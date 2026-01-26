#!/usr/bin/env python3
"""Simple test D_8b using top-k bins - no TTA"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import random
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, str(Path(__file__).parent))

from test_direction_models import DirectionModel, load_ply

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048
NUM_BINS = 8

def circular_error(pred, gt):
    diff = pred - gt
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))

def hungarian_match_error(pred_angles, gt_angles):
    n = len(pred_angles)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1 - np.cos(pred_angles[i] - gt_angles[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = []
    for i, j in zip(row_ind, col_ind):
        errors.append(circular_error(pred_angles[i], gt_angles[j]))
    return errors

DIRECTION_TO_ANGLE = {'+X': 0.0, '+Z': np.pi/2, '-X': np.pi, '-Z': 3*np.pi/2}
SYMMETRY_TO_CAT = {'1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
                   '旋转对称': 'no_front', '无正面': 'no_front'}

def main():
    print("Testing D_8b with top-k bins (no TTA)")

    with open('data/test_set/test_set.json') as f:
        all_samples = json.load(f)

    valid_samples = []
    for s in all_samples:
        sym_name = s.get('symmetry_name', '')
        if sym_name not in SYMMETRY_TO_CAT:
            continue
        cat = SYMMETRY_TO_CAT[sym_name]
        direction = s.get('front_direction')
        if cat in ['1_front', '2_fronts', '4_fronts']:
            if not direction or direction not in DIRECTION_TO_ANGLE:
                continue
            gt_angle = DIRECTION_TO_ANGLE[direction]
        else:
            gt_angle = 0.0
        k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4}.get(cat, 0)
        valid_samples.append({'file': s['file'], 'category': cat, 'gt_angle': gt_angle, 'k': k})

    random.seed(42)
    if len(valid_samples) > 350:
        valid_samples = random.sample(valid_samples, 350)

    print(f"Loaded {len(valid_samples)} samples")

    # Load model
    ckpt_path = 'checkpoints/D_8b_20251218_203046/best_error.pth'
    model = DirectionModel(mode='discrete', num_bins=8).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    print("Model loaded")

    data_root = Path('data/full_mn40_normal_resampled_ply')
    bin_width = 2 * np.pi / NUM_BINS

    def rotate_y(points, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
        return points @ R.T

    errors_1f = []
    errors_2f = []
    errors_4f = []

    for i, sample in enumerate(valid_samples):
        if sample['k'] == 0:
            continue

        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        # 8x TTA
        avg_probs = np.zeros(NUM_BINS)
        for rot_idx in range(8):
            rot_angle = rot_idx * bin_width  # 0, 45, 90, ...
            pts_rot = rotate_y(pts, rot_angle)
            pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(pts_tensor)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            # Undo rotation: shift probs back by rot_idx
            shifted = np.roll(probs, -rot_idx)
            avg_probs += shifted

        probs = avg_probs / 8

        gt = sample['gt_angle']
        cat = sample['category']

        if cat == '1_front':
            top_bin = np.argmax(probs)
            pred_angle = top_bin * bin_width
            err = circular_error(pred_angle, gt)
            errors_1f.append(err)

        elif cat == '2_fronts':
            top2_bins = np.argsort(probs)[-2:]
            pred_angles = [b * bin_width for b in top2_bins]
            gt_angles = [gt, (gt + np.pi) % (2 * np.pi)]
            errs = hungarian_match_error(pred_angles, gt_angles)
            errors_2f.extend(errs)

        elif cat == '4_fronts':
            top4_bins = np.argsort(probs)[-4:]
            pred_angles = [b * bin_width for b in top4_bins]
            gt_angles = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
            errs = hungarian_match_error(pred_angles, gt_angles)
            errors_4f.extend(errs)

    print("\nD_8b Top-k Results (no TTA)")
    print("-" * 40)
    if errors_1f:
        print(f"1-front (top-1): {np.degrees(np.mean(errors_1f)):.2f}° (n={len(errors_1f)})")
    if errors_2f:
        print(f"2-front (top-2): {np.degrees(np.mean(errors_2f)):.2f}° (n={len(errors_2f)//2} samples)")
    if errors_4f:
        print(f"4-front (top-4): {np.degrees(np.mean(errors_4f)):.2f}° (n={len(errors_4f)//4} samples)")

if __name__ == '__main__':
    main()

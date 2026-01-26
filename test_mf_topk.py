#!/usr/bin/env python3
"""Test MF_1c using top-k peaks for k-front prediction"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import random
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, str(Path(__file__).parent))

from test_direction_models import DirectionModel, load_ply

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

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
    return [circular_error(pred_angles[i], gt_angles[j]) for i, j in zip(row_ind, col_ind)]

DIRECTION_TO_ANGLE = {'+X': 0.0, '+Z': np.pi/2, '-X': np.pi, '-Z': 3*np.pi/2}
SYMMETRY_TO_CAT = {'1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
                   '旋转对称': 'no_front', '无正面': 'no_front'}

def main():
    print("Testing MF_1c with top-k peaks")

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
            k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4}[cat]
            valid_samples.append({'file': s['file'], 'category': cat, 'gt_angle': gt_angle, 'k': k})

    random.seed(42)
    if len(valid_samples) > 350:
        valid_samples = random.sample(valid_samples, 350)

    print(f"Loaded {len(valid_samples)} samples")

    # Load MF model
    model = DirectionModel(mode='mf', num_peaks=4).to(DEVICE)
    ckpt = torch.load('checkpoints/MF_1c_mu_only_20251217_005015/best_error.pth',
                      map_location=DEVICE, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    print("Model loaded")

    data_root = Path('data/full_mn40_normal_resampled_ply')
    errors_1f, errors_2f, errors_4f = [], [], []

    for sample in valid_samples:
        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue
        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_vec, kappa = model(pts_tensor)  # mu_vec: [1,4,2], kappa: [1,4]

        mu_vec = mu_vec[0].cpu().numpy()  # [4, 2] - (cos, sin)
        mu = np.arctan2(mu_vec[:, 1], mu_vec[:, 0])  # [4] angles
        mu = (mu + 2*np.pi) % (2*np.pi)  # normalize to [0, 2pi]
        kappa = kappa[0].cpu().numpy()  # [4]
        gt = sample['gt_angle']
        cat = sample['category']

        # Sort by kappa (confidence), pick top-k
        sorted_idx = np.argsort(kappa)[::-1]  # descending

        if cat == '1_front':
            pred = mu[sorted_idx[0]]
            errors_1f.append(circular_error(pred, gt))

        elif cat == '2_fronts':
            pred_angles = [float(mu[sorted_idx[i]]) for i in range(2)]
            gt_angles = [gt, (gt + np.pi) % (2*np.pi)]
            errors_2f.extend(hungarian_match_error(pred_angles, gt_angles))

        elif cat == '4_fronts':
            pred_angles = [float(mu[i]) for i in range(4)]
            gt_angles = [(gt + j*np.pi/2) % (2*np.pi) for j in range(4)]
            errors_4f.extend(hungarian_match_error(pred_angles, gt_angles))

    print("\nMF_1c Top-k Results")
    print("-" * 40)
    if errors_1f:
        print(f"1-front: {np.degrees(np.mean(errors_1f)):.2f}° (n={len(errors_1f)})")
    if errors_2f:
        print(f"2-front: {np.degrees(np.mean(errors_2f)):.2f}° (n={len(errors_2f)//2})")
    if errors_4f:
        print(f"4-front: {np.degrees(np.mean(errors_4f)):.2f}° (n={len(errors_4f)//4})")

if __name__ == '__main__':
    main()

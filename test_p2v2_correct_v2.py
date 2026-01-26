#!/usr/bin/env python3
"""
Test P2v2 Clean using EXACTLY the same method as training validation
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from models.probabilistic_orientation_net import ProbabilisticOrientationNet
from train_symmetry_classifier import SymmetryClassifier

class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    def forward(self, points, upright_vec=None):
        return self.classifier(points)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

DIRECTION_TO_ANGLE = {
    '+X': 0.0, '+Z': np.pi / 2, '-X': np.pi, '-Z': 3 * np.pi / 2,
}

SYMMETRY_TO_CAT = {
    '1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
    '旋转对称': 'no_front', '无正面': 'no_front',
}


def load_ply(filepath, num_points=2048):
    from plyfile import PlyData
    ply = PlyData.read(filepath)
    vertex = ply['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    elif len(pts) < num_points:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def circular_error(pred, gt):
    """Same as training: circular distance in radians"""
    diff = pred - gt
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def hungarian_errors(pred_peaks, gt_peaks):
    """Same as training: Hungarian matching for multi-peak"""
    n = len(pred_peaks)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1 - np.cos(pred_peaks[i] - gt_peaks[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = []
    for i, j in zip(row_ind, col_ind):
        errors.append(circular_error(pred_peaks[i], gt_peaks[j]))
    return errors


def load_p2v2_model(checkpoint_path):
    base_classifier = SymmetryClassifier(num_classes=5)
    wrapped_classifier = ClassifierWrapper(base_classifier).to(DEVICE)
    model = ProbabilisticOrientationNet(
        classifier=wrapped_classifier,
        backbone_dim=1024,
        expert_hidden_dim=256,
        freeze_classifier=True
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def main():
    print("=" * 60)
    print("Testing P2v2 Clean (same method as training validation)")
    print("=" * 60)

    # Load samples
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
        valid_samples.append({
            'file': s['file'],
            'category': cat,
            'gt_angle': gt_angle,
            'k': k,
        })

    random.seed(42)
    if len(valid_samples) > 350:
        valid_samples = random.sample(valid_samples, 350)

    print(f"Loaded {len(valid_samples)} test samples")

    # Load model
    p2v2_path = 'checkpoints/P2v2_Clean_20251230_165848/best.pth'
    model = load_p2v2_model(p2v2_path)
    print("Model loaded")

    data_root = Path('data/full_mn40_normal_resampled_ply')

    errors_1f = []
    errors_2f = []
    errors_4f = []
    kappas_1f = []
    kappas_2f = []
    kappas_4f = []

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

        pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(pts_tensor)

        gt = sample['gt_angle']
        cat = sample['category']

        # Use GT label to select head (same as training validation)
        if cat == '1_front':
            mu = output['head_1front']['mu'][0, 0].cpu().numpy()
            kappa = output['head_1front']['kappa'][0, 0].cpu().numpy()
            err = circular_error(mu, gt)
            errors_1f.append(err)
            kappas_1f.append(kappa)

        elif cat == '2_fronts':
            mu_2f = output['head_2front']['mu'][0].cpu().numpy()  # [2]
            kappa_2f = output['head_2front']['kappa'][0].cpu().numpy()  # [2]
            gt_peaks = [gt, (gt + np.pi) % (2 * np.pi)]
            pred_peaks = [mu_2f[j] for j in range(2)]
            errs = hungarian_errors(pred_peaks, gt_peaks)
            errors_2f.extend(errs)
            kappas_2f.extend([kappa_2f[j] for j in range(2)])

        elif cat == '4_fronts':
            mu_4f = output['head_4front']['mu'][0].cpu().numpy()  # [4]
            kappa_4f = output['head_4front']['kappa'][0].cpu().numpy()  # [4]
            gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
            pred_peaks = [mu_4f[j] for j in range(4)]
            errs = hungarian_errors(pred_peaks, gt_peaks)
            errors_4f.extend(errs)
            kappas_4f.extend([kappa_4f[j] for j in range(4)])

        if (i + 1) % 50 == 0:
            all_errs = errors_1f + errors_2f + errors_4f
            mean_err = np.degrees(np.mean(all_errs)) if all_errs else 0
            print(f"Progress: {i+1}/{len(valid_samples)}, Current mean error: {mean_err:.2f}°")

    # Results (same format as training)
    print("\n" + "=" * 60)
    print("P2v2 Clean Results (same method as training)")
    print("=" * 60)

    if errors_1f:
        errors_1f_deg = np.degrees(errors_1f)
        print(f"1-front: {np.mean(errors_1f_deg):.2f}° (median: {np.median(errors_1f_deg):.2f}°, n={len(errors_1f)})")
        print(f"  <5°: {100*np.mean(errors_1f_deg < 5):.1f}%, <10°: {100*np.mean(errors_1f_deg < 10):.1f}%")
        print(f"  kappa: {np.mean(kappas_1f):.1f}")

    if errors_2f:
        errors_2f_deg = np.degrees(errors_2f)
        print(f"2-front: {np.mean(errors_2f_deg):.2f}° (median: {np.median(errors_2f_deg):.2f}°, n={len(errors_2f)//2} samples, {len(errors_2f)} peaks)")
        print(f"  <5°: {100*np.mean(errors_2f_deg < 5):.1f}%, <10°: {100*np.mean(errors_2f_deg < 10):.1f}%")

    if errors_4f:
        errors_4f_deg = np.degrees(errors_4f)
        print(f"4-front: {np.mean(errors_4f_deg):.2f}° (median: {np.median(errors_4f_deg):.2f}°, n={len(errors_4f)//4} samples, {len(errors_4f)} peaks)")
        print(f"  <5°: {100*np.mean(errors_4f_deg < 5):.1f}%, <10°: {100*np.mean(errors_4f_deg < 10):.1f}%")

    # Overall
    all_errors_deg = []
    if errors_1f:
        all_errors_deg.extend(np.degrees(errors_1f))
    if errors_2f:
        all_errors_deg.extend(np.degrees(errors_2f))
    if errors_4f:
        all_errors_deg.extend(np.degrees(errors_4f))

    if all_errors_deg:
        print(f"\nOverall: {np.mean(all_errors_deg):.2f}° (median: {np.median(all_errors_deg):.2f}°)")
        print(f"  <10°: {100*np.mean(np.array(all_errors_deg) < 10):.1f}%")


if __name__ == '__main__':
    main()

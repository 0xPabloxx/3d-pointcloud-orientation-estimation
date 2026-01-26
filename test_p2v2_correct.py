#!/usr/bin/env python3
"""
Test P2v2 Clean using the actual model architecture
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

sys.path.insert(0, str(Path(__file__).parent))

from models.probabilistic_orientation_net import ProbabilisticOrientationNet
from train_symmetry_classifier import SymmetryClassifier

class ClassifierWrapper(torch.nn.Module):
    """Wrapper to match training structure"""
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
    '1个正面': '1_front', '1_front': '1_front',
    '2个正面': '2_fronts', '2_fronts': '2_fronts',
    '4个正面': '4_fronts', '4_fronts': '4_fronts',
    '旋转对称': 'no_front', 'symmetric': 'no_front', '完全对称': 'no_front',
    '无正面': 'no_front', 'no_front': 'no_front', '没有正面': 'no_front',
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
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def rotate_y(points, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
    return points @ R.T


def angular_distance(pred_angle, gt_angle, symmetry_k=1):
    if symmetry_k <= 0:
        return 0.0
    min_error = float('inf')
    for i in range(symmetry_k):
        candidate = (gt_angle + i * 2 * np.pi / symmetry_k) % (2 * np.pi)
        diff = pred_angle - candidate
        error = abs(np.arctan2(np.sin(diff), np.cos(diff)))
        min_error = min(min_error, error)
    return min_error


def load_p2v2_model(checkpoint_path, classifier_path):
    """Load P2v2 model using original architecture"""
    # Create classifier with correct structure - needs to be wrapped
    base_classifier = SymmetryClassifier(num_classes=5)
    wrapped_classifier = ClassifierWrapper(base_classifier).to(DEVICE)

    # Create P2v2 model with wrapped classifier
    model = ProbabilisticOrientationNet(
        classifier=wrapped_classifier,
        backbone_dim=1024,
        expert_hidden_dim=256,
        freeze_classifier=True
    ).to(DEVICE)

    # Load P2v2 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded P2v2 model successfully")
    return model


def main():
    print("=" * 60)
    print("Testing P2v2 Clean on 350 samples with 8x TTA")
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

    # Sample distribution
    cat_counts = defaultdict(int)
    for s in valid_samples:
        cat_counts[s['category']] += 1
    print(f"Distribution: 1_front={cat_counts['1_front']}, 2_fronts={cat_counts['2_fronts']}, 4_fronts={cat_counts['4_fronts']}, no_front={cat_counts['no_front']}")

    # Load model
    p2v2_path = 'checkpoints/P2v2_Clean_20251230_165848/best.pth'
    classifier_path = 'checkpoints/CleanClassifier_20251229_220630/best.pth'
    model = load_p2v2_model(p2v2_path, classifier_path)

    data_root = Path('data/full_mn40_normal_resampled_ply')
    errors_by_cat = defaultdict(list)
    all_errors = []

    # Category to head mapping
    cat_to_label = {'1_front': 0, '2_fronts': 1, '4_fronts': 2}

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
        pred_angles = []
        for rot_idx in range(8):
            rot_angle = rot_idx * 2 * np.pi / 8
            pts_rot = rotate_y(pts, rot_angle)
            pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(pts_tensor)

            # Use CLASSIFIER prediction to select head (end-to-end)
            weights = output['weights'][0]  # [5] softmax probabilities
            pred_class = weights.argmax().item()  # 0=1f, 1=2f, 2=4f, 3=rot, 4=nofront

            if pred_class == 0:  # 1_front
                mu = output['head_1front']['mu'][0, 0].item()
            elif pred_class == 1:  # 2_fronts
                kappas = output['head_2front']['kappa'][0]
                max_idx = kappas.argmax().item()
                mu = output['head_2front']['mu'][0, max_idx].item()
            elif pred_class == 2:  # 4_fronts
                kappas = output['head_4front']['kappa'][0]
                max_idx = kappas.argmax().item()
                mu = output['head_4front']['mu'][0, max_idx].item()
            else:  # 3=rot, 4=nofront - classifier says no direction
                mu = 0.0  # default, will have high error

            pred_angle = (mu - rot_angle) % (2 * np.pi)
            pred_angles.append(pred_angle)

        # Circular mean
        sin_sum = sum(np.sin(a) for a in pred_angles)
        cos_sum = sum(np.cos(a) for a in pred_angles)
        avg_pred_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

        error = angular_distance(avg_pred_angle, sample['gt_angle'], sample['k'])
        error_deg = np.degrees(error)

        errors_by_cat[sample['category']].append(error_deg)
        all_errors.append(error_deg)

        if (i + 1) % 50 == 0:
            current_mean = np.mean(all_errors) if all_errors else 0
            print(f"Progress: {i+1}/{len(valid_samples)}, Current mean error: {current_mean:.2f}°")

    # Results
    print("\n" + "=" * 60)
    print("P2v2 Clean Results")
    print("=" * 60)
    print(f"Overall: {np.mean(all_errors):.2f}° (median: {np.median(all_errors):.2f}°, n={len(all_errors)})")
    for cat in ['1_front', '2_fronts', '4_fronts']:
        if errors_by_cat[cat]:
            print(f"  {cat}: {np.mean(errors_by_cat[cat]):.2f}° (n={len(errors_by_cat[cat])})")

    # Save results
    results = {
        'model': 'P2v2_Clean',
        'overall_mean': float(np.mean(all_errors)),
        'overall_median': float(np.median(all_errors)),
        'num_samples': len(all_errors),
    }
    for cat in ['1_front', '2_fronts', '4_fronts']:
        if errors_by_cat[cat]:
            results[f'{cat}_mean'] = float(np.mean(errors_by_cat[cat]))
            results[f'{cat}_count'] = len(errors_by_cat[cat])

    output_path = Path(f'results/p2v2_clean_correct_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

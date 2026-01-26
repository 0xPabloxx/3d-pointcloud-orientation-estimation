#!/usr/bin/env python3
"""
Test MuOnly model on test set with unified Hungarian matching angular error.

All cases use 4 vs 4 Hungarian matching:
- 1-front: GT replicated 4x (same direction)
- 2-front: GT 2 directions replicated 2x each
- 4-front: GT 4 directions
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_muonly_full import FullMoEDataset, BaselineDirectionModel, collate_fn


def circular_distance(pred: float, gt: float) -> float:
    """Compute circular distance in radians."""
    diff = abs(pred - gt)
    return min(diff, 2 * np.pi - diff)


def hungarian_angular_error(pred_peaks: np.ndarray, gt_peaks: np.ndarray) -> list:
    """
    Compute angular errors using Hungarian matching.

    Args:
        pred_peaks: (4,) predicted angles in radians
        gt_peaks: (4,) ground truth angles in radians

    Returns:
        List of 4 angular errors in radians
    """
    n = len(gt_peaks)
    cost = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost[i, j] = circular_distance(pred_peaks[i], gt_peaks[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    errors = [cost[i, j] for i, j in zip(row_ind, col_ind)]
    return errors


def create_gt_peaks(gt_angle: float, label: int) -> np.ndarray:
    """
    Create 4 GT peaks based on symmetry label.

    Args:
        gt_angle: Base ground truth angle in radians
        label: 0=1-front, 1=2-front, 2=4-front

    Returns:
        (4,) array of GT angles
    """
    if label == 0:  # 1-front: replicate 4x
        return np.array([gt_angle, gt_angle, gt_angle, gt_angle])

    elif label == 1:  # 2-front: 2 directions, each replicated 2x
        angle2 = (gt_angle + np.pi) % (2 * np.pi)
        return np.array([gt_angle, gt_angle, angle2, angle2])

    elif label == 2:  # 4-front: 4 distinct directions
        return np.array([
            gt_angle,
            (gt_angle + np.pi / 2) % (2 * np.pi),
            (gt_angle + np.pi) % (2 * np.pi),
            (gt_angle + 3 * np.pi / 2) % (2 * np.pi),
        ])

    else:
        raise ValueError(f"Invalid label: {label}")


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()

    results = {
        '1-front': {'errors': [], 'samples': []},
        '2-front': {'errors': [], 'samples': []},
        '4-front': {'errors': [], 'samples': []},
    }

    label_to_name = {0: '1-front', 1: '2-front', 2: '4-front'}

    for batch in dataloader:
        points = batch['points'].to(device)
        gt_angles = batch['gt_angle'].numpy()
        gt_labels = batch['gt_label'].numpy()
        files = batch['file']

        outputs = model(points)
        pred_mu = outputs['mu'].cpu().numpy()  # (B, 4)

        for i in range(len(gt_labels)):
            label = gt_labels[i]

            # Skip rot-sym and no-front (labels 3, 4)
            if label not in [0, 1, 2]:
                continue

            gt_angle = gt_angles[i]
            pred_peaks = pred_mu[i]  # (4,)

            # Create 4 GT peaks based on symmetry type
            gt_peaks = create_gt_peaks(gt_angle, label)

            # Hungarian matching to get 4 errors
            errors = hungarian_angular_error(pred_peaks, gt_peaks)

            # Convert to degrees
            errors_deg = [e * 180 / np.pi for e in errors]

            name = label_to_name[label]
            results[name]['errors'].extend(errors_deg)
            results[name]['samples'].append({
                'file': files[i],
                'gt_angle': float(gt_angle),
                'pred_peaks': pred_peaks.tolist(),
                'errors_deg': errors_deg,
                'mean_error': float(np.mean(errors_deg)),
            })

    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("Angular Error Results (Hungarian Matching, 4 vs 4)")
    print("=" * 70)

    all_errors = []

    for name in ['1-front', '2-front', '4-front']:
        errors = np.array(results[name]['errors'])
        if len(errors) == 0:
            continue

        all_errors.extend(errors)

        print(f"\n{name} ({len(errors)} peaks from {len(results[name]['samples'])} samples):")
        print(f"  Mean error:   {np.mean(errors):.2f}°")
        print(f"  Median error: {np.median(errors):.2f}°")
        print(f"  Std error:    {np.std(errors):.2f}°")
        print(f"  <5°:  {100 * np.mean(errors < 5):.1f}%")
        print(f"  <10°: {100 * np.mean(errors < 10):.1f}%")
        print(f"  <15°: {100 * np.mean(errors < 15):.1f}%")
        print(f"  >45°: {100 * np.mean(errors > 45):.1f}%")
        print(f"  >90°: {100 * np.mean(errors > 90):.1f}%")

    # Overall
    all_errors = np.array(all_errors)
    print(f"\nOverall ({len(all_errors)} peaks):")
    print(f"  Mean error:   {np.mean(all_errors):.2f}°")
    print(f"  Median error: {np.median(all_errors):.2f}°")
    print(f"  <5°:  {100 * np.mean(all_errors < 5):.1f}%")
    print(f"  <10°: {100 * np.mean(all_errors < 10):.1f}%")
    print(f"  <15°: {100 * np.mean(all_errors < 15):.1f}%")

    return all_errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/MuOnly_Full_20260109_234110/best.pth')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = BaselineDirectionModel(backbone_dim=1024, hidden_dim=512).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load test dataset with random rotation
    print("\nLoading test dataset (with random rotation)...")
    test_dataset = FullMoEDataset(
        annotation_file=args.annotation_file,
        data_dir=args.data_dir,
        split='test',
        num_points=2048,
        augment=True,  # Random rotation
        num_rotations=1,  # 1 random rotation per sample
        seed=42,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model, test_loader, device)

    # Print results
    all_errors = print_results(results)

    # Save results
    if args.save_results:
        output_file = Path(args.checkpoint).parent / 'angular_error_results.json'
        save_data = {
            'checkpoint': args.checkpoint,
            'summary': {
                '1-front': {
                    'mean': float(np.mean(results['1-front']['errors'])),
                    'median': float(np.median(results['1-front']['errors'])),
                    'std': float(np.std(results['1-front']['errors'])),
                    'count': len(results['1-front']['errors']),
                },
                '2-front': {
                    'mean': float(np.mean(results['2-front']['errors'])),
                    'median': float(np.median(results['2-front']['errors'])),
                    'std': float(np.std(results['2-front']['errors'])),
                    'count': len(results['2-front']['errors']),
                },
                '4-front': {
                    'mean': float(np.mean(results['4-front']['errors'])),
                    'median': float(np.median(results['4-front']['errors'])),
                    'std': float(np.std(results['4-front']['errors'])),
                    'count': len(results['4-front']['errors']),
                },
                'overall': {
                    'mean': float(np.mean(all_errors)),
                    'median': float(np.median(all_errors)),
                    'count': len(all_errors),
                },
            },
        }
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

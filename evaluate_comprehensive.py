"""
Comprehensive evaluation comparing different matching methods:
1. Argmax + Min Distance (single prediction)
2. Top-K peaks + Hungarian matching (multi-peak matching)
3. KL Divergence (distribution comparison)

Usage:
    python evaluate_comprehensive.py --all
    python evaluate_comprehensive.py --checkpoint checkpoints/D_16b_20251219_081853
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

sys.path.insert(0, str(Path(__file__).parent))

from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn
from train_direction import (
    PointNetPlusPlusEncoder,
    DiscreteDirectionHead,
    ProjectionRegressionHead,
)


# =============================================================================
# Core Functions
# =============================================================================

def circular_distance(angle1, angle2):
    """Circular distance in radians"""
    diff = angle1 - angle2
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def find_top_k_peaks(probs, k, num_bins):
    """Find top-K peaks in probability distribution"""
    bin_width = 2 * np.pi / num_bins

    # Get top-k bin indices
    top_k_bins = torch.topk(probs, k).indices.cpu().numpy()
    top_k_angles = top_k_bins * bin_width

    return top_k_angles


def get_equivalent_directions(gt_angle, num_equiv):
    """Get all equivalent directions for a category"""
    if num_equiv == 1:
        return [gt_angle]
    elif num_equiv == 2:
        return [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
    elif num_equiv == 4:
        return [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
    else:
        return [gt_angle]


def hungarian_matching(pred_angles, gt_angles):
    """
    Hungarian algorithm for optimal matching between predictions and GT.

    Args:
        pred_angles: List of predicted angles (radians)
        gt_angles: List of GT equivalent angles (radians)

    Returns:
        Total matching cost (sum of matched distances)
        List of (pred_idx, gt_idx, error) tuples
    """
    n_pred = len(pred_angles)
    n_gt = len(gt_angles)

    # Build cost matrix
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_angles):
        for j, gt in enumerate(gt_angles):
            cost_matrix[i, j] = circular_distance(pred, gt)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute total cost and matches
    matches = []
    total_cost = 0
    for i, j in zip(row_ind, col_ind):
        error = cost_matrix[i, j]
        matches.append((i, j, error))
        total_cost += error

    return total_cost, matches


def compute_kl_divergence(pred_probs, gt_distribution):
    """Compute KL divergence between prediction and GT distribution"""
    # Add small epsilon for numerical stability
    eps = 1e-10
    pred_probs = pred_probs + eps
    gt_distribution = gt_distribution + eps

    # Normalize
    pred_probs = pred_probs / pred_probs.sum()
    gt_distribution = gt_distribution / gt_distribution.sum()

    # KL(GT || pred) - how well pred matches GT
    kl = (gt_distribution * np.log(gt_distribution / pred_probs)).sum()
    return kl


def create_gt_distribution(gt_angle, num_equiv, num_bins, temperature=5.0):
    """Create GT probability distribution for KL comparison"""
    bin_width = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * bin_width

    equiv_angles = get_equivalent_directions(gt_angle, num_equiv)

    # Create soft distribution using von Mises-like kernel
    dist = np.zeros(num_bins)
    for angle in equiv_angles:
        for i, center in enumerate(bin_centers):
            dist[i] += np.exp(temperature * np.cos(center - angle))

    # Normalize
    dist = dist / dist.sum()
    return dist


# =============================================================================
# Evaluation Methods
# =============================================================================

def evaluate_argmax_min_distance(pred_probs, gt_angle, category, num_bins):
    """
    Method 1: Argmax + Min Distance
    - Take argmax prediction
    - Find minimum distance to any equivalent GT direction
    """
    bin_width = 2 * np.pi / num_bins
    pred_bin = pred_probs.argmax().item()
    pred_angle = pred_bin * bin_width

    if category == '1_front':
        num_equiv = 1
    elif category == '2_fronts':
        num_equiv = 2
    elif category == '4_fronts':
        num_equiv = 4
    else:
        return None

    equiv_angles = get_equivalent_directions(gt_angle, num_equiv)
    min_error = min(circular_distance(pred_angle, a) for a in equiv_angles)

    return min_error * 180 / np.pi  # Convert to degrees


def evaluate_topk_hungarian(pred_probs, gt_angle, category, num_bins, k=None):
    """
    Method 2: Top-K + Hungarian Matching
    - Take top-K predictions (K = num_equiv)
    - Use Hungarian algorithm to match with equivalent GT directions
    - Return average matched error
    """
    if category == '1_front':
        num_equiv = 1
    elif category == '2_fronts':
        num_equiv = 2
    elif category == '4_fronts':
        num_equiv = 4
    else:
        return None

    k = k or num_equiv

    pred_angles = find_top_k_peaks(pred_probs, k, num_bins)
    gt_angles = get_equivalent_directions(gt_angle, num_equiv)

    total_cost, matches = hungarian_matching(pred_angles, gt_angles)
    avg_error = total_cost / len(matches)

    return avg_error * 180 / np.pi  # Convert to degrees


def evaluate_kl_divergence(pred_probs, gt_angle, category, num_bins, temperature=5.0):
    """
    Method 3: KL Divergence
    - Compare predicted distribution with ideal GT distribution
    - Lower is better
    """
    if category == '1_front':
        num_equiv = 1
    elif category == '2_fronts':
        num_equiv = 2
    elif category == '4_fronts':
        num_equiv = 4
    else:
        return None

    gt_dist = create_gt_distribution(gt_angle, num_equiv, num_bins, temperature)
    pred_np = pred_probs.cpu().numpy()

    kl = compute_kl_divergence(pred_np, gt_dist)
    return kl


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_checkpoint_comprehensive(checkpoint_dir, num_aug=10, device='cuda'):
    """Comprehensive evaluation with all methods"""

    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get('exp_name', Path(checkpoint_dir).name)
    num_bins = config.get('num_bins', 8)
    gt_mode = config.get('gt_mode', 'projection')
    temperature = config.get('temperature', 5.0)

    print(f"\n{'='*70}")
    print(f"Evaluating: {exp_name}")
    print(f"  num_bins: {num_bins}, gt_mode: {gt_mode}")
    print(f"{'='*70}")

    # Load model
    encoder = PointNetPlusPlusEncoder().to(device)
    if gt_mode == 'dr':
        head = ProjectionRegressionHead(num_dirs=num_bins).to(device)
    else:
        head = DiscreteDirectionHead(num_bins=num_bins).to(device)

    ckpt_path = Path(checkpoint_dir) / 'best_error.pth'
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / 'best_loss.pth'

    if not ckpt_path.exists():
        print(f"Checkpoint not found in {checkpoint_dir}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    encoder.eval()
    head.eval()

    # Results storage
    results = {
        'exp_name': exp_name,
        'config': config,
        'methods': {}
    }

    for category in ['1_front', '4_fronts']:
        # Use augmented validation for fair evaluation
        dataset = DiscreteDirectionDataset(
            split='val',
            categories=[category],
            num_bins=num_bins,
            gt_mode=gt_mode,
            temperature=temperature,
            augment=True,  # Enable augmentation for fair eval
            augment_factor=num_aug,
        )

        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset, batch_size=32, shuffle=False,
            num_workers=4, collate_fn=collate_fn,
        )

        # Collect errors for each method
        errors_argmax = []
        errors_hungarian = []
        kl_values = []

        with torch.no_grad():
            for batch in loader:
                points = batch['points'].to(device)
                gt_angle = batch['gt_angle']
                categories_batch = batch['category']

                features = encoder(points)
                pred = head(features)
                pred_probs = F.softmax(pred, dim=-1)

                for b in range(len(categories_batch)):
                    cat = categories_batch[b]
                    if cat == 'no_front':
                        continue

                    probs = pred_probs[b]
                    gt = gt_angle[b].item()

                    # Method 1: Argmax + Min Distance
                    err1 = evaluate_argmax_min_distance(probs, gt, cat, num_bins)
                    if err1 is not None:
                        errors_argmax.append(err1)

                    # Method 2: Top-K + Hungarian
                    err2 = evaluate_topk_hungarian(probs, gt, cat, num_bins)
                    if err2 is not None:
                        errors_hungarian.append(err2)

                    # Method 3: KL Divergence
                    kl = evaluate_kl_divergence(probs, gt, cat, num_bins)
                    if kl is not None:
                        kl_values.append(kl)

        # Store results
        if category not in results['methods']:
            results['methods'][category] = {}

        if errors_argmax:
            results['methods'][category]['argmax_min_dist'] = {
                'mean': float(np.mean(errors_argmax)),
                'std': float(np.std(errors_argmax)),
                'median': float(np.median(errors_argmax)),
            }

        if errors_hungarian:
            results['methods'][category]['topk_hungarian'] = {
                'mean': float(np.mean(errors_hungarian)),
                'std': float(np.std(errors_hungarian)),
                'median': float(np.median(errors_hungarian)),
            }

        if kl_values:
            results['methods'][category]['kl_divergence'] = {
                'mean': float(np.mean(kl_values)),
                'std': float(np.std(kl_values)),
                'median': float(np.median(kl_values)),
            }

        # Print results
        print(f"\n  {category}:")
        if errors_argmax:
            print(f"    Argmax+MinDist: {np.mean(errors_argmax):.2f}° ± {np.std(errors_argmax):.2f}°")
        if errors_hungarian:
            print(f"    Top-K+Hungarian: {np.mean(errors_hungarian):.2f}° ± {np.std(errors_hungarian):.2f}°")
        if kl_values:
            print(f"    KL Divergence: {np.mean(kl_values):.4f} ± {np.std(kl_values):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint directory')
    parser.add_argument('--all', action='store_true', help='Evaluate all D/DR checkpoints')
    parser.add_argument('--num_aug', type=int, default=10, help='Number of augmentations')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.all:
        checkpoints_dir = Path('checkpoints')

        # All D/DR checkpoints
        d_checkpoints = [
            'D_8a_20251218_171207',
            'D_8b_20251218_203046',
            'D_8c_20251218_234110',
            'D_8d_20251219_024151',
            'D_16a_20251219_053127',
            'D_16b_20251219_081853',
        ]

        dr_checkpoints = [
            'DR_8a_20251219_215002',
            'DR_8b_20251220_004513',
            'DR_16a_20251220_033656',
        ]

        all_results = []

        print("\n" + "="*80)
        print("  D Series Comprehensive Evaluation")
        print("="*80)

        for ckpt in d_checkpoints:
            ckpt_path = checkpoints_dir / ckpt
            if ckpt_path.exists():
                result = evaluate_checkpoint_comprehensive(
                    str(ckpt_path), args.num_aug, args.device
                )
                if result:
                    all_results.append(result)

        print("\n" + "="*80)
        print("  DR Series Comprehensive Evaluation")
        print("="*80)

        for ckpt in dr_checkpoints:
            ckpt_path = checkpoints_dir / ckpt
            if ckpt_path.exists():
                result = evaluate_checkpoint_comprehensive(
                    str(ckpt_path), args.num_aug, args.device
                )
                if result:
                    all_results.append(result)

        # Print summary table
        print("\n" + "="*80)
        print("  Summary Table: Argmax + Min Distance vs Top-K + Hungarian")
        print("="*80)

        print(f"\n{'Experiment':<12} | {'1_front':<25} | {'4_fronts':<25}")
        print(f"{'':12} | {'ArgMax':>10} {'Hungarian':>12} | {'ArgMax':>10} {'Hungarian':>12}")
        print("-"*78)

        for r in all_results:
            exp = r['exp_name']
            methods = r.get('methods', {})

            # 1_front
            m1 = methods.get('1_front', {})
            arg1 = m1.get('argmax_min_dist', {}).get('mean', float('nan'))
            hun1 = m1.get('topk_hungarian', {}).get('mean', float('nan'))

            # 4_fronts
            m4 = methods.get('4_fronts', {})
            arg4 = m4.get('argmax_min_dist', {}).get('mean', float('nan'))
            hun4 = m4.get('topk_hungarian', {}).get('mean', float('nan'))

            print(f"{exp:<12} | {arg1:>9.2f}° {hun1:>11.2f}° | {arg4:>9.2f}° {hun4:>11.2f}°")

        # KL Divergence summary
        print("\n" + "="*80)
        print("  Summary Table: KL Divergence (lower is better)")
        print("="*80)

        print(f"\n{'Experiment':<12} | {'1_front KL':>15} | {'4_fronts KL':>15}")
        print("-"*50)

        for r in all_results:
            exp = r['exp_name']
            methods = r.get('methods', {})

            kl1 = methods.get('1_front', {}).get('kl_divergence', {}).get('mean', float('nan'))
            kl4 = methods.get('4_fronts', {}).get('kl_divergence', {}).get('mean', float('nan'))

            print(f"{exp:<12} | {kl1:>15.4f} | {kl4:>15.4f}")

        # Save results
        output_path = 'evaluation_comprehensive.json'
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    elif args.checkpoint:
        evaluate_checkpoint_comprehensive(args.checkpoint, args.num_aug, args.device)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

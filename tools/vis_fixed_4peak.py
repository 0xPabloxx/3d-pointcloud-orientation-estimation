#!/usr/bin/env python
"""
Fixed 4-Peak von Mises 模型推理可视化

可视化预测 vs GT 的 von Mises 混合分布。
使用极坐标图展示 4 个峰的位置和集中度。

Usage:
    python tools/vis_fixed_4peak.py
    python tools/vis_fixed_4peak.py --checkpoint path/to/best.pth
    python tools/vis_fixed_4peak.py --num_samples 20 --save
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_fixed_4peak import Fixed4PeakModel
from datasets.fixed_4peak_dataset import Fixed4PeakDataset, collate_fn


def compute_von_mises_pdf(theta: np.ndarray, mu_angle: float, kappa: float) -> np.ndarray:
    """Compute von Mises PDF at given angles."""
    if kappa < 0.01:
        # Uniform distribution
        return np.ones_like(theta) / (2 * np.pi)

    # von Mises PDF
    from scipy.special import i0
    pdf = np.exp(kappa * np.cos(theta - mu_angle)) / (2 * np.pi * i0(kappa))
    return pdf


def compute_mixture_pdf(mu_angles: np.ndarray, kappas: np.ndarray,
                        theta: np.ndarray = None, weight: float = 0.25) -> np.ndarray:
    """Compute mixture of von Mises PDF."""
    if theta is None:
        theta = np.linspace(0, 2*np.pi, 360)

    mixture = np.zeros_like(theta)
    for mu, kappa in zip(mu_angles, kappas):
        mixture += weight * compute_von_mises_pdf(theta, mu, kappa)

    return mixture


def visualize_single_sample(ax, pred_mu, pred_kappa, gt_mu, gt_kappa,
                            category: str, filename: str):
    """Visualize a single sample on a polar axis."""
    theta = np.linspace(0, 2*np.pi, 360)

    # Convert mu (cos, sin) to angles
    pred_angles = np.arctan2(pred_mu[:, 1], pred_mu[:, 0])
    gt_angles = np.arctan2(gt_mu[:, 1], gt_mu[:, 0])

    # Compute PDFs
    pred_pdf = compute_mixture_pdf(pred_angles, pred_kappa, theta)
    gt_pdf = compute_mixture_pdf(gt_angles, gt_kappa, theta)

    # Normalize for visualization
    max_val = max(pred_pdf.max(), gt_pdf.max())
    if max_val > 0:
        pred_pdf = pred_pdf / max_val
        gt_pdf = gt_pdf / max_val

    # Plot PDFs
    ax.plot(theta, gt_pdf, 'g-', linewidth=2, label='GT', alpha=0.8)
    ax.plot(theta, pred_pdf, 'b--', linewidth=2, label='Pred', alpha=0.8)

    # Mark peak positions
    for i, (angle, kappa) in enumerate(zip(gt_angles, gt_kappa)):
        if kappa > 0.5:  # Active peak
            ax.scatter([angle], [1.05], c='green', s=80, marker='^', zorder=10)

    for i, (angle, kappa) in enumerate(zip(pred_angles, pred_kappa)):
        if kappa > 5:  # Significant peak
            ax.scatter([angle], [1.1], c='blue', s=60, marker='v', zorder=10)

    # Formatting
    ax.set_ylim(0, 1.2)
    ax.set_title(f'{category}\n{filename[:20]}...', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)

    return pred_angles, pred_kappa, gt_angles, gt_kappa


def visualize_predictions(model, dataset, device, num_samples: int = 16,
                          save_path: str = None, by_category: bool = True):
    """
    Visualize model predictions vs ground truth.

    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: torch device
        num_samples: Number of samples to visualize
        save_path: Path to save figure
        by_category: If True, sample equally from each category
    """
    model.eval()
    categories = Fixed4PeakDataset.CATEGORIES

    if by_category:
        # Sample equally from each category
        samples_per_cat = max(1, num_samples // len(categories))
        indices = []
        for cat in categories:
            cat_indices = dataset.get_category_samples(cat)
            if len(cat_indices) > 0:
                selected = np.random.choice(cat_indices,
                                           min(samples_per_cat, len(cat_indices)),
                                           replace=False)
                indices.extend(selected.tolist())
        indices = indices[:num_samples]
    else:
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # Create figure
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows),
                              subplot_kw={'projection': 'polar'})
    axes = np.array(axes).flatten()

    # Statistics
    stats_by_category = {cat: {'kl_divs': [], 'angle_errors': [], 'kappa_errors': []}
                         for cat in categories}

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get sample
            sample = dataset[idx]
            points = sample['points'].unsqueeze(0).to(device)
            gt_mu = sample['mu'].numpy()
            gt_kappa = sample['kappa'].numpy()
            category = sample['category_name']
            filename = sample['filename']

            # Inference
            pred_mu, pred_kappa = model(points)
            pred_mu = pred_mu[0].cpu().numpy()
            pred_kappa = pred_kappa[0].cpu().numpy()

            # Visualize
            pred_angles, pred_k, gt_angles, gt_k = visualize_single_sample(
                ax, pred_mu, pred_kappa, gt_mu, gt_kappa, category, filename
            )

            # Compute metrics
            # Angle error (find best matching)
            angle_errors = []
            for pa in pred_angles:
                min_error = min(abs((pa - ga + np.pi) % (2*np.pi) - np.pi) for ga in gt_angles)
                angle_errors.append(np.rad2deg(min_error))

            kappa_error = np.abs(pred_kappa - gt_kappa).mean()

            stats_by_category[category]['angle_errors'].extend(angle_errors)
            stats_by_category[category]['kappa_errors'].append(kappa_error)

    # Hide unused axes
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Fixed 4-Peak von Mises Predictions vs GT\n'
                 '(Green = GT, Blue = Prediction)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("Per-Category Statistics")
    print("=" * 60)
    for cat in categories:
        stats = stats_by_category[cat]
        if stats['angle_errors']:
            mean_angle = np.mean(stats['angle_errors'])
            mean_kappa = np.mean(stats['kappa_errors'])
            print(f"{cat:12s}: Angle Error = {mean_angle:6.2f}°, κ Error = {mean_kappa:6.2f}")

    return fig


def visualize_distribution_comparison(model, dataset, device,
                                       samples_per_category: int = 3,
                                       save_path: str = None):
    """
    Create a detailed comparison grid: rows = categories, cols = samples.
    Each cell shows polar plot with GT (green) and Pred (blue).
    """
    model.eval()
    categories = Fixed4PeakDataset.CATEGORIES
    n_cats = len(categories)

    fig, axes = plt.subplots(n_cats, samples_per_category,
                              figsize=(4*samples_per_category, 4*n_cats),
                              subplot_kw={'projection': 'polar'})

    with torch.no_grad():
        for row, cat in enumerate(categories):
            cat_indices = dataset.get_category_samples(cat)
            if len(cat_indices) == 0:
                continue

            selected = np.random.choice(cat_indices,
                                        min(samples_per_category, len(cat_indices)),
                                        replace=False)

            for col, idx in enumerate(selected):
                ax = axes[row, col] if n_cats > 1 else axes[col]

                sample = dataset[idx]
                points = sample['points'].unsqueeze(0).to(device)
                gt_mu = sample['mu'].numpy()
                gt_kappa = sample['kappa'].numpy()
                filename = sample['filename']

                pred_mu, pred_kappa = model(points)
                pred_mu = pred_mu[0].cpu().numpy()
                pred_kappa = pred_kappa[0].cpu().numpy()

                visualize_single_sample(ax, pred_mu, pred_kappa,
                                        gt_mu, gt_kappa, cat, filename)

                if col == 0:
                    ax.set_ylabel(cat, fontsize=12, labelpad=30)

    plt.suptitle('Fixed 4-Peak von Mises: Predictions by Category\n'
                 '(Green = GT, Blue = Prediction)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


def print_sample_predictions(model, dataset, device, num_samples: int = 5):
    """Print detailed predictions for a few samples."""
    model.eval()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print("\n" + "=" * 80)
    print("Detailed Predictions")
    print("=" * 80)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            points = sample['points'].unsqueeze(0).to(device)
            gt_mu = sample['mu'].numpy()
            gt_kappa = sample['kappa'].numpy()
            category = sample['category_name']
            filename = sample['filename']

            pred_mu, pred_kappa = model(points)
            pred_mu = pred_mu[0].cpu().numpy()
            pred_kappa = pred_kappa[0].cpu().numpy()

            # Convert to angles
            pred_angles = np.rad2deg(np.arctan2(pred_mu[:, 1], pred_mu[:, 0]))
            gt_angles = np.rad2deg(np.arctan2(gt_mu[:, 1], gt_mu[:, 0]))

            print(f"\n[{category}] {filename}")
            print("-" * 60)
            print(f"{'Peak':<6} {'GT Angle':<12} {'Pred Angle':<12} {'GT κ':<10} {'Pred κ':<10}")
            print("-" * 60)
            for i in range(4):
                print(f"{i+1:<6} {gt_angles[i]:>8.1f}°    {pred_angles[i]:>8.1f}°    "
                      f"{gt_kappa[i]:>8.1f}  {pred_kappa[i]:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Fixed 4-Peak Model Predictions')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/fixed4peak_20251210_123709/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to use')
    parser.add_argument('--save', action='store_true',
                        help='Save figures')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed category comparison')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = Fixed4PeakModel(encoder_dim=1024).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f})")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A') + 1}")

    # Load dataset
    dataset = Fixed4PeakDataset(
        mode='random',
        split=args.split,
        num_points=2048,
        augment=False  # No augmentation for visualization
    )
    print(f"Dataset: {len(dataset)} samples ({args.split} split)")

    # Print some detailed predictions
    print_sample_predictions(model, dataset, device, num_samples=5)

    # Visualize
    save_dir = Path('visualization/outputs')
    save_dir.mkdir(exist_ok=True)

    if args.detailed:
        save_path = str(save_dir / 'fixed4peak_by_category.png') if args.save else None
        visualize_distribution_comparison(
            model, dataset, device,
            samples_per_category=4,
            save_path=save_path
        )
    else:
        save_path = str(save_dir / 'fixed4peak_predictions.png') if args.save else None
        visualize_predictions(
            model, dataset, device,
            num_samples=args.num_samples,
            save_path=save_path,
            by_category=True
        )


if __name__ == '__main__':
    main()

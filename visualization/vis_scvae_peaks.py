#!/usr/bin/env python
"""
Visualization for sCVAE predictions.

Shows whether the model learned distinct peaks from stochastic sampling.

Usage:
    python visualization/vis_scvae_peaks.py --checkpoint checkpoints/scvae_xxx/best_model.pth
    python visualization/vis_scvae_peaks.py --checkpoint checkpoints/scvae_xxx/best_model.pth --wandb

Author: Claude
Created: 2025-12-08
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import build_model
from datasets import OrientationDataset


def visualize_scvae_predictions(model, dataset, device, num_samples=16,
                                 save_path=None, show=True):
    """
    Visualize sCVAE predictions on a polar plot.

    For each input, shows where the N stochastic samples predict directions.
    If the model learned 4 peaks, we should see clustering at 90° intervals.

    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: torch device
        num_samples: Number of point clouds to visualize
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib figure
    """
    model.eval()

    # Create figure with polar subplots
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows),
                              subplot_kw={'projection': 'polar'})
    axes = axes.flatten() if num_samples > 1 else [axes]

    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            ax = axes[i]

            # Get sample
            sample = dataset[idx]
            points = sample['points'].unsqueeze(0).to(device)
            gt_direction = sample['direction']
            gt_angle = math.atan2(gt_direction[1], gt_direction[0])

            # Get predictions
            output = model(points)
            mu = output['mu'][0]  # (N, 2)
            kappa = output['kappa'][0]  # (N,)

            # Convert to angles
            pred_angles = torch.atan2(mu[:, 1], mu[:, 0]).cpu().numpy()
            kappas = kappa.cpu().numpy()

            # Plot predictions as points
            # Size proportional to kappa (confidence)
            sizes = np.clip(kappas * 10, 20, 200)
            ax.scatter(pred_angles, np.ones_like(pred_angles),
                      s=sizes, c='blue', alpha=0.6, label='Predictions')

            # Plot GT direction
            ax.scatter([gt_angle], [1], s=100, c='red', marker='*',
                      label='GT', zorder=10)

            # Plot ideal 4-fold symmetric directions (if K=4)
            K = sample.get('K', 4)
            if K == 4:
                ideal_angles = [gt_angle + k * math.pi/2 for k in range(4)]
                ideal_angles = [a if a <= math.pi else a - 2*math.pi for a in ideal_angles]
                ax.scatter(ideal_angles, [0.8]*4, s=50, c='green', marker='x',
                          alpha=0.5, label='Ideal 4-fold')

            # Formatting
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_title(f"Sample {idx}\nκ: {kappas.mean():.1f}±{kappas.std():.1f}", fontsize=10)

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"sCVAE Predictions (N={mu.shape[0]} samples per input)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_peak_distribution(model, dataset, device, num_inputs=100,
                                 save_path=None, show=True):
    """
    Aggregate visualization: histogram of all predicted angles.

    If the model learned 4 peaks, the histogram should show 4 clusters.

    Args:
        model: Trained model
        dataset: Dataset
        device: torch device
        num_inputs: Number of point clouds to aggregate
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib figure
    """
    model.eval()

    all_angles = []
    all_kappas = []
    gt_angles = []

    indices = np.random.choice(len(dataset), min(num_inputs, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            points = sample['points'].unsqueeze(0).to(device)
            gt_direction = sample['direction']
            gt_angle = math.atan2(gt_direction[1], gt_direction[0])
            gt_angles.append(gt_angle)

            output = model(points)
            mu = output['mu'][0]  # (N, 2)
            kappa = output['kappa'][0]  # (N,)

            # Convert to angles, normalized relative to GT
            pred_angles = torch.atan2(mu[:, 1], mu[:, 0]).cpu().numpy()
            # Normalize to [0, 2π) relative to GT for comparison
            relative_angles = (pred_angles - gt_angle) % (2 * math.pi)

            all_angles.extend(relative_angles)
            all_kappas.extend(kappa.cpu().numpy())

    all_angles = np.array(all_angles)
    all_kappas = np.array(all_kappas)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of relative angles
    ax1 = axes[0]
    ax1.hist(np.degrees(all_angles), bins=72, range=(0, 360),
             alpha=0.7, color='blue', edgecolor='black')

    # Mark expected peaks at 0°, 90°, 180°, 270°
    for peak in [0, 90, 180, 270]:
        ax1.axvline(peak, color='red', linestyle='--', alpha=0.7, label=f'{peak}°' if peak == 0 else None)

    ax1.set_xlabel('Angle relative to GT (degrees)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of Predicted Angles\n({num_inputs} inputs × N samples each)')
    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.legend(['Expected peaks (4-fold)'])

    # Right: Polar histogram
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    bins = np.linspace(0, 2*np.pi, 73)
    counts, _ = np.histogram(all_angles, bins=bins)
    theta = (bins[:-1] + bins[1:]) / 2
    ax2.bar(theta, counts, width=2*np.pi/72, alpha=0.7, color='blue', edgecolor='black')

    # Mark expected directions
    for peak in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax2.axvline(peak, color='red', linestyle='--', alpha=0.7)

    ax2.set_title('Polar Distribution')

    plt.suptitle('sCVAE Peak Analysis', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    # Print statistics
    print("\n=== Peak Analysis ===")
    print(f"Total predictions: {len(all_angles)}")
    print(f"Mean kappa: {all_kappas.mean():.2f} ± {all_kappas.std():.2f}")

    # Check clustering around 4 expected peaks
    for peak_deg in [0, 90, 180, 270]:
        peak_rad = np.radians(peak_deg)
        # Count predictions within ±22.5° of peak
        diffs = np.abs((all_angles - peak_rad + np.pi) % (2*np.pi) - np.pi)
        near_peak = np.sum(diffs < np.radians(22.5))
        pct = 100 * near_peak / len(all_angles)
        print(f"  Near {peak_deg:3d}°: {near_peak:5d} ({pct:5.1f}%)")

    return fig


def create_wandb_plots(model, dataset, device, num_samples=16, num_inputs=100):
    """
    Create plots suitable for wandb logging.

    Returns:
        dict of wandb.Image objects
    """
    try:
        import wandb
    except ImportError:
        print("wandb not available")
        return {}

    import io
    from PIL import Image

    plots = {}

    # Individual predictions
    fig1 = visualize_scvae_predictions(model, dataset, device,
                                        num_samples=num_samples, show=False)
    buf = io.BytesIO()
    fig1.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['scvae/individual_predictions'] = wandb.Image(Image.open(buf))
    plt.close(fig1)

    # Aggregate distribution
    fig2 = visualize_peak_distribution(model, dataset, device,
                                        num_inputs=num_inputs, show=False)
    buf = io.BytesIO()
    fig2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['scvae/peak_distribution'] = wandb.Image(Image.open(buf))
    plt.close(fig2)

    return plots


def main():
    parser = argparse.ArgumentParser(description='Visualize sCVAE predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply',
                        help='Path to point cloud data')
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json',
                        help='Path to annotations')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of individual samples to visualize')
    parser.add_argument('--num_inputs', type=int, default=100,
                        help='Number of inputs for aggregate analysis')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save figures')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to wandb')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Build model
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model: {config.get('backbone', {}).get('type', 'unknown')} + {config.get('head', {}).get('type', 'unknown')}")

    # Load dataset
    k_filter = config.get('data', {}).get('k_filter', [4])
    dataset = OrientationDataset(
        args.annotation_file,
        args.data_dir,
        num_points=config.get('data', {}).get('num_points', 1024),
        split='test',
        k_filter=k_filter,
        align_pointcloud=True,
        augment=False
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize
    print("\n=== Visualizing Individual Predictions ===")
    fig1 = visualize_scvae_predictions(
        model, dataset, device,
        num_samples=args.num_samples,
        save_path=output_dir / 'scvae_individual.png',
        show=not args.wandb
    )

    print("\n=== Visualizing Peak Distribution ===")
    fig2 = visualize_peak_distribution(
        model, dataset, device,
        num_inputs=args.num_inputs,
        save_path=output_dir / 'scvae_distribution.png',
        show=not args.wandb
    )

    # Wandb logging
    if args.wandb:
        import wandb
        wandb.init(project='forwardnet', name='scvae_visualization')
        plots = create_wandb_plots(model, dataset, device,
                                   num_samples=args.num_samples,
                                   num_inputs=args.num_inputs)
        wandb.log(plots)
        wandb.finish()
        print("\nLogged to wandb")

    print(f"\nFigures saved to: {output_dir}")


if __name__ == '__main__':
    main()

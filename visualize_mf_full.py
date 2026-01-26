#!/usr/bin/env python3
"""
MF Full Model Visualization

For MF Full models, visualize predictions on test samples with:
- Single polar plot (no experts like P2V2)
- Title: "object_name category" (e.g., "chair 1-front")
- Error in degrees shown above the polar plot
- Legend at bottom

Usage:
    python visualize_mf_full.py --checkpoint checkpoints/MF_Full_4Cat_20260107_234055/best_error.pth
    python visualize_mf_full.py --checkpoint checkpoints/MF_Full_4Cat_20260107_234055/best_error.pth --num_samples 5
    python visualize_mf_full.py --checkpoint checkpoints/MF_Full_4Cat_20260107_234055/best_error.pth --category 1_front
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import vonmises
import json
import argparse
import torch
import random

from test_direction_models import DirectionModel

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'cm',
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

CATEGORY_DISPLAY = {
    '1_front': '1-front',
    '2_fronts': '2-front',
    '4_fronts': '4-front',
    'no_front': 'no-front',
}

DIRECTION_TO_ANGLE = {
    '+X': 0.0, '+Z': np.pi / 2, '-X': np.pi, '-Z': 3 * np.pi / 2,
}

SYMMETRY_TO_CAT = {
    '1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
    '旋转对称': 'no_front', '无正面': 'no_front',
}


def load_ply(filepath, num_points=2048):
    """Load PLY point cloud"""
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


def rotate_points_y(points, angle):
    """Rotate points around Y axis (in XZ plane)

    Args:
        points: (N, 3) point cloud
        angle: rotation angle in radians
    Returns:
        rotated points (N, 3)
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return points @ R.T


def load_mf_model(checkpoint_path, num_peaks=4):
    """Load MF Full model"""
    model = DirectionModel(mode='mf', num_peaks=num_peaks).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Handle different checkpoint formats
    if 'encoder_state_dict' in ckpt and 'head_state_dict' in ckpt:
        state_dict = {}
        for k, v in ckpt['encoder_state_dict'].items():
            state_dict[f'encoder.{k}'] = v
        for k, v in ckpt['head_state_dict'].items():
            state_dict[f'head.{k}'] = v
        model.load_state_dict(state_dict)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def mixture_pdf(theta, mus, kappas, weights):
    """Compute von Mises mixture PDF"""
    pdf = np.zeros_like(theta, dtype=float)
    for mu, kappa, w in zip(mus, kappas, weights):
        if kappa < 0.01:
            pdf += w / (2 * np.pi)
        else:
            pdf += w * vonmises.pdf(theta, kappa=kappa, loc=mu)
    return pdf


def render_pointcloud_topdown(points, ax, cmap='coolwarm', point_size=3.0):
    """Render point cloud top-down view"""
    if len(points) > 3000:
        indices = np.random.choice(len(points), 3000, replace=False)
        pts = points[indices]
    else:
        pts = points

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(x, z, c=y, cmap=cmap, s=point_size, alpha=0.9)

    max_range = max(np.abs(pts).max(), 0.5)
    ax.set_xlim(-max_range * 1.1, max_range * 1.1)
    ax.set_ylim(-max_range * 1.1, max_range * 1.1)
    ax.set_aspect('equal')
    ax.axis('off')


def circular_error_deg(pred, gt):
    """Compute circular error in degrees"""
    diff = pred - gt
    return abs(np.degrees(np.arctan2(np.sin(diff), np.cos(diff))))


def hungarian_error_deg(pred_peaks, gt_peaks):
    """Compute mean angular error with Hungarian matching"""
    from scipy.optimize import linear_sum_assignment
    n_pred = len(pred_peaks)
    n_gt = len(gt_peaks)

    # Build cost matrix
    cost = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            diff = pred_peaks[i] - gt_peaks[j]
            cost[i, j] = abs(np.arctan2(np.sin(diff), np.cos(diff)))

    row_ind, col_ind = linear_sum_assignment(cost)
    errors = []
    for i, j in zip(row_ind, col_ind):
        errors.append(np.degrees(cost[i, j]))
    return np.mean(errors)


def visualize_mf_sample(model, points, sample_name, save_path,
                        gt_angle=None, gt_category=None,
                        model_name=None, no_rotate=False):
    """Visualize MF model prediction for one sample

    Args:
        model: MF model
        points: normalized point cloud (N, 3)
        sample_name: sample name for title
        save_path: output path
        gt_angle: ground truth angle (optional)
        gt_category: ground truth category (optional)
        model_name: model name for display
        no_rotate: if True, skip random rotation
    """
    # Random rotation (rotate both point cloud and GT angle together)
    if no_rotate:
        rotation_angle = 0.0
        rotated_points = points
        rotated_gt_angle = gt_angle
    else:
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotated_points = rotate_points_y(points, rotation_angle)
        if gt_angle is not None:
            # GT angle rotates with the point cloud (subtract rotation)
            rotated_gt_angle = (gt_angle - rotation_angle) % (2 * np.pi)
        else:
            rotated_gt_angle = None

    # Forward pass with rotated points
    pts_tensor = torch.from_numpy(rotated_points).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        mu, kappa = model(pts_tensor)

    # Convert mu to angles
    mu_angle = torch.atan2(mu[:, :, 1], mu[:, :, 0])  # (B, num_peaks)
    mu_angle = (mu_angle + 2 * np.pi) % (2 * np.pi)
    mu_np = mu_angle[0].cpu().numpy()  # (4,)
    kappa_np = kappa[0].cpu().numpy()  # (4,)

    # Calculate error using rotated GT angle
    error = None
    if rotated_gt_angle is not None and gt_category is not None:
        if gt_category == '1_front':
            # Find best peak for 1-front
            errors = [circular_error_deg(mu_np[j], rotated_gt_angle) for j in range(4)]
            error = min(errors)
        elif gt_category == '2_fronts':
            gt_peaks = [rotated_gt_angle, (rotated_gt_angle + np.pi) % (2 * np.pi)]
            error = hungarian_error_deg(mu_np.tolist(), gt_peaks)
        elif gt_category == '4_fronts':
            gt_peaks = [(rotated_gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
            error = hungarian_error_deg(mu_np.tolist(), gt_peaks)

    # Create figure
    fig = plt.figure(figsize=(8, 9), facecolor='white')

    # Polar subplot
    ax_polar = fig.add_subplot(111, projection='polar')

    # Compute von Mises mixture PDF
    theta = np.linspace(0, 2 * np.pi, 360)
    weights = kappa_np / kappa_np.sum() if kappa_np.sum() > 0 else np.ones(4) / 4
    pdf = mixture_pdf(theta, mu_np, kappa_np, weights)

    # Normalize for display
    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background ring
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # Mixture curve (fill)
    line_color = '#27ae60'
    fill_color = '#27ae60'
    fill_alpha = 0.3
    pred_line_color = '#3498db'
    gt_line_color = '#e74c3c'

    ax_polar.fill(theta, pdf_display, color=fill_color, alpha=fill_alpha, zorder=3)
    ax_polar.plot(theta, pdf_display, color=line_color, linewidth=2.5, zorder=5)

    # Predicted direction lines (blue) - all 4 peaks
    for mu in mu_np:
        ax_polar.plot(
            [mu, mu], [0.30, 1.05],
            color=pred_line_color, linewidth=2.5,
            solid_capstyle='round',
            zorder=8,
        )

    # GT direction lines (red dashed) - use rotated GT angle
    if rotated_gt_angle is not None and gt_category is not None:
        if gt_category == '1_front':
            gt_angles = [rotated_gt_angle]
        elif gt_category == '2_fronts':
            gt_angles = [rotated_gt_angle, (rotated_gt_angle + np.pi) % (2 * np.pi)]
        elif gt_category == '4_fronts':
            gt_angles = [(rotated_gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        else:
            gt_angles = []

        for angle in gt_angles:
            ax_polar.plot(
                [angle, angle], [0.30, 1.05],
                color=gt_line_color, linewidth=2.5,
                linestyle='--',
                solid_capstyle='round',
                zorder=10,
            )

    # Polar style
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([0.5, 0.7, 0.9])
    ax_polar.set_yticklabels(['', '', ''])
    ax_polar.set_xticks([])
    ax_polar.set_xticklabels([])
    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # Center point cloud
    pos = ax_polar.get_position()
    pc_size = 0.30
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2

    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=2, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(rotated_points, ax_center, point_size=3.0)

    # Axis indicators
    arrow_len = 0.15
    ax_center.annotate('', xy=(0.5 + arrow_len, 0.5), xytext=(0.5, 0.5),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      annotation_clip=False)
    ax_center.annotate('', xy=(0.5, 0.5 + arrow_len), xytext=(0.5, 0.5),
                      arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                      annotation_clip=False)
    ax_center.text(0.5 + arrow_len + 0.03, 0.5, 'X', fontsize=10, color='red',
                  ha='left', va='center', transform=ax_center.transAxes, fontweight='bold')
    ax_center.text(0.5, 0.5 + arrow_len + 0.03, 'Z', fontsize=10, color='blue',
                  ha='center', va='bottom', transform=ax_center.transAxes, fontweight='bold')

    # Title with error
    obj_name = sample_name.split('/')[0]
    cat_display = CATEGORY_DISPLAY.get(gt_category, gt_category) if gt_category else ''
    main_title = f'{obj_name}  {cat_display}'

    # Subtitle with error
    if error is not None:
        subtitle = f'Error: {error:.1f}°'
    else:
        subtitle = ''

    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.97)
    if subtitle:
        ax_polar.set_title(subtitle, fontsize=15, pad=20)

    # Legend at bottom center
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=3, label=r'Predicted $\hat{\phi}$'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, linestyle='--', label=r'GT $\phi_{gt}$'),
        Patch(facecolor='#27ae60', alpha=0.3, edgecolor='#27ae60', linewidth=2, label=r'$p(\phi)$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=3, framealpha=0.95, edgecolor='#cccccc', fontsize=12)

    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {save_path}")
    plt.close(fig)

    return {
        'mu': mu_np,
        'kappa': kappa_np,
        'error': error,
    }


def load_test_samples():
    """Load test set samples"""
    test_set_path = project_root / 'data/test_set/test_set.json'
    if test_set_path.exists():
        with open(test_set_path) as f:
            return json.load(f)

    # Fallback to annotations
    with open(project_root / 'data_annotation/symmetry_annotations.json') as f:
        annotations = json.load(f)

    samples = []
    for file_path, ann in annotations.items():
        symmetry_name = ann.get('symmetry_name', '')
        if symmetry_name not in SYMMETRY_TO_CAT:
            continue
        category = SYMMETRY_TO_CAT[symmetry_name]
        direction = ann.get('front_direction')
        gt_angle = DIRECTION_TO_ANGLE.get(direction) if direction else None

        samples.append({
            'file': file_path,
            'category': category,
            'front_direction': direction,
            'gt_angle': gt_angle,
            'symmetry_name': symmetry_name,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description='Visualize MF Full model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sample', type=str, help='Specific sample path')
    parser.add_argument('--category', type=str,
                       choices=['1_front', '2_fronts', '4_fronts', 'no_front'],
                       help='Filter by GT category')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='paper_figures/mf_full')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_peaks', type=int, default=4,
                       help='Number of peaks in MF model')
    parser.add_argument('--no_rotate', action='store_true',
                       help='Do not randomly rotate point cloud and GT')

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load model
    print(f"Loading MF model from: {args.checkpoint}")
    model = load_mf_model(args.checkpoint, args.num_peaks)
    print("Model loaded successfully!")

    # Get model name from checkpoint path
    ckpt_path = Path(args.checkpoint)
    model_name = ckpt_path.parent.name

    # Load samples
    samples = load_test_samples()
    print(f"Loaded {len(samples)} samples")

    # Organize by category
    samples_by_cat = {'1_front': [], '2_fronts': [], '4_fronts': [], 'no_front': []}
    for s in samples:
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
            gt_angle = None

        samples_by_cat[cat].append({
            'file': s['file'],
            'category': cat,
            'gt_angle': gt_angle,
        })

    for cat, lst in samples_by_cat.items():
        print(f"  {cat}: {len(lst)} samples")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select samples
    selected = []

    if args.sample:
        for cat, lst in samples_by_cat.items():
            for s in lst:
                if s['file'] == args.sample:
                    selected.append(s)
                    break
            if selected:
                break
        if not selected:
            print(f"Sample not found: {args.sample}")
            return
    elif args.category:
        candidates = samples_by_cat.get(args.category, [])
        if not candidates:
            print(f"No samples for category: {args.category}")
            return
        num = min(args.num_samples, len(candidates))
        indices = np.random.choice(len(candidates), num, replace=False)
        selected = [candidates[i] for i in indices]
    else:
        # Select from all directional categories
        all_directional = samples_by_cat['1_front'] + samples_by_cat['2_fronts'] + samples_by_cat['4_fronts']
        num = min(args.num_samples, len(all_directional))
        indices = np.random.choice(len(all_directional), num, replace=False)
        selected = [all_directional[i] for i in indices]

    # Visualize
    data_root = Path('data/full_mn40_normal_resampled_ply')

    print(f"\nVisualizing {len(selected)} samples...")
    print("=" * 60)

    for i, sample in enumerate(selected):
        ply_path = data_root / sample['file']
        if not ply_path.exists():
            print(f"File not found: {ply_path}")
            continue

        # Load point cloud
        points = load_ply(str(ply_path), NUM_POINTS)

        # Output filename
        safe_name = sample['file'].replace('/', '_').replace('.ply', '')
        save_path = output_dir / f'{safe_name}_mf.png'

        # Visualize
        result = visualize_mf_sample(
            model=model,
            points=points,
            sample_name=sample['file'],
            save_path=save_path,
            gt_angle=sample['gt_angle'],
            gt_category=sample['category'],
            model_name=model_name,
            no_rotate=args.no_rotate,
        )

        # Print summary
        print(f"\n[{i+1}] {sample['file']}")
        print(f"    GT Category: {sample['category']}")
        if result['error'] is not None:
            print(f"    Error: {result['error']:.1f}")
        print(f"    mu: [{', '.join([f'{np.degrees(m):.1f}' for m in result['mu']])}]")
        print(f"    kappa: [{', '.join([f'{k:.1f}' for k in result['kappa']])}]")

    print("\n" + "=" * 60)
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()

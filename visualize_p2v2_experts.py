#!/usr/bin/env python3
"""
P2V2 Clean 3-Expert Head Visualization

For each object, visualize all 3 expert heads' predictions:
- head_1front: 1 peak von Mises
- head_2front: 2 peak von Mises mixture
- head_4front: 4 peak von Mises mixture

Also shows classifier prediction (1-front, 2-front, 4-front, or no-front)

Usage:
    python visualize_p2v2_experts.py --sample airplane/airplane_0001.ply
    python visualize_p2v2_experts.py --num_samples 5
    python visualize_p2v2_experts.py --category 1_front --num_samples 3
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

from models.probabilistic_orientation_net import ProbabilisticOrientationNet
from train_symmetry_classifier import SymmetryClassifier

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

# Category mapping
CATEGORY_NAMES = {
    0: '1-front',
    1: '2-front',
    2: '4-front',
    3: 'rotational',
    4: 'no-front',
}

DIRECTION_TO_ANGLE = {
    '+X': 0.0, '+Z': np.pi / 2, '-X': np.pi, '-Z': 3 * np.pi / 2,
}

SYMMETRY_TO_CAT = {
    '1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
    '旋转对称': 'no_front', '无正面': 'no_front',
}


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    def forward(self, points, upright_vec=None):
        return self.classifier(points)


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


def random_rotate_y(points, gt_angle=None):
    """Apply random rotation around Y axis to point cloud and gt angle

    Args:
        points: (N, 3) point cloud
        gt_angle: ground truth angle in radians (optional)

    Returns:
        rotated_points: (N, 3) rotated point cloud
        rotated_gt_angle: rotated gt angle (or None if gt_angle was None)
        rotation_angle: the random rotation angle applied
    """
    # Random rotation angle in [0, 2*pi)
    rotation_angle = np.random.uniform(0, 2 * np.pi)

    # Rotation matrix around Y axis
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)

    # Rotate point cloud
    rotated_points = points @ R.T

    # Rotate gt angle (subtract rotation since we rotated the object)
    # If we rotate the object by rotation_angle, the "front" direction
    # in the world frame stays the same, but relative to the rotated object
    # it appears to have rotated in the opposite direction
    if gt_angle is not None:
        rotated_gt_angle = (gt_angle - rotation_angle) % (2 * np.pi)
    else:
        rotated_gt_angle = None

    return rotated_points, rotated_gt_angle, rotation_angle


def load_p2v2_model(checkpoint_path):
    """Load P2V2 model"""
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


def plot_single_expert_mvm_style(fig, ax_polar, mus, kappas, weights, points, title,
                                  gt_angle=None, gt_category=None,
                                  line_color='#27ae60', fill_color='#27ae60', fill_alpha=0.3,
                                  pred_line_color='#3498db', gt_line_color='#e74c3c'):
    """Plot single expert head's von Mises mixture in MVM style

    Large centered point cloud with polar distribution around it.
    """
    # Compute PDF
    theta = np.linspace(0, 2 * np.pi, 360)
    weights_norm = np.array(weights) / np.sum(weights)
    pdf = mixture_pdf(theta, mus, kappas, weights_norm)

    # Normalize for display (map to 0.35-0.9 range like mvm)
    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background ring
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # Mixture curve (fill)
    ax_polar.fill(theta, pdf_display, color=fill_color, alpha=fill_alpha, zorder=3)
    ax_polar.plot(theta, pdf_display, color=line_color, linewidth=2.5, zorder=5)

    # Predicted direction lines (blue)
    for mu in mus:
        ax_polar.plot(
            [mu, mu], [0.30, 1.05],
            color=pred_line_color, linewidth=2.5,
            solid_capstyle='round',
            zorder=8,
        )

    # GT direction lines (red) if provided
    if gt_angle is not None and gt_category is not None:
        if gt_category == '1_front':
            gt_angles = [gt_angle]
        elif gt_category == '2_fronts':
            gt_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        elif gt_category == '4_fronts':
            gt_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
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

    # Get axis position for centered point cloud
    pos = ax_polar.get_position()
    pc_size = 0.30  # 30% of subplot size (like mvm)
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2

    # Point cloud inset (large, centered)
    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=2, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(points, ax_center, point_size=3.0)

    # Axis indicators (larger)
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


def visualize_3_experts(model, points, sample_name, save_path,
                        gt_angle=None, gt_category=None):
    """Visualize all 3 expert heads for one sample

    Args:
        model: P2V2 model
        points: normalized point cloud (N, 3)
        sample_name: sample name for title
        save_path: output path
        gt_angle: ground truth angle (optional)
        gt_category: ground truth category (optional)
    """
    # Forward pass
    pts_tensor = torch.from_numpy(points).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(pts_tensor)

    # Extract predictions
    # 1-front head
    mu_1f = output['head_1front']['mu'][0].cpu().numpy()  # [1]
    kappa_1f = output['head_1front']['kappa'][0].cpu().numpy()  # [1]

    # 2-front head
    mu_2f = output['head_2front']['mu'][0].cpu().numpy()  # [2]
    kappa_2f = output['head_2front']['kappa'][0].cpu().numpy()  # [2]

    # 4-front head
    mu_4f = output['head_4front']['mu'][0].cpu().numpy()  # [4]
    kappa_4f = output['head_4front']['kappa'][0].cpu().numpy()  # [4]

    # Classifier weights
    weights = output['weights'][0].cpu().numpy()  # [5]
    pred_class = np.argmax(weights)
    pred_label = CATEGORY_NAMES[pred_class]

    # Classifier confidence for each directional category
    p_1f = weights[0]
    p_2f = weights[1]
    p_4f = weights[2]

    # Calculate angular errors for each head
    def circular_error_deg(pred, gt):
        """Compute circular error in degrees"""
        diff = pred - gt
        return abs(np.degrees(np.arctan2(np.sin(diff), np.cos(diff))))

    def hungarian_error_deg(pred_peaks, gt_peaks):
        """Compute mean angular error with Hungarian matching"""
        from scipy.optimize import linear_sum_assignment
        n = len(pred_peaks)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = 1 - np.cos(pred_peaks[i] - gt_peaks[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        errors = []
        for i, j in zip(row_ind, col_ind):
            errors.append(circular_error_deg(pred_peaks[i], gt_peaks[j]))
        return np.mean(errors)

    # Compute errors for each head (using GT angle)
    if gt_angle is not None:
        error_1f = circular_error_deg(mu_1f[0], gt_angle)
        gt_2f = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        error_2f = hungarian_error_deg(list(mu_2f), gt_2f)
        gt_4f = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        error_4f = hungarian_error_deg(list(mu_4f), gt_4f)
    else:
        error_1f = error_2f = error_4f = None

    # Create figure with 3 subplots - larger size for better visibility
    fig = plt.figure(figsize=(18, 8), facecolor='white')

    # Create subplots with proper spacing - more room at top for titles
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.78, bottom=0.08, wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')

    # Determine which expert is correct based on GT category
    correct_expert = None
    if gt_category == '1_front':
        correct_expert = 1
    elif gt_category == '2_fronts':
        correct_expert = 2
    elif gt_category == '4_fronts':
        correct_expert = 4

    # Plot 1-front head
    plot_single_expert_mvm_style(
        fig, ax1,
        mus=list(mu_1f),
        kappas=list(kappa_1f),
        weights=[1.0],
        points=points,
        title=f'1-Front Head',
        gt_angle=gt_angle,
        gt_category=gt_category if gt_category == '1_front' else None,
    )

    # Plot 2-front head
    plot_single_expert_mvm_style(
        fig, ax2,
        mus=list(mu_2f),
        kappas=list(kappa_2f),
        weights=[0.5, 0.5],
        points=points,
        title=f'2-Front Head',
        gt_angle=gt_angle,
        gt_category=gt_category if gt_category == '2_fronts' else None,
    )

    # Plot 4-front head
    plot_single_expert_mvm_style(
        fig, ax3,
        mus=list(mu_4f),
        kappas=list(kappa_4f),
        weights=[0.25, 0.25, 0.25, 0.25],
        points=points,
        title=f'4-Front Head',
        gt_angle=gt_angle,
        gt_category=gt_category if gt_category == '4_fronts' else None,
    )

    # Subplot titles with error
    def make_title(name, error):
        if error is not None:
            return f'{name}\nError: {error:.1f}°'
        return name

    ax1.set_title(make_title('1-Front', error_1f), fontsize=15, pad=20)
    ax2.set_title(make_title('2-Front', error_2f), fontsize=15, pad=20)
    ax3.set_title(make_title('4-Front', error_4f), fontsize=15, pad=20)

    # Add red dashed border around the correct expert (only around the polar plot, not the title)
    import matplotlib.patches as mpatches
    axes_list = [ax1, ax2, ax3]
    expert_nums = [1, 2, 4]
    for ax, exp_num in zip(axes_list, expert_nums):
        if correct_expert == exp_num:
            # Get the axis position and add a rectangle border (below the title)
            pos = ax.get_position()
            # Adjust to only wrap the polar plot area, not the title
            rect = mpatches.FancyBboxPatch(
                (pos.x0 - 0.01, pos.y0 - 0.01),
                pos.width + 0.02, pos.height + 0.02,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=3, edgecolor='#e74c3c', facecolor='none',
                linestyle='--', transform=fig.transFigure, zorder=100
            )
            fig.patches.append(rect)

    # Main title - single line: object_name category
    gt_label_map = {'1_front': '1-front', '2_fronts': '2-front', '4_fronts': '4-front', 'no_front': 'no-front'}
    if gt_category:
        gt_label = gt_label_map.get(gt_category, gt_category)
        # Extract object name (e.g., "chair" from "chair/chair_0094.ply")
        obj_name = sample_name.split('/')[0]
        main_title = f'{obj_name}  {gt_label}'
    else:
        main_title = sample_name

    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.97)

    # Legend at bottom center
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=3, label=r'Predicted $\hat{\phi}$'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, linestyle='--', label=r'GT $\phi_{gt}$'),
        Patch(facecolor='#27ae60', alpha=0.3, edgecolor='#27ae60', linewidth=2, label=r'$p(\phi)$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=3, framealpha=0.95, edgecolor='#cccccc', fontsize=14)

    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {save_path}")
    plt.close(fig)

    return {
        'pred_label': pred_label,
        'pred_class': pred_class,
        'weights': weights,
        'mu_1f': mu_1f,
        'kappa_1f': kappa_1f,
        'mu_2f': mu_2f,
        'kappa_2f': kappa_2f,
        'mu_4f': mu_4f,
        'kappa_4f': kappa_4f,
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
    parser = argparse.ArgumentParser(description='Visualize P2V2 3 expert heads')
    parser.add_argument('--sample', type=str, help='Specific sample path (e.g., airplane/airplane_0001.ply)')
    parser.add_argument('--category', type=str,
                       choices=['1_front', '2_fronts', '4_fronts', 'no_front'],
                       help='Filter by GT category')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--all', action='store_true', help='Visualize all directional samples (1/2/4-front)')
    parser.add_argument('--output_dir', type=str, default='paper_figures/p2v2_experts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_path', type=str,
                       default='checkpoints/P2v2_Clean_20251230_165848/best.pth',
                       help='P2V2 model checkpoint path')

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load model
    print(f"Loading P2V2 model from: {args.model_path}")
    model = load_p2v2_model(args.model_path)
    print("Model loaded successfully!")

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

    # Select samples to visualize
    selected = []

    if args.sample:
        # Specific sample
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
    elif getattr(args, 'all', False):
        # All directional samples
        all_directional = samples_by_cat['1_front'] + samples_by_cat['2_fronts'] + samples_by_cat['4_fronts']
        selected = all_directional
        print(f"Processing ALL {len(selected)} directional samples...")
    elif args.category:
        # Specific category
        candidates = samples_by_cat.get(args.category, [])
        if not candidates:
            print(f"No samples for category: {args.category}")
            return
        num = min(args.num_samples, len(candidates))
        indices = np.random.choice(len(candidates), num, replace=False)
        selected = [candidates[i] for i in indices]
    else:
        # Random from all categories
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

        # Apply random rotation to point cloud and gt angle
        points, rotated_gt_angle, rot_angle = random_rotate_y(points, sample['gt_angle'])
        print(f"  Applied random rotation: {np.degrees(rot_angle):.1f}°")

        # Output filename
        safe_name = sample['file'].replace('/', '_').replace('.ply', '')
        save_path = output_dir / f'{safe_name}_3experts.png'

        # Visualize with rotated point cloud and rotated gt angle
        result = visualize_3_experts(
            model=model,
            points=points,
            sample_name=sample['file'],
            save_path=save_path,
            gt_angle=rotated_gt_angle,
            gt_category=sample['category'],
        )

        # Print summary
        print(f"\n[{i+1}] {sample['file']}")
        print(f"    GT Category: {sample['category']}")
        print(f"    Predicted: {result['pred_label']}")
        print(f"    Classifier weights: 1f={result['weights'][0]:.3f}, 2f={result['weights'][1]:.3f}, "
              f"4f={result['weights'][2]:.3f}, rot={result['weights'][3]:.3f}, no={result['weights'][4]:.3f}")
        print(f"    1-front head: mu={np.degrees(result['mu_1f'][0]):.1f}deg, kappa={result['kappa_1f'][0]:.1f}")
        print(f"    2-front head: mu=[{np.degrees(result['mu_2f'][0]):.1f}, {np.degrees(result['mu_2f'][1]):.1f}]deg")
        print(f"    4-front head: mu=[" + ", ".join([f"{np.degrees(m):.1f}" for m in result['mu_4f']]) + "]deg")

    print("\n" + "=" * 60)
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()

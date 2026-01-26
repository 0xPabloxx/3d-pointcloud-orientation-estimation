#!/usr/bin/env python3
"""
D/DR Series Visualization Script

For each sample, visualize the discrete direction prediction from D/DR series models.
Shows predicted probability distribution as polar bar chart with GT direction.

Format matches p2v2_experts style:
- Top: object_name  symmetry_type (e.g., "chair  1-front")
- Single polar plot with predicted distribution
- Bottom: legend and Error in degrees

Usage:
    python visualize_d_series.py --model D_8b --num_samples 5
    python visualize_d_series.py --model D_16a --category 1_front --num_samples 3
    python visualize_d_series.py --all_models --num_samples 2
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
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    '旋转对称': 'no_front', '无正面': 'no_front', '完全对称': 'no_front',
}

# ============================================================================
# Model Definitions (same as test_direction_models.py)
# ============================================================================
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0].permute(0, 2, 1)
        return new_xyz, new_points


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(32, 0.8, 128, 256 + 3, [256, 512, out_channels])
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, points):
        xyz = points
        feat = None
        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        xyz, feat = self.sa3(xyz, feat)
        x = feat.max(dim=1)[0]
        x = self.fc(x)
        return x


class DiscreteDirectionHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, num_bins=8):
        super().__init__()
        self.num_bins = num_bins
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_bins),
        )

    def forward(self, x):
        return self.head(x)


class ProjectionRegressionHead(nn.Module):
    """DR series: projection regression with tanh"""
    def __init__(self, in_channels=1024, hidden_channels=512, num_dirs=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_dirs),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.head(x)


class DirectionModel(nn.Module):
    """Unified direction prediction model"""
    def __init__(self, mode='discrete', num_bins=8):
        super().__init__()
        self.mode = mode
        self.num_bins = num_bins
        self.encoder = PointNetPlusPlusEncoder()

        if mode == 'discrete':
            self.head = DiscreteDirectionHead(num_bins=num_bins)
        elif mode == 'dr':
            self.head = ProjectionRegressionHead(num_dirs=num_bins)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# ============================================================================
# Data loading utilities
# ============================================================================
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
    """Rotate point cloud around Y axis (in XZ plane)

    Args:
        points: (N, 3) point cloud
        angle: rotation angle in radians (positive = counter-clockwise when viewed from above)

    Returns:
        rotated points (N, 3)
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return points @ R.T


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


# ============================================================================
# Model loading
# ============================================================================
def find_model_checkpoint(model_name, checkpoints_dir='checkpoints'):
    """Find model checkpoint path"""
    checkpoints_path = Path(checkpoints_dir)

    # Try exact match first
    for d in checkpoints_path.iterdir():
        if d.is_dir() and model_name in d.name:
            # Check for best_error.pth or best.pth
            if (d / 'best_error.pth').exists():
                return d / 'best_error.pth', d.name
            elif (d / 'best.pth').exists():
                return d / 'best.pth', d.name

    return None, None


def load_d_model(checkpoint_path, mode='discrete', num_bins=8):
    """Load D/DR model"""
    model = DirectionModel(mode=mode, num_bins=num_bins)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Handle different checkpoint formats
    if 'encoder_state_dict' in checkpoint and 'head_state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['encoder_state_dict'].items():
            state_dict[f'encoder.{k}'] = v
        for k, v in checkpoint['head_state_dict'].items():
            state_dict[f'head.{k}'] = v
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"  [Warning] State dict mismatch: {e}")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  [Info] Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# Error computation
# ============================================================================
def angular_distance(pred_angle, gt_angle, symmetry_k=1):
    """Compute angular error considering symmetry"""
    if symmetry_k <= 0:  # no_front
        return 0.0

    min_error = float('inf')
    for i in range(symmetry_k):
        candidate = (gt_angle + i * 2 * np.pi / symmetry_k) % (2 * np.pi)
        diff = pred_angle - candidate
        error = abs(np.arctan2(np.sin(diff), np.cos(diff)))
        min_error = min(min_error, error)

    return min_error


# ============================================================================
# Visualization
# ============================================================================
def plot_d_series_prediction(
    fig, ax_polar, pred_probs, points, num_bins,
    gt_angle=None, gt_category=None, model_name='',
    line_color='#27ae60', fill_color='#27ae60', fill_alpha=0.3,
    pred_line_color='#3498db', gt_line_color='#e74c3c',
    num_pred_lines=1
):
    """Plot D-series discrete prediction as polar bar chart

    Similar to MVM style but with discrete bars instead of continuous curve.
    num_pred_lines: number of top-k predicted directions to draw (1, 2, or 4)
    """
    bin_width = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * bin_width

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background ring
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # Bin backgrounds (alternating colors)
    for i in range(num_bins):
        color = '#f0f4f8' if i % 2 == 0 else '#fafbfc'
        ax_polar.bar(
            bin_centers[i], 0.65,
            width=bin_width * 0.95,
            bottom=0.35,
            color=color,
            edgecolor='#d0d7de',
            linewidth=0.5,
            zorder=1,
        )

    # Predicted distribution bars
    pred_probs_np = pred_probs.cpu().numpy() if isinstance(pred_probs, torch.Tensor) else pred_probs
    pred_max = pred_probs_np.max() if pred_probs_np.max() > 0 else 1.0
    pred_display = pred_probs_np / pred_max * 0.55  # Normalize to fit in display range

    # Color gradient based on probability
    colors = plt.cm.Greens(0.35 + pred_probs_np / pred_max * 0.55)

    ax_polar.bar(
        bin_centers, pred_display,
        width=bin_width * 0.80,
        bottom=0.35,
        color=colors,
        edgecolor='#1e8449',
        linewidth=1.2,
        zorder=3,
    )

    # Predicted direction lines (blue) - top-k argmax
    top_k_indices = np.argsort(pred_probs_np)[::-1][:num_pred_lines]
    pred_angles = [idx * bin_width for idx in top_k_indices]

    for i, pred_angle in enumerate(pred_angles):
        # First line (highest prob) is solid and thicker
        if i == 0:
            ax_polar.plot(
                [pred_angle, pred_angle], [0.30, 1.05],
                color=pred_line_color, linewidth=2.5,
                solid_capstyle='round',
                zorder=8,
            )
        else:
            # Other lines are slightly thinner
            ax_polar.plot(
                [pred_angle, pred_angle], [0.30, 1.05],
                color=pred_line_color, linewidth=2.0,
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
    pc_size = 0.30
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2

    # Point cloud inset (large, centered)
    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=2, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(points, ax_center, point_size=3.0)

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

    return pred_angles[0]  # Return the top-1 predicted angle


def visualize_d_series_sample(
    model, points, sample_name, save_path,
    gt_angle=None, gt_category=None, num_bins=8, model_name='', mode='discrete',
    multi_pred_lines=False
):
    """Visualize D-series prediction for one sample

    Args:
        model: D/DR model
        points: normalized point cloud (N, 3)
        sample_name: sample name for title (e.g., "chair/chair_0094.ply")
        save_path: output path
        gt_angle: ground truth angle (optional, will be rotated)
        gt_category: ground truth category (e.g., '1_front', '2_fronts', '4_fronts')
        num_bins: number of bins (8 or 16)
        model_name: model name for display
        mode: 'discrete' or 'dr'
        multi_pred_lines: if True, draw top-k prediction lines based on gt_category
                          (2 for 2_fronts, 4 for 4_fronts)
    """
    # Random rotation around Y axis
    rotation_angle = np.random.uniform(0, 2 * np.pi)

    # Rotate point cloud
    rotated_points = rotate_points_y(points, rotation_angle)

    # Rotate GT angle (if exists)
    # When point cloud rotates counter-clockwise by angle, the GT direction
    # relative to the rotated point cloud shifts by -angle
    if gt_angle is not None:
        rotated_gt_angle = (gt_angle - rotation_angle) % (2 * np.pi)
    else:
        rotated_gt_angle = None

    # Forward pass with rotated points
    pts_tensor = torch.from_numpy(rotated_points).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(pts_tensor)

    # Get probabilities
    pred_probs = F.softmax(output[0], dim=-1).cpu().numpy()

    # Get predicted angle
    pred_idx = np.argmax(pred_probs)
    bin_width = 2 * np.pi / num_bins
    pred_angle = pred_idx * bin_width

    # Compute error using rotated GT angle
    if rotated_gt_angle is not None and gt_category in ['1_front', '2_fronts', '4_fronts']:
        if gt_category == '1_front':
            k = 1
        elif gt_category == '2_fronts':
            k = 2
        elif gt_category == '4_fronts':
            k = 4
        else:
            k = 0
        error_rad = angular_distance(pred_angle, rotated_gt_angle, k)
        error_deg = np.degrees(error_rad)
    else:
        error_deg = None

    # Create figure - similar size to p2v2_experts but single plot
    fig = plt.figure(figsize=(8, 8), facecolor='white')

    # Main polar plot
    ax_polar = fig.add_subplot(111, projection='polar')

    # Determine number of prediction lines to draw
    if multi_pred_lines:
        if gt_category == '2_fronts':
            num_pred_lines = 2
        elif gt_category == '4_fronts':
            num_pred_lines = 4
        else:
            num_pred_lines = 1
    else:
        num_pred_lines = 1

    # Use rotated points and rotated GT angle for visualization
    plot_d_series_prediction(
        fig, ax_polar, pred_probs, rotated_points, num_bins,
        gt_angle=rotated_gt_angle, gt_category=gt_category,
        model_name=model_name, num_pred_lines=num_pred_lines
    )

    # Title with error
    if error_deg is not None:
        ax_polar.set_title(f'Error: {error_deg:.1f}°', fontsize=15, pad=20)

    # Main title - object_name  category
    gt_label_map = {'1_front': '1-front', '2_fronts': '2-front', '4_fronts': '4-front', 'no_front': 'no-front'}
    if gt_category:
        gt_label = gt_label_map.get(gt_category, gt_category)
        obj_name = sample_name.split('/')[0]
        main_title = f'{obj_name}  {gt_label}'
    else:
        main_title = sample_name

    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.97)

    # Legend at bottom center
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=3, label=r'Predicted $\hat{\phi}$'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, linestyle='--', label=r'GT $\phi_{gt}$'),
        Patch(facecolor='#27ae60', alpha=0.5, edgecolor='#1e8449', linewidth=2, label=r'$p(\phi)$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, framealpha=0.95, edgecolor='#cccccc', fontsize=12)

    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {save_path}")
    plt.close(fig)

    return {
        'pred_angle_deg': np.degrees(pred_angle),
        'rotated_gt_angle_deg': np.degrees(rotated_gt_angle) if rotated_gt_angle is not None else None,
        'rotation_angle_deg': np.degrees(rotation_angle),
        'error_deg': error_deg,
        'pred_probs': pred_probs,
    }


def load_test_samples():
    """Load test set samples"""
    test_set_path = project_root / 'data/test_set/test_set.json'
    if test_set_path.exists():
        with open(test_set_path) as f:
            return json.load(f)
    return []


def get_model_config(model_name):
    """Get model configuration from name"""
    # Default config
    mode = 'discrete'
    num_bins = 8

    # Parse from name
    if 'DR' in model_name or model_name.startswith('DR_'):
        mode = 'dr'

    if '16' in model_name:
        num_bins = 16
    elif '8' in model_name:
        num_bins = 8

    return mode, num_bins


def main():
    parser = argparse.ArgumentParser(description='Visualize D/DR series predictions')
    parser.add_argument('--model', type=str, default='D_8b',
                        help='Model name (e.g., D_8b, D_16a, DR_8a)')
    parser.add_argument('--sample', type=str,
                        help='Specific sample path (e.g., airplane/airplane_0001.ply)')
    parser.add_argument('--category', type=str,
                        choices=['1_front', '2_fronts', '4_fronts', 'no_front'],
                        help='Filter by GT category')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--all_models', action='store_true',
                        help='Visualize with all D/DR models')
    parser.add_argument('--all', action='store_true',
                        help='Visualize ALL samples in test set (directional categories)')
    parser.add_argument('--multi_pred', action='store_true',
                        help='Draw top-k prediction lines (2 for 2_fronts, 4 for 4_fronts)')
    parser.add_argument('--output_dir', type=str, default='paper_figures/d_series_vis')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models to test
    if args.all_models:
        models_to_test = ['D_8a', 'D_8b', 'D_8c', 'D_8d', 'D_16a', 'D_16b',
                         'DR_8a', 'DR_8b', 'DR_16a']
    else:
        models_to_test = [args.model]

    # Load samples
    samples = load_test_samples()
    print(f"Loaded {len(samples)} test samples")

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

    # Select samples
    if args.sample:
        # Specific sample
        selected = []
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
        # Specific category
        candidates = samples_by_cat.get(args.category, [])
        if not candidates:
            print(f"No samples for category: {args.category}")
            return
        if args.all:
            # All samples of this category
            selected = candidates
            print(f"  Selected ALL {len(selected)} samples from {args.category}")
        else:
            num = min(args.num_samples, len(candidates))
            indices = np.random.choice(len(candidates), num, replace=False)
            selected = [candidates[i] for i in indices]
    elif args.all:
        # ALL samples from directional categories
        all_directional = samples_by_cat['1_front'] + samples_by_cat['2_fronts'] + samples_by_cat['4_fronts']
        selected = all_directional
        print(f"  Selected ALL {len(selected)} directional samples")
    else:
        # Random from directional categories
        all_directional = samples_by_cat['1_front'] + samples_by_cat['2_fronts'] + samples_by_cat['4_fronts']
        num = min(args.num_samples, len(all_directional))
        indices = np.random.choice(len(all_directional), num, replace=False)
        selected = [all_directional[i] for i in indices]

    print(f"\nVisualizing {len(selected)} samples with {len(models_to_test)} models...")
    print("=" * 60)

    data_root = Path('data/full_mn40_normal_resampled_ply')

    # Process each model
    for model_name in models_to_test:
        print(f"\n=== Model: {model_name} ===")

        # Find checkpoint
        checkpoint_path, full_name = find_model_checkpoint(model_name, args.checkpoints_dir)
        if checkpoint_path is None:
            print(f"  [Error] Checkpoint not found for {model_name}")
            continue

        print(f"  Checkpoint: {checkpoint_path}")

        # Get model config
        mode, num_bins = get_model_config(full_name)
        print(f"  Mode: {mode}, Bins: {num_bins}")

        # Load model
        try:
            model = load_d_model(checkpoint_path, mode=mode, num_bins=num_bins)
            print(f"  Model loaded successfully!")
        except Exception as e:
            print(f"  [Error] Failed to load model: {e}")
            continue

        # Create output subdir for this model
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Visualize each sample
        for i, sample in enumerate(selected):
            ply_path = data_root / sample['file']
            if not ply_path.exists():
                print(f"  [Warning] File not found: {ply_path}")
                continue

            # Load point cloud
            points = load_ply(str(ply_path), NUM_POINTS)

            # Output filename
            safe_name = sample['file'].replace('/', '_').replace('.ply', '')
            save_path = model_output_dir / f'{safe_name}_{model_name}.png'

            # Visualize
            result = visualize_d_series_sample(
                model=model,
                points=points,
                sample_name=sample['file'],
                save_path=save_path,
                gt_angle=sample['gt_angle'],
                gt_category=sample['category'],
                num_bins=num_bins,
                model_name=model_name,
                mode=mode,
                multi_pred_lines=args.multi_pred,
            )

            # Print summary
            print(f"\n  [{i+1}] {sample['file']}")
            print(f"      GT Category: {sample['category']}")
            print(f"      Predicted: {result['pred_angle_deg']:.1f}°")
            if result['error_deg'] is not None:
                print(f"      Error: {result['error_deg']:.1f}°")

    print("\n" + "=" * 60)
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()

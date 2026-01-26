#!/usr/bin/env python3
"""
Three Models Comparison Visualization

For each sample, visualize predictions from 3 models side by side:
- D16a: Discrete 16-bin prediction (bar chart)
- MF_mu_only: Mixture of 4 von Mises peaks
- P2v2_Clean: Expert head matching GT category

Style matches existing paper_figures visualizations (mvm, d_series, p2v2_experts).

Usage:
    python visualize_three_models_comparison.py --top_k 10
    python visualize_three_models_comparison.py --samples chair/chair_0412.ply,chair/chair_0685.ply
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
import matplotlib.patches as mpatches
from scipy.stats import vonmises
from scipy.optimize import linear_sum_assignment
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

DIRECTION_TO_ANGLE = {
    '+X': 0.0, '+Z': np.pi / 2, '-X': np.pi, '-Z': 3 * np.pi / 2,
}

SYMMETRY_TO_CAT = {
    '1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
    '旋转对称': 'no_front', '无正面': 'no_front', '完全对称': 'no_front',
}

CATEGORY_DISPLAY = {
    '1_front': '1-front', '2_fronts': '2-front', '4_fronts': '4-front',
}


# ============================================================================
# Model Definitions
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
    return points[batch_indices, idx, :]


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


# D16a Model
class DiscreteDirectionHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, num_bins=16):
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


class D16aModel(nn.Module):
    def __init__(self, num_bins=16):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder()
        self.head = DiscreteDirectionHead(num_bins=num_bins)
        self.num_bins = num_bins

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.head(feat)
        return logits


# MF Model
class MixturePeakHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, num_peaks=4):
        super().__init__()
        self.num_peaks = num_peaks
        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x):
        h = self.shared(x)
        mu_vec = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu_vec = F.normalize(mu_vec, dim=-1)
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])
        mu = (mu + 2 * np.pi) % (2 * np.pi)
        kappa = F.softplus(self.kappa_head(h))
        return mu, kappa


class MFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder()
        self.head = MixturePeakHead()

    def forward(self, x):
        feat = self.encoder(x)
        mu, kappa = self.head(feat)
        return mu, kappa


# P2v2 Model
class P2v2SetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp_channels, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all
        last_ch = in_channel + 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in mlp_channels:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)
            new_points = grouped_xyz if points is None else torch.cat([grouped_xyz, points.unsqueeze(1)], -1)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(0.5, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            normed = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                new_points = torch.cat([normed, grouped_pts], -1)
            else:
                new_points = normed
        x = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, 3)[0]
        return new_xyz, x.permute(0, 2, 1)


class P2v2ExpertHead(nn.Module):
    def __init__(self, in_dim=1024, num_peaks=1, hidden_dim=256):
        super().__init__()
        self.num_peaks = num_peaks
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, num_peaks * 2)
        self.fc_kappa = nn.Linear(hidden_dim, num_peaks)

    def forward(self, features):
        B = features.size(0)
        hidden = self.hidden(features)
        mu_raw = self.fc_mu(hidden).view(B, self.num_peaks, 2)
        mu = torch.atan2(mu_raw[:, :, 1], mu_raw[:, :, 0])
        kappa = F.softplus(self.fc_kappa(hidden))
        return mu, kappa


class P2v2DirectionModel(nn.Module):
    def __init__(self, backbone_dim=1024):
        super().__init__()
        self.sa1 = P2v2SetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = P2v2SetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = P2v2SetAbstraction(None, None, 256, [256, 512, backbone_dim], group_all=True)
        self.head_1front = P2v2ExpertHead(backbone_dim, num_peaks=1)
        self.head_2front = P2v2ExpertHead(backbone_dim, num_peaks=2)
        self.head_4front = P2v2ExpertHead(backbone_dim, num_peaks=4)

    def forward(self, x, category=None):
        B = x.size(0)
        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.view(B, -1)

        if category == '1_front':
            mu, kappa = self.head_1front(global_feat)
        elif category == '2_fronts':
            mu, kappa = self.head_2front(global_feat)
        elif category == '4_fronts':
            mu, kappa = self.head_4front(global_feat)
        else:
            mu, kappa = self.head_1front(global_feat)
        return mu, kappa


# ============================================================================
# Utility Functions
# ============================================================================
def load_ply(filepath, num_points=2048):
    from plyfile import PlyData
    import hashlib
    ply = PlyData.read(filepath)
    vertex = ply['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)

    # Use deterministic sampling based on filepath (hashlib is deterministic)
    seed = int(hashlib.md5(str(filepath).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)

    if len(pts) > num_points:
        idx = rng.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    elif len(pts) < num_points:
        idx = rng.choice(len(pts), num_points, replace=True)
        pts = pts[idx]
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def rotate_y(points, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
    return points @ R.T


def circular_distance(angle1, angle2):
    diff = angle1 - angle2
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def hungarian_match_angles(pred_angles, gt_angles):
    n_pred = len(pred_angles)
    n_gt = len(gt_angles)
    if n_pred == 0 or n_gt == 0:
        return 0.0
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(gt_angles):
        for j, pred in enumerate(pred_angles):
            cost_matrix[i, j] = circular_distance(gt, pred)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_errors = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    return np.mean(matched_errors)


def mixture_pdf(theta, mus, kappas, weights):
    pdf = np.zeros_like(theta, dtype=float)
    for mu, kappa, w in zip(mus, kappas, weights):
        if kappa < 0.01:
            pdf += w / (2 * np.pi)
        else:
            pdf += w * vonmises.pdf(theta, kappa=kappa, loc=mu)
    return pdf


def render_pointcloud_topdown(points, ax, cmap='coolwarm', point_size=2.0):
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
# Model Loading
# ============================================================================
def load_d16a_model(checkpoint_path):
    model = D16aModel(num_bins=16)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = {}
    for k, v in checkpoint['encoder_state_dict'].items():
        state_dict[f'encoder.{k}'] = v
    for k, v in checkpoint['head_state_dict'].items():
        state_dict[f'head.{k}'] = v
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_mf_model(checkpoint_path):
    model = MFModel()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = {}
    for k, v in checkpoint['encoder_state_dict'].items():
        state_dict[f'encoder.{k}'] = v
    for k, v in checkpoint['head_state_dict'].items():
        state_dict[f'head.{k}'] = v
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_p2v2_model(checkpoint_path):
    model = P2v2DirectionModel()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    full_state = checkpoint['model_state_dict']
    new_state = {}
    for k, v in full_state.items():
        if k.startswith('sa1.') or k.startswith('sa2.') or k.startswith('sa3.'):
            new_state[k] = v
        elif k.startswith('head_1front.') or k.startswith('head_2front.') or k.startswith('head_4front.'):
            new_state[k] = v
    model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_d16a_polar(fig, ax_polar, logits, points, gt_angle, k, num_bins=16):
    """Plot D16a discrete bin prediction as bar chart"""
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    bin_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    bin_width = 2 * np.pi / num_bins

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background ring
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # Bar chart for probabilities
    prob_max = probs.max() if probs.max() > 0 else 1.0
    bar_heights = 0.35 + (probs / prob_max) * 0.55

    bars = ax_polar.bar(bin_angles, bar_heights - 0.35, width=bin_width * 0.8,
                        bottom=0.35, color='#3498db', alpha=0.7, zorder=3)

    # Highlight top-k bins
    top_k_idx = np.argsort(probs)[-k:]
    for idx in top_k_idx:
        bars[idx].set_color('#2980b9')
        bars[idx].set_alpha(1.0)

    # GT direction lines
    gt_angles = [(gt_angle + i * 2 * np.pi / k) % (2 * np.pi) for i in range(k)]
    for angle in gt_angles:
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#e74c3c', linewidth=2.5,
                      linestyle='--', solid_capstyle='round', zorder=10)

    # Predicted direction (top-k bin centers average via Hungarian)
    pred_angles = bin_angles[top_k_idx].tolist()
    for angle in pred_angles:
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#27ae60', linewidth=2.5,
                      solid_capstyle='round', zorder=8)

    # Style
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([])
    ax_polar.set_xticks([])
    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # Point cloud inset
    pos = ax_polar.get_position()
    pc_size = 0.22
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2
    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=1.5, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(points, ax_center, point_size=1.5)

    # Compute error
    error = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(error)


def plot_mf_polar(fig, ax_polar, mu, kappa, points, gt_angle, k):
    """Plot MF model as von Mises mixture"""
    mu_np = mu.cpu().numpy() if torch.is_tensor(mu) else mu
    kappa_np = kappa.cpu().numpy() if torch.is_tensor(kappa) else kappa

    # Sort by kappa
    sorted_idx = np.argsort(kappa_np)[::-1]
    mu_sorted = mu_np[sorted_idx]
    kappa_sorted = kappa_np[sorted_idx]

    # For visualization: use all 4 peaks
    theta = np.linspace(0, 2 * np.pi, 360)
    weights = np.ones(4) / 4
    pdf = mixture_pdf(theta, mu_sorted, kappa_sorted, weights)

    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # PDF curve
    ax_polar.fill(theta, pdf_display, color='#9b59b6', alpha=0.3, zorder=3)
    ax_polar.plot(theta, pdf_display, color='#9b59b6', linewidth=2.5, zorder=5)

    # GT direction lines
    if k == 1:
        gt_angles = [gt_angle] * 4  # For 1-front, we match 4 peaks to same GT
        pred_angles = mu_sorted.tolist()
    elif k == 2:
        gt_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        pred_angles = mu_sorted[:2].tolist()
    else:  # k == 4
        gt_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        pred_angles = mu_sorted.tolist()

    for angle in gt_angles[:k]:  # Only show k GT lines
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#e74c3c', linewidth=2.5,
                      linestyle='--', solid_capstyle='round', zorder=10)

    # Predicted peaks - always show all 4 mu predictions for MF model
    for angle in mu_sorted:
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#27ae60', linewidth=2.5,
                      solid_capstyle='round', zorder=8)

    # Style
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([])
    ax_polar.set_xticks([])
    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # Point cloud inset
    pos = ax_polar.get_position()
    pc_size = 0.22
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2
    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=1.5, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(points, ax_center, point_size=1.5)

    # Compute error (with Hungarian matching)
    error = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(error)


def plot_p2v2_polar(fig, ax_polar, mu, kappa, points, gt_angle, k):
    """Plot P2v2 expert head as von Mises mixture"""
    mu_np = mu.cpu().numpy() if torch.is_tensor(mu) else mu
    kappa_np = kappa.cpu().numpy() if torch.is_tensor(kappa) else kappa

    # Sort by kappa
    sorted_idx = np.argsort(kappa_np)[::-1]
    mu_sorted = mu_np[sorted_idx]
    kappa_sorted = kappa_np[sorted_idx]

    # PDF for visualization
    theta = np.linspace(0, 2 * np.pi, 360)
    weights = np.ones(k) / k
    pdf = mixture_pdf(theta, mu_sorted[:k], kappa_sorted[:k], weights)

    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # Background
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # PDF curve
    ax_polar.fill(theta, pdf_display, color='#e67e22', alpha=0.3, zorder=3)
    ax_polar.plot(theta, pdf_display, color='#e67e22', linewidth=2.5, zorder=5)

    # GT and predicted directions
    gt_angles = [(gt_angle + i * 2 * np.pi / k) % (2 * np.pi) for i in range(k)]
    pred_angles = [(mu_sorted[i] + 2 * np.pi) % (2 * np.pi) for i in range(k)]

    for angle in gt_angles:
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#e74c3c', linewidth=2.5,
                      linestyle='--', solid_capstyle='round', zorder=10)

    for angle in pred_angles:
        ax_polar.plot([angle, angle], [0.30, 1.05], color='#27ae60', linewidth=2.5,
                      solid_capstyle='round', zorder=8)

    # Style
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([])
    ax_polar.set_xticks([])
    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # Point cloud inset
    pos = ax_polar.get_position()
    pc_size = 0.22
    pc_x = pos.x0 + (pos.width - pc_size) / 2
    pc_y = pos.y0 + (pos.height - pc_size) / 2
    ax_center = fig.add_axes([pc_x, pc_y, pc_size, pc_size])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=1.5, zorder=0)
    ax_center.add_patch(circle)
    render_pointcloud_topdown(points, ax_center, point_size=1.5)

    # Compute error
    error = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(error)


def visualize_three_models(d16a_model, mf_model, p2v2_model, sample_info, points, save_path):
    """Create 3-column visualization for one sample"""
    file_path = sample_info['file']
    category = sample_info['category']
    gt_angle = sample_info['gt_angle']
    k = sample_info['k']
    rotation = sample_info.get('rotation', 0)

    # Apply rotation - point cloud and GT rotate together
    # rotation is already in radians from sample_info
    pts_rot = rotate_y(points, rotation)
    # rotate_y rotates points CLOCKWISE in XZ plane (viewed from +Y)
    # So GT direction also rotates clockwise: new_gt = gt - rotation
    rotated_gt = (gt_angle - rotation) % (2 * np.pi)

    pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).float().to(DEVICE)

    # Get predictions
    with torch.no_grad():
        d16a_logits = d16a_model(pts_tensor)[0].cpu().numpy()
        mf_mu, mf_kappa = mf_model(pts_tensor)
        mf_mu, mf_kappa = mf_mu[0], mf_kappa[0]

        category_map = {1: '1_front', 2: '2_fronts', 4: '4_fronts'}
        p2v2_mu, p2v2_kappa = p2v2_model(pts_tensor, category=category_map[k])
        p2v2_mu, p2v2_kappa = p2v2_mu[0], p2v2_kappa[0]

    # Create figure
    fig = plt.figure(figsize=(15, 6), facecolor='white')
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.80, bottom=0.12, wspace=0.08)

    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')

    # Plot each model
    d16a_error = plot_d16a_polar(fig, ax1, d16a_logits, pts_rot, rotated_gt, k)
    mf_error = plot_mf_polar(fig, ax2, mf_mu, mf_kappa, pts_rot, rotated_gt, k)
    p2v2_error = plot_p2v2_polar(fig, ax3, p2v2_mu, p2v2_kappa, pts_rot, rotated_gt, k)

    # Titles with error
    ax1.set_title(f'Discrete\nError: {d16a_error:.1f}°', fontsize=13, pad=15)
    ax2.set_title(f'MvM\nError: {mf_error:.1f}°', fontsize=13, pad=15)
    ax3.set_title(f'MoE\nError: {p2v2_error:.1f}°', fontsize=13, pad=15)

    # Main title - simple format: object_name category
    obj_name = file_path.split('/')[0]
    cat_display = CATEGORY_DISPLAY.get(category, category)
    fig.suptitle(f'{obj_name}  {cat_display}',
                 fontsize=16, fontweight='bold', y=0.95)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#27ae60', linewidth=3, label='Predicted'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, linestyle='--', label='GT'),
        Patch(facecolor='#3498db', alpha=0.7, label='Discrete $p(\\phi)$'),
        Patch(facecolor='#9b59b6', alpha=0.3, edgecolor='#9b59b6', linewidth=2, label='MvM $p(\\phi)$'),
        Patch(facecolor='#e67e22', alpha=0.3, edgecolor='#e67e22', linewidth=2, label='MoE $p(\\phi)$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.01), fontsize=10, framealpha=0.95)

    plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {save_path}")

    return d16a_error, mf_error, p2v2_error


def main():
    parser = argparse.ArgumentParser(description='Visualize three models comparison')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top samples to visualize')
    parser.add_argument('--samples', type=str, help='Comma-separated sample paths')
    parser.add_argument('--csv_path', type=str, default='paper_figures/three_models_comparison_20260108_221249.csv')
    parser.add_argument('--output_dir', type=str, default='paper_figures/three_models_vis')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    print("Loading models...")
    d16a_model = load_d16a_model('checkpoints/D_16a_20251219_053127/best_error.pth')
    mf_model = load_mf_model('checkpoints/MF_1c_mu_only_20251217_005015/best_error.pth')
    p2v2_model = load_p2v2_model('checkpoints/P2v2_Clean_20251230_165848/best.pth')
    print("  Models loaded.")

    # Load annotations for GT info
    with open('data_annotation/symmetry_annotations.json') as f:
        annotations = json.load(f)

    # Load CSV data for rotation info
    import csv
    csv_samples = {}
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_samples[row['file']] = {
                'file': row['file'],
                'category': row['category'],
                'k': int(row['k']),
                'rotation': float(row['rotation_deg']) * np.pi / 180,
                'mean_error': float(row['mean_error']),
            }

    # Get samples to visualize
    if args.samples:
        sample_files = args.samples.split(',')
    else:
        # Sort by mean error and take top-k
        samples = sorted(csv_samples.values(), key=lambda x: x['mean_error'])
        samples = samples[:args.top_k]
        sample_files = [s['file'] for s in samples]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path('data/full_mn40_normal_resampled_ply')

    # Visualize each sample
    for i, file_path in enumerate(sample_files):
        print(f"\n[{i+1}/{len(sample_files)}] {file_path}")

        # Get annotation info
        ann = annotations.get(file_path)
        if not ann:
            print(f"  Warning: No annotation for {file_path}")
            continue

        symmetry_name = ann.get('symmetry_name', '')
        category = SYMMETRY_TO_CAT.get(symmetry_name)
        if not category or category == 'no_front':
            continue

        direction = ann.get('front_direction')
        gt_angle = DIRECTION_TO_ANGLE.get(direction, 0.0)
        k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4}.get(category, 1)

        # Load point cloud
        ply_path = data_root / file_path
        if not ply_path.exists():
            print(f"  Warning: File not found {ply_path}")
            continue

        points = load_ply(str(ply_path), NUM_POINTS)

        # Get rotation from CSV
        rotation = 0.0
        if file_path in csv_samples:
            rotation = csv_samples[file_path]['rotation']

        sample_info = {
            'file': file_path,
            'category': category,
            'gt_angle': gt_angle,
            'k': k,
            'rotation': rotation,
        }

        # Check existing files to determine starting index
        existing_files = list(output_dir.glob('*.png'))
        start_idx = len(existing_files) if args.samples else 0
        save_name = f"{start_idx + i + 1:02d}_{file_path.replace('/', '_').replace('.ply', '')}.png"
        save_path = output_dir / save_name

        visualize_three_models(d16a_model, mf_model, p2v2_model, sample_info, points, save_path)

    print(f"\nVisualization saved to: {output_dir}")


if __name__ == '__main__':
    main()

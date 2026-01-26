#!/usr/bin/env python3
"""
Evaluate D16a, MF_mu_only, and P2v2_Clean on TRAINING SET samples with random rotation.

Output format: One row per point cloud with all three models' angular errors for comparison.

Angular error calculation (using GT's k-front protocol):
- D16a (discrete): Select top-k bins, compute average direction, Hungarian match with GT
- MF_mu_only: 1-front: 4 peaks vs 4 copies of GT; 2-front: top-2 peaks vs 2 GTs; 4-front: 4 peaks vs 4 GTs
- P2v2_Clean: Use corresponding expert head (1/2/4-front)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

# Direction to angle mapping
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


# ============================================================================
# PointNet++ Components
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


# ============================================================================
# D16a Discrete Head
# ============================================================================
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


# ============================================================================
# MF mu_only Head
# ============================================================================
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


# ============================================================================
# P2v2 Clean Model
# ============================================================================
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
        cos_val = mu_raw[:, :, 0]
        sin_val = mu_raw[:, :, 1]
        mu = torch.atan2(sin_val, cos_val)
        kappa = F.softplus(self.fc_kappa(hidden))
        return {'mu': mu, 'kappa': kappa}


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
            out = self.head_1front(global_feat)
        elif category == '2_fronts':
            out = self.head_2front(global_feat)
        elif category == '4_fronts':
            out = self.head_4front(global_feat)
        else:
            out = self.head_1front(global_feat)
        return out['mu'], out['kappa']


# ============================================================================
# Data loading utilities
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
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def rotate_y(points, angle):
    """Rotate points around Y axis"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
    return points @ R.T


def circular_distance(angle1, angle2):
    """Compute circular distance in radians"""
    diff = angle1 - angle2
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def hungarian_match_angles(pred_angles, gt_angles):
    """
    Hungarian matching between predicted and ground truth angles.
    Returns the minimum total angular error.
    """
    n_pred = len(pred_angles)
    n_gt = len(gt_angles)

    if n_pred == 0 or n_gt == 0:
        return 0.0, []

    # Build cost matrix
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(gt_angles):
        for j, pred in enumerate(pred_angles):
            cost_matrix[i, j] = circular_distance(gt, pred)

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_errors = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    mean_error = np.mean(matched_errors) if matched_errors else 0.0

    return mean_error, matched_errors


# ============================================================================
# Model loading
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
# Evaluation functions for each model
# ============================================================================
def get_d16a_topk_angles(logits, k, num_bins=16):
    """Get top-k bin center angles from D16a logits."""
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    bin_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    top_indices = np.argsort(probs)[-k:][::-1]
    top_angles = bin_angles[top_indices]
    return top_angles.tolist()


def get_mf_all_peaks(mu, kappa):
    """Get all 4 peak directions from MF model, sorted by kappa."""
    mu_np = mu.cpu().numpy()
    kappa_np = kappa.cpu().numpy()
    sorted_idx = np.argsort(kappa_np)[::-1]
    sorted_angles = mu_np[sorted_idx]
    return sorted_angles.tolist()


def evaluate_d16a(model, pts_tensor, gt_angle, k):
    """
    Evaluate D16a for k-front: select top-k bins, Hungarian match with k GTs.
    """
    with torch.no_grad():
        logits = model(pts_tensor)
    pred_angles = get_d16a_topk_angles(logits[0], k)
    gt_angles = [(gt_angle + i * 2 * np.pi / k) % (2 * np.pi) for i in range(k)]
    mean_error, _ = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(mean_error)


def evaluate_mf(model, pts_tensor, gt_angle, k):
    """
    Evaluate MF mu_only for k-front.
    - 1-front: all 4 peaks vs 4 copies of same GT
    - 2-front: top-2 peaks vs 2 GTs (180° apart)
    - 4-front: all 4 peaks vs 4 GTs (90° apart)
    """
    with torch.no_grad():
        mu, kappa = model(pts_tensor)
    all_pred_angles = get_mf_all_peaks(mu[0], kappa[0])

    if k == 1:
        gt_angles = [gt_angle] * 4
        pred_angles = all_pred_angles
    elif k == 2:
        gt_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        pred_angles = all_pred_angles[:2]
    elif k == 4:
        gt_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        pred_angles = all_pred_angles
    else:
        return 0.0

    mean_error, _ = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(mean_error)


def evaluate_p2v2(model, pts_tensor, gt_angle, k):
    """
    Evaluate P2v2 Clean using the corresponding expert head for k-front.
    """
    category_map = {1: '1_front', 2: '2_fronts', 4: '4_fronts'}
    category = category_map.get(k, '1_front')

    with torch.no_grad():
        mu, kappa = model(pts_tensor, category=category)

    mu_np = mu[0].cpu().numpy()
    kappa_np = kappa[0].cpu().numpy()
    sorted_idx = np.argsort(kappa_np)[::-1]
    pred_angles = [mu_np[sorted_idx[i]] for i in range(min(k, len(sorted_idx)))]
    pred_angles = [(a + 2 * np.pi) % (2 * np.pi) for a in pred_angles]
    gt_angles = [(gt_angle + i * 2 * np.pi / k) % (2 * np.pi) for i in range(k)]
    mean_error, _ = hungarian_match_angles(pred_angles, gt_angles)
    return np.degrees(mean_error)


# ============================================================================
# Main evaluation
# ============================================================================
def load_training_samples():
    """Load training samples (70% split, stratified by category)"""
    annotation_file = 'data_annotation/symmetry_annotations.json'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    samples = []
    excluded_directions = {'OBLIQUE', 'MULTI'}

    for file_path, ann in annotations.items():
        symmetry_name = ann.get('symmetry_name', '')
        if symmetry_name not in SYMMETRY_TO_CAT:
            continue
        category = SYMMETRY_TO_CAT[symmetry_name]

        if category in ['1_front', '2_fronts', '4_fronts']:
            direction = ann.get('front_direction')
            if not direction or direction in excluded_directions:
                continue
            if direction not in DIRECTION_TO_ANGLE:
                continue
            gt_angle = DIRECTION_TO_ANGLE[direction]
        else:
            continue

        k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4}.get(category, 0)
        if k == 0:
            continue

        samples.append({
            'file': file_path,
            'category': category,
            'gt_angle': gt_angle,
            'k': k,
        })

    # 70% train split (stratified)
    rng = np.random.RandomState(42)
    category_samples = defaultdict(list)
    for sample in samples:
        category_samples[sample['category']].append(sample)

    train_samples = []
    for cat, cat_samples in category_samples.items():
        indices = rng.permutation(len(cat_samples))
        n_train = int(len(cat_samples) * 0.7)
        train_idx = indices[:n_train]
        train_samples.extend([cat_samples[i] for i in train_idx])

    return train_samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Evaluating D16a, MF_mu_only, P2v2_Clean on Training Set")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    d16a_path = 'checkpoints/D_16a_20251219_053127/best_error.pth'
    mf_path = 'checkpoints/MF_1c_mu_only_20251217_005015/best_error.pth'
    p2v2_path = 'checkpoints/P2v2_Clean_20251230_165848/best.pth'

    d16a_model = load_d16a_model(d16a_path)
    mf_model = load_mf_model(mf_path)
    p2v2_model = load_p2v2_model(p2v2_path)
    print("  Models loaded.")

    # Load training samples
    print("\nLoading training samples...")
    train_samples = load_training_samples()
    print(f"  Found {len(train_samples)} training samples")

    cat_counts = defaultdict(int)
    for s in train_samples:
        cat_counts[s['category']] += 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    # Sample subset
    random.seed(args.seed)
    num_samples = min(args.num_samples, len(train_samples))
    eval_samples = random.sample(train_samples, num_samples)
    print(f"\n  Evaluating {num_samples} samples...")

    data_root = Path('data/full_mn40_normal_resampled_ply')

    # Results: one row per point cloud
    results = []

    for i, sample in enumerate(eval_samples):
        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        # Random rotation
        random_angle = random.uniform(0, 2 * np.pi)
        pts_rot = rotate_y(pts, random_angle)
        # rotate_y rotates points CLOCKWISE in XZ plane (viewed from +Y)
        # So GT direction also rotates clockwise: new_gt = gt - rotation
        rotated_gt = (sample['gt_angle'] - random_angle) % (2 * np.pi)

        pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

        k = sample['k']

        # Evaluate all three models using the GT's k-front protocol
        d16a_error = evaluate_d16a(d16a_model, pts_tensor, rotated_gt, k)
        mf_error = evaluate_mf(mf_model, pts_tensor, rotated_gt, k)
        p2v2_error = evaluate_p2v2(p2v2_model, pts_tensor, rotated_gt, k)

        # Compute mean and std across three models
        errors = [d16a_error, mf_error, p2v2_error]
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        results.append({
            'file': sample['file'],
            'category': sample['category'],
            'k': k,
            'rotation_deg': np.degrees(random_angle),
            'd16a_error': d16a_error,
            'mf_error': mf_error,
            'p2v2_error': p2v2_error,
            'mean_error': mean_error,
            'std_error': std_error,
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{num_samples}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    all_d16a = [r['d16a_error'] for r in results]
    all_mf = [r['mf_error'] for r in results]
    all_p2v2 = [r['p2v2_error'] for r in results]

    print(f"\nOverall (n={len(results)}):")
    print(f"  D16a:  mean={np.mean(all_d16a):.2f}°, median={np.median(all_d16a):.2f}°, std={np.std(all_d16a):.2f}°")
    print(f"  MF:    mean={np.mean(all_mf):.2f}°, median={np.median(all_mf):.2f}°, std={np.std(all_mf):.2f}°")
    print(f"  P2v2:  mean={np.mean(all_p2v2):.2f}°, median={np.median(all_p2v2):.2f}°, std={np.std(all_p2v2):.2f}°")

    # By category
    for cat in ['1_front', '2_fronts', '4_fronts']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            d16a_errs = [r['d16a_error'] for r in cat_results]
            mf_errs = [r['mf_error'] for r in cat_results]
            p2v2_errs = [r['p2v2_error'] for r in cat_results]
            print(f"\n{cat} (n={len(cat_results)}):")
            print(f"  D16a:  mean={np.mean(d16a_errs):.2f}°, median={np.median(d16a_errs):.2f}°")
            print(f"  MF:    mean={np.mean(mf_errs):.2f}°, median={np.median(mf_errs):.2f}°")
            print(f"  P2v2:  mean={np.mean(p2v2_errs):.2f}°, median={np.median(p2v2_errs):.2f}°")

    # Save results
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV: one row per point cloud
    csv_path = output_dir / f'three_models_comparison_{timestamp}.csv'
    with open(csv_path, 'w') as f:
        f.write("file,category,k,rotation_deg,d16a_error,mf_error,p2v2_error,mean_error,std_error\n")
        for r in results:
            f.write(f"{r['file']},{r['category']},{r['k']},{r['rotation_deg']:.2f},"
                    f"{r['d16a_error']:.2f},{r['mf_error']:.2f},{r['p2v2_error']:.2f},"
                    f"{r['mean_error']:.2f},{r['std_error']:.2f}\n")
    print(f"\nCSV saved to: {csv_path}")

    # JSON with full details
    json_path = output_dir / f'three_models_comparison_{timestamp}.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(results),
        'overall': {
            'd16a': {'mean': np.mean(all_d16a), 'median': np.median(all_d16a), 'std': np.std(all_d16a)},
            'mf': {'mean': np.mean(all_mf), 'median': np.median(all_mf), 'std': np.std(all_mf)},
            'p2v2': {'mean': np.mean(all_p2v2), 'median': np.median(all_p2v2), 'std': np.std(all_p2v2)},
        },
        'by_category': {},
    }
    for cat in ['1_front', '2_fronts', '4_fronts']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            summary['by_category'][cat] = {
                'count': len(cat_results),
                'd16a': {'mean': np.mean([r['d16a_error'] for r in cat_results])},
                'mf': {'mean': np.mean([r['mf_error'] for r in cat_results])},
                'p2v2': {'mean': np.mean([r['p2v2_error'] for r in cat_results])},
            }

    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == '__main__':
    main()

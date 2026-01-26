#!/usr/bin/env python3
"""
Test P2v2 Clean model only on 350 test samples
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

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

# 方向到角度映射
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


class P2v2SetAbstraction(nn.Module):
    """P2v2使用的SetAbstraction层"""
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
    """P2v2的ExpertHead - 输出von Mises分布参数"""
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
    """P2v2 Direction Model"""
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
# 数据加载和测试
# ============================================================================
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
    print(f"Loaded P2v2 model with {len(new_state)} parameters")
    return model


def main():
    print("=" * 60)
    print("Testing P2v2 Clean on 350 samples with 8x TTA")
    print("=" * 60)

    # 加载测试样本
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

    # 统计样本分布
    cat_counts = defaultdict(int)
    for s in valid_samples:
        cat_counts[s['category']] += 1
    print(f"Sample distribution: 1_front={cat_counts['1_front']}, 2_fronts={cat_counts['2_fronts']}, 4_fronts={cat_counts['4_fronts']}, no_front={cat_counts['no_front']}")

    # 加载模型
    p2v2_path = 'checkpoints/P2v2_Clean_20251230_165848/best.pth'
    model = load_p2v2_model(p2v2_path)

    data_root = Path('data/full_mn40_normal_resampled_ply')
    errors_by_cat = defaultdict(list)
    all_errors = []

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
                mu, kappa = model(pts_tensor, category=sample['category'])

            max_kappa_idx = kappa[0].argmax()
            pred_angle = mu[0, max_kappa_idx].item()
            pred_angle = (pred_angle - rot_angle) % (2 * np.pi)
            pred_angles.append(pred_angle)

        # 圆周平均
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

    # 统计结果
    print("\n" + "=" * 60)
    print("P2v2 Clean Results")
    print("=" * 60)
    print(f"Overall: {np.mean(all_errors):.2f}° (median: {np.median(all_errors):.2f}°, n={len(all_errors)})")
    for cat in ['1_front', '2_fronts', '4_fronts']:
        if errors_by_cat[cat]:
            print(f"  {cat}: {np.mean(errors_by_cat[cat]):.2f}° (n={len(errors_by_cat[cat])})")

    # 保存结果
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

    output_path = Path(f'results/p2v2_clean_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

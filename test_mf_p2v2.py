#!/usr/bin/env python3
"""
Test MF series and P2v2 Clean models on 350 test samples
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
# PointNet++ Encoder (same as train_direction.py)
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


class MixturePeakHead(nn.Module):
    """MF系列输出头"""
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
    """MF系列模型"""
    def __init__(self):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder()
        self.head = MixturePeakHead()

    def forward(self, x):
        feat = self.encoder(x)
        mu, kappa = self.head(feat)
        return mu, kappa


# ============================================================================
# 数据加载
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


# ============================================================================
# 主测试函数
# ============================================================================
def load_mf_model(checkpoint_path):
    """加载MF系列模型"""
    model = MFModel()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    encoder_dict = checkpoint['encoder_state_dict']
    head_dict = checkpoint['head_state_dict']

    state_dict = {}
    for k, v in encoder_dict.items():
        state_dict[f'encoder.{k}'] = v
    for k, v in head_dict.items():
        state_dict[f'head.{k}'] = v

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def test_model(model, samples, data_root, num_rotations=8, model_name="Model"):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    errors_by_cat = defaultdict(list)
    all_errors = []
    predictions = []  # 记录详细预测

    for i, sample in enumerate(samples):
        if sample['k'] == 0:  # Skip no_front
            continue

        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        # TTA
        pred_angles = []
        for rot_idx in range(num_rotations):
            if num_rotations > 1:
                rot_angle = rot_idx * 2 * np.pi / num_rotations
                pts_rot = rotate_y(pts, rot_angle)
            else:
                pts_rot = pts
                rot_angle = 0.0

            pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mu, kappa = model(pts_tensor)

            # 选择kappa最大的峰
            max_kappa_idx = kappa[0].argmax()
            pred_angle = mu[0, max_kappa_idx].item()

            # 还原旋转
            pred_angle = (pred_angle - rot_angle) % (2 * np.pi)
            pred_angles.append(pred_angle)

        # 圆周平均
        sin_sum = sum(np.sin(a) for a in pred_angles)
        cos_sum = sum(np.cos(a) for a in pred_angles)
        avg_pred_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

        # 计算误差
        error = angular_distance(avg_pred_angle, sample['gt_angle'], sample['k'])
        error_deg = np.degrees(error)

        errors_by_cat[sample['category']].append(error_deg)
        all_errors.append(error_deg)

        # 记录预测详情
        predictions.append({
            'file': sample['file'],
            'category': sample['category'],
            'gt_angle_deg': np.degrees(sample['gt_angle']),
            'pred_angle_deg': np.degrees(avg_pred_angle),
            'error_deg': error_deg,
        })

        if (i + 1) % 50 == 0:
            current_mean = np.mean(all_errors) if all_errors else 0
            print(f"  Progress: {i+1}/{len(samples)}, Current mean error: {current_mean:.2f}°")

    # 统计结果
    results = {
        'model': model_name,
        'overall_mean': np.mean(all_errors) if all_errors else 0,
        'overall_median': np.median(all_errors) if all_errors else 0,
        'num_samples': len(all_errors),
    }

    for cat, errs in errors_by_cat.items():
        results[f'{cat}_mean'] = np.mean(errs) if errs else 0
        results[f'{cat}_count'] = len(errs)

    print(f"\n  Results for {model_name}:")
    print(f"    Overall: {results['overall_mean']:.2f}° (median: {results['overall_median']:.2f}°, n={results['num_samples']})")
    for cat in ['1_front', '2_fronts', '4_fronts']:
        if f'{cat}_mean' in results:
            print(f"    {cat}: {results[f'{cat}_mean']:.2f}° (n={results[f'{cat}_count']})")

    return results, predictions


# ============================================================================
# P2v2 Model Components (Mixture of Experts)
# ============================================================================

class P2v2SetAbstraction(nn.Module):
    """P2v2使用的SetAbstraction层 - 与probabilistic_orientation_net.py相同"""
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
            idx = query_ball_point(0.5, self.nsample, xyz, new_xyz)  # radius=0.5 default
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
        x = torch.max(x, 3)[0]  # (B, mlp[-1], npoint)
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
        self.fc_mu = nn.Linear(hidden_dim, num_peaks * 2)  # cos, sin
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
    """P2v2 Direction Model (without classifier, for testing only)"""
    def __init__(self, backbone_dim=1024):
        super().__init__()
        # Shared PointNet++ Backbone
        self.sa1 = P2v2SetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = P2v2SetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = P2v2SetAbstraction(None, None, 256, [256, 512, backbone_dim], group_all=True)

        # Expert Heads
        self.head_1front = P2v2ExpertHead(backbone_dim, num_peaks=1)
        self.head_2front = P2v2ExpertHead(backbone_dim, num_peaks=2)
        self.head_4front = P2v2ExpertHead(backbone_dim, num_peaks=4)

    def forward(self, x, category=None):
        """
        Args:
            x: [B, N, 3] point cloud
            category: category name to select head ('1_front', '2_fronts', '4_fronts')
        Returns:
            mu: [B, num_peaks] angles
            kappa: [B, num_peaks] concentration
        """
        B = x.size(0)

        # Backbone
        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.view(B, -1)

        # Select head based on category
        if category == '1_front':
            out = self.head_1front(global_feat)
        elif category == '2_fronts':
            out = self.head_2front(global_feat)
        elif category == '4_fronts':
            out = self.head_4front(global_feat)
        else:
            out = self.head_1front(global_feat)  # default

        return out['mu'], out['kappa']


def load_p2v2_model(checkpoint_path):
    """加载P2v2模型的direction部分"""
    model = P2v2DirectionModel()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    full_state = checkpoint['model_state_dict']

    # 提取direction backbone和heads
    new_state = {}

    # sa1, sa2, sa3 直接匹配
    for k, v in full_state.items():
        # Direction backbone: sa1, sa2, sa3 (not classifier.classifier.encoder...)
        if k.startswith('sa1.') or k.startswith('sa2.') or k.startswith('sa3.'):
            new_state[k] = v
        # Direction heads
        elif k.startswith('head_1front.') or k.startswith('head_2front.') or k.startswith('head_4front.'):
            new_state[k] = v

    # 加载状态
    model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()

    print(f"  Loaded P2v2 model with {len(new_state)} parameters")
    return model


def test_p2v2_model(checkpoint_path, samples, data_root, num_rotations=8):
    """测试P2v2模型"""
    print(f"\n{'='*60}")
    print(f"Testing: P2v2_Clean")
    print(f"{'='*60}")

    model = load_p2v2_model(checkpoint_path)

    errors_by_cat = defaultdict(list)
    all_errors = []
    predictions = []

    for i, sample in enumerate(samples):
        if sample['k'] == 0:  # Skip no_front/symmetric
            continue

        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        # TTA
        pred_angles = []
        for rot_idx in range(num_rotations):
            if num_rotations > 1:
                rot_angle = rot_idx * 2 * np.pi / num_rotations
                pts_rot = rotate_y(pts, rot_angle)
            else:
                pts_rot = pts
                rot_angle = 0.0

            pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mu, kappa = model(pts_tensor, category=sample['category'])

            # 选择kappa最大的峰
            max_kappa_idx = kappa[0].argmax()
            pred_angle = mu[0, max_kappa_idx].item()

            # 还原旋转
            pred_angle = (pred_angle - rot_angle) % (2 * np.pi)
            pred_angles.append(pred_angle)

        # 圆周平均
        sin_sum = sum(np.sin(a) for a in pred_angles)
        cos_sum = sum(np.cos(a) for a in pred_angles)
        avg_pred_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

        # 计算误差
        error = angular_distance(avg_pred_angle, sample['gt_angle'], sample['k'])
        error_deg = np.degrees(error)

        errors_by_cat[sample['category']].append(error_deg)
        all_errors.append(error_deg)

        predictions.append({
            'file': sample['file'],
            'category': sample['category'],
            'gt_angle_deg': np.degrees(sample['gt_angle']),
            'pred_angle_deg': np.degrees(avg_pred_angle),
            'error_deg': error_deg,
        })

        if (i + 1) % 50 == 0:
            current_mean = np.mean(all_errors) if all_errors else 0
            print(f"  Progress: {i+1}/{len(samples)}, Current mean error: {current_mean:.2f}°")

    # 统计结果
    results = {
        'model': 'P2v2_Clean',
        'overall_mean': np.mean(all_errors) if all_errors else 0,
        'overall_median': np.median(all_errors) if all_errors else 0,
        'num_samples': len(all_errors),
    }

    for cat, errs in errors_by_cat.items():
        results[f'{cat}_mean'] = np.mean(errs) if errs else 0
        results[f'{cat}_count'] = len(errs)

    print(f"\n  Results for P2v2_Clean:")
    print(f"    Overall: {results['overall_mean']:.2f}° (median: {results['overall_median']:.2f}°, n={results['num_samples']})")
    for cat in ['1_front', '2_fronts', '4_fronts']:
        if f'{cat}_mean' in results:
            print(f"    {cat}: {results[f'{cat}_mean']:.2f}° (n={results[f'{cat}_count']})")

    return results, predictions


def main():
    # 加载测试样本
    with open('data/test_set/test_set.json') as f:
        all_samples = json.load(f)

    # 筛选并选择350个
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

    data_root = Path('data/full_mn40_normal_resampled_ply')
    all_results = []

    # MF系列
    mf_checkpoints = [
        ('MF_1a', 'checkpoints/MF_1a_combined_20251216_190021/best_error.pth'),
        ('MF_1b', 'checkpoints/MF_1b_reverse_kl_20251216_220645/best_error.pth'),
        ('MF_1c', 'checkpoints/MF_1c_mu_only_20251217_005015/best_error.pth'),
        ('MF_1d', 'checkpoints/MF_1d_heavy_kappa_20251217_030241/best_error.pth'),
        ('MF_1e', 'checkpoints/MF_1e_kappa_mu_only_20251217_053138/best_error.pth'),
    ]

    print("\n" + "#"*70)
    print("# Testing MF Series")
    print("#"*70)

    for name, path in mf_checkpoints:
        if Path(path).exists():
            try:
                model = load_mf_model(path)
                results, preds = test_model(model, valid_samples, data_root,
                                           num_rotations=8, model_name=name)
                results['series'] = 'MF'
                all_results.append(results)
            except Exception as e:
                print(f"  [Error] {name}: {e}")

    # P2v2 Clean - 使用MoE架构，每个类别有独立的方向预测头
    print("\n" + "#"*70)
    print("# Testing P2v2 Clean")
    print("#"*70)

    p2v2_path = 'checkpoints/P2v2_Clean_20251230_165848/best.pth'
    if Path(p2v2_path).exists():
        try:
            results, preds = test_p2v2_model(p2v2_path, valid_samples, data_root)
            results['series'] = 'P2v2'
            all_results.append(results)
        except Exception as e:
            import traceback
            print(f"  [Error] P2v2 Clean: {e}")
            traceback.print_exc()

    # 汇总
    print("\n" + "="*70)
    print("SUMMARY - MF Series + P2v2 (350 samples, 8x TTA)")
    print("="*70)

    for r in sorted(all_results, key=lambda x: x['overall_mean']):
        print(f"  {r['model']}: {r['overall_mean']:.2f}° (median: {r['overall_median']:.2f}°)")

    # 保存结果
    output_path = Path(f'results/mf_p2v2_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

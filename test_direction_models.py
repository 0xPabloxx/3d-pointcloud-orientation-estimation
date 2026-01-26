#!/usr/bin/env python3
"""
Direction Model Test Script - 方向预测模型测试脚本

在350个测试样本上测试D系列、MF系列、P2v2系列模型
支持旋转增强 (Test Time Augmentation)
配置wandb实时日志

用法:
    python test_direction_models.py --num_samples 350 --num_rotations 8
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not available")

# ============================================================================
# 配置
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

# 对称名映射到方向预测类别
SYMMETRY_TO_DIRECTION_CAT = {
    '1个正面': '1_front', '1_front': '1_front',
    '2个正面': '2_fronts', '2_fronts': '2_fronts',
    '4个正面': '4_fronts', '4_fronts': '4_fronts',
    '旋转对称': 'no_front', 'symmetric': 'no_front', '完全对称': 'no_front',
    '无正面': 'no_front', 'no_front': 'no_front', '没有正面': 'no_front',
}

# 方向到角度映射 (XZ平面)
DIRECTION_TO_ANGLE = {
    '+X': 0.0,
    '+Z': np.pi / 2,
    '-X': np.pi,
    '-Z': 3 * np.pi / 2,
    '+X+Z': np.pi / 4,
    '-X+Z': 3 * np.pi / 4,
    '-X-Z': 5 * np.pi / 4,
    '+X-Z': 7 * np.pi / 4,
}


# ============================================================================
# 模型定义 (与train_direction.py一致)
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


class MixtureVonMisesHead(nn.Module):
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
        mu = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu = F.normalize(mu, dim=-1)
        kappa = F.softplus(self.kappa_head(h))
        return mu, kappa


class DiscreteResidualHead(nn.Module):
    """DR系列: 离散分类 + 残差回归"""
    def __init__(self, in_channels=1024, hidden_channels=512, num_bins=8):
        super().__init__()
        self.num_bins = num_bins
        self.bin_width = 2 * np.pi / num_bins

        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_bins),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_bins),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.cls_head(h)
        residuals = self.reg_head(h) * (self.bin_width / 2)
        return logits, residuals


class DirectionModel(nn.Module):
    """统一的方向预测模型"""
    def __init__(self, mode='discrete', num_bins=8, num_peaks=4):
        super().__init__()
        self.mode = mode
        self.num_bins = num_bins
        self.encoder = PointNetPlusPlusEncoder()

        if mode == 'discrete':
            self.head = DiscreteDirectionHead(num_bins=num_bins)
        elif mode == 'mf':
            self.head = MixtureVonMisesHead(num_peaks=num_peaks)
        elif mode == 'dr':
            self.head = DiscreteResidualHead(num_bins=num_bins)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# ============================================================================
# 数据加载工具
# ============================================================================
def load_ply(filepath, num_points=2048):
    """加载PLY点云"""
    from plyfile import PlyData
    ply = PlyData.read(filepath)
    vertex = ply['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)

    # 随机采样
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    elif len(pts) < num_points:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]

    # 归一化到单位球
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    return pts


def rotate_y(points, angle):
    """绕Y轴旋转点云"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return points @ R.T


# ============================================================================
# 误差计算
# ============================================================================
def angular_distance(pred_angle, gt_angle, symmetry_k=1):
    """
    计算角度误差，考虑对称性

    Args:
        pred_angle: 预测角度 (弧度)
        gt_angle: GT角度 (弧度)
        symmetry_k: 对称折数 (1, 2, 4)
    """
    if symmetry_k <= 0:  # no_front
        return 0.0

    min_error = float('inf')
    for i in range(symmetry_k):
        candidate = (gt_angle + i * 2 * np.pi / symmetry_k) % (2 * np.pi)
        diff = pred_angle - candidate
        # 圆周距离
        error = abs(np.arctan2(np.sin(diff), np.cos(diff)))
        min_error = min(min_error, error)

    return min_error


def get_pred_angle_discrete(logits, num_bins):
    """从discrete模型输出获取预测角度"""
    pred_bin = logits.argmax(dim=-1)
    bin_width = 2 * np.pi / num_bins
    return pred_bin.float() * bin_width


def get_pred_angle_dr(logits, residuals, num_bins):
    """从DR模型输出获取预测角度"""
    pred_bin = logits.argmax(dim=-1)
    bin_width = 2 * np.pi / num_bins
    base_angle = pred_bin.float() * bin_width
    residual = residuals[torch.arange(len(pred_bin)), pred_bin]
    return base_angle + residual


def get_pred_angle_mf(mu, kappa):
    """从MF模型输出获取预测角度 (选择kappa最大的peak)"""
    # mu: (B, num_peaks, 2), kappa: (B, num_peaks)
    max_kappa_idx = kappa.argmax(dim=-1)  # (B,)
    batch_idx = torch.arange(len(max_kappa_idx))
    best_mu = mu[batch_idx, max_kappa_idx]  # (B, 2)
    pred_angle = torch.atan2(best_mu[:, 1], best_mu[:, 0])  # XZ平面
    pred_angle = (pred_angle + 2 * np.pi) % (2 * np.pi)
    return pred_angle


# ============================================================================
# 主测试类
# ============================================================================
class DirectionModelTester:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE

        # 加载测试样本
        self.samples = self._load_test_samples()
        print(f"Loaded {len(self.samples)} test samples")

        # 初始化wandb
        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.init(
                project="ForwardNet-DirectionTest",
                name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

        # 模型列表
        self.models_to_test = self._find_models()

    def _load_test_samples(self):
        """加载350个测试样本"""
        test_set_path = Path(self.args.test_set)
        with open(test_set_path) as f:
            all_samples = json.load(f)

        # 过滤有效样本并选择350个
        valid_samples = []
        for s in all_samples:
            sym_name = s.get('symmetry_name', '')
            if sym_name not in SYMMETRY_TO_DIRECTION_CAT:
                continue

            cat = SYMMETRY_TO_DIRECTION_CAT[sym_name]
            direction = s.get('front_direction')

            # 对于有方向的类别，需要有效的方向
            if cat in ['1_front', '2_fronts', '4_fronts']:
                if not direction or direction not in DIRECTION_TO_ANGLE:
                    continue
                gt_angle = DIRECTION_TO_ANGLE[direction]
            else:
                gt_angle = 0.0  # no_front

            # 确定K值
            if cat == '1_front':
                k = 1
            elif cat == '2_fronts':
                k = 2
            elif cat == '4_fronts':
                k = 4
            else:
                k = 0  # no_front

            valid_samples.append({
                'file': s['file'],
                'category': cat,
                'gt_angle': gt_angle,
                'k': k,
                'symmetry_name': sym_name,
            })

        # 随机选择指定数量
        random.seed(42)
        if len(valid_samples) > self.args.num_samples:
            valid_samples = random.sample(valid_samples, self.args.num_samples)

        return valid_samples

    def _find_models(self):
        """查找所有要测试的模型"""
        checkpoints_dir = Path(self.args.checkpoints_dir)
        models = []

        # D系列
        for d in sorted(checkpoints_dir.glob("D_*")):
            if (d / "best_error.pth").exists():
                # 从目录名解析num_bins
                name = d.name
                if '16' in name:
                    num_bins = 16
                else:
                    num_bins = 8
                models.append({
                    'name': name,
                    'path': d / "best_error.pth",
                    'mode': 'discrete',
                    'num_bins': num_bins,
                    'series': 'D'
                })

        # DR系列
        for d in sorted(checkpoints_dir.glob("DR_*")):
            if (d / "best_error.pth").exists():
                name = d.name
                if '16' in name:
                    num_bins = 16
                else:
                    num_bins = 8
                models.append({
                    'name': name,
                    'path': d / "best_error.pth",
                    'mode': 'dr',
                    'num_bins': num_bins,
                    'series': 'DR'
                })

        # MF系列
        for d in sorted(checkpoints_dir.glob("MF_*")):
            if (d / "best_error.pth").exists():
                models.append({
                    'name': d.name,
                    'path': d / "best_error.pth",
                    'mode': 'mf',
                    'num_bins': 8,  # MF不用bins
                    'series': 'MF'
                })

        # P2v2系列
        for d in sorted(checkpoints_dir.glob("P2v2_*")):
            if (d / "best.pth").exists():
                models.append({
                    'name': d.name,
                    'path': d / "best.pth",
                    'mode': 'mf',  # P2v2也使用MF输出格式
                    'num_bins': 8,
                    'series': 'P2v2'
                })

        return models

    def _load_model(self, model_info):
        """加载模型"""
        mode = model_info['mode']
        num_bins = model_info['num_bins']

        model = DirectionModel(mode=mode, num_bins=num_bins)

        checkpoint = torch.load(model_info['path'], map_location=self.device, weights_only=False)

        # 处理不同的checkpoint格式
        if 'encoder_state_dict' in checkpoint and 'head_state_dict' in checkpoint:
            # train_direction.py格式: 分开保存encoder和head
            encoder_dict = checkpoint['encoder_state_dict']
            head_dict = checkpoint['head_state_dict']
            # 添加前缀 (encoder keys: sa1.xxx -> encoder.sa1.xxx, head keys: head.0.xxx -> head.head.0.xxx)
            state_dict = {}
            for k, v in encoder_dict.items():
                state_dict[f'encoder.{k}'] = v
            for k, v in head_dict.items():
                # head的state dict已经有head.前缀，需要加head.
                state_dict[f'head.{k}'] = v
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 尝试加载
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  [Warning] State dict mismatch: {e}")
            # 尝试部分加载
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"  [Info] Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def test_model(self, model_info):
        """测试单个模型"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_info['name']}")
        print(f"  Mode: {model_info['mode']}, Series: {model_info['series']}")
        print(f"{'='*60}")

        try:
            model = self._load_model(model_info)
        except Exception as e:
            print(f"  [Error] Failed to load model: {e}")
            return None

        errors_by_cat = defaultdict(list)
        all_errors = []

        data_root = Path(self.args.data_root)
        num_rotations = self.args.num_rotations

        for i, sample in enumerate(self.samples):
            if sample['k'] == 0:  # Skip no_front
                continue

            # 加载点云
            ply_path = data_root / sample['file']
            if not ply_path.exists():
                continue

            try:
                pts = load_ply(str(ply_path), NUM_POINTS)
            except Exception as e:
                continue

            # TTA: 多次旋转取平均
            pred_angles = []
            for rot_idx in range(num_rotations):
                if num_rotations > 1:
                    rot_angle = rot_idx * 2 * np.pi / num_rotations
                    pts_rot = rotate_y(pts, rot_angle)
                else:
                    pts_rot = pts
                    rot_angle = 0.0

                pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(self.device)

                # 推理
                output = model(pts_tensor)

                # 获取预测角度
                if model_info['mode'] == 'discrete':
                    pred_angle = get_pred_angle_discrete(output, model_info['num_bins'])
                elif model_info['mode'] == 'dr':
                    logits, residuals = output
                    pred_angle = get_pred_angle_dr(logits, residuals, model_info['num_bins'])
                elif model_info['mode'] == 'mf':
                    mu, kappa = output
                    pred_angle = get_pred_angle_mf(mu, kappa)

                # 还原旋转
                pred_angle = (pred_angle.item() - rot_angle) % (2 * np.pi)
                pred_angles.append(pred_angle)

            # 使用圆周平均
            sin_sum = sum(np.sin(a) for a in pred_angles)
            cos_sum = sum(np.cos(a) for a in pred_angles)
            avg_pred_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

            # 计算误差
            error = angular_distance(avg_pred_angle, sample['gt_angle'], sample['k'])
            error_deg = np.degrees(error)

            errors_by_cat[sample['category']].append(error_deg)
            all_errors.append(error_deg)

            # 实时进度
            if (i + 1) % 50 == 0:
                current_mean = np.mean(all_errors) if all_errors else 0
                print(f"  Progress: {i+1}/{len(self.samples)}, Current mean error: {current_mean:.2f}°")

        # 统计结果
        results = {
            'model': model_info['name'],
            'series': model_info['series'],
            'mode': model_info['mode'],
            'overall_mean_error': np.mean(all_errors) if all_errors else 0,
            'overall_median_error': np.median(all_errors) if all_errors else 0,
            'num_samples': len(all_errors),
        }

        for cat, errs in errors_by_cat.items():
            results[f'{cat}_mean'] = np.mean(errs) if errs else 0
            results[f'{cat}_count'] = len(errs)

        # 打印结果
        print(f"\n  Results for {model_info['name']}:")
        print(f"    Overall: {results['overall_mean_error']:.2f}° (n={results['num_samples']})")
        for cat in ['1_front', '2_fronts', '4_fronts']:
            if f'{cat}_mean' in results:
                print(f"    {cat}: {results[f'{cat}_mean']:.2f}° (n={results[f'{cat}_count']})")

        # 记录到wandb
        if self.use_wandb:
            wandb.log({
                f"{model_info['series']}/{model_info['name']}/mean_error": results['overall_mean_error'],
                f"{model_info['series']}/{model_info['name']}/median_error": results['overall_median_error'],
            })

        return results

    def run(self):
        """运行所有测试"""
        print(f"\n{'#'*70}")
        print(f"# Direction Model Testing")
        print(f"# Test samples: {len(self.samples)}")
        print(f"# Models to test: {len(self.models_to_test)}")
        print(f"# TTA rotations: {self.args.num_rotations}")
        print(f"{'#'*70}")

        all_results = []

        for model_info in self.models_to_test:
            result = self.test_model(model_info)
            if result:
                all_results.append(result)

        # 汇总报告
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        # 按系列分组
        for series in ['D', 'DR', 'MF', 'P2v2']:
            series_results = [r for r in all_results if r['series'] == series]
            if series_results:
                print(f"\n{series} Series:")
                for r in sorted(series_results, key=lambda x: x['overall_mean_error']):
                    print(f"  {r['model']}: {r['overall_mean_error']:.2f}°")

        # 保存结果
        output_path = Path(self.args.output_dir) / f"direction_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        if self.use_wandb:
            wandb.finish()

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Test direction prediction models')
    parser.add_argument('--test_set', type=str,
                        default='data/test_set/test_set.json',
                        help='Path to test set JSON')
    parser.add_argument('--data_root', type=str,
                        default='data/full_mn40_normal_resampled_ply',
                        help='Root directory for PLY files')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str,
                        default='results/direction_test',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=350,
                        help='Number of test samples')
    parser.add_argument('--num_rotations', type=int, default=8,
                        help='Number of TTA rotations')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    tester = DirectionModelTester(args)
    tester.run()


if __name__ == '__main__':
    main()

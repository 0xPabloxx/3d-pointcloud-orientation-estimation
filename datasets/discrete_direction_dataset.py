"""
Discrete Direction Dataset - 离散方向预测数据集

将方向预测问题转化为分类问题:
- 8 bins: 每45°一个bin [+X, +X+Z, +Z, -X+Z, -X, -X-Z, -Z, +X-Z]
- 16 bins: 每22.5°一个bin

用法:
    dataset = DiscreteDirectionDataset(split='train', num_bins=8)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class DiscreteDirectionDataset(Dataset):
    """
    离散方向预测数据集

    将连续方向转化为离散bin的概率分布

    Args:
        annotation_file: 标注JSON文件路径
        data_root: 点云数据根目录
        split: 数据划分 ('train', 'val', 'test', 'all')
        categories: 要包含的类别列表 ['1_front', '4_fronts', 'no_front']
        num_bins: 离散bin数量 (8 or 16)
        split_ratio: 划分比例 (train, val, test)
        seed: 随机种子
        num_points: 采样点数
        normalize: 是否归一化到单位球
        augment: 是否启用数据增强（Y轴随机旋转）
        augment_factor: 数据增强倍数
        label_smoothing: 标签平滑系数 (0表示不平滑, 仅用于one-hot模式)
        gt_mode: GT生成模式 ('projection' 或 'onehot')
        temperature: 投影softmax的温度参数 (仅用于projection模式)
    """

    # 方向到角度的映射 (XZ平面，Y轴向上)
    DIRECTION_TO_ANGLE = {
        '+X': 0.0,
        '+Z': np.pi / 2,
        '-X': np.pi,
        '-Z': 3 * np.pi / 2,
    }

    def __init__(
        self,
        annotation_file: str = 'data_annotation/symmetry_annotations.json',
        data_root: str = 'data/full_mn40_normal_resampled_ply',
        split: str = 'all',
        categories: List[str] = ['1_front', '4_fronts', 'no_front'],
        num_bins: int = 8,
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
        num_points: int = 2048,
        normalize: bool = True,
        augment: bool = False,
        augment_factor: int = 10,
        label_smoothing: float = 0.0,
        gt_mode: str = 'projection',  # 'projection' or 'onehot'
        temperature: float = 5.0,  # 投影softmax温度
    ):
        self.annotation_file = annotation_file
        self.data_root = Path(data_root)
        self.split = split
        self.categories = categories
        self.num_bins = num_bins
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        self.label_smoothing = label_smoothing
        self.gt_mode = gt_mode
        self.temperature = temperature

        # bin角度 (中心角度)
        self.bin_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        self.bin_width = 2 * np.pi / num_bins

        # 加载标注
        self.samples = self._load_annotations()

        # 划分数据集
        self._split_data()

        # 统计信息
        self._print_stats()

    def _load_annotations(self) -> List[dict]:
        """加载并过滤标注数据"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        for file_path, ann in annotations.items():
            symmetry_name = ann.get('symmetry_name', '')

            # 确定类别
            category = None
            if symmetry_name == '1个正面':
                category = '1_front'
            elif symmetry_name == '4个正面':
                category = '4_fronts'
            elif symmetry_name in ['无正面', '旋转对称']:
                category = 'no_front'

            # 检查是否在要求的类别中
            if category not in self.categories:
                continue

            # 获取方向信息
            direction = ann.get('front_direction')

            # 对于有方向的类别，排除 OBLIQUE 和 MULTI
            if category in ['1_front', '4_fronts']:
                if not direction or direction in excluded_directions:
                    continue
                if direction not in self.DIRECTION_TO_ANGLE:
                    continue

            # 检查文件存在
            ply_path = self.data_root / file_path
            if not ply_path.exists():
                continue

            # 获取角度，no_front 类别没有方向，设为0
            if category == 'no_front':
                angle = 0.0  # no_front 不使用角度
            else:
                angle = self.DIRECTION_TO_ANGLE[direction]

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'category': category,
                'direction': direction,
                'angle': angle,
            })

        return samples

    def _split_data(self):
        """划分训练/验证/测试集（分层采样）"""
        if self.split == 'all':
            return

        # 按类别分层采样
        rng = np.random.RandomState(self.seed)

        category_samples = {}
        for sample in self.samples:
            cat = sample['category']
            if cat not in category_samples:
                category_samples[cat] = []
            category_samples[cat].append(sample)

        selected_samples = []

        for cat, cat_samples in category_samples.items():
            indices = rng.permutation(len(cat_samples))
            n_total = len(cat_samples)
            n_train = int(n_total * self.split_ratio[0])
            n_val = int(n_total * self.split_ratio[1])

            if self.split == 'train':
                selected_idx = indices[:n_train]
            elif self.split == 'val':
                selected_idx = indices[n_train:n_train + n_val]
            elif self.split == 'test':
                selected_idx = indices[n_train + n_val:]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            selected_samples.extend([cat_samples[i] for i in selected_idx])

        self.samples = selected_samples

    def _print_stats(self):
        """打印数据集统计信息"""
        category_counts = {}
        for sample in self.samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        aug_str = f" x{self.augment_factor}" if self.augment else ""
        print(f"[DiscreteDirectionDataset] {self.split}: {len(self.samples)} samples{aug_str} = {len(self)} total")
        if self.gt_mode == 'projection':
            print(f"  num_bins: {self.num_bins}, gt_mode: {self.gt_mode}, temperature: {self.temperature}")
        else:
            print(f"  num_bins: {self.num_bins}, gt_mode: {self.gt_mode}, label_smoothing: {self.label_smoothing}")
        for cat, count in sorted(category_counts.items()):
            print(f"  - {cat}: {count}")

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        sample = self.samples[sample_idx]

        # 加载点云
        points = self._load_ply(sample['ply_path'])

        # 采样点数
        if self.num_points and len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
            points = points[choice]
        elif self.num_points and len(points) < self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=True)
            points = points[choice]

        # 归一化
        if self.normalize:
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist

        # 数据增强：Y轴随机旋转
        angle_offset = 0.0
        if self.augment:
            angle_offset = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ R.T

        # 生成离散GT
        category = sample['category']
        gt_probs = self._generate_discrete_gt(category, sample['angle'], angle_offset)

        # 计算主方向角度（用于评估）
        gt_angle = (sample['angle'] + angle_offset) % (2 * np.pi)

        return {
            'points': torch.from_numpy(points).float(),
            'gt_probs': torch.from_numpy(gt_probs).float(),
            'gt_angle': torch.tensor(gt_angle).float(),
            'category': category,
            'file': sample['file'],
        }

    def _generate_discrete_gt(self, category: str, base_angle: float, angle_offset: float) -> np.ndarray:
        """生成离散方向GT

        三种模式的区别：
        ┌─────────────┬────────────────────────────────────────────────────────────┐
        │ 模式        │ 说明                                                        │
        ├─────────────┼────────────────────────────────────────────────────────────┤
        │ onehot      │ 【槽分类】将圆周分成N个槽，输出落在哪个槽的one-hot概率      │
        │             │ 输出: 概率分布，和=1，例如 [0,0,1,0,0,0,0,0]               │
        │             │ 损失: CrossEntropy                                         │
        ├─────────────┼────────────────────────────────────────────────────────────┤
        │ projection  │ 【软槽分类】同上，但用cos投影+softmax生成soft label        │
        │             │ 输出: 概率分布，和=1，例如 [0.01,0.05,0.8,0.1,0.02,...]    │
        │             │ 损失: CrossEntropy / KL                                    │
        ├─────────────┼────────────────────────────────────────────────────────────┤
        │ regression  │ 【基础方向投影】计算方向在N个基础方向上的投影值（点积）     │
        │             │ 输出: 投影值，范围[-1,1]，例如 [0.87,0.97,0.5,-0.26,...]   │
        │             │ 损失: MSE / Cosine                                         │
        │             │ 本质: 回归问题，保持物理意义                               │
        ├─────────────┼────────────────────────────────────────────────────────────┤
        │ dr          │ 【DR系列】投影+Softmax，GT是原始cos投影值                  │
        │             │ 输出: 投影值，范围[-1,1]，在loss中会做softmax              │
        │             │ 损失: KL (对pred和gt都做softmax后比较)                     │
        │             │ 本质: 投影回归 + 概率分布比较                              │
        └─────────────┴────────────────────────────────────────────────────────────┘
        """
        if self.gt_mode == 'projection':
            return self._generate_gt_projection(category, base_angle, angle_offset)
        elif self.gt_mode in ['regression', 'dr']:
            # DR模式和regression模式GT相同（原始cos投影值）
            # 区别在于loss：regression用MSE，DR在loss中做softmax后用KL
            return self._generate_gt_regression(category, base_angle, angle_offset)
        else:
            return self._generate_gt_onehot(category, base_angle, angle_offset)

    def _generate_gt_projection(self, category: str, base_angle: float, angle_offset: float) -> np.ndarray:
        """
        投影 + Softmax 方法生成GT

        优点:
        - 无量化误差，保留角度信息
        - 梯度平滑，训练稳定
        - soft label包含更多信息
        """
        if category == 'no_front':
            # 无正面 -> 均匀分布
            return np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float32)

        angle = (base_angle + angle_offset) % (2 * np.pi)

        if category == '1_front':
            # 单方向投影
            projections = np.cos(angle - self.bin_angles)
            gt_probs = self._softmax(projections * self.temperature)

        elif category == '4_fronts':
            # 4个方向叠加
            gt_probs = np.zeros(self.num_bins, dtype=np.float32)
            for i in range(4):
                a = (angle + i * np.pi / 2) % (2 * np.pi)
                projections = np.cos(a - self.bin_angles)
                gt_probs += self._softmax(projections * self.temperature)
            gt_probs /= 4  # 归一化

        return gt_probs.astype(np.float32)

    def _generate_gt_onehot(self, category: str, base_angle: float, angle_offset: float) -> np.ndarray:
        """
        One-hot 方法生成GT (原始方法)
        """
        gt_probs = np.zeros(self.num_bins, dtype=np.float32)

        if category == '1_front':
            angle = (base_angle + angle_offset) % (2 * np.pi)
            bin_idx = self._angle_to_bin(angle)
            gt_probs[bin_idx] = 1.0

        elif category == '4_fronts':
            base = (base_angle + angle_offset) % (2 * np.pi)
            for i in range(4):
                angle = (base + i * np.pi / 2) % (2 * np.pi)
                bin_idx = self._angle_to_bin(angle)
                gt_probs[bin_idx] = 0.25

        elif category == 'no_front':
            gt_probs[:] = 1.0 / self.num_bins

        # 应用标签平滑
        if self.label_smoothing > 0:
            gt_probs = gt_probs * (1 - self.label_smoothing) + self.label_smoothing / self.num_bins

        return gt_probs

    def _generate_gt_regression(self, category: str, base_angle: float, angle_offset: float) -> np.ndarray:
        """
        【基础方向投影】真正的投影回归方法

        与分类方法的本质区别：
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 分类方法 (onehot/projection):                                          │
        │   - 输出是概率分布，和=1                                               │
        │   - 本质是"方向落在哪个槽"的分类问题                                   │
        │   - 用 Softmax + CrossEntropy                                          │
        │                                                                        │
        │ 投影回归方法 (regression):                                             │
        │   - 输出是方向向量在各基础方向上的投影值（点积）                       │
        │   - pᵢ = d · bᵢ = cos(θ - θᵢ)，范围 [-1, 1]                          │
        │   - 保持物理意义：正投影=方向对齐，负投影=方向相反                     │
        │   - 用 Tanh + MSE/Cosine Loss                                          │
        │                                                                        │
        │ 数学原理：                                                             │
        │   方向向量 d = (cos θ, sin θ)                                         │
        │   基础向量 bᵢ = (cos θᵢ, sin θᵢ)                                      │
        │   投影值  pᵢ = d · bᵢ = cos θ cos θᵢ + sin θ sin θᵢ = cos(θ - θᵢ)   │
        └────────────────────────────────────────────────────────────────────────┘

        Returns:
            投影值数组，形状 (num_bins,)，范围 [-1, 1]
            - 1_front: 单峰投影模式，最大值在方向对应位置
            - 4_fronts: 四峰投影模式，4个方向各有一个峰
            - no_front: 全零（无明确方向）
        """
        if category == 'no_front':
            # 无正面 -> 全零投影（无明确方向）
            # 注意：不是均匀分布！全零表示"没有任何方向的偏好"
            return np.zeros(self.num_bins, dtype=np.float32)

        angle = (base_angle + angle_offset) % (2 * np.pi)

        if category == '1_front':
            # 单方向：直接计算投影值，不做 softmax
            # pᵢ = cos(θ - θᵢ)，最大值=1出现在 θ=θᵢ 时
            gt_proj = np.cos(angle - self.bin_angles)

        elif category == '4_fronts':
            # 4个方向叠加投影
            # 每个方向贡献一个 cos 峰，叠加后有4个峰
            gt_proj = np.zeros(self.num_bins, dtype=np.float32)
            for i in range(4):
                a = (angle + i * np.pi / 2) % (2 * np.pi)
                gt_proj += np.cos(a - self.bin_angles)
            # 归一化到 [-1, 1] 范围
            gt_proj = gt_proj / 4.0

        return gt_proj.astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        x_shifted = x - x.max()
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()

    def _angle_to_bin(self, angle: float) -> int:
        """将角度映射到最近的bin"""
        # 将角度归一化到 [0, 2π)
        angle = angle % (2 * np.pi)
        # 找到最近的bin
        bin_idx = int(np.round(angle / self.bin_width)) % self.num_bins
        return bin_idx

    def _load_ply(self, ply_path: str) -> np.ndarray:
        """加载PLY文件"""
        with open(ply_path, 'r') as f:
            lines = f.readlines()

        header_end = 0
        num_vertices = 0
        for i, line in enumerate(lines):
            if 'element vertex' in line:
                num_vertices = int(line.split()[-1])
            if 'end_header' in line:
                header_end = i + 1
                break

        points = []
        for line in lines[header_end:header_end + num_vertices]:
            coords = line.strip().split()[:3]
            points.append([float(x) for x in coords])

        return np.array(points, dtype=np.float32)


def collate_fn(batch):
    """自定义collate函数"""
    return {
        'points': torch.stack([b['points'] for b in batch]),
        'gt_probs': torch.stack([b['gt_probs'] for b in batch]),
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'category': [b['category'] for b in batch],
        'file': [b['file'] for b in batch],
    }


if __name__ == '__main__':
    print("Testing DiscreteDirectionDataset...")

    # 测试各种配置
    configs = [
        # Projection模式 (推荐)
        {'num_bins': 8, 'gt_mode': 'projection', 'temperature': 5.0, 'categories': ['1_front']},
        {'num_bins': 8, 'gt_mode': 'projection', 'temperature': 5.0, 'categories': ['4_fronts']},
        {'num_bins': 8, 'gt_mode': 'projection', 'temperature': 5.0, 'categories': ['no_front']},
        {'num_bins': 8, 'gt_mode': 'projection', 'temperature': 5.0, 'categories': ['1_front', '4_fronts', 'no_front']},
        {'num_bins': 16, 'gt_mode': 'projection', 'temperature': 5.0, 'categories': ['1_front', '4_fronts', 'no_front']},
        # One-hot模式 (对比)
        {'num_bins': 8, 'gt_mode': 'onehot', 'label_smoothing': 0.0, 'categories': ['1_front']},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Bins: {config['num_bins']}, Mode: {config['gt_mode']}")
        if config['gt_mode'] == 'projection':
            print(f"Temperature: {config['temperature']}")
        else:
            print(f"Label Smoothing: {config.get('label_smoothing', 0)}")
        print(f"Categories: {config['categories']}")
        print(f"{'='*60}")

        dataset = DiscreteDirectionDataset(
            split='train',
            num_bins=config['num_bins'],
            categories=config['categories'],
            gt_mode=config['gt_mode'],
            temperature=config.get('temperature', 5.0),
            label_smoothing=config.get('label_smoothing', 0.0),
            augment=False,  # 关闭增强方便观察
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  category: {sample['category']}")
            print(f"  gt_angle: {np.degrees(sample['gt_angle'].item()):.1f}°")
            print(f"  gt_probs: {np.round(sample['gt_probs'].numpy(), 3)}")
            print(f"  gt_probs sum: {sample['gt_probs'].sum():.4f}")
            print(f"  max prob bin: {sample['gt_probs'].argmax().item()} ({sample['gt_probs'].max():.3f})")

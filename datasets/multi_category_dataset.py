"""
Multi-Category Dataset - 多类别方向预测数据集

支持四种对称类型:
- 1_front (1个正面): 4峰同方向
- 2_fronts (2个正面): 4峰中2对方向相同，间隔180°
- 4_fronts (4个正面): 4峰间隔90°
- no_front (无正面): κ=0，均匀分布

用法:
    # 使用所有类别
    dataset = MultiCategoryDataset(split='train', categories=['1_front', '2_fronts', '4_fronts', 'no_front'])

    # 只使用特定类别
    dataset = MultiCategoryDataset(split='train', categories=['1_front'])
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class MultiCategoryDataset(Dataset):
    """
    多类别方向预测数据集

    支持 1_front, 2_fronts, 4_fronts, no_front 四种对称类型

    Args:
        annotation_file: 标注JSON文件路径
        data_root: 点云数据根目录
        split: 数据划分 ('train', 'val', 'test', 'all')
        categories: 要包含的类别列表
        split_ratio: 划分比例 (train, val, test)
        seed: 随机种子
        num_points: 采样点数
        normalize: 是否归一化到单位球
        augment: 是否启用数据增强（Y轴随机旋转）
        augment_factor: 数据增强倍数
        gt_kappa: 有方向类别的GT kappa值
    """

    # 类别名称映射
    CATEGORY_MAP = {
        '1_front': '1个正面',
        '2_fronts': '2个正面',
        '4_fronts': '4个正面',
        'no_front': '无正面',
        'symmetric': '旋转对称',  # 也算无正面
    }

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
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
        num_points: int = 2048,
        normalize: bool = True,
        augment: bool = False,
        augment_factor: int = 10,
        gt_kappa: float = 10.0,
    ):
        self.annotation_file = annotation_file
        self.data_root = Path(data_root)
        self.split = split
        self.categories = categories
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        self.gt_kappa = gt_kappa

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
            elif symmetry_name == '2个正面':
                category = '2_fronts'
            elif symmetry_name == '4个正面':
                category = '4_fronts'
            elif symmetry_name == '无正面':
                category = 'no_front'
            elif symmetry_name == '旋转对称':
                category = 'symmetric'

            # 检查是否在要求的类别中
            if category not in self.categories:
                # 兼容: symmetric 可以被 no_front 包含
                if category == 'symmetric' and 'no_front' in self.categories:
                    category = 'no_front'  # 合并到 no_front
                else:
                    continue

            # 获取方向信息
            direction = ann.get('front_direction')

            # 对于有方向的类别，排除 OBLIQUE 和 MULTI
            if category in ['1_front', '2_fronts', '4_fronts']:
                if not direction or direction in excluded_directions:
                    continue
                if direction not in self.DIRECTION_TO_ANGLE:
                    continue

            # 检查文件存在
            ply_path = self.data_root / file_path
            if not ply_path.exists():
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'category': category,
                'direction': direction,
                'angle': self.DIRECTION_TO_ANGLE.get(direction, 0.0),
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
        print(f"[MultiCategoryDataset] {self.split}: {len(self.samples)} samples{aug_str} = {len(self)} total")
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

        # 生成GT
        category = sample['category']
        gt_mu, gt_kappa, gt_weights = self._generate_gt(
            category, sample['angle'], angle_offset
        )

        # 计算主方向（用于评估）
        if category == '1_front':
            gt_angle = (sample['angle'] + angle_offset) % (2 * np.pi)
        elif category == '2_fronts':
            gt_angle = (sample['angle'] + angle_offset) % (2 * np.pi)  # 第一个峰的角度
        elif category == '4_fronts':
            gt_angle = (sample['angle'] + angle_offset) % (2 * np.pi)  # 第一个峰的角度
        else:  # no_front
            gt_angle = 0.0  # 无意义

        return {
            'points': torch.from_numpy(points).float(),
            'gt_mu': torch.from_numpy(gt_mu).float(),
            'gt_kappa': torch.from_numpy(gt_kappa).float(),
            'gt_weights': torch.from_numpy(gt_weights).float(),
            'gt_angle': torch.tensor(gt_angle).float(),
            'category': category,
            'file': sample['file'],
        }

    def _generate_gt(self, category: str, base_angle: float, angle_offset: float):
        """生成GT标签"""
        gt_mu = np.zeros((4, 2), dtype=np.float32)
        gt_kappa = np.zeros(4, dtype=np.float32)
        gt_weights = np.full(4, 0.25, dtype=np.float32)  # 固定均匀权重

        if category == '1_front':
            # 4峰同方向
            angle = (base_angle + angle_offset) % (2 * np.pi)
            for i in range(4):
                gt_mu[i] = [np.cos(angle), np.sin(angle)]
                gt_kappa[i] = self.gt_kappa

        elif category == '2_fronts':
            # 2峰间隔180° (两对相同方向)
            base = (base_angle + angle_offset) % (2 * np.pi)
            for i in range(4):
                # Peak 0,2 在 base 方向，Peak 1,3 在 base+π 方向
                angle = (base + (i % 2) * np.pi) % (2 * np.pi)
                gt_mu[i] = [np.cos(angle), np.sin(angle)]
                gt_kappa[i] = self.gt_kappa

        elif category == '4_fronts':
            # 4峰间隔90°
            base = (base_angle + angle_offset) % (2 * np.pi)
            for i in range(4):
                angle = (base + i * np.pi / 2) % (2 * np.pi)
                gt_mu[i] = [np.cos(angle), np.sin(angle)]
                gt_kappa[i] = self.gt_kappa

        elif category in ['no_front', 'symmetric']:
            # 无方向，κ=0 表示均匀分布
            # mu 可以是任意值，设为默认的90°间隔
            for i in range(4):
                angle = i * np.pi / 2
                gt_mu[i] = [np.cos(angle), np.sin(angle)]
                gt_kappa[i] = 0.0  # 关键：κ=0

        return gt_mu, gt_kappa, gt_weights

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
        'gt_mu': torch.stack([b['gt_mu'] for b in batch]),
        'gt_kappa': torch.stack([b['gt_kappa'] for b in batch]),
        'gt_weights': torch.stack([b['gt_weights'] for b in batch]),
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'category': [b['category'] for b in batch],
        'file': [b['file'] for b in batch],
    }


if __name__ == '__main__':
    print("Testing MultiCategoryDataset...")

    # 测试各种配置
    configs = [
        {'categories': ['1_front']},
        {'categories': ['4_fronts']},
        {'categories': ['no_front']},
        {'categories': ['1_front', '4_fronts', 'no_front']},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Categories: {config['categories']}")
        print(f"{'='*60}")

        for split in ['train', 'val', 'test']:
            dataset = MultiCategoryDataset(
                split=split,
                categories=config['categories'],
                augment=(split == 'train'),
                augment_factor=5
            )

            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\nSample from {split}:")
                print(f"  category: {sample['category']}")
                print(f"  points: {sample['points'].shape}")
                print(f"  gt_mu: {sample['gt_mu']}")
                print(f"  gt_kappa: {sample['gt_kappa']}")
                print(f"  gt_weights: {sample['gt_weights']}")

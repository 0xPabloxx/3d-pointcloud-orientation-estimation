"""
Single Front Dataset - 单正面方向预测数据集

从 symmetry_annotations.json 加载所有 "1个正面" 的样本
排除 OBLIQUE 和 MULTI 方向的数据

用法:
    dataset = SingleFrontDataset(split='train', augment=True, augment_factor=10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class SingleFrontDataset(Dataset):
    """
    单正面方向预测数据集

    从标注文件加载 "1个正面" 类型的样本，生成4峰von Mises GT（只有1个峰有效）

    Args:
        annotation_file: 标注JSON文件路径
        data_root: 点云数据根目录
        split: 数据划分 ('train', 'val', 'test', 'all')
        split_ratio: 划分比例 (train, val, test)
        seed: 随机种子
        num_points: 采样点数
        normalize: 是否归一化到单位球
        augment: 是否启用数据增强（Y轴随机旋转）
        augment_factor: 数据增强倍数
    """

    # 方向到角度的映射 (XZ平面，Y轴向上)
    # +X: 0°, +Z: 90°, -X: 180°, -Z: 270°
    DIRECTION_TO_ANGLE = {
        '+X': 0.0,
        '+Z': np.pi / 2,
        '-X': np.pi,
        '-Z': 3 * np.pi / 2,
        # N/A 用于旋转对称，这里不会用到
    }

    def __init__(
        self,
        annotation_file: str = 'data_annotation/symmetry_annotations.json',
        data_root: str = 'data/full_mn40_normal_resampled_ply',
        split: str = 'all',
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
        num_points: int = 2048,
        normalize: bool = True,
        augment: bool = False,
        augment_factor: int = 10,
        gt_kappa: float = 10.0,  # GT峰的kappa值
    ):
        self.annotation_file = annotation_file
        self.data_root = Path(data_root)
        self.split = split
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

        aug_str = f" x{self.augment_factor} (online random)" if self.augment else ""
        print(f"[SingleFrontDataset] {split}: {len(self.samples)} samples{aug_str} = {len(self)} total")

    def _load_annotations(self) -> List[dict]:
        """加载并过滤标注数据"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        for file_path, ann in annotations.items():
            # 只要 "1个正面" 类型
            if ann.get('symmetry_name') != '1个正面':
                continue

            # 排除 OBLIQUE 和 MULTI
            direction = ann.get('front_direction')
            if not direction or direction in excluded_directions:
                continue

            # 检查方向是否有效
            if direction not in self.DIRECTION_TO_ANGLE:
                print(f"[Warning] Unknown direction '{direction}' for {file_path}, skipping")
                continue

            # 构建样本
            ply_path = self.data_root / file_path
            if not ply_path.exists():
                print(f"[Warning] PLY file not found: {ply_path}, skipping")
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'direction': direction,
                'angle': self.DIRECTION_TO_ANGLE[direction],
            })

        return samples

    def _split_data(self):
        """划分训练/验证/测试集"""
        if self.split == 'all':
            return

        # 固定随机种子确保划分一致
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(self.samples))

        n_total = len(self.samples)
        n_train = int(n_total * self.split_ratio[0])
        n_val = int(n_total * self.split_ratio[1])

        if self.split == 'train':
            selected = indices[:n_train]
        elif self.split == 'val':
            selected = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.samples = [self.samples[i] for i in selected]

    def __len__(self):
        # 数据集大小 = 样本数 x 增强倍数
        # 每次 __getitem__ 时在线随机增强（不固定种子）
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        # 映射到原始样本索引
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

        # 数据增强：Y轴随机旋转（每次调用都随机，不固定种子）
        angle_offset = 0.0
        if self.augment:
            # 真正的随机旋转，每个epoch每个样本都不同
            angle_offset = np.random.uniform(0, 2 * np.pi)

            # 旋转点云（绕Y轴）
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ R.T

        # 计算GT角度（考虑增强旋转）
        gt_angle = (sample['angle'] + angle_offset) % (2 * np.pi)

        # 生成4峰GT（4个峰都指向同一方向，权重各0.25）
        # mu: (4, 2) - [cos, sin]
        # kappa: (4,)
        gt_mu = np.zeros((4, 2), dtype=np.float32)
        gt_kappa = np.zeros(4, dtype=np.float32)

        # 4个峰都指向同一个正确方向，全部有效
        for i in range(4):
            gt_mu[i] = [np.cos(gt_angle), np.sin(gt_angle)]
            gt_kappa[i] = self.gt_kappa

        return {
            'points': torch.from_numpy(points).float(),  # (N, 3)
            'gt_mu': torch.from_numpy(gt_mu).float(),    # (4, 2)
            'gt_kappa': torch.from_numpy(gt_kappa).float(),  # (4,)
            'gt_angle': torch.tensor(gt_angle).float(),  # scalar
            'file': sample['file'],
            'direction': sample['direction'],
        }

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
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'file': [b['file'] for b in batch],
        'direction': [b['direction'] for b in batch],
    }


if __name__ == '__main__':
    # 测试数据集
    print("Testing SingleFrontDataset...")

    for split in ['train', 'val', 'test']:
        dataset = SingleFrontDataset(split=split, augment=(split == 'train'))
        print(f"{split}: {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  points: {sample['points'].shape}")
            print(f"  gt_mu: {sample['gt_mu'].shape}")
            print(f"  gt_kappa: {sample['gt_kappa']}")
            print(f"  gt_angle: {sample['gt_angle']:.2f} rad")

"""
Symmetry Classifier Dataset - 对称性类型分类数据集

从 symmetry_annotations.json 加载所有标注样本，用于5分类任务
排除 OBLIQUE 和 MULTI 方向的数据

类别:
    0: 1个正面
    1: 2个正面
    2: 4个正面
    3: 旋转对称
    4: 无正面

用法:
    dataset = SymmetryClassifierDataset(split='train', augment=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class SymmetryClassifierDataset(Dataset):
    """
    对称性类型分类数据集

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

    # 类别映射
    CLASS_NAMES = ['1个正面', '2个正面', '4个正面', '旋转对称', '无正面']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

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

        # 加载标注
        self.samples = self._load_annotations()

        # 划分数据集
        self._split_data()

        # 统计类别分布
        self._print_stats()

    def _load_annotations(self) -> List[dict]:
        """加载并过滤标注数据"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        for file_path, ann in annotations.items():
            symmetry_name = ann.get('symmetry_name')
            direction = ann.get('front_direction')

            # 必须有对称类型和方向
            if not symmetry_name or not direction:
                continue

            # 排除 OBLIQUE 和 MULTI
            if direction in excluded_directions:
                continue

            # 映射类别名称（处理旧数据中的别名）
            if symmetry_name in ['完全对称']:
                symmetry_name = '旋转对称'
            elif symmetry_name in ['没有正面']:
                symmetry_name = '无正面'

            # 检查类别是否有效
            if symmetry_name not in self.CLASS_TO_IDX:
                print(f"[Warning] Unknown symmetry '{symmetry_name}' for {file_path}, skipping")
                continue

            # 检查文件是否存在
            ply_path = self.data_root / file_path
            if not ply_path.exists():
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'symmetry_name': symmetry_name,
                'class_idx': self.CLASS_TO_IDX[symmetry_name],
                'direction': direction,
            })

        return samples

    def _split_data(self):
        """划分训练/验证/测试集（按类别分层划分）"""
        if self.split == 'all':
            return

        # 按类别分组
        by_class = {i: [] for i in range(len(self.CLASS_NAMES))}
        for i, sample in enumerate(self.samples):
            by_class[sample['class_idx']].append(i)

        # 对每个类别进行划分
        selected_indices = []
        rng = np.random.RandomState(self.seed)

        for class_idx, indices in by_class.items():
            indices = np.array(indices)
            rng.shuffle(indices)

            n_total = len(indices)
            n_train = int(n_total * self.split_ratio[0])
            n_val = int(n_total * self.split_ratio[1])

            if self.split == 'train':
                selected_indices.extend(indices[:n_train])
            elif self.split == 'val':
                selected_indices.extend(indices[n_train:n_train + n_val])
            elif self.split == 'test':
                selected_indices.extend(indices[n_train + n_val:])

        self.samples = [self.samples[i] for i in selected_indices]

    def _print_stats(self):
        """打印数据集统计"""
        class_counts = {name: 0 for name in self.CLASS_NAMES}
        for sample in self.samples:
            class_counts[sample['symmetry_name']] += 1

        aug_str = f" x{self.augment_factor} (online random)" if self.augment else ""
        print(f"[SymmetryClassifierDataset] {self.split}: {len(self.samples)} samples{aug_str} = {len(self)} total")
        for name, count in class_counts.items():
            print(f"  {name}: {count}")

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
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ R.T

        return {
            'points': torch.from_numpy(points).float(),  # (N, 3)
            'label': torch.tensor(sample['class_idx']).long(),  # scalar
            'file': sample['file'],
            'symmetry_name': sample['symmetry_name'],
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

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = np.zeros(len(self.CLASS_NAMES))
        for sample in self.samples:
            class_counts[sample['class_idx']] += 1

        # 逆频率权重
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(self.CLASS_NAMES)

        return torch.from_numpy(weights).float()


def collate_fn(batch):
    """自定义collate函数"""
    return {
        'points': torch.stack([b['points'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'file': [b['file'] for b in batch],
        'symmetry_name': [b['symmetry_name'] for b in batch],
    }


if __name__ == '__main__':
    # 测试数据集
    print("Testing SymmetryClassifierDataset...")

    for split in ['train', 'val', 'test']:
        dataset = SymmetryClassifierDataset(split=split, augment=(split == 'train'))
        print()

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  points: {sample['points'].shape}")
            print(f"  label: {sample['label']} ({dataset.CLASS_NAMES[sample['label']]})")

    # 测试类别权重
    train_dataset = SymmetryClassifierDataset(split='train')
    weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {weights}")

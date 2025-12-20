"""
Fixed 4-Peak von Mises Dataset
用于训练固定4峰混合von Mises分布的数据加载器

数据集位置: data/symmetry_classification_gt/
支持两种采样模式:
1. random: 从所有样本中随机采样
2. balanced: 每个batch中各类别样本数量平衡

数据增强:
- augment=True 启用在线Y轴随机旋转增强
- augment_factor=10 将数据集扩展10倍 (1000 → 10000)
- 同时旋转点云和GT mu

用法:
    # 随机采样模式
    dataset = Fixed4PeakDataset(mode='random')

    # 带数据增强（训练时使用）
    dataset = Fixed4PeakDataset(mode='random', split='train', augment=True, augment_factor=10)

    # 平衡采样模式（每个batch各类别等量）
    dataset = Fixed4PeakDataset(mode='balanced')
    sampler = dataset.get_balanced_sampler(batch_size=20)  # 每类4个
    loader = DataLoader(dataset, batch_sampler=sampler)

    # 单类别模式
    dataset = Fixed4PeakDataset(mode='single', category='1_front')
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Optional, Tuple
import random


class Fixed4PeakDataset(Dataset):
    """
    Fixed 4-Peak von Mises数据集

    Args:
        data_root: 数据根目录
        mode: 采样模式 ('random', 'balanced', 'single')
        category: 当mode='single'时，指定类别
        split: 数据划分 ('train', 'val', 'test', 'all')
        split_ratio: 划分比例 (train, val, test)
        seed: 随机种子
        num_points: 采样点数（None表示使用全部点）
        normalize: 是否归一化点云到单位球
        augment: 是否启用数据增强（Y轴随机旋转）
        augment_factor: 数据增强倍数（每个样本生成多少个增强版本）
    """

    CATEGORIES = ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']

    def __init__(
        self,
        data_root: str = 'data/symmetry_classification_gt',
        mode: str = 'random',
        category: Optional[str] = None,
        categories: Optional[List[str]] = None,  # 新增：支持多类别过滤
        split: str = 'all',
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
        num_points: Optional[int] = 2048,
        normalize: bool = True,
        augment: bool = False,
        augment_factor: int = 10
    ):
        self.data_root = data_root
        self.mode = mode
        self.category = category
        self.categories = categories  # 新增
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1

        # 验证参数
        assert mode in ['random', 'balanced', 'single'], f"Invalid mode: {mode}"
        if mode == 'single':
            assert category in self.CATEGORIES, f"Invalid category: {category}"
        if categories is not None:
            for cat in categories:
                assert cat in self.CATEGORIES, f"Invalid category: {cat}"

        # 加载数据集信息
        self.dataset_info = self._load_dataset_info()

        # 按类别组织数据
        self.data_by_category = self._organize_by_category()

        # 根据split划分数据
        self.samples = self._split_data()

        # 建立索引映射
        self._build_index()

    def _load_dataset_info(self) -> Dict:
        """加载dataset_info.json"""
        info_path = os.path.join(self.data_root, 'dataset_info.json')
        with open(info_path, 'r') as f:
            return json.load(f)

    def _organize_by_category(self) -> Dict[str, List[str]]:
        """按类别组织文件名"""
        data_by_cat = {cat: [] for cat in self.CATEGORIES}
        for filename, info in self.dataset_info.items():
            cat = info['category']
            if cat in data_by_cat:
                data_by_cat[cat].append(filename)

        # 排序确保可复现
        for cat in data_by_cat:
            data_by_cat[cat].sort()

        return data_by_cat

    def _split_data(self) -> List[str]:
        """根据split划分数据"""
        rng = np.random.RandomState(self.seed)

        all_samples = []

        # 确定要使用的类别
        if self.categories is not None:
            # 新增：使用指定的多个类别
            categories = self.categories
        elif self.mode == 'single' and self.category:
            categories = [self.category]
        else:
            categories = self.CATEGORIES

        for cat in categories:
            files = self.data_by_category[cat].copy()
            rng.shuffle(files)

            n = len(files)
            n_train = int(n * self.split_ratio[0])
            n_val = int(n * self.split_ratio[1])

            if self.split == 'train':
                selected = files[:n_train]
            elif self.split == 'val':
                selected = files[n_train:n_train + n_val]
            elif self.split == 'test':
                selected = files[n_train + n_val:]
            else:  # 'all'
                selected = files

            all_samples.extend(selected)

        return all_samples

    def _build_index(self):
        """建立类别到样本索引的映射"""
        self.category_indices = {cat: [] for cat in self.CATEGORIES}
        for idx, filename in enumerate(self.samples):
            cat = self.dataset_info[filename]['category']
            self.category_indices[cat].append(idx)

    def __len__(self) -> int:
        return len(self.samples) * self.augment_factor

    def _get_rotation_matrix_y(self, angle_rad: float) -> np.ndarray:
        """获取Y轴旋转矩阵"""
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)

    def _rotate_mu(self, mu: np.ndarray, angle_rad: float) -> np.ndarray:
        """旋转mu (cos θ, sin θ) -> (cos(θ+angle), sin(θ+angle))

        使用2D旋转矩阵:
        [cos(a), -sin(a)]   [cos θ]   [cos(θ+a)]
        [sin(a),  cos(a)] × [sin θ] = [sin(θ+a)]
        """
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_2d = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)
        # mu: (4, 2) -> 转置后旋转再转置回来
        return (rot_2d @ mu.T).T

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 计算真实样本索引（增强时虚拟扩展了数据集长度）
        base_idx = idx % len(self.samples)

        filename = self.samples[base_idx]
        info = self.dataset_info[filename]
        category = info['category']

        # 读取点云
        ply_path = os.path.join(self.data_root, category, filename)
        points = self._read_ply(ply_path)

        # 读取GT
        gt_path = os.path.join(self.data_root, category, filename.replace('.ply', '_gt.txt'))
        gt = self._read_gt(gt_path)

        # 采样点数
        if self.num_points and len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif self.num_points and len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]

        # 归一化（在旋转前进行，确保围绕原点旋转）
        if self.normalize:
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
            if max_dist > 0:
                points = points / max_dist

        # 分离GT的各个部分
        weights = gt[:, 0]      # (4,)
        mu = gt[:, 1:3]         # (4, 2) - [cos, sin]
        kappa = gt[:, 3]        # (4,)

        # 数据增强：随机Y轴旋转
        if self.augment:
            # 生成随机旋转角度 (0 到 2π)
            rotation_angle = np.random.uniform(0, 2 * np.pi)

            # 旋转点云（Y轴旋转）
            rot_matrix = self._get_rotation_matrix_y(rotation_angle)
            points = points @ rot_matrix.T

            # 旋转GT mu
            mu = self._rotate_mu(mu, rotation_angle)

        # 转换为tensor
        points = torch.from_numpy(points).float()
        weights = torch.from_numpy(weights).float()
        mu = torch.from_numpy(mu).float()
        kappa = torch.from_numpy(kappa).float()

        # 类别标签
        category_idx = self.CATEGORIES.index(category)

        return {
            'points': points,              # (N, 3)
            'weights': weights,            # (4,)
            'mu': mu,                      # (4, 2)
            'kappa': kappa,                # (4,)
            'category': category_idx,      # int
            'category_name': category,     # str
            'filename': filename           # str
        }

    def _read_ply(self, filepath: str) -> np.ndarray:
        """读取PLY文件"""
        points = []
        with open(filepath, 'r') as f:
            in_header = True
            vertex_count = 0
            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line == 'end_header':
                        in_header = False
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(points) >= vertex_count:
                            break
        return np.array(points, dtype=np.float32)

    def _read_gt(self, filepath: str) -> np.ndarray:
        """读取GT文件"""
        gt = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    weight = float(parts[0])
                    mu_cos = float(parts[1])
                    mu_sin = float(parts[2])
                    kappa = float(parts[3])
                    gt.append([weight, mu_cos, mu_sin, kappa])
        return np.array(gt, dtype=np.float32)

    def get_balanced_sampler(self, batch_size: int) -> 'BalancedBatchSampler':
        """获取平衡采样器"""
        return BalancedBatchSampler(self, batch_size)

    def get_category_samples(self, category: str) -> List[int]:
        """获取某类别的所有样本索引"""
        return self.category_indices.get(category, [])

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total_base': len(self.samples),
            'total_augmented': len(self),
            'augment_factor': self.augment_factor,
            'augment_enabled': self.augment,
            'by_category': {cat: len(indices) for cat, indices in self.category_indices.items()}
        }
        return stats


class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    确保每个batch中各类别样本数量相等

    Args:
        dataset: Fixed4PeakDataset实例
        batch_size: 批次大小（必须能被类别数整除）
    """

    def __init__(self, dataset: Fixed4PeakDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_categories = len(dataset.CATEGORIES)

        # 确保batch_size能被类别数整除
        assert batch_size % self.n_categories == 0, \
            f"batch_size ({batch_size}) must be divisible by n_categories ({self.n_categories})"

        self.samples_per_category = batch_size // self.n_categories

        # 计算epoch中的batch数
        min_samples = min(len(dataset.category_indices[cat]) for cat in dataset.CATEGORIES)
        self.n_batches = min_samples // self.samples_per_category

    def __iter__(self):
        # 为每个类别随机打乱索引
        category_indices = {}
        for cat in self.dataset.CATEGORIES:
            indices = self.dataset.category_indices[cat].copy()
            random.shuffle(indices)
            category_indices[cat] = indices

        # 生成batch
        for batch_idx in range(self.n_batches):
            batch = []
            for cat in self.dataset.CATEGORIES:
                start = batch_idx * self.samples_per_category
                end = start + self.samples_per_category
                batch.extend(category_indices[cat][start:end])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class SingleCategorySampler(Sampler):
    """
    单类别采样器
    只从指定类别中采样

    Args:
        dataset: Fixed4PeakDataset实例
        category: 类别名称
        num_samples: 每个epoch的样本数（None表示使用全部）
    """

    def __init__(
        self,
        dataset: Fixed4PeakDataset,
        category: str,
        num_samples: Optional[int] = None
    ):
        self.dataset = dataset
        self.category = category
        self.indices = dataset.category_indices[category]
        self.num_samples = num_samples or len(self.indices)

    def __iter__(self):
        indices = self.indices.copy()
        random.shuffle(indices)

        if self.num_samples <= len(indices):
            return iter(indices[:self.num_samples])
        else:
            # 重复采样
            return iter(random.choices(indices, k=self.num_samples))

    def __len__(self):
        return self.num_samples


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数
    """
    return {
        'points': torch.stack([item['points'] for item in batch]),
        'weights': torch.stack([item['weights'] for item in batch]),
        'mu': torch.stack([item['mu'] for item in batch]),
        'kappa': torch.stack([item['kappa'] for item in batch]),
        'category': torch.tensor([item['category'] for item in batch]),
        'category_name': [item['category_name'] for item in batch],
        'filename': [item['filename'] for item in batch]
    }


# ============== 测试代码 ==============
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("Testing Fixed4PeakDataset")
    print("=" * 60)

    # 测试1: 随机模式
    print("\n[Test 1] Random mode, all data")
    dataset = Fixed4PeakDataset(mode='random', split='all')
    print(f"  Total samples: {len(dataset)}")
    print(f"  Stats: {dataset.get_stats()}")

    # 测试单个样本
    sample = dataset[0]
    print(f"  Sample keys: {sample.keys()}")
    print(f"  Points shape: {sample['points'].shape}")
    print(f"  Weights: {sample['weights']}")
    print(f"  Mu shape: {sample['mu'].shape}")
    print(f"  Kappa: {sample['kappa']}")
    print(f"  Category: {sample['category_name']}")

    # 测试2: 平衡采样模式
    print("\n[Test 2] Balanced sampling")
    dataset = Fixed4PeakDataset(mode='balanced', split='train')
    sampler = dataset.get_balanced_sampler(batch_size=20)  # 每类4个
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    print(f"  Train samples: {len(dataset)}")
    print(f"  Batches per epoch: {len(sampler)}")

    # 检查一个batch的类别分布
    for batch in loader:
        categories = batch['category_name']
        from collections import Counter
        dist = Counter(categories)
        print(f"  Batch category distribution: {dict(dist)}")
        break

    # 测试3: 单类别模式
    print("\n[Test 3] Single category mode")
    dataset = Fixed4PeakDataset(mode='single', category='1_front', split='all')
    print(f"  1_front samples: {len(dataset)}")

    # 测试4: Train/Val/Test划分
    print("\n[Test 4] Train/Val/Test split")
    for split in ['train', 'val', 'test']:
        dataset = Fixed4PeakDataset(mode='random', split=split)
        print(f"  {split}: {len(dataset)} samples")

    # 测试5: 数据增强模式
    print("\n[Test 5] Data augmentation (10x)")
    dataset = Fixed4PeakDataset(mode='random', split='train', augment=True, augment_factor=10)
    print(f"  Stats: {dataset.get_stats()}")

    # 验证同一个base样本的多个增强版本
    print("\n  Checking augmentation on same base sample:")
    base_sample = dataset[0]  # 第0个样本
    aug_sample = dataset[len(dataset.samples)]  # 第0个样本的第2个增强版本

    print(f"    Base sample filename: {base_sample['filename']}")
    print(f"    Aug sample filename: {aug_sample['filename']} (should be same)")
    print(f"    Base mu[0]: {base_sample['mu'][0].numpy()}")
    print(f"    Aug mu[0]:  {aug_sample['mu'][0].numpy()}")

    # 验证mu的模长仍为1（归一化）
    mu_norm_base = torch.norm(base_sample['mu'][0]).item()
    mu_norm_aug = torch.norm(aug_sample['mu'][0]).item()
    print(f"    Base mu[0] norm: {mu_norm_base:.6f}")
    print(f"    Aug mu[0] norm: {mu_norm_aug:.6f}")

    # 验证角度确实不同
    base_angle = torch.atan2(base_sample['mu'][0, 1], base_sample['mu'][0, 0]).item()
    aug_angle = torch.atan2(aug_sample['mu'][0, 1], aug_sample['mu'][0, 0]).item()
    print(f"    Base angle: {np.rad2deg(base_angle):.1f}°")
    print(f"    Aug angle: {np.rad2deg(aug_angle):.1f}°")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

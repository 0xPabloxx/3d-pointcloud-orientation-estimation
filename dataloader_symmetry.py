#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对称性分类数据集 (Symmetry Classification Dataset)

功能：
- 加载标注好的对称性数据
- 随机旋转增强（围绕upright轴）
- 支持train/val/test划分
- 输出点云和K值标签

作者: Claude
创建日期: 2025-11-25
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict


def read_ply(file_path):
    """读取PLY文件"""
    with open(file_path, 'r') as f:
        while True:
            if f.readline().strip() == "end_header":
                break
        try:
            points = np.loadtxt(f)
        except Exception as e:
            raise RuntimeError(f"读取点云数据时出错: {e}")
    return points


def sample_points(points, num_points=10000):
    """采样固定数量的点"""
    if points.shape[0] >= num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        idx = np.random.choice(points.shape[0], num_points, replace=True)
    return points[idx, :]


def normalize_pc(pc):
    """归一化点云到单位球"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / max_dist
    return pc


def rotate_around_y(pc, theta):
    """围绕Y轴旋转点云

    Args:
        pc: (N, 3) 点云
        theta: 旋转角度（弧度）

    Returns:
        rotated_pc: (N, 3) 旋转后的点云
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Y轴旋转矩阵
    rotation_matrix = np.array([
        [cos_t,  0, sin_t],
        [0,      1, 0    ],
        [-sin_t, 0, cos_t]
    ])

    rotated_pc = pc @ rotation_matrix.T
    return rotated_pc


def rotate_direction_vector(direction, theta):
    """旋转方向向量（围绕Y轴）

    Args:
        direction: str, 如 '-Z', '+X' 等
        theta: 旋转角度（弧度）

    Returns:
        rotated_vec: (3,) numpy array, 旋转后的方向向量
    """
    # 方向到向量的映射
    dir_to_vec = {
        '-Z': np.array([0, 0, -1]),
        '+Z': np.array([0, 0, 1]),
        '-X': np.array([-1, 0, 0]),
        '+X': np.array([1, 0, 0]),
        '-Y': np.array([0, -1, 0]),
        '+Y': np.array([0, 1, 0])
    }

    vec = dir_to_vec[direction].astype(np.float32)

    # Y轴旋转矩阵
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation_matrix = np.array([
        [cos_t,  0, sin_t],
        [0,      1, 0    ],
        [-sin_t, 0, cos_t]
    ])

    rotated_vec = rotation_matrix @ vec
    return rotated_vec


class SymmetryDataset(Dataset):
    """对称性分类数据集

    标签映射:
        K=-1 (没有正面) → 0
        K=0  (完全对称) → 1
        K=1  (1个正面)  → 2
        K=2  (2个正面)  → 3
        K=4  (4个正面)  → 4
    """

    def __init__(self,
                 annotation_file: str,
                 data_dir: str,
                 split: str = 'train',
                 num_points: int = 10000,
                 augment: bool = True,
                 split_ratio: tuple = (0.7, 0.15, 0.15)):
        """
        Args:
            annotation_file: 标注文件路径 (JSON)
            data_dir: 数据集根目录
            split: 'train', 'val', 'test'
            num_points: 采样点数
            augment: 是否使用旋转增强
            split_ratio: (train, val, test) 比例
        """
        self.annotation_file = annotation_file
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment and (split == 'train')  # 只在训练时增强

        # K值到类别索引的映射
        self.K_to_idx = {-1: 0, 0: 1, 1: 2, 2: 3, 4: 4}
        self.idx_to_K = {v: k for k, v in self.K_to_idx.items()}

        # 加载标注
        self.annotations = self._load_annotations()

        # 划分数据集
        self.samples = self._split_dataset(split_ratio)

        print(f"[{split.upper()}] 数据集大小: {len(self.samples)}")
        self._print_class_distribution()

    def _load_annotations(self):
        """加载标注文件"""
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        # 过滤掉未完成标注的样本
        valid_annotations = {}
        for file_path, ann in annotations.items():
            if ann.get('K') is not None and ann.get('front_direction'):
                valid_annotations[file_path] = ann

        print(f"总标注数: {len(valid_annotations)}")
        return valid_annotations

    def _split_dataset(self, split_ratio):
        """划分数据集，按K值分层采样"""
        train_ratio, val_ratio, test_ratio = split_ratio
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio必须和为1"

        # 按K值分组
        K_to_samples = defaultdict(list)
        for file_path, ann in self.annotations.items():
            K = ann['K']
            K_to_samples[K].append(file_path)

        train_samples = []
        val_samples = []
        test_samples = []

        # 对每个K值进行分层采样
        np.random.seed(42)  # 固定随机种子，确保可复现
        for K, samples in K_to_samples.items():
            samples = np.array(samples)
            np.random.shuffle(samples)

            n = len(samples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train+n_val])
            test_samples.extend(samples[n_train+n_val:])

        # 打乱顺序
        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        np.random.shuffle(test_samples)

        # 返回对应split的样本
        if self.split == 'train':
            return train_samples
        elif self.split == 'val':
            return val_samples
        else:  # test
            return test_samples

    def _print_class_distribution(self):
        """打印类别分布"""
        K_counts = defaultdict(int)
        for sample in self.samples:
            K = self.annotations[sample]['K']
            K_counts[K] += 1

        print(f"[{self.split.upper()}] 类别分布:")
        for K in sorted(K_counts.keys()):
            idx = self.K_to_idx[K]
            count = K_counts[K]
            percentage = count / len(self.samples) * 100
            print(f"  K={K:2d} (class {idx}): {count:4d} ({percentage:5.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            points: (num_points, 3) tensor
            upright_vec: (3,) tensor, 总是 [0, 1, 0]
            label: int, 0-4的类别索引
            K_value: int, 原始K值（用于调试）
            front_dir_vec: (3,) tensor, 旋转后的正面方向向量（用于验证）
        """
        # 1. 获取样本信息
        file_path = self.samples[idx]
        annotation = self.annotations[file_path]

        K_value = annotation['K']
        front_direction = annotation['front_direction']

        # 2. 读取点云
        full_path = self.data_dir / file_path
        pc = read_ply(str(full_path))

        # 只取xyz坐标（可能包含法向量）
        if pc.shape[1] > 3:
            pc = pc[:, :3]

        # 3. 归一化
        pc = normalize_pc(pc)

        # 4. 随机旋转（仅训练时）
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            pc = rotate_around_y(pc, theta)
            front_dir_vec = rotate_direction_vector(front_direction, theta)
        else:
            # 验证/测试时不旋转
            theta = 0
            front_dir_vec = rotate_direction_vector(front_direction, 0)

        # 5. 采样固定点数
        pc = sample_points(pc, self.num_points)

        # 6. Upright方向（总是Y轴正方向）
        upright_vec = np.array([0, 1, 0], dtype=np.float32)

        # 7. 类别标签
        label = self.K_to_idx[K_value]

        # 8. 转为tensor
        points = torch.tensor(pc, dtype=torch.float32)
        upright_vec = torch.tensor(upright_vec, dtype=torch.float32)
        front_dir_vec = torch.tensor(front_dir_vec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'points': points,           # (num_points, 3)
            'upright_vec': upright_vec, # (3,)
            'label': label,             # scalar
            'K_value': K_value,         # scalar (for debugging)
            'front_dir_vec': front_dir_vec  # (3,) (for debugging)
        }


def get_symmetry_dataloaders(annotation_file, data_dir, batch_size=32, num_workers=4, num_points=10000):
    """获取train/val/test dataloader

    Args:
        annotation_file: 标注文件路径
        data_dir: 数据集根目录
        batch_size: batch大小
        num_workers: 数据加载线程数
        num_points: 采样点数

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='train',
        num_points=num_points,
        augment=True
    )

    val_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='val',
        num_points=num_points,
        augment=False
    )

    test_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='test',
        num_points=num_points,
        augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据集
    annotation_file = 'data_annotation/symmetry_annotations.json'
    data_dir = 'data/full_mn40_normal_resampled_ply'

    print("="*80)
    print("测试 SymmetryDataset")
    print("="*80)

    # 创建数据集
    train_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='train',
        num_points=10000,
        augment=True
    )

    val_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='val',
        num_points=10000,
        augment=False
    )

    test_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='test',
        num_points=10000,
        augment=False
    )

    # 测试一个样本
    print("\n" + "="*80)
    print("测试样本加载")
    print("="*80)
    sample = train_dataset[0]
    print(f"Points shape: {sample['points'].shape}")
    print(f"Upright vec: {sample['upright_vec']}")
    print(f"Label: {sample['label']} (K={sample['K_value']})")
    print(f"Front direction vec: {sample['front_dir_vec']}")

    # 测试DataLoader
    print("\n" + "="*80)
    print("测试 DataLoader")
    print("="*80)
    train_loader, val_loader, test_loader = get_symmetry_dataloaders(
        annotation_file=annotation_file,
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        num_points=10000
    )

    batch = next(iter(train_loader))
    print(f"Batch points shape: {batch['points'].shape}")
    print(f"Batch upright_vec shape: {batch['upright_vec'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Batch labels: {batch['label']}")
    print(f"Batch K values: {batch['K_value']}")

    print("\n✅ 数据集测试完成！")

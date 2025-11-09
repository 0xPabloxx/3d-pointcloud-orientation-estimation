#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlassBox数据加载器 - 支持旋转数据增强（用于MvM训练）

详细说明：
- 数据源: ModelNet40 glassbox类别的PLY点云 + 预生成的MvM GT
- 数据增强: 12旋转（30°间隔）+ 可选点云抖动
- GT格式: MvM分布参数 (μ, κ, π) × K=4
- 返回: (点云xyz, K, GT_pi, GT_mu, GT_kappa, 类别ID)

核心功能：
1. rotate_pointcloud_y(): Y轴旋转（保持upright不变）
2. GlassBoxDatasetAugmented: 数据集类，自动应用旋转增强

使用方法：
    from dataloader_glassbox_augmented import GlassBoxDatasetAugmented
    dataset = GlassBoxDatasetAugmented(
        gt_root="data/MN40_multi_peak_vM_gt/glass_box",
        ply_root="data/full_mn40_normal_resampled_2d_rotated_ply/glass_box",
        split="train",
        rotation_angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    )

作者: Claude
创建日期: 2025-11-09
最后修改: 2025-11-09
关联文档: experiment_20251109_init_fix_results.md
配套训练脚本: train_pointnetpp_mvm_glassbox_augmented.py
"""
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def read_ply(p):
    """读取 .ply 文件，返回 N×3 的 numpy 数组"""
    with open(p, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == "end_header":
                break
        pts = np.loadtxt(f, dtype=np.float32)[:, :3]
    return pts


def sample_pts(arr, num=10000):
    """随机采样num个点"""
    n = len(arr)
    if n == 0:
        return arr
    idx = np.random.choice(n, num, replace=(n < num))
    return arr[idx]


def rotate_pointcloud_y(xyz, angle_rad):
    """
    绕Y轴旋转点云（水平旋转）

    Args:
        xyz: (N, 3) numpy array
        angle_rad: 旋转角度（弧度）

    Returns:
        rotated: (N, 3) numpy array
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 绕Y轴旋转矩阵
    rot_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)

    return xyz @ rot_matrix.T


def add_jitter(xyz, std=0.01, clip=0.05):
    """添加高斯噪声"""
    noise = np.random.normal(0, std, xyz.shape).astype(np.float32)
    noise = np.clip(noise, -clip, clip)
    return xyz + noise


class GlassBoxDatasetAugmented(Dataset):
    """
    GlassBox专用数据集，支持旋转数据增强

    Args:
        samples: list of (ply_path, gt_txt_path, category)
        num_points: 每个点云采样的点数
        max_K: 最大峰数量（glassbox固定为4）
        rotation_angles: 旋转增强的角度列表（度数），如[0, 30, 60, ..., 330]
        apply_jitter: 是否添加点云抖动
        jitter_std: 抖动的标准差
    """

    def __init__(
        self,
        samples,
        num_points=10000,
        max_K=4,
        rotation_angles=None,
        apply_jitter=False,
        jitter_std=0.01
    ):
        self.base_samples = list(samples)
        self.num_points = num_points
        self.max_K = max_K
        self.apply_jitter = apply_jitter
        self.jitter_std = jitter_std

        # 默认12个旋转角度：每30度一个
        if rotation_angles is None:
            rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        self.rotation_angles = rotation_angles

        # 扩展样本：每个原始样本 × 旋转次数
        self.samples = []
        for ply_p, gt_txt, category in self.base_samples:
            for angle_deg in rotation_angles:
                self.samples.append((ply_p, gt_txt, category, angle_deg))

        print(f"[GlassBoxDataset] Base samples: {len(self.base_samples)}, "
              f"Augmented samples: {len(self.samples)} "
              f"(×{len(rotation_angles)} rotations)")

    @staticmethod
    def _read_mvM(gt_path, max_K=4):
        """读取von Mises mixture参数"""
        mus = []
        kappas = []
        ws = []

        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        if len(lines) < 2:
            raise RuntimeError(f"GT file too short: {gt_path}")

        # 读取K值
        parts = lines[0].split()
        K = int(parts[1])

        # 读取每个峰的参数
        data_lines = lines[2:]
        for ln in data_lines:
            vals = ln.split()
            if len(vals) >= 3:
                mus.append(float(vals[0]))
                kappas.append(float(vals[1]))
                ws.append(float(vals[2]))

        # padding到max_K
        while len(mus) < max_K:
            mus.append(0.0)
            kappas.append(0.0)
            ws.append(0.0)

        arr = np.stack([mus, kappas, ws], axis=1)[:max_K]
        return torch.tensor(arr, dtype=torch.float32), K

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ply_p, gt_txt, category, angle_deg = self.samples[idx]

        # 读取点云
        if not os.path.exists(ply_p):
            raise FileNotFoundError(f"PLY not found: {ply_p}")

        xyz_np = read_ply(ply_p)

        # 旋转点云
        angle_rad = np.deg2rad(angle_deg)
        xyz_rotated = rotate_pointcloud_y(xyz_np, angle_rad)

        # 随机采样
        sampled = sample_pts(xyz_rotated, self.num_points)

        # 添加抖动（可选）
        if self.apply_jitter:
            sampled = add_jitter(sampled, self.jitter_std)

        xyz = torch.from_numpy(sampled.astype(np.float32))

        # 读取GT参数
        if not os.path.exists(gt_txt):
            raise FileNotFoundError(f"GT txt not found: {gt_txt}")

        vm_params, K = self._read_mvM(gt_txt, self.max_K)

        # 调整GT的mu值：旋转angle_deg后，正面方向也要跟着旋转
        # mu_new = mu_old - angle_rad (逆时针旋转angle相当于正面方向顺时针转)
        vm_params_adjusted = vm_params.clone()
        for i in range(K):
            old_mu = vm_params[i, 0].item()
            new_mu = (old_mu - angle_rad) % (2 * math.pi)
            # 标准化到[-π, π]
            if new_mu > math.pi:
                new_mu -= 2 * math.pi
            vm_params_adjusted[i, 0] = new_mu

        # 返回：点云、调整后的GT、峰数量、旋转角度（用于调试）
        return xyz, vm_params_adjusted, K, torch.tensor(angle_deg, dtype=torch.float32)
"""
Fixed 4-Peak von Mises 训练脚本

训练目标：预测固定4峰的混合von Mises分布参数
- μ: (B, 4, 2) - 4个峰的方向 (cos, sin)
- κ: (B, 4) - 4个峰的集中度

Loss设计：
- 纯 KL Divergence Loss（离散化）
- 不加额外的 μ/κ 监督，完全信任 KL 的梯度引导
- 避免了匈牙利匹配问题（简单max匹配会导致mode collapse）

用法：
    python train_fixed_4peak.py --config configs/fixed_4peak.yaml
    python train_fixed_4peak.py --batch_size 32 --epochs 100
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.fixed_4peak_dataset import Fixed4PeakDataset, BalancedBatchSampler, collate_fn


# ============== Loss Functions ==============

class FixedWeightDiscretizedKLLoss(nn.Module):
    """
    KL Divergence Loss + κ 正则化 for Mixture of von Mises distributions

    关键设计：
    - 主 Loss: KL Divergence (分布形状匹配)
    - 辅助 Loss: κ 正则化 (防止 κ 坍塌到 0)

    问题背景：
    - 纯 KL loss 会让模型学到 κ≈0 (均匀分布) 的"安全"输出
    - 因为 symmetric/no_front 类别的 GT 就是均匀分布
    - 当 κ→0 时，μ 的梯度消失，模型学不动方向

    解决方案：
    - 添加 κ 监督：强制预测的 κ 接近 GT 的 κ
    - 使用匈牙利匹配来对齐预测峰和 GT 峰
    """

    def __init__(
        self,
        n_bins: int = 360,
        eps: float = 1e-10,
        lambda_kl: float = 1.0,     # KL 权重 (设为 0 则不使用 KL)
        lambda_kappa: float = 5.0,  # κ 监督权重
        lambda_mu: float = 2.0,     # μ 监督权重 (角度 loss 范围 [0, 2])
        reverse_kl: bool = False    # 新增：使用 Reverse KL
    ):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps
        self.weight = 0.25  # 固定权重
        self.lambda_kl = lambda_kl
        self.lambda_kappa = lambda_kappa
        self.lambda_mu = lambda_mu
        self.reverse_kl = reverse_kl

        # 预计算网格点 [0, 2π)
        grid = torch.linspace(0, 2 * np.pi, n_bins + 1)[:-1]  # 不包含 2π (与0重复)
        self.register_buffer('grid', grid)  # (n_bins,)

        # 网格宽度 (用于积分)
        self.bin_width = 2 * np.pi / n_bins

    def _mu_to_angle(self, mu: torch.Tensor) -> torch.Tensor:
        """
        将 (cos, sin) 向量转换为角度 [0, 2π)

        Args:
            mu: (B, 4, 2) - [cos, sin]

        Returns:
            angles: (B, 4) - 角度值 [0, 2π)
        """
        # atan2(sin, cos) 返回 [-π, π]
        angles = torch.atan2(mu[..., 1], mu[..., 0])
        # 转换到 [0, 2π)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        return angles

    def _log_von_mises_pdf(
        self,
        theta: torch.Tensor,
        mu: torch.Tensor,
        kappa: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 von Mises PDF 的对数值 (数值稳定版本)

        log VM(θ; μ, κ) = κ cos(θ - μ) - log(2π) - [log(I₀ᵉ(κ)) + κ]

        其中 I₀ᵉ(κ) = I₀(κ) * exp(-κ) 是 scaled Bessel function

        Args:
            theta: (n_bins,) - 网格点角度
            mu: (B, 4) - 各分量的方向角度
            kappa: (B, 4) - 各分量的集中度

        Returns:
            log_pdf: (B, 4, n_bins) - 各分量在各网格点的 log PDF
        """
        # theta: (n_bins,) -> (1, 1, n_bins)
        # mu: (B, 4) -> (B, 4, 1)
        # kappa: (B, 4) -> (B, 4, 1)
        theta = theta.view(1, 1, -1)
        mu = mu.unsqueeze(-1)
        kappa = kappa.unsqueeze(-1)

        # cos(θ - μ)
        cos_diff = torch.cos(theta - mu)  # (B, 4, n_bins)

        # log(I₀ᵉ(κ)) where I₀ᵉ(κ) = I₀(κ) * exp(-κ)
        # 使用 torch.special.i0e 保证数值稳定
        log_i0e = torch.log(torch.special.i0e(kappa) + self.eps)  # (B, 4, 1)

        # log VM = κ cos(θ-μ) - log(2π) - log(I₀ᵉ(κ)) - κ
        log_pdf = kappa * cos_diff - np.log(2 * np.pi) - log_i0e - kappa

        return log_pdf  # (B, 4, n_bins)

    def _compute_mixture_pdf(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Mixture of von Mises 的 PDF

        PDF(θ) = Σᵢ 0.25 × VM(θ; μᵢ, κᵢ)

        Args:
            mu: (B, 4, 2) - 方向向量 [cos, sin]
            kappa: (B, 4) - 集中度

        Returns:
            pdf: (B, n_bins) - 混合分布在网格点的 PDF
        """
        # 转换 mu 向量为角度
        mu_angles = self._mu_to_angle(mu)  # (B, 4)

        # 计算各分量的 log PDF
        log_pdf_components = self._log_von_mises_pdf(
            self.grid, mu_angles, kappa
        )  # (B, 4, n_bins)

        # 转回 PDF 空间
        pdf_components = torch.exp(log_pdf_components)  # (B, 4, n_bins)

        # 加权求和 (固定权重 0.25)
        mixture_pdf = self.weight * pdf_components.sum(dim=1)  # (B, n_bins)

        return mixture_pdf

    def _hungarian_match(
        self,
        pred_mus: torch.Tensor,
        pred_kappas: torch.Tensor,
        gt_mus: torch.Tensor,
        gt_kappas: torch.Tensor
    ) -> tuple:
        """
        使用匈牙利算法匹配预测峰和 GT 峰

        Cost = 角度距离 + κ 距离

        Returns:
            matched_pred_kappas: (B, 4) - 重排后的预测 κ (保持梯度)
            matched_gt_kappas: (B, 4) - 对应的 GT κ
            matched_pred_angles: (B, 4) - 重排后的预测角度 (保持梯度)
            matched_gt_angles: (B, 4) - 对应的 GT 角度
        """
        from scipy.optimize import linear_sum_assignment

        B = pred_mus.shape[0]
        device = pred_mus.device

        # 转换为角度
        pred_angles = self._mu_to_angle(pred_mus)  # (B, 4)
        gt_angles = self._mu_to_angle(gt_mus)  # (B, 4)

        # 收集匹配后的索引
        pred_indices = []
        gt_indices = []

        for b in range(B):
            # 计算 cost matrix: 角度距离 (考虑周期性)
            # cost[i, j] = 1 - cos(pred_angle[i] - gt_angle[j])
            pred_ang = pred_angles[b]  # (4,)
            gt_ang = gt_angles[b]  # (4,)

            # (4, 1) - (1, 4) -> (4, 4)
            angle_diff = pred_ang.unsqueeze(1) - gt_ang.unsqueeze(0)
            angle_cost = 1 - torch.cos(angle_diff)  # [0, 2]

            # κ 差异作为辅助 cost (归一化到 [0, 1])
            kappa_diff = torch.abs(pred_kappas[b].unsqueeze(1) - gt_kappas[b].unsqueeze(0)) / 10.0

            # 总 cost
            cost_matrix = angle_cost + 0.1 * kappa_diff

            # 匈牙利匹配 (在 detach 后的张量上计算，不影响梯度)
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            pred_indices.append(torch.tensor(row_ind, device=device))
            gt_indices.append(torch.tensor(col_ind, device=device))

        # 使用 gather 保持梯度流
        pred_indices = torch.stack(pred_indices)  # (B, 4)
        gt_indices = torch.stack(gt_indices)  # (B, 4)

        # 重排 kappa (保持梯度)
        matched_pred_kappas = torch.gather(pred_kappas, 1, pred_indices)
        matched_gt_kappas = torch.gather(gt_kappas, 1, gt_indices)

        # 重排角度 (保持梯度)
        matched_pred_angles = torch.gather(pred_angles, 1, pred_indices)
        matched_gt_angles = torch.gather(gt_angles, 1, gt_indices)

        return matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles

    def forward(
        self,
        pred_mus: torch.Tensor,
        pred_kappas: torch.Tensor,
        gt_mus: torch.Tensor,
        gt_kappas: torch.Tensor
    ) -> dict:
        """
        计算 KL Divergence Loss + κ 监督 + μ 监督

        设计决策：
        - 主 Loss: KL Divergence (分布形状匹配)
        - 辅助 Loss 1: κ 监督 (防止 κ 坍塌到 0)
        - 辅助 Loss 2: μ 监督 (强制角度方向正确)

        Args:
            pred_mus: (B, 4, 2) - 预测的方向向量 [cos, sin]
            pred_kappas: (B, 4) - 预测的集中度
            gt_mus: (B, 4, 2) - GT 方向向量 [cos, sin]
            gt_kappas: (B, 4) - GT 集中度 (无效峰 kappa=0)

        Returns:
            dict: {
                'loss': 总 loss,
                'kl_div': KL 散度,
                'kappa_loss': κ 监督 loss,
                'mu_loss': μ 监督 loss
            }
        """
        # === 1. KL Loss (可选) ===
        if self.lambda_kl > 0:
            pred_pdf = self._compute_mixture_pdf(pred_mus, pred_kappas)  # (B, n_bins)

            # GT 不需要梯度
            with torch.no_grad():
                gt_pdf = self._compute_mixture_pdf(gt_mus, gt_kappas)  # (B, n_bins)

            pred_pdf = pred_pdf + self.eps
            gt_pdf = gt_pdf + self.eps

            if self.reverse_kl:
                # Reverse KL: KL(Pred || GT) - 倾向于mode-seeking
                kl_per_bin = pred_pdf * (torch.log(pred_pdf) - torch.log(gt_pdf))
            else:
                # Forward KL: KL(GT || Pred) - 倾向于mode-covering
                kl_per_bin = gt_pdf * (torch.log(gt_pdf) - torch.log(pred_pdf))

            kl_div = kl_per_bin.sum(dim=1) * self.bin_width
            kl_loss = kl_div.mean()
        else:
            kl_loss = torch.tensor(0.0, device=pred_mus.device)

        # === 2. 匈牙利匹配获取对齐的 κ 和 μ ===
        matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles = \
            self._hungarian_match(pred_mus, pred_kappas, gt_mus, gt_kappas)

        # === 3. κ 监督 Loss ===
        # Smooth L1 loss for κ (对大误差更鲁棒)
        # 归一化到 [0, 1] 范围 (GT κ 最大为 10)
        kappa_loss = F.smooth_l1_loss(matched_pred_kappas / 10.0, matched_gt_kappas / 10.0)

        # === 4. μ 监督 Loss ===
        # 角度距离: 1 - cos(θ_pred - θ_gt), 范围 [0, 2]
        # 对于无效峰 (GT κ=0)，不计算 μ loss
        angle_diff = matched_pred_angles - matched_gt_angles
        angle_loss_per_peak = 1 - torch.cos(angle_diff)  # (B, 4), 范围 [0, 2]

        # 创建有效峰掩码 (GT κ > 0 表示有效峰)
        valid_mask = (matched_gt_kappas > 0).float()  # (B, 4)

        # 只对有效峰计算 μ loss
        if valid_mask.sum() > 0:
            mu_loss = (angle_loss_per_peak * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            mu_loss = torch.tensor(0.0, device=pred_mus.device)

        # === 5. Total Loss ===
        total_loss = self.lambda_kl * kl_loss + self.lambda_kappa * kappa_loss + self.lambda_mu * mu_loss

        return {
            'loss': total_loss,
            'kl_div': kl_loss,
            'kappa_loss': kappa_loss,
            'mu_loss': mu_loss
        }


# ============== Model ==============

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS)

    Args:
        xyz: (B, N, 3) 点云坐标
        npoint: 采样点数

    Returns:
        centroids: (B, npoint) 采样点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # 随机选择第一个点
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    根据索引获取点

    Args:
        points: (B, N, C) 点云
        idx: (B, S) 或 (B, S, K) 索引

    Returns:
        indexed_points: (B, S, C) 或 (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball Query - 在半径内找邻居点

    Args:
        radius: 球查询半径
        nsample: 每个球内最大采样点数
        xyz: (B, N, 3) 所有点
        new_xyz: (B, S, 3) 中心点

    Returns:
        group_idx: (B, S, nsample) 邻居点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # 计算距离矩阵
    sqrdists = torch.cdist(new_xyz, xyz, p=2) ** 2  # (B, S, N)

    # 在半径内的点
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N  # 超出半径的设为N（无效）

    # 排序，取最近的nsample个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # 如果邻居不足nsample个，用第一个邻居填充
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer

    采样 + 分组 + PointNet局部特征提取
    """

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list, group_all: bool = False):
        """
        Args:
            npoint: FPS采样点数
            radius: Ball Query半径
            nsample: 每个球内采样点数
            in_channel: 输入通道数
            mlp: MLP各层输出通道数列表
            group_all: 是否对所有点做全局聚合
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel + 3  # +3 for relative xyz
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) 点坐标
            points: (B, N, C) 点特征，可为None

        Returns:
            new_xyz: (B, S, 3) 采样后的点坐标
            new_points: (B, S, D) 采样后的点特征
        """
        if self.group_all:
            # 全局聚合
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # FPS采样
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            # Ball Query分组
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, S, nsample, 3)

            # 相对坐标
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        # (B, S, nsample, C) -> (B, C, S, nsample)
        grouped_points = grouped_points.permute(0, 3, 1, 2)

        # PointNet: 逐点MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling within each group
        new_points = torch.max(grouped_points, dim=3)[0]  # (B, D, S)
        new_points = new_points.permute(0, 2, 1)  # (B, S, D)

        return new_xyz, new_points


class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ 编码器 (SSG - Single Scale Grouping)

    层级结构:
    - SA1: 1024 -> 512 点
    - SA2: 512 -> 128 点
    - SA3: 128 -> 全局聚合

    输出: 1024维全局特征
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1024):
        super().__init__()

        # Set Abstraction layers
        # SA1: 1024 points -> 512 points
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channels - 3,  # 只有xyz时为0
            mlp=[64, 64, 128]
        )

        # SA2: 512 points -> 128 points
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128,
            mlp=[128, 128, 256]
        )

        # SA3: 128 points -> 1 point (全局)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256,
            mlp=[256, 512, out_channels],
            group_all=True
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) 点云坐标

        Returns:
            global_features: (B, out_channels) 全局特征
        """
        B, N, C = x.shape

        # 只用xyz坐标
        xyz = x[:, :, :3]
        points = None  # 没有额外特征

        # 层级特征提取
        xyz, points = self.sa1(xyz, points)  # (B, 512, 3), (B, 512, 128)
        xyz, points = self.sa2(xyz, points)  # (B, 128, 3), (B, 128, 256)
        xyz, points = self.sa3(xyz, points)  # (B, 1, 3), (B, 1, 1024)

        # 全局特征
        global_features = points.squeeze(1)  # (B, 1024)

        return global_features


# 保留原始PointNet作为备选
class PointNetEncoder(nn.Module):
    """简单的PointNet编码器 (备用)"""

    def __init__(self, in_channels: int = 3, out_channels: int = 1024):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, out_channels)

        return x


class Fixed4PeakHead(nn.Module):
    """
    Fixed 4-Peak预测头

    输出:
    - μ: (B, 4, 2) - 4个峰的方向 (cos, sin)，归一化
    - κ: (B, 4) - 4个峰的集中度，非负

    初始化策略:
    - μ head 的 bias 初始化为 4 个随机方向，避免所有峰从相同方向开始
    - 这有助于避免峰之间的对称性问题，加速收敛
    """

    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.3)

        # μ head: 输出 4 * 2 = 8 个值
        self.mu_head = nn.Linear(hidden_channels, 8)

        # κ head: 输出 4 个值
        self.kappa_head = nn.Linear(hidden_channels, 4)

        # 初始化 μ head 的 bias 为 4 个随机方向
        self._init_mu_bias()

    def _init_mu_bias(self):
        """
        初始化 μ head 的 bias 为 2 对相对方向

        策略：峰 1 和 峰 3 相差 180°（一对），峰 2 和 峰 4 相差 180°（另一对）
        这样对于 2-fronts，网络更容易学到"选择峰 1+3 或 峰 2+4"

        结构：
        - Peak 1: θ1 (随机)
        - Peak 2: θ2 = θ1 + 90° (与 Peak 1 垂直)
        - Peak 3: 180° (与 Peak 1 相对)
        - Peak 4: 270° (与 Peak 2 相对)

        使用固定角度以确保实验可重现性。
        """
        with torch.no_grad():
            # 固定角度初始化: [0°, 90°, 180°, 270°]
            # 确保消融实验的可重现性
            angles_deg = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float32)

            # 转换为弧度
            angles_rad = torch.deg2rad(angles_deg)

            # 计算 (cos, sin)
            cos_vals = torch.cos(angles_rad)
            sin_vals = torch.sin(angles_rad)

            # 组合为 bias: [cos0, sin0, cos1, sin1, cos2, sin2, cos3, sin3]
            bias = torch.zeros(8)
            for i in range(4):
                bias[i * 2] = cos_vals[i]
                bias[i * 2 + 1] = sin_vals[i]

            # 设置 bias
            self.mu_head.bias.data = bias

            # 将 weight 初始化为较小的值，让 bias 主导初始输出
            nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)

            print(f"  μ head initialized with FIXED directions: [0°, 90°, 180°, 270°]")

        # 初始化 κ head
        # GT κ 范围: 0 (无方向) 到 10 (有方向)
        # 初始化 bias=5，使初始 κ≈5，偏向多数样本
        self._init_kappa_bias()

    def _init_kappa_bias(self):
        """
        初始化 κ head 的 bias，使初始输出 κ≈5

        GT分布: 60%样本需要κ=10 (1_front/2_fronts/4_fronts)
                40%样本需要κ=0  (symmetric/no_front)

        选择 bias=5 的理由:
        - softplus(5) ≈ 5.01，初始 κ≈5
        - 到 κ=10 只需调整 +5
        - 偏向多数样本(60%需要高κ)
        - 简单直观: bias值 ≈ 期望的初始κ值
        """
        with torch.no_grad():
            self.kappa_head.bias.data.fill_(5.0)

            # 将 weight 初始化为较小的值，让 bias 主导初始输出
            nn.init.xavier_uniform_(self.kappa_head.weight, gain=0.1)

            print(f"  κ head initialized with bias=5.0 (initial κ≈5)")

    def forward(self, x, min_kappa: float = 0.0):
        """
        Forward pass

        Args:
            x: (B, in_channels) - 编码器输出特征
            min_kappa: κ 的硬性下限 (默认 0，设为 5.0 可强制尖峰分布)

        Returns:
            mu: (B, 4, 2) - 4个峰的方向
            kappa: (B, 4) - 4个峰的集中度 (≥ min_kappa)
        """
        # x: (B, in_channels)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))

        # μ: (B, 8) -> (B, 4, 2) -> normalize
        mu = self.mu_head(x).view(-1, 4, 2)
        mu = F.normalize(mu, dim=-1)

        # κ: (B, 4) -> softplus确保非负 + 硬性下限
        # 当 min_kappa > 0 时，强制分布始终尖峰，避免 κ 坍塌
        kappa = F.softplus(self.kappa_head(x)) + min_kappa

        return mu, kappa


class Fixed4PeakModel(nn.Module):
    """完整的Fixed 4-Peak模型 (使用PointNet++编码器)"""

    def __init__(self, encoder_dim: int = 1024, min_kappa: float = 0.0):
        """
        Args:
            encoder_dim: 编码器输出维度
            min_kappa: κ 的硬性下限 (默认 0，设为 5.0 可强制尖峰分布防止坍塌)
        """
        super().__init__()
        self.min_kappa = min_kappa
        # 使用 PointNet++ 编码器 (层级特征学习)
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=encoder_dim)
        self.head = Fixed4PeakHead(in_channels=encoder_dim)

        if min_kappa > 0:
            print(f"  κ 硬性下限: {min_kappa} (强制尖峰分布，防止 κ 坍塌)")

    def forward(self, points):
        """
        Args:
            points: (B, N, 3)

        Returns:
            mu: (B, 4, 2)
            kappa: (B, 4) - 值 ≥ min_kappa
        """
        features = self.encoder(points)
        mu, kappa = self.head(features, min_kappa=self.min_kappa)
        return mu, kappa


# ============== Training ==============

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 生成清晰的实验名称
        # 如果指定了exp_name则使用，否则自动生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.exp_name:
            self.exp_name = f"{args.exp_name}_{timestamp}"
        else:
            # 格式: fixed4peak_pn++_{loss_config}_{timestamp}
            loss_parts = []
            if args.lambda_kl > 0:
                loss_parts.append(f"kl{args.lambda_kl}")
            if args.lambda_kappa > 0:
                loss_parts.append(f"kappa{args.lambda_kappa}")
            if args.lambda_mu > 0:
                loss_parts.append(f"mu{args.lambda_mu}")
            if args.min_kappa > 0:
                loss_parts.append(f"minK{args.min_kappa}")
            loss_config = "_".join(loss_parts) if loss_parts else "no_loss"
            self.exp_name = f"fixed4peak_pn++_{loss_config}_{timestamp}"

        self.output_dir = os.path.join('checkpoints', self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

        # 数据集 (训练集启用数据增强)
        self.train_dataset = Fixed4PeakDataset(
            mode='random',  # 使用 random 模式配合 shuffle
            split='train',
            num_points=args.num_points,
            augment=True,
            augment_factor=args.augment_factor,
            categories=args.categories  # 类别过滤
        )
        self.val_dataset = Fixed4PeakDataset(
            mode='random',
            split='val',
            num_points=args.num_points,
            augment=False,  # 验证集不增强
            categories=args.categories  # 类别过滤
        )

        # DataLoader
        # 训练集: 使用 shuffle，不用 BalancedBatchSampler（增强后每个样本都会随机旋转）
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=True  # 避免最后一个 batch 太小
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

        # 模型
        self.model = Fixed4PeakModel(
            encoder_dim=args.encoder_dim,
            min_kappa=args.min_kappa
        ).to(self.device)

        # Loss: 可配置的 loss 组合
        self.criterion = FixedWeightDiscretizedKLLoss(
            n_bins=360,
            lambda_kl=args.lambda_kl,
            lambda_kappa=args.lambda_kappa,
            lambda_mu=args.lambda_mu,
            reverse_kl=args.reverse_kl
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint (if specified)
        if args.resume:
            self._load_checkpoint(args.resume)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # WandB (可选)
        self.use_wandb = args.wandb
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=self.exp_name,
                config=vars(args),
                dir=self.output_dir,
                resume="allow" if args.resume else None
            )
            wandb.watch(self.model, log='all', log_freq=100)

        print(f"Device: {self.device}")
        print(f"Encoder: PointNet++ (SSG)")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_dataset)} (base: {len(self.train_dataset.samples)}, augment: x{args.augment_factor})")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        loss_desc = f"λ_KL={args.lambda_kl}, λ_κ={args.lambda_kappa}, λ_μ={args.lambda_mu}"
        print(f"Loss: {loss_desc}")
        print(f"Output dir: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.url}")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_mu = batch['mu'].to(self.device)
            gt_kappa = batch['kappa'].to(self.device)

            # Forward
            pred_mu, pred_kappa = self.model(points)

            # Loss (KL Divergence)
            loss_dict = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa)

            # Backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录
            for k, v in loss_dict.items():
                metrics[k].append(v.item())

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"KL={loss_dict['kl_div'].item():.4f}, "
                      f"κ={loss_dict['kappa_loss'].item():.4f}")

        # 平均
        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        metrics = defaultdict(list)
        category_metrics = defaultdict(lambda: defaultdict(list))

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_mu = batch['mu'].to(self.device)
            gt_kappa = batch['kappa'].to(self.device)
            categories = batch['category_name']

            # Forward
            pred_mu, pred_kappa = self.model(points)

            # Loss
            loss_dict = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa)

            for k, v in loss_dict.items():
                metrics[k].append(v.item())

            # 按类别统计
            for i, cat in enumerate(categories):
                cat_loss = self.criterion(
                    pred_mu[i:i+1], pred_kappa[i:i+1],
                    gt_mu[i:i+1], gt_kappa[i:i+1]
                )
                for k, v in cat_loss.items():
                    category_metrics[cat][k].append(v.item())

        # 平均
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        # 计算per-category平均
        avg_category_metrics = {}
        print("\n  Per-category metrics:")
        for cat in Fixed4PeakDataset.CATEGORIES:
            if cat in category_metrics:
                avg_category_metrics[cat] = {
                    k: np.mean(v) for k, v in category_metrics[cat].items()
                }
                cat_kl = avg_category_metrics[cat]['kl_div']
                cat_kappa = avg_category_metrics[cat]['kappa_loss']
                cat_mu = avg_category_metrics[cat]['mu_loss']
                print(f"    {cat:12s}: KL={cat_kl:.4f}, κ={cat_kappa:.4f}, μ={cat_mu:.4f}")

        return avg_metrics, avg_category_metrics

    def train(self):
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        epoch_times = []  # 记录每个epoch的时间
        training_start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 40)

            # Train
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            train_time = time.time() - t0

            # Validate
            val_metrics, val_category_metrics = self.validate()
            epoch_total_time = time.time() - t0
            epoch_times.append(epoch_total_time)

            # Update scheduler
            self.scheduler.step()

            # 计算ETA
            avg_epoch_time = np.mean(epoch_times[-10:])  # 用最近10个epoch的平均时间
            remaining_epochs = self.args.epochs - epoch - 1
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_hour, eta_min = divmod(eta_min, 60)

            elapsed = time.time() - training_start_time
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            elapsed_hour, elapsed_min = divmod(elapsed_min, 60)

            # Log
            lr = self.scheduler.get_last_lr()[0]
            print(f"\n  Train: KL={train_metrics['kl_div']:.4f}, κ={train_metrics['kappa_loss']:.4f}, μ={train_metrics['mu_loss']:.4f}, time={train_time:.1f}s")
            print(f"  Val:   KL={val_metrics['kl_div']:.4f}, κ={val_metrics['kappa_loss']:.4f}, μ={val_metrics['mu_loss']:.4f}")
            print(f"  LR: {lr:.6f}")
            print(f"  Elapsed: {elapsed_hour:02d}:{elapsed_min:02d}:{elapsed_sec:02d} | "
                  f"ETA: {eta_hour:02d}:{eta_min:02d}:{eta_sec:02d} | "
                  f"~{avg_epoch_time:.1f}s/epoch")

            # TensorBoard
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/kl_div', train_metrics['kl_div'], epoch)
            self.writer.add_scalar('train/mu_loss', train_metrics['mu_loss'], epoch)
            self.writer.add_scalar('train/kappa_loss', train_metrics['kappa_loss'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/kl_div', val_metrics['kl_div'], epoch)
            self.writer.add_scalar('val/mu_loss', val_metrics['mu_loss'], epoch)
            self.writer.add_scalar('val/kappa_loss', val_metrics['kappa_loss'], epoch)
            self.writer.add_scalar('lr', lr, epoch)

            # WandB
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/kl_div': train_metrics['kl_div'],
                    'train/mu_loss': train_metrics['mu_loss'],
                    'train/kappa_loss': train_metrics['kappa_loss'],
                    'val/loss': val_metrics['loss'],
                    'val/kl_div': val_metrics['kl_div'],
                    'val/mu_loss': val_metrics['mu_loss'],
                    'val/kappa_loss': val_metrics['kappa_loss'],
                    'lr': lr
                }
                # Per-category metrics
                for cat, cat_metrics in val_category_metrics.items():
                    log_dict[f'val/{cat}/kl_div'] = cat_metrics['kl_div']
                    log_dict[f'val/{cat}/kappa_loss'] = cat_metrics['kappa_loss']
                    log_dict[f'val/{cat}/mu_loss'] = cat_metrics['mu_loss']
                wandb.log(log_dict)

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, is_best)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 60)

        self.writer.close()
        if self.use_wandb:
            wandb.finish()

    def _load_checkpoint(self, checkpoint_path: str):
        """从checkpoint恢复训练状态"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Scheduler state可选（新训练可能不需要）
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"  Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val loss so far: {self.best_val_loss:.6f}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }

        # 保存最新
        torch.save(checkpoint, os.path.join(self.output_dir, 'latest.pth'))

        # 保存最佳
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best.pth'))
            print(f"  [*] New best model saved!")

        # 定期保存
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'epoch_{epoch+1}.pth'))


# ============== Testing ==============

@torch.no_grad()
def test(checkpoint_path: str, args):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = Fixed4PeakModel(encoder_dim=args.encoder_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 测试数据
    test_dataset = Fixed4PeakDataset(
        mode='random',
        split='test',
        num_points=args.num_points
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    criterion = FixedWeightDiscretizedKLLoss(n_bins=360).to(device)
    metrics = defaultdict(list)
    category_metrics = defaultdict(lambda: defaultdict(list))

    print("\nTesting...")
    for batch in test_loader:
        points = batch['points'].to(device)
        gt_mu = batch['mu'].to(device)
        gt_kappa = batch['kappa'].to(device)
        categories = batch['category_name']

        pred_mu, pred_kappa = model(points)
        loss_dict = criterion(pred_mu, pred_kappa, gt_mu, gt_kappa)

        for k, v in loss_dict.items():
            metrics[k].append(v.item())

        for i, cat in enumerate(categories):
            cat_loss = criterion(
                pred_mu[i:i+1], pred_kappa[i:i+1],
                gt_mu[i:i+1], gt_kappa[i:i+1]
            )
            for k, v in cat_loss.items():
                category_metrics[cat][k].append(v.item())

    # 结果
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Overall KL Divergence: {np.mean(metrics['kl_div']):.4f}")

    print("\nPer-category KL:")
    for cat in Fixed4PeakDataset.CATEGORIES:
        if cat in category_metrics:
            print(f"  {cat:12s}: KL={np.mean(category_metrics[cat]['kl_div']):.4f}")


# ============== Main ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Fixed 4-Peak von Mises Training')

    # Data
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augment_factor', type=int, default=10, help='数据增强倍数')

    # Model
    parser.add_argument('--encoder_dim', type=int, default=1024)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss weights
    parser.add_argument('--lambda_kl', type=float, default=1.0,
                        help='KL loss weight (set to 0 to disable KL)')
    parser.add_argument('--lambda_kappa', type=float, default=5.0,
                        help='κ supervision weight')
    parser.add_argument('--lambda_mu', type=float, default=2.0,
                        help='μ supervision weight')

    # Model constraints
    parser.add_argument('--min_kappa', type=float, default=0.0,
                        help='κ 硬性下限 (设为 5.0 强制尖峰分布，防止 κ 坍塌)')

    # Test
    parser.add_argument('--test', action='store_true', help='Run test only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for testing')

    # Data filtering
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='只使用指定类别 (e.g., --categories 4_fronts 2_fronts)')

    # KL direction
    parser.add_argument('--reverse_kl', action='store_true',
                        help='使用 Reverse KL: KL(Pred||GT) 而非 Forward KL: KL(GT||Pred)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练')

    # WandB logging
    parser.add_argument('--wandb', action='store_true',
                        help='启用WandB日志')
    parser.add_argument('--wandb_project', type=str, default='forwardnet',
                        help='WandB项目名')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称 (默认自动生成)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.test:
        if args.checkpoint is None:
            print("Please specify --checkpoint for testing")
            sys.exit(1)
        test(args.checkpoint, args)
    else:
        trainer = Trainer(args)
        trainer.train()

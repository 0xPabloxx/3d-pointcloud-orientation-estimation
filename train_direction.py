"""
Unified Direction Training Script - 统一方向预测训练脚本

支持两种模式:
- MF (Mixture of von Mises): 使用混合von Mises分布
- D (Discrete): 使用离散方向bin分类

用法:
    # MF系列 (mixture von Mises)
    python train_direction.py --mode mf --exp_name MF_1a --loss_type combined

    # D系列 (discrete bins)
    python train_direction.py --mode discrete --exp_name D_8a --num_bins 8
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# PointNet++ 辅助函数
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
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


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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
    """PointNet++ Encoder (与SF/EXP系列一致)"""
    def __init__(self, in_channels: int = 3, out_channels: int = 1024):
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
from datasets.multi_category_dataset import MultiCategoryDataset, collate_fn as mf_collate_fn
from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn as d_collate_fn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Model Heads
# =============================================================================

class MixtureVonMisesHead(nn.Module):
    """
    混合von Mises分布输出头 (用于MF系列)

    输出: mu (B, 4, 2), kappa (B, 4)
    """
    def __init__(self, in_channels=1024, hidden_channels=512, num_peaks=4):
        super().__init__()
        self.num_peaks = num_peaks

        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # mu: 方向向量 (会被归一化)
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)

        # kappa: 集中度 (通过softplus保证非负)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels) 编码器特征
        Returns:
            mu: (B, 4, 2) 归一化方向向量
            kappa: (B, 4) 集中度参数
        """
        h = self.shared(x)

        # mu
        mu = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu = F.normalize(mu, dim=-1)

        # kappa (使用softplus，可以学到0)
        kappa = F.softplus(self.kappa_head(h))

        return mu, kappa


class DiscreteDirectionHead(nn.Module):
    """
    【槽分类】离散方向分类头 (用于D系列 onehot/projection 模式)

    将方向预测视为分类问题：圆周分成 num_bins 个槽，预测落在哪个槽

    输出: logits (B, num_bins) - 未归一化的分数，需经过 Softmax 转为概率
    损失: CrossEntropy / KL Divergence
    """
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
        """
        Args:
            x: (B, in_channels) 编码器特征
        Returns:
            logits: (B, num_bins) 未归一化的分数
        """
        return self.head(x)


class ProjectionRegressionHead(nn.Module):
    """
    【基础方向投影】投影回归头 (用于D系列 regression 模式)

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 与 DiscreteDirectionHead (槽分类) 的本质区别：                             │
    ├────────────────────────────────────────────────────────────────────────────┤
    │                                                                            │
    │  槽分类 (DiscreteDirectionHead):                                           │
    │    点云 → 特征 → [logits] → Softmax → [概率分布] → argmax → 角度          │
    │                                ↑                                           │
    │                           强制和=1                                         │
    │    输出含义: "方向落在第i个槽的概率"                                       │
    │                                                                            │
    │  投影回归 (ProjectionRegressionHead):                                      │
    │    点云 → 特征 → [投影值] → Tanh → [-1,1]范围 → 解码 → 角度               │
    │                               ↑                                            │
    │                          保持物理意义                                      │
    │    输出含义: "方向向量在第i个基础方向上的投影 pᵢ = d·bᵢ = cos(θ-θᵢ)"     │
    │                                                                            │
    │  数学原理:                                                                 │
    │    8个基础方向向量 bᵢ = (cos θᵢ, sin θᵢ), θᵢ = i × 45°                  │
    │    方向向量 d = (cos θ, sin θ)                                            │
    │    投影值 pᵢ = d · bᵢ = cos(θ - θᵢ)                                      │
    │                                                                            │
    │  特性:                                                                     │
    │    - 投影值范围 [-1, 1]，不是概率                                          │
    │    - pᵢ = 1 表示方向与基础方向 i 完全对齐                                 │
    │    - pᵢ = -1 表示方向与基础方向 i 完全相反                                │
    │    - pᵢ = 0 表示方向与基础方向 i 垂直                                     │
    │    - 对向投影相反: p_{i+4} = -p_i (例如 p_+X = -p_-X)                     │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    输出: projections (B, num_dirs) - 投影值，范围 [-1, 1]
    损失: MSE / Cosine Similarity
    """
    def __init__(self, in_channels=1024, hidden_channels=512, num_dirs=8):
        super().__init__()
        self.num_dirs = num_dirs

        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_dirs),
            nn.Tanh(),  # 关键区别：用 Tanh 而不是不加激活，输出范围 [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels) 编码器特征
        Returns:
            projections: (B, num_dirs) 投影值，范围 [-1, 1]
        """
        return self.head(x)


# =============================================================================
# Loss Functions
# =============================================================================

class MixtureVonMisesLoss(nn.Module):
    """
    混合von Mises分布损失函数 (用于MF系列)

    支持三种loss类型:
    - combined: KL + kappa + mu (推荐)
    - reverse_kl: Reverse KL + kappa + mu
    - mu_only: 只使用mu损失
    """
    def __init__(
        self,
        loss_type='combined',
        lambda_kl=1.0,
        lambda_kappa=5.0,
        lambda_mu=2.0,
        grid_size=360,
        eps=1e-8,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.lambda_kappa = lambda_kappa
        self.lambda_mu = lambda_mu
        self.grid_size = grid_size
        self.eps = eps

        # 预计算角度网格
        self.register_buffer(
            'grid',
            torch.linspace(0, 2 * np.pi, grid_size, dtype=torch.float32)
        )

    def forward(self, pred_mu, pred_kappa, gt_mu, gt_kappa, categories=None):
        """
        计算损失

        Args:
            pred_mu: (B, 4, 2) 预测方向
            pred_kappa: (B, 4) 预测集中度
            gt_mu: (B, 4, 2) GT方向
            gt_kappa: (B, 4) GT集中度
            categories: list of str, 类别标签
        """
        losses = {}

        # 固定权重
        weights = torch.full((pred_mu.shape[0], 4), 0.25, device=pred_mu.device)

        # KL Loss
        if self.loss_type in ['combined', 'reverse_kl']:
            pred_pdf = self._compute_mixture_pdf(pred_mu, pred_kappa, weights)
            gt_pdf = self._compute_mixture_pdf(gt_mu, gt_kappa, weights)

            if self.loss_type == 'combined':
                # Forward KL: KL(gt || pred)
                kl_loss = (gt_pdf * (torch.log(gt_pdf + self.eps) - torch.log(pred_pdf + self.eps))).sum(dim=-1).mean()
            else:
                # Reverse KL: KL(pred || gt)
                kl_loss = (pred_pdf * (torch.log(pred_pdf + self.eps) - torch.log(gt_pdf + self.eps))).sum(dim=-1).mean()

            losses['kl_loss'] = kl_loss

        # Kappa Loss (Hungarian matching)
        if self.loss_type != 'mu_only':
            kappa_loss = self._hungarian_kappa_loss(pred_kappa, gt_kappa, pred_mu, gt_mu)
            losses['kappa_loss'] = kappa_loss

        # Mu Loss (Hungarian matching)
        mu_loss = self._hungarian_mu_loss(pred_mu, gt_mu, gt_kappa)
        losses['mu_loss'] = mu_loss

        # 组合总损失
        if self.loss_type == 'combined':
            total_loss = (
                self.lambda_kl * losses['kl_loss'] +
                self.lambda_kappa * losses['kappa_loss'] +
                self.lambda_mu * losses['mu_loss']
            )
        elif self.loss_type == 'reverse_kl':
            total_loss = (
                self.lambda_kl * losses['kl_loss'] +
                self.lambda_kappa * losses['kappa_loss'] +
                self.lambda_mu * losses['mu_loss']
            )
        elif self.loss_type == 'mu_only':
            total_loss = self.lambda_mu * losses['mu_loss']
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        losses['total_loss'] = total_loss
        return losses

    def _compute_mixture_pdf(self, mu, kappa, weights):
        """计算混合von Mises的PDF"""
        B = mu.shape[0]
        device = mu.device
        grid = self.grid.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, G)

        # 计算每个component的角度
        angles = torch.atan2(mu[..., 1], mu[..., 0])  # (B, 4)
        angles = angles.unsqueeze(-1)  # (B, 4, 1)

        # von Mises PDF
        kappa_exp = kappa.unsqueeze(-1)  # (B, 4, 1)
        log_pdf = kappa_exp * torch.cos(grid - angles)

        # 对于kappa接近0的情况，PDF接近均匀分布
        # I_0(kappa) ≈ 1 when kappa ≈ 0
        log_normalizer = torch.log(2 * np.pi * torch.special.i0(kappa_exp) + self.eps)
        log_pdf = log_pdf - log_normalizer

        # 混合
        component_pdf = torch.exp(log_pdf)  # (B, 4, G)
        weights_exp = weights.unsqueeze(-1)  # (B, 4, 1)
        mixture_pdf = (weights_exp * component_pdf).sum(dim=1)  # (B, G)

        # 归一化
        mixture_pdf = mixture_pdf / (mixture_pdf.sum(dim=-1, keepdim=True) + self.eps)

        return mixture_pdf

    def _hungarian_kappa_loss(self, pred_kappa, gt_kappa, pred_mu, gt_mu):
        """使用Hungarian算法匹配后计算kappa损失"""
        from scipy.optimize import linear_sum_assignment

        B = pred_kappa.shape[0]
        total_loss = 0.0

        for b in range(B):
            # 计算角度距离矩阵
            pred_angles = torch.atan2(pred_mu[b, :, 1], pred_mu[b, :, 0])
            gt_angles = torch.atan2(gt_mu[b, :, 1], gt_mu[b, :, 0])

            cost_matrix = torch.zeros(4, 4)
            for i in range(4):
                for j in range(4):
                    angle_diff = torch.abs(pred_angles[i] - gt_angles[j])
                    angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
                    cost_matrix[i, j] = angle_diff

            # Hungarian匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # 计算匹配后的kappa损失
            for i, j in zip(row_ind, col_ind):
                total_loss += F.mse_loss(pred_kappa[b, i], gt_kappa[b, j])

        return total_loss / B

    def _hungarian_mu_loss(self, pred_mu, gt_mu, gt_kappa):
        """使用Hungarian算法匹配后计算mu损失 (只对有效峰)"""
        from scipy.optimize import linear_sum_assignment

        B = pred_mu.shape[0]
        total_loss = 0.0
        valid_count = 0

        for b in range(B):
            # 只考虑有效峰 (kappa > 0)
            valid_mask = gt_kappa[b] > 0.1
            if not valid_mask.any():
                continue

            # 计算角度距离矩阵
            pred_angles = torch.atan2(pred_mu[b, :, 1], pred_mu[b, :, 0])
            gt_angles = torch.atan2(gt_mu[b, :, 1], gt_mu[b, :, 0])

            cost_matrix = torch.zeros(4, 4)
            for i in range(4):
                for j in range(4):
                    angle_diff = torch.abs(pred_angles[i] - gt_angles[j])
                    angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
                    cost_matrix[i, j] = angle_diff

            # Hungarian匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # 计算匹配后的mu损失 (只对有效峰)
            for i, j in zip(row_ind, col_ind):
                if valid_mask[j]:
                    # 角度损失 (1 - cos(diff))
                    angle_diff = pred_angles[i] - gt_angles[j]
                    total_loss += 1 - torch.cos(angle_diff)
                    valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=pred_mu.device)

        return total_loss / valid_count


class DiscreteDirectionLoss(nn.Module):
    """
    【槽分类】离散方向分类损失 (用于D系列 onehot/projection 模式)

    支持:
    - CrossEntropy (soft labels)
    - Focal Loss
    - KL Divergence
    """
    def __init__(self, loss_type='ce', focal_gamma=2.0):
        super().__init__()
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, logits, gt_probs, categories=None):
        """
        计算损失

        Args:
            logits: (B, num_bins) 未归一化的预测
            gt_probs: (B, num_bins) GT概率分布
            categories: list of str, 类别标签
        """
        pred_probs = F.softmax(logits, dim=-1)

        if self.loss_type == 'ce':
            # Cross Entropy with soft labels
            loss = -(gt_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        elif self.loss_type == 'focal':
            # Focal Loss
            ce_loss = -(gt_probs * F.log_softmax(logits, dim=-1))
            pt = pred_probs
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = (focal_weight * ce_loss).sum(dim=-1).mean()

        elif self.loss_type == 'kl':
            # KL Divergence
            loss = F.kl_div(F.log_softmax(logits, dim=-1), gt_probs, reduction='batchmean')

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return {'total_loss': loss, 'ce_loss': loss}


class ProjectionRegressionLoss(nn.Module):
    """
    【基础方向投影】投影回归损失 (用于D系列 regression 模式)

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 与 DiscreteDirectionLoss (槽分类损失) 的本质区别：                         │
    ├────────────────────────────────────────────────────────────────────────────┤
    │                                                                            │
    │  槽分类损失 (DiscreteDirectionLoss):                                       │
    │    - 输入: logits (未归一化) vs gt_probs (概率分布)                        │
    │    - 损失: CrossEntropy，最小化概率分布之间的差异                          │
    │    - 目标: 让预测概率分布接近GT概率分布                                    │
    │                                                                            │
    │  投影回归损失 (ProjectionRegressionLoss):                                  │
    │    - 输入: pred_proj (投影值 [-1,1]) vs gt_proj (投影值 [-1,1])            │
    │    - 损失: MSE / Cosine，最小化投影向量之间的差异                          │
    │    - 目标: 让预测的投影模式接近GT投影模式                                  │
    │                                                                            │
    │  为什么用 Cosine Loss:                                                     │
    │    投影向量的"形状"比"幅度"更重要                                          │
    │    例如 [0.5, 0.9, 0.5, -0.1, ...] 和 [0.25, 0.45, 0.25, -0.05, ...]      │
    │    表示相同的方向，只是幅度不同                                            │
    │    Cosine Loss 只关注形状（角度），忽略幅度                                │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    支持:
    - mse: Mean Squared Error，直接比较投影值
    - cosine: Cosine Similarity Loss，比较投影向量的"形状"
    - combined: MSE + Cosine 组合
    """
    def __init__(self, loss_type='cosine', lambda_mse=1.0, lambda_cos=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_mse = lambda_mse
        self.lambda_cos = lambda_cos

    def forward(self, pred_proj, gt_proj, categories=None):
        """
        计算损失

        Args:
            pred_proj: (B, num_dirs) 预测的投影值，范围 [-1, 1]
            gt_proj: (B, num_dirs) GT投影值，范围 [-1, 1]
            categories: list of str, 类别标签 (用于可能的类别特定处理)

        Returns:
            dict: 包含各项损失
        """
        losses = {}

        if self.loss_type == 'mse':
            # MSE Loss: 直接比较投影值
            mse_loss = F.mse_loss(pred_proj, gt_proj)
            losses['mse_loss'] = mse_loss
            losses['total_loss'] = mse_loss

        elif self.loss_type == 'cosine':
            # Cosine Similarity Loss: 比较投影向量的方向（形状）
            # cosine_similarity 范围 [-1, 1]，1 表示完全相同
            # 损失 = 1 - cosine_similarity，范围 [0, 2]
            cos_sim = F.cosine_similarity(pred_proj, gt_proj, dim=-1)
            cos_loss = (1 - cos_sim).mean()
            losses['cos_loss'] = cos_loss
            losses['total_loss'] = cos_loss

        elif self.loss_type == 'combined':
            # MSE + Cosine 组合
            mse_loss = F.mse_loss(pred_proj, gt_proj)
            cos_sim = F.cosine_similarity(pred_proj, gt_proj, dim=-1)
            cos_loss = (1 - cos_sim).mean()

            losses['mse_loss'] = mse_loss
            losses['cos_loss'] = cos_loss
            losses['total_loss'] = self.lambda_mse * mse_loss + self.lambda_cos * cos_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return losses


class ProjectionSoftmaxLoss(nn.Module):
    """
    【DR系列】投影 + Softmax 损失

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 与 D系列 (DiscreteDirectionLoss) 的根本区别：                              │
    ├────────────────────────────────────────────────────────────────────────────┤
    │                                                                            │
    │  D系列 (槽分类):                                                           │
    │    网络输出: logits (任意值，无物理意义)                                   │
    │    处理: softmax(logits) → 概率分布                                        │
    │    含义: "方向落在哪个bin的概率"                                           │
    │                                                                            │
    │  DR系列 (基础方向投影):                                                    │
    │    网络输出: 投影值回归 (tanh约束到[-1,1]，有物理意义)                     │
    │    处理: softmax(投影值) → 概率分布                                        │
    │    含义: "与各基础方向的相似度分布"                                        │
    │                                                                            │
    │  关键区别:                                                                 │
    │    D系列的logits是任意值，网络可以输出任何值来达到目标分布                 │
    │    DR系列的投影值有物理约束，必须回归到cos(θ-θᵢ)的形式                    │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    输入:
      - pred_proj: (B, num_dirs) 网络预测的投影值，tanh约束到[-1,1]
      - gt_proj: (B, num_dirs) GT投影值 = cos(θ-θᵢ)，范围[-1,1]

    处理:
      - pred_probs = softmax(pred_proj)
      - gt_probs = softmax(gt_proj)
      - loss = KL(pred_probs, gt_probs) 或 CE

    损失类型:
      - ce: CrossEntropy
      - kl: KL Divergence
    """
    def __init__(self, loss_type='kl'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_proj, gt_proj, categories=None):
        """
        计算损失

        Args:
            pred_proj: (B, num_dirs) 预测的投影值，范围 [-1, 1] (来自tanh)
            gt_proj: (B, num_dirs) GT投影值，范围 [-1, 1] (cos投影)
            categories: list of str, 类别标签

        Returns:
            dict: 包含各项损失
        """
        # 对预测和GT都做softmax，得到概率分布
        pred_probs = F.softmax(pred_proj, dim=-1)
        gt_probs = F.softmax(gt_proj, dim=-1)

        if self.loss_type == 'ce':
            # CrossEntropy: -sum(gt * log(pred))
            loss = -torch.sum(gt_probs * torch.log(pred_probs + 1e-10), dim=-1).mean()

        elif self.loss_type == 'kl':
            # KL Divergence: sum(gt * log(gt/pred))
            loss = F.kl_div(
                torch.log(pred_probs + 1e-10),
                gt_probs,
                reduction='batchmean'
            )

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return {'total_loss': loss, 'kl_loss': loss}


def compute_angle_error_dr(pred_proj, gt_angle, categories, num_dirs):
    """
    【DR系列】计算角度误差

    从投影值重建方向：
    1. 对投影值做softmax得到概率分布
    2. 用概率加权求期望方向向量
    3. 计算与GT的角度差

    Args:
        pred_proj: (B, num_dirs) 预测的投影值
        gt_angle: (B,) GT角度
        categories: list of str
        num_dirs: int, 基础方向数量

    Returns:
        mean_error: 平均角度误差（度）
    """
    B = pred_proj.shape[0]
    errors = []

    # 基础方向角度
    dir_angles = torch.linspace(0, 2 * np.pi, num_dirs + 1)[:-1].to(pred_proj.device)

    for b in range(B):
        cat = categories[b]
        if cat == 'no_front':
            continue

        # softmax得到概率
        probs = F.softmax(pred_proj[b], dim=-1)

        # 期望方向向量 (加权平均)
        cos_dirs = torch.cos(dir_angles)
        sin_dirs = torch.sin(dir_angles)
        pred_x = (probs * cos_dirs).sum()
        pred_y = (probs * sin_dirs).sum()
        pred_angle = torch.atan2(pred_y, pred_x)

        # 计算角度误差
        gt = gt_angle[b]

        if cat == '1_front':
            diff = pred_angle - gt
            diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
        elif cat == '4_fronts':
            min_diff = float('inf')
            for i in range(4):
                candidate = (gt + i * np.pi / 2) % (2 * np.pi)
                angle_diff = pred_angle - candidate
                circular_diff = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
                min_diff = min(min_diff, circular_diff.item())
            diff = min_diff

        errors.append(np.degrees(diff if isinstance(diff, float) else diff.item()))

    return np.mean(errors) if errors else 0.0


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_angle_error_mf(pred_mu, pred_kappa, gt_angle, categories):
    """
    计算MF系列的角度误差

    对于1_front: 使用最大kappa峰的方向
    对于4_fronts: 使用最近的峰
    对于no_front: 跳过
    """
    B = pred_mu.shape[0]
    errors = []

    for b in range(B):
        cat = categories[b]

        if cat == 'no_front':
            continue

        # 找到最大kappa的峰
        max_idx = pred_kappa[b].argmax()
        pred_angle = torch.atan2(pred_mu[b, max_idx, 1], pred_mu[b, max_idx, 0])

        # 计算角度误差
        gt = gt_angle[b]

        if cat == '1_front':
            # 直接比较 (正确的圆周距离)
            diff = pred_angle - gt
            diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
        elif cat == '4_fronts':
            # 与最近的90°倍数比较
            min_diff = float('inf')
            for i in range(4):
                candidate = (gt + i * np.pi / 2) % (2 * np.pi)
                angle_diff = pred_angle - candidate
                # 正确的圆周距离
                circular_diff = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
                min_diff = min(min_diff, circular_diff.item())
            diff = torch.tensor(min_diff)

        errors.append(diff.item())

    if len(errors) == 0:
        return 0.0

    return np.mean(errors) * 180 / np.pi


def compute_angle_error_discrete(logits, gt_angle, categories, num_bins):
    """
    计算D系列的角度误差
    """
    B = logits.shape[0]
    errors = []
    bin_width = 2 * np.pi / num_bins

    pred_probs = F.softmax(logits, dim=-1)

    for b in range(B):
        cat = categories[b]

        if cat == 'no_front':
            continue

        # 使用argmax找到预测的bin
        pred_bin = pred_probs[b].argmax()
        pred_angle = pred_bin.float() * bin_width

        gt = gt_angle[b]

        if cat == '1_front':
            # 正确的圆周距离
            diff = pred_angle - gt
            diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
        elif cat == '4_fronts':
            min_diff = float('inf')
            for i in range(4):
                candidate = (gt + i * np.pi / 2) % (2 * np.pi)
                angle_diff = pred_angle - candidate
                # 正确的圆周距离
                circular_diff = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
                min_diff = min(min_diff, circular_diff.item())
            diff = torch.tensor(min_diff)

        errors.append(diff.item())

    if len(errors) == 0:
        return 0.0

    return np.mean(errors) * 180 / np.pi


def compute_angle_error_regression(pred_proj, gt_angle, categories, num_dirs):
    """
    【基础方向投影】计算 regression 模式的角度误差

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 与 compute_angle_error_discrete (槽分类) 的区别：                          │
    ├────────────────────────────────────────────────────────────────────────────┤
    │                                                                            │
    │  槽分类:                                                                   │
    │    pred_probs → argmax → 槽索引 → 槽中心角度                              │
    │    精度受限于槽宽度 (8槽 = 45°)                                           │
    │                                                                            │
    │  投影回归 (本函数):                                                        │
    │    方法1: pred_proj → argmax → 最大投影的基础方向 (与槽分类类似)          │
    │    方法2: pred_proj → 加权平均 → 连续角度 (更精确)                        │
    │    方法3: pred_proj → 向量重建 → arctan2 → 连续角度 (最精确)              │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    Args:
        pred_proj: (B, num_dirs) 预测的投影值，范围 [-1, 1]
        gt_angle: (B,) GT角度 (弧度)
        categories: list of str, 类别标签
        num_dirs: 基础方向数量

    Returns:
        平均角度误差 (度)
    """
    B = pred_proj.shape[0]
    errors = []
    device = pred_proj.device

    # 基础方向角度
    dir_angles = torch.linspace(0, 2 * np.pi, num_dirs + 1, device=device)[:-1]  # 不包含 2π

    for b in range(B):
        cat = categories[b]

        if cat == 'no_front':
            continue

        proj = pred_proj[b]  # (num_dirs,)

        # 方法3: 向量重建法 (最精确)
        # 从投影值重建方向向量，然后用 atan2 计算角度
        # d' = Σ pᵢ * bᵢ，其中 bᵢ = (cos θᵢ, sin θᵢ)
        cos_dirs = torch.cos(dir_angles)  # (num_dirs,)
        sin_dirs = torch.sin(dir_angles)  # (num_dirs,)

        # 只用正投影来重建（负投影表示相反方向）
        weights = F.relu(proj)  # 只保留正投影
        if weights.sum() < 1e-6:
            # 如果所有投影都是负的，用原始投影
            weights = proj - proj.min() + 1e-6

        # 加权平均重建方向向量
        reconstructed_x = (weights * cos_dirs).sum()
        reconstructed_y = (weights * sin_dirs).sum()

        # 计算预测角度
        pred_angle = torch.atan2(reconstructed_y, reconstructed_x)
        # 归一化到 [0, 2π)
        pred_angle = pred_angle % (2 * np.pi)

        gt = gt_angle[b]

        if cat == '1_front':
            # 正确的圆周距离
            diff = pred_angle - gt
            diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
        elif cat == '4_fronts':
            # 与最近的90°倍数比较
            min_diff = float('inf')
            for i in range(4):
                candidate = (gt + i * np.pi / 2) % (2 * np.pi)
                angle_diff = pred_angle - candidate
                circular_diff = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
                min_diff = min(min_diff, circular_diff.item())
            diff = torch.tensor(min_diff)

        errors.append(diff.item())

    if len(errors) == 0:
        return 0.0

    return np.mean(errors) * 180 / np.pi


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建checkpoint目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"checkpoints/{args.exp_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        # 初始化数据集
        self._init_datasets()

        # 初始化模型
        self._init_model()

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # 初始化loss
        self._init_loss()

        # WandB
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                config=vars(args)
            )

        self.best_val_loss = float('inf')
        self.best_val_error = float('inf')
        self.start_epoch = 1

        # Resume from checkpoint if specified
        if args.resume:
            self._load_checkpoint(args.resume)

    def _save_config(self):
        config = vars(self.args)
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {self.exp_dir / 'config.json'}")

    def _init_datasets(self):
        args = self.args
        categories = args.categories.split(',')

        if args.mode == 'mf':
            self.train_dataset = MultiCategoryDataset(
                split='train',
                categories=categories,
                num_points=args.num_points,
                augment=True,
                augment_factor=args.augment_factor,
                gt_kappa=args.gt_kappa,
            )
            self.val_dataset = MultiCategoryDataset(
                split='val',
                categories=categories,
                num_points=args.num_points,
                augment=False,
            )
            collate_fn = mf_collate_fn
        else:  # discrete
            self.train_dataset = DiscreteDirectionDataset(
                split='train',
                categories=categories,
                num_bins=args.num_bins,
                num_points=args.num_points,
                augment=True,
                augment_factor=args.augment_factor,
                gt_mode=args.gt_mode,
                temperature=args.temperature,
                label_smoothing=args.label_smoothing,
            )
            self.val_dataset = DiscreteDirectionDataset(
                split='val',
                categories=categories,
                num_bins=args.num_bins,
                num_points=args.num_points,
                augment=False,
                gt_mode=args.gt_mode,
                temperature=args.temperature,
                label_smoothing=0.0,  # val不用平滑
            )
            collate_fn = d_collate_fn

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _init_model(self):
        args = self.args

        # 共享的encoder (使用PointNet++，与SF/EXP系列一致)
        self.encoder = PointNetPlusPlusEncoder(
            in_channels=3,
            out_channels=1024,
        ).to(self.device)

        # 模式特定的head
        if args.mode == 'mf':
            self.head = MixtureVonMisesHead(
                in_channels=1024,
                hidden_channels=512,
                num_peaks=4,
            ).to(self.device)
        elif args.mode == 'discrete' and args.gt_mode == 'regression':
            # 【基础方向投影】投影回归模式
            # 与槽分类的区别见 ProjectionRegressionHead 文档
            self.head = ProjectionRegressionHead(
                in_channels=1024,
                hidden_channels=512,
                num_dirs=args.num_bins,  # 基础方向数量
            ).to(self.device)
            print(f"[Model] Using ProjectionRegressionHead (基础方向投影回归)")
        elif args.mode == 'discrete' and args.gt_mode == 'dr':
            # 【DR系列】投影 + Softmax 模式
            # 复用 ProjectionRegressionHead，输出tanh投影值，然后在loss中做softmax
            self.head = ProjectionRegressionHead(
                in_channels=1024,
                hidden_channels=512,
                num_dirs=args.num_bins,
            ).to(self.device)
            print(f"[Model] Using ProjectionRegressionHead for DR (投影+Softmax)")
        else:
            # 【槽分类】离散方向分类模式 (onehot/projection)
            self.head = DiscreteDirectionHead(
                in_channels=1024,
                hidden_channels=512,
                num_bins=args.num_bins,
            ).to(self.device)
            print(f"[Model] Using DiscreteDirectionHead (槽分类)")

        # 组合模型
        self.model = nn.ModuleDict({
            'encoder': self.encoder,
            'head': self.head,
        })

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    def _init_loss(self):
        args = self.args

        if args.mode == 'mf':
            self.criterion = MixtureVonMisesLoss(
                loss_type=args.loss_type,
                lambda_kl=args.lambda_kl,
                lambda_kappa=args.lambda_kappa,
                lambda_mu=args.lambda_mu,
            )
        elif args.mode == 'discrete' and args.gt_mode == 'regression':
            # 【基础方向投影】投影回归损失
            # 与槽分类损失的区别见 ProjectionRegressionLoss 文档
            self.criterion = ProjectionRegressionLoss(
                loss_type=args.reg_loss_type,  # mse, cosine, combined
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
            )
            print(f"[Loss] Using ProjectionRegressionLoss (投影回归损失, type={args.reg_loss_type})")
        elif args.mode == 'discrete' and args.gt_mode == 'dr':
            # 【DR系列】投影 + Softmax 损失
            # pred和gt都做softmax后用KL比较
            self.criterion = ProjectionSoftmaxLoss(
                loss_type=args.d_loss_type,  # ce or kl
            )
            print(f"[Loss] Using ProjectionSoftmaxLoss (DR投影+Softmax损失, type={args.d_loss_type})")
        else:
            # 【槽分类】离散方向分类损失 (onehot/projection)
            self.criterion = DiscreteDirectionLoss(
                loss_type=args.d_loss_type,
                focal_gamma=args.focal_gamma,
            )
            print(f"[Loss] Using DiscreteDirectionLoss (槽分类损失, type={args.d_loss_type})")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            B = points.shape[0]

            # Forward (PointNet++ 输入 (B, N, 3)，输出 (B, 1024))
            features = self.encoder(points)

            if self.args.mode == 'mf':
                pred_mu, pred_kappa = self.head(features)
                gt_mu = batch['gt_mu'].to(self.device)
                gt_kappa = batch['gt_kappa'].to(self.device)
                losses = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa, batch['category'])
            elif self.args.mode == 'discrete' and self.args.gt_mode == 'regression':
                # 【基础方向投影】回归模式
                # head 输出投影值 [-1, 1]，gt_probs 实际是投影值（不是概率）
                pred_proj = self.head(features)
                gt_proj = batch['gt_probs'].to(self.device)  # 复用 gt_probs 字段存储投影值
                losses = self.criterion(pred_proj, gt_proj, batch['category'])
            elif self.args.mode == 'discrete' and self.args.gt_mode == 'dr':
                # 【DR系列】投影 + Softmax 模式
                # head 输出投影值 [-1, 1]，loss中会对pred和gt都做softmax
                pred_proj = self.head(features)
                gt_proj = batch['gt_probs'].to(self.device)  # GT也是cos投影值
                losses = self.criterion(pred_proj, gt_proj, batch['category'])
            else:
                # 【槽分类】分类模式
                logits = self.head(features)
                gt_probs = batch['gt_probs'].to(self.device)
                losses = self.criterion(logits, gt_probs, batch['category'])

            loss = losses['total_loss']

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

            # 日志
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_errors = []

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            B = points.shape[0]

            features = self.encoder(points)

            if self.args.mode == 'mf':
                pred_mu, pred_kappa = self.head(features)
                gt_mu = batch['gt_mu'].to(self.device)
                gt_kappa = batch['gt_kappa'].to(self.device)
                gt_angle = batch['gt_angle'].to(self.device)

                losses = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa, batch['category'])
                error = compute_angle_error_mf(pred_mu, pred_kappa, gt_angle, batch['category'])
            elif self.args.mode == 'discrete' and self.args.gt_mode == 'regression':
                # 【基础方向投影】回归模式
                pred_proj = self.head(features)
                gt_proj = batch['gt_probs'].to(self.device)  # 复用 gt_probs 字段存储投影值
                gt_angle = batch['gt_angle'].to(self.device)

                losses = self.criterion(pred_proj, gt_proj, batch['category'])
                # 使用专用的投影回归角度误差计算函数
                error = compute_angle_error_regression(pred_proj, gt_angle, batch['category'], self.args.num_bins)
            elif self.args.mode == 'discrete' and self.args.gt_mode == 'dr':
                # 【DR系列】投影 + Softmax 模式
                pred_proj = self.head(features)
                gt_proj = batch['gt_probs'].to(self.device)
                gt_angle = batch['gt_angle'].to(self.device)

                losses = self.criterion(pred_proj, gt_proj, batch['category'])
                # 使用DR专用的角度误差计算函数（从softmax概率重建方向）
                error = compute_angle_error_dr(pred_proj, gt_angle, batch['category'], self.args.num_bins)
            else:
                # 【槽分类】分类模式
                logits = self.head(features)
                gt_probs = batch['gt_probs'].to(self.device)
                gt_angle = batch['gt_angle'].to(self.device)

                losses = self.criterion(logits, gt_probs, batch['category'])
                error = compute_angle_error_discrete(logits, gt_angle, batch['category'], self.args.num_bins)

            total_loss += losses['total_loss'].item() * B
            total_samples += B
            all_errors.append(error)

        avg_loss = total_loss / total_samples
        avg_error = np.mean(all_errors) if all_errors else 0.0

        return avg_loss, avg_error

    def train(self):
        print(f"\nStarting training: {self.args.exp_name}")
        print(f"Mode: {self.args.mode}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.start_epoch} -> {self.args.epochs}")
        print(f"="*60)

        # Append mode if resuming, write mode if starting fresh
        log_mode = 'a' if self.start_epoch > 1 else 'w'
        log_file = open(self.exp_dir / 'training_log.txt', log_mode)

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_error = self.validate(epoch)

            # 学习率调度
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # 日志
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_error={val_error:.2f}°, lr={lr:.6f}"
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()

            # WandB
            if self.args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/angle_error_deg': val_error,
                    'lr': lr,
                })

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_loss.pth', epoch, val_loss, val_error)

            if val_error < self.best_val_error:
                self.best_val_error = val_error
                self._save_checkpoint('best_error.pth', epoch, val_loss, val_error)

            # 每个epoch都保存latest.pth，用于暂停/恢复
            self._save_checkpoint('latest.pth', epoch, val_loss, val_error)

        log_file.write(f"\nBest val loss: {self.best_val_loss:.4f}\n")
        log_file.write(f"Best val error: {self.best_val_error:.2f}°\n")
        log_file.close()

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val error: {self.best_val_error:.2f}°")
        print(f"Checkpoints saved to: {self.exp_dir}")

        if self.args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    def _save_checkpoint(self, filename, epoch, val_loss, val_error):
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_error': val_error,
            'best_val_loss': self.best_val_loss,
            'best_val_error': self.best_val_error,
            'args': vars(self.args),
        }
        torch.save(checkpoint, self.exp_dir / filename)
        print(f"  Saved: {filename} (val_loss={val_loss:.4f}, val_error={val_error:.2f}°)")

    def _load_checkpoint(self, resume_dir):
        """Load checkpoint and resume training"""
        resume_path = Path(resume_dir)

        # Try to load latest.pth first, then best_loss.pth
        checkpoint_file = None
        for fname in ['latest.pth', 'best_loss.pth', 'best_error.pth']:
            if (resume_path / fname).exists():
                checkpoint_file = resume_path / fname
                break

        if checkpoint_file is None:
            print(f"[Warning] No checkpoint found in {resume_dir}, starting from scratch")
            return

        print(f"[Resume] Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        # Load model states
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.head.load_state_dict(checkpoint['head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
        self.best_val_error = checkpoint.get('best_val_error', checkpoint['val_error'])

        print(f"[Resume] Resuming from epoch {self.start_epoch}")
        print(f"[Resume] Best val_loss: {self.best_val_loss:.4f}, Best val_error: {self.best_val_error:.2f}°")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Direction Training')

    # 基础参数
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--mode', type=str, choices=['mf', 'discrete'], required=True,
                        help='Training mode: mf (mixture von Mises) or discrete')
    parser.add_argument('--categories', type=str, default='1_front,4_fronts',
                        help='Categories to train on (comma-separated)')

    # 数据参数
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--augment_factor', type=int, default=10, help='Data augmentation factor')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # MF模式参数
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['combined', 'reverse_kl', 'mu_only'],
                        help='Loss type for MF mode')
    parser.add_argument('--lambda_kl', type=float, default=1.0, help='KL loss weight')
    parser.add_argument('--lambda_kappa', type=float, default=5.0, help='Kappa loss weight')
    parser.add_argument('--lambda_mu', type=float, default=2.0, help='Mu loss weight')
    parser.add_argument('--gt_kappa', type=float, default=10.0, help='GT kappa value')

    # Discrete模式参数
    # ┌────────────────────────────────────────────────────────────────────────────┐
    # │ gt_mode 三种模式的区别：                                                   │
    # ├────────────────────────────────────────────────────────────────────────────┤
    # │ onehot:     【槽分类】将圆周分成N个槽，输出one-hot概率                     │
    # │ projection: 【软槽分类】同上，但用cos投影+softmax生成soft label           │
    # │ regression: 【基础方向投影】输出方向在N个基础方向上的投影值（回归问题）   │
    # └────────────────────────────────────────────────────────────────────────────┘
    parser.add_argument('--num_bins', type=int, default=8,
                        help='Number of direction bins/基础方向数量')
    parser.add_argument('--d_loss_type', type=str, default='ce',
                        choices=['ce', 'focal', 'kl'],
                        help='Loss type for discrete classification mode (onehot/projection)')
    parser.add_argument('--gt_mode', type=str, default='projection',
                        choices=['projection', 'onehot', 'regression', 'dr'],
                        help='GT mode: projection/onehot (D槽分类), regression (投影回归), dr (DR投影+softmax)')
    parser.add_argument('--temperature', type=float, default=5.0,
                        help='Temperature for projection softmax (higher = sharper, only for projection mode)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (only for onehot mode)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (only for classification mode)')

    # Regression模式参数 (gt_mode='regression' 时使用)
    # ┌────────────────────────────────────────────────────────────────────────────┐
    # │ 投影回归与槽分类的本质区别：                                               │
    # │   槽分类: Softmax → 概率分布(和=1) → CrossEntropy                         │
    # │   投影回归: Tanh → 投影值[-1,1] → MSE/Cosine Loss                         │
    # └────────────────────────────────────────────────────────────────────────────┘
    parser.add_argument('--reg_loss_type', type=str, default='cosine',
                        choices=['mse', 'cosine', 'combined'],
                        help='Loss type for regression mode: mse, cosine, or combined')
    parser.add_argument('--lambda_mse', type=float, default=1.0,
                        help='MSE loss weight (for regression combined mode)')
    parser.add_argument('--lambda_cos', type=float, default=1.0,
                        help='Cosine loss weight (for regression combined mode)')

    # WandB
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation',
                        help='WandB project name')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume from (e.g., checkpoints/DR_8a_20251219_144931)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

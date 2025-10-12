# models/pointnet_pp_mvm2d.py
# -*- coding: utf-8 -*-
"""
PointNet++ → Mixture of von Mises (2D yaw) 预测网络（修正版，带 mu 输出保护）
无 upright 条件输入，所有 head 输入 256 维。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_pp_8dir import PointNetSetAbstraction


def _maybe_transpose_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """确保输入为 (B,3,N)。若是 (B,N,3) 就转置。"""
    assert xyz.dim() == 3, f"xyz should be 3D tensor, got {xyz.shape}"
    B, A, C = xyz.shape
    if C == 3:
        # 输入形状 (B, N, 3)，transpose 成 (B, 3, N)
        xyz = xyz.transpose(1, 2).contiguous()
    elif A == 3:
        # 已经是 (B, 3, N)
        pass
    else:
        raise ValueError(f"xyz must be (B,N,3) or (B,3,N), got {xyz.shape}")
    return xyz


class PointNetPPMvM(nn.Module):
    """
    PointNet++ backbone + 混合 von Mises 参数头（2D 平面角度）
    输出:
        mu     (B,K)   — 角度 ∈ [-π, π]
        kappa  (B,K)   — ≥ 0
        weight (B,K)   — 混合分量权重，sum = 1
    """

    def __init__(
        self,
        max_K: int = 4,
        kappa_max: float = 80.0,
        p_drop: float = 0.4,
        temp: float = 0.7,
    ):
        super().__init__()
        self.max_K = max_K
        self.kappa_max = float(kappa_max)
        self.temp = float(temp)

        # ---------- PointNet++ Backbone ----------
        self.sa1 = PointNetSetAbstraction(128, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

        # ---------- 全局特征层 ----------
        self.fc1 = nn.Linear(1024, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.drop = nn.Dropout(p_drop)

        hidden = 256
        self.head_pi = nn.Linear(hidden, max_K)
        self.head_mu = nn.Linear(hidden, max_K * 2)
        self.head_kappa = nn.Linear(hidden, max_K)

        # **注意**：这里 bias 和 weight 的初始化可以微调
        nn.init.zeros_(self.head_pi.weight)
        nn.init.zeros_(self.head_pi.bias)
        nn.init.zeros_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.constant_(self.head_kappa.bias, 0.0)

    def _global_feat(self, xyz: torch.Tensor) -> torch.Tensor:
        B = xyz.size(0)
        xyz_bn3 = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        l1_xyz, l1_pts = self.sa1(xyz_bn3, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)  # (B, 1, 1024)
        x = l3_pts.view(B, -1)
        x = self.drop(F.relu(self.ln1(self.fc1(x))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        return x  # (B, 256)

    def forward(self, xyz: torch.Tensor):
        xyz = _maybe_transpose_xyz(xyz)
        feat = self._global_feat(xyz)  # (B, hidden)

        # Mixture 权重头
        logit_pi = self.head_pi(feat) / self.temp
        weight = F.softmax(logit_pi, dim=-1)  # (B, K)

        # mu_raw 输出
        mu_raw = self.head_mu(feat).view(-1, self.max_K, 2)  # (B, K, 2)

        # 检查 mu_raw 是否异常（调试用）
        if not torch.isfinite(mu_raw).all():
            print("mu_raw non-finite:", mu_raw)

        # 归一化为单位向量（带 eps 防止除以 0）
        mu_unit = F.normalize(mu_raw, dim=-1, eps=1e-4)
        c = mu_unit[..., 0]
        s = mu_unit[..., 1]

        # 计算 norm，用于检测是否退化为零向量
        norm = torch.sqrt(c * c + s * s)
        if (norm < 1e-3).any():
            # 若某些分量太小，做 fallback 处理，避免 c=s=0
            # 这里将那些分量设定为默认方向 (c=1, s=0) → mu = 0
            mask = (norm < 1e-3)
            c = torch.where(mask, torch.ones_like(c), c)
            s = torch.where(mask, torch.zeros_like(s), s)

        mu = torch.atan2(s, c)  # (B, K)

        # 检查 mu 是否为 nan/inf（调试用）
        if not torch.isfinite(mu).all():
            print("mu non-finite:", mu, "from c,s:", c, s)

        # kappa 输出
        kappa_raw = self.head_kappa(feat)
        kappa = F.softplus(kappa_raw) + 1e-6
        if self.kappa_max is not None:
            kappa = kappa.clamp_max(self.kappa_max)

        return mu, kappa, weight


@torch.no_grad()
def mvm_density_on_grid(mu, kappa, weight, num=360, device=None):
    """在 [0, 2π) 上采样 num 个角度，返回混合密度。"""
    B, K = mu.shape
    device = device or mu.device
    theta = torch.linspace(0.0, 2 * math.pi, steps=num, device=device, dtype=mu.dtype)
    theta = theta[:-1]  # [0,2π)
    theta = theta[None, None, :]  # (1, 1, num)
    mu = mu[..., None]
    kappa = kappa[..., None]
    w = weight[..., None]
    vm = torch.exp(kappa * torch.cos(theta - mu)) / (2 * math.pi * torch.i0(kappa))
    p = (w * vm).sum(dim=1)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    return theta.squeeze(), p

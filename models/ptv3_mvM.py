# models/ptv3_mvM.py
# -*- coding: utf-8 -*-
"""
Point Transformer V3 → Mixture of von Mises (2D yaw) 预测网络

与 pointnet_pp_mvM.py 和 dgcnn_mvM.py 结构对应，使用 PTv3 作为 backbone
用于性能对比实验。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .point_transformer_v3 import PointTransformerV3, PointTransformerV3Lite


def _maybe_transpose_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """确保输入为 (B,N,3)。若是 (B,3,N) 就转置。"""
    assert xyz.dim() == 3, f"xyz should be 3D tensor, got {xyz.shape}"
    B, A, C = xyz.shape
    if A == 3 and C != 3:
        xyz = xyz.transpose(1, 2).contiguous()  # (B,3,N) -> (B,N,3)
    return xyz


class PTv3MvM(nn.Module):
    """
    Point Transformer V3 backbone + 混合 von Mises 参数头（2D 平面角度）

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
        temp: float = 2.0,
        # PTv3 specific params
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        window_size: int = 64,
        use_lite: bool = True,  # Use lite version for small point clouds
    ):
        super().__init__()
        self.max_K = max_K
        self.kappa_max = float(kappa_max)
        self.temp = float(temp)

        # ---------- PTv3 Backbone ----------
        if use_lite:
            self.backbone = PointTransformerV3Lite(
                in_channels=3,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
                window_size=window_size,
                drop_rate=p_drop,
            )
            backbone_out_dim = embed_dim
        else:
            self.backbone = PointTransformerV3(
                in_channels=3,
                embed_dims=[64, 128, 256, 512],
                num_heads=[2, 4, 8, 16],
                depths=[2, 2, 2, 2],
                window_size=window_size,
                drop_rate=p_drop,
            )
            backbone_out_dim = 512

        # ---------- 全局特征层 ----------
        self.fc1 = nn.Linear(backbone_out_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.drop = nn.Dropout(p_drop)

        hidden = 256
        self.head_pi = nn.Linear(hidden, max_K)
        self.head_mu = nn.Linear(hidden, max_K * 2)
        self.head_kappa = nn.Linear(hidden, max_K)

        # head_pi保持zeros初始化（对softmax是合理的）
        nn.init.zeros_(self.head_pi.weight)
        nn.init.zeros_(self.head_pi.bias)

        # head_mu: 预设4个方向初始化，打破对称性
        # 预设角度：0°, 90°, 180°, 270°
        initial_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

        nn.init.zeros_(self.head_mu.weight)

        with torch.no_grad():
            for i, angle in enumerate(initial_angles):
                if i < max_K:
                    self.head_mu.bias[2 * i] = math.cos(angle)
                    self.head_mu.bias[2 * i + 1] = math.sin(angle)

        nn.init.constant_(self.head_kappa.bias, 0.0)

    def _global_feat(self, xyz: torch.Tensor) -> torch.Tensor:
        """Extract global features using PTv3 backbone."""
        feat = self.backbone(xyz)  # (B, backbone_out_dim)
        x = self.drop(F.relu(self.ln1(self.fc1(feat))))
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
    theta = theta[:-1]
    theta = theta[None, None, :]
    mu = mu[..., None]
    kappa = kappa[..., None]
    w = weight[..., None]
    vm = torch.exp(kappa * torch.cos(theta - mu)) / (2 * math.pi * torch.i0(kappa))
    p = (w * vm).sum(dim=1)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    return theta.squeeze(), p

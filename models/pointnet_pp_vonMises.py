# models/pointnet_pp_vonMises.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_pp_8dir import PointNetSetAbstraction


class PointNetPPVonMises(nn.Module):
    """
    PointNet++ backbone + von Mises parameter head
    输出 (μ, κ)

    使用 2D 单位向量 + atan2 方法处理角度的周期性问题：
    - 网络输出 (cos θ, sin θ) 形式的 2D 向量
    - 归一化后用 atan2 恢复角度
    - 避免了 tanh 在边界处的梯度消失问题
    """
    def __init__(self, kappa_max: float = 80.0):
        super().__init__()
        self.kappa_max = kappa_max

        # PointNet++ backbone
        self.sa1 = PointNetSetAbstraction(128, 32,   0, [ 64,  64, 128])
        self.sa2 = PointNetSetAbstraction( 32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)

        # 分离的 head: mu 输出 2D 向量，kappa 输出标量
        self.head_mu = nn.Linear(256, 2)     # 输出 (cos, sin)
        self.head_kappa = nn.Linear(256, 1)  # 输出 κ

        # 初始化 head_mu 的 bias 为 (1, 0)，对应 mu=0
        nn.init.zeros_(self.head_mu.weight)
        with torch.no_grad():
            self.head_mu.bias[0] = 1.0  # cos(0) = 1
            self.head_mu.bias[1] = 0.0  # sin(0) = 0

        nn.init.constant_(self.head_kappa.bias, 0.0)

    def forward(self, xyz):
        B = xyz.size(0)
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)

        # mu: 2D 单位向量 → atan2 → 角度
        mu_raw = self.head_mu(x)  # (B, 2)
        mu_unit = F.normalize(mu_raw, dim=-1, eps=1e-4)
        c = mu_unit[:, 0]  # cos(θ)
        s = mu_unit[:, 1]  # sin(θ)

        # 防止零向量导致 NaN
        norm = torch.sqrt(c * c + s * s)
        mask = (norm < 1e-3)
        if mask.any():
            c = torch.where(mask, torch.ones_like(c), c)
            s = torch.where(mask, torch.zeros_like(s), s)

        mu = torch.atan2(s, c)  # (B,) 范围 [-π, π]

        # kappa: softplus 保证非负 + 上限裁剪
        kappa_raw = self.head_kappa(x).squeeze(-1)  # (B,)
        kappa = F.softplus(kappa_raw) + 1e-6
        if self.kappa_max is not None:
            kappa = kappa.clamp_max(self.kappa_max)

        return mu, kappa

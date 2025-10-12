# models/pointnet_pp_vonMises.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_pp_8dir import PointNetSetAbstraction

class PointNetPPVonMises(nn.Module):
    """
    PointNet++ backbone + von Mises parameter head
    输出 (μ, κ)
    """
    def __init__(self):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(128, 32,   0, [ 64,  64, 128])
        self.sa2 = PointNetSetAbstraction( 32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None,256,[256, 512,1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop= nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)  # 输出 μ, κ

    def forward(self, xyz):
        B = xyz.size(0)
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _,    l3_pts  = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        out = self.fc3(x)
        mu    = torch.tanh(out[:, 0]) * np.pi     # 限制 μ∈[-π, π]
        kappa = F.softplus(out[:, 1])             # κ≥0
        return mu, kappa

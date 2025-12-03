#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对称性分类网络 (Symmetry Classifier)

包含两个版本用于消融实验：
1. SymmetryClassifier_GlobalConcat: upright vector在全局特征后concat
2. SymmetryClassifier_PointConcat: upright vector复制到每个点

基于PointNet++架构

作者: Claude
创建日期: 2025-11-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import index_points, query_ball_point


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction层"""
    def __init__(self, npoint, nsample, in_channel, mlp_channels, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all

        last_ch = in_channel + 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in mlp_channels:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)
            new_points = grouped_xyz if points is None else torch.cat([grouped_xyz, points.unsqueeze(1)], -1)
        else:
            fps_idx = torch.stack([torch.randperm(N)[:self.npoint] for _ in range(B)]).to(xyz.device)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(new_xyz, xyz, self.nsample)
            grouped_xyz = index_points(xyz, idx)
            normed = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                new_points = torch.cat([normed, grouped_pts], -1)
            else:
                new_points = normed

        x = new_points.permute(0, 3, 1, 2)  # (B, 3+D, npoint, nsample)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, 3)[0]  # (B, mlp[-1], npoint)
        return new_xyz, x.permute(0, 2, 1)


class SymmetryClassifier_GlobalConcat(nn.Module):
    """
    对称性分类器 - 版本1（Ablation Exp 1.1）

    架构：
        PointNet++ Encoder → global_feat (1024)
        Concat upright_vec (3) → (1027)
        MLP → 5 classes

    输入：
        points: (B, N, 3) 点云
        upright_vec: (B, 3) 竖直方向向量

    输出：
        logits: (B, 5) 5个类别的logits
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

        # Classifier MLP (1024 + 3 → num_classes)
        self.fc1 = nn.Linear(1024 + 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, points, upright_vec):
        """
        Args:
            points: (B, N, 3) 点云
            upright_vec: (B, 3) 竖直方向向量

        Returns:
            logits: (B, num_classes)
        """
        B = points.size(0)

        # PointNet++ Encoder
        l1_xyz, l1_pts = self.sa1(points, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)

        # Global feature
        global_feat = l3_pts.view(B, -1)  # (B, 1024)

        # Concat upright vector
        feat = torch.cat([global_feat, upright_vec], dim=1)  # (B, 1027)

        # Classifier MLP
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        logits = self.fc4(x)  # (B, num_classes)

        return logits


class SymmetryClassifier_PointConcat(nn.Module):
    """
    对称性分类器 - 版本2（Ablation Exp 1.2）

    架构：
        Input: (N, 3+3=6) [xyz + upright_vec复制N次]
        PointNet++ Encoder → global_feat (1024)
        MLP → 5 classes

    输入：
        points: (B, N, 3) 点云
        upright_vec: (B, 3) 竖直方向向量

    输出：
        logits: (B, 5) 5个类别的logits
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # PointNet++ Encoder (输入6维：xyz + upright_vec)
        # 注意：in_channel是除xyz之外的额外特征维度
        self.sa1 = PointNetSetAbstraction(512, 32, 3, [64, 64, 128])  # in_channel=3 (upright_vec)
        self.sa2 = PointNetSetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

        # Classifier MLP (1024 → num_classes)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, points, upright_vec):
        """
        Args:
            points: (B, N, 3) 点云
            upright_vec: (B, 3) 竖直方向向量

        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = points.shape

        # 复制upright_vec到每个点
        upright_expanded = upright_vec.unsqueeze(1).expand(B, N, 3)  # (B, N, 3)

        # Concat到每个点的特征
        # 注意：这里我们不直接concat，而是把upright_vec作为points传入
        # PointNetSetAbstraction会自动处理

        # PointNet++ Encoder
        l1_xyz, l1_pts = self.sa1(points, upright_expanded)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)

        # Global feature
        global_feat = l3_pts.view(B, -1)  # (B, 1024)

        # Classifier MLP
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        logits = self.fc4(x)  # (B, num_classes)

        return logits


class SymmetryClassifier_NoUpright(nn.Module):
    """
    对称性分类器 - Baseline版本（Ablation Exp 1.3）

    架构：
        PointNet++ Encoder → global_feat (1024)
        MLP → 5 classes
        (不使用upright vector)

    输入：
        points: (B, N, 3) 点云

    输出：
        logits: (B, 5) 5个类别的logits
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

        # Classifier MLP (1024 → num_classes)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, points, upright_vec=None):
        """
        Args:
            points: (B, N, 3) 点云
            upright_vec: (B, 3) 竖直方向向量（忽略，为了接口兼容）

        Returns:
            logits: (B, num_classes)
        """
        B = points.size(0)

        # PointNet++ Encoder
        l1_xyz, l1_pts = self.sa1(points, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)

        # Global feature
        global_feat = l3_pts.view(B, -1)  # (B, 1024)

        # Classifier MLP
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        logits = self.fc4(x)  # (B, num_classes)

        return logits


if __name__ == '__main__':
    # 测试网络
    print("=" * 80)
    print("测试对称性分类网络")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建测试数据
    B = 4
    N = 10000
    points = torch.randn(B, N, 3).to(device)
    upright_vec = torch.tensor([[0, 1, 0]] * B, dtype=torch.float32).to(device)

    # 测试三个模型
    models = {
        'GlobalConcat': SymmetryClassifier_GlobalConcat(),
        'PointConcat': SymmetryClassifier_PointConcat(),
        'NoUpright': SymmetryClassifier_NoUpright()
    }

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        print(f"\n{'='*80}")
        print(f"测试 {name}")
        print(f"{'='*80}")

        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {num_params:,}")

        # Forward pass
        with torch.no_grad():
            logits = model(points, upright_vec)
            print(f"输入 points shape: {points.shape}")
            print(f"输入 upright_vec shape: {upright_vec.shape}")
            print(f"输出 logits shape: {logits.shape}")
            print(f"输出 logits: {logits[0]}")

            # 预测类别
            preds = torch.argmax(logits, dim=1)
            print(f"预测类别: {preds}")

    print(f"\n{'='*80}")
    print("✅ 网络测试完成！")
    print(f"{'='*80}")

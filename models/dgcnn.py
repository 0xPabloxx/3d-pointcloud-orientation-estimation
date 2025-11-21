# models/dgcnn.py
# -*- coding: utf-8 -*-
"""
DGCNN (Dynamic Graph CNN) Backbone for Point Cloud Processing

Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM TOG 2019
Implementation based on: https://github.com/antao97/dgcnn.pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Compute k nearest neighbors for each point.

    Args:
        x: (B, C, N) point features
        k: number of neighbors

    Returns:
        idx: (B, N, k) indices of k nearest neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for EdgeConv.

    Args:
        x: (B, C, N) point features
        k: number of neighbors
        idx: precomputed knn indices

    Returns:
        feature: (B, 2*C, N, k) edge features [x_j - x_i, x_i]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (B, 2*C, N, k)


class DGCNN(nn.Module):
    """
    DGCNN backbone that outputs global features.

    This is designed to match the output dimension of PointNet++ backbone (1024-dim)
    for easy comparison and interchangeability.
    """

    def __init__(self, k=20, emb_dims=1024, dropout=0.5):
        """
        Args:
            k: number of nearest neighbors for EdgeConv
            emb_dims: embedding dimension (default 1024 to match PointNet++)
            dropout: dropout rate
        """
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        # EdgeConv layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) or (B, N, 3) point cloud

        Returns:
            feat: (B, emb_dims*2) global features (max + avg pooling)
        """
        batch_size = x.size(0)

        # Handle input shape
        if x.size(1) != 3:
            x = x.transpose(1, 2).contiguous()  # (B, 3, N)

        # EdgeConv 1
        x = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x = self.conv1(x)  # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # EdgeConv 2
        x = get_graph_feature(x1, k=self.k)  # (B, 128, N, k)
        x = self.conv2(x)  # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # EdgeConv 3
        x = get_graph_feature(x2, k=self.k)  # (B, 128, N, k)
        x = self.conv3(x)  # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        # EdgeConv 4
        x = get_graph_feature(x3, k=self.k)  # (B, 256, N, k)
        x = self.conv4(x)  # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        # Concatenate all EdgeConv outputs
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        # Final conv
        x = self.conv5(x)  # (B, emb_dims, N)

        # Global pooling (max + avg)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, emb_dims)
        x = torch.cat((x1, x2), 1)  # (B, emb_dims*2)

        return x


class DGCNNCls(nn.Module):
    """
    DGCNN for classification tasks.
    """

    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40):
        super().__init__()
        self.backbone = DGCNN(k=k, emb_dims=emb_dims, dropout=dropout)

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = self.backbone(x)  # (B, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

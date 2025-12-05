"""
Backbone factory for point cloud feature extraction.

Supported backbones:
- PointNet++: Hierarchical point set learning
- DGCNN: Dynamic graph CNN
- PointTransformerV3: Transformer-based (if available)

All backbones output a fixed-size global feature vector.

Author: Claude
Created: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add models directory to path
models_dir = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_dir))

from models.base import index_points, query_ball_point


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer."""

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


class PointNetPPBackbone(nn.Module):
    """
    PointNet++ backbone that outputs global features.

    Output dimension: 1024
    """

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim

        self.sa1 = PointNetSetAbstraction(128, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, 1024], group_all=True)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud

        Returns:
            features: (B, output_dim) global features
        """
        B = x.size(0)

        # Handle input shape: expect (B, N, 3)
        if x.size(-1) != 3:
            x = x.transpose(1, 2).contiguous()

        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)

        features = l3_pts.view(B, -1)  # (B, 1024)
        return features


class DGCNNBackbone(nn.Module):
    """
    DGCNN backbone that outputs global features.

    Output dimension: emb_dims * 2 (default 2048, using max + avg pooling)
    """

    def __init__(self, k: int = 20, emb_dims: int = 1024):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.output_dim = emb_dims * 2

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

    def _knn(self, x, k):
        """Compute k nearest neighbors."""
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def _get_graph_feature(self, x, k=20, idx=None):
        """Construct edge features for EdgeConv."""
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        if idx is None:
            idx = self._knn(x, k=k)

        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) or (B, 3, N) point cloud

        Returns:
            features: (B, output_dim) global features
        """
        batch_size = x.size(0)

        # Handle input shape: need (B, 3, N)
        if x.size(1) != 3:
            x = x.transpose(1, 2).contiguous()

        # EdgeConv layers
        x = self._get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self._get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self._get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self._get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Global pooling (max + avg)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        return x


def get_backbone(backbone_config: dict) -> nn.Module:
    """
    Factory function to create backbone based on config.

    Args:
        backbone_config: dict with keys:
            - type: 'pointnet++', 'dgcnn', 'pointtransformer'
            - Additional backbone-specific parameters

    Returns:
        Backbone module with .output_dim attribute
    """
    backbone_type = backbone_config.get('type', 'pointnet++').lower()

    if backbone_type in ['pointnet++', 'pointnetpp', 'pointnet2']:
        return PointNetPPBackbone(
            output_dim=backbone_config.get('output_dim', 1024)
        )
    elif backbone_type == 'dgcnn':
        return DGCNNBackbone(
            k=backbone_config.get('k', 20),
            emb_dims=backbone_config.get('emb_dims', 1024)
        )
    elif backbone_type in ['pointtransformer', 'ptv3']:
        # TODO: Add Point Transformer V3 support
        raise NotImplementedError("Point Transformer V3 backbone not yet integrated")
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


# Quick test
if __name__ == '__main__':
    # Test PointNet++
    print("Testing PointNet++ backbone...")
    pn_backbone = get_backbone({'type': 'pointnet++'})
    x = torch.randn(2, 1024, 3)
    out = pn_backbone(x)
    print(f"  Input: {x.shape}, Output: {out.shape}, output_dim: {pn_backbone.output_dim}")

    # Test DGCNN
    print("Testing DGCNN backbone...")
    dg_backbone = get_backbone({'type': 'dgcnn', 'k': 20})
    out = dg_backbone(x)
    print(f"  Input: {x.shape}, Output: {out.shape}, output_dim: {dg_backbone.output_dim}")

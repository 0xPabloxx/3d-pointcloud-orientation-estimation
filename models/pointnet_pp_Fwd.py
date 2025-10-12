# models/pointnet_pp_8dir_sampled.py
import torch, torch.nn as nn, torch.nn.functional as F
from .base import index_points, query_ball_point              # 详见原仓库

# ---------- Set-Abstraction ----------
class PointNetSetAbstraction(nn.Module):
    """
    与 PointNet++ 论文/官方实现一致的 SA Layer
    npoint     : 采样中心点数 (None 时整体 Pool)
    nsample    : 每个中心点查询的近邻点数
    in_channel : 输入特征维度 (XYZ 之外)
    mlp_channels : 多层 MLP 输出通道列表
    group_all  : 是否整体汇聚
    """
    def __init__(self, npoint, nsample, in_channel,
                 mlp_channels, group_all=False):
        super().__init__()
        self.npoint    = npoint
        self.nsample   = nsample
        self.group_all = group_all

        last_ch = in_channel + 3                          # +XYZ
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for ch in mlp_channels:
            self.convs.append(nn.Conv2d(last_ch, ch, 1))
            self.bns  .append(nn.BatchNorm2d(ch))
            last_ch = ch                                  # 链式更新

    def forward(self, xyz, points):
        """
        xyz   : (B,N,3)  原始坐标
        points: (B,N,D)  额外特征 or None
        return: new_xyz (B,S,3), new_points (B,S,C_out)
        """
        B, N, _ = xyz.size()

        if self.group_all:
            new_xyz      = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz  = xyz.unsqueeze(1)               # (B,1,N,3)
            new_points   = grouped_xyz if points is None \
                           else torch.cat([grouped_xyz, points.unsqueeze(1)], -1)
        else:
            # ① FPS 采样中心点
            fps_idx  = torch.stack(
                [torch.randperm(N, device=xyz.device)[:self.npoint]
                 for _ in range(B)]
            )                                            # (B,S)
            new_xyz  = index_points(xyz, fps_idx)        # (B,S,3)

            # ② ball-query / kNN 获取邻域
            idx       = query_ball_point(new_xyz, xyz, self.nsample)  # (B,S,K)
            grouped_xyz  = index_points(xyz, idx)                     # (B,S,K,3)
            normed_xyz   = grouped_xyz - new_xyz.unsqueeze(2)         # 局部归一化

            if points is not None:
                grouped_pts = index_points(points, idx)              # (B,S,K,D)
                new_points  = torch.cat([normed_xyz, grouped_pts], -1)
            else:
                new_points  = normed_xyz                              # 只用坐标

        # ③ MLP + max-pool
        x = new_points.permute(0, 3, 1, 2)            # (B,3+D,S,K)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, 3)[0]                        # (B,C_out,S)
        return new_xyz, x.permute(0, 2, 1)            # (B,S,C_out)

# ---------- 8 个水平基向量 ----------
DIRS_8 = torch.tensor([
    [ 0.0000, 0.0, -1.0000], [ 0.7071, 0.0, -0.7071],
    [ 1.0000, 0.0,  0.0000], [ 0.7071, 0.0,  0.7071],
    [ 0.0000, 0.0,  1.0000], [-0.7071, 0.0,  0.7071],
    [-1.0000, 0.0,  0.0000], [-0.7071, 0.0, -0.7071],
])                                   # (8,3)  :contentReference[oaicite:1]{index=1}

# ---------- PointNet++ 主干 + 3-D 向量头 ----------
class PointNetPPFwd(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(128, 32,   0, [ 64,  64, 128])
        self.sa2 = PointNetSetAbstraction( 32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None,256,[256, 512,1024],
                                          group_all=True)
        self.fc1 = nn.Linear(1024, 512);  self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256);   self.bn2 = nn.BatchNorm1d(256)
        self.drop= nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 3)      # 输出 3-D 向量

    def forward(self, xyz):               # xyz:(B,N,3)
        B = xyz.size(0)
        l1_xyz, l1 = self.sa1(xyz, None)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        _,       g = self.sa3(l2_xyz, l2)           # (B,1,1024)
        x = g.view(B, -1)                           # (B,1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        return F.normalize(self.fc3(x), dim=1)      # 单位化 (B,3)

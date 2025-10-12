import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import index_points, query_ball_point

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp_channels, group_all=False):
        super().__init__()
        self.npoint   = npoint
        self.nsample  = nsample
        self.group_all= group_all

        last_ch = in_channel + 3
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for out_ch in mlp_channels:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B,1,3,device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)
            new_points = grouped_xyz if points is None else torch.cat([grouped_xyz, points.unsqueeze(1)],-1)
        else:
            fps_idx = torch.stack([torch.randperm(N)[:self.npoint] for _ in range(B)]).to(xyz.device)
            new_xyz = index_points(xyz, fps_idx)
            idx     = query_ball_point(new_xyz, xyz, self.nsample)
            grouped_xyz = index_points(xyz, idx)
            normed = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                new_points = torch.cat([normed, grouped_pts], -1)
            else:
                new_points = normed

        x = new_points.permute(0,3,1,2)  # (B,3+D,npoint,nsample)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, 3)[0]          # (B,mlp[-1],npoint)
        return new_xyz, x.permute(0,2,1)

class PointNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(128, 32,   0, [ 64,  64, 128])
        self.sa2 = PointNetSetAbstraction( 32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None,None,256, [256, 512,1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop= nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        B = x.size(0)
        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _,    l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        return self.fc3(x)

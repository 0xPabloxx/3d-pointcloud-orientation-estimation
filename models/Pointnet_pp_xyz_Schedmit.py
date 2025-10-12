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


class PointNetPPXYZ_Schedmit(nn.Module):
    def __init__(self):
        super().__init__()
        # 保留原有的 Set Abstraction 层
        self.sa1 = PointNetSetAbstraction(128, 32,   0, [ 64,  64, 128])
        self.sa2 = PointNetSetAbstraction( 32, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None,None,256, [256, 512,1024], group_all=True)

        # 全连接特征提取层
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop= nn.Dropout(0.5)

        # 删除旧的 fc3 = nn.Linear(256, 3)
        # 新增三个回归头，分别预测原始 X、Y、Z 轴旋转后方向
        #self.head_x = nn.Linear(256, 3)
        self.head_y = nn.Linear(256, 3)
        self.head_z = nn.Linear(256, 3)

    def forward(self, x):
        B = x.size(0)
        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _,    l3_pts = self.sa3(l2_xyz, l2_pts)
        feat = l3_pts.view(B, -1)

        feat = F.relu(self.bn1(self.fc1(feat)))
        feat = F.relu(self.bn2(self.fc2(feat)))
        feat = self.drop(feat)

        # # 直接回归，不做 L2 归一化
        # v1 = self.head_x(feat)  # (B,3)
        # v2 = self.head_y(feat)
        # v3 = self.head_z(feat)

        # 分别回归三个轴向，并做 L2 归一化保证单位向量
        #v2 v3 shape[16,3]
        # v1 = F.normalize(self.head_x(feat), p=2, dim=1)  # 预测 X 轴方向
        v2 = F.normalize(self.head_y(feat), p=2, dim=1)  # 预测 Y 轴方向
        v3 = F.normalize(self.head_z(feat), p=2, dim=1)  # 预测 Z 轴方向

        #print(v2.shape,v3.shape)#

        return  v2, v3
        #z forward
        #y upright
        # 2) 第一基向量 e3 = v3
        # e3 = v3  # 保证与输入 v3 保持一致 :contentReference[oaicite:5]{index=5}
        #
        # # 3) 计算第二基向量 e2
        # proj_v2_on_e3 = (v2 * e3).sum(dim=1, keepdim=True) * e3  # 投影 :contentReference[oaicite:6]{index=6}
        # u2 = v2 - proj_v2_on_e3
        # e2 = F.normalize(u2, p=2, dim=1)  # 归一化残差 :contentReference[oaicite:7]{index=7}
        #
        # # 4) 计算第三基向量 e1
        # proj_v1_on_e3 = (v1 * e3).sum(dim=1, keepdim=True) * e3
        # proj_v1_on_e2 = (v1 * e2).sum(dim=1, keepdim=True) * e2
        # u1 = v1 - proj_v1_on_e3 - proj_v1_on_e2
        # e1 = F.normalize(u1, p=2, dim=1)  # 最后归一化 :contentReference[oaicite:8]{index=8}
        #
        # # 5) 按顺序返回 (e1,e2,e3)
        # return e1, e2, e3


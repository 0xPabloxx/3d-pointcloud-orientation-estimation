import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------ 辅助函数 ------------------

def farthest_point_sample(xyz, npoint):
    """
    输入:
        xyz: 点云坐标 [B, N, 3]
        npoint: 采样点数
    输出:
        centroids: [B, npoint] 每个batch采样的点索引
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    输入:
        points: [B, N, C]
        idx: 采样索引, shape: [B, npoint] 或 [B, npoint, nsample]
    输出:
        new_points: 采样后的点, 形状与 idx 对应
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    输入:
        radius: 查询半径
        nsample: 每个局部区域最大采样点数
        xyz: 全部点云坐标 [B, N, 3]
        new_xyz: 中心点坐标 [B, npoint, 3]
    输出:
        group_idx: [B, npoint, nsample] 邻域点索引
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    # 计算新中心点与所有点的距离
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, -1)
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N  # 超出半径的点标记为非法索引 N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# ------------------ Set Abstraction 模块 ------------------

class SimpleSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        参数:
            npoint: 采样中心点数（group_all 为 False 时使用）
            radius: 分组查询半径
            nsample: 每个局部区域采样点数
            in_channel: 输入特征维数（不含坐标）
            mlp: list，每层输出通道数，第一层输入维度为 in_channel+3
        """
        super(SimpleSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channel + 3
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        输入:
            xyz: [B, 3, N] 原始坐标
            points: [B, D, N] 其他特征（可为 None）
        输出:
            new_xyz: [B, 3, npoint] 采样中心点坐标
            new_points: [B, mlp[-1], npoint] 聚合后的局部特征
        """
        B, _, N = xyz.shape
        # 转换为 [B, N, 3]
        xyz = xyz.transpose(2, 1).contiguous()
        if points is not None:
            points = points.transpose(2, 1).contiguous()  # [B, N, D]
        # 使用 FPS 采样中心点
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)               # [B, npoint, 3]
        # 查询邻域内的点
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # [B, npoint, nsample]
        grouped_xyz = index_points(xyz, group_idx)  # [B, npoint, nsample, 3]
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # 归一化为相对坐标
        if points is not None:
            grouped_points = index_points(points, group_idx)  # [B, npoint, nsample, D]
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
        else:
            new_points = grouped_xyz
        # 转换为 [B, (3+D), nsample, npoint] 以便进行1x1卷积（MLP）
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        # 在局部区域内 max pooling，得到 [B, mlp[-1], npoint]
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.transpose(2, 1).contiguous()  # 转为 [B, 3, npoint]
        return new_xyz, new_points

class SimpleSetAbstractionGroupAll(nn.Module):
    def __init__(self, in_channel, mlp):
        """
        全局聚合模块：对所有点进行分组（group_all=True）
        参数:
            in_channel: 输入特征维数（不含坐标）
            mlp: list，每层输出通道数，第一层输入维度为 in_channel+3
        """
        super(SimpleSetAbstractionGroupAll, self).__init__()
        last_channel = in_channel + 3
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        输入:
            xyz: [B, 3, N]
            points: [B, D, N]
        输出:
            new_xyz: [B, 3, 1] 全局中心（通常为0向量）
            new_points: [B, mlp[-1], 1] 全局特征
        """
        B, _, N = xyz.shape
        # 将所有点组合在一起
        xyz = xyz.transpose(2, 1).contiguous()  # [B, N, 3]
        if points is not None:
            points = points.transpose(2, 1).contiguous()  # [B, N, D]
            new_points = torch.cat([xyz, points], dim=-1)   # [B, N, 3+D]
        else:
            new_points = xyz  # [B, N, 3]
        new_points = new_points.unsqueeze(2)  # [B, N, 1, 3+D]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # [B, 3+D, 1, N]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 3, keepdim=True)[0]  # [B, mlp[-1], 1, 1]
        new_points = new_points.squeeze(3)  # [B, mlp[-1], 1]
        # 全局中心点取平均，也可以取0
        new_xyz = torch.mean(xyz, dim=1, keepdim=True).transpose(2,1).contiguous()  # [B, 3, 1]
        return new_xyz, new_points

# ------------------ PointNet++ 分类模型 ------------------

class PointNetPlusPlusCls(nn.Module):
    def __init__(self, num_classes=40, normal_channel=True):
        """
        参数:
            num_classes: 分类类别数
            normal_channel: 如果True, 输入点云通道数为6 (XYZ + 法向量)，否则为3
        """
        super(PointNetPlusPlusCls, self).__init__()
        in_channel = 3 if not normal_channel else 3  # 额外特征将在 forward 中分离
        self.normal_channel = normal_channel

        # 第一层 SA: 从原始点云采样512个中心点，邻域半径0.2, 每个邻域32个点, MLP: [64, 64, 128]
        self.sa1 = SimpleSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128])
        # 第二层 SA: 从 SA1 的结果采样128个中心点，邻域半径0.4, 每个邻域64个点, MLP: [128, 128, 256]
        self.sa2 = SimpleSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        # 第三层 SA: 全局聚合，group_all=True, MLP: [256, 512, 1024]
        self.sa3 = SimpleSetAbstractionGroupAll(in_channel=256, mlp=[256, 512, 1024])

        # 分类全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        输入:
            x: [B, C, N] 点云数据，其中如果 normal_channel=True, C=6 (前3为坐标，后3为法向量)
        输出:
            x: [B, num_classes] 分类得分（log_softmax）
        """
        B, C, N = x.size()
        if self.normal_channel:
            xyz = x[:, :3, :]  # [B, 3, N]
            points = x[:, 3:, :]  # [B, 3, N]
        else:
            xyz = x
            points = None

        # SA1
        l1_xyz, l1_points = self.sa1(xyz, points)  # l1_xyz: [B, 3, 512], l1_points: [B, 128, 512]
        # SA2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_xyz: [B, 3, 128], l2_points: [B, 256, 128]
        # SA3（全局）
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_xyz: [B, 3, 1], l3_points: [B, 1024, 1]

        # 全局特征: 展平至 [B, 1024]
        x = l3_points.view(B, 1024)
        # 分类全连接层
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# ------------------ 损失函数 ------------------

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss

# ------------------ Demo 测试函数 ------------------

def demo():
    # 参数设置
    batch_size = 32        # 每个 batch 中的点云数
    num_points = 1024      # 每个点云包含的点数
    normal_channel = True  # 如果 True，则输入点云通道为6 (例如 XYZ + 法向量)
    num_channels = 6 if normal_channel else 3
    num_classes = 40       # 分类任务类别数

    # 生成随机点云数据，形状为 [batch_size, num_channels, num_points]
    point_clouds = torch.rand(batch_size, num_channels, num_points)
    # 生成对应的随机标签（整数，范围在 [0, num_classes-1]）
    labels = torch.randint(0, num_classes, (batch_size,))

    # 初始化模型和损失函数
    model = PointNetPlusPlusCls(num_classes=num_classes, normal_channel=normal_channel)
    criterion = get_loss()

    # 设定模型为训练模式（dropout 等会生效）
    model.train()

    # 前向传播得到预测结果
    preds = model(point_clouds)  # preds: [batch_size, num_classes]

    # 计算损失（分类损失）
    loss = criterion(preds, labels)

    print("模型输出 shape:", preds.shape)
    print("损失值:", loss.item())

if __name__ == "__main__":
    demo()

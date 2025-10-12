import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不依赖 GUI

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import time

# ----------------------------
# 1. 数据读取与预处理函数
# ----------------------------
def read_ply(file_path):
    """
    读取 .ply 文件，跳过 header 部分，返回点云数据（假设每个顶点仅含有 x, y, z 三个属性）
    """
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line.strip() == "end_header":
                break
        try:
            points = np.loadtxt(f)
        except Exception as e:
            raise RuntimeError(f"读取点云数据时出错: {e}")
    return points

def sample_points(points, num_points=10000):
    """
    从点云中随机采样固定数量的点。若点数不足，则使用放回采样
    """
    if points.shape[0] >= num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
    return points[indices, :]

# ----------------------------
# 2. 自定义数据集类
# ----------------------------
class PointCloudDataset(Dataset):
    def __init__(self, rotated_dir, file_list, num_points=1024):
        """
        rotated_dir: 旋转后的点云文件所在的目录（包含 *.ply 文件与对应的正向向量文件 *.txt）
        file_list: 文件名列表
        num_points: 每个点云采样点数
        """
        self.rotated_dir = rotated_dir
        self.file_list = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.rotated_dir, file_name)
        try:
            points = read_ply(file_path)
        except Exception as e:
            raise RuntimeError(f"加载文件 {file_path} 时出错: {e}")
        points = sample_points(points, self.num_points)
        # 转换为 tensor，形状 (num_points, 3)
        points = torch.tensor(points, dtype=torch.float32)

        # 读取对应的正向向量文件，假设与 .ply 同名但扩展名为 .txt
        target_file = file_path.replace('.ply', '.txt')
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"正向向量文件不存在: {target_file}")
        with open(target_file, 'r') as f:
            vector_str = f.read().strip().split()
            if len(vector_str) < 3:
                raise ValueError(f"正向向量文件格式错误: {target_file}")
            target = [float(val) for val in vector_str[:3]]
        target = torch.tensor(target, dtype=torch.float32)
        return points, target

# ----------------------------
# 3. 模型定义：简化版 PointNet++（输出 3 维向量，与原 SimplePointNet 输出保持一致）
# ----------------------------

# 辅助函数：根据索引选取点
def index_points(points, idx):
    """
    输入:
        points: (B, N, C) 点云数据
        idx: (B, S) 或 (B, S, K) 采样的索引
    输出:
        new_points: 选取的点云数据，形状对应 idx
    """
    device = points.device
    B = points.shape[0]
    if idx.dim() == 2:
        S = idx.shape[1]
        batch_indices = torch.arange(B, device=device).view(B, 1).repeat(1, S)
        new_points = points[batch_indices, idx]
    elif idx.dim() == 3:
        S, K = idx.shape[1], idx.shape[2]
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).repeat(1, S, K)
        new_points = points[batch_indices, idx]
    return new_points

def square_distance(src, dst):
    """
    计算 src 与 dst 之间的平方欧氏距离
    src: (B, N, C)
    dst: (B, M, C)
    输出:
        dist: (B, N, M)
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist

def query_ball_point(new_xyz, xyz, nsample):
    """
    对于每个新的中心点 new_xyz，从原始点云 xyz 中找出其最近的 nsample 个点。
    new_xyz: (B, npoint, 3)
    xyz: (B, N, 3)
    输出:
        idx: (B, npoint, nsample) 每个中心点对应的点索引
    """
    dist = square_distance(new_xyz, xyz)  # (B, npoint, N)
    _, idx = dist.topk(nsample, largest=False, sorted=False)
    return idx

# Set Abstraction
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp_channels, group_all=False):
        """
        npoint: 采样的中心点数。如果 group_all=True，此处无效
        nsample: 每个中心点选取的近邻点数。如果 group_all=True，此处无效
        in_channel: 输入特征维度（不含 xyz 坐标）
        mlp_channels: 列表形式的 MLP 输出尺寸，将用于局部特征提取
        group_all: 如果为 True，则对整个点云进行全局特征提取
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel + 3  # 输入为相对坐标加上原始特征（若有）
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        输入:
            xyz: (B, N, 3) 原始坐标
            points: (B, N, D) 原始特征，如果没有则为 None
        输出:
            new_xyz: (B, npoint, 3) 采样后的中心点坐标（group_all 时为 (B, 1, 3)）
            new_points: (B, npoint, mlp_channels[-1]) 局部特征描述符
        """
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)  # (B, 1, N, 3+D)
            else:
                new_points = grouped_xyz  # (B, 1, N, 3)
        else:
            # 随机采样 npoint 个中心点（实际可使用 farthest point sampling）
            fps_idx = torch.stack([torch.randperm(N)[:self.npoint] for _ in range(B)]).to(xyz.device)  # (B, npoint)
            new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
            # 找到每个中心点的邻域
            idx = query_ball_point(new_xyz, xyz, self.nsample)  # (B, npoint, nsample)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, 3+D)
            else:
                new_points = grouped_xyz_norm  # (B, npoint, nsample, 3)
        # 转换维度以适应 2D 卷积：(B, C, npoint, nsample)
        new_points = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        return new_xyz, new_points

# PointNet++
class PointNetPP(nn.Module):
    def __init__(self):
        super(PointNetPP, self).__init__()
        # 第一层 SA 模块：采样 128 个点，每个点选取 32 个邻居，提取局部特征，输出维度为 128
        self.sa1 = PointNetSetAbstraction(npoint=128, nsample=32, in_channel=0, mlp_channels=[64, 64, 128], group_all=False)
        # 第二层 SA 模块：在第一层的中心点上采样 32 个点，每个采样点选取 32 个邻居，提取特征，输出维度为 256
        self.sa2 = PointNetSetAbstraction(npoint=32, nsample=32, in_channel=128, mlp_channels=[128, 128, 256], group_all=False)
        # 第三层 SA 模块：全局特征提取，group_all=True，输出全局特征 1024 维
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256, mlp_channels=[256, 512, 1024], group_all=True)
        # 全连接层，将全局特征映射到 3 维输出
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        """
        输入:
            x: (B, N, 3) 点云坐标
        输出:
            y: (B, 3) 回归输出，例如物体正向向量
        """
        B, N, _ = x.shape
        # 第一层：无初始特征，传入 None
        l1_xyz, l1_points = self.sa1(x, None)             # l1_xyz: (B, 128, 3), l1_points: (B, 128, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)       # l2_xyz: (B, 32, 3),  l2_points: (B, 32, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)       # l3_xyz: (B, 1, 3),   l3_points: (B, 1, 1024)
        x = l3_points.view(B, -1)                           # (B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))                   # (B, 512)
        x = F.relu(self.bn2(self.fc2(x)))                   # (B, 256)
        x = self.dropout(x)
        x = self.fc3(x)                                     # (B, 3)
        return x

# ----------------------------
# 4. 训练、验证、测试与可视化
# ----------------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=100):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for points, target in train_loader:
            points = points.to(device)  # (B, N, 3)
            target = target.to(device)  # (B, 3)
            optimizer.zero_grad()
            outputs = model(points)  # (B, 3)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for points, target in val_loader:
                points = points.to(device)
                target = target.to(device)
                outputs = model(points)
                loss = criterion(outputs, target)
                running_val_loss += loss.item() * points.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        # 保存验证集上 loss 最低的模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for points, target in test_loader:
            points = points.to(device)
            target = target.to(device)
            outputs = model(points)
            # 打印每个样本的输出与目标对比
            for p, t in zip(outputs, target):
                mse_sample = torch.mean((p - t) ** 2)
                print(f'Manual MSE loss for this sample: {mse_sample.item()}')
            loss_manual = torch.mean((outputs - target) ** 2)
            print(f'Manual computed batch MSE loss: {loss_manual.item()}')
            loss = criterion(outputs, target)
            print(f'Criterion computed batch MSE loss: {loss.item()}')
            running_loss += loss.item() * points.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")
    return test_loss

# ----------------------------
# 5. 主函数：数据加载、模型训练及可视化
# ----------------------------
def main():
    # 设置随机种子，保证结果可复现
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    label = "bookshelf"  # 此处 label 可根据实际情况修改，例如 "bottle" 或其他类别
    rotated_dir = "/home/pablo/ForwardNet/data/modelnet40_normal_resampled_rotated_ply/" + label
    original_dir = "/home/pablo/ForwardNet/data/modelnet40_normal_resampled_ply/" + label

    # 获取旋转后数据的所有文件（假设文件后缀为 .ply）
    all_files = [f for f in os.listdir(rotated_dir) if f.endswith(".ply")]
    all_files.sort()  # 排序以确保相同名字对应原始与旋转数据

    # 数据集划分比例：70% 训练，15% 验证，15% 测试
    total_num = len(all_files)
    print(f"数据总数量：{total_num}")
    train_num = int(0.7 * total_num)
    val_num = int(0.15 * total_num)
    test_num = total_num - train_num - val_num

    random.shuffle(all_files)
    train_files = all_files[:train_num]
    val_files = all_files[train_num:train_num+val_num]
    test_files = all_files[train_num+val_num:]
    print(f"总样本数: {total_num}, 训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

    num_points = 10000  # 每个点云采样点数
    train_dataset = PointCloudDataset(rotated_dir, train_files, num_points)
    val_dataset   = PointCloudDataset(rotated_dir, val_files, num_points)
    test_dataset  = PointCloudDataset(rotated_dir, test_files, num_points)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" device: {device}")

    # PointNet++
    model = PointNetPP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200  # 根据数据量和任务复杂度调整 epoch 数

    start_time = time.time()
    model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training time: {elapsed/60:.2f} 分钟")

    test_loss = test_model(model, test_loader, criterion, device)

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{label} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("bookshelf_pointnet++.png")
    plt.show()

if __name__ == '__main__':
    main()

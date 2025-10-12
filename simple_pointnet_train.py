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
# 3. 模型定义：简化版 PointNet++（类似 PointNet 架构）
# ----------------------------
class SimplePointNet(nn.Module):
    def __init__(self):
        super(SimplePointNet, self).__init__()
        # 卷积模块：输入 (B,3,N)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        # 全连接层：将全局特征变换到 3 维输出
        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # 转换为 (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N)
        # 全局 max pooling
        x = torch.max(x, 2)[0]  # (B, 256)
        x = F.relu(self.bn4(self.fc1(x)))  # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 3) => 输出正向向量预测
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

    # 载入验证时表现最好的模型
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
            # print
            for p, t in zip(outputs, target):
                # single sample
                mse_sample = torch.mean((p - t) ** 2)
                print(f'Manual MSE loss for this sample: {mse_sample.item()}')

            #  batch  MSE loss
            loss_manual = torch.mean((outputs - target) ** 2)
            print(f'Manual computed batch MSE loss: {loss_manual.item()}')

            # nn.MSE loss
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

    label = "chair"  # 此处 label 可根据实际情况修改，例如 "bottle" 或其他类别
    # 定义数据所在路径（请根据实际情况修改为 WSL2 下的路径）
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

    # 打乱文件顺序
    random.shuffle(all_files)
    train_files = all_files[:train_num]
    val_files = all_files[train_num:train_num+val_num]
    test_files = all_files[train_num+val_num:]
    print(f"总样本数: {total_num}, 训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

    # 创建数据集
    num_points = 10000  # 每个点云采样点数
    train_dataset = PointCloudDataset(rotated_dir, train_files, num_points)
    val_dataset   = PointCloudDataset(rotated_dir, val_files, num_points)
    test_dataset  = PointCloudDataset(rotated_dir, test_files, num_points)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用 device: {device}")

    # 创建模型、损失函数、优化器
    model = SimplePointNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 200  # 根据数据量和任务复杂度调整 epoch 数

    # 开始训练
    start_time = time.time()
    model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"训练总耗时: {elapsed/60:.2f} 分钟")

    # 测试模型
    test_loss = test_model(model, test_loader, criterion, device)

    # 可视化训练 & 验证 loss 曲线
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{label} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # 保存图像到文件后再显示
    plt.savefig("chair_simplepointnet_training_validation_loss.png")
    plt.show()

if __name__ == '__main__':
    main()

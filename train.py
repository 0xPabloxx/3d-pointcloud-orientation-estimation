import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import PointCloudDataset  # 注意这里的导入路径
#from models import PointNetPPXYZ    # 已替换为 PointNetPPWithAxes
# from models import PointTransformer
# from models import PointNet
from models import  PointNetPPXYZ_Schedmit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNetPPXYZ_Schedmit().to(device)

# ---------------- 配置区 ----------------
# ModelNet40 数据所在根目录，每个子文件夹为一个 label，如 "airplane", "bench", ...
DATA_ROOT    = "/home/pablo/ForwardNet/data/full_mn40_normal_resampled_3d_rotated_ply"
# 结果与输出根目录
RESULTS_ROOT = "/home/pablo/ForwardNet/results"
OUTPUT_ROOT  = "/home/pablo/ForwardNet/results/output"
# 选择训练的标签区间 (1-based 索引); 如 (1,10)，或 (20,30)，或 None 表示全部
# 9 chair
LABEL_RANGE  = None
# 每个标签的训练参数
NUM_POINTS   = 10000
BATCH_SIZE   = 16
NUM_EPOCHS   = 200
LR           = 1e-3
SEED         = 42

# ---- ply 读写工具，带 axes 注释 ----
def read_ply_vertices(file_path):
    """
    读取 ASCII PLY 文件的顶点部分，返回 Nx3 numpy 数组。
    """
    verts = []
    with open(file_path, 'r') as f:
        # 找到 header 结束
        while True:
            line = f.readline()
            if not line or line.strip() == 'end_header':
                break
        # 读取所有顶点行
        for l in f:
            parts = l.strip().split()
            if len(parts) < 3:
                continue
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(verts)


#without colors
def write_ply_with_axes(vertices, axes, out_file):
    """
    写入带 axes（原点+三个向量端点）和 edges 注释的 ASCII PLY。
      - vertices: (N,3) numpy array 原始点云
      - axes: list of three (3,) numpy 向量 [vx, vy, vz]
      - out_file: 输出路径
    """
    num_pts = vertices.shape[0]
    tot_pts = num_pts + 4  # 原始 N + 4 个附加点

    with open(out_file, 'w') as f:
        # 1) Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {tot_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element edge 3\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # 2) 四个附加顶点：原点 + 三个预测向量端点  (索引 0..3)
        f.write("0.000000 0.000000 0.000000\n")  # 原点
        for vec in axes:
            f.write(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n")

        # 3) 原始顶点：索引 4..4+N-1
        for pt in vertices:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # 4) 三条边（不含颜色）：0->1, 0->2, 0->3
        f.write("0 1\n")
        f.write("0 2\n")
        f.write("0 3\n")

    print(f"Saved ply with axes and edges (no color): {out_file}")
#with colors
# def write_ply_with_axes(vertices, axes, out_file):
#     num_pts = len(vertices)
#     tot_pts = num_pts + 4
#
#     with open(out_file,'w') as f:
#         # Header 保留颜色声明
#         f.write("ply\nformat ascii 1.0\n")
#         f.write(f"element vertex {tot_pts}\n")
#         f.write("property float x\nproperty float y\nproperty float z\n")
#         f.write("element edge 3\n")
#         f.write("property int vertex1\nproperty int vertex2\n")
#         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
#         f.write("end_header\n")
#
#         # 1) 附加顶点：原点 + 三轴端点 （索引0..3）
#         f.write("0.0 0.0 0.0\n")
#         for vec in axes:
#             f.write(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n")
#
#         # 2) 原始顶点：索引4..4+N-1
#         for pt in vertices:
#             f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
#
#         # 3) 三条有颜色的边
#         f.write("0 1 255 0 0\n")
#         f.write("0 2 0 255 0\n")
#         f.write("0 3 0 0 255\n")






def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Avg MSE')
    plt.legend(); plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve: {save_path}")


# 单标签训练函数
def train_single_label(label_name, label_idx, total_labels, device):
    print(f"\n=== [{label_idx}/{total_labels}] Training label: {label_name} ===")

    data_dir = os.path.join(DATA_ROOT, label_name)
    all_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.ply'))
    random.shuffle(all_files)

    # 划分集
    n_train = int(0.7 * len(all_files))
    n_val   = int(0.15 * len(all_files))

    train_ds = PointCloudDataset(data_dir, all_files[:n_train], num_points=NUM_POINTS)
    val_ds   = PointCloudDataset(data_dir, all_files[n_train:n_train+n_val], num_points=NUM_POINTS)
    test_ds  = PointCloudDataset(data_dir, all_files[n_train+n_val:], num_points=NUM_POINTS)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_files = all_files[n_train+n_val:]

    # 初始化模型 & 优化器
    model = PointNetPPXYZ_Schedmit().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    best_val = float('inf'); best_state = None

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        tot = 0.0
        for pts, tgt in train_ld:
            pts = pts.to(device)
            gx, gy, gz = tgt[:,0].to(device), tgt[:,1].to(device), tgt[:,2].to(device)
            optimizer.zero_grad()
            #vx, vy, vz = model(pts)
            vy, vz = model(pts)
            pred_loss = (criterion(vy, gy) + criterion(vz, gz)) / 2.0
            dot_prod = (vy * vz).sum(dim=1)  # [B]
            orth_loss = dot_prod.pow(2).mean()
            λ = 0.1
            loss =pred_loss + λ * orth_loss
            loss.backward()
            optimizer.step()
            tot += loss.item() * pts.size(0)
        train_losses.append(tot / len(train_ld.dataset))

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for pts, tgt in val_ld:
                pts = pts.to(device)
                gx, gy, gz = tgt[:,0].to(device), tgt[:,1].to(device), tgt[:,2].to(device)
                vy, vz = model(pts)
                pred_loss = (criterion(vy, gy) + criterion(vz, gz)) / 2.0
                dot_prod = (vy * vz).sum(dim=1)  # [B]
                orth_loss = dot_prod.pow(2).mean()
                λ = 0.1
                loss = pred_loss + λ * orth_loss
                tot += loss.item() * pts.size(0)
        val_losses.append(tot / len(val_ld.dataset))

        print(f"[{label_name}] Epoch {epoch}/{NUM_EPOCHS}  Train {train_losses[-1]:.4f}  Val {val_losses[-1]:.4f}")

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = model.state_dict()

    # 保存该标签最优模型
    if best_state:
        model.load_state_dict(best_state)

    # 绘图并保存
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    curve_path = os.path.join(RESULTS_ROOT, f"{label_name}_loss_curve.png")
    plot_losses(train_losses, val_losses, curve_path)

    # 记录最终验证损失
    with open(os.path.join(RESULTS_ROOT, "summary.txt"), "a") as f:
        f.write(f"{label_name}\t{best_val:.6f}\n")

    # 随机选 10 个样本保存带预测向量 PLY
    for idx, fname in enumerate(random.sample(test_files, min(10, len(test_files))), start=1):
        sample_ply = os.path.join(data_dir, fname)
        # 读取点云
        verts = read_ply_vertices(sample_ply)
        pts = torch.tensor(verts, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            vy, vz = model(pts)
        vx = torch.cross(vy, vz, dim=1)
        vx_hat = F.normalize(vx, dim=1)
        axes = [vx_hat[0].cpu().numpy(), vy[0].cpu().numpy(), vz[0].cpu().numpy()]

        out_dir = os.path.join(OUTPUT_ROOT, label_name)
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(fname)[0]
        # 在文件名里加上序号 idx，就不会覆盖
        out_file = os.path.join(out_dir, f"{base}_pred_{idx}.ply")
        write_ply_with_axes(verts, axes, out_file)

    return best_val

def main():
    # 固定随机种子
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # 列出所有标签目录，过滤隐藏文件夹
    all_labels = sorted([
        d for d in os.listdir(DATA_ROOT)
        if not d.startswith('.')
           and os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    if LABEL_RANGE is not None:
        lo, hi = LABEL_RANGE
        selected = all_labels[lo-1:hi]
    else:
        selected = all_labels

    print(f"Will train labels {selected[0]} … {selected[-1]}  (total {len(selected)})")

    # 清空 summary 文件
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    open(os.path.join(RESULTS_ROOT, "summary.txt"), "w").close()

    # 逐标签训练
    for idx, label_name in enumerate(selected, start=1):
        train_single_label(label_name, idx, len(selected), device)

    print("=== All labels trained. Summary written to summary.txt ===")


if __name__ == '__main__':
    main()
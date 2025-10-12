import os
import numpy as np
import torch
from torch.utils.data import Dataset

def read_ply(file_path):
    with open(file_path, 'r') as f:
        while True:
            if f.readline().strip() == "end_header":
                break
        try:
            points = np.loadtxt(f)
        except Exception as e:
            raise RuntimeError(f"读取点云数据时出错: {e}")
    return points

def sample_points(points, num_points=10000):
    if points.shape[0] >= num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        idx = np.random.choice(points.shape[0], num_points, replace=True)
    return points[idx, :]

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, file_list, num_points=1024):
        self.data_dir   = data_dir
        self.file_list  = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. 读取并采样点云
        fname    = self.file_list[idx]
        ply_path = os.path.join(self.data_dir, fname)
        pts_np   = read_ply(ply_path)                       # Nx3 或 Nx6
        pts_np   = sample_points(pts_np, self.num_points)   # (num_points, D)
        pts      = torch.tensor(pts_np, dtype=torch.float32)

        # 2. 读取三行方向向量，构造 (3,3) 的 tgt
        txt_path = ply_path.replace('.ply', '.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"正向向量文件不存在: {txt_path}")
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError(f"正向向量文件格式错误 (少于3行): {txt_path}")

        vecs = []
        for i in range(3):
            parts = lines[i].strip().split()
            if len(parts) != 3:
                raise ValueError(f"第 {i+1} 行向量格式错误: {lines[i]}")
            vecs.append([float(x) for x in parts])

        tgt = torch.tensor(vecs, dtype=torch.float32)  # (3,3)

        return pts, tgt

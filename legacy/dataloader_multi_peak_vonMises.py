import os
import numpy as np
import torch
from torch.utils.data import Dataset

def read_ply(p):
    """
    读取 .ply 文件，返回 N×3 的 numpy 数组
    """
    with open(p, "r") as f:
        # 跳过 header 行直到 "end_header"
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == "end_header":
                break
        pts = np.loadtxt(f, dtype=np.float32)[:, :3]
    return pts

def sample_pts(arr, num=10000):
    n = len(arr)
    if n == 0:
        return arr
    idx = np.random.choice(n, num, replace=(n < num))
    return arr[idx]

class PointCloudDatasetMvM(Dataset):
    def __init__(self, samples, num_points, max_K=4, label_map=None):
        self.samples = list(samples)  # 每项 (ply_path, gt_txt_path, category)
        self.num_points = num_points
        self.max_K = max_K
        # label_map 是外部传入的 dict
        self.label_map = label_map or {cat: i for i, cat in enumerate(sorted(set(s[2] for s in samples)))}

    @staticmethod
    def _read_mvM(gt_path, max_K=4):
        mus = []
        kappas = []
        ws = []
        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if len(lines) < 2:
            raise RuntimeError(f"GT file too short or malformed: {gt_path}")
        parts = lines[0].split()
        if len(parts) < 2:
            raise RuntimeError(f"GT file K line malformed: {gt_path}")
        K = int(parts[1])
        data_lines = lines[2:]
        for ln in data_lines:
            vals = ln.split()
            if len(vals) >= 3:
                mu_v = float(vals[0])
                kappa_v = float(vals[1])
                w_v = float(vals[2])
                mus.append(mu_v)
                kappas.append(kappa_v)
                ws.append(w_v)
        while len(mus) < max_K:
            mus.append(0.0)
            kappas.append(0.0)
            ws.append(0.0)
        arr = np.stack([mus, kappas, ws], axis=1)[:max_K]
        return torch.tensor(arr, dtype=torch.float32), K

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ply_p, gt_txt, category = self.samples[idx]
        # 调试打印（可注释掉）
        # print(f"[DEBUG] idx {idx}, ply: {ply_p}, gt: {gt_txt}")

        if not os.path.exists(ply_p):
            raise FileNotFoundError(f"PLY not found: {ply_p}")
        xyz_np = read_ply(ply_p)
        sampled = sample_pts(xyz_np, self.num_points)
        xyz = torch.from_numpy(sampled.astype(np.float32))

        gt_path = gt_txt
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT txt not found: {gt_path}, for ply: {ply_p}")

        vm_params, K = self._read_mvM(gt_path, self.max_K)
        label_idx = self.label_map[category]
        return xyz, vm_params, K, torch.tensor(label_idx, dtype=torch.long)

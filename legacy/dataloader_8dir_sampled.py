# dataloader_8dir_sampled.py
import os, numpy as np, torch
from torch.utils.data import Dataset

# ---------- 读 / 采样 ----------
def read_ply(p):
    with open(p, "r") as f:
        while f.readline().strip() != "end_header":
            pass
        pts = np.loadtxt(f, dtype=np.float32)[:, :3]
    return pts

def sample_pts(arr, num=10_000):
    idx = np.random.choice(len(arr), num, replace=len(arr) < num)
    return arr[idx]

# ---------- Dataset ----------
class PointCloudDataset(Dataset):
    """
    samples     : [(ply_path, prob8_path, label_str), ...]
    uniform_set : 若类别在其中或 _8dir.txt 缺失 → 返回均匀分布
    返回:
      pts        FloatTensor (N,3)
      prob_gt    FloatTensor (8,)
      label_idx  int
    """
    def __init__(self, samples, num_points, uniform_set, label_map=None):
        self.samples     = list(samples)
        self.num_points  = num_points
        self.uniform_set = set(uniform_set)
        # label→id
        self.label2id = label_map or {}
        if not self.label2id:
            for _, _, lbl in self.samples:
                if lbl not in self.label2id:
                    self.label2id[lbl] = len(self.label2id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ply_p, prob_p, lbl = self.samples[idx]

        # 点云采样
        xyz = torch.from_numpy(sample_pts(read_ply(ply_p), self.num_points))

        # 8-维概率
        if (lbl in self.uniform_set) or (not os.path.exists(prob_p)):
            prob = torch.full((8,), 0.125, dtype=torch.float32)
        else:
            try:
                arr = np.loadtxt(prob_p, dtype=np.float32).flatten()
                prob = torch.tensor(arr[:8], dtype=torch.float32)
            except Exception:
                prob = torch.full((8,), 0.125, dtype=torch.float32)

        return xyz, prob, self.label2id[lbl]


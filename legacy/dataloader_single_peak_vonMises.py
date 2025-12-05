# dataloader_single_peak_vonMises.py
import os, numpy as np, torch
from torch.utils.data import Dataset

def read_ply(p):
    with open(p, "r") as f:
        while f.readline().strip() != "end_header":
            pass
        pts = np.loadtxt(f, dtype=np.float32)[:, :3]
    return pts

def sample_pts(arr, num=10_000):
    idx = np.random.choice(len(arr), num, replace=len(arr) < num)
    return arr[idx]

class PointCloudDatasetVonMises(Dataset):
    """
    samples : [(ply_path, label_str), ...]
    返回:
      pts       FloatTensor (N,3)
      vm_params FloatTensor (2,) -> [μ, κ]
      label_idx int
    """
    def __init__(self, samples, num_points, label_map=None):
        self.samples    = list(samples)
        self.num_points = num_points
        self.label2id   = label_map or {}
        if not self.label2id:
            for _, lbl in self.samples:
                if lbl not in self.label2id:
                    self.label2id[lbl] = len(self.label2id)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_vm(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            #mu, kappa = map(float, lines[1].split()[:2])  # 第二行
            mu, kappa = map(float, lines[0].split()[:2])  # 第一行非注释
        except Exception:
            mu, kappa = 0.0, 0.0
        return mu, max(kappa, 0.0)

    def __getitem__(self, idx):
        ply_p, lbl = self.samples[idx]
        xyz = torch.from_numpy(sample_pts(read_ply(ply_p), self.num_points))
        vm_path = ply_p.with_name(ply_p.stem + "_single_peak_vM_gt.txt")
        mu, kappa = self._read_vm(vm_path)
        return xyz, torch.tensor([mu, kappa], dtype=torch.float32), self.label2id[lbl]

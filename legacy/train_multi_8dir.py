#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合 six-class 样本 → PointNet++ 回归向量 → 投影 8dir → MSE
输出：
  ~/ForwardNet/results/
      ├── best_model.pth
      ├── summary.txt       (label\tloss  + Overall)
      └── figs/
            ├── overall_loss.png
            └── <label>_loss.png × N
"""
import os, time, random, json, numpy as np, torch
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from dataloader_8dir_sampled import PointCloudDataset
from ForwardNet.models.pointnet_pp_Fwd import PointNetPPFwd, DIRS_8

# ---------- 路径配置 ----------
ROOT = Path("/home/pablo/ForwardNet/data/chair_toilet_sofa_plant_bowl_bottle")
RES  = Path("/home/pablo/ForwardNet/results"); RES.mkdir(parents=True, exist_ok=True)
FIGS = RES / "figs"                          # 单独文件夹放曲线
FIGS.mkdir(parents=True, exist_ok=True)

# ---------- 超参 ----------
NUM_POINTS = 10_000
BATCH      = 16
EPOCHS     = 200
LR         = 1e-3
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIRS_8_T = DIRS_8.to(device)
UNIFORM  = {"bottle", "bowl", "plant"}

# ---------- 工具函数 ----------
def proj_probs(vec):                 # vec (B,3) → (B,8)
    v = F.normalize(vec, dim=1)
    sims = (v @ DIRS_8_T.T).clamp(min=0)
    return sims / sims.sum(dim=1, keepdim=True).clamp(min=1e-8)

def plot_curve(xs, ys_dict, title, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)     # <- 新增，确保目录存在
    plt.figure()
    for k, (tr, va) in ys_dict.items():
        plt.plot(xs, tr, label=f'{k}-Tr')
        plt.plot(xs, va, label=f'{k}-Val', ls='--')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()

# ---------- 构造“全混合”样本 ----------
labels = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
label_map = {lbl: i for i, lbl in enumerate(labels)}
samples = [(ply,
            ply.with_name(ply.stem + "_8dir.txt"),
            lbl)
           for lbl in labels
           for ply in (ROOT / lbl).glob("*.ply")]
random.shuffle(samples)

n_total = len(samples)
n_tr = int(0.70 * n_total); n_va = int(0.15 * n_total)
train_ds = PointCloudDataset(samples[:n_tr]        , NUM_POINTS, UNIFORM, label_map)
val_ds   = PointCloudDataset(samples[n_tr:n_tr+n_va], NUM_POINTS, UNIFORM, label_map)
test_ds  = PointCloudDataset(samples[n_tr+n_va:]   , NUM_POINTS, UNIFORM, label_map)

tr_loader = DataLoader(train_ds, BATCH, shuffle=True , num_workers=4, pin_memory=True)
va_loader = DataLoader(val_ds  , BATCH, shuffle=False, num_workers=4, pin_memory=True)
te_loader = DataLoader(test_ds , BATCH, shuffle=False, num_workers=4, pin_memory=True)

print(f"Samples  train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")
print("Labels   :", ", ".join(labels))

# ---------- 训练 ----------
model = PointNetPPFwd().to(device)
opt   = optim.Adam(model.parameters(), lr=LR)
crit  = nn.MSELoss()

hist_overall = {'train': [], 'val': []}
hist_label   = {lbl: {'train': [], 'val': []} for lbl in labels}
best_val = float('inf'); best_state = None

t0_global = time.time()
for ep in range(1, EPOCHS + 1):
    ep_start = time.time()
    for phase, loader in [('train', tr_loader), ('val', va_loader)]:
        model.train() if phase == 'train' else model.eval()
        agg_loss, n_seen = 0.0, 0
        bucket_loss = {lbl: 0.0 for lbl in labels}; bucket_cnt = {lbl: 0 for lbl in labels}

        for pts, prob_gt, lbl_idx in loader:
            pts, prob_gt = pts.to(device), prob_gt.to(device)
            if phase == 'train':
                opt.zero_grad()
            pred = proj_probs(model(pts))
            loss = crit(pred, prob_gt)
            if phase == 'train':
                loss.backward(); opt.step()

            bs = pts.size(0)
            agg_loss += loss.item() * bs; n_seen += bs
            for i, lid in enumerate(lbl_idx):
                lbl = labels[lid.item()]
                bucket_loss[lbl] += crit(pred[i], prob_gt[i]).item()
                bucket_cnt [lbl] += 1

        epoch_loss = agg_loss / n_seen
        hist_overall[phase].append(epoch_loss)
        for lbl in labels:
            hist_label[lbl][phase].append(bucket_loss[lbl] / max(1, bucket_cnt[lbl]))

        if phase == 'val' and epoch_loss < best_val:
            best_val, best_state = epoch_loss, model.state_dict()

    # ---- 日志 ----
    elapsed = time.time() - ep_start
    avg_ep  = (time.time() - t0_global) / ep
    eta     = avg_ep * (EPOCHS - ep)
    h, m = divmod(int(eta), 60)
    print(f"Ep {ep:03d}/{EPOCHS}  ⏱ {elapsed:4.0f}s  ETA {h//60:02d}:{h%60:02d}:{m:02d}  "
          f"Train {hist_overall['train'][-1]:.4f}  Val {hist_overall['val'][-1]:.4f}")

# ---------- 保存模型 ----------
torch.save(best_state, RES / "best_model.pth")

# ---------- 曲线 ----------
xs = range(1, EPOCHS + 1)
plot_curve(xs, {'overall': (hist_overall['train'], hist_overall['val'])},
           "Overall Loss", FIGS / "overall_loss.png")
for lbl in labels:
    plot_curve(xs, {lbl: (hist_label[lbl]['train'], hist_label[lbl]['val'])},
               f"{lbl} Loss", FIGS / f"{lbl}_loss.png")

# ---------- 测试 & summary ----------
model.load_state_dict(best_state); model.eval()
label_loss = {lbl: 0.0 for lbl in labels}; label_cnt = {lbl: 0 for lbl in labels}
total_sum, total_cnt = 0.0, 0
with torch.no_grad():
    for pts, prob_gt, lbl_idx in te_loader:
        pts, prob_gt = pts.to(device), prob_gt.to(device)
        pred = proj_probs(model(pts))
        batch_mse = crit(pred, prob_gt).item() * pts.size(0)
        total_sum += batch_mse; total_cnt += pts.size(0)
        for i, lid in enumerate(lbl_idx):
            lbl = labels[lid.item()]
            label_loss[lbl] += crit(pred[i], prob_gt[i]).item()
            label_cnt [lbl] += 1

for lbl in labels:
    label_loss[lbl] /= max(1, label_cnt[lbl])
overall_loss = total_sum / total_cnt

with open(RES / "summary.txt", "w") as f:
    for lbl in labels:
        f.write(f"{lbl}\t{label_loss[lbl]:.6f}\n")
    f.write(f"Overall\t{overall_loss:.6f}\n")

print("训练结束！所有文件已保存至:", RES)

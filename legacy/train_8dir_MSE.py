#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ 输出 8-logits，直接 softmax 后与 *_8dir.txt 概率做 MSE。
训练结果、曲线与模型权重均保存到 ~/ForwardNet/results
"""
import time, random, json, numpy as np, torch
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from dataloader_8dir_sampled import PointCloudDataset
from models.pointnet_pp_8dir import PointNetPP8Dir, DIRS_8   # ← 网络输出 8 logits

# ---------- 路径 ----------
ROOT = Path("/home/pablo/ForwardNet/data/chair_toilet_sofa_plant_bowl_bottle")
RES  = Path("/home/pablo/ForwardNet/results");  RES.mkdir(parents=True, exist_ok=True)
FIGS = RES / "figs";                             FIGS.mkdir(parents=True, exist_ok=True)

# ---------- 超参 ----------
NUM_POINTS, BATCH, EPOCHS, LR = 10_000, 16, 200, 1e-3
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNIFORM = {"bottle", "bowl", "plant"}

# ---------- 绘图工具 ----------
def plot_curve(xs, ys_dict, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for k,(tr,va) in ys_dict.items():
        plt.plot(xs, tr, label=f'{k}-Tr')
        plt.plot(xs, va, ls='--', label=f'{k}-Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

# ---------- 构造数据 ----------
labels    = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
label_map = {lbl:i for i,lbl in enumerate(labels)}

samples = [(ply,
            ply.with_name(ply.stem + "_8dir.txt"),
            lbl)
           for lbl in labels
           for ply in (ROOT/lbl).glob("*.ply")]
random.shuffle(samples)

n_total = len(samples); n_tr = int(.7*n_total); n_va = int(.15*n_total)
train_ds = PointCloudDataset(samples[:n_tr]        , NUM_POINTS, UNIFORM, label_map)
val_ds   = PointCloudDataset(samples[n_tr:n_tr+n_va], NUM_POINTS, UNIFORM, label_map)
test_ds  = PointCloudDataset(samples[n_tr+n_va:]   , NUM_POINTS, UNIFORM, label_map)

tr_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True , num_workers=4, pin_memory=True)
va_loader = DataLoader(val_ds  , batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
te_loader = DataLoader(test_ds , batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

print(f"Samples  train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

# ---------- 训练 ----------
model = PointNetPP8Dir().to(device)
opt   = optim.Adam(model.parameters(), lr=LR)
crit  = nn.MSELoss()

hist_overall = {'train': [], 'val': []}
hist_label   = {l:{'train':[], 'val':[]} for l in labels}
best_val, best_state = float('inf'), None
t0 = time.time()

# gt = ground truth
for ep in range(1, EPOCHS+1):
    ep_t = time.time()
    for phase, loader in (('train', tr_loader), ('val', va_loader)):
        model.train() if phase=='train' else model.eval()
        agg_loss, cnt = 0.0, 0
        bucket_loss   = {l:0.0 for l in labels}; bucket_cnt = {l:0 for l in labels}
        for pts, prob_gt, lbl_idx in loader:
            pts, prob_gt = pts.to(device), prob_gt.to(device)
            if phase == 'train': opt.zero_grad()
            logits = model(pts)                # (B,8)
            prob   = F.softmax(logits, dim=1)
            loss   = crit(prob, prob_gt)
            if phase == 'train': loss.backward(); opt.step()

            bs = pts.size(0)
            agg_loss += loss.item()*bs; cnt += bs
            for i, lid in enumerate(lbl_idx):
                lbl = labels[lid.item()]
                bucket_loss[lbl] += crit(prob[i], prob_gt[i]).item()
                bucket_cnt [lbl] += 1

        epoch_loss = agg_loss / cnt
        hist_overall[phase].append(epoch_loss)
        for l in labels:
            hist_label[l][phase].append(bucket_loss[l]/max(1,bucket_cnt[l]))

        if phase=='val' and epoch_loss < best_val:
            best_val, best_state = epoch_loss, model.state_dict()

    # 日志
    elapsed = time.time() - ep_t
    eta = (time.time()-t0)/ep * (EPOCHS-ep)
    h, m = divmod(int(eta), 60)
    print(f"Ep {ep:03}/{EPOCHS}  ⏱{elapsed:2.0f}s  ETA {h//60:02d}:{h%60:02d}:{m:02d}  "
          f"Train {hist_overall['train'][-1]:.4f}  Val {hist_overall['val'][-1]:.4f}")

# ---------- 保存 ----------
torch.save(best_state, RES/"best_model.pth")

xs = range(1, EPOCHS+1)
plot_curve(xs, {'overall':(hist_overall['train'],hist_overall['val'])},
           "Overall Loss", FIGS/"overall_loss.png")
for l in labels:
    plot_curve(xs,{l:(hist_label[l]['train'],hist_label[l]['val'])},
               f"{l} Loss", FIGS/f"{l}_loss.png")

# ---------- 测试 ----------
model.load_state_dict(best_state)
model.eval(); label_mse, label_cnt = {l:0.0 for l in labels}, {l:0 for l in labels}
tot_sum, tot_cnt = 0.0, 0
with torch.no_grad():
    for pts, prob_gt, lbl_idx in te_loader:
        pts, prob_gt = pts.to(device), prob_gt.to(device)
        prob = F.softmax(model(pts), dim=1)
        tot_sum += crit(prob, prob_gt).item()*pts.size(0); tot_cnt += pts.size(0)
        for i,lid in enumerate(lbl_idx):
            l = labels[lid.item()]
            label_mse[l] += crit(prob[i], prob_gt[i]).item()
            label_cnt [l] += 1
for l in labels: label_mse[l] /= label_cnt[l]
overall = tot_sum / tot_cnt

with open(RES/"summary.txt","w") as f:
    for l in labels: f.write(f"{l}\t{label_mse[l]:.6f}\n")
    f.write(f"Overall\t{overall:.6f}\n")

print("训练结束！所有输出位于:", RES)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from dataloader_single_peak_vonMises import PointCloudDatasetVonMises
from models.pointnet_pp_vonMises import PointNetPPVonMises

ROOT = Path("/home/pablo/ForwardNet/data/chair_toilet_sofa_plant_bowl_bottle")
RES  = Path("/home/pablo/ForwardNet/results/single_peak_vonMises_KL_1006_2"); RES.mkdir(parents=True, exist_ok=True)
FIGS = RES / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

NUM_POINTS, BATCH, EPOCHS, LR = 10_000, 16, 200, 1e-3
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    i0_p, i1_p = torch.special.i0(kappa_p), torch.special.i1(kappa_p)
    i0_q = torch.special.i0(kappa_q)
    a1_p = torch.where(kappa_p<=1e-6, torch.zeros_like(kappa_p), i1_p/i0_p)
    delta = mu_p - mu_q
    return torch.log(i0_q) - torch.log(i0_p) + kappa_p*a1_p - kappa_q*a1_p*torch.cos(delta)

def plot_curve(xs, ys_dict, title, path):
    plt.figure()
    for k,(tr,va) in ys_dict.items():
        plt.plot(xs,tr,label=f"{k}-Tr"); plt.plot(xs,va,"--",label=f"{k}-Val")
    plt.xlabel("Epoch"); plt.ylabel("KL"); plt.title(title); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()


# ---------- dataset ----------
labels = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
label_map = {l:i for i,l in enumerate(labels)}

samples = []
for lbl in labels:
    for vm_file in (ROOT / lbl).glob("*_single_peak_vM_gt.txt"):
        ply_name = vm_file.name.replace("_single_peak_vM_gt.txt", ".ply")
        ply_path = vm_file.with_name(ply_name)
        if ply_path.exists():
            samples.append((ply_path, lbl))

random.shuffle(samples)
n_total = len(samples)
n_tr = int(0.7 * n_total)
n_va = int(0.15 * n_total)

train_ds = PointCloudDatasetVonMises(samples[:n_tr], NUM_POINTS, label_map)
val_ds   = PointCloudDatasetVonMises(samples[n_tr:n_tr+n_va], NUM_POINTS, label_map)
test_ds  = PointCloudDatasetVonMises(samples[n_tr+n_va:], NUM_POINTS, label_map)

tr_loader = DataLoader(train_ds, BATCH, True, num_workers=4, pin_memory=True)
va_loader = DataLoader(val_ds, BATCH, False, num_workers=4, pin_memory=True)
te_loader = DataLoader(test_ds, BATCH, False, num_workers=4, pin_memory=True)

print(f"Samples found: {len(samples)} | train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")


# ---------- train ----------
model = PointNetPPVonMises().to(device)
opt = optim.Adam(model.parameters(), lr=LR)
hist = {"train":[], "val":[]}
best_val=float("inf"); best_state=None
t0=time.time()

for ep in range(1,EPOCHS+1):
    for phase,loader in (("train",tr_loader),("val",va_loader)):
        model.train() if phase=="train" else model.eval()
        total, cnt = 0.,0
        for xyz, vm_gt, _ in loader:
            xyz, vm_gt = xyz.to(device), vm_gt.to(device)
            mu_gt, kappa_gt = vm_gt[:,0], vm_gt[:,1]
            if phase=="train": opt.zero_grad()
            mu_pred, kappa_pred = model(xyz)
            loss_vec = kl_von_mises(mu_pred, kappa_pred, mu_gt, kappa_gt)
            loss = loss_vec.mean()
            if phase=="train":
                loss.backward(); opt.step()
            total += loss.item()*xyz.size(0); cnt+=xyz.size(0)
        avg = total/max(cnt,1)
        hist[phase].append(avg)
        if phase=="val" and avg<best_val:
            best_val,best_state=avg,model.state_dict()
    print(f"Ep {ep:03}/{EPOCHS} Train {hist['train'][-1]:.4f} Val {hist['val'][-1]:.4f}")

torch.save(best_state, RES/"vonMises_best.pth")
xs=range(1,EPOCHS+1)
plot_curve(xs,{"KL":(hist["train"],hist["val"])}, "von Mises KL", FIGS/"loss.png")

# ---------- test ----------
model.load_state_dict(best_state); model.eval()
tot_sum,tot_cnt=0.,0
with torch.no_grad():
    for xyz, vm_gt, _ in te_loader:
        xyz, vm_gt = xyz.to(device), vm_gt.to(device)
        mu_gt, kappa_gt = vm_gt[:,0], vm_gt[:,1]
        mu_pred, kappa_pred = model(xyz)
        loss_vec=kl_von_mises(mu_pred,kappa_pred,mu_gt,kappa_gt)
        tot_sum+=loss_vec.sum().item(); tot_cnt+=xyz.size(0)
print(f"Test KL = {tot_sum/max(tot_cnt,1):.6f}")

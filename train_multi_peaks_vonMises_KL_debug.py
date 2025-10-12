#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, random, re, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader_multi_peak_vonMises import PointCloudDatasetMvM
from models.pointnet_pp_mvM import PointNetPPMvM

ROOT = Path("/home/pablo/ForwardNet/data/MN40_multi_peak_vM_gt")
PLY_ROOT = Path("/home/pablo/ForwardNet/data/full_mn40_normal_resampled_2d_rotated_ply")
RES = Path("/home/pablo/ForwardNet/results/multi_peak_vonMises_KL_debug")
RES.mkdir(parents=True, exist_ok=True)
FIGS = RES / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

NUM_POINTS = 10_000
BATCH = 16
EPOCHS = 100
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 开启 autograd 异常检测
torch.autograd.set_detect_anomaly(True)

def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    # clamp to safe区间
    kappa_p = torch.clamp(kappa_p, 1e-6, 500.0)
    kappa_q = torch.clamp(kappa_q, 1e-6, 500.0)

    i0_p = torch.special.i0(kappa_p)
    i1_p = torch.special.i1(kappa_p)
    i0_q = torch.special.i0(kappa_q)

    # 检查 i0, i1 是否有异常
    if not torch.isfinite(i0_p).all():
        print("Non-finite i0_p:", i0_p)
    if not torch.isfinite(i0_q).all():
        print("Non-finite i0_q:", i0_q)
    if not torch.isfinite(i1_p).all():
        print("Non-finite i1_p:", i1_p)

    A_p = i1_p / i0_p  # 可能有 inf

    delta = mu_p - mu_q
    # wrap delta 至 [-π, π]
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    # 计算 KL
    KL = torch.log(i0_q / i0_p) + A_p * (kappa_p - kappa_q * torch.cos(delta))

    # 检查结果
    if not torch.isfinite(KL).all():
        print("KL non-finite:", KL, "inputs:", mu_p, kappa_p, mu_q, kappa_q)
    return KL


def match_loss(mu_pred, kappa_pred, w_pred, vm_gt, _, K_gt):
    B = mu_pred.size(0)
    loss_vec = torch.zeros(B, device=device)

    for b in range(B):
        K = int(K_gt[b].item())
        if K <= 0:
            loss_vec[b] = 0.0
            continue

        μp = mu_pred[b, :K]
        κp = kappa_pred[b, :K]
        wp = w_pred[b, :K]

        μg = vm_gt[b, :K, 0]
        κg = vm_gt[b, :K, 1]

        # 打印预测 vs gt
        print(f"[Batch {b}] K = {K}")
        print("  μp:", μp.detach().cpu().numpy())
        print("  κp:", κp.detach().cpu().numpy())
        print("  wp:", wp.detach().cpu().numpy())
        print("  μg:", μg.detach().cpu().numpy())
        print("  κg:", κg.detach().cpu().numpy())

        cost = torch.zeros((K, K), device=device)
        for i in range(K):
            for j in range(K):
                cost_ij = kl_von_mises(μp[i], κp[i], μg[j], κg[j])
                cost[i, j] = cost_ij
                if not torch.isfinite(cost_ij):
                    print(f"  cost[{i},{j}] non-finite:", cost_ij.item())

        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

        cost_np = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost_np)

        matched = [cost[r, c].item() for r, c in zip(row, col)]
        print("  matched cost:", matched)
        matched_ws = wp[row]
        ws_sum = torch.sum(matched_ws)
        print("  matched_ws:", matched_ws.detach().cpu().numpy(), "sum:", ws_sum.item())

        loss_bc = torch.sum(matched_ws * cost[row, col]) / (ws_sum + 1e-8)
        if not torch.isfinite(loss_bc):
            print("  loss_bc non-finite:", loss_bc)
        loss_vec[b] = loss_bc

    # 检查整个 loss_vec
    if not torch.isfinite(loss_vec).all():
        print("loss_vec has non-finite:", loss_vec)
    return loss_vec


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip())


def plot_curve(xs, ys_dict, title, path):
    plt.figure(figsize=(12, 8))
    for k in sorted(ys_dict.keys()):
        tr, va = ys_dict[k]
        plt.plot(xs, tr, label=f"{k}-Train")
        plt.plot(xs, va, "--", label=f"{k}-Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_label_curve(xs, train_vals, val_vals, label_name, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(xs, train_vals, label="Train")
    plt.plot(xs, val_vals, "--", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title(f"{label_name} - KL Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_total_curve(xs, total_train, total_val, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(xs, total_train, label="Total-Train")
    plt.plot(xs, total_val, "--", label="Total-Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("Overall KL Loss (Total)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_summary_txt(path_txt: Path, categories, hist, test_kl=None, best_val_epoch=None):
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("=== Multi-Peak von Mises KL Summary ===\n")
        if best_val_epoch is not None:
            f.write(f"Best Total Val Epoch: {best_val_epoch}\n")
        if test_kl is not None:
            f.write(f"Test KL: {test_kl:.6f}\n")
        f.write("\n-- Per-Category (last epoch) --\n")
        last = len(hist["total"]["train"]) - 1
        def _fmt(x):
            try:
                return f"{float(x):.6f}"
            except:
                return "nan"
        f.write(f"[TOTAL] Train={_fmt(hist['total']['train'][last])} "
                f"Val={_fmt(hist['total']['val'][last])}\n")
        for cat in categories:
            tr = hist[cat]["train"][last] if len(hist[cat]["train"]) > 0 else float("nan")
            va = hist[cat]["val"][last] if len(hist[cat]["val"]) > 0 else float("nan")
            f.write(f"[{cat}] Train={_fmt(tr)} Val={_fmt(va)}\n")


def main():
    if not ROOT.exists():
        raise RuntimeError(f"ROOT not exists: {ROOT}")
    gt_txts = list(ROOT.rglob("*_multi_peak_vM_gt.txt"))
    if len(gt_txts) == 0:
        raise RuntimeError("No GT txts found under ROOT")

    categories = sorted(set(txt.parent.name for txt in gt_txts))
    label_map = {cat: i for i, cat in enumerate(categories)}

    samples = []
    for txt in gt_txts:
        category = txt.parent.name
        base = txt.stem.replace("_multi_peak_vM_gt", "")
        ply_path = PLY_ROOT / category / (base + ".ply")
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found for GT: {txt} -> {ply_path}")
        samples.append((ply_path, txt, category))
    random.shuffle(samples)
    n_total = len(samples)
    n_tr = int(0.7 * n_total)
    n_va = int(0.15 * n_total)
    train_ds = PointCloudDatasetMvM(samples[:n_tr], NUM_POINTS, max_K=4, label_map=label_map)
    val_ds = PointCloudDatasetMvM(samples[n_tr:n_tr+n_va], NUM_POINTS, max_K=4, label_map=label_map)
    test_ds = PointCloudDatasetMvM(samples[n_tr+n_va:], NUM_POINTS, max_K=4, label_map=label_map)

    tr_loader = DataLoader(train_ds, BATCH, True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(val_ds, BATCH, False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(test_ds, BATCH, False, num_workers=4, pin_memory=True)

    print(f"Samples: {n_total} | train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    model = PointNetPPMvM().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    hist = {"total": {"train": [], "val": []}}
    for cat in categories:
        hist[cat] = {"train": [], "val": []}

    best_val = float("inf")
    best_state = None
    best_val_epoch = None

    total_time = 0.0
    epoch_times = []

    for ep in range(1, EPOCHS + 1):
        epoch_start = time.time()
        for phase, loader in (("train", tr_loader), ("val", va_loader)):
            if phase == "train":
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            total_cnt = 0
            cat_total = {cat: 0.0 for cat in categories}
            cat_cnt = {cat: 0 for cat in categories}

            data_times = []
            fwd_times = []
            loss_times = []
            bwd_times = [] if phase == "train" else None

            for batch_idx, (xyz, vm_gt, K, labels) in enumerate(loader):
                t0 = time.time()
                xyz = xyz.to(device)
                vm_gt = vm_gt.to(device)
                K = K.to(device)
                labels = labels.to(device)
                data_times.append(time.time() - t0)

                if phase == "train":
                    opt.zero_grad()

                t1 = time.time()
                mu_pred, kappa_pred, w_pred = model(xyz)
                fwd_times.append(time.time() - t1)

                # 检查预测输出
                if not torch.isfinite(mu_pred).all():
                    print("Non-finite mu_pred in batch", batch_idx, mu_pred)
                if not torch.isfinite(kappa_pred).all():
                    print("Non-finite kappa_pred in batch", batch_idx, kappa_pred)
                if not torch.isfinite(w_pred).all():
                    print("Non-finite w_pred in batch", batch_idx, w_pred)

                t2 = time.time()
                loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, vm_gt, K)
                loss = loss_vec.mean()
                loss_times.append(time.time() - t2)

                # 检查 loss_vec / loss
                if not torch.isfinite(loss_vec).all():
                    print("Non-finite loss_vec at batch", batch_idx, loss_vec)
                if not torch.isfinite(loss):
                    print("Non-finite loss at batch", batch_idx, loss)

                if phase == "train":
                    t3 = time.time()
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    bwd_times.append(time.time() - t3)

                    # 检查梯度
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            print("Non-finite grad in", name, param.grad)

                total_loss += loss_vec.sum().item()
                total_cnt += xyz.size(0)

                for cid, cat in enumerate(categories):
                    mask = (labels == cid)
                    if mask.any():
                        cat_total[cat] += loss_vec[mask].sum().item()
                        cat_cnt[cat] += mask.sum().item()

            # 结束 batch loop
            # …（保持你之前打印、记录等逻辑）…

            # 简略打印 epoch 内的一些信息
            print(f"After epoch {ep} phase {phase}, avg_loss = {total_loss / max(total_cnt,1):.6f}")

        # epoch 结束后的逻辑（保存、验证、打印等）…

    # 训练／测试结束部分同之前版本…

if __name__ == "__main__":
    main()

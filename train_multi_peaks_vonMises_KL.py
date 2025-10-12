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
RES = Path("/home/pablo/ForwardNet/results/multi_peak_vonMises_KL_1012_1")
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
torch.autograd.set_detect_anomaly(False)

def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    kappa_p = torch.clamp(kappa_p, 1e-6, 500.0)
    kappa_q = torch.clamp(kappa_q, 1e-6, 500.0)

    i0_p = torch.special.i0(kappa_p)
    i1_p = torch.special.i1(kappa_p)
    i0_q = torch.special.i0(kappa_q)

    A_p = i1_p / i0_p

    delta = mu_p - mu_q
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    KL = torch.log(i0_q / i0_p) + A_p * (kappa_p - kappa_q * torch.cos(delta))
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

        cost = torch.zeros((K, K), device=device)
        for i in range(K):
            for j in range(K):
                cost[i, j] = kl_von_mises(μp[i], κp[i], μg[j], κg[j])

        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        cost_np = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost_np)

        matched_ws = wp[row]
        ws_sum = torch.sum(matched_ws) + 1e-8
        loss_bc = torch.sum(matched_ws * cost[row, col]) / ws_sum
        loss_vec[b] = loss_bc
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
            raise FileNotFoundError(f"PLY not found: {ply_path}, for GT: {txt}")
        samples.append((str(ply_path), str(txt), category))

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

            data_to_device_times = []
            forward_times = []
            loss_times = []
            backward_times = [] if phase == "train" else None

            for xyz, vm_gt, K, labels in loader:
                data_start = time.time()
                xyz = xyz.to(device)
                vm_gt = vm_gt.to(device)
                K = K.to(device)
                labels = labels.to(device)
                data_to_device_times.append(time.time() - data_start)

                if phase == "train":
                    opt.zero_grad()

                fwd_start = time.time()
                mu_pred, kappa_pred, w_pred = model(xyz)
                forward_times.append(time.time() - fwd_start)

                loss_start = time.time()
                loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, vm_gt, K)
                loss = loss_vec.mean()
                loss_times.append(time.time() - loss_start)

                if phase == "train":
                    bwd_start = time.time()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    backward_times.append(time.time() - bwd_start)

                total_loss += loss_vec.sum().item()
                total_cnt += xyz.size(0)

                for cid, cat in enumerate(categories):
                    mask = (labels == cid)
                    if mask.any():
                        cat_total[cat] += loss_vec[mask].sum().item()
                        cat_cnt[cat] += mask.sum().item()

            phase_time = time.time() - (epoch_start + sum(epoch_times[:0]))
            avg_data = np.mean(data_to_device_times) if data_to_device_times else 0.0
            avg_fwd = np.mean(forward_times) if forward_times else 0.0
            avg_loss = np.mean(loss_times) if loss_times else 0.0
            avg_bwd = np.mean(backward_times) if (phase == "train" and backward_times) else 0.0

            avg_total = total_loss / max(total_cnt, 1)
            hist["total"][phase].append(avg_total)
            for cat in categories:
                if cat_cnt[cat] > 0:
                    hist[cat][phase].append(cat_total[cat] / cat_cnt[cat])
                else:
                    hist[cat][phase].append(float("nan"))

            if phase == "val" and avg_total < best_val:
                best_val = avg_total
                best_state = model.state_dict()
                best_val_epoch = ep

            if phase == "train":
                train_phase_time = phase_time
                train_avg_data, train_avg_fwd, train_avg_loss, train_avg_bwd = avg_data, avg_fwd, avg_loss, avg_bwd
            else:
                val_phase_time = phase_time
                val_avg_data, val_avg_fwd, val_avg_loss = avg_data, avg_fwd, avg_loss

        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times) if len(epoch_times) > 1 else epoch_time
        remaining_epochs = EPOCHS - ep
        eta = remaining_epochs * avg_epoch_time
        eta_str = f" | ETA: {eta/60:.1f}m" if remaining_epochs > 0 else ""

        train_detail = f"(avg/batch: data={train_avg_data:.1f}s fwd={train_avg_fwd:.1f}s loss={train_avg_loss:.1f}s bwd={train_avg_bwd:.1f}s)"
        val_detail = f"(avg/batch: data={val_avg_data:.1f}s fwd={val_avg_fwd:.1f}s loss={val_avg_loss:.1f}s)"
        print(f"Ep {ep:03}/{EPOCHS} Train {hist['total']['train'][-1]:.4f} Val {hist['total']['val'][-1]:.4f} | "
              f"Train: {train_phase_time:.1f}s {train_detail} | Val: {val_phase_time:.1f}s {val_detail} | "
              f"Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}m{eta_str}")

    print(f"Training completed in total {total_time/60:.1f} minutes.")
    if best_state is not None:
        torch.save(best_state, RES / "mvM_best.pth")

    xs = list(range(1, EPOCHS + 1))
    for cat in categories:
        cat_safe = _sanitize(cat)
        plot_label_curve(xs, hist[cat]["train"], hist[cat]["val"], cat, FIGS / f"loss_{cat_safe}.png")
    plot_total_curve(xs, hist["total"]["train"], hist["total"]["val"], FIGS / "loss_total.png")
    ys_dict = {"total": (hist["total"]["train"], hist["total"]["val"])}
    for cat in categories:
        ys_dict[cat] = (hist[cat]["train"], hist[cat]["val"])
    plot_curve(xs, ys_dict, "Multi-Peak von Mises KL Loss", FIGS / "loss_overview.png")

    model.load_state_dict(best_state)
    model.eval()
    tot_sum = 0.0
    tot_cnt = 0
    with torch.no_grad():
        for xyz, vm_gt, K, _ in te_loader:
            xyz = xyz.to(device)
            vm_gt = vm_gt.to(device)
            K = K.to(device)
            mu_pred, kappa_pred, w_pred = model(xyz)
            loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, vm_gt, K)
            tot_sum += loss_vec.sum().item()
            tot_cnt += xyz.size(0)
    test_kl = tot_sum / max(tot_cnt, 1)
    print(f"Test KL = {test_kl:.6f}")
    write_summary_txt(RES / "results.txt", categories, hist, test_kl=test_kl, best_val_epoch=best_val_epoch)

if __name__ == "__main__":
    main()

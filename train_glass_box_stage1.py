#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glass Box 专用 Stage-1 训练脚本：
- 仅使用 glass_box 数据（4 个对称正面）
- PointNet++ backbone + MvM head，但 loss 固定 κ 与权重，专注学习 4 个 μ
- 输出训练/验证曲线与简单极坐标可视化
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from dataloader_multi_peak_vonMises import PointCloudDatasetMvM
from models.pointnet_pp_mvM import PointNetPPMvM

DATA_ROOT = Path("/home/pablo/ForwardNet/data")
GT_DIR = DATA_ROOT / "MN40_multi_peak_vM_gt" / "glass_box"
PLY_DIR = DATA_ROOT / "full_mn40_normal_resampled_2d_rotated_ply" / "glass_box"
DEFAULT_RES = Path("/home/pablo/ForwardNet/results/glass_box_stage1")


def parse_args():
    parser = argparse.ArgumentParser(description="Glass Box 4-peak Stage-1 training (fixed κ, uniform weight)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="若设置，仅使用前 N 个样本（可用于过拟合测试）")
    parser.add_argument("--const-kappa", type=float, default=8.0, help="Stage-1 中用于预测分布的固定 κ")
    parser.add_argument("--loss-scale", type=float, default=1.0, help="匹配后的 KL 将乘以该系数，便于验证损失缩放假设")
    parser.add_argument("--results", type=Path, default=DEFAULT_RES, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(limit=None):
    if not GT_DIR.exists():
        raise RuntimeError(f"GT_DIR 不存在: {GT_DIR}")
    if not PLY_DIR.exists():
        raise RuntimeError(f"PLY_DIR 不存在: {PLY_DIR}")
    txts = sorted(GT_DIR.glob("*_multi_peak_vM_gt.txt"))
    if len(txts) == 0:
        raise RuntimeError(f"{GT_DIR} 下没有 *_multi_peak_vM_gt.txt")
    if limit is not None:
        txts = txts[:limit]
    samples = []
    for txt in txts:
        base = txt.stem.replace("_multi_peak_vM_gt", "")
        ply = PLY_DIR / f"{base}.ply"
        if not ply.exists():
            raise FileNotFoundError(f"PLY 缺失: {ply}, 对应 {txt}")
        samples.append((str(ply), str(txt), "glass_box"))
    return samples


def split_samples(samples, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(samples)
    n_total = len(samples)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    return train, val, test


def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    kappa_p = torch.clamp(kappa_p, 1e-6, 500.0)
    kappa_q = torch.clamp(kappa_q, 1e-6, 500.0)
    i0_p = torch.special.i0(kappa_p)
    i1_p = torch.special.i1(kappa_p)
    i0_q = torch.special.i0(kappa_q)
    A_p = i1_p / i0_p
    delta = mu_p - mu_q
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return torch.log(i0_q / i0_p) + A_p * (kappa_p - kappa_q * torch.cos(delta))


def stage1_loss(mu_pred, vm_gt, K_gt, const_kappa, loss_scale=1.0):
    device = mu_pred.device
    B = mu_pred.size(0)
    losses = torch.zeros(B, device=device)
    const_kappa = float(const_kappa)
    for b in range(B):
        K = int(K_gt[b].item())
        if K <= 0:
            continue
        if K != 4:
            raise ValueError(f"Stage-1 期望 K=4，实际 {K}")
        mu_p = mu_pred[b, :K]
        mu_g = vm_gt[b, :K, 0]
        kappa_g = vm_gt[b, :K, 1]
        kappa_p = torch.full_like(mu_p, const_kappa)
        cost = torch.zeros((K, K), device=device)
        for i in range(K):
            for j in range(K):
                cost[i, j] = kl_von_mises(mu_p[i], kappa_p[i], mu_g[j], kappa_g[j])
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        losses[b] = cost[row, col].mean() * loss_scale
    return losses


def viz_polar(mu_pred, mu_gt, save_path):
    theta_pred = mu_pred.detach().cpu().numpy()
    theta_gt = mu_gt.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.scatter(theta_gt, np.ones_like(theta_gt), c="tab:blue", label="GT", s=60)
    ax.scatter(theta_pred, np.ones_like(theta_pred) * 0.7, c="tab:orange", label="Pred", s=60, marker="x")
    ax.set_title("Glass Box Peaks (GT vs Pred)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_epoch(model, loader, optimizer, phase, const_kappa, loss_scale, device):
    if phase == "train":
        model.train()
    else:
        model.eval()
    total = 0.0
    count = 0
    for xyz, vm_gt, K, _ in loader:
        xyz = xyz.to(device)
        vm_gt = vm_gt.to(device)
        K = K.to(device)
        if phase == "train":
            optimizer.zero_grad()
        mu_pred, _, _ = model(xyz)
        loss_vec = stage1_loss(mu_pred, vm_gt, K, const_kappa, loss_scale=loss_scale)
        loss = loss_vec.mean()
        if phase == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss_vec.sum().item()
        count += loss_vec.numel()
    avg = total / max(count, 1)
    return avg


def evaluate(model, loader, const_kappa, loss_scale, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xyz, vm_gt, K, _ in loader:
            xyz = xyz.to(device)
            vm_gt = vm_gt.to(device)
            K = K.to(device)
            mu_pred, _, _ = model(xyz)
            loss_vec = stage1_loss(mu_pred, vm_gt, K, const_kappa, loss_scale=loss_scale)
            total += loss_vec.sum().item()
            count += loss_vec.numel()
    return total / max(count, 1)


def visualize_best(model, loader, out_dir, const_kappa, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (xyz, vm_gt, K, _) in enumerate(loader):
            xyz = xyz.to(device)
            vm_gt = vm_gt.to(device)
            mu_pred, _, _ = model(xyz)
            for b in range(min(xyz.size(0), 4)):
                if int(K[b].item()) != 4:
                    continue
                save_path = out_dir / f"sample_{idx:02d}_b{b}.png"
                viz_polar(mu_pred[b, :4], vm_gt[b, :4, 0], save_path)
            break


def plot_loss(xs, train, val, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(xs, train, label="Train")
    plt.plot(xs, val, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL (uniform-κ)")
    plt.title("Glass Box Stage-1 Loss")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_samples(limit=args.limit)
    train_s, val_s, test_s = split_samples(samples, args.val_ratio, args.test_ratio)

    label_map = {"glass_box": 0}
    train_ds = PointCloudDatasetMvM(train_s, args.num_points, max_K=4, label_map=label_map)
    val_ds = PointCloudDatasetMvM(val_s, args.num_points, max_K=4, label_map=label_map)
    test_ds = PointCloudDatasetMvM(test_s, args.num_points, max_K=4, label_map=label_map)

    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    args.results.mkdir(parents=True, exist_ok=True)
    figs_dir = args.results / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    model = PointNetPPMvM(max_K=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, "train", args.const_kappa, args.loss_scale, device)
        val_loss = run_epoch(model, val_loader, optimizer, "val", args.const_kappa, args.loss_scale, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"[Ep {epoch:03d}] Train {train_loss:.4f} | Val {val_loss:.4f} | Best {best_val:.4f}")

    if best_state is not None:
        torch.save(best_state, args.results / "best_stage1.pth")
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, args.const_kappa, args.loss_scale, device)
    print(f"Test KL (uniform κ) = {test_loss:.4f}")

    xs = list(range(1, args.epochs + 1))
    plot_loss(xs, history["train"], history["val"], args.results / "loss_curve.png")
    visualize_best(model, val_loader, figs_dir, args.const_kappa, device)

    summary = args.results / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("=== Glass Box Stage-1 ===\n")
        f.write(f"Train samples: {len(train_ds)}\nVal samples: {len(val_ds)}\nTest samples: {len(test_ds)}\n")
        f.write(f"Best Val KL: {best_val:.6f}\n")
        f.write(f"Test KL: {test_loss:.6f}\n")
        f.write(f"Const κ: {args.const_kappa}\n")
        f.write(f"Loss scale: {args.loss_scale}\n")
        f.write(f"Limit samples: {args.limit}\n")


if __name__ == "__main__":
    main()

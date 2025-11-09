#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练PointNet++ + MvM模型在glassbox类别上（带12旋转增强）

详细说明：
- 模型/方法: PointNet++ backbone + MvM预测头（K=4峰，固定）
- 数据集: ModelNet40 glassbox类别 (271样本，train:val:test = 217:54:271)
- 数据增强: 12旋转增强（30°间隔：0°, 30°, ..., 330°）+ 点云抖动
- 训练策略: KL散度loss + Hungarian匹配，Adam优化器，学习率衰减
- 输出: 模型checkpoints、训练日志、可视化（极坐标图）

核心特性：
1. 预设角度初始化 [0°, 90°, 180°, 270°] 打破对称性
2. 每10个epoch保存可视化，追踪训练进度
3. Best model保存（基于验证集loss）

使用方法：
    python train_pointnetpp_mvm_glassbox_augmented.py

输出位置：
    results/glassbox_only_YYYYMMDD_HHMMSS/
    ├── checkpoints/best_model.pth
    ├── figs/predictions_epoch_*.png
    └── (训练日志在终端输出，可用tee重定向)

作者: Claude
创建日期: 2025-11-09
最后修改: 2025-11-09
关联文档: experiment_20251109_init_fix_results.md
实验ID: exp_20251109_init_fix
"""
import time
import random
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader_glassbox_augmented import GlassBoxDatasetAugmented
from models.pointnet_pp_mvM import PointNetPPMvM


# ============ 配置参数 ============
ROOT = Path("/home/pablo/ForwardNet-claude/data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")

# 实验命名
EXP_NAME = f"glassbox_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RES = Path(f"/home/pablo/ForwardNet-claude/results/{EXP_NAME}")
RES.mkdir(parents=True, exist_ok=True)
FIGS = RES / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RES / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 超参数
NUM_POINTS = 10_000
BATCH = 8  # 降低batch size（数据量少）
EPOCHS = 50  # 实验1：先跑50 epoch观察效果
LR = 5e-4  # 降低学习率以稳定训练
SEED = 42

# 数据增强配置
ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12个角度
APPLY_JITTER = True  # 训练时添加抖动
JITTER_STD = 0.01

# 随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")

# 保存配置
config_txt = RES / "config.txt"
with open(config_txt, "w") as f:
    f.write(f"Experiment: {EXP_NAME}\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Device: {device}\n\n")
    f.write(f"NUM_POINTS: {NUM_POINTS}\n")
    f.write(f"BATCH: {BATCH}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"LR: {LR}\n")
    f.write(f"SEED: {SEED}\n\n")
    f.write(f"ROTATION_ANGLES: {ROTATION_ANGLES}\n")
    f.write(f"APPLY_JITTER: {APPLY_JITTER}\n")
    f.write(f"JITTER_STD: {JITTER_STD}\n")


# ============ Loss函数 ============
def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    """计算两个von Mises分布的KL散度"""
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


def match_loss(mu_pred, kappa_pred, w_pred, vm_gt, K_gt):
    """
    基于匈牙利算法的匹配loss

    Args:
        mu_pred: (B, max_K) 预测的mu
        kappa_pred: (B, max_K) 预测的kappa
        w_pred: (B, max_K) 预测的weight
        vm_gt: (B, max_K, 3) GT的[mu, kappa, weight]
        K_gt: (B,) 每个样本的实际峰数量

    Returns:
        loss_vec: (B,) 每个样本的loss
    """
    B = mu_pred.size(0)
    loss_vec = torch.zeros(B, device=device)

    for b in range(B):
        K = int(K_gt[b].item())
        if K <= 0:
            loss_vec[b] = 0.0
            continue

        # 预测值
        μp = mu_pred[b, :K]
        κp = kappa_pred[b, :K]
        wp = w_pred[b, :K]

        # GT值
        μg = vm_gt[b, :K, 0]
        κg = vm_gt[b, :K, 1]

        # 计算cost矩阵 (K, K)
        cost = torch.zeros((K, K), device=device)
        for i in range(K):
            for j in range(K):
                cost[i, j] = kl_von_mises(μp[i], κp[i], μg[j], κg[j])

        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

        # 匈牙利匹配
        cost_np = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost_np)

        # 计算加权loss
        matched_ws = wp[row]
        ws_sum = torch.sum(matched_ws) + 1e-8
        loss_bc = torch.sum(matched_ws * cost[row, col]) / ws_sum

        loss_vec[b] = loss_bc

    return loss_vec


# ============ 可视化函数 ============
def plot_loss_curve(xs, train_vals, val_vals, out_path, title="GlassBox Training Loss"):
    """绘制训练/验证loss曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(xs, train_vals, label="Train", linewidth=2)
    plt.plot(xs, val_vals, "--", label="Val", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("KL Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_prediction(model, dataset, num_samples=4, out_path=None):
    """
    可视化模型预测的MvM分布（极坐标图）

    Args:
        model: 训练好的模型
        dataset: 数据集
        num_samples: 可视化的样本数量
        out_path: 保存路径
    """
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= 4:
                break

            xyz, vm_gt, K, angle_deg = dataset[idx]
            xyz = xyz.unsqueeze(0).to(device)

            # 预测
            mu_pred, kappa_pred, w_pred = model(xyz)
            mu_pred = mu_pred[0].cpu().numpy()
            kappa_pred = kappa_pred[0].cpu().numpy()
            w_pred = w_pred[0].cpu().numpy()

            # GT
            K_val = int(K) if not isinstance(K, torch.Tensor) else int(K.item())
            mu_gt = vm_gt[:K_val, 0].numpy()
            kappa_gt = vm_gt[:K_val, 1].numpy()
            w_gt = vm_gt[:K_val, 2].numpy()

            # 绘制极坐标图
            ax = axes[i]
            theta = np.linspace(0, 2 * np.pi, 360)

            # 计算GT分布
            p_gt = np.zeros_like(theta)
            for k in range(K_val):
                p_gt += w_gt[k] * np.exp(kappa_gt[k] * np.cos(theta - mu_gt[k]))
            p_gt = p_gt / (p_gt.sum() + 1e-8)

            # 计算预测分布
            p_pred = np.zeros_like(theta)
            for k in range(K_val):
                p_pred += w_pred[k] * np.exp(kappa_pred[k] * np.cos(theta - mu_pred[k]))
            p_pred = p_pred / (p_pred.sum() + 1e-8)

            # 绘图
            ax.plot(theta, p_gt, 'b-', linewidth=2, label='GT', alpha=0.7)
            ax.plot(theta, p_pred, 'r--', linewidth=2, label='Pred', alpha=0.7)

            # 标记峰位置
            for k in range(K_val):
                ax.plot([mu_gt[k], mu_gt[k]], [0, w_gt[k]], 'b-', linewidth=1, alpha=0.5)
                ax.plot([mu_pred[k], mu_pred[k]], [0, w_pred[k]], 'r--', linewidth=1, alpha=0.5)

            ax.set_ylim(0, max(p_gt.max(), p_pred.max()) * 1.2)
            ax.set_title(f"Sample {idx} (rot={angle_deg:.0f}°)", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()


# ============ 主训练函数 ============
def main():
    print(f"\n{'='*60}")
    print(f"  GlassBox-Only Training - 4-Peak von Mises Mixture")
    print(f"{'='*60}\n")

    # 检查数据路径
    if not ROOT.exists():
        raise RuntimeError(f"GT root not exists: {ROOT}")
    if not PLY_ROOT.exists():
        raise RuntimeError(f"PLY root not exists: {PLY_ROOT}")

    # 收集glassbox样本
    gt_txts = list(ROOT.glob("*_multi_peak_vM_gt.txt"))
    if len(gt_txts) == 0:
        raise RuntimeError(f"No GT files found in {ROOT}")

    samples = []
    for txt in gt_txts:
        base = txt.stem.replace("_multi_peak_vM_gt", "")
        ply_path = PLY_ROOT / (base + ".ply")
        if not ply_path.exists():
            print(f"[Warning] PLY not found for {txt.name}, skipping...")
            continue
        samples.append((str(ply_path), str(txt), "glass_box"))

    print(f"[Data] Found {len(samples)} glassbox samples")

    # 划分训练/验证/测试集 (7:2:1)
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"[Split] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # 创建数据集（训练集用增强，验证/测试集不增强）
    train_ds = GlassBoxDatasetAugmented(
        train_samples,
        NUM_POINTS,
        max_K=4,
        rotation_angles=ROTATION_ANGLES,
        apply_jitter=APPLY_JITTER,
        jitter_std=JITTER_STD
    )

    val_ds = GlassBoxDatasetAugmented(
        val_samples,
        NUM_POINTS,
        max_K=4,
        rotation_angles=[0],  # 验证集不增强
        apply_jitter=False
    )

    test_ds = GlassBoxDatasetAugmented(
        test_samples,
        NUM_POINTS,
        max_K=4,
        rotation_angles=[0],
        apply_jitter=False
    )

    # DataLoader
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Dataset] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 初始化模型
    model = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n[Training] Starting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_cnt = 0

        for xyz, vm_gt, K, _ in train_loader:
            xyz = xyz.to(device)
            vm_gt = vm_gt.to(device)
            K = K.to(device)

            optimizer.zero_grad()

            # 前向传播
            mu_pred, kappa_pred, w_pred = model(xyz)

            # 计算loss
            loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, K)
            loss = loss_vec.mean()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_vec.sum().item()
            train_cnt += xyz.size(0)

        avg_train_loss = train_loss / max(train_cnt, 1)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_cnt = 0

        with torch.no_grad():
            for xyz, vm_gt, K, _ in val_loader:
                xyz = xyz.to(device)
                vm_gt = vm_gt.to(device)
                K = K.to(device)

                mu_pred, kappa_pred, w_pred = model(xyz)
                loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, K)

                val_loss += loss_vec.sum().item()
                val_cnt += xyz.size(0)

        avg_val_loss = val_loss / max(val_cnt, 1)

        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")

        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s | "
              f"Best: {best_val_loss:.4f}@{best_epoch}")

        # 定期保存checkpoint和可视化
        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), CKPT_DIR / f"model_epoch_{epoch:03d}.pth")

            # 可视化预测结果
            vis_path = FIGS / f"predictions_epoch_{epoch:03d}.png"
            visualize_prediction(model, val_ds, num_samples=4, out_path=vis_path)
            print(f"  -> Saved visualization to {vis_path.name}")

    print(f"\n[Training] Completed! Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}\n")

    # ========== 测试阶段 ==========
    print("[Testing] Evaluating on test set...")
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth"))
    model.eval()

    test_loss = 0.0
    test_cnt = 0

    with torch.no_grad():
        for xyz, vm_gt, K, _ in test_loader:
            xyz = xyz.to(device)
            vm_gt = vm_gt.to(device)
            K = K.to(device)

            mu_pred, kappa_pred, w_pred = model(xyz)
            loss_vec = match_loss(mu_pred, kappa_pred, w_pred, vm_gt, K)

            test_loss += loss_vec.sum().item()
            test_cnt += xyz.size(0)

    avg_test_loss = test_loss / max(test_cnt, 1)
    print(f"[Test] Loss: {avg_test_loss:.4f}\n")

    # ========== 保存结果 ==========
    # 绘制loss曲线
    xs = list(range(1, EPOCHS + 1))
    plot_loss_curve(xs, history["train_loss"], history["val_loss"],
                    FIGS / "loss_curve.png", title="GlassBox Training Loss")

    # 保存最终可视化
    visualize_prediction(model, test_ds, num_samples=4,
                        out_path=FIGS / "final_predictions.png")

    # 保存结果摘要
    with open(RES / "results.txt", "w") as f:
        f.write(f"=== GlassBox Training Results ===\n\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n")
        f.write(f"Test Loss: {avg_test_loss:.6f}\n\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.6f}\n")

    print(f"[Results] All outputs saved to: {RES}\n")
    print("="*60)


if __name__ == "__main__":
    main()
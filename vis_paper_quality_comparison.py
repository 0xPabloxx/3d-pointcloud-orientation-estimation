#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文级可视化：高质量实验对比图

生成适合论文使用的对比可视化：
1. Loss曲线对比（并排）
2. 极坐标预测对比（多样本）
3. 参数分布对比（μ, κ, weight）
4. 误差分析可视化

作者: Claude
创建日期: 2025-11-13
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from dataloader_glassbox_augmented import GlassBoxDatasetAugmented
from models.pointnet_pp_mvM import PointNetPPMvM


# ============ 配置 ============
ROOT = Path("/home/pablo/ForwardNet-claude/data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")

MODEL1_PATH = Path("results/glassbox_only_20251109_183051/checkpoints/best_model.pth")
MODEL2_PATH = Path("results/glassbox_no_augment_20251109_201200/checkpoints/best_model.pth")

LOSS_CURVE1 = Path("results/glassbox_only_20251109_183051/figs/loss_curve.png")
LOSS_CURVE2 = Path("results/glassbox_no_augment_20251109_201200/figs/loss_curve.png")

OUTPUT_DIR = Path("results/paper_quality_visualizations_20251113")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_POINTS = 10_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置论文级绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


print(f"[Device] {device}")
print(f"[Output] {OUTPUT_DIR}")


# ============ 图1: Loss曲线对比（并排） ============
def create_loss_curve_comparison():
    """创建并排的loss曲线对比图"""
    print("\n[Creating] Loss curve comparison...")

    if not LOSS_CURVE1.exists() or not LOSS_CURVE2.exists():
        print("[Warning] Loss curve images not found, skipping...")
        return

    # 读取现有的loss曲线图
    img1 = Image.open(LOSS_CURVE1)
    img2 = Image.open(LOSS_CURVE2)

    # 创建并排对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Experiment 1: With Data Augmentation\nBest Val Loss: 0.001719 @ Epoch 45',
                      fontsize=13, fontweight='bold', pad=15)

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Experiment 2: Without Data Augmentation\nBest Val Loss: 0.006047 @ Epoch 46',
                      fontsize=13, fontweight='bold', pad=15)

    plt.suptitle('Training Loss Comparison: Impact of Data Augmentation',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig1_loss_curve_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[Saved] {out_path}")


# ============ 图2: 极坐标预测对比（精选样本） ============
@torch.no_grad()
def create_polar_comparison(model1, model2, dataset, num_samples=9):
    """创建3x3的极坐标预测对比网格"""
    print("\n[Creating] High-quality polar plot comparison...")

    model1.eval()
    model2.eval()

    # 精选有代表性的样本
    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(num_samples, 3, figure=fig, hspace=0.35, wspace=0.25)

    for i, idx in enumerate(indices):
        xyz, vm_gt, K, angle_deg = dataset[idx]
        xyz = xyz.unsqueeze(0).to(device)

        # 预测
        mu_pred1, kappa_pred1, w_pred1 = model1(xyz)
        mu_pred2, kappa_pred2, w_pred2 = model2(xyz)

        mu_pred1 = mu_pred1[0].cpu().numpy()
        kappa_pred1 = kappa_pred1[0].cpu().numpy()
        w_pred1 = w_pred1[0].cpu().numpy()

        mu_pred2 = mu_pred2[0].cpu().numpy()
        kappa_pred2 = kappa_pred2[0].cpu().numpy()
        w_pred2 = w_pred2[0].cpu().numpy()

        # GT
        K_val = int(K) if not isinstance(K, torch.Tensor) else int(K.item())
        mu_gt = vm_gt[:K_val, 0].numpy()
        kappa_gt = vm_gt[:K_val, 1].numpy()
        w_gt = vm_gt[:K_val, 2].numpy()

        theta = np.linspace(0, 2 * np.pi, 360)

        # 计算分布
        p_gt = np.zeros_like(theta)
        for k in range(K_val):
            p_gt += w_gt[k] * np.exp(kappa_gt[k] * np.cos(theta - mu_gt[k]))
        p_gt = p_gt / (p_gt.sum() + 1e-8)

        p_pred1 = np.zeros_like(theta)
        for k in range(K_val):
            p_pred1 += w_pred1[k] * np.exp(kappa_pred1[k] * np.cos(theta - mu_pred1[k]))
        p_pred1 = p_pred1 / (p_pred1.sum() + 1e-8)

        p_pred2 = np.zeros_like(theta)
        for k in range(K_val):
            p_pred2 += w_pred2[k] * np.exp(kappa_pred2[k] * np.cos(theta - mu_pred2[k]))
        p_pred2 = p_pred2 / (p_pred2.sum() + 1e-8)

        y_max = max(p_gt.max(), p_pred1.max(), p_pred2.max()) * 1.15

        # GT
        ax_gt = fig.add_subplot(gs[i, 0], projection='polar')
        ax_gt.plot(theta, p_gt, 'b-', linewidth=2.5, label='GT')
        for k in range(K_val):
            ax_gt.plot([mu_gt[k], mu_gt[k]], [0, w_gt[k]], 'b-', linewidth=2, alpha=0.7)
            ax_gt.scatter([mu_gt[k]], [w_gt[k]], c='blue', s=80, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.set_ylim(0, y_max)
        ax_gt.set_title(f'Ground Truth' if i == 0 else '', fontsize=12, fontweight='bold', pad=10)
        ax_gt.grid(True, alpha=0.3, linestyle='--')

        # 实验1
        ax_pred1 = fig.add_subplot(gs[i, 1], projection='polar')
        ax_pred1.plot(theta, p_gt, 'b-', linewidth=1.5, alpha=0.25, label='GT')
        ax_pred1.plot(theta, p_pred1, 'r-', linewidth=2.5, label='Pred (Aug)')
        for k in range(K_val):
            ax_pred1.plot([mu_pred1[k], mu_pred1[k]], [0, w_pred1[k]], 'r-', linewidth=2, alpha=0.7)
            ax_pred1.scatter([mu_pred1[k]], [w_pred1[k]], c='red', s=80, zorder=5, edgecolors='white', linewidths=1)
        ax_pred1.set_ylim(0, y_max)
        ax_pred1.set_title(f'With Augmentation' if i == 0 else '', fontsize=12, fontweight='bold', pad=10)
        if i == 0:
            ax_pred1.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_pred1.grid(True, alpha=0.3, linestyle='--')

        # 实验2
        ax_pred2 = fig.add_subplot(gs[i, 2], projection='polar')
        ax_pred2.plot(theta, p_gt, 'b-', linewidth=1.5, alpha=0.25, label='GT')
        ax_pred2.plot(theta, p_pred2, 'g-', linewidth=2.5, label='Pred (No Aug)')
        for k in range(K_val):
            ax_pred2.plot([mu_pred2[k], mu_pred2[k]], [0, w_pred2[k]], 'g-', linewidth=2, alpha=0.7)
            ax_pred2.scatter([mu_pred2[k]], [w_pred2[k]], c='green', s=80, zorder=5, edgecolors='white', linewidths=1)
        ax_pred2.set_ylim(0, y_max)
        ax_pred2.set_title(f'Without Augmentation' if i == 0 else '', fontsize=12, fontweight='bold', pad=10)
        if i == 0:
            ax_pred2.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_pred2.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('von Mises Mixture Distribution Predictions: 9 Sample Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    out_path = OUTPUT_DIR / "fig2_polar_comparison_9samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[Saved] {out_path}")


# ============ 图3: 参数分布对比 ============
@torch.no_grad()
def create_parameter_distribution_comparison(model1, model2, dataset, num_samples=100):
    """对比两个模型预测的参数分布（μ, κ, weight）"""
    print("\n[Creating] Parameter distribution comparison...")

    model1.eval()
    model2.eval()

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # 收集参数
    mu_pred1_all = []
    kappa_pred1_all = []
    w_pred1_all = []

    mu_pred2_all = []
    kappa_pred2_all = []
    w_pred2_all = []

    mu_gt_all = []
    kappa_gt_all = []
    w_gt_all = []

    for idx in indices:
        xyz, vm_gt, K, _ = dataset[idx]
        xyz = xyz.unsqueeze(0).to(device)

        # 预测
        mu_pred1, kappa_pred1, w_pred1 = model1(xyz)
        mu_pred2, kappa_pred2, w_pred2 = model2(xyz)

        K_val = int(K) if not isinstance(K, torch.Tensor) else int(K.item())

        mu_pred1_all.extend(mu_pred1[0, :K_val].cpu().numpy())
        kappa_pred1_all.extend(kappa_pred1[0, :K_val].cpu().numpy())
        w_pred1_all.extend(w_pred1[0, :K_val].cpu().numpy())

        mu_pred2_all.extend(mu_pred2[0, :K_val].cpu().numpy())
        kappa_pred2_all.extend(kappa_pred2[0, :K_val].cpu().numpy())
        w_pred2_all.extend(w_pred2[0, :K_val].cpu().numpy())

        mu_gt_all.extend(vm_gt[:K_val, 0].numpy())
        kappa_gt_all.extend(vm_gt[:K_val, 1].numpy())
        w_gt_all.extend(vm_gt[:K_val, 2].numpy())

    # 转换为numpy数组
    mu_pred1_all = np.array(mu_pred1_all)
    kappa_pred1_all = np.array(kappa_pred1_all)
    w_pred1_all = np.array(w_pred1_all)

    mu_pred2_all = np.array(mu_pred2_all)
    kappa_pred2_all = np.array(kappa_pred2_all)
    w_pred2_all = np.array(w_pred2_all)

    mu_gt_all = np.array(mu_gt_all)
    kappa_gt_all = np.array(kappa_gt_all)
    w_gt_all = np.array(w_gt_all)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # μ 分布（极坐标直方图）
    ax = axes[0, 0]
    bins = np.linspace(0, 2*np.pi, 25)
    ax.hist(mu_gt_all, bins=bins, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(mu_pred1_all, bins=bins, alpha=0.6, label='Exp1 (Aug)', color='red', density=True)
    ax.set_xlabel('μ (radians)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Mean Direction (μ) Distribution - Exp1', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(mu_gt_all, bins=bins, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(mu_pred2_all, bins=bins, alpha=0.6, label='Exp2 (No Aug)', color='green', density=True)
    ax.set_xlabel('μ (radians)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Mean Direction (μ) Distribution - Exp2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # κ 分布
    ax = axes[1, 0]
    bins_kappa = np.linspace(0, max(kappa_gt_all.max(), kappa_pred1_all.max(), kappa_pred2_all.max()), 30)
    ax.hist(kappa_gt_all, bins=bins_kappa, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(kappa_pred1_all, bins=bins_kappa, alpha=0.6, label='Exp1 (Aug)', color='red', density=True)
    ax.set_xlabel('κ (concentration)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Concentration (κ) Distribution - Exp1', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(kappa_gt_all, bins=bins_kappa, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(kappa_pred2_all, bins=bins_kappa, alpha=0.6, label='Exp2 (No Aug)', color='green', density=True)
    ax.set_xlabel('κ (concentration)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Concentration (κ) Distribution - Exp2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Weight 分布
    ax = axes[0, 2]
    bins_w = np.linspace(0, 1, 25)
    ax.hist(w_gt_all, bins=bins_w, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(w_pred1_all, bins=bins_w, alpha=0.6, label='Exp1 (Aug)', color='red', density=True)
    ax.set_xlabel('Weight (π)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weight (π) Distribution - Exp1', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.hist(w_gt_all, bins=bins_w, alpha=0.4, label='GT', color='blue', density=True)
    ax.hist(w_pred2_all, bins=bins_w, alpha=0.6, label='Exp2 (No Aug)', color='green', density=True)
    ax.set_xlabel('Weight (π)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weight (π) Distribution - Exp2', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'MvM Parameter Distributions ({num_samples} samples × 4 peaks)',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig3_parameter_distribution_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[Saved] {out_path}")


# ============ 主函数 ============
def main():
    print(f"\n{'='*70}")
    print(f"  Paper-Quality Visualization Generation")
    print(f"{'='*70}\n")

    # 图1: Loss曲线对比
    create_loss_curve_comparison()

    # 加载模型和数据
    print("\n[Loading] Models and dataset...")

    # 收集测试样本
    gt_txts = list(ROOT.glob("*_multi_peak_vM_gt.txt"))
    samples = []
    for txt in gt_txts:
        base = txt.stem.replace("_multi_peak_vM_gt", "")
        ply_path = PLY_ROOT / (base + ".ply")
        if ply_path.exists():
            samples.append((str(ply_path), str(txt), "glass_box"))

    np.random.seed(42)
    np.random.shuffle(samples)
    test_samples = samples[:100]  # 使用前100个样本

    test_ds = GlassBoxDatasetAugmented(
        test_samples,
        NUM_POINTS,
        max_K=4,
        rotation_angles=[0],
        apply_jitter=False
    )

    # 加载模型
    model1 = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
    model1.load_state_dict(torch.load(MODEL1_PATH, map_location=device, weights_only=False))

    model2 = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
    model2.load_state_dict(torch.load(MODEL2_PATH, map_location=device, weights_only=False))

    # 图2: 极坐标对比
    create_polar_comparison(model1, model2, test_ds, num_samples=9)

    # 图3: 参数分布对比
    create_parameter_distribution_comparison(model1, model2, test_ds, num_samples=100)

    print(f"\n{'='*70}")
    print(f"  All visualizations generated successfully!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

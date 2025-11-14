#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证weight修复效果 - 只使用独立测试集
"""

import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
from models.pointnet_pp_mvM import PointNetPPMvM
from dataloader_glassbox_augmented import GlassBoxDatasetAugmented

# 设置随机种子与训练脚本一致
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 加载最佳模型
CKPT_PATH = Path("/home/pablo/ForwardNet-claude/results/glassbox_fixed_weight_20251114_130044/checkpoints/best_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("  Weight Distribution Verification - TEST SET ONLY")
print("="*60)

# 加载模型
model = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=2.0).to(device)
model.load_state_dict(torch.load(CKPT_PATH, weights_only=False))
model.eval()

# 使用与训练脚本完全相同的数据划分逻辑
ROOT = Path("/home/pablo/ForwardNet-claude/data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")

gt_txts = list(ROOT.glob("*_multi_peak_vM_gt.txt"))
samples = []
for txt in gt_txts:
    base = txt.stem.replace("_multi_peak_vM_gt", "")
    ply_path = PLY_ROOT / (base + ".ply")
    if ply_path.exists():
        samples.append((str(ply_path), str(txt), "glass_box"))

# 与训练脚本相同的划分（7:2:1）
random.shuffle(samples)
n_total = len(samples)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)

train_samples = samples[:n_train]
val_samples = samples[n_train:n_train + n_val]
test_samples = samples[n_train + n_val:]  # 只使用这部分！

print(f"\n[Data Split] Total: {n_total}")
print(f"  Train: {len(train_samples)} samples")
print(f"  Val:   {len(val_samples)} samples")
print(f"  Test:  {len(test_samples)} samples ← Using only this")

# 创建测试集（不增强）
test_ds = GlassBoxDatasetAugmented(
    test_samples,
    1024,
    max_K=4,
    rotation_angles=[0],
    apply_jitter=False
)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

print(f"\n[Model] Loaded from: {CKPT_PATH.name}")
print(f"[Model] temp = {model.temp}")

# 收集所有预测的weights和计算loss
all_weights = []
all_mu_pred = []
all_kappa_pred = []
all_mu_gt = []
all_kappa_gt = []
all_w_gt = []

from scipy.optimize import linear_sum_assignment

def compute_kl_loss_single(mu_p, kappa_p, w_p, mu_g, kappa_g, w_g):
    """计算单个样本的KL loss（使用Hungarian matching）"""
    K = len(mu_g)
    cost_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            # 简化的KL散度估计（只用μ和κ）
            mu_diff = abs(mu_p[i] - mu_g[j])
            mu_diff = min(mu_diff, 2*np.pi - mu_diff)  # circular distance
            kappa_diff = abs(kappa_p[i] - kappa_g[j])
            cost_matrix[i, j] = mu_diff + 0.1 * kappa_diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 匹配后的μ和κ差异
    mu_errors = []
    kappa_errors = []
    for i, j in zip(row_ind, col_ind):
        mu_diff = abs(mu_p[i] - mu_g[j])
        mu_diff = min(mu_diff, 2*np.pi - mu_diff)
        mu_errors.append(mu_diff)
        kappa_errors.append(abs(kappa_p[i] - kappa_g[j]))

    return np.mean(mu_errors), np.mean(kappa_errors), col_ind

with torch.no_grad():
    total_loss = 0.0
    n_samples = 0

    for xyz, vm_gt, K_gt, _ in test_loader:
        xyz = xyz.to(device)
        vm_gt = vm_gt.to(device)

        mu_pred, kappa_pred, w_pred = model(xyz)

        # 收集weights (B, K)
        all_weights.append(w_pred.cpu().numpy())

        # 收集参数用于分析
        batch_size = xyz.size(0)
        for b in range(batch_size):
            K = K_gt[b].item()
            mu_p = mu_pred[b, :K].cpu().numpy()
            kappa_p = kappa_pred[b, :K].cpu().numpy()
            w_p = w_pred[b, :K].cpu().numpy()

            mu_g = vm_gt[b, :K, 0].cpu().numpy()
            kappa_g = vm_gt[b, :K, 1].cpu().numpy()
            w_g = vm_gt[b, :K, 2].cpu().numpy()

            all_mu_pred.append(mu_p)
            all_kappa_pred.append(kappa_p)
            all_mu_gt.append(mu_g)
            all_kappa_gt.append(kappa_g)
            all_w_gt.append(w_g)

            n_samples += 1

# 合并所有批次
all_weights = np.concatenate(all_weights, axis=0)  # (N, 4)
N = all_weights.shape[0]

print(f"\n{'='*60}")
print(f"  Weight Distribution Analysis - TEST SET (N={N})")
print(f"{'='*60}")

# 统计分析
mean_weights = all_weights.mean(axis=0)
std_weights = all_weights.std(axis=0)
min_weights = all_weights.min(axis=0)
max_weights = all_weights.max(axis=0)

print(f"\n{'Statistic':<12} | {'Peak 1':>8} | {'Peak 2':>8} | {'Peak 3':>8} | {'Peak 4':>8}")
print(f"{'-'*60}")
print(f"{'Mean':<12} | {mean_weights[0]:8.4f} | {mean_weights[1]:8.4f} | {mean_weights[2]:8.4f} | {mean_weights[3]:8.4f}")
print(f"{'Std':<12} | {std_weights[0]:8.4f} | {std_weights[1]:8.4f} | {std_weights[2]:8.4f} | {std_weights[3]:8.4f}")
print(f"{'Min':<12} | {min_weights[0]:8.4f} | {min_weights[1]:8.4f} | {min_weights[2]:8.4f} | {min_weights[3]:8.4f}")
print(f"{'Max':<12} | {max_weights[0]:8.4f} | {max_weights[1]:8.4f} | {max_weights[2]:8.4f} | {max_weights[3]:8.4f}")

# 理想值对比
ideal = np.array([0.25, 0.25, 0.25, 0.25])
deviation = np.abs(mean_weights - ideal)
total_deviation = deviation.sum()

print(f"\n{'='*60}")
print(f"  Deviation from Ideal [0.25, 0.25, 0.25, 0.25]")
print(f"{'='*60}")
print(f"Per-peak deviation: {deviation}")
print(f"Total deviation: {total_deviation:.4f}")
print(f"Max deviation: {deviation.max():.4f}")

# 判断修复成功
if deviation.max() < 0.05:
    print(f"\n✅ Weight修复成功！所有峰的平均权重接近0.25")
elif deviation.max() < 0.10:
    print(f"\n⚠️  Weight有所改善但仍有偏差（最大偏差{deviation.max():.4f}）")
else:
    print(f"\n❌ Weight仍存在显著偏差（最大偏差{deviation.max():.4f}）")

# 检查样本级别的均匀性
weight_ratios = all_weights.max(axis=1) / (all_weights.min(axis=1) + 1e-6)
print(f"\n{'='*60}")
print(f"  Sample-level Weight Balance")
print(f"{'='*60}")
print(f"Max/Min weight ratio:")
print(f"  Mean: {weight_ratios.mean():.2f}")
print(f"  Median: {np.median(weight_ratios):.2f}")
print(f"  Std: {weight_ratios.std():.2f}")
print(f"  Min: {weight_ratios.min():.2f}")
print(f"  Max: {weight_ratios.max():.2f}")

if weight_ratios.mean() < 2.0:
    print(f"✅ 样本级别权重分布均匀")
elif weight_ratios.mean() < 5.0:
    print(f"⚠️  样本级别权重分布有轻微不均")
else:
    print(f"❌ 样本级别权重分布不均匀")

# 显示几个样本示例
print(f"\n{'='*60}")
print(f"  Sample Examples (first 5 from test set)")
print(f"{'='*60}")
for i in range(min(5, N)):
    w = all_weights[i]
    print(f"Sample {i+1}: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}]  (sum={w.sum():.3f})")

print(f"\n{'='*60}")
print(f"  Parameter Accuracy Analysis")
print(f"{'='*60}")

# 分析μ和κ的误差
mu_errors = []
kappa_errors = []
for i in range(N):
    mu_err, kappa_err, _ = compute_kl_loss_single(
        all_mu_pred[i], all_kappa_pred[i], all_weights[i],
        all_mu_gt[i], all_kappa_gt[i], all_w_gt[i]
    )
    mu_errors.append(mu_err)
    kappa_errors.append(kappa_err)

mu_errors = np.array(mu_errors)
kappa_errors = np.array(kappa_errors)

print(f"\nμ (angle) error:")
print(f"  Mean: {np.rad2deg(mu_errors.mean()):.2f}°")
print(f"  Median: {np.rad2deg(np.median(mu_errors)):.2f}°")
print(f"  Std: {np.rad2deg(mu_errors.std()):.2f}°")

print(f"\nκ (concentration) error:")
print(f"  Mean: {kappa_errors.mean():.3f}")
print(f"  Median: {np.median(kappa_errors):.3f}")
print(f"  Std: {kappa_errors.std():.3f}")

print(f"\n{'='*60}\n")

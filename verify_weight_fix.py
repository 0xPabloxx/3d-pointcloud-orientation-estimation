#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证weight修复效果 - 定量分析weight分布
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from models.pointnet_pp_mvM import PointNetPPMvM
from dataloader_glassbox_augmented import GlassBoxDatasetAugmented

# 加载最佳模型
CKPT_PATH = Path("/home/pablo/ForwardNet-claude/results/glassbox_fixed_weight_20251114_130044/checkpoints/best_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("  Weight Distribution Verification")
print("="*60)

# 加载模型
model = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=2.0).to(device)
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

# 加载所有glass_box数据（用于全面验证）
ROOT = Path("/home/pablo/ForwardNet-claude/data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")

# 收集所有glassbox样本
gt_txts = list(ROOT.glob("*_multi_peak_vM_gt.txt"))
samples = []
for txt in gt_txts:
    base = txt.stem.replace("_multi_peak_vM_gt", "")
    ply_path = PLY_ROOT / (base + ".ply")
    if ply_path.exists():
        samples.append((str(ply_path), str(txt), "glass_box"))

print(f"[Data] Found {len(samples)} glassbox samples")

# 使用所有样本验证（不增强）
test_ds = GlassBoxDatasetAugmented(
    samples,
    1024,
    max_K=4,
    rotation_angles=[0],
    apply_jitter=False
)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

print(f"\n[Model] Loaded from: {CKPT_PATH.name}")
print(f"[Model] temp = {model.temp}")
print(f"[Data] Test samples: {len(test_loader.dataset)}")

# 收集所有预测的weights
all_weights = []

with torch.no_grad():
    for xyz, vm_gt, K_gt, _ in test_loader:
        xyz = xyz.to(device)
        mu_pred, kappa_pred, w_pred = model(xyz)

        # 收集weights (B, K)
        all_weights.append(w_pred.cpu().numpy())

# 合并所有批次
all_weights = np.concatenate(all_weights, axis=0)  # (N, 4)
N = all_weights.shape[0]

print(f"\n{'='*60}")
print(f"  Weight Distribution Analysis (N={N} samples)")
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
if deviation.max() < 0.05:  # 每个峰偏差小于5%
    print(f"\n✅ Weight修复成功！所有峰的平均权重接近0.25")
elif deviation.max() < 0.10:
    print(f"\n⚠️  Weight有所改善但仍有偏差（最大偏差{deviation.max():.4f}）")
else:
    print(f"\n❌ Weight仍存在显著偏差（最大偏差{deviation.max():.4f}）")

# 检查样本级别的均匀性
# 计算每个样本的最大weight和最小weight的比值
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

if weight_ratios.mean() < 2.0:  # 平均比值小于2表示相对均匀
    print(f"✅ 样本级别权重分布均匀")
elif weight_ratios.mean() < 5.0:
    print(f"⚠️  样本级别权重分布有轻微不均")
else:
    print(f"❌ 样本级别权重分布不均匀")

# 显示几个样本示例
print(f"\n{'='*60}")
print(f"  Sample Examples")
print(f"{'='*60}")
for i in range(min(5, N)):
    w = all_weights[i]
    print(f"Sample {i+1}: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}]  (sum={w.sum():.3f})")

print(f"\n{'='*60}\n")

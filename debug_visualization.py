#!/usr/bin/env python3
"""
调试可视化：检查GT和预测是否真的有4个峰
"""
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from dataloader_glassbox_augmented import GlassBoxDatasetAugmented
from models.pointnet_pp_mvM import PointNetPPMvM

# 配置
ROOT = Path("data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")
MODEL_PATH = Path("results/glassbox_only_20251109_183051/checkpoints/best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 收集样本
gt_txts = list(ROOT.glob("*_multi_peak_vM_gt.txt"))
samples = []
for txt in gt_txts[:10]:  # 只取10个样本
    base = txt.stem.replace("_multi_peak_vM_gt", "")
    ply_path = PLY_ROOT / (base + ".ply")
    if ply_path.exists():
        samples.append((str(ply_path), str(txt), "glass_box"))

# 创建数据集
dataset = GlassBoxDatasetAugmented(samples, 10000, max_K=4, rotation_angles=[0], apply_jitter=False)

# 加载模型
model = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()

print(f"\n{'='*70}")
print(f"  Debugging Visualization: Checking GT and Predictions")
print(f"{'='*70}\n")

# 检查前3个样本
for idx in range(min(3, len(dataset))):
    xyz, vm_gt, K, angle_deg = dataset[idx]

    K_val = int(K) if not isinstance(K, torch.Tensor) else int(K.item())

    print(f"\n{'='*70}")
    print(f"Sample {idx}:")
    print(f"{'='*70}")
    print(f"K = {K_val}")

    # GT参数
    print(f"\nGround Truth Parameters:")
    for k in range(K_val):
        mu_rad = vm_gt[k, 0].item()
        mu_deg = np.rad2deg(mu_rad)
        kappa = vm_gt[k, 1].item()
        weight = vm_gt[k, 2].item()
        print(f"  Peak {k+1}: μ={mu_rad:7.3f} rad ({mu_deg:7.2f}°), κ={kappa:6.2f}, π={weight:.4f}")

    # 预测
    with torch.no_grad():
        xyz_batch = xyz.unsqueeze(0).to(device)
        mu_pred, kappa_pred, w_pred = model(xyz_batch)

        mu_pred = mu_pred[0].cpu().numpy()
        kappa_pred = kappa_pred[0].cpu().numpy()
        w_pred = w_pred[0].cpu().numpy()

    print(f"\nModel Predictions:")
    for k in range(K_val):
        mu_rad = mu_pred[k]
        mu_deg = np.rad2deg(mu_rad)
        kappa = kappa_pred[k]
        weight = w_pred[k]
        print(f"  Peak {k+1}: μ={mu_rad:7.3f} rad ({mu_deg:7.2f}°), κ={kappa:6.2f}, π={weight:.4f}")

    # 计算并绘制分布
    theta = np.linspace(0, 2*np.pi, 360)

    # GT分布
    p_gt = np.zeros_like(theta)
    for k in range(K_val):
        mu_k = vm_gt[k, 0].item()
        kappa_k = vm_gt[k, 1].item()
        weight_k = vm_gt[k, 2].item()
        p_gt += weight_k * np.exp(kappa_k * np.cos(theta - mu_k))

    # 预测分布
    p_pred = np.zeros_like(theta)
    for k in range(K_val):
        mu_k = mu_pred[k]
        kappa_k = kappa_pred[k]
        weight_k = w_pred[k]
        p_pred += weight_k * np.exp(kappa_k * np.cos(theta - mu_k))

    # 检查峰的数量
    from scipy.signal import find_peaks

    peaks_gt, _ = find_peaks(p_gt, height=p_gt.max() * 0.3)
    peaks_pred, _ = find_peaks(p_pred, height=p_pred.max() * 0.3)

    print(f"\n检测到的峰数量:")
    print(f"  GT分布: {len(peaks_gt)} 个峰")
    print(f"  预测分布: {len(peaks_pred)} 个峰")

    if len(peaks_gt) < 4:
        print(f"  ⚠️ 警告：GT分布检测到的峰少于4个！")
    if len(peaks_pred) < 4:
        print(f"  ⚠️ 警告：预测分布检测到的峰少于4个！")

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

    # GT
    ax = axes[0]
    ax.plot(theta, p_gt, 'b-', linewidth=2, label='Distribution')
    for k in range(K_val):
        mu_k = vm_gt[k, 0].item()
        weight_k = vm_gt[k, 2].item()
        ax.plot([mu_k, mu_k], [0, weight_k], 'r-', linewidth=3, alpha=0.7)
        ax.scatter([mu_k], [weight_k], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
        ax.text(mu_k, weight_k + 0.05, f'Peak{k+1}', ha='center', fontsize=9, color='red')
    ax.set_title(f'Ground Truth (K={K_val}, {len(peaks_gt)} peaks detected)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 预测
    ax = axes[1]
    ax.plot(theta, p_pred, 'g-', linewidth=2, label='Distribution')
    for k in range(K_val):
        mu_k = mu_pred[k]
        weight_k = w_pred[k]
        ax.plot([mu_k, mu_k], [0, weight_k], 'r-', linewidth=3, alpha=0.7)
        ax.scatter([mu_k], [weight_k], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
        ax.text(mu_k, weight_k + 0.05, f'Peak{k+1}', ha='center', fontsize=9, color='red')
    ax.set_title(f'Prediction (K={K_val}, {len(peaks_pred)} peaks detected)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'debug_sample_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  保存: debug_sample_{idx}.png")

print(f"\n{'='*70}")
print(f"  调试完成！检查生成的 debug_sample_*.png 文件")
print(f"{'='*70}\n")

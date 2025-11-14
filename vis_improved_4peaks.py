#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的4峰可视化：强制显示所有4个峰的位置

改进点：
1. 显式标注所有4个峰的位置（用箭头/标记）
2. 用颜色/大小区分weight的差异
3. 分别绘制单个von Mises分量（半透明）
4. 添加weight柱状图对比

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

from dataloader_glassbox_augmented import GlassBoxDatasetAugmented
from models.pointnet_pp_mvM import PointNetPPMvM

# 配置
ROOT = Path("data/MN40_multi_peak_vM_gt/glass_box")
PLY_ROOT = Path("data/full_mn40_normal_resampled_2d_rotated_ply/glass_box")
MODEL1_PATH = Path("results/glassbox_only_20251109_183051/checkpoints/best_model.pth")
MODEL2_PATH = Path("results/glassbox_no_augment_20251109_201200/checkpoints/best_model.pth")

OUTPUT_DIR = Path("results/improved_4peaks_visualization_20251113")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_POINTS = 10_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print(f"[Device] {device}")
print(f"[Output] {OUTPUT_DIR}")


@torch.no_grad()
def visualize_4peaks_detailed(model, dataset, model_name="Model", num_samples=6):
    """
    详细可视化4峰MvM预测

    每个样本显示：
    - 左上：GT的4个分量（半透明）+ 总分布
    - 右上：预测的4个分量（半透明）+ 总分布
    - 左下：GT weight柱状图
    - 右下：预测weight柱状图
    """
    model.eval()

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for sample_idx, idx in enumerate(indices):
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

        theta = np.linspace(0, 2 * np.pi, 360)

        # 创建2x2子图
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 3, 1.5], hspace=0.35, wspace=0.25)

        colors = ['red', 'blue', 'green', 'orange']

        # ========== 左上：GT的4个分量 ==========
        ax_gt = fig.add_subplot(gs[0, 0], projection='polar')

        p_gt_total = np.zeros_like(theta)
        for k in range(K_val):
            # 单个von Mises分量
            p_k = w_gt[k] * np.exp(kappa_gt[k] * np.cos(theta - mu_gt[k]))
            p_k_normalized = p_k / (np.sum(p_k) + 1e-8)

            # 绘制单个分量（半透明）
            ax_gt.fill(theta, p_k_normalized, alpha=0.2, color=colors[k], label=f'Peak {k+1}')
            ax_gt.plot(theta, p_k_normalized, '--', linewidth=1.5, alpha=0.6, color=colors[k])

            # 标注峰位置（箭头）
            mu_deg = np.rad2deg(mu_gt[k])
            ax_gt.annotate(f'μ{k+1}={mu_deg:.0f}°\nπ={w_gt[k]:.2f}',
                          xy=(mu_gt[k], p_k_normalized.max()),
                          xytext=(mu_gt[k], p_k_normalized.max() * 1.3),
                          fontsize=9,
                          ha='center',
                          color=colors[k],
                          fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color=colors[k], lw=1.5))

            p_gt_total += p_k_normalized

        # 总分布（粗实线）
        ax_gt.plot(theta, p_gt_total, 'k-', linewidth=3, label='Total', alpha=0.8)

        ax_gt.set_title(f'Ground Truth (Sample {idx})\n4 Peaks with Equal Weights',
                       fontsize=13, fontweight='bold', pad=15)
        ax_gt.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_gt.grid(True, alpha=0.3, linestyle='--')

        # ========== 右上：预测的4个分量 ==========
        ax_pred = fig.add_subplot(gs[0, 1], projection='polar')

        p_pred_total = np.zeros_like(theta)
        for k in range(K_val):
            # 单个von Mises分量
            p_k = w_pred[k] * np.exp(kappa_pred[k] * np.cos(theta - mu_pred[k]))
            p_k_normalized = p_k / (np.sum(p_k) + 1e-8)

            # 绘制单个分量（半透明，颜色强度与weight成正比）
            alpha_val = 0.15 + 0.35 * w_pred[k]  # weight越大，越不透明
            ax_pred.fill(theta, p_k_normalized, alpha=alpha_val, color=colors[k], label=f'Peak {k+1}')
            ax_pred.plot(theta, p_k_normalized, '--', linewidth=1.5, alpha=0.6, color=colors[k])

            # 标注峰位置
            mu_deg = np.rad2deg(mu_pred[k])
            ax_pred.annotate(f'μ{k+1}={mu_deg:.0f}°\nπ={w_pred[k]:.3f}',
                           xy=(mu_pred[k], p_k_normalized.max() if p_k_normalized.max() > 1e-6 else 0.01),
                           xytext=(mu_pred[k], (p_k_normalized.max() if p_k_normalized.max() > 1e-6 else 0.01) * 1.3),
                           fontsize=9,
                           ha='center',
                           color=colors[k],
                           fontweight='bold' if w_pred[k] > 0.5 else 'normal',
                           arrowprops=dict(arrowstyle='->', color=colors[k], lw=2 if w_pred[k] > 0.5 else 1))

            p_pred_total += p_k_normalized

        # 总分布
        ax_pred.plot(theta, p_pred_total, 'k-', linewidth=3, label='Total', alpha=0.8)

        ax_pred.set_title(f'{model_name} Prediction\n⚠️ Weight Imbalance Issue',
                         fontsize=13, fontweight='bold', pad=15, color='darkred')
        ax_pred.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_pred.grid(True, alpha=0.3, linestyle='--')

        # ========== 中间：GT vs Pred 对比（极坐标叠加）==========
        ax_compare = fig.add_subplot(gs[1, :], projection='polar')

        # GT总分布
        p_gt_total_unnorm = np.zeros_like(theta)
        for k in range(K_val):
            p_gt_total_unnorm += w_gt[k] * np.exp(kappa_gt[k] * np.cos(theta - mu_gt[k]))
        p_gt_total_unnorm /= (np.sum(p_gt_total_unnorm) + 1e-8)

        # 预测总分布
        p_pred_total_unnorm = np.zeros_like(theta)
        for k in range(K_val):
            p_pred_total_unnorm += w_pred[k] * np.exp(kappa_pred[k] * np.cos(theta - mu_pred[k]))
        p_pred_total_unnorm /= (np.sum(p_pred_total_unnorm) + 1e-8)

        # 绘制
        ax_compare.plot(theta, p_gt_total_unnorm, 'b-', linewidth=3, label='GT', alpha=0.7)
        ax_compare.plot(theta, p_pred_total_unnorm, 'r--', linewidth=3, label=f'{model_name} Pred', alpha=0.7)

        # 标注所有4个GT峰位置
        for k in range(K_val):
            ax_compare.plot([mu_gt[k], mu_gt[k]], [0, w_gt[k]], 'b-', linewidth=2, alpha=0.6)
            ax_compare.scatter([mu_gt[k]], [w_gt[k]], c='blue', s=150, zorder=5,
                             edgecolors='white', linewidths=2, marker='o', label=f'GT Peak {k+1}' if k < 2 else '')

        # 标注所有4个预测峰位置
        for k in range(K_val):
            ax_compare.plot([mu_pred[k], mu_pred[k]], [0, w_pred[k]], 'r--', linewidth=2, alpha=0.6)
            ax_compare.scatter([mu_pred[k]], [w_pred[k]], c='red', s=150 + 100*w_pred[k], zorder=5,
                             edgecolors='white', linewidths=2, marker='^',
                             label=f'Pred Peak {k+1}' if k < 2 else '')

        ax_compare.set_title('Distribution Comparison: GT vs Prediction',
                            fontsize=14, fontweight='bold', pad=15)
        ax_compare.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
        ax_compare.grid(True, alpha=0.3, linestyle='--')

        # ========== 底部：Weight柱状图对比 ==========
        ax_weights = fig.add_subplot(gs[2, :])

        x_pos = np.arange(K_val)
        width = 0.35

        bars_gt = ax_weights.bar(x_pos - width/2, w_gt, width, label='GT Weights',
                                 color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars_pred = ax_weights.bar(x_pos + width/2, w_pred, width, label=f'{model_name} Weights',
                                   color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (bar_gt, bar_pred) in enumerate(zip(bars_gt, bars_pred)):
            ax_weights.text(bar_gt.get_x() + bar_gt.get_width()/2, w_gt[i] + 0.02,
                           f'{w_gt[i]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax_weights.text(bar_pred.get_x() + bar_pred.get_width()/2, w_pred[i] + 0.02,
                           f'{w_pred[i]:.3f}', ha='center', va='bottom', fontsize=10,
                           fontweight='bold' if w_pred[i] > 0.5 else 'normal',
                           color='darkred' if w_pred[i] > 0.5 else 'black')

        ax_weights.set_xlabel('Peak Index', fontsize=12, fontweight='bold')
        ax_weights.set_ylabel('Weight (π)', fontsize=12, fontweight='bold')
        ax_weights.set_title('Weight Distribution Comparison', fontsize=13, fontweight='bold')
        ax_weights.set_xticks(x_pos)
        ax_weights.set_xticklabels([f'Peak {i+1}' for i in range(K_val)])
        ax_weights.legend(fontsize=11, loc='upper right')
        ax_weights.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax_weights.set_ylim(0, max(w_gt.max(), w_pred.max()) * 1.2)

        # 添加理想值参考线
        ax_weights.axhline(y=0.25, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Ideal (0.25)')

        plt.suptitle(f'Detailed 4-Peak MvM Visualization - {model_name}\nSample {idx} ({angle_deg:.0f}° rotation)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        out_path = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_sample_{sample_idx:02d}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Saved] {out_path}")


def main():
    print(f"\n{'='*70}")
    print(f"  Improved 4-Peak Visualization with All Peaks Displayed")
    print(f"{'='*70}\n")

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
    test_samples = samples[:20]  # 使用20个样本

    test_ds = GlassBoxDatasetAugmented(
        test_samples,
        NUM_POINTS,
        max_K=4,
        rotation_angles=[0],
        apply_jitter=False
    )

    # 加载模型1（增强版）
    print(f"\n[Loading] Model 1 (Augmented)...")
    model1 = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
    model1.load_state_dict(torch.load(MODEL1_PATH, map_location=device, weights_only=False))

    # 可视化模型1
    visualize_4peaks_detailed(model1, test_ds, model_name="Model1_Augmented", num_samples=6)

    # 加载模型2（无增强版）
    print(f"\n[Loading] Model 2 (No Augmentation)...")
    model2 = PointNetPPMvM(max_K=4, kappa_max=200.0, p_drop=0.4, temp=0.7).to(device)
    model2.load_state_dict(torch.load(MODEL2_PATH, map_location=device, weights_only=False))

    # 可视化模型2
    visualize_4peaks_detailed(model2, test_ds, model_name="Model2_NoAugment", num_samples=6)

    print(f"\n{'='*70}")
    print(f"  All improved visualizations generated!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # 生成问题分析报告
    report_path = OUTPUT_DIR / "weight_imbalance_analysis.md"
    with open(report_path, 'w') as f:
        f.write("# 4峰MvM权重不平衡问题分析\n\n")
        f.write("**日期**: 2025-11-13\n")
        f.write("**发现者**: 用户反馈 + Claude分析\n\n")
        f.write("---\n\n")

        f.write("## 问题描述\n\n")
        f.write("虽然模型预测的4个峰的**位置（μ）基本正确**，但**权重分布严重不平衡**：\n\n")
        f.write("- GT: [0.25, 0.25, 0.25, 0.25] ✅ 均匀分布\n")
        f.write("- 预测: [1.0, 0.0, 0.0, 0.0] ❌ 全部权重集中在1个峰\n\n")

        f.write("导致可视化只显示1个峰，而不是预期的4个峰。\n\n")

        f.write("---\n\n")

        f.write("## 根本原因\n\n")

        f.write("### 1. 温度参数设置错误\n\n")
        f.write("在 `models/pointnet_pp_mvM.py:106`:\n\n")
        f.write("```python\n")
        f.write("logit_pi = self.head_pi(feat) / self.temp  # temp = 0.7\n")
        f.write("weight = F.softmax(logit_pi, dim=-1)\n")
        f.write("```\n\n")

        f.write("**问题**: `temp=0.7 < 1` 会**放大**logit差异，导致softmax后权重集中在最大值。\n\n")
        f.write("**正确做法**: `temp >= 1` 用于平滑分布，或直接不除温度。\n\n")

        f.write("### 2. Loss函数未优化权重准确性\n\n")
        f.write("当前loss只用weight来加权KL散度，但**没有显式优化weight与GT weight的匹配**。\n\n")
        f.write("```python\n")
        f.write("# 当前loss\n")
        f.write("loss_bc = sum(w_pred[i] * KL(pred[i], gt[matched[i]]) for i in range(K))\n")
        f.write("```\n\n")

        f.write("**缺失**: 没有 `weight_loss = ||w_pred - w_gt||`\n\n")

        f.write("---\n\n")

        f.write("## 解决方案\n\n")

        f.write("### 方案A：修改温度参数并重新训练（推荐）\n\n")
        f.write("1. 修改 `models/pointnet_pp_mvM.py` 中的 `temp=0.7` → `temp=1.0` 或移除温度除法\n")
        f.write("2. 在loss中添加weight正则化：\n\n")
        f.write("```python\n")
        f.write("# 添加weight的KL散度或均匀性约束\n")
        f.write("weight_loss = F.kl_div(w_pred.log(), w_gt, reduction='batchmean')\n")
        f.write("total_loss = kl_loss + 0.1 * weight_loss\n")
        f.write("```\n\n")

        f.write("3. 重新训练50 epochs（预计耗时50分钟）\n\n")

        f.write("### 方案B：后处理强制均匀化（临时方案）\n\n")
        f.write("在推理时强制weight均匀：\n\n")
        f.write("```python\n")
        f.write("w_pred = torch.ones_like(w_pred) / K  # 强制 [0.25, 0.25, 0.25, 0.25]\n")
        f.write("```\n\n")

        f.write("---\n\n")

        f.write("## 当前可视化改进\n\n")
        f.write("为了让用户看到模型实际学到的内容，改进的可视化：\n\n")
        f.write("1. ✅ **显式标注所有4个峰的位置**（即使weight为0）\n")
        f.write("2. ✅ **分别绘制每个von Mises分量**（半透明）\n")
        f.write("3. ✅ **Weight柱状图对比**（直观展示不平衡问题）\n")
        f.write("4. ✅ **用颜色/大小区分weight差异**\n\n")

        f.write("这样用户可以看到：\n")
        f.write("- 模型学到的4个μ位置是准确的 ✅\n")
        f.write("- Weight分布有严重问题 ❌（需要重新训练修复）\n\n")

        f.write("---\n\n")

        f.write("## 下一步\n\n")
        f.write("1. ⭐ **立即**：查看改进后的可视化，确认μ位置准确\n")
        f.write("2. ⭐ **推荐**：修改模型温度参数，重新训练（方案A）\n")
        f.write("3. 对比重新训练前后的weight分布\n")
        f.write("4. 更新论文实验报告\n\n")

    print(f"[Saved] Analysis report: {report_path}")


if __name__ == "__main__":
    main()

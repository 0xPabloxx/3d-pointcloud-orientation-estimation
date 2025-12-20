#!/usr/bin/env python
"""
可视化训练过程中预测方向的变化。

比较不同epoch的模型预测分布，观察训练如何影响多峰结构。

Usage:
    python visualization/vis_training_progression.py --checkpoint_dir checkpoints/scvae_glassbox_xxx/

Author: Claude
Created: 2025-12-08
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import build_model
from datasets import OrientationDataset


def load_model_from_checkpoint(checkpoint_path, device):
    """加载模型检查点。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 0)
    return model, config, epoch


def get_predictions(model, dataset, device, num_inputs=50):
    """获取模型预测的角度分布。"""
    all_angles = []
    all_kappas = []
    gt_angles = []

    indices = np.random.choice(len(dataset), min(num_inputs, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            points = sample['points'].unsqueeze(0).to(device)
            gt_direction = sample['direction']
            gt_angle = math.atan2(gt_direction[1], gt_direction[0])
            gt_angles.append(gt_angle)

            output = model(points)
            mu = output['mu'][0]  # (N, 2)
            kappa = output['kappa'][0]  # (N,)

            # 转换为相对于GT的角度
            pred_angles = torch.atan2(mu[:, 1], mu[:, 0]).cpu().numpy()
            relative_angles = (pred_angles - gt_angle) % (2 * math.pi)

            all_angles.extend(relative_angles)
            all_kappas.extend(kappa.cpu().numpy())

    return np.array(all_angles), np.array(all_kappas)


def visualize_progression(checkpoint_dir, device, num_inputs=100, save_path=None):
    """
    可视化训练过程中预测分布的变化。

    Args:
        checkpoint_dir: 检查点目录
        device: torch设备
        num_inputs: 每个检查点使用的输入数量
        save_path: 保存路径
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 查找所有检查点
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch*.pth'))
    if (checkpoint_dir / 'best_model.pth').exists():
        # 添加best model作为最后一个
        pass

    if not checkpoints:
        print("未找到检查点文件！")
        return

    print(f"找到 {len(checkpoints)} 个检查点:")
    for cp in checkpoints:
        print(f"  - {cp.name}")

    # 加载第一个检查点获取配置
    _, config, _ = load_model_from_checkpoint(checkpoints[0], device)

    # 加载数据集
    k_filter = config.get('data', {}).get('k_filter', [4])
    dataset = OrientationDataset(
        config.get('data', {}).get('annotation_file', 'data_annotation/symmetry_annotations.json'),
        config.get('data', {}).get('data_dir', 'data/full_mn40_normal_resampled_ply'),
        num_points=config.get('data', {}).get('num_points', 1024),
        split='test',
        k_filter=k_filter,
        align_pointcloud=True,
        augment=False
    )

    # 固定随机种子确保每个检查点使用相同的样本
    np.random.seed(42)

    # 创建图形
    n_checkpoints = len(checkpoints)
    fig, axes = plt.subplots(2, n_checkpoints, figsize=(5*n_checkpoints, 10))

    if n_checkpoints == 1:
        axes = axes.reshape(2, 1)

    all_data = []

    for i, cp_path in enumerate(checkpoints):
        print(f"\n处理检查点: {cp_path.name}")

        # 重置随机种子
        np.random.seed(42)

        model, _, epoch = load_model_from_checkpoint(cp_path, device)
        angles, kappas = get_predictions(model, dataset, device, num_inputs)

        all_data.append({
            'epoch': epoch,
            'angles': angles,
            'kappas': kappas
        })

        # 上方: 极坐标直方图
        ax1 = plt.subplot(2, n_checkpoints, i+1, projection='polar')
        bins = np.linspace(0, 2*np.pi, 73)
        counts, _ = np.histogram(angles, bins=bins)
        theta = (bins[:-1] + bins[1:]) / 2
        ax1.bar(theta, counts, width=2*np.pi/72, alpha=0.7, color='blue', edgecolor='black')

        # 标记期望的4个峰
        for peak in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ax1.axvline(peak, color='red', linestyle='--', alpha=0.5)

        ax1.set_title(f'Epoch {epoch}\nκ={kappas.mean():.0f}±{kappas.std():.0f}', fontsize=12)

        # 下方: 普通直方图
        ax2 = axes[1, i]
        ax2.hist(np.degrees(angles), bins=72, range=(0, 360), alpha=0.7, color='blue', edgecolor='black')

        # 标记期望的4个峰
        for peak in [0, 90, 180, 270]:
            ax2.axvline(peak, color='red', linestyle='--', alpha=0.5, label=f'{peak}°' if i==0 and peak==0 else None)

        ax2.set_xlabel('Angle relative to GT (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

        # 计算每个峰附近的比例
        peak_info = []
        for peak_deg in [0, 90, 180, 270]:
            peak_rad = np.radians(peak_deg)
            diffs = np.abs((angles - peak_rad + np.pi) % (2*np.pi) - np.pi)
            near_peak = np.sum(diffs < np.radians(22.5))
            pct = 100 * near_peak / len(angles)
            peak_info.append(f'{peak_deg}°:{pct:.0f}%')

        ax2.text(0.5, -0.15, ' | '.join(peak_info), transform=ax2.transAxes,
                ha='center', fontsize=9, color='darkgreen')

    plt.suptitle('sCVAE Prediction Distribution During Training\n(Red dashed = expected peak positions for K=4)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {save_path}")

    plt.show()

    # 打印汇总统计
    print("\n" + "="*60)
    print("训练过程汇总")
    print("="*60)
    for data in all_data:
        print(f"\nEpoch {data['epoch']}:")
        print(f"  预测数量: {len(data['angles'])}")
        print(f"  平均κ: {data['kappas'].mean():.1f} ± {data['kappas'].std():.1f}")

        for peak_deg in [0, 90, 180, 270]:
            peak_rad = np.radians(peak_deg)
            diffs = np.abs((data['angles'] - peak_rad + np.pi) % (2*np.pi) - np.pi)
            near_peak = np.sum(diffs < np.radians(22.5))
            pct = 100 * near_peak / len(data['angles'])
            print(f"  {peak_deg:3d}°附近: {near_peak:4d} ({pct:5.1f}%)")

    return fig


def visualize_single_sample_progression(checkpoint_dir, device, sample_idx=0, save_path=None):
    """
    可视化单个样本在不同epoch下的预测变化。

    展示同一个点云输入，在训练不同阶段产生的8个样本预测。
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch*.pth'))

    if not checkpoints:
        print("未找到检查点文件！")
        return

    # 加载配置和数据集
    _, config, _ = load_model_from_checkpoint(checkpoints[0], device)
    k_filter = config.get('data', {}).get('k_filter', [4])
    dataset = OrientationDataset(
        config.get('data', {}).get('annotation_file', 'data_annotation/symmetry_annotations.json'),
        config.get('data', {}).get('data_dir', 'data/full_mn40_normal_resampled_ply'),
        num_points=config.get('data', {}).get('num_points', 1024),
        split='test',
        k_filter=k_filter,
        align_pointcloud=True,
        augment=False
    )

    # 获取固定样本
    sample = dataset[sample_idx]
    points = sample['points'].unsqueeze(0).to(device)
    gt_direction = sample['direction']
    gt_angle = math.atan2(gt_direction[1], gt_direction[0])

    n_checkpoints = len(checkpoints)
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(5*n_checkpoints, 5),
                             subplot_kw={'projection': 'polar'})

    if n_checkpoints == 1:
        axes = [axes]

    for i, cp_path in enumerate(checkpoints):
        model, _, epoch = load_model_from_checkpoint(cp_path, device)

        with torch.no_grad():
            output = model(points)
            mu = output['mu'][0]  # (N, 2)
            kappa = output['kappa'][0]  # (N,)

        pred_angles = torch.atan2(mu[:, 1], mu[:, 0]).cpu().numpy()
        kappas = kappa.cpu().numpy()

        ax = axes[i]

        # 绘制预测点 (大小与kappa成正比)
        sizes = np.clip(kappas / 10, 20, 200)
        ax.scatter(pred_angles, np.ones_like(pred_angles), s=sizes, c='blue', alpha=0.6, label='Pred')

        # 绘制GT方向
        ax.scatter([gt_angle], [1], s=150, c='red', marker='*', label='GT', zorder=10)

        # 绘制K=4的等效方向
        for k in range(1, 4):
            equiv_angle = gt_angle + k * np.pi/2
            ax.scatter([equiv_angle], [0.9], s=80, c='green', marker='x', alpha=0.5)

        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_title(f'Epoch {epoch}\nκ: {kappas.mean():.0f}±{kappas.std():.0f}', fontsize=12)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Single Sample (idx={sample_idx}) Prediction Evolution\nBlue=8 samples, Red star=GT, Green X=equiv directions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='可视化训练过程中的预测变化')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='检查点目录路径')
    parser.add_argument('--num_inputs', type=int, default=100,
                        help='分析的输入数量')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='如果指定，则可视化单个样本的变化')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_idx is not None:
        # 可视化单个样本
        visualize_single_sample_progression(
            args.checkpoint_dir, device,
            sample_idx=args.sample_idx,
            save_path=output_dir / f'sample_{args.sample_idx}_progression.png'
        )
    else:
        # 可视化整体分布变化
        visualize_progression(
            args.checkpoint_dir, device,
            num_inputs=args.num_inputs,
            save_path=output_dir / 'training_progression.png'
        )


if __name__ == '__main__':
    main()

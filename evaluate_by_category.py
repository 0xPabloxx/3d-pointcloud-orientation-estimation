"""
按类别评估D/DR系列模型

用法:
    python evaluate_by_category.py --checkpoint checkpoints/D_16a_20251219_053127
    python evaluate_by_category.py --checkpoint checkpoints/DR_8a_20251219_215002
    python evaluate_by_category.py --all  # 评估所有D/DR系列
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn

# 从 train_direction.py 导入模型组件
from train_direction import (
    PointNetPlusPlusEncoder,
    DiscreteDirectionHead,
    ProjectionRegressionHead,
)


# =============================================================================
# Evaluation Functions
# =============================================================================

def circular_distance(angle1, angle2):
    """计算圆周距离 (弧度)"""
    diff = angle1 - angle2
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def compute_errors_by_category(pred_probs, gt_angles, categories, num_bins):
    """
    按类别计算角度误差

    Returns:
        dict: {category: [error1, error2, ...]}
    """
    B = pred_probs.shape[0]
    bin_width = 2 * np.pi / num_bins

    category_errors = defaultdict(list)

    for b in range(B):
        cat = categories[b]

        if cat == 'no_front':
            # no_front没有角度误差的概念，跳过
            continue

        # argmax预测
        pred_bin = pred_probs[b].argmax().item()
        pred_angle = pred_bin * bin_width
        gt = gt_angles[b].item()

        if cat == '1_front':
            error = circular_distance(pred_angle, gt)
        elif cat == '4_fronts':
            # 找到4个等效方向中最近的
            min_error = float('inf')
            for i in range(4):
                candidate = (gt + i * np.pi / 2) % (2 * np.pi)
                error = circular_distance(pred_angle, candidate)
                min_error = min(min_error, error)
            error = min_error
        else:
            continue

        # 转为度
        error_deg = error * 180 / np.pi
        category_errors[cat].append(error_deg)

    return category_errors


def evaluate_checkpoint(checkpoint_dir, device='cuda'):
    """评估单个checkpoint"""

    # 加载配置
    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get('exp_name', Path(checkpoint_dir).name)
    num_bins = config.get('num_bins', 8)
    gt_mode = config.get('gt_mode', 'projection')
    temperature = config.get('temperature', 5.0)

    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"  num_bins: {num_bins}, gt_mode: {gt_mode}, temperature: {temperature}")
    print(f"{'='*60}")

    # 加载模型
    encoder = PointNetPlusPlusEncoder().to(device)

    if gt_mode == 'dr':
        head = ProjectionRegressionHead(num_dirs=num_bins).to(device)
    else:
        head = DiscreteDirectionHead(num_bins=num_bins).to(device)

    # 加载权重
    ckpt_path = Path(checkpoint_dir) / 'best_error.pth'
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / 'best_loss.pth'

    if not ckpt_path.exists():
        print(f"Checkpoint not found in {checkpoint_dir}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 加载权重
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])

    encoder.eval()
    head.eval()

    # 准备数据集 - 分别评估每个类别
    results = {}

    for category in ['1_front', '4_fronts']:
        dataset = DiscreteDirectionDataset(
            split='val',
            categories=[category],
            num_bins=num_bins,
            gt_mode=gt_mode,
            temperature=temperature,
            augment=False,
        )

        if len(dataset) == 0:
            print(f"  {category}: No samples")
            continue

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        all_errors = []

        with torch.no_grad():
            for batch in loader:
                points = batch['points'].to(device)
                gt_angle = batch['gt_angle'].to(device)
                categories_batch = batch['category']

                # 前向传播
                features = encoder(points)
                pred = head(features)

                # 计算概率
                pred_probs = F.softmax(pred, dim=-1)

                # 按类别计算误差
                errors = compute_errors_by_category(
                    pred_probs.cpu(),
                    gt_angle.cpu(),
                    categories_batch,
                    num_bins
                )

                for cat, errs in errors.items():
                    all_errors.extend(errs)

        if len(all_errors) > 0:
            mean_error = np.mean(all_errors)
            std_error = np.std(all_errors)
            median_error = np.median(all_errors)
            max_error = np.max(all_errors)
            min_error = np.min(all_errors)

            results[category] = {
                'mean': mean_error,
                'std': std_error,
                'median': median_error,
                'max': max_error,
                'min': min_error,
                'count': len(all_errors),
            }

            print(f"  {category}: mean={mean_error:.2f}°, std={std_error:.2f}°, "
                  f"median={median_error:.2f}°, n={len(all_errors)}")

    # 计算整体误差 (包含所有类别)
    all_dataset = DiscreteDirectionDataset(
        split='val',
        categories=['1_front', '4_fronts', 'no_front'],
        num_bins=num_bins,
        gt_mode=gt_mode,
        temperature=temperature,
        augment=False,
    )

    all_loader = DataLoader(
        all_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    all_errors = []
    with torch.no_grad():
        for batch in all_loader:
            points = batch['points'].to(device)
            gt_angle = batch['gt_angle'].to(device)
            categories_batch = batch['category']

            features = encoder(points)
            pred = head(features)
            pred_probs = F.softmax(pred, dim=-1)

            errors = compute_errors_by_category(
                pred_probs.cpu(), gt_angle.cpu(), categories_batch, num_bins
            )
            for errs in errors.values():
                all_errors.extend(errs)

    if len(all_errors) > 0:
        results['overall'] = {
            'mean': np.mean(all_errors),
            'std': np.std(all_errors),
            'median': np.median(all_errors),
            'max': np.max(all_errors),
            'min': np.min(all_errors),
            'count': len(all_errors),
        }
        print(f"  overall: mean={results['overall']['mean']:.2f}°, "
              f"std={results['overall']['std']:.2f}°, n={len(all_errors)}")

    return {
        'exp_name': exp_name,
        'config': config,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint directory')
    parser.add_argument('--all', action='store_true', help='Evaluate all D/DR checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.all:
        # 找到所有D/DR系列的checkpoint
        checkpoints_dir = Path('checkpoints')

        # D系列
        d_checkpoints = [
            'D_8a_20251218_171207',
            'D_8b_20251218_203046',
            'D_8c_20251218_234110',
            'D_8d_20251219_024151',
            'D_16a_20251219_053127',
            'D_16b_20251219_081853',
        ]

        # DR系列
        dr_checkpoints = [
            'DR_8a_20251219_215002',
            'DR_8b_20251220_004513',
            'DR_16a_20251220_033656',
        ]

        all_results = []

        print("\n" + "="*80)
        print("  D Series Evaluation")
        print("="*80)

        for ckpt in d_checkpoints:
            ckpt_path = checkpoints_dir / ckpt
            if ckpt_path.exists():
                result = evaluate_checkpoint(str(ckpt_path), args.device)
                if result:
                    all_results.append(result)

        print("\n" + "="*80)
        print("  DR Series Evaluation")
        print("="*80)

        for ckpt in dr_checkpoints:
            ckpt_path = checkpoints_dir / ckpt
            if ckpt_path.exists():
                result = evaluate_checkpoint(str(ckpt_path), args.device)
                if result:
                    all_results.append(result)

        # 打印汇总表格
        print("\n" + "="*80)
        print("  Summary Table")
        print("="*80)
        print(f"\n{'Experiment':<15} {'1_front':<15} {'4_fronts':<15} {'Overall':<15}")
        print("-"*60)

        for r in all_results:
            exp = r['exp_name']
            r1 = r['results'].get('1_front', {}).get('mean', float('nan'))
            r4 = r['results'].get('4_fronts', {}).get('mean', float('nan'))
            ro = r['results'].get('overall', {}).get('mean', float('nan'))
            print(f"{exp:<15} {r1:>10.2f}°    {r4:>10.2f}°    {ro:>10.2f}°")

        # 保存结果到JSON
        output_path = 'evaluation_by_category.json'
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args.device)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

"""
正确的 Hungarian 匹配评估

关键改进：
1. 找真正的峰（局部最大值），而不是简单的 Top-K
2. 用 Hungarian 做一对一匹配，不允许重复匹配
3. 同时评估两种场景：
   - 场景A：只需要预测 1 个正确方向（用 Argmax + Min Distance）
   - 场景B：需要预测所有 K 个等效方向（用 Peak Detection + Hungarian）
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn
from train_direction import PointNetPlusPlusEncoder, DiscreteDirectionHead


def circular_distance(a1, a2):
    """圆周距离 (弧度)"""
    diff = a1 - a2
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def find_local_peaks(probs, min_height=0.05):
    """
    找局部最大值（真正的峰），而不是简单的 Top-K

    Args:
        probs: 概率分布 (numpy array)
        min_height: 峰的最小高度阈值

    Returns:
        峰的 bin 索引列表
    """
    n = len(probs)
    peaks = []

    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n

        # 局部最大值：比左右邻居都大，且超过阈值
        if probs[i] >= probs[left] and probs[i] >= probs[right] and probs[i] > min_height:
            peaks.append(i)

    # 按概率排序
    peaks = sorted(peaks, key=lambda x: probs[x], reverse=True)
    return peaks


def hungarian_matching(pred_angles, gt_angles):
    """
    Hungarian 算法一对一匹配

    Args:
        pred_angles: 预测角度列表 (弧度)
        gt_angles: GT角度列表 (弧度)

    Returns:
        平均匹配误差，匹配列表
    """
    n_pred = len(pred_angles)
    n_gt = len(gt_angles)

    if n_pred == 0:
        return float('inf'), []

    # 构建代价矩阵
    # 如果预测数 != GT数，用较小的那个做匹配
    n = min(n_pred, n_gt)

    cost_matrix = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_angles):
        for j, gt in enumerate(gt_angles):
            cost_matrix[i, j] = circular_distance(pred, gt)

    # Hungarian 算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 计算匹配结果
    matches = []
    total_error = 0
    for i, j in zip(row_ind, col_ind):
        error = cost_matrix[i, j]
        matches.append({
            'pred_idx': i,
            'gt_idx': j,
            'pred_angle': pred_angles[i] * 180 / np.pi,
            'gt_angle': gt_angles[j] * 180 / np.pi,
            'error': error * 180 / np.pi
        })
        total_error += error

    avg_error = total_error / len(matches) if matches else float('inf')
    return avg_error * 180 / np.pi, matches


def evaluate_sample(pred_probs, gt_angle, category, num_bins):
    """
    评估单个样本

    Returns:
        dict: {
            'argmax_min_dist': 场景A误差（只需1个预测）,
            'hungarian': 场景B误差（需要K个预测）,
            'n_peaks_detected': 检测到的峰数,
            'n_peaks_expected': 期望的峰数,
        }
    """
    bin_width = 2 * np.pi / num_bins
    probs = pred_probs.cpu().numpy()

    # 确定期望峰数
    if category == '1_front':
        n_expected = 1
        gt_angles = [gt_angle]
    elif category == '2_fronts':
        n_expected = 2
        gt_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
    elif category == '4_fronts':
        n_expected = 4
        gt_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
    else:
        return None

    # ===== 场景A: Argmax + Min Distance =====
    pred_bin = pred_probs.argmax().item()
    pred_angle_argmax = pred_bin * bin_width
    error_argmax = min(circular_distance(pred_angle_argmax, a) for a in gt_angles)

    # ===== 场景B: Peak Detection + Hungarian =====
    peaks = find_local_peaks(probs, min_height=0.05)

    # 取前 n_expected 个峰（如果有的话）
    peaks_to_use = peaks[:n_expected] if len(peaks) >= n_expected else peaks
    pred_angles_peaks = [p * bin_width for p in peaks_to_use]

    if len(pred_angles_peaks) > 0:
        error_hungarian, matches = hungarian_matching(pred_angles_peaks, gt_angles)
    else:
        error_hungarian = float('inf')
        matches = []

    return {
        'argmax_min_dist': error_argmax * 180 / np.pi,
        'hungarian': error_hungarian,
        'n_peaks_detected': len(peaks),
        'n_peaks_expected': n_expected,
        'peaks_used': len(peaks_to_use),
    }


def evaluate_checkpoint(checkpoint_dir, device='cuda'):
    """评估 checkpoint"""

    config_path = Path(checkpoint_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get('exp_name')
    num_bins = config.get('num_bins', 8)
    gt_mode = config.get('gt_mode', 'projection')
    temperature = config.get('temperature', 5.0)

    print(f"\n{'='*70}")
    print(f"  {exp_name} (bins={num_bins})")
    print(f"{'='*70}")

    # 加载模型
    encoder = PointNetPlusPlusEncoder().to(device)
    head = DiscreteDirectionHead(num_bins=num_bins).to(device)

    ckpt_path = Path(checkpoint_dir) / 'best_error.pth'
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / 'best_loss.pth'

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    encoder.eval()
    head.eval()

    results = {}

    for category in ['1_front', '4_fronts']:
        dataset = DiscreteDirectionDataset(
            split='val',
            categories=[category],
            num_bins=num_bins,
            gt_mode=gt_mode,
            temperature=temperature,
            augment=True,
            augment_factor=10,
        )

        if len(dataset) == 0:
            continue

        loader = DataLoader(dataset, batch_size=32, shuffle=False,
                           num_workers=4, collate_fn=collate_fn)

        errors_argmax = []
        errors_hungarian = []
        peak_counts = []

        with torch.no_grad():
            for batch in loader:
                points = batch['points'].to(device)
                gt_angle_batch = batch['gt_angle']
                categories_batch = batch['category']

                features = encoder(points)
                pred = head(features)
                pred_probs = F.softmax(pred, dim=-1)

                for b in range(len(categories_batch)):
                    cat = categories_batch[b]
                    result = evaluate_sample(
                        pred_probs[b],
                        gt_angle_batch[b].item(),
                        cat,
                        num_bins
                    )
                    if result:
                        errors_argmax.append(result['argmax_min_dist'])
                        errors_hungarian.append(result['hungarian'])
                        peak_counts.append(result['n_peaks_detected'])

        results[category] = {
            'argmax_min_dist': np.mean(errors_argmax),
            'hungarian': np.mean(errors_hungarian),
            'avg_peaks': np.mean(peak_counts),
        }

        n_expected = 4 if category == '4_fronts' else 1
        print(f"\n  {category} (期望 {n_expected} 个峰):")
        print(f"    检测到的平均峰数: {np.mean(peak_counts):.2f}")
        print(f"    场景A (只需1个正确): Argmax+MinDist = {np.mean(errors_argmax):.2f}°")
        print(f"    场景B (需要{n_expected}个正确): Hungarian = {np.mean(errors_hungarian):.2f}°")

    return results


def main():
    print("\n" + "="*70)
    print("  正确的 Hungarian 匹配评估")
    print("  场景A: 只需预测 1 个正确方向 (Argmax + Min Distance)")
    print("  场景B: 需要预测所有 K 个等效方向 (Peak Detection + Hungarian)")
    print("="*70)

    checkpoints = [
        'checkpoints/D_8a_20251218_171207',
        'checkpoints/D_8b_20251218_203046',
        'checkpoints/D_16a_20251219_053127',
        'checkpoints/D_16b_20251219_081853',
    ]

    all_results = {}
    for ckpt in checkpoints:
        if Path(ckpt).exists():
            result = evaluate_checkpoint(ckpt)
            exp_name = Path(ckpt).name.split('_2025')[0]
            all_results[exp_name] = result

    # 汇总表
    print("\n" + "="*70)
    print("  汇总对比")
    print("="*70)

    print(f"\n{'模型':<10} | {'1_front':<20} | {'4_fronts':<30}")
    print(f"{'':10} | {'Argmax':>8} {'Hungarian':>10} | {'Argmax':>8} {'Hungarian':>10} {'峰数':>8}")
    print("-"*75)

    for exp_name, result in all_results.items():
        r1 = result.get('1_front', {})
        r4 = result.get('4_fronts', {})

        print(f"{exp_name:<10} | "
              f"{r1.get('argmax_min_dist', 0):>7.2f}° {r1.get('hungarian', 0):>9.2f}° | "
              f"{r4.get('argmax_min_dist', 0):>7.2f}° {r4.get('hungarian', 0):>9.2f}° "
              f"{r4.get('avg_peaks', 0):>7.2f}")


if __name__ == '__main__':
    main()

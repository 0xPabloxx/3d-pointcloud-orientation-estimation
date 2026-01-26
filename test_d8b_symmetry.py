#!/usr/bin/env python3
"""
测试D_8b模型在no_front和旋转对称类别上的均匀分布输出能力

检验模型是否能识别出这两类应该输出8个bins均等概率的对称性
不使用旋转增强，只测试模型本身的对称性识别能力
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))

from test_direction_models import DirectionModel, load_ply

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048
NUM_BINS = 8

# 旋转对称或无正面的symmetry_name
SYMMETRIC_NAMES = {'旋转对称', 'symmetric', '完全对称', '无正面', 'no_front', '没有正面'}


def entropy(probs, eps=1e-10):
    """计算概率分布的熵"""
    return -np.sum(probs * np.log(probs + eps))


def max_entropy(n_bins):
    """均匀分布的最大熵"""
    return np.log(n_bins)


def is_uniform_distribution(probs, threshold_entropy_ratio=0.95, threshold_max_diff=0.05):
    """
    判断概率分布是否接近均匀分布

    使用两个标准:
    1. 熵接近最大熵 (> threshold_entropy_ratio * max_entropy)
    2. 最大值和最小值之差小于阈值
    """
    h = entropy(probs)
    h_max = max_entropy(len(probs))
    entropy_ratio = h / h_max

    max_diff = np.max(probs) - np.min(probs)

    return entropy_ratio, max_diff


def main():
    print("=" * 80)
    print("测试D_8b模型在no_front和旋转对称类别上的均匀分布输出")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()

    # 加载测试集
    with open('data/test_set/test_set.json') as f:
        all_samples = json.load(f)

    # 筛选出no_front和旋转对称的样本
    symmetric_samples = []
    for s in all_samples:
        sym_name = s.get('symmetry_name', '')
        if sym_name in SYMMETRIC_NAMES:
            symmetric_samples.append(s)

    print(f"总测试样本: {len(all_samples)}")
    print(f"对称性样本 (no_front + 旋转对称): {len(symmetric_samples)}")

    # 统计各类型
    name_counts = defaultdict(int)
    for s in symmetric_samples:
        name_counts[s['symmetry_name']] += 1
    print("详细分布:")
    for name, cnt in sorted(name_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {cnt}")
    print()

    # 加载模型
    ckpt_path = 'checkpoints/D_8b_20251218_203046/best_error.pth'
    model = DirectionModel(mode='discrete', num_bins=8).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    print(f"模型加载完成: {ckpt_path}")
    print()

    data_root = Path('data/full_mn40_normal_resampled_ply')

    # 测试
    results = []

    print("测试中...")
    with torch.no_grad():
        for i, sample in enumerate(symmetric_samples):
            if (i + 1) % 20 == 0:
                print(f"  处理: {i+1}/{len(symmetric_samples)}")

            ply_path = data_root / sample['file']
            if not ply_path.exists():
                print(f"  [Warning] 文件不存在: {ply_path}")
                continue

            try:
                pts = load_ply(str(ply_path), NUM_POINTS)
            except Exception as e:
                print(f"  [Error] 加载失败: {ply_path}, {e}")
                continue

            # 直接推理，不做旋转增强
            pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)
            logits = model(pts_tensor)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            entropy_ratio, max_diff = is_uniform_distribution(probs)

            results.append({
                'file': sample['file'],
                'category': sample.get('category', ''),
                'symmetry_name': sample['symmetry_name'],
                'probs': probs,
                'entropy_ratio': entropy_ratio,
                'max_diff': max_diff,
            })

    print(f"\n测试完成: {len(results)} 个样本")
    print()

    # 分析结果
    print("=" * 80)
    print("均匀分布分析")
    print("=" * 80)

    # 按不同阈值统计
    thresholds = [
        (0.99, 0.02, "严格 (熵>99%max, diff<0.02)"),
        (0.95, 0.05, "中等 (熵>95%max, diff<0.05)"),
        (0.90, 0.10, "宽松 (熵>90%max, diff<0.10)"),
        (0.85, 0.15, "很宽松 (熵>85%max, diff<0.15)"),
    ]

    print("\n按阈值统计均匀分布比例:")
    print("-" * 60)
    for ent_thresh, diff_thresh, desc in thresholds:
        count = sum(1 for r in results
                   if r['entropy_ratio'] >= ent_thresh and r['max_diff'] <= diff_thresh)
        ratio = count / len(results) if results else 0
        print(f"{desc}: {count}/{len(results)} = {ratio*100:.1f}%")

    # 整体统计
    entropy_ratios = [r['entropy_ratio'] for r in results]
    max_diffs = [r['max_diff'] for r in results]

    print(f"\n熵比率统计 (1.0=完美均匀):")
    print(f"  均值: {np.mean(entropy_ratios):.4f}")
    print(f"  中位数: {np.median(entropy_ratios):.4f}")
    print(f"  最小: {np.min(entropy_ratios):.4f}")
    print(f"  最大: {np.max(entropy_ratios):.4f}")

    print(f"\n概率最大差值统计 (0.0=完美均匀, 理论完美=0.0):")
    print(f"  均值: {np.mean(max_diffs):.4f}")
    print(f"  中位数: {np.median(max_diffs):.4f}")
    print(f"  最小: {np.min(max_diffs):.4f}")
    print(f"  最大: {np.max(max_diffs):.4f}")

    # 输出一些具体例子
    print("\n" + "=" * 80)
    print("具体样本分析 (排序按熵比率)")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x['entropy_ratio'], reverse=True)

    print("\n最接近均匀分布的5个样本:")
    print("-" * 60)
    for r in sorted_results[:5]:
        probs_str = ", ".join([f"{p:.3f}" for p in r['probs']])
        print(f"{r['file']}")
        print(f"  类型: {r['symmetry_name']}, 熵比率: {r['entropy_ratio']:.4f}, 最大差: {r['max_diff']:.4f}")
        print(f"  概率: [{probs_str}]")

    print("\n最远离均匀分布的5个样本:")
    print("-" * 60)
    for r in sorted_results[-5:]:
        probs_str = ", ".join([f"{p:.3f}" for p in r['probs']])
        print(f"{r['file']}")
        print(f"  类型: {r['symmetry_name']}, 熵比率: {r['entropy_ratio']:.4f}, 最大差: {r['max_diff']:.4f}")
        print(f"  概率: [{probs_str}]")

    # 按symmetry_name分组统计
    print("\n" + "=" * 80)
    print("按对称类型分组统计")
    print("=" * 80)

    by_name = defaultdict(list)
    for r in results:
        by_name[r['symmetry_name']].append(r)

    for name in sorted(by_name.keys()):
        items = by_name[name]
        ent_mean = np.mean([r['entropy_ratio'] for r in items])
        diff_mean = np.mean([r['max_diff'] for r in items])
        # 宽松标准
        uniform_count = sum(1 for r in items if r['entropy_ratio'] >= 0.90 and r['max_diff'] <= 0.10)
        print(f"\n{name} (n={len(items)}):")
        print(f"  平均熵比率: {ent_mean:.4f}")
        print(f"  平均最大差: {diff_mean:.4f}")
        print(f"  宽松标准均匀比例: {uniform_count}/{len(items)} = {uniform_count/len(items)*100:.1f}%")

    # 理想分布对比
    print("\n" + "=" * 80)
    print("参考: 理想均匀分布")
    print("=" * 80)
    ideal_probs = np.ones(NUM_BINS) / NUM_BINS
    ideal_ent, ideal_diff = is_uniform_distribution(ideal_probs)
    print(f"理想均匀分布: [{', '.join([f'{p:.3f}' for p in ideal_probs])}]")
    print(f"理想熵比率: {ideal_ent:.4f}, 理想最大差: {ideal_diff:.4f}")


if __name__ == '__main__':
    main()

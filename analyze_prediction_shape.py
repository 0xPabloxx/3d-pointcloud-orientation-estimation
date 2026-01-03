"""
分析模型预测的分布形状：是单峰还是多峰？
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn
from train_direction import PointNetPlusPlusEncoder, DiscreteDirectionHead
from torch.utils.data import DataLoader


def count_peaks(probs, threshold=0.1):
    """
    统计概率分布中的峰数量
    峰定义：概率 > threshold 且是局部最大值
    """
    probs = probs.cpu().numpy()
    n = len(probs)
    peaks = []

    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n
        # 局部最大值且超过阈值
        if probs[i] > threshold and probs[i] >= probs[left] and probs[i] >= probs[right]:
            peaks.append((i, probs[i]))

    return len(peaks), peaks


def analyze_checkpoint(checkpoint_dir, device='cuda'):
    """分析模型预测分布"""

    config_path = Path(checkpoint_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get('exp_name')
    num_bins = config.get('num_bins', 8)
    gt_mode = config.get('gt_mode', 'projection')
    temperature = config.get('temperature', 5.0)

    print(f"\n{'='*60}")
    print(f"Analyzing: {exp_name} (num_bins={num_bins})")
    print(f"{'='*60}")

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

    for category in ['1_front', '4_fronts']:
        dataset = DiscreteDirectionDataset(
            split='val',
            categories=[category],
            num_bins=num_bins,
            gt_mode=gt_mode,
            temperature=temperature,
            augment=True,
            augment_factor=5,
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False,
                           num_workers=4, collate_fn=collate_fn)

        peak_counts = []
        example_probs = []

        with torch.no_grad():
            for batch in loader:
                points = batch['points'].to(device)
                features = encoder(points)
                pred = head(features)
                pred_probs = F.softmax(pred, dim=-1)

                for b in range(pred_probs.shape[0]):
                    n_peaks, peaks = count_peaks(pred_probs[b], threshold=0.1)
                    peak_counts.append(n_peaks)

                    if len(example_probs) < 3:
                        example_probs.append(pred_probs[b].cpu().numpy())

        # 统计
        peak_dist = {}
        for p in peak_counts:
            peak_dist[p] = peak_dist.get(p, 0) + 1

        print(f"\n  {category}:")
        print(f"    峰数量分布: {dict(sorted(peak_dist.items()))}")
        print(f"    平均峰数: {np.mean(peak_counts):.2f}")

        # 打印几个示例分布
        print(f"    示例预测分布:")
        for i, probs in enumerate(example_probs[:2]):
            probs_str = " ".join([f"{p:.2f}" for p in probs])
            n_peaks, _ = count_peaks(torch.tensor(probs), threshold=0.1)
            print(f"      样本{i+1} ({n_peaks}峰): [{probs_str}]")


if __name__ == '__main__':
    # 分析 D_16b (最佳模型)
    analyze_checkpoint('checkpoints/D_16b_20251219_081853')

    # 分析 D_8a
    analyze_checkpoint('checkpoints/D_8a_20251218_171207')

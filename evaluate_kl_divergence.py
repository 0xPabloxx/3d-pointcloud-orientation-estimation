"""
基于 KL Divergence 的分布相似度评估

不检测峰，直接比较预测分布和理想 GT 分布的相似度
KL 散度越小，预测越好
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader

from datasets.discrete_direction_dataset import DiscreteDirectionDataset, collate_fn
from train_direction import PointNetPlusPlusEncoder, DiscreteDirectionHead, ProjectionRegressionHead


def create_gt_distribution(gt_angle, category, num_bins, temperature=5.0):
    """
    生成理想的 GT 分布

    Args:
        gt_angle: 基础 GT 角度 (弧度)
        category: '1_front', '2_fronts', '4_fronts'
        num_bins: bin 数量
        temperature: 分布的集中程度（越大越尖锐）

    Returns:
        numpy array: 归一化的概率分布
    """
    bin_width = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * bin_width

    # 确定等效方向
    if category == '1_front':
        equiv_angles = [gt_angle]
    elif category == '2_fronts':
        equiv_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
    elif category == '4_fronts':
        equiv_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
    else:
        # no_front: 均匀分布
        return np.ones(num_bins) / num_bins

    # 用 von Mises-like kernel 生成分布
    # p(bin_i) ∝ sum_k exp(τ * cos(bin_i - angle_k))
    dist = np.zeros(num_bins)
    for angle in equiv_angles:
        for i, center in enumerate(bin_centers):
            dist[i] += np.exp(temperature * np.cos(center - angle))

    # 归一化
    dist = dist / dist.sum()
    return dist


def kl_divergence(p, q, eps=1e-10):
    """
    计算 KL(P || Q)
    P: GT 分布
    Q: 预测分布
    """
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def js_divergence(p, q, eps=1e-10):
    """
    计算 Jensen-Shannon Divergence (对称版本的 KL)
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2
    """
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def evaluate_checkpoint(checkpoint_dir, device='cuda'):
    """评估 checkpoint"""

    config_path = Path(checkpoint_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get('exp_name')
    num_bins = config.get('num_bins', 8)
    gt_mode = config.get('gt_mode', 'projection')
    temperature = config.get('temperature', 5.0)

    print(f"\n{'='*60}")
    print(f"  {exp_name} (bins={num_bins}, τ={temperature})")
    print(f"{'='*60}")

    # 加载模型
    encoder = PointNetPlusPlusEncoder().to(device)
    if gt_mode == 'dr':
        head = ProjectionRegressionHead(num_dirs=num_bins).to(device)
    else:
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

        kl_values = []
        js_values = []

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
                    if cat == 'no_front':
                        continue

                    # 预测分布
                    pred_dist = pred_probs[b].cpu().numpy()

                    # 理想 GT 分布
                    gt_dist = create_gt_distribution(
                        gt_angle_batch[b].item(),
                        cat,
                        num_bins,
                        temperature
                    )

                    # 计算 KL 和 JS
                    kl = kl_divergence(gt_dist, pred_dist)
                    js = js_divergence(gt_dist, pred_dist)

                    kl_values.append(kl)
                    js_values.append(js)

        results[category] = {
            'kl_mean': float(np.mean(kl_values)),
            'kl_std': float(np.std(kl_values)),
            'js_mean': float(np.mean(js_values)),
            'js_std': float(np.std(js_values)),
        }

        print(f"\n  {category}:")
        print(f"    KL Divergence: {np.mean(kl_values):.4f} ± {np.std(kl_values):.4f}")
        print(f"    JS Divergence: {np.mean(js_values):.4f} ± {np.std(js_values):.4f}")

    return results


def main():
    print("\n" + "="*60)
    print("  KL Divergence 分布相似度评估")
    print("  不检测峰，直接比较预测分布和理想 GT 分布")
    print("  值越小 = 分布越相似 = 预测越好")
    print("="*60)

    # D 系列
    d_checkpoints = [
        'checkpoints/D_8a_20251218_171207',
        'checkpoints/D_8b_20251218_203046',
        'checkpoints/D_8c_20251218_234110',
        'checkpoints/D_8d_20251219_024151',
        'checkpoints/D_16a_20251219_053127',
        'checkpoints/D_16b_20251219_081853',
    ]

    # DR 系列
    dr_checkpoints = [
        'checkpoints/DR_8a_20251219_215002',
        'checkpoints/DR_8b_20251220_004513',
        'checkpoints/DR_16a_20251220_033656',
    ]

    all_results = {}

    print("\n" + "="*60)
    print("  D 系列")
    print("="*60)
    for ckpt in d_checkpoints:
        if Path(ckpt).exists():
            result = evaluate_checkpoint(ckpt)
            exp_name = Path(ckpt).name.split('_2025')[0]
            all_results[exp_name] = result

    print("\n" + "="*60)
    print("  DR 系列")
    print("="*60)
    for ckpt in dr_checkpoints:
        if Path(ckpt).exists():
            result = evaluate_checkpoint(ckpt)
            exp_name = Path(ckpt).name.split('_2025')[0]
            all_results[exp_name] = result

    # 汇总表
    print("\n" + "="*70)
    print("  汇总：KL Divergence (越小越好)")
    print("="*70)

    print(f"\n{'模型':<10} | {'1_front KL':>15} | {'4_fronts KL':>15} | {'平均 KL':>12}")
    print("-"*60)

    for exp_name, result in sorted(all_results.items()):
        kl1 = result.get('1_front', {}).get('kl_mean', float('nan'))
        kl4 = result.get('4_fronts', {}).get('kl_mean', float('nan'))
        avg_kl = (kl1 + kl4) / 2 if not (np.isnan(kl1) or np.isnan(kl4)) else float('nan')

        print(f"{exp_name:<10} | {kl1:>15.4f} | {kl4:>15.4f} | {avg_kl:>12.4f}")

    # 按平均 KL 排序
    print("\n" + "="*70)
    print("  排名 (按平均 KL，越小越好)")
    print("="*70)

    rankings = []
    for exp_name, result in all_results.items():
        kl1 = result.get('1_front', {}).get('kl_mean', float('nan'))
        kl4 = result.get('4_fronts', {}).get('kl_mean', float('nan'))
        if not (np.isnan(kl1) or np.isnan(kl4)):
            avg_kl = (kl1 + kl4) / 2
            rankings.append((exp_name, kl1, kl4, avg_kl))

    rankings.sort(key=lambda x: x[3])

    print(f"\n{'排名':>4} {'模型':<10} {'1_front':>12} {'4_fronts':>12} {'平均':>12}")
    print("-"*55)
    for i, (name, kl1, kl4, avg) in enumerate(rankings, 1):
        print(f"{i:>4} {name:<10} {kl1:>12.4f} {kl4:>12.4f} {avg:>12.4f}")

    # 保存结果
    with open('evaluation_kl_divergence.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 evaluation_kl_divergence.json")


if __name__ == '__main__':
    main()

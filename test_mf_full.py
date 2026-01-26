#!/usr/bin/env python3
"""
MF Full Model Test Script - 在测试集上评估MF模型

评估指标:
1. 1-front, 2-front, 4-front 的角度误差
2. no-front/rot-sym 的对称性识别准确率 (kappa阈值)

用法:
    python test_mf_full.py --checkpoint checkpoints/MF_Full_4Cat_20260107_234055/best_error.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from datasets.multi_category_dataset import MultiCategoryDataset, collate_fn
from test_direction_models import DirectionModel
from torch.utils.data import DataLoader


def circular_error(pred, gt):
    """计算循环角度误差 (弧度)"""
    pred = pred % (2 * np.pi)
    gt = gt % (2 * np.pi)
    diff = abs(pred - gt)
    return min(diff, 2 * np.pi - diff)


def hungarian_matching(pred_peaks, gt_peaks):
    """Hungarian匹配，返回每个GT peak的最小误差"""
    n_pred = len(pred_peaks)
    n_gt = len(gt_peaks)

    # 构建cost矩阵
    cost = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            cost[i, j] = circular_error(pred_peaks[i], gt_peaks[j])

    # Hungarian匹配
    row_ind, col_ind = linear_sum_assignment(cost)

    # 返回匹配后的误差
    errors = [cost[i, j] for i, j in zip(row_ind, col_ind)]
    return errors


def evaluate_2front_correct(pred_peaks, gt_angle):
    """
    正确评估2-front:
    - GT有2个方向: gt_angle 和 gt_angle + π
    - 从4个预测peaks中找最接近这2个GT方向的
    """
    gt_peaks = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]

    # 对每个GT peak，找最接近的pred peak
    errors = []
    used_pred = set()

    for gt in gt_peaks:
        best_err = float('inf')
        best_idx = -1
        for i, pred in enumerate(pred_peaks):
            if i in used_pred:
                continue
            err = circular_error(pred, gt)
            if err < best_err:
                best_err = err
                best_idx = i
        errors.append(best_err)
        if best_idx >= 0:
            used_pred.add(best_idx)

    return errors


class MFFullTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = DirectionModel(mode='mf', num_peaks=4).to(self.device)

        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.head.load_state_dict(checkpoint['head_state_dict'])
        self.model.eval()

        print(f"Loaded checkpoint: {args.checkpoint}")

        # Test dataset
        self.test_dataset = MultiCategoryDataset(
            annotation_file=args.annotation_file,
            data_root=args.data_root,
            split='test',
            categories=['1_front', '2_fronts', '4_fronts', 'no_front'],
            num_points=args.num_points,
            augment=False,
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_fn
        )

        print(f"Test samples: {len(self.test_dataset)}")

    @torch.no_grad()
    def evaluate(self):
        """在测试集上评估"""

        # 收集结果
        results_1f = []  # [(error, kappa), ...]
        results_2f = []
        results_4f = []
        results_nf = []  # [(max_kappa, mean_kappa, is_symmetric), ...]
        results_rs = []  # rot-sym (旋转对称)

        # 按类别统计
        category_counts = defaultdict(int)

        for batch in self.test_loader:
            points = batch['points'].to(self.device)
            gt_angle = batch['gt_angle'].numpy()
            categories = batch['category']

            mu, kappa = self.model(points)

            # 转换mu从向量到角度
            mu_angle = torch.atan2(mu[:, :, 1], mu[:, :, 0])
            mu_angle = (mu_angle + 2 * np.pi) % (2 * np.pi)
            mu_np = mu_angle.cpu().numpy()
            kappa_np = kappa.cpu().numpy()

            for i, cat in enumerate(categories):
                category_counts[cat] += 1
                pred_peaks = mu_np[i]  # (4,)
                pred_kappas = kappa_np[i]  # (4,)
                gt = gt_angle[i]

                if cat == '1_front':
                    # 取最接近GT的peak
                    errs = [circular_error(pred_peaks[j], gt) for j in range(4)]
                    best_idx = np.argmin(errs)
                    results_1f.append({
                        'error': errs[best_idx],
                        'kappa': pred_kappas[best_idx],
                        'all_kappas': pred_kappas.copy(),
                    })

                elif cat == '2_fronts':
                    # 正确的2-front评估
                    errors = evaluate_2front_correct(pred_peaks, gt)
                    results_2f.append({
                        'errors': errors,
                        'mean_error': np.mean(errors),
                        'kappas': pred_kappas.copy(),
                    })

                elif cat == '4_fronts':
                    # Hungarian匹配
                    gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    errors = hungarian_matching(pred_peaks.tolist(), gt_peaks)
                    results_4f.append({
                        'errors': errors,
                        'mean_error': np.mean(errors),
                        'kappas': pred_kappas.copy(),
                    })

                elif cat in ['no_front', 'symmetric']:
                    max_kappa = np.max(pred_kappas)
                    mean_kappa = np.mean(pred_kappas)
                    results_nf.append({
                        'max_kappa': max_kappa,
                        'mean_kappa': mean_kappa,
                        'all_kappas': pred_kappas.copy(),
                        'category': cat,
                    })

        return {
            '1_front': results_1f,
            '2_fronts': results_2f,
            '4_fronts': results_4f,
            'no_front': results_nf,
            'counts': dict(category_counts),
        }

    def print_results(self, results):
        """打印评估结果"""

        print("\n" + "=" * 70)
        print("MF Full Model - Test Set Evaluation")
        print("=" * 70)

        counts = results['counts']
        print(f"\nTest set distribution:")
        for cat, count in counts.items():
            print(f"  {cat}: {count}")

        # === 1-front Results ===
        print("\n" + "-" * 50)
        print("1-FRONT Angular Error")
        print("-" * 50)

        if results['1_front']:
            errors_deg = [r['error'] * 180 / np.pi for r in results['1_front']]
            kappas = [r['kappa'] for r in results['1_front']]

            print(f"  Samples: {len(errors_deg)}")
            print(f"  Mean Error: {np.mean(errors_deg):.2f}°")
            print(f"  Median Error: {np.median(errors_deg):.2f}°")
            print(f"  Std: {np.std(errors_deg):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")
            print(f"  Mean κ: {np.mean(kappas):.2f}")

        # === 2-front Results ===
        print("\n" + "-" * 50)
        print("2-FRONT Angular Error")
        print("-" * 50)

        if results['2_fronts']:
            all_errors = []
            for r in results['2_fronts']:
                all_errors.extend(r['errors'])
            errors_deg = [e * 180 / np.pi for e in all_errors]
            mean_errors = [r['mean_error'] * 180 / np.pi for r in results['2_fronts']]

            print(f"  Samples: {len(results['2_fronts'])} (x2 peaks = {len(errors_deg)} errors)")
            print(f"  Mean Error (per peak): {np.mean(errors_deg):.2f}°")
            print(f"  Median Error (per peak): {np.median(errors_deg):.2f}°")
            print(f"  Mean Error (per sample): {np.mean(mean_errors):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")

        # === 4-front Results ===
        print("\n" + "-" * 50)
        print("4-FRONT Angular Error")
        print("-" * 50)

        if results['4_fronts']:
            all_errors = []
            for r in results['4_fronts']:
                all_errors.extend(r['errors'])
            errors_deg = [e * 180 / np.pi for e in all_errors]
            mean_errors = [r['mean_error'] * 180 / np.pi for r in results['4_fronts']]

            print(f"  Samples: {len(results['4_fronts'])} (x4 peaks = {len(errors_deg)} errors)")
            print(f"  Mean Error (per peak): {np.mean(errors_deg):.2f}°")
            print(f"  Median Error (per peak): {np.median(errors_deg):.2f}°")
            print(f"  Mean Error (per sample): {np.mean(mean_errors):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")

        # === No-front / Symmetric Recognition ===
        print("\n" + "-" * 50)
        print("NO-FRONT / ROT-SYM Recognition (via κ threshold)")
        print("-" * 50)

        if results['no_front']:
            max_kappas = [r['max_kappa'] for r in results['no_front']]
            mean_kappas = [r['mean_kappa'] for r in results['no_front']]

            print(f"  Samples: {len(results['no_front'])}")
            print(f"\n  Max κ statistics:")
            print(f"    Mean: {np.mean(max_kappas):.3f}")
            print(f"    Median: {np.median(max_kappas):.3f}")
            print(f"    Min: {np.min(max_kappas):.3f}")
            print(f"    Max: {np.max(max_kappas):.3f}")

            print(f"\n  Mean κ statistics:")
            print(f"    Mean: {np.mean(mean_kappas):.3f}")
            print(f"    Median: {np.median(mean_kappas):.3f}")

            print(f"\n  Recognition Accuracy (κ_max < threshold):")
            for thresh in [0.1, 0.5, 1.0, 2.0, 5.0]:
                acc = 100 * np.mean(np.array(max_kappas) < thresh)
                print(f"    κ_max < {thresh}: {acc:.1f}%")

            print(f"\n  Recognition Accuracy (κ_mean < threshold):")
            for thresh in [0.1, 0.5, 1.0, 2.0]:
                acc = 100 * np.mean(np.array(mean_kappas) < thresh)
                print(f"    κ_mean < {thresh}: {acc:.1f}%")

        # === Overall Summary ===
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)

        # Directional error summary
        all_dir_errors = []
        if results['1_front']:
            all_dir_errors.extend([r['error'] * 180 / np.pi for r in results['1_front']])
        if results['2_fronts']:
            for r in results['2_fronts']:
                all_dir_errors.extend([e * 180 / np.pi for e in r['errors']])
        if results['4_fronts']:
            for r in results['4_fronts']:
                all_dir_errors.extend([e * 180 / np.pi for e in r['errors']])

        if all_dir_errors:
            print(f"\n  Combined Directional Error (1f + 2f + 4f):")
            print(f"    Mean: {np.mean(all_dir_errors):.2f}°")
            print(f"    Median: {np.median(all_dir_errors):.2f}°")
            print(f"    <10°: {100*np.mean(np.array(all_dir_errors) < 10):.1f}%")

        # Symmetry recognition summary
        if results['no_front']:
            max_kappas = [r['max_kappa'] for r in results['no_front']]
            print(f"\n  Symmetry Recognition (no_front + rot_sym):")
            print(f"    κ_max < 1.0: {100*np.mean(np.array(max_kappas) < 1.0):.1f}%")
            print(f"    κ_max < 0.5: {100*np.mean(np.array(max_kappas) < 0.5):.1f}%")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Test MF Full Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_root', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    tester = MFFullTester(args)
    results = tester.evaluate()
    tester.print_results(results)


if __name__ == '__main__':
    main()

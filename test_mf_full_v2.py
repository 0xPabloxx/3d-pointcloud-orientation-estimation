#!/usr/bin/env python3
"""
MF Full Model Test Script V2 - 改进版测试

改进点:
1. 测试时随机旋转，记录旋转后的GT方向
2. 统一使用Hungarian matching:
   - 1-front: GT复制4份，4个mu做hungarian matching
   - 2-front: 2个GT复制为4个，4个mu做hungarian matching
   - 4-front: 直接4个GT对4个mu做hungarian matching

用法:
    python test_mf_full_v2.py --checkpoint checkpoints/MF_Full_4Cat_20260107_234055/best_error.pth
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
    """
    Hungarian匹配，返回每个GT peak的最小误差

    Args:
        pred_peaks: (n,) 预测的角度
        gt_peaks: (m,) GT的角度

    Returns:
        errors: 每个匹配对的误差
    """
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


class MFFullTesterV2:
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

        # Test dataset - 注意这里不使用augment，我们手动做随机旋转
        self.test_dataset = MultiCategoryDataset(
            annotation_file=args.annotation_file,
            data_root=args.data_root,
            split='test',
            categories=['1_front', '2_fronts', '4_fronts', 'no_front'],
            num_points=args.num_points,
            augment=False,  # 不使用dataset的augment
        )

        # 不用DataLoader，手动处理以便做随机旋转
        print(f"Test samples: {len(self.test_dataset)}")

    def rotate_points_y(self, points, angle):
        """绕Y轴旋转点云"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        return points @ R.T

    @torch.no_grad()
    def evaluate(self, num_rotations=10):
        """
        在测试集上评估，每个样本测试num_rotations次随机旋转
        """

        # 收集结果
        results_1f = []
        results_2f = []
        results_4f = []
        results_nf = []

        # 按类别统计
        category_counts = defaultdict(int)

        rng = np.random.RandomState(self.args.seed)

        for idx in range(len(self.test_dataset)):
            sample = self.test_dataset.samples[idx]
            category = sample['category']
            base_angle = sample['angle']
            category_counts[category] += 1

            # 读取原始点云
            file_path = os.path.join(self.test_dataset.data_root, sample['file'])
            points = self._load_points(file_path)

            # 对每个样本进行num_rotations次随机旋转测试
            for _ in range(num_rotations):
                # 随机旋转角度
                rot_angle = rng.uniform(0, 2 * np.pi)

                # 旋转点云
                rotated_points = self.rotate_points_y(points, rot_angle)

                # 旋转后的GT方向
                rotated_gt = (base_angle + rot_angle) % (2 * np.pi)

                # 模型推理
                points_tensor = torch.from_numpy(rotated_points).float().unsqueeze(0).to(self.device)
                mu, kappa = self.model(points_tensor)

                # 转换mu从向量到角度
                mu_angle = torch.atan2(mu[:, :, 1], mu[:, :, 0])
                mu_angle = (mu_angle + 2 * np.pi) % (2 * np.pi)
                pred_peaks = mu_angle[0].cpu().numpy()  # (4,)
                pred_kappas = kappa[0].cpu().numpy()  # (4,)

                if category == '1_front':
                    # 1-front: GT复制4份，用Hungarian matching
                    gt_peaks = [rotated_gt] * 4
                    errors = hungarian_matching(pred_peaks.tolist(), gt_peaks)
                    results_1f.append({
                        'errors': errors,
                        'mean_error': np.mean(errors),
                        'kappas': pred_kappas.copy(),
                    })

                elif category == '2_fronts':
                    # 2-front: 2个GT复制成4个，用Hungarian matching
                    gt1 = rotated_gt
                    gt2 = (rotated_gt + np.pi) % (2 * np.pi)
                    gt_peaks = [gt1, gt1, gt2, gt2]  # 每个GT复制一份
                    errors = hungarian_matching(pred_peaks.tolist(), gt_peaks)
                    results_2f.append({
                        'errors': errors,
                        'mean_error': np.mean(errors),
                        'kappas': pred_kappas.copy(),
                    })

                elif category == '4_fronts':
                    # 4-front: 直接4个GT对4个mu
                    gt_peaks = [(rotated_gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    errors = hungarian_matching(pred_peaks.tolist(), gt_peaks)
                    results_4f.append({
                        'errors': errors,
                        'mean_error': np.mean(errors),
                        'kappas': pred_kappas.copy(),
                    })

                elif category in ['no_front', 'symmetric']:
                    max_kappa = np.max(pred_kappas)
                    mean_kappa = np.mean(pred_kappas)
                    results_nf.append({
                        'max_kappa': max_kappa,
                        'mean_kappa': mean_kappa,
                        'all_kappas': pred_kappas.copy(),
                        'category': category,
                    })

        return {
            '1_front': results_1f,
            '2_fronts': results_2f,
            '4_fronts': results_4f,
            'no_front': results_nf,
            'counts': dict(category_counts),
            'num_rotations': num_rotations,
        }

    def _load_points(self, file_path):
        """加载并预处理点云"""
        from plyfile import PlyData

        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

        # 随机采样到固定点数
        if len(points) >= self.args.num_points:
            indices = np.random.choice(len(points), self.args.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.args.num_points, replace=True)
        points = points[indices]

        # 归一化
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist

        return points.astype(np.float32)

    def print_results(self, results):
        """打印评估结果"""

        print("\n" + "=" * 70)
        print("MF Full Model - Test Set Evaluation V2 (with random rotation)")
        print("=" * 70)

        counts = results['counts']
        num_rot = results['num_rotations']
        print(f"\nTest set distribution (x{num_rot} rotations each):")
        for cat, count in counts.items():
            print(f"  {cat}: {count} samples x {num_rot} = {count * num_rot} evaluations")

        # === 1-front Results ===
        print("\n" + "-" * 50)
        print("1-FRONT Angular Error (GT copied to 4, Hungarian matching)")
        print("-" * 50)

        if results['1_front']:
            all_errors = []
            for r in results['1_front']:
                all_errors.extend(r['errors'])
            errors_deg = [e * 180 / np.pi for e in all_errors]
            mean_errors = [r['mean_error'] * 180 / np.pi for r in results['1_front']]

            print(f"  Evaluations: {len(results['1_front'])} (x4 peaks = {len(errors_deg)} errors)")
            print(f"  Mean Error (per peak): {np.mean(errors_deg):.2f}°")
            print(f"  Median Error (per peak): {np.median(errors_deg):.2f}°")
            print(f"  Mean Error (per sample): {np.mean(mean_errors):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")

        # === 2-front Results ===
        print("\n" + "-" * 50)
        print("2-FRONT Angular Error (2 GT copied to 4, Hungarian matching)")
        print("-" * 50)

        if results['2_fronts']:
            all_errors = []
            for r in results['2_fronts']:
                all_errors.extend(r['errors'])
            errors_deg = [e * 180 / np.pi for e in all_errors]
            mean_errors = [r['mean_error'] * 180 / np.pi for r in results['2_fronts']]

            print(f"  Evaluations: {len(results['2_fronts'])} (x4 peaks = {len(errors_deg)} errors)")
            print(f"  Mean Error (per peak): {np.mean(errors_deg):.2f}°")
            print(f"  Median Error (per peak): {np.median(errors_deg):.2f}°")
            print(f"  Mean Error (per sample): {np.mean(mean_errors):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")

        # === 4-front Results ===
        print("\n" + "-" * 50)
        print("4-FRONT Angular Error (4 GT vs 4 pred, Hungarian matching)")
        print("-" * 50)

        if results['4_fronts']:
            all_errors = []
            for r in results['4_fronts']:
                all_errors.extend(r['errors'])
            errors_deg = [e * 180 / np.pi for e in all_errors]
            mean_errors = [r['mean_error'] * 180 / np.pi for r in results['4_fronts']]

            print(f"  Evaluations: {len(results['4_fronts'])} (x4 peaks = {len(errors_deg)} errors)")
            print(f"  Mean Error (per peak): {np.mean(errors_deg):.2f}°")
            print(f"  Median Error (per peak): {np.median(errors_deg):.2f}°")
            print(f"  Mean Error (per sample): {np.mean(mean_errors):.2f}°")
            print(f"  <5°: {100*np.mean(np.array(errors_deg) < 5):.1f}%")
            print(f"  <10°: {100*np.mean(np.array(errors_deg) < 10):.1f}%")
            print(f"  <15°: {100*np.mean(np.array(errors_deg) < 15):.1f}%")
            print(f"  >45°: {100*np.mean(np.array(errors_deg) > 45):.1f}%")

        # === No-front / Symmetric Recognition ===
        print("\n" + "-" * 50)
        print("NO-FRONT / ROT-SYM Recognition (via kappa threshold)")
        print("-" * 50)

        if results['no_front']:
            max_kappas = [r['max_kappa'] for r in results['no_front']]
            mean_kappas = [r['mean_kappa'] for r in results['no_front']]

            print(f"  Evaluations: {len(results['no_front'])}")
            print(f"\n  Max kappa statistics:")
            print(f"    Mean: {np.mean(max_kappas):.3f}")
            print(f"    Median: {np.median(max_kappas):.3f}")
            print(f"    Min: {np.min(max_kappas):.3f}")
            print(f"    Max: {np.max(max_kappas):.3f}")

            print(f"\n  Recognition Accuracy (kappa_max < threshold):")
            for thresh in [0.1, 0.5, 1.0, 2.0, 5.0]:
                acc = 100 * np.mean(np.array(max_kappas) < thresh)
                print(f"    kappa_max < {thresh}: {acc:.1f}%")

        # === Overall Summary ===
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)

        # Directional error summary
        all_dir_errors = []
        if results['1_front']:
            for r in results['1_front']:
                all_dir_errors.extend([e * 180 / np.pi for e in r['errors']])
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
            print(f"    <15°: {100*np.mean(np.array(all_dir_errors) < 15):.1f}%")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Test MF Full Model V2')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_root', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_rotations', type=int, default=10,
                        help='Number of random rotations per sample')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    tester = MFFullTesterV2(args)
    results = tester.evaluate(num_rotations=args.num_rotations)
    tester.print_results(results)


if __name__ == '__main__':
    main()

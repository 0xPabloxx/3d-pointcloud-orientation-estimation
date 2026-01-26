#!/usr/bin/env python3
"""
测试 Symmetry Classifier 在测试集上的表现

使用 data/test_set/test_set.json 作为测试数据
评估 CleanClassifier 模型的分类准确率和其他指标

输出：
- 整体准确率
- 各类别准确率
- 混淆矩阵
- 分类报告 (precision, recall, F1)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from train_symmetry_classifier import SymmetryClassifier


# 类别映射
CLASS_NAMES = ['1个正面', '2个正面', '4个正面', '旋转对称', '无正面']

# 使用 symmetry_name 而不是 K 值，因为原始标注中 K 值定义有问题
SYMMETRY_NAME_TO_CLASS = {
    '1个正面': 0,
    '1_front': 0,
    '2个正面': 1,
    '2_fronts': 1,
    '4个正面': 2,
    '4_fronts': 2,
    '旋转对称': 3,
    '完全对称': 3,
    'symmetric': 3,
    '无正面': 4,
    '没有正面': 4,
    'no_front': 4,
}


def load_ply(ply_path: str) -> np.ndarray:
    """加载PLY文件"""
    with open(ply_path, 'r') as f:
        lines = f.readlines()

    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if 'element vertex' in line:
            num_vertices = int(line.split()[-1])
        if 'end_header' in line:
            header_end = i + 1
            break

    points = []
    for line in lines[header_end:header_end + num_vertices]:
        coords = line.strip().split()[:3]
        points.append([float(x) for x in coords])

    return np.array(points, dtype=np.float32)


def preprocess_points(points: np.ndarray, num_points: int = 2048) -> np.ndarray:
    """预处理点云：采样和归一化"""
    # 采样
    if len(points) > num_points:
        choice = np.random.choice(len(points), num_points, replace=False)
        points = points[choice]
    elif len(points) < num_points:
        choice = np.random.choice(len(points), num_points, replace=True)
        points = points[choice]

    # 归一化到单位球
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    return points


def compute_metrics(y_true, y_pred, class_names):
    """计算分类指标"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    # 整体准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # 各类别准确率
    class_accuracy = {}
    for i, name in enumerate(class_names):
        mask = np.array(y_true) == i
        if mask.sum() > 0:
            class_accuracy[name] = (np.array(y_pred)[mask] == i).mean()
        else:
            class_accuracy[name] = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'class_accuracy': class_accuracy,
    }


def test_classifier(checkpoint_path, test_set_path, data_root, num_rotations=1, device='cuda'):
    """测试分类器"""
    print("=" * 80)
    print("Symmetry Classifier 测试")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test set: {test_set_path}")
    print(f"Data root: {data_root}")
    print(f"Num rotations: {num_rotations}")
    print(f"Device: {device}")
    print()

    # 加载模型
    print("Loading model...")
    model = SymmetryClassifier(num_classes=5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 处理不同格式的checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 加载测试集
    print("\nLoading test set...")
    with open(test_set_path, 'r') as f:
        test_samples = json.load(f)
    print(f"Total samples: {len(test_samples)}")

    # 统计类别分布
    class_counts = defaultdict(int)
    for sample in test_samples:
        class_counts[sample['symmetry_name']] += 1
    print("Class distribution:")
    for name in CLASS_NAMES:
        print(f"  {name}: {class_counts.get(name, 0)}")

    # 测试
    print("\nRunning inference...")
    y_true = []
    y_pred = []
    y_prob = []
    predictions = []

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processing {i+1}/{len(test_samples)}...", flush=True)

            file_path = sample['file']
            symmetry_name = sample['symmetry_name']

            # 获取真实标签 (使用 symmetry_name 而不是 K)
            if symmetry_name not in SYMMETRY_NAME_TO_CLASS:
                print(f"  [Warning] Unknown symmetry_name='{symmetry_name}' for {file_path}, skipping")
                continue

            true_class = SYMMETRY_NAME_TO_CLASS[symmetry_name]

            # 加载点云
            ply_path = os.path.join(data_root, file_path)
            if not os.path.exists(ply_path):
                print(f"  [Warning] File not found: {ply_path}")
                continue

            points = load_ply(ply_path)

            # 多次旋转测试取平均
            all_probs = []
            for rot in range(num_rotations):
                # 随机Y轴旋转
                if rot > 0:
                    angle = np.random.uniform(0, 2 * np.pi)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    R = np.array([
                        [cos_a, 0, sin_a],
                        [0, 1, 0],
                        [-sin_a, 0, cos_a]
                    ])
                    rotated_points = points @ R.T
                else:
                    rotated_points = points.copy()

                # 预处理
                processed = preprocess_points(rotated_points)
                points_tensor = torch.from_numpy(processed).float().unsqueeze(0).to(device)

                # 推理 (SymmetryClassifier 不需要 upright_vec)
                logits = model(points_tensor)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

            # 平均概率
            avg_probs = np.mean(all_probs, axis=0)[0]
            pred_class = np.argmax(avg_probs)

            y_true.append(true_class)
            y_pred.append(pred_class)
            y_prob.append(avg_probs)

            predictions.append({
                'file': file_path,
                'category': sample.get('category', ''),
                'true_class': int(true_class),
                'true_name': CLASS_NAMES[true_class],
                'pred_class': int(pred_class),
                'pred_name': CLASS_NAMES[pred_class],
                'correct': true_class == pred_class,
                'probabilities': avg_probs.tolist(),
            })

    # 计算指标
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    print(f"\n总体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    print("\n各类别指标:")
    print("-" * 70)
    print(f"{'类别':<12} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'样本数':>8}")
    print("-" * 70)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} {metrics['class_accuracy'][name]:>10.4f} "
              f"{metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} "
              f"{metrics['f1'][i]:>10.4f} {metrics['support'][i]:>8}")
    print("-" * 70)

    # 宏平均
    macro_precision = np.mean(metrics['precision'])
    macro_recall = np.mean(metrics['recall'])
    macro_f1 = np.mean(metrics['f1'])
    print(f"{'宏平均':<12} {'':>10} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f}")

    # 加权平均
    weights = metrics['support'] / metrics['support'].sum()
    weighted_precision = np.sum(metrics['precision'] * weights)
    weighted_recall = np.sum(metrics['recall'] * weights)
    weighted_f1 = np.sum(metrics['f1'] * weights)
    print(f"{'加权平均':<12} {'':>10} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f}")

    print("\n混淆矩阵:")
    print("-" * 70)
    header = "真实\\预测  " + "  ".join([f"{name[:4]:>6}" for name in CLASS_NAMES])
    print(header)
    print("-" * 70)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name[:8]:<10} " + "  ".join([f"{metrics['confusion_matrix'][i,j]:>6}" for j in range(len(CLASS_NAMES))])
        print(row)

    # 按类别统计错误
    print("\n错误分析 (按类别):")
    print("-" * 70)
    errors_by_category = defaultdict(list)
    for pred in predictions:
        if not pred['correct']:
            errors_by_category[pred['category']].append(pred)

    for category in sorted(errors_by_category.keys()):
        errors = errors_by_category[category]
        print(f"  {category}: {len(errors)} 个错误")
        for err in errors[:3]:  # 只显示前3个
            print(f"    - {err['file']}: {err['true_name']} -> {err['pred_name']}")
        if len(errors) > 3:
            print(f"    ... 还有 {len(errors)-3} 个")

    return metrics, predictions


def save_results(metrics, predictions, output_dir, checkpoint_name):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果
    results = {
        'timestamp': timestamp,
        'checkpoint': checkpoint_name,
        'accuracy': float(metrics['accuracy']),
        'class_accuracy': {k: float(v) for k, v in metrics['class_accuracy'].items()},
        'precision': metrics['precision'].tolist(),
        'recall': metrics['recall'].tolist(),
        'f1': metrics['f1'].tolist(),
        'support': metrics['support'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'class_names': CLASS_NAMES,
        'predictions': predictions,
    }

    result_file = os.path.join(output_dir, f'classifier_test_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {result_file}")

    # 保存简要摘要
    summary_file = os.path.join(output_dir, f'classifier_summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Symmetry Classifier 测试结果摘要\n")
        f.write("=" * 80 + "\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"模型: {checkpoint_name}\n")
        f.write(f"\n总体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write("\n各类别准确率:\n")
        for name in CLASS_NAMES:
            f.write(f"  {name}: {metrics['class_accuracy'][name]:.4f}\n")
        f.write("\n混淆矩阵:\n")
        f.write("真实\\预测  " + "  ".join([f"{name[:4]:>6}" for name in CLASS_NAMES]) + "\n")
        for i, name in enumerate(CLASS_NAMES):
            row = f"{name[:8]:<10} " + "  ".join([f"{metrics['confusion_matrix'][i,j]:>6}" for j in range(len(CLASS_NAMES))])
            f.write(row + "\n")
    print(f"摘要已保存到: {summary_file}")

    return result_file, summary_file


def run_multi_augmentation_experiments(checkpoint_path, test_set_path, data_root, output_dir, device):
    """运行多组数据增强实验 (1x, 8x, 10x, 12x)"""
    augmentation_factors = [1, 8, 10, 12]
    all_results = {}

    print("\n" + "=" * 80)
    print("多组数据增强实验")
    print("=" * 80)

    for num_rot in augmentation_factors:
        print(f"\n{'='*60}")
        print(f"实验: {num_rot}x 数据增强 (TTA)")
        print(f"{'='*60}")

        metrics, predictions = test_classifier(
            checkpoint_path=checkpoint_path,
            test_set_path=test_set_path,
            data_root=data_root,
            num_rotations=num_rot,
            device=device,
        )

        all_results[f'{num_rot}x'] = {
            'num_rotations': num_rot,
            'accuracy': float(metrics['accuracy']),
            'class_accuracy': {k: float(v) for k, v in metrics['class_accuracy'].items()},
            'macro_f1': float(np.mean(metrics['f1'])),
            'weighted_f1': float(np.sum(metrics['f1'] * metrics['support'] / metrics['support'].sum())),
        }

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("多组实验结果汇总")
    print("=" * 80)
    print(f"\n{'增强倍数':<12} {'总体准确率':>12} {'Macro F1':>12} {'Weighted F1':>12}")
    print("-" * 50)
    for key, result in all_results.items():
        print(f"{key:<12} {result['accuracy']*100:>11.2f}% {result['macro_f1']:>12.4f} {result['weighted_f1']:>12.4f}")

    print("\n各类别准确率:")
    print(f"{'增强倍数':<10}", end="")
    for name in CLASS_NAMES:
        print(f" {name[:6]:>8}", end="")
    print()
    print("-" * 60)
    for key, result in all_results.items():
        print(f"{key:<10}", end="")
        for name in CLASS_NAMES:
            print(f" {result['class_accuracy'].get(name, 0)*100:>7.1f}%", end="")
        print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test Symmetry Classifier on test set')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/CleanClassifier_20251229_220630/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test_set', type=str,
                        default='data/test_set/test_set.json',
                        help='Path to test set JSON')
    parser.add_argument('--data_root', type=str,
                        default='data/full_mn40_normal_resampled_ply',
                        help='Root directory of point cloud data')
    parser.add_argument('--num_rotations', type=int, default=10,
                        help='Number of random rotations for test-time augmentation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--multi_aug', action='store_true',
                        help='Run multiple augmentation experiments (1x, 8x, 10x, 12x)')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.multi_aug:
        # 运行多组实验
        all_results = run_multi_augmentation_experiments(
            checkpoint_path=args.checkpoint,
            test_set_path=args.test_set,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=args.device,
        )

        # 保存汇总结果
        summary = {
            'timestamp': timestamp,
            'checkpoint': checkpoint_name,
            'experiments': all_results,
        }
        summary_file = os.path.join(args.output_dir, f'classifier_multi_aug_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果已保存到: {summary_file}")

        # 保存可读文本报告
        report_file = os.path.join(args.output_dir, f'classifier_multi_aug_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Symmetry Classifier 多组数据增强实验报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"模型: {checkpoint_name}\n")
            f.write("\n结果汇总:\n")
            f.write(f"{'增强倍数':<12} {'总体准确率':>12} {'Macro F1':>12} {'Weighted F1':>12}\n")
            f.write("-" * 50 + "\n")
            for key, result in all_results.items():
                f.write(f"{key:<12} {result['accuracy']*100:>11.2f}% {result['macro_f1']:>12.4f} {result['weighted_f1']:>12.4f}\n")
            f.write("\n各类别准确率:\n")
            header = f"{'增强倍数':<10}"
            for name in CLASS_NAMES:
                header += f" {name[:6]:>8}"
            f.write(header + "\n")
            f.write("-" * 60 + "\n")
            for key, result in all_results.items():
                line = f"{key:<10}"
                for name in CLASS_NAMES:
                    line += f" {result['class_accuracy'].get(name, 0)*100:>7.1f}%"
                f.write(line + "\n")
        print(f"报告已保存到: {report_file}")

    else:
        # 运行单次测试
        metrics, predictions = test_classifier(
            checkpoint_path=args.checkpoint,
            test_set_path=args.test_set,
            data_root=args.data_root,
            num_rotations=args.num_rotations,
            device=args.device,
        )

        # 保存结果
        save_results(metrics, predictions, args.output_dir, checkpoint_name)

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

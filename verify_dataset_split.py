#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分验证脚本

功能：
- 验证train/val/test划分是否正确
- 检查是否有数据泄露（样本重复出现在多个集合）
- 验证划分比例是否符合预期
- 检查随机种子是否固定（多次运行结果相同）

使用方法：
    python verify_dataset_split.py

作者: Claude
创建日期: 2025-11-25
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataloader_symmetry import SymmetryDataset


def verify_split_integrity(annotation_file, data_dir):
    """验证数据集划分的完整性"""

    print("=" * 80)
    print("数据集划分验证")
    print("=" * 80)

    # 创建3个数据集
    train_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='train',
        num_points=10000,
        augment=True
    )

    val_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='val',
        num_points=10000,
        augment=False
    )

    test_dataset = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='test',
        num_points=10000,
        augment=False
    )

    # 获取样本列表
    train_samples = set(train_dataset.samples)
    val_samples = set(val_dataset.samples)
    test_samples = set(test_dataset.samples)

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)

    print(f"\n📊 数据集大小:")
    print(f"  Train: {len(train_samples):4d} ({len(train_samples)/total_samples*100:5.1f}%)")
    print(f"  Val:   {len(val_samples):4d} ({len(val_samples)/total_samples*100:5.1f}%)")
    print(f"  Test:  {len(test_samples):4d} ({len(test_samples)/total_samples*100:5.1f}%)")
    print(f"  Total: {total_samples:4d}")

    # 1. 检查样本是否有重复
    print(f"\n🔍 检查1: 数据泄露检测（样本重复）")
    train_val_overlap = train_samples & val_samples
    train_test_overlap = train_samples & test_samples
    val_test_overlap = val_samples & test_samples

    has_leak = False
    if train_val_overlap:
        print(f"  ❌ 错误: Train和Val有 {len(train_val_overlap)} 个重复样本!")
        print(f"     重复样本: {list(train_val_overlap)[:5]}...")
        has_leak = True

    if train_test_overlap:
        print(f"  ❌ 错误: Train和Test有 {len(train_test_overlap)} 个重复样本!")
        print(f"     重复样本: {list(train_test_overlap)[:5]}...")
        has_leak = True

    if val_test_overlap:
        print(f"  ❌ 错误: Val和Test有 {len(val_test_overlap)} 个重复样本!")
        print(f"     重复样本: {list(val_test_overlap)[:5]}...")
        has_leak = True

    if not has_leak:
        print(f"  ✅ 通过: 没有数据泄露，所有样本唯一")

    # 2. 检查样本总数是否正确
    print(f"\n🔍 检查2: 样本总数")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)

    # 过滤有效标注
    valid_annotations = {
        k: v for k, v in all_annotations.items()
        if v.get('K') is not None and v.get('front_direction')
    }

    union_samples = train_samples | val_samples | test_samples

    if len(union_samples) == len(valid_annotations):
        print(f"  ✅ 通过: 所有 {len(valid_annotations)} 个样本都被正确划分")
    else:
        print(f"  ❌ 错误: 划分后样本数 ({len(union_samples)}) != 标注总数 ({len(valid_annotations)})")
        missing = set(valid_annotations.keys()) - union_samples
        extra = union_samples - set(valid_annotations.keys())
        if missing:
            print(f"     缺失样本: {list(missing)[:5]}...")
        if extra:
            print(f"     多余样本: {list(extra)[:5]}...")

    # 3. 检查划分比例
    print(f"\n🔍 检查3: 划分比例")
    expected_train_ratio = 0.70
    expected_val_ratio = 0.15
    expected_test_ratio = 0.15

    actual_train_ratio = len(train_samples) / total_samples
    actual_val_ratio = len(val_samples) / total_samples
    actual_test_ratio = len(test_samples) / total_samples

    tolerance = 0.05  # 允许5%误差

    ratio_ok = True
    if abs(actual_train_ratio - expected_train_ratio) > tolerance:
        print(f"  ⚠️  警告: Train比例 {actual_train_ratio:.2f} 偏离预期 {expected_train_ratio:.2f}")
        ratio_ok = False

    if abs(actual_val_ratio - expected_val_ratio) > tolerance:
        print(f"  ⚠️  警告: Val比例 {actual_val_ratio:.2f} 偏离预期 {expected_val_ratio:.2f}")
        ratio_ok = False

    if abs(actual_test_ratio - expected_test_ratio) > tolerance:
        print(f"  ⚠️  警告: Test比例 {actual_test_ratio:.2f} 偏离预期 {expected_test_ratio:.2f}")
        ratio_ok = False

    if ratio_ok:
        print(f"  ✅ 通过: 划分比例符合预期 (70%/15%/15%)")
    else:
        print(f"  提示: 样本数少时，比例可能略有偏差，属正常现象")

    # 4. 检查类别分布
    print(f"\n🔍 检查4: 类别分布（分层采样验证）")

    def get_K_distribution(samples, annotations):
        """统计K值分布"""
        K_counts = defaultdict(int)
        for sample in samples:
            K = annotations[sample]['K']
            K_counts[K] += 1
        return K_counts

    train_K_dist = get_K_distribution(train_samples, valid_annotations)
    val_K_dist = get_K_distribution(val_samples, valid_annotations)
    test_K_dist = get_K_distribution(test_samples, valid_annotations)

    print(f"\n  K值分布:")
    print(f"  {'K值':<6} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print(f"  {'-'*40}")

    all_K_values = sorted(set(train_K_dist.keys()) | set(val_K_dist.keys()) | set(test_K_dist.keys()))
    for K in all_K_values:
        t = train_K_dist.get(K, 0)
        v = val_K_dist.get(K, 0)
        e = test_K_dist.get(K, 0)
        total = t + v + e
        print(f"  K={K:<4} {t:<8} {v:<8} {e:<8} {total:<8}")

    # 5. 检查可复现性（随机种子）
    print(f"\n🔍 检查5: 可复现性验证（固定随机种子）")

    # 再次创建数据集
    train_dataset_2 = SymmetryDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='train',
        num_points=10000,
        augment=True
    )

    train_samples_2 = set(train_dataset_2.samples)

    if train_samples == train_samples_2:
        print(f"  ✅ 通过: 多次运行划分结果相同（seed=42固定）")
    else:
        print(f"  ❌ 错误: 多次运行划分结果不同！随机种子可能未固定！")
        diff = train_samples ^ train_samples_2
        print(f"     差异样本数: {len(diff)}")

    # 6. 检查数据增强设置
    print(f"\n🔍 检查6: 数据增强设置")
    if train_dataset.augment:
        print(f"  ✅ Train: 使用旋转增强")
    else:
        print(f"  ⚠️  Train: 未使用增强（可能不是最优设置）")

    if not val_dataset.augment:
        print(f"  ✅ Val: 不使用增强")
    else:
        print(f"  ❌ Val: 使用了增强（错误！验证集不应增强）")

    if not test_dataset.augment:
        print(f"  ✅ Test: 不使用增强")
    else:
        print(f"  ❌ Test: 使用了增强（严重错误！测试集不应增强）")

    # 总结
    print(f"\n" + "=" * 80)
    print(f"验证总结")
    print(f"=" * 80)

    all_checks_passed = (
        not has_leak and
        len(union_samples) == len(valid_annotations) and
        train_samples == train_samples_2 and
        not val_dataset.augment and
        not test_dataset.augment
    )

    if all_checks_passed:
        print(f"\n✅ 所有检查通过！数据集划分正确，可以开始训练。")
    else:
        print(f"\n⚠️  发现问题！请检查上述错误并修复。")

    return all_checks_passed


def main():
    annotation_file = 'data_annotation/symmetry_annotations.json'
    data_dir = 'data/full_mn40_normal_resampled_ply'

    # 检查文件是否存在
    if not Path(annotation_file).exists():
        print(f"❌ 错误: 标注文件不存在: {annotation_file}")
        return

    if not Path(data_dir).exists():
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        return

    # 运行验证
    verify_split_integrity(annotation_file, data_dir)

    print(f"\n💡 提示:")
    print(f"  - 如果所有检查通过，可以开始训练实验")
    print(f"  - 如果发现数据泄露，必须修复后重新训练")
    print(f"  - 任何在测试集上的评估都只能运行一次！")
    print(f"")


if __name__ == '__main__':
    main()

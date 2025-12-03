#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量标注glassbox为4个正面

功能：
- 自动为glassbox类别的所有样本标注
- K=4 (4个正面)
- front_direction='-Z' (正面为-Z方向)
- aligned=True (无需矫正)
- 不丢失其他已有标注

使用方法：
    python batch_annotate_glassbox.py

作者: Claude
创建日期: 2025-11-25
"""

import json
from pathlib import Path


def batch_annotate_glassbox(annotation_file, data_dir, dry_run=False):
    """批量标注glassbox类别

    Args:
        annotation_file: 标注文件路径
        data_dir: 数据集根目录
        dry_run: 如果为True，只显示将要标注的样本，不实际保存
    """

    annotation_file = Path(annotation_file)
    data_dir = Path(data_dir)

    # 1. 加载现有标注
    print("=" * 80)
    print("批量标注glassbox")
    print("=" * 80)

    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"\n📂 加载现有标注: {len(annotations)} 个样本")
    else:
        annotations = {}
        print(f"\n📂 创建新标注文件")

    # 统计现有标注
    existing_by_category = {}
    for file_path, ann in annotations.items():
        category = file_path.split('/')[0]
        existing_by_category[category] = existing_by_category.get(category, 0) + 1

    print(f"\n当前标注统计:")
    for category, count in sorted(existing_by_category.items()):
        print(f"  {category}: {count} 个")

    # 2. 扫描glassbox目录
    glass_box_dir = data_dir / 'glass_box'

    if not glass_box_dir.exists():
        print(f"\n❌ 错误: glassbox目录不存在: {glass_box_dir}")
        return

    ply_files = sorted(glass_box_dir.glob('*.ply'))
    print(f"\n🔍 扫描glassbox目录: 找到 {len(ply_files)} 个.ply文件")

    # 3. 批量标注
    new_annotations = 0
    updated_annotations = 0
    skipped_annotations = 0

    for ply_file in ply_files:
        # 相对路径（相对于data_dir）
        relative_path = f"glass_box/{ply_file.name}"

        # 检查是否已标注
        if relative_path in annotations:
            existing_ann = annotations[relative_path]
            # 如果已经标注为K=4且front_direction=-Z，跳过
            if existing_ann.get('K') == 4 and existing_ann.get('front_direction') == '-Z':
                skipped_annotations += 1
                continue
            else:
                # 更新标注
                updated_annotations += 1
                action = "更新"
        else:
            # 新标注
            new_annotations += 1
            action = "新增"

        # 添加标注
        annotations[relative_path] = {
            'K': 4,
            'symmetry_name': '4个正面',
            'front_direction': '-Z',
            'aligned': True,
            'file': relative_path
        }

        if dry_run:
            print(f"  [{action}] {relative_path}")

    # 4. 统计结果
    print(f"\n📊 标注统计:")
    print(f"  新增标注: {new_annotations} 个")
    print(f"  更新标注: {updated_annotations} 个")
    print(f"  跳过（已标注）: {skipped_annotations} 个")
    print(f"  总标注数: {len(annotations)} 个")

    # 5. 验证没有丢失数据
    final_by_category = {}
    for file_path, ann in annotations.items():
        category = file_path.split('/')[0]
        final_by_category[category] = final_by_category.get(category, 0) + 1

    print(f"\n最终标注统计:")
    for category, count in sorted(final_by_category.items()):
        old_count = existing_by_category.get(category, 0)
        if old_count > 0 and count != old_count:
            print(f"  {category}: {count} 个 (原来: {old_count}) ⚠️  数量变化!")
        else:
            print(f"  {category}: {count} 个")

    # 6. 保存
    if dry_run:
        print(f"\n⚠️  DRY RUN 模式，未保存文件")
        print(f"如果确认无误，去掉 --dry-run 参数重新运行")
        return annotations, False
    else:
        # 备份原文件
        if annotation_file.exists():
            backup_file = annotation_file.with_suffix('.json.backup')
            with open(annotation_file, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(backup_data)
            print(f"\n💾 备份原文件: {backup_file}")

        # 保存新标注
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"✅ 保存标注文件: {annotation_file}")
        print(f"总标注数: {len(annotations)} 个")

        return annotations, True


def verify_annotations(annotations):
    """验证标注完整性"""
    print(f"\n" + "=" * 80)
    print("验证标注完整性")
    print("=" * 80)

    # 统计各类别
    category_stats = {}
    K_stats = {}

    for file_path, ann in annotations.items():
        category = file_path.split('/')[0]
        K = ann.get('K')
        front_dir = ann.get('front_direction')

        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'K_distribution': {},
                'direction_distribution': {}
            }

        category_stats[category]['total'] += 1

        if K is not None:
            category_stats[category]['K_distribution'][K] = \
                category_stats[category]['K_distribution'].get(K, 0) + 1
            K_stats[K] = K_stats.get(K, 0) + 1

        if front_dir:
            category_stats[category]['direction_distribution'][front_dir] = \
                category_stats[category]['direction_distribution'].get(front_dir, 0) + 1

    # 打印统计
    print(f"\n各类别统计:")
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(f"\n  {category}: {stats['total']} 个")
        print(f"    K值分布: {stats['K_distribution']}")
        print(f"    方向分布: {stats['direction_distribution']}")

    print(f"\n总体K值分布:")
    for K in sorted(K_stats.keys()):
        print(f"  K={K}: {K_stats[K]} 个")

    # 检查glassbox
    if 'glass_box' in category_stats:
        gb_stats = category_stats['glass_box']
        print(f"\n✅ glassbox标注验证:")
        print(f"  总数: {gb_stats['total']} 个")

        if gb_stats['K_distribution'].get(4) == gb_stats['total']:
            print(f"  ✅ 所有样本都标注为K=4")
        else:
            print(f"  ⚠️  部分样本K值不是4: {gb_stats['K_distribution']}")

        if gb_stats['direction_distribution'].get('-Z') == gb_stats['total']:
            print(f"  ✅ 所有样本方向都为-Z")
        else:
            print(f"  ⚠️  部分样本方向不是-Z: {gb_stats['direction_distribution']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量标注glassbox为4个正面')
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json',
                        help='标注文件路径')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply',
                        help='数据集根目录')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示将要标注的样本，不实际保存')

    args = parser.parse_args()

    # 执行批量标注
    annotations, saved = batch_annotate_glassbox(
        args.annotation_file,
        args.data_dir,
        dry_run=args.dry_run
    )

    if annotations:
        # 验证标注
        verify_annotations(annotations)

    if saved:
        print(f"\n" + "=" * 80)
        print(f"✅ 批量标注完成!")
        print(f"=" * 80)
        print(f"\n可以运行以下命令验证数据集划分:")
        print(f"  python verify_dataset_split.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出标注为CSV和Markdown格式

从JSON导出到CSV和Markdown，用于查看和分析

使用方法：
    python export_annotations.py

作者: Claude
创建日期: 2025-11-25
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def export_to_csv(annotations, csv_file):
    """导出为CSV格式"""
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['file', 'K', 'symmetry_name', 'front_direction', 'aligned'])

        # Data
        for file_path in sorted(annotations.keys()):
            ann = annotations[file_path]
            writer.writerow([
                file_path,
                ann.get('K', ''),
                ann.get('symmetry_name', ''),
                ann.get('front_direction', ''),
                ann.get('aligned', '')
            ])

    print(f"✅ 导出CSV: {csv_file} ({len(annotations)} 条)")


def export_to_markdown(annotations, md_file):
    """导出为Markdown格式"""

    # 统计信息
    category_stats = defaultdict(lambda: {
        'total': 0,
        'K_dist': defaultdict(int),
        'dir_dist': defaultdict(int),
        'aligned': 0,
        'not_aligned': 0
    })

    K_stats = defaultdict(int)
    dir_stats = defaultdict(int)

    for file_path, ann in annotations.items():
        category = file_path.split('/')[0]
        K = ann.get('K')
        front_dir = ann.get('front_direction')
        aligned = ann.get('aligned', False)

        category_stats[category]['total'] += 1

        if K is not None:
            category_stats[category]['K_dist'][K] += 1
            K_stats[K] += 1

        if front_dir:
            category_stats[category]['dir_dist'][front_dir] += 1
            dir_stats[front_dir] += 1

        if aligned:
            category_stats[category]['aligned'] += 1
        else:
            category_stats[category]['not_aligned'] += 1

    # 写入Markdown
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 对称性标注报告\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"**总标注数**: {len(annotations)} 个\n\n")

        # K值分布
        f.write("### K值分布\n\n")
        f.write("| K值 | 含义 | 数量 |\n")
        f.write("|-----|------|------|\n")
        K_names = {
            -1: "没有正面",
            0: "完全对称",
            1: "1个正面",
            2: "2个正面",
            4: "4个正面"
        }
        for K in sorted(K_stats.keys()):
            name = K_names.get(K, "未知")
            count = K_stats[K]
            f.write(f"| {K} | {name} | {count} |\n")

        # 方向分布
        f.write("\n### 正面方向分布\n\n")
        f.write("| 方向 | 数量 |\n")
        f.write("|------|------|\n")
        for direction in sorted(dir_stats.keys()):
            count = dir_stats[direction]
            f.write(f"| {direction} | {count} |\n")

        # 各类别统计
        f.write("\n## 各类别统计\n\n")

        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            f.write(f"### {category}\n\n")
            f.write(f"**总数**: {stats['total']} 个\n\n")

            # K值分布
            if stats['K_dist']:
                f.write("**K值分布**:\n")
                for K in sorted(stats['K_dist'].keys()):
                    count = stats['K_dist'][K]
                    name = K_names.get(K, "未知")
                    f.write(f"- K={K} ({name}): {count} 个\n")
                f.write("\n")

            # 方向分布
            if stats['dir_dist']:
                f.write("**方向分布**:\n")
                for direction in sorted(stats['dir_dist'].keys()):
                    count = stats['dir_dist'][direction]
                    f.write(f"- {direction}: {count} 个\n")
                f.write("\n")

            # 对齐情况
            f.write(f"**对齐情况**: {stats['aligned']} 已对齐, {stats['not_aligned']} 未对齐\n\n")

        # 详细列表
        f.write("## 详细标注列表\n\n")
        f.write("| 文件 | K值 | 对称性 | 正面方向 | 已对齐 |\n")
        f.write("|------|-----|--------|----------|--------|\n")

        for file_path in sorted(annotations.keys())[:100]:  # 只显示前100个
            ann = annotations[file_path]
            f.write(f"| {file_path} | {ann.get('K', '')} | {ann.get('symmetry_name', '')} | "
                   f"{ann.get('front_direction', '')} | {ann.get('aligned', '')} |\n")

        if len(annotations) > 100:
            f.write(f"\n*（共 {len(annotations)} 个标注，仅显示前100个）*\n")

    print(f"✅ 导出Markdown: {md_file}")


def main():
    annotation_file = Path('symmetry_annotations.json')
    csv_file = Path('symmetry_annotations.csv')
    md_file = Path('symmetry_annotations.md')

    if not annotation_file.exists():
        print(f"❌ 标注文件不存在: {annotation_file}")
        return

    # 读取JSON
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"📂 读取标注: {len(annotations)} 个样本")

    # 导出
    export_to_csv(annotations, csv_file)
    export_to_markdown(annotations, md_file)

    print(f"\n✅ 导出完成!")


if __name__ == '__main__':
    main()

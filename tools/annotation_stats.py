#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注速度与时间统计工具

功能：
- 统计标注速度（样本/小时）
- 估算完成时间
- 显示各类别标注进度
- 轻量级，不占用太多资源

使用方法：
    python annotation_stats.py

作者: Claude
创建日期: 2025-11-25
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def load_annotations(annotation_file):
    """加载标注数据"""
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def calculate_annotation_speed(annotations):
    """
    计算标注速度
    假设每个标注保存时都包含index字段，可以粗略估算
    """
    if not annotations:
        return None, None, None

    # 读取CSV文件获取时间戳（如果有的话）
    # 由于JSON没有时间戳，我们用文件修改时间粗略估算
    json_file = Path("/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations.json")

    if json_file.exists():
        # 文件最后修改时间
        last_modified = datetime.fromtimestamp(json_file.stat().st_mtime)

        # 简单估算：假设从今天开始标注
        # （实际情况需要记录每个标注的时间戳）
        return last_modified, None, None

    return None, None, None


def scan_data_directory(data_dir):
    """扫描数据目录统计总样本数"""
    categories = {}
    total_samples = 0

    for cat_dir in sorted(data_dir.iterdir()):
        if cat_dir.is_dir():
            ply_files = list(cat_dir.glob("*.ply"))
            if ply_files:
                categories[cat_dir.name] = len(ply_files)
                total_samples += len(ply_files)

    return categories, total_samples


def calculate_progress_stats(annotations, categories):
    """计算各类别标注进度"""
    category_stats = {}

    for cat_name, total in categories.items():
        annotated = 0
        for file_path, ann in annotations.items():
            if file_path.startswith(cat_name + '/'):
                if ann.get('K') is not None and ann.get('front_direction'):
                    annotated += 1

        remaining = total - annotated
        progress = (annotated / total * 100) if total > 0 else 0

        category_stats[cat_name] = {
            'total': total,
            'annotated': annotated,
            'remaining': remaining,
            'progress': progress
        }

    return category_stats


def estimate_completion_time(total_annotated, total_remaining, samples_per_hour):
    """估算完成时间"""
    if samples_per_hour > 0:
        hours_needed = total_remaining / samples_per_hour
        return timedelta(hours=hours_needed)
    return None


def display_stats(annotations, categories, category_stats):
    """显示统计信息"""
    print("\n" + "="*80)
    print("📊 标注进度与速度统计")
    print("="*80)

    total_samples = sum(cat['total'] for cat in category_stats.values())
    total_annotated = sum(cat['annotated'] for cat in category_stats.values())
    total_remaining = total_samples - total_annotated
    overall_progress = (total_annotated / total_samples * 100) if total_samples > 0 else 0

    # 整体进度
    print(f"\n🎯 整体进度")
    print(f"  总样本数: {total_samples:,}")
    print(f"  已标注: {total_annotated:,} ({overall_progress:.2f}%)")
    print(f"  剩余: {total_remaining:,} ({100-overall_progress:.2f}%)")

    # 进度条
    filled = int(overall_progress / 2)
    bar = "█" * filled + "░" * (50 - filled)
    print(f"  [{bar}] {overall_progress:.1f}%")

    # 速度估算（基于最近标注时间）
    json_file = Path("/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations.json")
    if json_file.exists() and total_annotated > 0:
        last_modified = datetime.fromtimestamp(json_file.stat().st_mtime)
        now = datetime.now()

        print(f"\n⏱️  时间信息")
        print(f"  最后标注时间: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # 简单估算（假设用户输入标注速度）
        print(f"\n💡 完成时间估算")
        print(f"  请根据您的标注速度估算：")

        speeds = [10, 20, 30, 50, 100]  # 样本/小时
        for speed in speeds:
            time_needed = estimate_completion_time(total_annotated, total_remaining, speed)
            if time_needed:
                days = time_needed.total_seconds() / 86400
                hours = time_needed.total_seconds() / 3600
                print(f"    @ {speed:3d} 样本/小时 → {hours:7.1f} 小时 ({days:6.2f} 天)")

    # 各类别进度（显示前10个剩余最多的）
    print(f"\n📋 类别进度（剩余最多的前10个）")
    print(f"  {'类别':<20} {'已标注':<10} {'剩余':<10} {'进度':<15}")
    print("  " + "-"*60)

    sorted_cats = sorted(
        category_stats.items(),
        key=lambda x: x[1]['remaining'],
        reverse=True
    )[:10]

    for cat_name, stats in sorted_cats:
        progress_bar_mini = "█" * int(stats['progress'] / 10) + "░" * (10 - int(stats['progress'] / 10))
        print(f"  {cat_name:<20} {stats['annotated']:<10} {stats['remaining']:<10} "
              f"{progress_bar_mini} {stats['progress']:5.1f}%")

    # 完成的类别
    completed = [cat for cat, stats in category_stats.items() if stats['remaining'] == 0]
    if completed:
        print(f"\n✅ 已完成的类别 ({len(completed)}个):")
        print(f"  {', '.join(completed)}")

    print("\n" + "="*80)

    # 标注效率提示
    if total_annotated > 0:
        print("\n💡 提示：")
        print("  - 每标注100个样本后休息10分钟可提高效率")
        print("  - 建议按类别集中标注，避免频繁切换")
        print("  - 使用快捷键可以提高标注速度")
        print("")


def main():
    DATA_DIR = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply")
    ANNOTATION_FILE = Path("/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations.json")

    # 加载标注
    annotations = load_annotations(ANNOTATION_FILE)

    # 扫描数据目录
    categories, total_samples = scan_data_directory(DATA_DIR)

    # 计算进度统计
    category_stats = calculate_progress_stats(annotations, categories)

    # 显示统计
    display_stats(annotations, categories, category_stats)


if __name__ == "__main__":
    main()

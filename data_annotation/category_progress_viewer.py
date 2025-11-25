#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别标注进度查看器

功能：
- 显示所有类别的标注进度
- 选择要标注的类别
- 保证不丢失任何已有标注

作者: Claude
创建日期: 2025-11-25
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict


def load_annotations(annotation_file):
    """加载已有标注"""
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def scan_categories(data_dir):
    """扫描所有类别"""
    categories = {}
    for cat_dir in sorted(data_dir.iterdir()):
        if cat_dir.is_dir():
            ply_files = list(cat_dir.glob("*.ply"))
            if ply_files:
                categories[cat_dir.name] = {
                    'path': cat_dir,
                    'total': len(ply_files)
                }
    return categories


def calculate_progress(data_dir, annotations):
    """计算各类别进度"""
    categories = scan_categories(data_dir)
    progress = {}

    for cat_name, cat_info in categories.items():
        total = cat_info['total']
        annotated = 0

        # 统计该类别已标注数量
        for file_path, ann in annotations.items():
            # 检查文件是否属于该类别
            if file_path.startswith(cat_name + '/'):
                # 检查是否完整标注（有对称性和方向）
                if ann.get('K') is not None and ann.get('front_direction'):
                    annotated += 1

        progress[cat_name] = {
            'total': total,
            'annotated': annotated,
            'remaining': total - annotated,
            'progress': (annotated / total * 100) if total > 0 else 0
        }

    return progress


def show_progress(progress):
    """显示进度表格"""
    print("\n" + "="*90)
    print("📊 类别标注进度统计".center(90))
    print("="*90)

    # 按剩余数量排序
    sorted_cats = sorted(progress.items(), key=lambda x: x[1]['remaining'], reverse=True)

    print(f"\n{'序号':<6}{'类别名称':<25}{'总数':<10}{'已标注':<10}{'剩余':<10}{'进度条':<20}{'状态':<10}")
    print("-"*90)

    for idx, (cat_name, prog) in enumerate(sorted_cats, 1):
        # 进度条
        filled = int(prog['progress'] / 10)
        bar = "█" * filled + "░" * (10 - filled)
        progress_str = f"{bar} {prog['progress']:5.1f}%"

        # 状态
        if prog['remaining'] == 0:
            status = "✅ 完成"
        elif prog['annotated'] == 0:
            status = "⭕ 未开始"
        else:
            status = "⏳ 进行中"

        print(f"{idx:<6}{cat_name:<25}{prog['total']:<10}{prog['annotated']:<10}"
              f"{prog['remaining']:<10}{progress_str:<20}{status:<10}")

    print("="*90)

    # 汇总统计
    total_samples = sum(p['total'] for p in progress.values())
    total_annotated = sum(p['annotated'] for p in progress.values())
    total_remaining = total_samples - total_annotated

    print(f"\n📈 总体进度：")
    print(f"   总样本数: {total_samples}")
    print(f"   已标注: {total_annotated} ({total_annotated/total_samples*100:.1f}%)")
    print(f"   剩余: {total_remaining} ({total_remaining/total_samples*100:.1f}%)")
    print()


def main():
    DATA_DIR = Path("/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply")
    ANNOTATION_FILE = Path("/home/pablo/ForwardNet-claude/data/symmetry_annotations.json")

    print("\n" + "="*90)
    print("🏷️  类别标注工具 - 进度查看器".center(90))
    print("="*90)

    # 加载标注
    print("\n📂 加载标注数据...")
    annotations = load_annotations(ANNOTATION_FILE)
    print(f"   ✅ 已加载 {len(annotations)} 个标注")

    # 计算进度
    print("📊 计算类别进度...")
    progress = calculate_progress(DATA_DIR, annotations)

    # 显示进度
    show_progress(progress)

    # 交互选择
    print("💡 选择要标注的类别：")
    print("   - 输入类别名称（如：airplane）")
    print("   - 输入序号（如：1）")
    print("   - 输入 'q' 退出")
    print()

    while True:
        choice = input("请选择 (输入类别名或序号): ").strip()

        if choice.lower() == 'q':
            print("\n👋 退出")
            break

        # 确定选择的类别
        selected_cat = None

        if choice.isdigit():
            # 序号选择
            idx = int(choice) - 1
            sorted_cats = sorted(progress.items(), key=lambda x: x[1]['remaining'], reverse=True)
            if 0 <= idx < len(sorted_cats):
                selected_cat = sorted_cats[idx][0]
        else:
            # 类别名称
            if choice in progress:
                selected_cat = choice

        if not selected_cat:
            print("❌ 无效的选择，请重试")
            continue

        # 显示选择的类别信息
        cat_prog = progress[selected_cat]
        print(f"\n{'='*90}")
        print(f"📂 已选择类别: {selected_cat}")
        print(f"   总数: {cat_prog['total']}")
        print(f"   已标注: {cat_prog['annotated']}")
        print(f"   剩余: {cat_prog['remaining']}")
        print(f"   进度: {cat_prog['progress']:.1f}%")
        print(f"{'='*90}\n")

        # 确认
        confirm = input(f"确认标注 '{selected_cat}' 类别？(y/n): ").strip().lower()

        if confirm == 'y':
            cat_dir = DATA_DIR / selected_cat

            print(f"\n🚀 启动标注工具...")
            print(f"   数据目录: {cat_dir}")
            print(f"   标注文件: {ANNOTATION_FILE}")
            print(f"   ⚠️  标注会保存到同一个JSON文件，不会丢失已有进度！")
            print(f"\n按 Ctrl+C 可以随时停止\n")

            try:
                # 启动Web工具
                subprocess.run([
                    "python", "annotate_symmetry_web_v2.py",
                    "--data_dir", str(cat_dir),
                    "--output", str(ANNOTATION_FILE),
                    "--port", "8051"
                ])
            except KeyboardInterrupt:
                print("\n\n✅ 标注已停止")

            # 重新计算进度
            print("\n📊 重新加载进度...")
            annotations = load_annotations(ANNOTATION_FILE)
            progress = calculate_progress(DATA_DIR, annotations)
            show_progress(progress)

        print()


if __name__ == "__main__":
    main()

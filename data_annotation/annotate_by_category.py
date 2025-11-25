#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按类别优先标注工具

功能：
- 选择要标注的类别（单个或多个）
- 按优先级顺序标注
- 跳过已标注的类别
- 显示各类别的标注进度

使用方法：
    # 方式1：交互式选择类别
    python annotate_by_category.py

    # 方式2：直接指定类别
    python annotate_by_category.py --categories airplane chair table

    # 方式3：按优先级列表标注
    python annotate_by_category.py --priority_file priority_list.txt

作者: Claude
创建日期: 2025-11-25
"""

import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict


class CategoryAnnotationManager:
    """类别标注管理器"""

    def __init__(self, data_dir: Path, annotation_file: Path):
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file)

        # 扫描所有类别
        self.categories = self._scan_categories()

        # 加载已有标注
        self.annotations = self._load_annotations()

        # 统计各类别进度
        self.category_progress = self._calculate_category_progress()

    def _scan_categories(self):
        """扫描所有类别目录"""
        categories = []
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir():
                ply_files = list(item.glob("*.ply"))
                if ply_files:
                    categories.append({
                        'name': item.name,
                        'path': item,
                        'total': len(ply_files)
                    })
        return categories

    def _load_annotations(self):
        """加载已有标注"""
        if self.annotation_file.exists():
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _calculate_category_progress(self):
        """计算各类别的标注进度"""
        progress = {}

        for cat in self.categories:
            cat_name = cat['name']
            total = cat['total']

            # 统计已标注数量
            annotated = 0
            for file_path, ann in self.annotations.items():
                if file_path.startswith(cat_name + '/'):
                    if ann.get('front_direction'):  # 完整标注
                        annotated += 1

            progress[cat_name] = {
                'total': total,
                'annotated': annotated,
                'remaining': total - annotated,
                'progress': annotated / total * 100 if total > 0 else 0
            }

        return progress

    def show_categories(self):
        """显示所有类别和进度"""
        print("\n" + "="*80)
        print("📂 ModelNet40 类别列表")
        print("="*80)
        print(f"\n{'序号':<6}{'类别名称':<20}{'总数':<10}{'已标注':<10}{'剩余':<10}{'进度':<15}")
        print("-"*80)

        for idx, cat in enumerate(self.categories, 1):
            name = cat['name']
            prog = self.category_progress[name]

            status = "✅" if prog['remaining'] == 0 else "⏳"
            progress_bar = self._create_progress_bar(prog['progress'])

            print(f"{idx:<6}{name:<20}{prog['total']:<10}{prog['annotated']:<10}"
                  f"{prog['remaining']:<10}{progress_bar:<15} {status}")

        print("="*80 + "\n")

    def _create_progress_bar(self, percentage):
        """创建进度条"""
        filled = int(percentage / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {percentage:5.1f}%"

    def interactive_select(self):
        """交互式选择类别"""
        self.show_categories()

        print("💡 选择要标注的类别：")
        print("  - 输入序号（如：1）")
        print("  - 输入多个序号，用空格分隔（如：1 5 10）")
        print("  - 输入类别名称（如：airplane chair）")
        print("  - 输入 'all' 标注所有未完成的类别")
        print("  - 输入 'top5' 标注剩余数量最多的前5个类别")
        print()

        user_input = input("请选择: ").strip()

        if user_input.lower() == 'all':
            # 所有未完成的类别
            selected = [cat['name'] for cat in self.categories
                       if self.category_progress[cat['name']]['remaining'] > 0]
        elif user_input.lower() == 'top5':
            # 剩余最多的前5个
            sorted_cats = sorted(self.categories,
                               key=lambda c: self.category_progress[c['name']]['remaining'],
                               reverse=True)
            selected = [cat['name'] for cat in sorted_cats[:5]
                       if self.category_progress[cat['name']]['remaining'] > 0]
        elif user_input.isdigit() or all(x.isdigit() for x in user_input.split()):
            # 序号选择
            indices = [int(x) - 1 for x in user_input.split()]
            selected = [self.categories[i]['name'] for i in indices
                       if 0 <= i < len(self.categories)]
        else:
            # 类别名称
            selected = user_input.split()

        return selected

    def create_category_subset(self, category_names):
        """创建包含指定类别的临时目录"""
        temp_dir = Path("data/temp_annotation_subset")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 清空临时目录
        for item in temp_dir.iterdir():
            if item.is_symlink() or item.is_dir():
                item.unlink()

        # 创建符号链接
        total_files = 0
        for cat_name in category_names:
            src = self.data_dir / cat_name
            if src.exists():
                dst = temp_dir / cat_name
                dst.symlink_to(src.resolve())
                prog = self.category_progress[cat_name]
                total_files += prog['remaining']
                print(f"  ✅ {cat_name}: {prog['remaining']} 个待标注")

        print(f"\n📊 总计: {total_files} 个待标注样本")
        return temp_dir

    def launch_annotation_tool(self, categories):
        """启动标注工具"""
        if not categories:
            print("❌ 未选择任何类别")
            return

        print("\n" + "="*80)
        print("🚀 启动标注工具")
        print("="*80)

        # 显示选择的类别
        print("\n📋 已选择的类别:")
        for cat_name in categories:
            prog = self.category_progress.get(cat_name)
            if prog:
                print(f"  - {cat_name}: {prog['remaining']}/{prog['total']} 待标注")

        # 创建临时子集
        print("\n🔨 创建临时数据集...")
        temp_dir = self.create_category_subset(categories)

        # 启动工具
        print("\n🌐 启动Web标注工具...")
        print("="*80 + "\n")

        cmd = [
            "python", "annotate_symmetry_web_v2.py",
            "--data_dir", str(temp_dir),
            "--output", str(self.annotation_file),
            "--port", "8051"
        ]

        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n✅ 标注工具已关闭")
        finally:
            # 清理临时目录
            print("\n🧹 清理临时文件...")
            for item in temp_dir.iterdir():
                if item.is_symlink():
                    item.unlink()
            # temp_dir.rmdir()  # 保留目录，方便调试


def main():
    parser = argparse.ArgumentParser(description="按类别优先标注工具")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply',
        help='数据集目录'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default='/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations.json',
        help='标注文件路径'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        help='要标注的类别（如：airplane chair table）'
    )
    parser.add_argument(
        '--priority_file',
        type=str,
        help='优先级列表文件（每行一个类别名）'
    )

    args = parser.parse_args()

    # 初始化管理器
    manager = CategoryAnnotationManager(
        Path(args.data_dir),
        Path(args.annotation_file)
    )

    # 确定要标注的类别
    if args.categories:
        # 命令行指定
        selected_categories = args.categories
    elif args.priority_file:
        # 从文件读取
        with open(args.priority_file, 'r') as f:
            selected_categories = [line.strip() for line in f if line.strip()]
    else:
        # 交互式选择
        selected_categories = manager.interactive_select()

    # 验证类别名称
    valid_categories = [cat['name'] for cat in manager.categories]
    selected_categories = [c for c in selected_categories if c in valid_categories]

    if not selected_categories:
        print("❌ 未找到有效的类别")
        return

    # 启动标注工具
    manager.launch_annotation_tool(selected_categories)


if __name__ == "__main__":
    main()

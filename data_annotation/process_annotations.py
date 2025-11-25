#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注数据处理工具集

功能：
- 按对称性分类数据
- 筛选需要矫正的数据
- 统计分析
- 生成矫正脚本

使用方法：
    python process_annotations.py --mode classify
    python process_annotations.py --mode filter_misaligned
    python process_annotations.py --mode stats
    python process_annotations.py --mode generate_correction_script

作者: Claude
创建日期: 2025-11-25
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict


class AnnotationProcessor:
    """标注数据处理器"""

    def __init__(self, annotation_file: Path):
        self.annotation_file = Path(annotation_file)
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """加载标注数据"""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def classify_by_symmetry(self, output_dir: Path):
        """按对称性分类数据，生成分类文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 分类
        classified = defaultdict(list)
        for file_path, ann in self.annotations.items():
            k = ann['K']
            symmetry_name = ann.get('symmetry_name', f'K={k}')
            classified[symmetry_name].append(ann)

        print("="*60)
        print("📊 按对称性分类")
        print("="*60)

        # 保存每个类别到单独的JSON文件
        for symmetry_name, items in sorted(classified.items()):
            # JSON
            json_file = output_dir / f"{symmetry_name.replace(' ', '_')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)

            # TXT（文件路径列表）
            txt_file = output_dir / f"{symmetry_name.replace(' ', '_')}_files.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write(item['file'] + '\n')

            print(f"  {symmetry_name}: {len(items)}个样本")
            print(f"    → JSON: {json_file}")
            print(f"    → TXT: {txt_file}")

        print()
        return classified

    def filter_misaligned(self, output_dir: Path):
        """筛选需要矫正的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        misaligned = []
        aligned = []

        for file_path, ann in self.annotations.items():
            if ann.get('aligned', False):
                aligned.append(ann)
            else:
                misaligned.append(ann)

        print("="*60)
        print("⚠️  筛选需要矫正的数据")
        print("="*60)
        print(f"  总数: {len(self.annotations)}")
        print(f"  已对齐: {len(aligned)} ({len(aligned)/len(self.annotations)*100:.1f}%)")
        print(f"  需矫正: {len(misaligned)} ({len(misaligned)/len(self.annotations)*100:.1f}%)")
        print()

        # 保存需要矫正的数据
        # 1. JSON
        json_file = output_dir / "misaligned_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(misaligned, f, indent=2, ensure_ascii=False)
        print(f"  JSON: {json_file}")

        # 2. CSV（含矫正信息）
        csv_file = output_dir / "misaligned_data.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file', 'K', 'symmetry_name', 'current_direction',
                'rotation_needed', 'category'
            ])
            writer.writeheader()
            for ann in misaligned:
                rotation = self._calculate_rotation(ann.get('front_direction'))
                category = ann['file'].split('/')[0] if '/' in ann['file'] else 'unknown'
                writer.writerow({
                    'file': ann['file'],
                    'K': ann['K'],
                    'symmetry_name': ann.get('symmetry_name', ''),
                    'current_direction': ann.get('front_direction', ''),
                    'rotation_needed': rotation,
                    'category': category
                })
        print(f"  CSV: {csv_file}")

        # 3. TXT（文件路径列表）
        txt_file = output_dir / "misaligned_files.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for ann in misaligned:
                f.write(ann['file'] + '\n')
        print(f"  TXT: {txt_file}")

        # 按方向分组统计
        print("\n  按当前方向分组：")
        direction_groups = defaultdict(list)
        for ann in misaligned:
            direction = ann.get('front_direction', 'unknown')
            direction_groups[direction].append(ann)

        for direction, items in sorted(direction_groups.items()):
            rotation = self._calculate_rotation(direction)
            print(f"    {direction}: {len(items)}个 → {rotation}")

        print()
        return misaligned, aligned

    def generate_statistics(self):
        """生成统计信息"""
        print("="*60)
        print("📈 统计分析")
        print("="*60)

        # 1. 对称性分布
        symmetry_stats = defaultdict(int)
        for ann in self.annotations.values():
            symmetry_name = ann.get('symmetry_name', f"K={ann['K']}")
            symmetry_stats[symmetry_name] += 1

        print("\n  对称性分布：")
        for symmetry, count in sorted(symmetry_stats.items()):
            pct = count / len(self.annotations) * 100
            print(f"    {symmetry}: {count} ({pct:.1f}%)")

        # 2. 方向分布
        direction_stats = defaultdict(int)
        for ann in self.annotations.values():
            direction = ann.get('front_direction', 'unknown')
            direction_stats[direction] += 1

        print("\n  正面方向分布：")
        for direction, count in sorted(direction_stats.items()):
            pct = count / len(self.annotations) * 100
            status = "✅" if direction == "-Z" else "⚠️"
            print(f"    {status} {direction}: {count} ({pct:.1f}%)")

        # 3. 对齐状态
        aligned_count = sum(1 for ann in self.annotations.values() if ann.get('aligned', False))
        misaligned_count = len(self.annotations) - aligned_count

        print(f"\n  对齐状态：")
        print(f"    ✅ 已对齐: {aligned_count} ({aligned_count/len(self.annotations)*100:.1f}%)")
        print(f"    ⚠️  需矫正: {misaligned_count} ({misaligned_count/len(self.annotations)*100:.1f}%)")

        # 4. 按类别统计
        category_stats = defaultdict(lambda: {'total': 0, 'aligned': 0, 'misaligned': 0})
        for ann in self.annotations.values():
            category = ann['file'].split('/')[0] if '/' in ann['file'] else 'unknown'
            category_stats[category]['total'] += 1
            if ann.get('aligned', False):
                category_stats[category]['aligned'] += 1
            else:
                category_stats[category]['misaligned'] += 1

        print(f"\n  按类别统计（前10个）：")
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        for i, (category, stats) in enumerate(sorted_categories[:10]):
            misalign_rate = stats['misaligned'] / stats['total'] * 100
            print(f"    {i+1}. {category}: {stats['total']}个 "
                  f"(矫正率 {misalign_rate:.1f}%)")

        print()

    def generate_correction_script(self, output_file: Path):
        """生成矫正脚本（示例）"""
        output_file = Path(output_file)

        misaligned = [ann for ann in self.annotations.values()
                     if not ann.get('aligned', False)]

        print("="*60)
        print("🔧 生成矫正脚本")
        print("="*60)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write("数据矫正脚本（自动生成）\n\n")
            f.write(f"需要矫正的数据: {len(misaligned)}个\n")
            f.write('"""\n\n')
            f.write("import numpy as np\n")
            f.write("from pathlib import Path\n\n")
            f.write("# 矫正列表\n")
            f.write("corrections = [\n")

            for ann in misaligned:
                rotation = self._get_rotation_matrix_code(ann.get('front_direction'))
                f.write(f"    {{\n")
                f.write(f"        'file': '{ann['file']}',\n")
                f.write(f"        'direction': '{ann.get('front_direction', '')}',\n")
                f.write(f"        'rotation': '{rotation}',\n")
                f.write(f"    }},\n")

            f.write("]\n\n")
            f.write("# TODO: 实现实际的矫正逻辑\n")
            f.write("# 例如：加载PLY文件，应用旋转，保存到新位置\n")

        print(f"  脚本已生成: {output_file}")
        print(f"  包含 {len(misaligned)} 个需要矫正的数据")
        print()

    def _calculate_rotation(self, current_direction):
        """计算需要的旋转"""
        rotation_map = {
            '-Z': '无需旋转（已对齐）',
            '+Z': '绕X轴旋转180°',
            '-X': '绕Y轴旋转-90°',
            '+X': '绕Y轴旋转+90°',
            '-Y': '绕X轴旋转+90°',
            '+Y': '绕X轴旋转-90°',
        }
        return rotation_map.get(current_direction, '未知方向')

    def _get_rotation_matrix_code(self, current_direction):
        """获取旋转矩阵代码（示例）"""
        rotation_code_map = {
            '+Z': 'rotate_x(180)',
            '-X': 'rotate_y(-90)',
            '+X': 'rotate_y(90)',
            '-Y': 'rotate_x(90)',
            '+Y': 'rotate_x(-90)',
        }
        return rotation_code_map.get(current_direction, 'identity')


def main():
    parser = argparse.ArgumentParser(description="标注数据处理工具")
    parser.add_argument(
        '--annotation_file',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/symmetry_annotations.json',
        help='标注JSON文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['classify', 'filter_misaligned', 'stats', 'generate_correction_script', 'all'],
        default='all',
        help='处理模式'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/processed',
        help='输出目录'
    )

    args = parser.parse_args()

    processor = AnnotationProcessor(Path(args.annotation_file))
    output_dir = Path(args.output_dir)

    if args.mode in ['classify', 'all']:
        classify_dir = output_dir / "classified_by_symmetry"
        processor.classify_by_symmetry(classify_dir)

    if args.mode in ['filter_misaligned', 'all']:
        misaligned_dir = output_dir / "misaligned"
        processor.filter_misaligned(misaligned_dir)

    if args.mode in ['stats', 'all']:
        processor.generate_statistics()

    if args.mode in ['generate_correction_script', 'all']:
        script_file = output_dir / "correct_data.py"
        processor.generate_correction_script(script_file)

    print("="*60)
    print("✅ 处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()

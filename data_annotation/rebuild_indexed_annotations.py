#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建索引化的标注数据结构

新的数据结构支持：
1. 快速按category查询
2. 快速按K值（对称类型）查询
3. 快速找到需要矫正的数据
4. O(1)复杂度的查询

数据结构：
{
  "version": "2.0",
  "annotations": {file_path: annotation_data},
  "indices": {
    "by_category": {category: [file_paths]},
    "by_K": {K: [file_paths]},
    "by_aligned": {true/false: [file_paths]},
    "need_correction": [file_paths]  // aligned=False的快速查询
  },
  "stats": {统计信息}
}

使用方法：
    python rebuild_indexed_annotations.py

作者: Claude
创建日期: 2025-11-25
"""

import json
from pathlib import Path
from collections import defaultdict


def build_indices(annotations):
    """构建索引"""
    indices = {
        'by_category': defaultdict(list),
        'by_K': defaultdict(list),
        'by_aligned': {'true': [], 'false': []},
        'need_correction': []  # aligned=False的快速查询
    }

    for file_path, ann in annotations.items():
        # 按category索引
        category = file_path.split('/')[0]
        indices['by_category'][category].append(file_path)

        # 按K值索引
        K = ann.get('K')
        if K is not None:
            indices['by_K'][str(K)].append(file_path)

        # 按aligned索引
        aligned = ann.get('aligned', False)
        if aligned:
            indices['by_aligned']['true'].append(file_path)
        else:
            indices['by_aligned']['false'].append(file_path)
            indices['need_correction'].append(file_path)

    # 转换defaultdict为普通dict（方便JSON序列化）
    indices['by_category'] = dict(indices['by_category'])
    indices['by_K'] = dict(indices['by_K'])

    return indices


def calculate_stats(annotations, indices):
    """计算统计信息"""
    stats = {
        'total': len(annotations),
        'version': '2.0',
        'by_category': {},
        'by_K': {},
        'aligned_count': len(indices['by_aligned']['true']),
        'need_correction_count': len(indices['need_correction'])
    }

    # 按category统计
    for category, file_list in indices['by_category'].items():
        # 统计该category的K值分布
        K_dist = defaultdict(int)
        for file_path in file_list:
            K = annotations[file_path].get('K')
            if K is not None:
                K_dist[K] += 1

        stats['by_category'][category] = {
            'total': len(file_list),
            'K_distribution': dict(K_dist)
        }

    # 按K值统计
    for K, file_list in indices['by_K'].items():
        stats['by_K'][K] = len(file_list)

    return stats


def rebuild_indexed_structure(old_json_file, new_json_file):
    """重建索引化的数据结构"""
    print("=" * 80)
    print("重建索引化标注数据")
    print("=" * 80)

    # 1. 加载旧数据
    with open(old_json_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"\n📂 加载标注: {len(annotations)} 个")

    # 2. 构建索引
    print(f"🔍 构建索引...")
    indices = build_indices(annotations)

    print(f"  ✅ Category索引: {len(indices['by_category'])} 个类别")
    print(f"  ✅ K值索引: {len(indices['by_K'])} 种K值")
    print(f"  ✅ 需要矫正: {len(indices['need_correction'])} 个")

    # 3. 计算统计
    print(f"📊 计算统计...")
    stats = calculate_stats(annotations, indices)

    # 4. 构建新结构
    indexed_data = {
        'version': '2.0',
        'annotations': annotations,
        'indices': indices,
        'stats': stats
    }

    # 5. 保存
    with open(new_json_file, 'w', encoding='utf-8') as f:
        json.dump(indexed_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 保存索引化数据: {new_json_file}")
    print(f"文件大小: {new_json_file.stat().st_size / 1024:.1f} KB")

    # 6. 打印统计
    print(f"\n" + "=" * 80)
    print("数据统计")
    print("=" * 80)
    print(f"\n总标注数: {stats['total']}")
    print(f"需要矫正: {stats['need_correction_count']} 个")
    print(f"已对齐: {stats['aligned_count']} 个")

    print(f"\nK值分布:")
    for K, count in sorted(stats['by_K'].items(), key=lambda x: int(x[0])):
        K_names = {'-1': '没有正面', '0': '完全对称', '1': '1个正面', '2': '2个正面', '4': '4个正面'}
        name = K_names.get(K, '未知')
        print(f"  K={K:2s} ({name}): {count:4d} 个")

    print(f"\n各类别分布:")
    for category in sorted(stats['by_category'].keys()):
        count = stats['by_category'][category]['total']
        print(f"  {category:15s}: {count:4d} 个")

    return indexed_data


def demo_queries(indexed_data):
    """演示快速查询"""
    print(f"\n" + "=" * 80)
    print("快速查询演示")
    print("=" * 80)

    indices = indexed_data['indices']

    # 查询1: 获取某个category的所有数据
    print(f"\n查询1: glass_box类别的所有数据")
    glass_box_files = indices['by_category'].get('glass_box', [])
    print(f"  找到 {len(glass_box_files)} 个文件")
    print(f"  示例: {glass_box_files[:3]}")

    # 查询2: 获取所有单正面（K=1）的数据
    print(f"\n查询2: 所有K=1（单正面）的数据")
    K1_files = indices['by_K'].get('1', [])
    print(f"  找到 {len(K1_files)} 个文件")
    print(f"  涉及类别: {set(f.split('/')[0] for f in K1_files)}")

    # 查询3: 所有需要矫正的数据
    print(f"\n查询3: 所有需要矫正的数据（aligned=False）")
    need_correction = indices['need_correction']
    print(f"  找到 {len(need_correction)} 个文件")
    if need_correction:
        print(f"  示例: {need_correction[:5]}")
        categories = defaultdict(int)
        for f in need_correction:
            cat = f.split('/')[0]
            categories[cat] += 1
        print(f"  按类别: {dict(categories)}")

    print(f"\n✅ 查询速度: O(1) 常数时间复杂度")


def create_query_helper():
    """创建查询辅助函数"""
    code = '''#!/usr/bin/env python3
"""
快速查询辅助函数

使用方法:
    from query_annotations import AnnotationQuery

    query = AnnotationQuery('symmetry_annotations_indexed.json')

    # 查询某个category
    files = query.get_by_category('airplane')

    # 查询某个K值
    files = query.get_by_K(1)

    # 查询需要矫正的
    files = query.get_need_correction()
"""

import json

class AnnotationQuery:
    def __init__(self, indexed_json_file):
        with open(indexed_json_file, 'r') as f:
            self.data = json.load(f)
        self.annotations = self.data['annotations']
        self.indices = self.data['indices']
        self.stats = self.data['stats']

    def get_by_category(self, category):
        """获取某个category的所有文件"""
        return self.indices['by_category'].get(category, [])

    def get_by_K(self, K):
        """获取某个K值的所有文件"""
        return self.indices['by_K'].get(str(K), [])

    def get_need_correction(self):
        """获取所有需要矫正的文件"""
        return self.indices['need_correction']

    def get_aligned(self):
        """获取所有已对齐的文件"""
        return self.indices['by_aligned']['true']

    def get_annotation(self, file_path):
        """获取某个文件的标注"""
        return self.annotations.get(file_path)

    def get_stats(self):
        """获取统计信息"""
        return self.stats
'''

    with open('query_annotations.py', 'w') as f:
        f.write(code)

    print(f"\n✅ 创建查询辅助: query_annotations.py")


def main():
    old_json = Path('symmetry_annotations.json')
    new_json = Path('symmetry_annotations_indexed.json')

    # 重建索引结构
    indexed_data = rebuild_indexed_structure(old_json, new_json)

    # 演示查询
    demo_queries(indexed_data)

    # 创建查询辅助
    create_query_helper()

    print(f"\n" + "=" * 80)
    print("✅ 重建完成!")
    print("=" * 80)
    print(f"\n现在可以使用 symmetry_annotations_indexed.json 进行快速查询")
    print(f"或使用 query_annotations.py 进行编程查询")


if __name__ == '__main__':
    main()

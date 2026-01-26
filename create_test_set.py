#!/usr/bin/env python3
"""
创建测试集 - 从标注数据中分层抽样

数据来源:
1. data_annotation/symmetry_annotations.csv (3399个标注样本)
2. data/symmetry_classification_gt/ (1140个p2v2训练样本)

测试集策略:
- 10%分层抽样（按类别和K值）
- 可选混入p2v2 clean训练样本
"""

import os
import json
import random
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_annotations(csv_path: str) -> pd.DataFrame:
    """加载标注CSV"""
    df = pd.read_csv(csv_path)
    df['category'] = df['file'].apply(lambda x: x.split('/')[0])
    df['sample_name'] = df['file'].apply(lambda x: x.split('/')[-1].replace('.ply', ''))
    return df


def load_p2v2_samples(gt_dir: str) -> Dict[str, dict]:
    """加载p2v2训练集样本信息"""
    info_path = os.path.join(gt_dir, 'dataset_info.json')
    if not os.path.exists(info_path):
        return {}

    with open(info_path, 'r') as f:
        data = json.load(f)

    # 已知的双词类别
    COMPOUND_CATEGORIES = {'glass_box', 'flower_pot'}

    # 已知的普通类别 (用于处理不同格式)
    SIMPLE_CATEGORIES = {'airplane', 'bathtub', 'bench', 'bookshelf', 'bottle',
                         'bowl', 'chair', 'cone', 'cup', 'desk', 'plant', 'wardrobe'}

    # 转换key格式:
    # 格式1: airplane_airplane_0001.ply -> airplane/airplane_0001.ply
    # 格式2: airplane_0001.ply -> airplane/airplane_0001.ply (短格式)
    # 格式3: glass_box_glass_box_0001.ply -> glass_box/glass_box_0001.ply
    result = {}
    for name, info in data.items():
        base = name.replace('.ply', '')

        # 检查是否是复合类别
        category = None
        item_name = None

        for compound in COMPOUND_CATEGORIES:
            if base.startswith(compound + '_'):
                category = compound
                remaining = base[len(compound)+1:]  # 去掉 "glass_box_"
                # 检查是否有重复: glass_box_0001 vs glass_box_glass_box_0001
                if remaining.startswith(compound + '_'):
                    item_name = remaining  # glass_box_0001
                else:
                    item_name = f"{compound}_{remaining}"  # 补全为 glass_box_0001
                break

        if category is None:
            # 普通类别处理
            parts = base.split('_')
            first_part = parts[0]

            if first_part in SIMPLE_CATEGORIES:
                category = first_part
                remaining = '_'.join(parts[1:])

                # 检查是否已经有类别前缀: airplane_0001 vs airplane_airplane_0001
                if remaining.startswith(first_part + '_'):
                    item_name = remaining  # airplane_0001
                else:
                    # 短格式: airplane_0001.ply (只有一个airplane)
                    # 需要补全为 airplane_airplane_0001
                    item_name = f"{first_part}_{remaining}"

        if category and item_name:
            file_path = f"{category}/{item_name}.ply"
            result[file_path] = info

    return result


def stratified_sample(df: pd.DataFrame, test_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分层抽样: 按类别和K值分层

    返回: (train_val_df, test_df)
    """
    random.seed(seed)

    test_indices = []
    train_val_indices = []

    # 按 (category, K) 分组
    groups = df.groupby(['category', 'K'])

    for (category, k), group in groups:
        indices = group.index.tolist()
        n_test = max(1, int(len(indices) * test_ratio))  # 至少1个

        random.shuffle(indices)
        test_indices.extend(indices[:n_test])
        train_val_indices.extend(indices[n_test:])

    test_df = df.loc[test_indices].copy()
    train_val_df = df.loc[train_val_indices].copy()

    return train_val_df, test_df


def add_p2v2_samples(test_df: pd.DataFrame, p2v2_samples: Dict[str, dict],
                      n_samples_per_category: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    从p2v2样本中添加一些到测试集

    注意: 只添加不在测试集中的样本
    """
    random.seed(seed)

    # 当前测试集中的文件
    existing_files = set(test_df['file'].tolist())

    # 按p2v2类别分组
    by_category = defaultdict(list)
    for file_path, info in p2v2_samples.items():
        if file_path not in existing_files:
            by_category[info['category']].append((file_path, info))

    # 从每个类别中随机选取
    additional_rows = []
    for category, samples in by_category.items():
        n_to_add = min(n_samples_per_category, len(samples))
        selected = random.sample(samples, n_to_add)

        for file_path, info in selected:
            # 转换K值
            if category == '1_front':
                k = 1
            elif category == '2_fronts':
                k = 2
            elif category == '4_fronts':
                k = 4
            elif category in ['no_front', 'symmetric']:
                k = -1  # 无正面/旋转对称
            else:
                k = 0

            # 构建行
            parts = file_path.split('/')
            obj_category = parts[0]
            row = {
                'file': file_path,
                'K': k,
                'symmetry_name': info['category'],
                'front_direction': info.get('original_front_direction', 'N/A'),
                'aligned': 'N/A',
                'needs_correction': 'NO',
                'rotation_needed': 'N/A',
                'category': obj_category,
                'sample_name': file_path.split('/')[-1].replace('.ply', ''),
                'source': 'p2v2'  # 标记来源
            }
            additional_rows.append(row)

    if additional_rows:
        additional_df = pd.DataFrame(additional_rows)
        test_df = pd.concat([test_df, additional_df], ignore_index=True)
        test_df['source'] = test_df.get('source', 'annotation').fillna('annotation')

    return test_df


def print_stats(df: pd.DataFrame, name: str):
    """打印数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"{name} 统计信息")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")

    print(f"\n按类别分布:")
    for cat in sorted(df['category'].unique()):
        count = len(df[df['category'] == cat])
        print(f"  {cat}: {count}")

    print(f"\n按K值分布:")
    for k in sorted(df['K'].unique()):
        count = len(df[df['K'] == k])
        print(f"  K={k}: {count}")

    if 'source' in df.columns:
        print(f"\n按来源分布:")
        for source in df['source'].unique():
            count = len(df[df['source'] == source])
            print(f"  {source}: {count}")


def save_test_set(test_df: pd.DataFrame, output_dir: str):
    """保存测试集"""
    os.makedirs(output_dir, exist_ok=True)

    # CSV格式
    csv_path = os.path.join(output_dir, 'test_set.csv')
    test_df.to_csv(csv_path, index=False)
    print(f"\n已保存CSV: {csv_path}")

    # JSON格式 (更方便加载)
    json_path = os.path.join(output_dir, 'test_set.json')
    records = test_df.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"已保存JSON: {json_path}")

    # 文件列表 (方便快速加载)
    files_path = os.path.join(output_dir, 'test_files.txt')
    with open(files_path, 'w') as f:
        for file in test_df['file'].tolist():
            f.write(file + '\n')
    print(f"已保存文件列表: {files_path}")

    return csv_path, json_path, files_path


def main():
    parser = argparse.ArgumentParser(description='创建测试集')
    parser.add_argument('--annotation-csv', type=str,
                        default='data_annotation/symmetry_annotations.csv',
                        help='标注CSV文件路径')
    parser.add_argument('--p2v2-dir', type=str,
                        default='data/symmetry_classification_gt',
                        help='p2v2训练数据目录')
    parser.add_argument('--output-dir', type=str,
                        default='data/test_set',
                        help='输出目录')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--p2v2-samples-per-cat', type=int, default=10,
                        help='每个p2v2类别添加的样本数 (默认: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no-p2v2', action='store_true',
                        help='不混入p2v2样本')

    args = parser.parse_args()

    print("="*60)
    print("创建测试集")
    print("="*60)

    # 1. 加载标注数据
    print(f"\n加载标注数据: {args.annotation_csv}")
    df = load_annotations(args.annotation_csv)
    print(f"  总样本数: {len(df)}")

    # 2. 分层抽样
    print(f"\n执行分层抽样 (test_ratio={args.test_ratio})...")
    train_val_df, test_df = stratified_sample(df, args.test_ratio, args.seed)
    print(f"  训练+验证集: {len(train_val_df)}")
    print(f"  测试集: {len(test_df)}")

    # 3. 混入p2v2样本 (可选)
    if not args.no_p2v2:
        print(f"\n加载p2v2样本: {args.p2v2_dir}")
        p2v2_samples = load_p2v2_samples(args.p2v2_dir)
        print(f"  可用样本: {len(p2v2_samples)}")

        print(f"\n添加p2v2样本 (每类别{args.p2v2_samples_per_cat}个)...")
        test_df = add_p2v2_samples(test_df, p2v2_samples,
                                    args.p2v2_samples_per_cat, args.seed)

    # 4. 打印统计信息
    print_stats(test_df, "测试集")

    # 5. 保存
    print(f"\n保存测试集到: {args.output_dir}")
    save_test_set(test_df, args.output_dir)

    print("\n完成!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
增加 1_front 和 4_fronts 类别的数据

处理流程：
1. 读取现有数据集，找出已使用的样本
2. 从标注中选取尚未使用的样本
3. 对每个新样本：旋转点云、生成 GT
4. 保存到数据集目录
"""

import json
import numpy as np
import os
from pathlib import Path
import shutil

# 配置
ANNOTATION_FILE = "data_annotation/symmetry_annotations.json"
PLY_SOURCE_DIR = "data/full_mn40_normal_resampled_ply"
OUTPUT_DIR = "/home/pablo/ForwardNet/data/symmetry_classification_gt"
DATASET_INFO_FILE = os.path.join(OUTPUT_DIR, "dataset_info.json")

# 方向映射
DIRECTION_MAP = {
    '+X': np.array([1, 0, 0]),
    '-X': np.array([-1, 0, 0]),
    '+Y': np.array([0, 1, 0]),
    '-Y': np.array([0, -1, 0]),
    '+Z': np.array([0, 0, 1]),
    '-Z': np.array([0, 0, -1]),
}

def read_ply(filepath):
    """读取 PLY 文件，返回点云 (N, 6) - xyz + normals"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 找到 header 结束
    header_end = 0
    vertex_count = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        if line.strip() == 'end_header':
            header_end = i + 1
            break

    # 读取点云数据
    points = []
    for i in range(header_end, header_end + vertex_count):
        parts = lines[i].strip().split()
        points.append([float(x) for x in parts[:6]])  # xyz + normals

    return np.array(points)

def write_ply(filepath, points):
    """写入 PLY 文件"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
"""
    with open(filepath, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f} {p[5]:.6f}\n")

def rotate_y(points, angle_deg):
    """绕 Y 轴旋转点云"""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Y 轴旋转矩阵
    R = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0],
        [-sin_a, 0, cos_a]
    ])

    # 旋转位置和法线
    rotated = points.copy()
    rotated[:, :3] = points[:, :3] @ R.T
    rotated[:, 3:6] = points[:, 3:6] @ R.T

    return rotated

def direction_to_angle(direction_str, rotation_deg):
    """
    将方向字符串转换为旋转后的角度

    坐标系（XZ平面，Y轴向上）：
        +Z (90°)
          ↑
    -X (180°)←───→+X (0°)
          ↓
        -Z (270°)
    """
    # 原始方向对应的角度
    base_angles = {
        '+X': 0,
        '-X': 180,
        '+Z': 90,
        '-Z': 270,
    }

    if direction_str not in base_angles:
        return None

    base_angle = base_angles[direction_str]
    # 旋转后的角度（顺时针旋转 = 角度减少）
    new_angle = (base_angle - rotation_deg) % 360

    return new_angle

def generate_gt(category, angle_deg, kappa=50.0):
    """
    生成 4 峰 GT

    格式：weight mu_cos mu_sin kappa (一行一个峰)
    """
    lines = ["# Mixture of 4 von Mises distributions",
             "# Format: weight, mu_cos, mu_sin, kappa (one peak per line)"]

    weight = 0.25

    if category == '1_front':
        # 只有第一个峰有 kappa > 0
        angles = [angle_deg, 0, 0, 0]
        kappas = [kappa, 0, 0, 0]
    elif category == '2_fronts':
        # 两个相对的峰
        angles = [angle_deg, (angle_deg + 180) % 360, 0, 0]
        kappas = [kappa, kappa, 0, 0]
    elif category == '4_fronts':
        # 四个峰，间隔 90°
        angles = [angle_deg, (angle_deg + 90) % 360,
                  (angle_deg + 180) % 360, (angle_deg + 270) % 360]
        kappas = [kappa, kappa, kappa, kappa]
    else:  # symmetric, no_front
        angles = [0, 90, 180, 270]
        kappas = [0, 0, 0, 0]

    for i in range(4):
        angle_rad = np.radians(angles[i])
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        lines.append(f"{weight:.4f} {cos_val:.6f} {sin_val:.6f} {kappas[i]:.1f}")

    return '\n'.join(lines)

def main():
    np.random.seed(42)  # 固定随机种子

    # 读取标注
    with open(ANNOTATION_FILE) as f:
        annotations = json.load(f)

    # 读取现有数据集信息
    with open(DATASET_INFO_FILE) as f:
        dataset_info = json.load(f)

    # 找出已使用的样本名 (dataset_info 中的 key 是 basename，如 "airplane_0001.ply")
    used_basenames = set(dataset_info.keys())
    print(f"现有数据集样本数: {len(used_basenames)}")

    # 统计各类别
    category_counts = {}
    for info in dataset_info.values():
        cat = info['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print(f"各类别数量: {category_counts}")

    # 目标数量配置
    TARGET_COUNTS = {
        '1_front': 270,   # 当前 200，增加 70
        '4_fronts': 270,  # 当前 200，增加 70
    }

    # 找出可用的新样本
    new_samples = {'1_front': [], '4_fronts': []}

    for name, info in annotations.items():
        # 用 basename 检查是否已使用
        basename = os.path.basename(name)
        if basename in used_basenames:
            continue

        k = info.get('K')
        direction = info.get('front_direction')

        # 排除 OBLIQUE 和 MULTI
        if direction in ['OBLIQUE', 'MULTI']:
            continue

        if k == 1 and direction in DIRECTION_MAP:
            new_samples['1_front'].append((name, info))
        elif k == 4 and direction in DIRECTION_MAP:
            new_samples['4_fronts'].append((name, info))

    # 随机打乱
    for cat in new_samples:
        np.random.shuffle(new_samples[cat])

    # 计算需要增加的数量
    to_add = {}
    for cat in ['1_front', '4_fronts']:
        current = category_counts.get(cat, 0)
        target = TARGET_COUNTS[cat]
        available = len(new_samples[cat])
        need = target - current
        actual_add = min(need, available)
        to_add[cat] = actual_add
        new_samples[cat] = new_samples[cat][:actual_add]  # 只取需要的数量

    print(f"\n增加计划:")
    for cat in ['1_front', '4_fronts']:
        current = category_counts.get(cat, 0)
        print(f"  {cat}: {current} → {current + to_add[cat]} (+{to_add[cat]})")

    # 处理新样本
    added_counts = {'1_front': 0, '4_fronts': 0}

    for category in ['1_front', '4_fronts']:
        output_subdir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_subdir, exist_ok=True)

        for name, info in new_samples[category]:
            direction = info['front_direction']

            # 标注文件中的 name 格式: "airplane/airplane_0001.ply"
            # 实际文件路径: data/full_mn40_normal_resampled_ply/airplane/airplane_0001.ply
            ply_path = os.path.join(PLY_SOURCE_DIR, name)

            # 输出文件名只保留 basename
            out_filename = os.path.basename(name)

            if not os.path.exists(ply_path):
                print(f"  警告: 找不到 {ply_path}")
                continue

            # 读取点云
            points = read_ply(ply_path)

            # 生成随机旋转角度
            rotation_deg = np.random.uniform(0, 360)

            # 旋转点云
            rotated_points = rotate_y(points, rotation_deg)

            # 计算旋转后的前向角度
            new_front_angle = direction_to_angle(direction, rotation_deg)

            if new_front_angle is None:
                print(f"  警告: 无法处理方向 {direction}")
                continue

            # 生成 GT
            gt_content = generate_gt(category, new_front_angle)

            # 保存文件
            out_ply_path = os.path.join(output_subdir, out_filename)
            out_gt_path = os.path.join(output_subdir, out_filename.replace('.ply', '_gt.txt'))

            write_ply(out_ply_path, rotated_points)
            with open(out_gt_path, 'w') as f:
                f.write(gt_content)

            # 更新 dataset_info (使用 out_filename 作为 key，与现有数据格式一致)
            dataset_info[out_filename] = {
                'category': category,
                'num_fronts': 1 if category == '1_front' else 4,
                'original_front_direction': direction,
                'rotation_applied_deg': rotation_deg,
                'new_front_angle_deg': new_front_angle,
                'ply_file': out_filename,
                'gt_file': out_filename.replace('.ply', '_gt.txt')
            }

            added_counts[category] += 1

    # 保存更新后的 dataset_info
    with open(DATASET_INFO_FILE, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n新增数据:")
    print(f"  1_front: +{added_counts['1_front']} (总计 {category_counts.get('1_front', 0) + added_counts['1_front']})")
    print(f"  4_fronts: +{added_counts['4_fronts']} (总计 {category_counts.get('4_fronts', 0) + added_counts['4_fronts']})")

    # 最终统计
    print(f"\n最终数据集统计:")
    for cat in ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']:
        count = sum(1 for info in dataset_info.values() if info['category'] == cat)
        print(f"  {cat}: {count} 个")

if __name__ == '__main__':
    main()

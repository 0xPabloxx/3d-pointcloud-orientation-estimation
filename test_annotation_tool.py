#!/usr/bin/env python3
"""测试标注工具的基本功能"""

import sys
from pathlib import Path
import numpy as np

# 测试1: PLY加载
print("=" * 60)
print("测试1: PLY文件加载")
print("=" * 60)

ply_file = Path("data/full_mn40_normal_resampled_2d_rotated_ply/glass_box/glass_box_0001.ply")

if not ply_file.exists():
    print(f"❌ PLY文件不存在: {ply_file}")
    sys.exit(1)

try:
    with open(ply_file, 'r') as f:
        lines = f.readlines()

    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if 'element vertex' in line:
            num_vertices = int(line.split()[-1])
        if 'end_header' in line:
            header_end = i + 1
            break

    points = []
    for line in lines[header_end:header_end + num_vertices]:
        coords = line.strip().split()[:3]
        points.append([float(x) for x in coords])

    points = np.array(points)
    print(f"✅ 成功加载PLY文件")
    print(f"   点云形状: {points.shape}")
    print(f"   点云范围: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

except Exception as e:
    print(f"❌ 加载PLY失败: {e}")
    sys.exit(1)

# 测试2: matplotlib导入
print("\n" + "=" * 60)
print("测试2: Matplotlib后端")
print("=" * 60)

try:
    import matplotlib
    print(f"✅ Matplotlib版本: {matplotlib.__version__}")
    print(f"   当前后端: {matplotlib.get_backend()}")

    # 尝试导入pyplot
    import matplotlib.pyplot as plt
    print(f"✅ pyplot导入成功")

except Exception as e:
    print(f"❌ Matplotlib导入失败: {e}")
    sys.exit(1)

# 测试3: 创建简单的3D图
print("\n" + "=" * 60)
print("测试3: 创建3D图并保存")
print("=" * 60)

try:
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Glass Box Sample')

    # 保存图片
    output_path = Path("test_pointcloud_visualization.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✅ 成功创建并保存3D图")
    print(f"   输出文件: {output_path}")

except Exception as e:
    print(f"❌ 创建3D图失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: JSON保存
print("\n" + "=" * 60)
print("测试4: JSON标注保存")
print("=" * 60)

try:
    import json

    test_annotations = {
        "glass_box/glass_box_0001.ply": {
            "file": "glass_box/glass_box_0001.ply",
            "K": 4,
            "name": "4-peak (四正面,90°)",
            "index": 0
        }
    }

    output_json = Path("test_annotations.json")
    with open(output_json, 'w') as f:
        json.dump(test_annotations, f, indent=2)

    # 读回验证
    with open(output_json, 'r') as f:
        loaded = json.load(f)

    print(f"✅ 成功保存和加载JSON")
    print(f"   标注文件: {output_json}")
    print(f"   内容: {json.dumps(loaded, indent=2, ensure_ascii=False)}")

except Exception as e:
    print(f"❌ JSON操作失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
print("\n建议：")
print("1. GUI版本可能在WSL中需要X server支持")
print("2. 可以使用Web版标注工具（基于Plotly Dash）")
print("3. 或者使用命令行+图片预览的简化版本")

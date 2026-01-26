"""
D/DR系列实验可视化 - 离散方向预测

功能：
- 从标注数据加载原始点云和GT方向
- GT和点云一起随机旋转（保持对齐）
- 支持5种类别：1_front, 2_fronts, 4_fronts, symmetric, no_front
- 支持拼图功能

Usage:
    python paper_figures/discrete_visualization.py --all_categories --combine
    python paper_figures/discrete_visualization.py --num_bins 8 --all_categories --combine
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import json
import argparse

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'cm',
})

# 方向到角度映射
DIRECTION_TO_ANGLE = {
    '+X': 0.0,
    '+Z': np.pi / 2,
    '-X': np.pi,
    '-Z': 3 * np.pi / 2,
}

# 类别映射
SYMMETRY_TO_CATEGORY = {
    '1个正面': '1_front',
    '2个正面': '2_fronts',
    '4个正面': '4_fronts',
    '旋转对称': 'symmetric',
    '完全对称': 'symmetric',
    '无正面': 'no_front',
    '没有正面': 'no_front',
}


def load_annotations(annotation_file='data_annotation/symmetry_annotations.json'):
    """加载标注数据"""
    with open(project_root / annotation_file) as f:
        return json.load(f)


def load_ply(ply_path):
    """加载PLY点云"""
    full_path = project_root / 'data/full_mn40_normal_resampled_ply' / ply_path
    with open(full_path, 'r') as f:
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

    return np.array(points, dtype=np.float32)


def normalize_points(points):
    """归一化点云"""
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points


def rotate_points_y(points, angle):
    """绕Y轴旋转点云（XZ平面内旋转）"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    return points @ R.T


def create_gt_distribution(gt_angle, category, num_bins, temperature=5.0):
    """生成GT分布"""
    bin_width = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * bin_width

    # 根据类别确定等效角度
    if category == '1_front':
        equiv_angles = [gt_angle] if gt_angle is not None else []
    elif category == '2_fronts':
        equiv_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)] if gt_angle is not None else []
    elif category == '4_fronts':
        equiv_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)] if gt_angle is not None else []
    elif category == 'symmetric':
        # 旋转对称：均匀分布
        return np.ones(num_bins) / num_bins
    else:  # no_front
        # 无正面：均匀分布
        return np.ones(num_bins) / num_bins

    if not equiv_angles:
        return np.ones(num_bins) / num_bins

    dist = np.zeros(num_bins)
    for angle in equiv_angles:
        for i, center in enumerate(bin_centers):
            dist[i] += np.exp(temperature * np.cos(center - angle))

    return dist / dist.sum()


def render_pointcloud_topdown(points, ax, cmap='coolwarm'):
    """渲染点云俯视图"""
    if len(points) > 3000:
        indices = np.random.choice(len(points), 3000, replace=False)
        pts = points[indices]
    else:
        pts = points

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(x, z, c=y, cmap=cmap, s=1.5, alpha=0.8)

    max_range = max(np.abs(pts).max(), 0.5)
    ax.set_xlim(-max_range * 1.1, max_range * 1.1)
    ax.set_ylim(-max_range * 1.1, max_range * 1.1)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_discrete_gt(
    gt_angle,
    category,
    num_bins,
    points,
    sample_name,
    original_direction,
    rotation_angle,
    save_path=None,
    temperature=5.0,
    figsize=(10, 10),
    show_legend=True,
    show_gt_line=True,
):
    """绘制离散方向GT可视化"""
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax_polar = fig.add_subplot(111, projection='polar')

    bin_width = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * bin_width

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # ========== 1. Bin背景 ==========
    for i in range(num_bins):
        color = '#f0f4f8' if i % 2 == 0 else '#fafbfc'
        ax_polar.bar(
            bin_centers[i], 1.0,
            width=bin_width * 0.98,
            bottom=0.35,
            color=color,
            edgecolor='#d0d7de',
            linewidth=0.5,
            zorder=1,
        )

    # ========== 2. GT分布（绿色）==========
    gt_dist = create_gt_distribution(gt_angle, category, num_bins, temperature)
    gt_display = gt_dist / gt_dist.max() * 0.55 if gt_dist.max() > 0 else gt_dist

    # 绿色渐变配色
    colors = plt.cm.Greens(0.35 + gt_dist / gt_dist.max() * 0.55)
    ax_polar.bar(
        bin_centers, gt_display,
        width=bin_width * 0.80,
        bottom=0.35,
        color=colors,
        edgecolor='#1e8449',
        linewidth=1.2,
        zorder=3,
        label=r'$p_{gt}(\phi)$',
    )

    # ========== 3. GT方向线（红色）==========
    if show_gt_line:
        if category == '1_front' and gt_angle is not None:
            gt_angles = [gt_angle]
        elif category == '2_fronts' and gt_angle is not None:
            gt_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        elif category == '4_fronts' and gt_angle is not None:
            gt_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        else:
            gt_angles = []  # symmetric/no_front 没有GT方向线

        for i, angle in enumerate(gt_angles):
            ax_polar.plot(
                [angle, angle], [0.30, 1.05],
                color='#e74c3c', linewidth=2.5,
                solid_capstyle='round',
                zorder=10,
                label=r'$\phi_{gt}$' if i == 0 else None,
            )

    # ========== 4. 极坐标样式（无标签）==========
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([0.5, 0.7, 0.9])
    ax_polar.set_yticklabels(['', '', ''])

    ax_polar.set_xticks([])
    ax_polar.set_xticklabels([])

    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # ========== 5. 中心点云俯视图 ==========
    ax_center = fig.add_axes([0.35, 0.35, 0.30, 0.30])
    circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                    facecolor='white', edgecolor='#666666', linewidth=2, zorder=0)
    ax_center.add_patch(circle)

    render_pointcloud_topdown(points, ax_center)

    # 坐标轴指示
    arrow_len = 0.12
    ax_center.annotate('', xy=(0.5 + arrow_len, 0.5), xytext=(0.5, 0.5),
                      arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                      annotation_clip=False)
    ax_center.annotate('', xy=(0.5, 0.5 + arrow_len), xytext=(0.5, 0.5),
                      arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                      annotation_clip=False)
    ax_center.text(0.5 + arrow_len + 0.02, 0.5, 'X', fontsize=8, color='red',
                  ha='left', va='center', transform=ax_center.transAxes)
    ax_center.text(0.5, 0.5 + arrow_len + 0.02, 'Z', fontsize=8, color='blue',
                  ha='center', va='bottom', transform=ax_center.transAxes)

    # ========== 6. 图例 ==========
    if show_legend:
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05),
                       framealpha=0.95, edgecolor='#cccccc')

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")

    plt.close(fig)


def get_valid_samples(annotations):
    """获取所有有效样本，按类别组织"""
    samples_by_category = {
        '1_front': [],
        '2_fronts': [],
        '4_fronts': [],
        'symmetric': [],
        'no_front': [],
    }

    for file_path, ann in annotations.items():
        symmetry_name = ann.get('symmetry_name', '')
        category = SYMMETRY_TO_CATEGORY.get(symmetry_name)
        if not category:
            continue

        direction = ann.get('front_direction')
        gt_angle = DIRECTION_TO_ANGLE.get(direction) if direction else None

        # 对于1/2/4_front，需要有有效的方向
        if category in ['1_front', '2_fronts', '4_fronts']:
            if gt_angle is None:
                continue

        samples_by_category[category].append({
            'file': file_path,
            'category': category,
            'direction': direction,
            'gt_angle': gt_angle,
        })

    return samples_by_category


def create_combined_figure(image_paths, output_path, ncols=5, nrows=2, show_gt_line=True):
    """将多张图片拼成一张大图，右上角只显示一次图例

    按列排列：同一类别的样本在同一列
    """
    from PIL import Image

    images = [Image.open(p) for p in image_paths]

    # 获取单张图的大小
    w, h = images[0].size

    # 创建大图
    combined = Image.new('RGB', (w * ncols, h * nrows), (255, 255, 255))

    # 按列排列（同一类别在同一列）
    for i, img in enumerate(images):
        col = i // nrows  # 每nrows个图一列
        row = i % nrows
        combined.paste(img, (col * w, row * h))

    # 添加图例到中间缝隙（两行之间）
    fig_legend, ax_legend = plt.subplots(figsize=(1.8, 1.0))
    ax_legend.axis('off')

    # 创建图例元素
    legend_elements = []
    if show_gt_line:
        legend_elements.append(Line2D([0], [0], color='#e74c3c', linewidth=2.5, label=r'$\phi_{gt}$'))
    legend_elements.append(Patch(facecolor='#27ae60', edgecolor='#1e8449', label=r'$p_{gt}(\phi)$'))
    ax_legend.legend(handles=legend_elements, loc='center', framealpha=0.95,
                    edgecolor='#cccccc', fontsize=12)

    # 保存图例
    legend_path = output_path.parent / 'temp_legend.png'
    fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight',
                      facecolor='white', edgecolor='none', transparent=True)
    plt.close(fig_legend)

    # 将图例粘贴到中间缝隙位置（两行之间，靠右）
    legend_img = Image.open(legend_path)
    legend_w, legend_h = legend_img.size
    # 放在最后一列和倒数第二列之间的缝隙，垂直居中
    x_pos = w * (ncols - 1) - legend_w // 2  # 最后两列之间
    y_pos = h - legend_h // 2  # 两行之间
    combined.paste(legend_img, (x_pos, y_pos))

    # 删除临时图例文件
    legend_path.unlink()

    combined.save(output_path, dpi=(300, 300))
    print(f"Combined figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize discrete direction GT')
    parser.add_argument('--sample', type=str, help='Specific sample path')
    parser.add_argument('--category', type=str,
                       choices=['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front'],
                       help='Filter by category')
    parser.add_argument('--all_categories', action='store_true',
                       help='Generate samples for all 5 categories')
    parser.add_argument('--num_samples', type=int, default=2, help='Samples per category')
    parser.add_argument('--num_bins', type=int, default=16, help='Number of bins')
    parser.add_argument('--output_dir', type=str, default='paper_figures/discrete_vis')
    parser.add_argument('--list', action='store_true', help='List available samples')
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--combine', action='store_true', help='Combine all into one figure')
    parser.add_argument('--ncols', type=int, default=5, help='Columns in combined figure')
    parser.add_argument('--no_gt_line', action='store_true', help='Do not draw GT direction lines')

    args = parser.parse_args()
    np.random.seed(args.seed)

    annotations = load_annotations()
    samples_by_category = get_valid_samples(annotations)

    print("Samples by category:")
    for cat, samples in samples_by_category.items():
        print(f"  {cat}: {len(samples)}")

    if args.list:
        print("\n=== Available Samples ===")
        for cat, samples in samples_by_category.items():
            print(f"\n{cat}: {len(samples)} samples")
            for s in samples[:5]:
                print(f"  {s['file']} (Dir: {s['direction']})")
            if len(samples) > 5:
                print(f"  ... and {len(samples) - 5} more")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要处理的类别
    if args.all_categories:
        categories_to_process = ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']
    elif args.category:
        categories_to_process = [args.category]
    elif args.sample:
        # 单个样本模式
        for cat, samples in samples_by_category.items():
            for s in samples:
                if s['file'] == args.sample:
                    categories_to_process = [cat]
                    break
        else:
            print(f"Error: Sample '{args.sample}' not found")
            return
    else:
        categories_to_process = ['1_front']  # 默认

    saved_paths = []

    # 处理每个类别
    for category in categories_to_process:
        samples = samples_by_category[category]
        if not samples:
            print(f"\nNo samples for {category}")
            continue

        print(f"\n=== Processing {category} ===")

        # 选择样本
        if args.sample:
            selected = [s for s in samples if s['file'] == args.sample]
        else:
            num = min(args.num_samples, len(samples))
            indices = np.random.choice(len(samples), num, replace=False)
            selected = [samples[i] for i in indices]

        for i, sample_info in enumerate(selected):
            file_path = sample_info['file']
            original_gt_angle = sample_info['gt_angle']
            original_direction = sample_info['direction']

            # 加载点云
            points = load_ply(file_path)
            points = normalize_points(points)

            # 随机旋转角度
            rotation_angle = np.random.uniform(0, 2 * np.pi)

            # 旋转点云
            rotated_points = rotate_points_y(points, rotation_angle)

            # 旋转GT角度（如果有）- 点云顺时针旋转，所以GT角度要减
            if original_gt_angle is not None:
                rotated_gt_angle = (original_gt_angle - rotation_angle) % (2 * np.pi)
            else:
                rotated_gt_angle = None

            # 保存文件名
            suffix = "_no_gt_line" if args.no_gt_line else ""
            save_name = f"{category}_{i+1:02d}_{file_path.replace('/', '_').replace('.ply', '')}_{args.num_bins}bins{suffix}.png"
            save_path = output_dir / save_name

            plot_discrete_gt(
                gt_angle=rotated_gt_angle,
                category=category,
                num_bins=args.num_bins,
                points=rotated_points,
                sample_name=file_path,
                original_direction=original_direction,
                rotation_angle=rotation_angle,
                save_path=save_path,
                temperature=args.temperature,
                show_legend=not args.combine,  # 拼图模式不显示单独图例
                show_gt_line=not args.no_gt_line,
            )
            saved_paths.append(save_path)

    # 拼图
    if args.combine and len(saved_paths) > 1:
        suffix = "_no_gt_line" if args.no_gt_line else ""
        combined_path = output_dir / f"combined_{args.num_bins}bins{suffix}.png"
        create_combined_figure(saved_paths, combined_path, ncols=len(categories_to_process),
                              nrows=args.num_samples, show_gt_line=not args.no_gt_line)

    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    main()

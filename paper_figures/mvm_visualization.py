"""
MvM (Mixture of von Mises) 极坐标可视化工具

功能：
- 从标注数据加载点云和GT方向
- GT和点云一起随机旋转（保持对齐）
- 绘制连续的 von Mises mixture 概率密度曲线
- 支持5种类别：1_front, 2_fronts, 4_fronts, symmetric, no_front

Usage:
    python paper_figures/mvm_visualization.py --all_categories --combine
    python paper_figures/mvm_visualization.py --category 1_front --num_samples 3
    python paper_figures/mvm_visualization.py --sample airplane/airplane_0001.ply
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
from scipy.stats import vonmises
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

# 方向到角度映射 (XZ平面内，+X=0, +Z=90度)
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


def mixture_pdf(theta, mus, kappas, weights):
    """计算 von Mises mixture PDF"""
    pdf = np.zeros_like(theta, dtype=float)
    for mu, kappa, w in zip(mus, kappas, weights):
        if kappa < 0.01:
            # Uniform distribution
            pdf += w / (2 * np.pi)
        else:
            pdf += w * vonmises.pdf(theta, kappa=kappa, loc=mu)
    return pdf


def create_vonmises_gt(gt_angle, category, kappa=10.0):
    """根据类别创建 von Mises mixture GT 分布

    Args:
        gt_angle: 主方向角度 (radians)
        category: 对称类别
        kappa: 集中度参数

    Returns:
        mus: list of angles in radians
        kappas: list of kappa values
        weights: list of weights (normalized)
    """
    if category == '1_front':
        mus = [gt_angle] if gt_angle is not None else [0.0]
        kappas_list = [kappa]
        weights = [1.0]
    elif category == '2_fronts':
        if gt_angle is not None:
            mus = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
        else:
            mus = [0.0, np.pi]
        kappas_list = [kappa, kappa]
        weights = [0.5, 0.5]
    elif category == '4_fronts':
        if gt_angle is not None:
            mus = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]
        else:
            mus = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        kappas_list = [kappa] * 4
        weights = [0.25] * 4
    elif category == 'symmetric':
        # 旋转对称：kappa=0 表示均匀分布
        mus = [0.0]
        kappas_list = [0.0]
        weights = [1.0]
    else:  # no_front
        mus = [0.0]
        kappas_list = [0.0]
        weights = [1.0]

    return mus, kappas_list, weights


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


def plot_vonmises_gt(
    gt_angle,
    category,
    points,
    sample_name,
    save_path=None,
    kappa=10.0,
    figsize=(10, 10),
    show_legend=True,
    fill_alpha=0.3,
    line_color='#27ae60',
    fill_color='#27ae60',
    gt_line_color='#e74c3c',
    show_components=False,
):
    """绘制 von Mises mixture 极坐标可视化

    Args:
        gt_angle: 主GT方向角度 (radians)
        category: 对称类别
        points: 点云数组 (N, 3)
        sample_name: 样本名称（用于标题）
        save_path: 保存路径
        kappa: von Mises 集中度参数
        figsize: 图像尺寸
        show_legend: 是否显示图例
        fill_alpha: 填充透明度
        line_color: 曲线颜色
        fill_color: 填充颜色
        gt_line_color: GT方向线颜色
        show_components: 是否显示各分量
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax_polar = fig.add_subplot(111, projection='polar')

    # 创建 von Mises mixture 参数
    mus, kappas, weights = create_vonmises_gt(gt_angle, category, kappa)

    # 计算 PDF
    theta = np.linspace(0, 2 * np.pi, 360)
    weights_norm = np.array(weights) / np.sum(weights)
    pdf = mixture_pdf(theta, mus, kappas, weights_norm)

    # 归一化 PDF 用于显示（映射到 0.35-0.9 范围）
    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # ========== 1. 背景圆环 ==========
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # ========== 2. 各分量（可选）==========
    if show_components and len(mus) > 1:
        colors_comp = plt.cm.Set2(np.linspace(0, 1, len(mus)))
        for i, (mu, k, w) in enumerate(zip(mus, kappas, weights_norm)):
            if k < 0.01:
                comp_pdf = np.ones_like(theta) / (2 * np.pi)
            else:
                comp_pdf = vonmises.pdf(theta, kappa=k, loc=mu)
            comp_display = 0.35 + (comp_pdf / pdf_max) * 0.55
            ax_polar.plot(theta, comp_display, color=colors_comp[i], linewidth=1.5,
                         linestyle='--', alpha=0.6, zorder=4)

    # ========== 3. Mixture 曲线（填充）==========
    ax_polar.fill(theta, pdf_display, color=fill_color, alpha=fill_alpha, zorder=3)
    ax_polar.plot(theta, pdf_display, color=line_color, linewidth=2.5, zorder=5,
                 label=r'$p(\phi)$')

    # ========== 4. GT 方向线 ==========
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
            color=gt_line_color, linewidth=2.5,
            solid_capstyle='round',
            zorder=10,
            label=r'$\phi_{gt}$' if i == 0 else None,
        )

    # ========== 5. 极坐标样式 ==========
    ax_polar.set_ylim(0, 1.2)
    ax_polar.set_yticks([0.5, 0.7, 0.9])
    ax_polar.set_yticklabels(['', '', ''])
    ax_polar.set_xticks([])
    ax_polar.set_xticklabels([])
    ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')
    ax_polar.spines['polar'].set_visible(True)
    ax_polar.spines['polar'].set_color('#999999')

    # ========== 6. 中心点云俯视图 ==========
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

    # ========== 7. 图例 ==========
    if show_legend:
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05),
                       framealpha=0.95, edgecolor='#cccccc')

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


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


def create_combined_figure(image_paths, output_path, ncols=5, nrows=2):
    """将多张图片拼成一张大图"""
    from PIL import Image

    images = [Image.open(p) for p in image_paths]
    w, h = images[0].size

    combined = Image.new('RGB', (w * ncols, h * nrows), (255, 255, 255))

    # 按列排列（同一类别在同一列）
    for i, img in enumerate(images):
        col = i // nrows
        row = i % nrows
        combined.paste(img, (col * w, row * h))

    # 添加图例
    fig_legend, ax_legend = plt.subplots(figsize=(1.8, 1.0))
    ax_legend.axis('off')

    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=2.5, label=r'$\phi_{gt}$'),
        Patch(facecolor='#27ae60', alpha=0.3, edgecolor='#27ae60', linewidth=2, label=r'$p(\phi)$'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', framealpha=0.95,
                    edgecolor='#cccccc', fontsize=12)

    legend_path = output_path.parent / 'temp_legend_mvm.png'
    fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight',
                      facecolor='white', edgecolor='none', transparent=True)
    plt.close(fig_legend)

    legend_img = Image.open(legend_path)
    legend_w, legend_h = legend_img.size
    x_pos = w * (ncols - 1) - legend_w // 2
    y_pos = h - legend_h // 2
    combined.paste(legend_img, (x_pos, y_pos))

    legend_path.unlink()

    combined.save(output_path, dpi=(300, 300))
    print(f"Combined figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize von Mises mixture GT distributions')
    parser.add_argument('--sample', type=str, help='Specific sample path (e.g., airplane/airplane_0001.ply)')
    parser.add_argument('--category', type=str,
                       choices=['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front'],
                       help='Filter by category')
    parser.add_argument('--all_categories', action='store_true',
                       help='Generate samples for all 5 categories')
    parser.add_argument('--num_samples', type=int, default=2, help='Samples per category')
    parser.add_argument('--kappa', type=float, default=10.0, help='Von Mises concentration parameter')
    parser.add_argument('--output_dir', type=str, default='paper_figures/mvm')
    parser.add_argument('--list', action='store_true', help='List available samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--combine', action='store_true', help='Combine all into one figure')
    parser.add_argument('--ncols', type=int, default=5, help='Columns in combined figure')
    parser.add_argument('--show_components', action='store_true', help='Show individual mixture components')
    parser.add_argument('--no_rotate', action='store_true', help='Do not randomly rotate point cloud')

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
        found = False
        for cat, samples in samples_by_category.items():
            for s in samples:
                if s['file'] == args.sample:
                    categories_to_process = [cat]
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"Error: Sample '{args.sample}' not found in annotations")
            return
    else:
        categories_to_process = ['1_front']

    saved_paths = []

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

            # 加载点云
            points = load_ply(file_path)
            points = normalize_points(points)

            # 随机旋转
            if args.no_rotate:
                rotation_angle = 0.0
                rotated_points = points
                rotated_gt_angle = original_gt_angle
            else:
                rotation_angle = np.random.uniform(0, 2 * np.pi)
                rotated_points = rotate_points_y(points, rotation_angle)
                if original_gt_angle is not None:
                    rotated_gt_angle = (original_gt_angle - rotation_angle) % (2 * np.pi)
                else:
                    rotated_gt_angle = None

            # 保存文件名
            save_name = f"{category}_{i+1:02d}_{file_path.replace('/', '_').replace('.ply', '')}_mvm.png"
            save_path = output_dir / save_name

            print(f"  [{i+1}] {file_path}")
            print(f"      Original GT: {np.degrees(original_gt_angle) if original_gt_angle else 'None'}deg")
            print(f"      Rotation: {np.degrees(rotation_angle):.1f}deg")
            print(f"      Rotated GT: {np.degrees(rotated_gt_angle) if rotated_gt_angle else 'None'}deg")

            plot_vonmises_gt(
                gt_angle=rotated_gt_angle,
                category=category,
                points=rotated_points,
                sample_name=file_path,
                save_path=save_path,
                kappa=args.kappa,
                show_legend=not args.combine,
                show_components=args.show_components,
            )
            saved_paths.append(save_path)

    # 拼图
    if args.combine and len(saved_paths) > 1:
        combined_path = output_dir / f"combined_mvm.png"
        create_combined_figure(saved_paths, combined_path, ncols=len(categories_to_process), nrows=args.num_samples)

    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    main()

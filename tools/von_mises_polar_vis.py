#!/usr/bin/env python
"""
Von Mises Mixture 极坐标可视化工具

功能：
- 中心显示点云俯视图
- 外圈显示 von Mises mixture 概率密度曲线（连续）
- 显示 GT 方向线

Usage:
    # 交互式Web界面
    python tools/von_mises_polar_vis.py --web --port 8089

    # 直接生成图片
    python tools/von_mises_polar_vis.py --mus 0 90 180 270 --kappas 10 --output test.png

    # 从MVM GT文件生成（点云+GT分布）
    python tools/von_mises_polar_vis.py --sample glass_box/glass_box_0195.ply \
        --output paper_figures/mvm/glass_box_0195_mvm.png

    # 批量生成（自动创建子目录）
    python tools/von_mises_polar_vis.py --batch --num-samples 12 --categories glass_box,bottle
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import vonmises
import argparse
import time
import json

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


def mixture_pdf(theta, mus, kappas, weights):
    """计算 von Mises mixture PDF"""
    pdf = np.zeros_like(theta, dtype=float)
    for mu, kappa, w in zip(mus, kappas, weights):
        if kappa < 0.01:
            # Uniform
            pdf += w / (2 * np.pi)
        else:
            pdf += w * vonmises.pdf(theta, kappa=kappa, loc=mu)
    return pdf


def normalize_weights(weights):
    """归一化权重"""
    arr = np.array(weights, dtype=float)
    return arr / arr.sum()


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


def read_mvm_gt(gt_path):
    """读取MVM GT文件，返回mu/kappa/weight数组"""
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError(f"Empty GT file: {gt_path}")

    data_lines = lines
    if lines[0].lower().startswith('k'):
        data_lines = lines[2:] if len(lines) > 2 else []

    if not data_lines:
        raise ValueError(f"No GT data rows in: {gt_path}")

    data = np.loadtxt(data_lines, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mu = data[:, 0]
    kappa = data[:, 1]
    if data.shape[1] >= 3:
        weight = data[:, 2]
    else:
        weight = np.ones_like(mu) / len(mu)

    return mu, kappa, weight


def resolve_gt_path(sample_path, gt_root):
    """根据sample路径解析对应的MVM GT文件路径"""
    sample_path = Path(sample_path)
    if len(sample_path.parts) < 2:
        raise ValueError("Sample path should include category/name, e.g. glass_box/glass_box_0195.ply")

    category = sample_path.parts[0]
    stem = sample_path.stem
    return Path(gt_root) / category / f"{stem}_multi_peak_vM_gt.txt"


def list_mvm_samples(gt_root, categories=None):
    """列出所有可用的MVM GT样本（与PLY匹配）"""
    gt_root = Path(gt_root)
    if categories:
        category_list = [c.strip() for c in categories if c.strip()]
    else:
        category_list = [p.name for p in gt_root.iterdir() if p.is_dir()]

    samples = []
    for category in category_list:
        cat_dir = gt_root / category
        if not cat_dir.is_dir():
            continue
        for gt_path in cat_dir.glob("*_multi_peak_vM_gt.txt"):
            stem = gt_path.stem.replace("_multi_peak_vM_gt", "")
            sample_rel = Path(category) / f"{stem}.ply"
            ply_path = project_root / 'data/full_mn40_normal_resampled_ply' / sample_rel
            if not ply_path.exists():
                continue
            samples.append({
                'category': category,
                'sample_rel': sample_rel,
                'gt_path': gt_path,
            })
    return samples


def render_pointcloud_topdown(points, ax, cmap='coolwarm', fixed_range=1.0, point_size=1.8, alpha=0.75):
    """渲染点云俯视图"""
    if len(points) > 3000:
        indices = np.random.choice(len(points), 3000, replace=False)
        pts = points[indices]
    else:
        pts = points

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(x, z, c=y, cmap=cmap, s=point_size, alpha=alpha)

    if fixed_range is None:
        max_range = max(np.abs(pts).max(), 0.5)
    else:
        max_range = fixed_range
    margin = max_range * 1.05
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect('equal')
    ax.axis('off')


def create_vonmises_gt(gt_angle, category, kappa=10.0):
    """根据类别创建 von Mises mixture GT 分布

    Returns:
        mus: list of angles in radians
        kappas: list of kappa values
        weights: list of weights (normalized)
    """
    if category == '1_front':
        mus = [gt_angle] if gt_angle is not None else []
        kappas_list = [kappa]
        weights = [1.0]
    elif category == '2_fronts':
        mus = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)] if gt_angle is not None else []
        kappas_list = [kappa, kappa]
        weights = [0.5, 0.5]
    elif category == '4_fronts':
        mus = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)] if gt_angle is not None else []
        kappas_list = [kappa] * 4
        weights = [0.25] * 4
    elif category == 'symmetric':
        # 旋转对称：用很小的 kappa 表示均匀分布
        mus = [0.0]
        kappas_list = [0.0]  # kappa=0 means uniform
        weights = [1.0]
    else:  # no_front
        mus = [0.0]
        kappas_list = [0.0]
        weights = [1.0]

    return mus, kappas_list, weights


def plot_vonmises_polar(
    mus,
    kappas,
    weights,
    gt_angles=None,
    points=None,
    sample_name=None,
    save_path=None,
    figsize=(10, 10),
    show_legend=True,
    fill_alpha=0.3,
    line_color='#27ae60',
    fill_color='#27ae60',
    gt_line_color='#e74c3c',
    show_components=False,
    pointcloud_range=1.0,
):
    """绘制 von Mises mixture 极坐标可视化

    Args:
        mus: list of mean angles in radians
        kappas: list of kappa values
        weights: list of weights
        gt_angles: list of GT direction angles to draw as lines
        points: point cloud array (N, 3) for center visualization
        sample_name: name for title
        save_path: path to save figure
        figsize: figure size
        show_legend: whether to show legend
        fill_alpha: transparency of filled area
        line_color: color of the mixture curve
        fill_color: color of the fill
        gt_line_color: color of GT direction lines
        show_components: whether to show individual components
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax_polar = fig.add_subplot(111, projection='polar')

    # 计算 PDF
    theta = np.linspace(0, 2 * np.pi, 360)
    weights_norm = normalize_weights(weights)
    pdf = mixture_pdf(theta, mus, kappas, weights_norm)

    # 归一化 PDF 用于显示（映射到 0.35-0.9 范围）
    pdf_max = pdf.max() if pdf.max() > 0 else 1.0
    pdf_display = 0.35 + (pdf / pdf_max) * 0.55

    ax_polar.set_theta_offset(0)
    ax_polar.set_theta_direction(1)

    # ========== 1. 背景圆环 ==========
    # 画一个浅色背景
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

    # ========== 2. 各分量（可选）==========
    if show_components and len(mus) > 1:
        colors = plt.cm.Set2(np.linspace(0, 1, len(mus)))
        for i, (mu, kappa, w) in enumerate(zip(mus, kappas, weights_norm)):
            if kappa < 0.01:
                comp_pdf = np.ones_like(theta) / (2 * np.pi)
            else:
                comp_pdf = vonmises.pdf(theta, kappa=kappa, loc=mu)
            comp_display = 0.35 + (comp_pdf / pdf_max) * 0.55
            ax_polar.plot(theta, comp_display, color=colors[i], linewidth=1.5,
                         linestyle='--', alpha=0.6, zorder=4)

    # ========== 3. Mixture 曲线（填充）==========
    ax_polar.fill(theta, pdf_display, color=fill_color, alpha=fill_alpha, zorder=3)
    ax_polar.plot(theta, pdf_display, color=line_color, linewidth=2.5, zorder=5,
                 label=r'$p(\phi)$ (von Mises mixture)')

    # ========== 4. GT 方向线 ==========
    if gt_angles:
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
    if points is not None:
        ax_center = fig.add_axes([0.35, 0.35, 0.30, 0.30])
        circle = Circle((0.5, 0.5), 0.48, transform=ax_center.transAxes,
                        facecolor='#fafafa', edgecolor='#666666', linewidth=2, zorder=0)
        ax_center.add_patch(circle)

        render_pointcloud_topdown(points, ax_center, fixed_range=pointcloud_range)

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


# ========== Web Interface ==========
def run_web_server(port=8089):
    """启动 Web 服务器"""
    from flask import Flask, render_template_string, jsonify, request, send_file
    import io
    import base64

    app = Flask(__name__)

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Von Mises Polar Visualizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 28px;
            background: linear-gradient(90deg, #4ecca3, #e94560);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .sidebar {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
        }
        .main {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        #plot-img {
            max-width: 100%;
            max-height: 700px;
            border-radius: 10px;
        }
        .section {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        .section-title {
            font-size: 14px;
            color: #e94560;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .control-row {
            margin-bottom: 12px;
        }
        .control-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
            font-size: 13px;
        }
        .control-label span:first-child { color: #aaa; }
        .control-label .value { color: #4ecca3; font-weight: bold; font-family: monospace; }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #333;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #e94560;
            cursor: pointer;
        }
        select, button, input[type="text"] {
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 10px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }
        button {
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            border: none;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .preset-btn {
            background: linear-gradient(135deg, #0f3460, #16213e);
            border: 1px solid #4ecca3;
            color: #4ecca3;
        }
        .preset-btn:hover {
            background: linear-gradient(135deg, #4ecca3, #1abc9c);
            color: #1a1a2e;
        }
        .peak-controls {
            background: rgba(15, 52, 96, 0.5);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .peak-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .peak-title { font-weight: bold; color: #4ecca3; }
        label { color: #aaa; font-size: 12px; }
        input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Von Mises Mixture Polar Visualizer</h1>
        <p style="color: #888; margin-top: 5px;">Continuous probability density on polar plot with point cloud</p>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="section">
                <div class="section-title">Presets</div>
                <button class="preset-btn" onclick="loadPreset('1_front')">1 Front (Single Peak)</button>
                <button class="preset-btn" onclick="loadPreset('2_fronts')">2 Fronts (Bimodal)</button>
                <button class="preset-btn" onclick="loadPreset('4_fronts')">4 Fronts (Quad)</button>
                <button class="preset-btn" onclick="loadPreset('symmetric')">Symmetric (Uniform)</button>
            </div>

            <div class="section">
                <div class="section-title">Global Kappa</div>
                <div class="control-row">
                    <div class="control-label">
                        <span>kappa</span>
                        <span class="value" id="global-kappa-val">10.0</span>
                    </div>
                    <input type="range" id="global-kappa" min="0" max="50" step="0.5" value="10"
                           oninput="applyGlobalKappa(this.value)">
                </div>
            </div>

            <div class="section">
                <div class="section-title">Peak Controls (up to 4)</div>
                <div id="peak-controls"></div>
            </div>

            <div class="section">
                <div class="section-title">Display Options</div>
                <label>
                    <input type="checkbox" id="show-components" onchange="updatePlot()">
                    Show individual components
                </label>
                <br><br>
                <label>
                    <input type="checkbox" id="show-gt-lines" checked onchange="updatePlot()">
                    Show GT direction lines
                </label>
            </div>

            <button onclick="updatePlot()">Update Plot</button>
        </div>

        <div class="main">
            <img id="plot-img" src="" alt="Loading...">
        </div>
    </div>

    <script>
        const MAX_PEAKS = 4;
        let peaks = [
            {active: true, mu: 180, kappa: 10, weight: 1},
            {active: false, mu: 90, kappa: 10, weight: 1},
            {active: false, mu: 180, kappa: 10, weight: 1},
            {active: false, mu: 270, kappa: 10, weight: 1},
        ];

        const colors = ['#e94560', '#4ecca3', '#f39c12', '#3498db'];

        function initControls() {
            const container = document.getElementById('peak-controls');
            container.innerHTML = '';

            for (let i = 0; i < MAX_PEAKS; i++) {
                const peak = peaks[i];
                const color = colors[i];

                const div = document.createElement('div');
                div.className = 'peak-controls';
                div.style.borderLeft = `3px solid ${color}`;
                div.innerHTML = `
                    <div class="peak-header">
                        <span class="peak-title" style="color: ${color}">Peak ${i + 1}</span>
                        <label>
                            <input type="checkbox" ${peak.active ? 'checked' : ''}
                                   onchange="togglePeak(${i}, this.checked)">
                            Active
                        </label>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>mu (angle)</span>
                            <span class="value" id="mu-val-${i}">${peak.mu} deg</span>
                        </div>
                        <input type="range" id="mu-${i}" min="0" max="360" step="1" value="${peak.mu}"
                               oninput="updatePeak(${i}, 'mu', this.value)" ${!peak.active ? 'disabled' : ''}>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>kappa</span>
                            <span class="value" id="kappa-val-${i}">${peak.kappa.toFixed(1)}</span>
                        </div>
                        <input type="range" id="kappa-${i}" min="0" max="50" step="0.5" value="${peak.kappa}"
                               oninput="updatePeak(${i}, 'kappa', this.value)" ${!peak.active ? 'disabled' : ''}>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>weight</span>
                            <span class="value" id="weight-val-${i}">${peak.weight.toFixed(2)}</span>
                        </div>
                        <input type="range" id="weight-${i}" min="0" max="2" step="0.05" value="${peak.weight}"
                               oninput="updatePeak(${i}, 'weight', this.value)" ${!peak.active ? 'disabled' : ''}>
                    </div>
                `;
                container.appendChild(div);
            }
        }

        function togglePeak(idx, active) {
            peaks[idx].active = active;
            const inputs = document.querySelectorAll(`#mu-${idx}, #kappa-${idx}, #weight-${idx}`);
            inputs.forEach(inp => inp.disabled = !active);
            updatePlot();
        }

        function updatePeak(idx, param, value) {
            peaks[idx][param] = parseFloat(value);
            const suffix = param === 'mu' ? ' deg' : '';
            document.getElementById(`${param}-val-${idx}`).textContent =
                param === 'mu' ? `${value} deg` : parseFloat(value).toFixed(param === 'kappa' ? 1 : 2);
            updatePlot();
        }

        function applyGlobalKappa(value) {
            document.getElementById('global-kappa-val').textContent = parseFloat(value).toFixed(1);
            peaks.forEach((peak, i) => {
                peak.kappa = parseFloat(value);
                document.getElementById(`kappa-${i}`).value = value;
                document.getElementById(`kappa-val-${i}`).textContent = parseFloat(value).toFixed(1);
            });
            updatePlot();
        }

        function loadPreset(name) {
            switch(name) {
                case '1_front':
                    peaks.forEach((p, i) => { p.active = i === 0; p.kappa = 10; p.weight = 1; });
                    peaks[0].mu = 180;
                    break;
                case '2_fronts':
                    peaks.forEach((p, i) => { p.active = i < 2; p.kappa = 10; p.weight = 1; });
                    peaks[0].mu = 180;
                    peaks[1].mu = 0;
                    break;
                case '4_fronts':
                    peaks.forEach((p, i) => { p.active = true; p.kappa = 10; p.weight = 1; });
                    peaks[0].mu = 0;
                    peaks[1].mu = 90;
                    peaks[2].mu = 180;
                    peaks[3].mu = 270;
                    break;
                case 'symmetric':
                    peaks.forEach((p, i) => { p.active = i === 0; p.weight = 1; });
                    peaks[0].mu = 0;
                    peaks[0].kappa = 0;
                    break;
            }
            initControls();
            updatePlot();
        }

        async function updatePlot() {
            const activePeaks = peaks.filter(p => p.active);
            if (activePeaks.length === 0) {
                activePeaks.push({mu: 0, kappa: 0, weight: 1});
            }

            const totalWeight = activePeaks.reduce((sum, p) => sum + p.weight, 0);
            const normalizedWeights = activePeaks.map(p => p.weight / totalWeight);

            const showComponents = document.getElementById('show-components').checked;
            const showGtLines = document.getElementById('show-gt-lines').checked;

            const resp = await fetch('/api/plot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    mus: activePeaks.map(p => p.mu),
                    kappas: activePeaks.map(p => p.kappa),
                    weights: normalizedWeights,
                    show_components: showComponents,
                    show_gt_lines: showGtLines,
                })
            });

            const data = await resp.json();
            document.getElementById('plot-img').src = 'data:image/png;base64,' + data.image;
        }

        initControls();
        updatePlot();
    </script>
</body>
</html>
'''

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/plot', methods=['POST'])
    def api_plot():
        data = request.json
        mus_deg = np.array(data['mus'])
        kappas = np.array(data['kappas'])
        weights = np.array(data['weights'])
        show_components = data.get('show_components', False)
        show_gt_lines = data.get('show_gt_lines', True)

        mus_rad = np.deg2rad(mus_deg)

        # GT 方向就是 mus
        gt_angles = mus_rad.tolist() if show_gt_lines else None

        # 生成图片
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax_polar = fig.add_subplot(111, projection='polar')

        theta = np.linspace(0, 2 * np.pi, 360)
        weights_norm = weights / weights.sum()
        pdf = mixture_pdf(theta, mus_rad, kappas, weights_norm)

        pdf_max = pdf.max() if pdf.max() > 0 else 1.0
        pdf_display = 0.35 + (pdf / pdf_max) * 0.55

        ax_polar.set_theta_offset(0)
        ax_polar.set_theta_direction(1)

        # 背景
        theta_bg = np.linspace(0, 2 * np.pi, 100)
        ax_polar.fill_between(theta_bg, 0.35, 1.0, color='#f0f4f8', alpha=0.5, zorder=1)

        # 分量
        if show_components and len(mus_rad) > 1:
            colors_comp = plt.cm.Set2(np.linspace(0, 1, len(mus_rad)))
            for i, (mu, kappa, w) in enumerate(zip(mus_rad, kappas, weights_norm)):
                if kappa < 0.01:
                    comp_pdf = np.ones_like(theta) / (2 * np.pi)
                else:
                    comp_pdf = vonmises.pdf(theta, kappa=kappa, loc=mu)
                comp_display = 0.35 + (comp_pdf / pdf_max) * 0.55
                ax_polar.plot(theta, comp_display, color=colors_comp[i], linewidth=1.5,
                             linestyle='--', alpha=0.6, zorder=4)

        # Mixture
        ax_polar.fill(theta, pdf_display, color='#27ae60', alpha=0.3, zorder=3)
        ax_polar.plot(theta, pdf_display, color='#27ae60', linewidth=2.5, zorder=5)

        # GT 线
        if gt_angles:
            for angle in gt_angles:
                ax_polar.plot([angle, angle], [0.30, 1.05], color='#e74c3c', linewidth=2.5, zorder=10)

        ax_polar.set_ylim(0, 1.2)
        ax_polar.set_yticks([])
        ax_polar.set_xticks([])
        ax_polar.grid(True, alpha=0.3, linestyle='-', color='#cccccc')

        # 保存到 buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'image': img_base64})

    print(f"\n{'='*60}")
    print("  Von Mises Polar Visualizer")
    print(f"{'='*60}")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='Von Mises Mixture Polar Visualizer')
    parser.add_argument('--web', action='store_true', help='Run web server')
    parser.add_argument('--port', type=int, default=8089, help='Web server port')

    # 直接生成模式
    parser.add_argument('--mus', type=float, nargs='+', help='Mean angles in degrees')
    parser.add_argument('--kappas', type=float, nargs='+', help='Kappa values')
    parser.add_argument('--weights', type=float, nargs='+', help='Weights')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--show-components', action='store_true')

    # 从GT文件生成
    parser.add_argument('--sample', type=str, help='Sample ply path (e.g. glass_box/glass_box_0195.ply)')
    parser.add_argument('--gt-file', type=str, help='Explicit MVM GT file path')
    parser.add_argument('--gt-root', type=str, default='data/MN40_multi_peak_vM_gt',
                        help='Root dir for MVM GT files')
    parser.add_argument('--rotation-deg', type=float, default=0.0,
                        help='Rotate point cloud around Y axis (degrees)')
    parser.add_argument('--pointcloud-range', type=float, default=1.0,
                        help='Fixed point cloud range (set to -1 for auto)')

    # 批量生成
    parser.add_argument('--batch', action='store_true', help='Generate a batch from MVM GT dataset')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples in batch')
    parser.add_argument('--categories', type=str,
                        help='Comma-separated category list (e.g. glass_box,bottle)')
    parser.add_argument('--output-dir', type=str, default='paper_figures/mvm',
                        help='Root output dir for batch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for batch')
    parser.add_argument('--random-rotation', action='store_true',
                        help='Apply random rotation per sample in batch')

    args = parser.parse_args()

    if args.web:
        run_web_server(args.port)
    elif args.batch:
        categories = args.categories.split(',') if args.categories else None
        samples = list_mvm_samples(args.gt_root, categories=categories)
        if not samples:
            raise RuntimeError("No valid MVM samples found for batch.")

        rng = np.random.default_rng(args.seed)
        count = min(args.num_samples, len(samples))
        selected = rng.choice(samples, size=count, replace=False)

        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        folder_name = time.strftime("batch_%Y%m%d_%H%M%S")
        batch_dir = output_root / folder_name
        suffix = 1
        while batch_dir.exists():
            batch_dir = output_root / f"{folder_name}_{suffix}"
            suffix += 1
        batch_dir.mkdir(parents=True, exist_ok=True)

        for item in selected:
            sample_rel = item['sample_rel']
            gt_path = item['gt_path']

            points = load_ply(sample_rel.as_posix())
            points = normalize_points(points)

            if args.random_rotation:
                rotation_rad = rng.uniform(0, 2 * np.pi)
            else:
                rotation_rad = np.deg2rad(args.rotation_deg)

            if abs(rotation_rad) > 1e-6:
                points = rotate_points_y(points, rotation_rad)

            mu, kappa, weight = read_mvm_gt(gt_path)
            mu = np.mod(mu - rotation_rad, 2 * np.pi)

            gt_angles = mu.tolist() if np.any(np.array(kappa) > 0.01) else None

            save_name = f"{sample_rel.as_posix().replace('/', '_').replace('.ply', '')}_mvm.png"
            save_path = batch_dir / save_name

            pointcloud_range = None if args.pointcloud_range < 0 else args.pointcloud_range

            plot_vonmises_polar(
                mus=mu,
                kappas=kappa,
                weights=weight,
                gt_angles=gt_angles,
                points=points,
                sample_name=sample_rel.as_posix(),
                save_path=save_path,
                show_components=args.show_components,
                pointcloud_range=pointcloud_range,
            )

        print(f"Batch saved: {batch_dir}")
    elif args.sample and args.output:
        points = load_ply(args.sample)
        points = normalize_points(points)

        rotation_rad = np.deg2rad(args.rotation_deg)
        if abs(rotation_rad) > 1e-6:
            points = rotate_points_y(points, rotation_rad)

        gt_path = Path(args.gt_file) if args.gt_file else resolve_gt_path(args.sample, args.gt_root)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        mu, kappa, weight = read_mvm_gt(gt_path)
        mu = np.mod(mu - rotation_rad, 2 * np.pi)

        gt_angles = mu.tolist() if np.any(np.array(kappa) > 0.01) else None

        pointcloud_range = None if args.pointcloud_range < 0 else args.pointcloud_range

        plot_vonmises_polar(
            mus=mu,
            kappas=kappa,
            weights=weight,
            gt_angles=gt_angles,
            points=points,
            sample_name=args.sample,
            save_path=args.output,
            show_components=args.show_components,
            pointcloud_range=pointcloud_range,
        )
    elif args.mus and args.output:
        mus_rad = np.deg2rad(args.mus)
        kappas = args.kappas if args.kappas else [10.0] * len(args.mus)
        weights = args.weights if args.weights else [1.0] * len(args.mus)

        if len(kappas) == 1:
            kappas = kappas * len(args.mus)
        if len(weights) == 1:
            weights = weights * len(args.mus)

        plot_vonmises_polar(
            mus=mus_rad,
            kappas=kappas,
            weights=weights,
            gt_angles=mus_rad.tolist(),
            save_path=args.output,
            show_components=args.show_components,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Web-based Ground Truth Verification Tool
验证旋转后的点云和von Mises ground truth是否对齐

Features:
- Support train/val/test split filtering
- Browse by category
- Interactive von Mises distribution plot

Usage:
    python tools/verify_gt_web.py --port 8060
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

# Data directory (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'symmetry_classification_gt'
CATEGORIES = ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']

# Load dataset_info
INFO_PATH = DATA_DIR / 'dataset_info.json'
if INFO_PATH.exists():
    with open(INFO_PATH, 'r') as f:
        DATASET_INFO = json.load(f)
else:
    DATASET_INFO = {}

# Split data (same logic as Fixed4PeakDataset)
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test
SEED = 42

def get_split_files():
    """Get files split into train/val/test using same logic as dataset"""
    rng = np.random.RandomState(SEED)
    splits = {'all': {}, 'train': {}, 'val': {}, 'test': {}}

    for cat in CATEGORIES:
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            for split in splits:
                splits[split][cat] = []
            continue

        files = [f for f in os.listdir(cat_dir) if f.endswith('.ply')]
        files.sort()

        # Shuffle with fixed seed
        files_shuffled = files.copy()
        rng.shuffle(files_shuffled)

        n = len(files_shuffled)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        splits['all'][cat] = files
        splits['train'][cat] = files_shuffled[:n_train]
        splits['val'][cat] = files_shuffled[n_train:n_train + n_val]
        splits['test'][cat] = files_shuffled[n_train + n_val:]

        # Sort for display
        for split in ['train', 'val', 'test']:
            splits[split][cat].sort()

    return splits

SPLITS = get_split_files()

def read_ply_points(filepath):
    """读取PLY点云"""
    points = []
    with open(filepath, 'r') as f:
        in_header = True
        vertex_count = 0
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
            else:
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(points) >= vertex_count:
                        break
    return np.array(points)

def read_gt_txt(filepath):
    """读取ground truth文件"""
    peaks = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                weight = float(parts[0])
                mu_cos = float(parts[1])
                mu_sin = float(parts[2])
                kappa = float(parts[3])
                mu_deg = np.rad2deg(np.arctan2(mu_sin, mu_cos)) % 360
                peaks.append({
                    'weight': weight,
                    'mu_cos': mu_cos,
                    'mu_sin': mu_sin,
                    'mu_deg': mu_deg,
                    'kappa': kappa
                })
    return peaks

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Ground Truth Verification</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { display: flex; gap: 20px; }
        .sidebar { width: 350px; }
        .main { flex: 1; }
        .panel { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        h2 { margin-top: 0; color: #e94560; }
        select, button { width: 100%; padding: 10px; margin: 5px 0; border: none; border-radius: 4px; }
        select { background: #0f3460; color: #fff; }
        button { background: #e94560; color: #fff; cursor: pointer; }
        button:hover { background: #ff6b6b; }
        .nav-buttons { display: flex; gap: 10px; }
        .nav-buttons button { flex: 1; }
        .info-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #0f3460; }
        .info-label { color: #888; }
        .info-value { color: #4ecca3; font-weight: bold; }
        .peak-info { background: #0f3460; padding: 10px; border-radius: 4px; margin: 5px 0; }
        .peak-active { border-left: 3px solid #e94560; }
        .peak-inactive { border-left: 3px solid #888; opacity: 0.6; }
        #progress { text-align: center; color: #888; margin-top: 10px; }
        .view-container { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
        .view-item { text-align: center; }
        .view-item img { width: 400px; height: 400px; object-fit: contain; background: #0f3460; border-radius: 4px; }
        #vonmises-container { width: 100%; height: 250px; }
        .plots-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .plot-box { background: #0f3460; border-radius: 8px; padding: 10px; }
        .plot-box h3 { text-align: center; color: #4ecca3; margin: 0 0 10px 0; font-size: 14px; }
        #polar-container { width: 100%; height: 350px; }
        .split-buttons { display: flex; gap: 5px; margin-bottom: 10px; }
        .split-btn { padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .split-btn.active { background: #e94560; color: white; }
        .split-btn:not(.active) { background: #0f3460; color: #888; }
        .split-btn:hover:not(.active) { background: #1a4a7a; }
        .stats { font-size: 12px; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Ground Truth Verification</h1>
    <div class="container">
        <div class="sidebar">
            <div class="panel">
                <h2>Dataset Split</h2>
                <div class="split-buttons">
                    <button class="split-btn active" data-split="all" onclick="setSplit('all')">All</button>
                    <button class="split-btn" data-split="train" onclick="setSplit('train')">Train</button>
                    <button class="split-btn" data-split="val" onclick="setSplit('val')">Val</button>
                    <button class="split-btn" data-split="test" onclick="setSplit('test')">Test</button>
                </div>
                <div id="split-stats" class="stats"></div>
            </div>

            <div class="panel">
                <h2>Category</h2>
                <select id="category" onchange="loadCategory()">
                    <option value="1_front">1 Front (1_front)</option>
                    <option value="2_fronts">2 Fronts (2_fronts)</option>
                    <option value="4_fronts">4 Fronts (4_fronts)</option>
                    <option value="symmetric">Symmetric</option>
                    <option value="no_front">No Front</option>
                </select>
                <select id="file" onchange="loadFile()"></select>
                <div class="nav-buttons">
                    <button onclick="prevFile()">← Prev</button>
                    <button onclick="nextFile()">Next →</button>
                    <button onclick="randomFile()">Random</button>
                </div>
                <div id="progress"></div>
            </div>

            <div class="panel">
                <h2>Rotation Info</h2>
                <div id="rotation-info"></div>
            </div>

            <div class="panel">
                <h2>Ground Truth Peaks</h2>
                <div id="peaks-info"></div>
            </div>
        </div>

        <div class="main">
            <div class="panel">
                <h2>Visualizations</h2>
                <div class="plots-row">
                    <div class="plot-box">
                        <h3>Top View (XZ Plane)</h3>
                        <div class="view-container">
                            <img id="topview" src="">
                        </div>
                    </div>
                    <div class="plot-box">
                        <h3>Polar Distribution (same as Model Vis)</h3>
                        <div id="polar-container"></div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h2>Von Mises PDF (Linear)</h2>
                <div id="vonmises-container"></div>
            </div>
        </div>
    </div>

    <script>
        let currentSplit = 'all';
        let currentCategory = '1_front';
        let currentFiles = [];
        let currentIndex = 0;
        let splitData = {};

        async function init() {
            const resp = await fetch('/api/splits');
            splitData = await resp.json();
            updateSplitStats();
            loadCategory();
        }

        function setSplit(split) {
            currentSplit = split;
            document.querySelectorAll('.split-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.split === split);
            });
            updateSplitStats();
            loadCategory();
        }

        function updateSplitStats() {
            const stats = splitData[currentSplit] || {};
            let total = 0;
            let html = '';
            for (const cat of ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']) {
                const count = (stats[cat] || []).length;
                total += count;
                html += `${cat}: ${count} | `;
            }
            html += `Total: ${total}`;
            document.getElementById('split-stats').innerHTML = html;
        }

        function loadCategory() {
            currentCategory = document.getElementById('category').value;
            currentFiles = (splitData[currentSplit] || {})[currentCategory] || [];

            const select = document.getElementById('file');
            select.innerHTML = currentFiles.map((f, i) =>
                `<option value="${f}">${i+1}. ${f}</option>`
            ).join('');
            currentIndex = 0;
            if (currentFiles.length > 0) loadFile();
            else {
                document.getElementById('progress').textContent = '0 / 0';
                document.getElementById('topview').src = '';
            }
        }

        function loadFile() {
            const file = document.getElementById('file').value;
            if (!file) return;

            currentIndex = currentFiles.indexOf(file);
            document.getElementById('progress').textContent =
                `${currentIndex + 1} / ${currentFiles.length}`;

            fetch(`/api/view/${currentCategory}/${file}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        console.error(data.error);
                        return;
                    }

                    // Show image
                    document.getElementById('topview').src = 'data:image/png;base64,' + data.image;

                    // Rotation info
                    let rotHtml = `
                        <div class="info-row">
                            <span class="info-label">Original Front:</span>
                            <span class="info-value">${data.info.original_front_direction || 'N/A'}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Rotation Applied:</span>
                            <span class="info-value">${data.info.rotation_applied_deg ? data.info.rotation_applied_deg.toFixed(1) + '°' : 'N/A'}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">New Front Angle:</span>
                            <span class="info-value">${data.info.new_front_angle_deg ? data.info.new_front_angle_deg.toFixed(1) + '°' : 'N/A'}</span>
                        </div>
                    `;
                    document.getElementById('rotation-info').innerHTML = rotHtml;

                    // Peaks info
                    let peaksHtml = '';
                    data.peaks.forEach((p, i) => {
                        const isActive = p.kappa > 0;
                        peaksHtml += `
                            <div class="peak-info ${isActive ? 'peak-active' : 'peak-inactive'}">
                                <strong>Peak ${i+1}</strong><br>
                                μ = ${p.mu_deg.toFixed(1)}°<br>
                                κ = ${p.kappa.toFixed(0)} ${isActive ? '(ACTIVE)' : ''}
                            </div>
                        `;
                    });
                    document.getElementById('peaks-info').innerHTML = peaksHtml;

                    // Plot von Mises (linear)
                    plotVonMises(data.peaks);

                    // Plot polar distribution (same as model visualizer)
                    plotPolar(data.peaks);
                });
        }

        function plotVonMises(peaks) {
            const theta = [];
            for (let i = 0; i <= 360; i++) theta.push(i);

            function vonMisesPDF(t, mu, kappa) {
                if (kappa === 0) return 1 / (2 * Math.PI);
                const tRad = t * Math.PI / 180;
                const muRad = mu * Math.PI / 180;
                const I0 = (k) => {
                    let sum = 1, term = 1;
                    for (let n = 1; n < 50; n++) {
                        term *= (k * k) / (4 * n * n);
                        sum += term;
                    }
                    return sum;
                };
                return Math.exp(kappa * Math.cos(tRad - muRad)) / (2 * Math.PI * I0(kappa));
            }

            const mixture = theta.map(t => {
                let sum = 0;
                peaks.forEach(p => {
                    sum += 0.25 * vonMisesPDF(t, p.mu_deg, p.kappa);
                });
                return sum;
            });

            const traces = [{
                x: theta,
                y: mixture,
                type: 'scatter',
                fill: 'tozeroy',
                fillcolor: 'rgba(78, 204, 163, 0.3)',
                line: { color: '#4ecca3', width: 2 },
                name: 'Mixture'
            }];

            // Add individual peaks
            peaks.forEach((p, i) => {
                if (p.kappa > 0) {
                    traces.push({
                        x: [p.mu_deg, p.mu_deg],
                        y: [0, Math.max(...mixture) * 1.1],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#e94560', width: 2, dash: 'dash' },
                        name: `Peak ${i+1}: ${p.mu_deg.toFixed(0)}°`
                    });
                }
            });

            Plotly.newPlot('vonmises-container', traces, {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'Angle (°)', color: '#888', gridcolor: '#333', range: [0, 360] },
                yaxis: { title: 'PDF', color: '#888', gridcolor: '#333' },
                margin: { t: 20, b: 50, l: 60, r: 20 },
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right', font: { color: '#888' } }
            });
        }

        function plotPolar(peaks) {
            // Compute von Mises PDF for polar plot (same as model visualizer)
            const theta = [];
            for (let i = 0; i <= 360; i++) theta.push(i);

            function vonMisesPDF(t, mu, kappa) {
                if (kappa === 0) return 1 / (2 * Math.PI);
                const tRad = t * Math.PI / 180;
                const muRad = mu * Math.PI / 180;
                const I0 = (k) => {
                    let sum = 1, term = 1;
                    for (let n = 1; n < 50; n++) {
                        term *= (k * k) / (4 * n * n);
                        sum += term;
                    }
                    return sum;
                };
                return Math.exp(kappa * Math.cos(tRad - muRad)) / (2 * Math.PI * I0(kappa));
            }

            const mixture = theta.map(t => {
                let sum = 0;
                peaks.forEach(p => {
                    sum += 0.25 * vonMisesPDF(t, p.mu_deg, p.kappa);
                });
                return sum;
            });

            // Normalize
            const maxVal = Math.max(...mixture);
            const normalized = maxVal > 0 ? mixture.map(v => v / maxVal) : mixture;

            // Create polar plot (same style as model visualizer)
            const polarData = [{
                type: 'scatterpolar',
                r: normalized,
                theta: theta,
                name: 'GT Distribution',
                line: { color: '#4ecca3', width: 3 },
                fill: 'toself',
                fillcolor: 'rgba(78, 204, 163, 0.3)'
            }];

            // Add peak markers
            peaks.forEach((p, i) => {
                if (p.kappa > 0) {
                    polarData.push({
                        type: 'scatterpolar',
                        r: [0, 1.05],
                        theta: [p.mu_deg, p.mu_deg],
                        mode: 'lines+markers',
                        name: `Peak ${i+1}: ${p.mu_deg.toFixed(0)}°`,
                        line: { color: '#e94560', width: 2 },
                        marker: { size: 10, symbol: 'triangle-up' }
                    });
                }
            });

            const polarLayout = {
                polar: {
                    radialaxis: { visible: true, range: [0, 1.15] },
                    angularaxis: { direction: 'clockwise' },
                    bgcolor: 'rgba(0,0,0,0)'
                },
                showlegend: true,
                legend: { x: 0.5, y: -0.15, xanchor: 'center', orientation: 'h', font: { color: '#888', size: 10 } },
                margin: { t: 30, b: 60, l: 30, r: 30 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#eee' }
            };

            Plotly.newPlot('polar-container', polarData, polarLayout);
        }

        function prevFile() {
            if (currentIndex > 0) {
                currentIndex--;
                document.getElementById('file').selectedIndex = currentIndex;
                loadFile();
            }
        }

        function nextFile() {
            if (currentIndex < currentFiles.length - 1) {
                currentIndex++;
                document.getElementById('file').selectedIndex = currentIndex;
                loadFile();
            }
        }

        function randomFile() {
            if (currentFiles.length > 0) {
                currentIndex = Math.floor(Math.random() * currentFiles.length);
                document.getElementById('file').selectedIndex = currentIndex;
                loadFile();
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') prevFile();
            if (e.key === 'ArrowRight') nextFile();
        });

        init();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/splits')
def get_splits():
    return jsonify(SPLITS)

@app.route('/api/files/<category>')
def get_files(category):
    return jsonify({'files': SPLITS['all'].get(category, [])})

@app.route('/api/view/<category>/<filename>')
def get_view(category, filename):
    """生成点云俯视图，并叠加GT方向箭头"""
    ply_path = DATA_DIR / category / filename
    gt_path = DATA_DIR / category / filename.replace('.ply', '_gt.txt')

    if not ply_path.exists():
        return jsonify({'error': f'PLY not found: {ply_path}'}), 404
    if not gt_path.exists():
        return jsonify({'error': f'GT not found: {gt_path}'}), 404

    try:
        points = read_ply_points(str(ply_path))
        peaks = read_gt_txt(str(gt_path))
        info = DATASET_INFO.get(filename, {})

        # 创建俯视图
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0f3460')
        ax.set_facecolor('#0f3460')

        # 绘制点云 (XZ平面)
        x_data = points[:, 0]  # X
        z_data = points[:, 2]  # Z
        ax.scatter(x_data, z_data, s=1, c='#4ecca3', alpha=0.6)

        # 计算点云范围
        max_range = max(np.abs(x_data).max(), np.abs(z_data).max()) * 1.2
        if max_range < 0.5:
            max_range = 0.5

        # 绘制GT方向箭头
        arrow_len = max_range * 0.6
        for i, p in enumerate(peaks):
            if p['kappa'] > 0:
                # 从原点画箭头
                dx = arrow_len * p['mu_cos']
                dz = arrow_len * p['mu_sin']
                ax.annotate('', xy=(dx, dz), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='#e94560', lw=3))
                # 标注角度
                label_dist = arrow_len * 1.15
                ax.text(label_dist * p['mu_cos'], label_dist * p['mu_sin'],
                       f"{p['mu_deg']:.0f}°",
                       color='#e94560', fontsize=14, ha='center', va='center',
                       fontweight='bold')

        # 绘制坐标轴标注
        ax.annotate('+X', xy=(max_range * 0.95, 0), color='#888', fontsize=14, ha='left', va='center')
        ax.annotate('-X', xy=(-max_range * 0.95, 0), color='#888', fontsize=14, ha='right', va='center')
        ax.annotate('+Z', xy=(0, max_range * 0.95), color='#888', fontsize=14, ha='center', va='bottom')
        ax.annotate('-Z', xy=(0, -max_range * 0.95), color='#888', fontsize=14, ha='center', va='top')

        ax.set_aspect('equal')
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_xlabel('X', color='#eee', fontsize=12)
        ax.set_ylabel('Z', color='#eee', fontsize=12)
        ax.set_title(f'{filename}\nOriginal: {info.get("original_front_direction", "?")} | Rotated: {info.get("rotation_applied_deg", 0):.1f}°',
                    color='#e94560', fontsize=12)
        ax.tick_params(colors='#888')
        ax.axhline(y=0, color='#333', linewidth=0.5)
        ax.axvline(x=0, color='#333', linewidth=0.5)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0f3460', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            'image': img_base64,
            'peaks': peaks,
            'info': info
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8060)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Ground Truth Verification Tool")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print("\nDataset statistics:")
    for split in ['all', 'train', 'val', 'test']:
        total = sum(len(SPLITS[split].get(cat, [])) for cat in CATEGORIES)
        print(f"  {split:5s}: {total:4d} samples")
    print("=" * 60)

    print(f"\nStarting server on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

#!/usr/bin/env python
"""
Web Tool for Visualizing Fixed 4-Peak Model Checkpoints

Features:
- Select any .pth checkpoint file
- Browse test/val samples by category
- Interactive polar plots (Plotly)
- Side-by-side GT vs Prediction comparison
- Detailed metrics per sample

Usage:
    python tools/vis_checkpoint_web.py
    python tools/vis_checkpoint_web.py --port 8070

Author: Claude
"""

import os
import sys
import glob
import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template_string, jsonify, request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_fixed_4peak import Fixed4PeakModel
from datasets.fixed_4peak_dataset import Fixed4PeakDataset

app = Flask(__name__)

# Global state
STATE = {
    'model': None,
    'checkpoint_path': None,
    'checkpoint_info': None,
    'dataset': None,
    'device': None,
}

CATEGORIES = ['1_front', '2_fronts', '4_fronts', 'symmetric', 'no_front']


def find_checkpoints(base_dir='checkpoints'):
    """Find all .pth files in checkpoint directories."""
    checkpoints = []

    # Search for .pth files
    for pth_file in glob.glob(f'{base_dir}/**/*.pth', recursive=True):
        rel_path = os.path.relpath(pth_file, base_dir)
        size_mb = os.path.getsize(pth_file) / (1024 * 1024)
        mtime = os.path.getmtime(pth_file)

        checkpoints.append({
            'path': pth_file,
            'name': rel_path,
            'size_mb': round(size_mb, 1),
            'mtime': mtime
        })

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    return checkpoints


def load_checkpoint(checkpoint_path: str):
    """Load a checkpoint and update global state."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    STATE['device'] = device

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    encoder_dim = checkpoint.get('args', {}).get('encoder_dim', 1024)
    model = Fixed4PeakModel(encoder_dim=encoder_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    STATE['model'] = model
    STATE['checkpoint_path'] = checkpoint_path
    STATE['checkpoint_info'] = {
        'epoch': checkpoint.get('epoch', -1) + 1,
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'args': checkpoint.get('args', {})
    }

    return STATE['checkpoint_info']


def load_dataset(split='val'):
    """Load dataset."""
    dataset = Fixed4PeakDataset(
        mode='random',
        split=split,
        num_points=2048,
        augment=False
    )
    STATE['dataset'] = dataset
    return dataset


def compute_von_mises_pdf(theta, mu_angle, kappa):
    """Compute von Mises PDF."""
    if kappa < 0.01:
        return np.ones_like(theta) / (2 * np.pi)

    from scipy.special import i0
    pdf = np.exp(kappa * np.cos(theta - mu_angle)) / (2 * np.pi * i0(kappa))
    return pdf


def generate_pointcloud_image(sample_idx: int, pred_angles=None, pred_kappa=None,
                               gt_angles=None, gt_kappa=None):
    """Generate top-view point cloud image with direction arrows."""
    if STATE['dataset'] is None:
        return None

    sample = STATE['dataset'][sample_idx]
    points = sample['points'].numpy()  # (N, 3)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f3460')
    ax.set_facecolor('#0f3460')

    # Plot point cloud (XZ plane, top view)
    x_data = points[:, 0]
    z_data = points[:, 2]
    ax.scatter(x_data, z_data, s=1, c='#4ecca3', alpha=0.6)

    # Calculate range
    max_range = max(np.abs(x_data).max(), np.abs(z_data).max()) * 1.2
    if max_range < 0.5:
        max_range = 0.5

    arrow_len = max_range * 0.6

    # Draw GT direction arrows (green)
    if gt_angles is not None and gt_kappa is not None:
        for i, (angle, kappa) in enumerate(zip(gt_angles, gt_kappa)):
            if kappa > 0.5:
                rad = np.deg2rad(angle)
                dx = arrow_len * np.cos(rad)
                dz = arrow_len * np.sin(rad)
                ax.annotate('', xy=(dx, dz), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='#4ecca3', lw=3))
                # Label
                label_dist = arrow_len * 1.15
                ax.text(label_dist * np.cos(rad), label_dist * np.sin(rad),
                       f"GT:{angle:.0f}°", color='#4ecca3', fontsize=10,
                       ha='center', va='center', fontweight='bold')

    # Draw Prediction direction arrows (red/pink)
    if pred_angles is not None and pred_kappa is not None:
        for i, (angle, kappa) in enumerate(zip(pred_angles, pred_kappa)):
            if kappa > 5:  # Significant prediction
                rad = np.deg2rad(angle)
                dx = arrow_len * 0.8 * np.cos(rad)
                dz = arrow_len * 0.8 * np.sin(rad)
                ax.annotate('', xy=(dx, dz), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='#e94560', lw=2, ls='--'))

    # Axis labels
    ax.annotate('+X', xy=(max_range * 0.95, 0), color='#888', fontsize=12, ha='left', va='center')
    ax.annotate('-X', xy=(-max_range * 0.95, 0), color='#888', fontsize=12, ha='right', va='center')
    ax.annotate('+Z', xy=(0, max_range * 0.95), color='#888', fontsize=12, ha='center', va='bottom')
    ax.annotate('-Z', xy=(0, -max_range * 0.95), color='#888', fontsize=12, ha='center', va='top')

    ax.set_aspect('equal')
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_xlabel('X', color='#eee', fontsize=12)
    ax.set_ylabel('Z', color='#eee', fontsize=12)
    ax.tick_params(colors='#888')
    ax.axhline(y=0, color='#333', linewidth=0.5)
    ax.axvline(x=0, color='#333', linewidth=0.5)

    # Title with legend
    ax.set_title('Top View (XZ Plane)\nGreen=GT, Red=Pred', color='#e94560', fontsize=12)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f3460', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return img_base64


def get_prediction(sample_idx: int):
    """Get prediction for a sample."""
    if STATE['model'] is None or STATE['dataset'] is None:
        return None

    model = STATE['model']
    dataset = STATE['dataset']
    device = STATE['device']

    sample = dataset[sample_idx]
    points = sample['points'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mu, pred_kappa = model(points)

    pred_mu = pred_mu[0].cpu().numpy()
    pred_kappa = pred_kappa[0].cpu().numpy()
    gt_mu = sample['mu'].numpy()
    gt_kappa = sample['kappa'].numpy()

    # Convert to angles (degrees)
    pred_angles = np.rad2deg(np.arctan2(pred_mu[:, 1], pred_mu[:, 0]))
    gt_angles = np.rad2deg(np.arctan2(gt_mu[:, 1], gt_mu[:, 0]))

    # Compute PDFs for plotting
    theta = np.linspace(0, 2*np.pi, 360)
    pred_angles_rad = np.deg2rad(pred_angles)
    gt_angles_rad = np.deg2rad(gt_angles)

    pred_pdf = np.zeros_like(theta)
    gt_pdf = np.zeros_like(theta)

    for i in range(4):
        pred_pdf += 0.25 * compute_von_mises_pdf(theta, pred_angles_rad[i], pred_kappa[i])
        gt_pdf += 0.25 * compute_von_mises_pdf(theta, gt_angles_rad[i], gt_kappa[i])

    # Normalize for display
    max_val = max(pred_pdf.max(), gt_pdf.max())
    if max_val > 0:
        pred_pdf = pred_pdf / max_val
        gt_pdf = gt_pdf / max_val

    # Generate point cloud image with arrows
    pointcloud_image = generate_pointcloud_image(
        sample_idx,
        pred_angles=pred_angles,
        pred_kappa=pred_kappa,
        gt_angles=gt_angles,
        gt_kappa=gt_kappa
    )

    return {
        'filename': sample['filename'],
        'category': sample['category_name'],
        'pred_angles': pred_angles.tolist(),
        'pred_kappa': pred_kappa.tolist(),
        'gt_angles': gt_angles.tolist(),
        'gt_kappa': gt_kappa.tolist(),
        'theta_deg': np.rad2deg(theta).tolist(),
        'pred_pdf': pred_pdf.tolist(),
        'gt_pdf': gt_pdf.tolist(),
        'pointcloud_image': pointcloud_image
    }


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fixed 4-Peak Model Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            padding: 20px;
            border-bottom: 2px solid #e94560;
        }
        .header h1 { font-size: 24px; margin-bottom: 10px; }
        .container { display: flex; height: calc(100vh - 80px); }

        .sidebar {
            width: 320px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #333;
        }
        .main { flex: 1; padding: 20px; overflow-y: auto; }

        .section { margin-bottom: 20px; }
        .section-title {
            font-size: 14px;
            color: #e94560;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        select, button {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #333;
            border-radius: 5px;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }
        select:focus, button:focus { outline: none; border-color: #e94560; }
        button {
            background: #e94560;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #ff6b6b; }
        button:disabled { background: #555; cursor: not-allowed; }

        .checkpoint-info {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            margin-bottom: 15px;
        }
        .checkpoint-info div { margin: 5px 0; }
        .checkpoint-info .label { color: #888; }
        .checkpoint-info .value { color: #4ecca3; font-weight: bold; }

        .sample-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #333;
            border-radius: 5px;
        }
        .sample-item {
            padding: 8px 10px;
            cursor: pointer;
            border-bottom: 1px solid #333;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        .sample-item:hover { background: #0f3460; }
        .sample-item.active { background: #e94560; }
        .sample-item .idx { color: #888; }

        .plot-container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .plot-box {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            min-height: 400px;
        }
        .plot-box h3 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .pointcloud-container {
            width: 100%;
            height: 350px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .pointcloud-img {
            max-width: 100%;
            max-height: 350px;
            border-radius: 5px;
            object-fit: contain;
        }
        #polar-plot, #peak-plot {
            width: 100%;
            height: 350px;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .metrics-table th, .metrics-table td {
            padding: 8px 12px;
            text-align: center;
            border: 1px solid #333;
        }
        .metrics-table th { background: #0f3460; color: #4ecca3; }
        .metrics-table td { background: #1a1a2e; }
        .metrics-table .gt { color: #4ecca3; }
        .metrics-table .pred { color: #e94560; }

        .category-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }
        .cat-1_front { background: #e94560; }
        .cat-2_fronts { background: #f39c12; }
        .cat-4_fronts { background: #3498db; }
        .cat-symmetric { background: #9b59b6; }
        .cat-no_front { background: #1abc9c; }

        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .nav-buttons button { flex: 1; }

        #loading {
            display: none;
            text-align: center;
            padding: 50px;
            color: #888;
        }

        .stats-summary {
            background: #0f3460;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Fixed 4-Peak von Mises Model Visualizer</h1>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="section">
                <div class="section-title">Checkpoint</div>
                <select id="checkpoint-select">
                    <option value="">-- Select checkpoint --</option>
                </select>
                <button id="load-btn" onclick="loadCheckpoint()">Load Checkpoint</button>

                <div class="checkpoint-info" id="checkpoint-info" style="display:none;">
                    <div><span class="label">Epoch:</span> <span class="value" id="info-epoch">-</span></div>
                    <div><span class="label">Val Loss:</span> <span class="value" id="info-loss">-</span></div>
                    <div><span class="label">Device:</span> <span class="value" id="info-device">-</span></div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Dataset</div>
                <select id="split-select">
                    <option value="val">Validation</option>
                    <option value="test">Test</option>
                    <option value="train">Train</option>
                </select>
                <select id="category-filter">
                    <option value="all">All Categories</option>
                    <option value="1_front">1_front</option>
                    <option value="2_fronts">2_fronts</option>
                    <option value="4_fronts">4_fronts</option>
                    <option value="symmetric">symmetric</option>
                    <option value="no_front">no_front</option>
                </select>
            </div>

            <div class="section">
                <div class="section-title">Samples</div>
                <div class="nav-buttons">
                    <button onclick="prevSample()">Prev</button>
                    <button onclick="nextSample()">Next</button>
                    <button onclick="randomSample()">Random</button>
                </div>
                <div class="sample-list" id="sample-list"></div>
            </div>
        </div>

        <div class="main">
            <div id="loading">Loading...</div>

            <div id="content" style="display:none;">
                <div class="stats-summary" id="sample-header">
                    <div class="stats-row">
                        <span>File: <strong id="current-file">-</strong></span>
                        <span>Category: <span class="category-badge" id="current-category">-</span></span>
                    </div>
                </div>

                <div class="plot-container">
                    <div class="plot-box">
                        <h3>Point Cloud (Top View)</h3>
                        <div class="pointcloud-container">
                            <img id="pointcloud-img" class="pointcloud-img" src="" alt="Point cloud top view">
                        </div>
                    </div>
                    <div class="plot-box">
                        <h3>Distribution Comparison</h3>
                        <div id="polar-plot"></div>
                    </div>
                    <div class="plot-box">
                        <h3>Peak Positions</h3>
                        <div id="peak-plot"></div>
                    </div>
                </div>

                <div class="plot-box">
                    <h3>Detailed Parameters</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Peak</th>
                                <th class="gt">GT Angle</th>
                                <th class="pred">Pred Angle</th>
                                <th class="gt">GT kappa</th>
                                <th class="pred">Pred kappa</th>
                                <th>Angle Error</th>
                            </tr>
                        </thead>
                        <tbody id="metrics-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let checkpoints = [];
        let samples = [];
        let filteredSamples = [];
        let currentIdx = 0;

        // Load checkpoints on page load
        async function init() {
            const resp = await fetch('/api/checkpoints');
            checkpoints = await resp.json();

            const select = document.getElementById('checkpoint-select');
            checkpoints.forEach(ckpt => {
                const opt = document.createElement('option');
                opt.value = ckpt.path;
                opt.textContent = `${ckpt.name} (${ckpt.size_mb}MB)`;
                select.appendChild(opt);
            });
        }

        async function loadCheckpoint() {
            const path = document.getElementById('checkpoint-select').value;
            if (!path) return;

            document.getElementById('load-btn').disabled = true;
            document.getElementById('load-btn').textContent = 'Loading...';

            try {
                const resp = await fetch('/api/load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        checkpoint: path,
                        split: document.getElementById('split-select').value
                    })
                });
                const data = await resp.json();

                if (data.error) {
                    alert(data.error);
                    return;
                }

                // Update info
                document.getElementById('checkpoint-info').style.display = 'block';
                document.getElementById('info-epoch').textContent = data.epoch;
                document.getElementById('info-loss').textContent =
                    data.best_val_loss ? data.best_val_loss.toFixed(4) : '-';
                document.getElementById('info-device').textContent = data.device;

                // Load samples
                samples = data.samples;
                filterSamples();

            } finally {
                document.getElementById('load-btn').disabled = false;
                document.getElementById('load-btn').textContent = 'Load Checkpoint';
            }
        }

        function filterSamples() {
            const cat = document.getElementById('category-filter').value;
            if (cat === 'all') {
                filteredSamples = samples;
            } else {
                filteredSamples = samples.filter(s => s.category === cat);
            }

            // Update sample list
            const list = document.getElementById('sample-list');
            list.innerHTML = '';
            filteredSamples.forEach((s, i) => {
                const div = document.createElement('div');
                div.className = 'sample-item' + (i === currentIdx ? ' active' : '');
                div.innerHTML = `
                    <span class="idx">#${i}</span>
                    <span>${s.filename.substring(0, 25)}...</span>
                    <span class="category-badge cat-${s.category}">${s.category}</span>
                `;
                div.onclick = () => selectSample(i);
                list.appendChild(div);
            });

            if (filteredSamples.length > 0) {
                selectSample(0);
            }
        }

        async function selectSample(idx) {
            if (idx < 0 || idx >= filteredSamples.length) return;
            currentIdx = idx;

            // Update list highlighting
            document.querySelectorAll('.sample-item').forEach((el, i) => {
                el.className = 'sample-item' + (i === idx ? ' active' : '');
            });

            document.getElementById('loading').style.display = 'block';
            document.getElementById('content').style.display = 'none';

            const sample = filteredSamples[idx];
            const resp = await fetch(`/api/predict/${sample.idx}`);
            const data = await resp.json();

            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').style.display = 'block';

            updateDisplay(data);
        }

        function updateDisplay(data) {
            // Update header
            document.getElementById('current-file').textContent = data.filename;
            const catBadge = document.getElementById('current-category');
            catBadge.textContent = data.category;
            catBadge.className = `category-badge cat-${data.category}`;

            // Point cloud image
            if (data.pointcloud_image) {
                document.getElementById('pointcloud-img').src =
                    'data:image/png;base64,' + data.pointcloud_image;
            }

            // Polar plot
            const polarData = [
                {
                    type: 'scatterpolar',
                    r: data.gt_pdf,
                    theta: data.theta_deg,
                    name: 'GT',
                    line: {color: '#4ecca3', width: 3},
                    fill: 'toself',
                    fillcolor: 'rgba(78, 204, 163, 0.2)'
                },
                {
                    type: 'scatterpolar',
                    r: data.pred_pdf,
                    theta: data.theta_deg,
                    name: 'Prediction',
                    line: {color: '#e94560', width: 3, dash: 'dash'},
                    fill: 'toself',
                    fillcolor: 'rgba(233, 69, 96, 0.2)'
                }
            ];

            const polarLayout = {
                polar: {
                    radialaxis: {visible: true, range: [0, 1.1]},
                    angularaxis: {direction: 'clockwise'}
                },
                showlegend: true,
                legend: {x: 0.5, y: -0.1, xanchor: 'center', orientation: 'h'},
                margin: {t: 30, b: 50, l: 30, r: 30},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#eee'},
                autosize: true
            };

            Plotly.react('polar-plot', polarData, polarLayout, {responsive: true});

            // Peak positions plot
            const peakData = [];

            // GT peaks
            for (let i = 0; i < 4; i++) {
                if (data.gt_kappa[i] > 0.5) {
                    peakData.push({
                        type: 'scatterpolar',
                        r: [0, 1],
                        theta: [data.gt_angles[i], data.gt_angles[i]],
                        mode: 'lines+markers',
                        name: `GT ${i+1}`,
                        line: {color: '#4ecca3', width: 2},
                        marker: {size: 10, symbol: 'triangle-up'},
                        showlegend: i === 0
                    });
                }
            }

            // Pred peaks
            for (let i = 0; i < 4; i++) {
                if (data.pred_kappa[i] > 5) {
                    peakData.push({
                        type: 'scatterpolar',
                        r: [0, 0.8],
                        theta: [data.pred_angles[i], data.pred_angles[i]],
                        mode: 'lines+markers',
                        name: `Pred ${i+1}`,
                        line: {color: '#e94560', width: 2, dash: 'dash'},
                        marker: {size: 8, symbol: 'triangle-down'},
                        showlegend: i === 0
                    });
                }
            }

            const peakLayout = {
                polar: {
                    radialaxis: {visible: false, range: [0, 1.2]},
                    angularaxis: {direction: 'clockwise'}
                },
                showlegend: true,
                legend: {x: 0.5, y: -0.1, xanchor: 'center', orientation: 'h'},
                margin: {t: 30, b: 50, l: 30, r: 30},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#eee'},
                autosize: true
            };

            Plotly.react('peak-plot', peakData, peakLayout, {responsive: true});

            // Metrics table
            const tbody = document.getElementById('metrics-body');
            tbody.innerHTML = '';

            for (let i = 0; i < 4; i++) {
                const angleError = Math.abs(
                    ((data.pred_angles[i] - data.gt_angles[i] + 180) % 360) - 180
                );

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td class="gt">${data.gt_angles[i].toFixed(1)}deg</td>
                    <td class="pred">${data.pred_angles[i].toFixed(1)}deg</td>
                    <td class="gt">${data.gt_kappa[i].toFixed(1)}</td>
                    <td class="pred">${data.pred_kappa[i].toFixed(1)}</td>
                    <td>${angleError.toFixed(1)}deg</td>
                `;
                tbody.appendChild(tr);
            }
        }

        function prevSample() { selectSample(currentIdx - 1); }
        function nextSample() { selectSample(currentIdx + 1); }
        function randomSample() {
            const idx = Math.floor(Math.random() * filteredSamples.length);
            selectSample(idx);
        }

        // Event listeners
        document.getElementById('category-filter').onchange = filterSamples;
        document.getElementById('split-select').onchange = loadCheckpoint;

        // Initialize
        init();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/checkpoints')
def api_checkpoints():
    return jsonify(find_checkpoints())


@app.route('/api/load', methods=['POST'])
def api_load():
    data = request.json
    checkpoint_path = data.get('checkpoint')
    split = data.get('split', 'val')

    try:
        info = load_checkpoint(checkpoint_path)
        dataset = load_dataset(split)

        # Get sample list with categories
        samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            samples.append({
                'idx': i,
                'filename': sample['filename'],
                'category': sample['category_name']
            })

        return jsonify({
            'epoch': info['epoch'],
            'best_val_loss': info['best_val_loss'],
            'device': str(STATE['device']),
            'samples': samples
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/predict/<int:idx>')
def api_predict(idx):
    result = get_prediction(idx)
    if result is None:
        return jsonify({'error': 'Model not loaded'})
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8070)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Fixed 4-Peak Model Visualizer")
    print(f"{'='*60}")
    print(f"Open in browser: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

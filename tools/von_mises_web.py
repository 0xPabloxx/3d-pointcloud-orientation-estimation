#!/usr/bin/env python
"""
Web-based Interactive von Mises Mixture Distribution Visualizer

Features:
- Adjust number of peaks (K)
- Slider controls for mu, kappa, and weights
- Real-time Plotly polar and Cartesian plots
- Preset configurations for common distributions

Usage:
    python tools/von_mises_web.py
    python tools/von_mises_web.py --port 8080

Author: Claude
"""

import argparse
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from scipy.stats import vonmises

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>von Mises Mixture Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        .header p { color: #888; margin-top: 5px; }

        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }

        .sidebar {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
        }

        .main {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .plot-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .plot-box {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
        }

        .plot-box h3 {
            text-align: center;
            margin-bottom: 10px;
            color: #4ecca3;
        }

        #polar-plot, #cartesian-plot {
            width: 100%;
            height: 450px;
        }

        .section {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        .section:last-child { border-bottom: none; }

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
            transition: background 0.2s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            background: #ff6b6b;
        }

        select, button {
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 10px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
        }

        select:focus, button:focus { outline: none; border-color: #e94560; }

        button {
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            border: none;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
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

        .peak-title {
            font-weight: bold;
            color: #4ecca3;
        }

        .peak-active {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
            color: #888;
        }

        .peak-active input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .info-box {
            background: rgba(78, 204, 163, 0.1);
            border: 1px solid #4ecca3;
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
        }

        .info-box .row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }

        .info-box .label { color: #888; }
        .info-box .value { color: #4ecca3; font-family: monospace; }

        .formula {
            font-family: 'Times New Roman', serif;
            font-style: italic;
            text-align: center;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>von Mises Mixture Distribution Visualizer</h1>
        <p>Interactive visualization of mixture of von Mises distributions</p>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="section">
                <div class="section-title">Presets</div>
                <button class="preset-btn" onclick="loadPreset('uniform')">Uniform (K=0)</button>
                <button class="preset-btn" onclick="loadPreset('single')">Single Peak</button>
                <button class="preset-btn" onclick="loadPreset('bimodal')">Bimodal (180deg)</button>
                <button class="preset-btn" onclick="loadPreset('quad')">4 Peaks (90deg)</button>
                <button class="preset-btn" onclick="loadPreset('asymmetric')">Asymmetric</button>
            </div>

            <div class="section">
                <div class="section-title">Global Settings</div>
                <div class="control-row">
                    <div class="control-label">
                        <span>Global Kappa</span>
                        <span class="value" id="global-kappa-val">10.0</span>
                    </div>
                    <input type="range" id="global-kappa" min="0.1" max="100" step="0.1" value="10"
                           oninput="applyGlobalKappa(this.value)">
                </div>
            </div>

            <div class="section">
                <div class="section-title">Peak Controls</div>
                <div id="peak-controls"></div>
            </div>

            <div class="section">
                <div class="section-title">Distribution Info</div>
                <div class="info-box">
                    <div class="row">
                        <span class="label">Active Peaks:</span>
                        <span class="value" id="info-peaks">4</span>
                    </div>
                    <div class="row">
                        <span class="label">Entropy (nats):</span>
                        <span class="value" id="info-entropy">-</span>
                    </div>
                    <div class="row">
                        <span class="label">Max Density:</span>
                        <span class="value" id="info-max">-</span>
                    </div>
                </div>
                <div class="formula">
                    p(θ) = Σ wᵢ · vM(θ; μᵢ, κᵢ)
                </div>
            </div>
        </div>

        <div class="main">
            <div class="plot-container">
                <div class="plot-box">
                    <h3>Polar View</h3>
                    <div id="polar-plot"></div>
                </div>
                <div class="plot-box">
                    <h3>Cartesian View</h3>
                    <div id="cartesian-plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const MAX_PEAKS = 6;
        let peaks = [
            {active: true, mu: 0, kappa: 10, weight: 1},
            {active: true, mu: 90, kappa: 10, weight: 1},
            {active: true, mu: 180, kappa: 10, weight: 1},
            {active: true, mu: 270, kappa: 10, weight: 1},
            {active: false, mu: 45, kappa: 10, weight: 1},
            {active: false, mu: 135, kappa: 10, weight: 1},
        ];

        const colors = ['#e94560', '#4ecca3', '#f39c12', '#3498db', '#9b59b6', '#1abc9c'];

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
                        <label class="peak-active">
                            <input type="checkbox" ${peak.active ? 'checked' : ''}
                                   onchange="togglePeak(${i}, this.checked)">
                            Active
                        </label>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>μ (angle)</span>
                            <span class="value" id="mu-val-${i}">${peak.mu}°</span>
                        </div>
                        <input type="range" id="mu-${i}" min="0" max="360" step="1" value="${peak.mu}"
                               oninput="updatePeak(${i}, 'mu', this.value)" ${!peak.active ? 'disabled' : ''}>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>κ (concentration)</span>
                            <span class="value" id="kappa-val-${i}">${peak.kappa.toFixed(1)}</span>
                        </div>
                        <input type="range" id="kappa-${i}" min="0" max="100" step="0.5" value="${peak.kappa}"
                               oninput="updatePeak(${i}, 'kappa', this.value)" ${!peak.active ? 'disabled' : ''}>
                    </div>
                    <div class="control-row">
                        <div class="control-label">
                            <span>Weight</span>
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
            document.getElementById(`${param}-val-${idx}`).textContent =
                param === 'mu' ? `${value}°` : parseFloat(value).toFixed(param === 'kappa' ? 1 : 2);
            updatePlot();
        }

        function applyGlobalKappa(value) {
            document.getElementById('global-kappa-val').textContent = parseFloat(value).toFixed(1);
            peaks.forEach((peak, i) => {
                if (peak.active) {
                    peak.kappa = parseFloat(value);
                    document.getElementById(`kappa-${i}`).value = value;
                    document.getElementById(`kappa-val-${i}`).textContent = parseFloat(value).toFixed(1);
                }
            });
            updatePlot();
        }

        function loadPreset(name) {
            switch(name) {
                case 'uniform':
                    peaks.forEach((p, i) => {
                        p.active = i === 0;
                        p.kappa = 0;
                    });
                    peaks[0].mu = 0;
                    break;
                case 'single':
                    peaks.forEach((p, i) => {
                        p.active = i === 0;
                        p.kappa = 20;
                    });
                    peaks[0].mu = 0;
                    break;
                case 'bimodal':
                    peaks.forEach((p, i) => {
                        p.active = i < 2;
                        p.kappa = 15;
                        p.weight = 1;
                    });
                    peaks[0].mu = 0;
                    peaks[1].mu = 180;
                    break;
                case 'quad':
                    peaks.forEach((p, i) => {
                        p.active = i < 4;
                        p.kappa = 10;
                        p.weight = 1;
                    });
                    peaks[0].mu = 0;
                    peaks[1].mu = 90;
                    peaks[2].mu = 180;
                    peaks[3].mu = 270;
                    break;
                case 'asymmetric':
                    peaks.forEach((p, i) => {
                        p.active = i < 3;
                        p.weight = 1;
                    });
                    peaks[0].mu = 30; peaks[0].kappa = 25; peaks[0].weight = 2;
                    peaks[1].mu = 150; peaks[1].kappa = 10; peaks[1].weight = 1;
                    peaks[2].mu = 270; peaks[2].kappa = 5; peaks[2].weight = 0.5;
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

            // Normalize weights
            const totalWeight = activePeaks.reduce((sum, p) => sum + p.weight, 0);
            const normalizedWeights = activePeaks.map(p => p.weight / totalWeight);

            const resp = await fetch('/api/compute', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    mus: activePeaks.map(p => p.mu),
                    kappas: activePeaks.map(p => p.kappa),
                    weights: normalizedWeights
                })
            });

            const data = await resp.json();

            // Update info
            document.getElementById('info-peaks').textContent = activePeaks.length;
            document.getElementById('info-entropy').textContent = data.entropy.toFixed(3);
            document.getElementById('info-max').textContent = data.max_density.toFixed(3);

            // Polar plot
            const polarTraces = [];

            // Component distributions (dashed)
            activePeaks.forEach((peak, i) => {
                polarTraces.push({
                    type: 'scatterpolar',
                    r: data.component_pdfs[i],
                    theta: data.theta_deg,
                    name: `Peak ${i+1} (μ=${peak.mu}°, κ=${peak.kappa.toFixed(1)})`,
                    line: {color: colors[peaks.indexOf(peak)], width: 2, dash: 'dash'},
                    opacity: 0.7
                });
            });

            // Mixture distribution (solid)
            polarTraces.push({
                type: 'scatterpolar',
                r: data.mixture_pdf,
                theta: data.theta_deg,
                name: 'Mixture',
                line: {color: '#ffffff', width: 3},
                fill: 'toself',
                fillcolor: 'rgba(255,255,255,0.1)'
            });

            const polarLayout = {
                polar: {
                    radialaxis: {visible: true, range: [0, data.max_density * 1.1]},
                    angularaxis: {direction: 'clockwise', rotation: 90},
                    bgcolor: 'rgba(0,0,0,0)'
                },
                showlegend: true,
                legend: {x: 0, y: -0.2, orientation: 'h', font: {size: 10}},
                margin: {t: 30, b: 80, l: 50, r: 50},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#eee'}
            };

            Plotly.react('polar-plot', polarTraces, polarLayout, {responsive: true});

            // Cartesian plot
            const cartTraces = [];

            activePeaks.forEach((peak, i) => {
                cartTraces.push({
                    x: data.theta_deg,
                    y: data.component_pdfs[i],
                    name: `Peak ${i+1}`,
                    line: {color: colors[peaks.indexOf(peak)], width: 2, dash: 'dash'},
                    opacity: 0.7
                });
            });

            cartTraces.push({
                x: data.theta_deg,
                y: data.mixture_pdf,
                name: 'Mixture',
                line: {color: '#ffffff', width: 3},
                fill: 'tozeroy',
                fillcolor: 'rgba(255,255,255,0.1)'
            });

            const cartLayout = {
                xaxis: {
                    title: 'θ (degrees)',
                    range: [0, 360],
                    tickvals: [0, 45, 90, 135, 180, 225, 270, 315, 360],
                    gridcolor: '#333',
                    zerolinecolor: '#555'
                },
                yaxis: {
                    title: 'Probability Density',
                    range: [0, data.max_density * 1.1],
                    gridcolor: '#333',
                    zerolinecolor: '#555'
                },
                showlegend: false,
                margin: {t: 30, b: 60, l: 60, r: 30},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#eee'}
            };

            Plotly.react('cartesian-plot', cartTraces, cartLayout, {responsive: true});
        }

        // Initialize
        initControls();
        updatePlot();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/compute', methods=['POST'])
def compute():
    data = request.json
    mus_deg = np.array(data['mus'])
    kappas = np.array(data['kappas'])
    weights = np.array(data['weights'])

    # Ensure weights sum to 1
    weights = weights / weights.sum()

    # Theta grid (0 to 360 degrees)
    theta_deg = np.linspace(0, 360, 361)
    theta_rad = np.deg2rad(theta_deg)
    mus_rad = np.deg2rad(mus_deg)

    # Compute component PDFs
    component_pdfs = []
    mixture_pdf = np.zeros_like(theta_rad)

    for mu, kappa, w in zip(mus_rad, kappas, weights):
        if kappa < 0.01:
            # Uniform distribution
            pdf = np.ones_like(theta_rad) / (2 * np.pi)
        else:
            pdf = vonmises.pdf(theta_rad, kappa=kappa, loc=mu)
        component_pdfs.append(pdf.tolist())
        mixture_pdf += w * pdf

    # Compute entropy (approximate)
    # H = -integral(p * log(p))
    p = mixture_pdf + 1e-10  # Avoid log(0)
    entropy = -np.trapezoid(p * np.log(p), theta_rad)

    return jsonify({
        'theta_deg': theta_deg.tolist(),
        'component_pdfs': component_pdfs,
        'mixture_pdf': mixture_pdf.tolist(),
        'max_density': float(mixture_pdf.max()),
        'entropy': float(entropy)
    })


def main():
    parser = argparse.ArgumentParser(description='von Mises Mixture Visualizer')
    parser.add_argument('--port', type=int, default=8088, help='Port number')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  von Mises Mixture Distribution Visualizer")
    print(f"{'='*60}")
    print(f"  Open in browser: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

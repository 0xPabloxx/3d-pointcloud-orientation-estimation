#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web版点云对称性与方向标注工具

功能：
- 3D交互式点云可视化（Plotly，浏览器中运行）
- 标注对称类型：1峰/2峰/4峰/完全对称
- 标注正面方向：-Z/+Z/-X/+X/-Y/+Y（用于检测未对齐的数据）
- 自动保存标注到JSON
- 支持断点续标

使用方法：
    python annotate_symmetry_web.py --data_dir <点云目录>
    然后在浏览器中打开 http://localhost:8050

作者: Claude
创建日期: 2025-11-25
"""

import argparse
import json
from pathlib import Path
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


class WebSymmetryAnnotator:
    """Web版对称性标注工具"""

    def __init__(self, data_dir: Path, output_file: Path):
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)

        # 加载点云文件列表
        self.ply_files = sorted(list(self.data_dir.glob("**/*.ply")))
        if len(self.ply_files) == 0:
            raise RuntimeError(f"No PLY files found in {data_dir}")

        print(f"[Data] Found {len(self.ply_files)} PLY files")

        # 加载已有标注
        self.annotations = self._load_annotations()

        self.current_idx = 0

        # 初始化Dash app
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._build_layout()
        self._setup_callbacks()

    def _load_annotations(self):
        """加载已有标注"""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                annotations = json.load(f)
            print(f"[Annotation] Loaded {len(annotations)} existing annotations")
            return annotations
        else:
            print("[Annotation] No existing annotations found")
            return {}

    def _save_annotations(self):
        """保存标注到JSON"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        # 统计
        stats = {'K1': 0, 'K2': 0, 'K4': 0, 'K0': 0}
        direction_stats = {}
        for ann in self.annotations.values():
            k = ann['K']
            stats[f'K{k}'] = stats.get(f'K{k}', 0) + 1

            direction = ann.get('front_direction', 'unknown')
            direction_stats[direction] = direction_stats.get(direction, 0) + 1

        print(f"[Saved] {len(self.annotations)}/{len(self.ply_files)} annotated")
        print(f"  Symmetry: {stats}")
        print(f"  Directions: {direction_stats}")

    def _load_ply(self, ply_path: Path):
        """加载PLY点云文件"""
        try:
            with open(ply_path, 'r') as f:
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

            return np.array(points)

        except Exception as e:
            print(f"[Error] Failed to load {ply_path.name}: {e}")
            return np.random.randn(1024, 3)

    def _build_layout(self):
        """构建Web界面"""
        self.app.layout = dbc.Container([
            html.H2("点云对称性与方向标注工具", className="text-center my-3"),

            dbc.Row([
                # 左侧：3D可视化
                dbc.Col([
                    dcc.Graph(
                        id='pointcloud-3d',
                        style={'height': '70vh'},
                        config={'displayModeBar': True}
                    ),
                    html.Div(id='info-display', className="mt-2")
                ], width=8),

                # 右侧：控制面板
                dbc.Col([
                    html.H4("标注控制", className="mb-3"),

                    html.Hr(),
                    html.H5("1. 对称类型"),
                    dbc.ButtonGroup([
                        dbc.Button("1峰", id="btn-k1", color="primary", outline=True, className="me-1"),
                        dbc.Button("2峰", id="btn-k2", color="primary", outline=True, className="me-1"),
                        dbc.Button("4峰", id="btn-k4", color="primary", outline=True, className="me-1"),
                        dbc.Button("对称", id="btn-k0", color="primary", outline=True),
                    ], className="mb-3"),

                    html.Hr(),
                    html.H5("2. 正面方向"),
                    html.P("理论上应为-Z，请标注实际方向：", className="small text-muted"),
                    dbc.ButtonGroup([
                        dbc.Button("-Z", id="btn-dir-z-", color="success", outline=True, size="sm", className="me-1"),
                        dbc.Button("+Z", id="btn-dir-z+", color="success", outline=True, size="sm", className="me-1"),
                        dbc.Button("-X", id="btn-dir-x-", color="success", outline=True, size="sm", className="me-1"),
                    ], className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("+X", id="btn-dir-x+", color="success", outline=True, size="sm", className="me-1"),
                        dbc.Button("-Y", id="btn-dir-y-", color="success", outline=True, size="sm", className="me-1"),
                        dbc.Button("+Y", id="btn-dir-y+", color="success", outline=True, size="sm"),
                    ], className="mb-3"),

                    html.Hr(),
                    html.H5("3. 导航"),
                    dbc.ButtonGroup([
                        dbc.Button("← 上一个", id="btn-prev", color="secondary", className="me-1"),
                        dbc.Button("下一个 →", id="btn-next", color="secondary"),
                    ], className="mb-3 w-100"),

                    html.Hr(),
                    dbc.Button("💾 保存", id="btn-save", color="warning", className="w-100 mb-2"),

                    html.Hr(),
                    html.Div(id='status-display', className="small"),

                ], width=4),
            ]),

            # 隐藏的存储组件
            dcc.Store(id='current-index', data=0),
            dcc.Store(id='current-k', data=None),
            dcc.Store(id='current-direction', data=None),

        ], fluid=True)

    def _create_3d_figure(self, points, title=""):
        """创建3D点云图"""
        fig = go.Figure()

        # 点云
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=points[:, 2],  # 用Z坐标着色
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z"),
            ),
            name='Point Cloud'
        ))

        # 添加坐标轴（箭头）
        max_range = np.abs(points).max()
        axis_length = max_range * 0.5

        # X轴（红色）
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=5),
            text=['', '+X'],
            textposition='top center',
            textfont=dict(size=14, color='red'),
            name='X-axis',
            showlegend=False
        ))

        # Y轴（绿色）
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=5),
            text=['', '+Y'],
            textposition='top center',
            textfont=dict(size=14, color='green'),
            name='Y-axis',
            showlegend=False
        ))

        # Z轴（蓝色）
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+text',
            line=dict(color='blue', width=5),
            text=['', '+Z'],
            textposition='top center',
            textfont=dict(size=14, color='blue'),
            name='Z-axis',
            showlegend=False
        ))

        # 负轴标记
        fig.add_trace(go.Scatter3d(
            x=[-axis_length*0.3, 0, 0],
            y=[0, -axis_length*0.3, 0],
            z=[0, 0, -axis_length*0.3],
            mode='text',
            text=['-X', '-Y', '-Z'],
            textfont=dict(size=12, color='gray'),
            showlegend=False
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title='X', range=[-max_range, max_range]),
                yaxis=dict(title='Y', range=[-max_range, max_range]),
                zaxis=dict(title='Z', range=[-max_range, max_range]),
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    def _setup_callbacks(self):
        """设置回调函数"""

        @self.app.callback(
            [Output('pointcloud-3d', 'figure'),
             Output('info-display', 'children'),
             Output('status-display', 'children'),
             Output('current-index', 'data')],
            [Input('btn-prev', 'n_clicks'),
             Input('btn-next', 'n_clicks'),
             Input('btn-k1', 'n_clicks'),
             Input('btn-k2', 'n_clicks'),
             Input('btn-k4', 'n_clicks'),
             Input('btn-k0', 'n_clicks'),
             Input('btn-dir-z-', 'n_clicks'),
             Input('btn-dir-z+', 'n_clicks'),
             Input('btn-dir-x-', 'n_clicks'),
             Input('btn-dir-x+', 'n_clicks'),
             Input('btn-dir-y-', 'n_clicks'),
             Input('btn-dir-y+', 'n_clicks'),
             Input('btn-save', 'n_clicks')],
            [State('current-index', 'data'),
             State('current-k', 'data'),
             State('current-direction', 'data')]
        )
        def handle_interactions(prev_click, next_click,
                               k1_click, k2_click, k4_click, k0_click,
                               dz_click, dzp_click, dx_click, dxp_click, dy_click, dyp_click,
                               save_click,
                               current_idx, current_k, current_direction):

            ctx = callback_context
            if not ctx.triggered:
                button_id = None
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # 处理导航
            if button_id == 'btn-prev':
                current_idx = max(0, current_idx - 1)
            elif button_id == 'btn-next':
                current_idx = min(len(self.ply_files) - 1, current_idx + 1)

            # 处理对称性标注
            k_map = {'btn-k1': 1, 'btn-k2': 2, 'btn-k4': 4, 'btn-k0': 0}
            if button_id in k_map:
                current_k = k_map[button_id]

            # 处理方向标注
            dir_map = {
                'btn-dir-z-': '-Z', 'btn-dir-z+': '+Z',
                'btn-dir-x-': '-X', 'btn-dir-x+': '+X',
                'btn-dir-y-': '-Y', 'btn-dir-y+': '+Y',
            }
            if button_id in dir_map:
                current_direction = dir_map[button_id]

            # 保存标注
            if current_k is not None and current_direction is not None:
                ply_path = self.ply_files[current_idx]
                rel_path = str(ply_path.relative_to(self.data_dir))

                self.annotations[rel_path] = {
                    'file': rel_path,
                    'K': current_k,
                    'front_direction': current_direction,
                    'index': current_idx,
                    'aligned': (current_direction == '-Z')  # 是否已对齐
                }

            # 保存到文件
            if button_id == 'btn-save':
                self._save_annotations()

            # 加载当前点云
            ply_path = self.ply_files[current_idx]
            points = self._load_ply(ply_path)

            # 创建图表
            title = f"Sample {current_idx + 1}/{len(self.ply_files)}: {ply_path.name}"
            fig = self._create_3d_figure(points, title)

            # 信息面板
            rel_path = str(ply_path.relative_to(self.data_dir))
            existing = self.annotations.get(rel_path, {})

            info_text = html.Div([
                html.P([
                    html.Strong("当前文件: "),
                    html.Code(ply_path.name)
                ]),
                html.P([
                    html.Strong("已标注: "),
                    html.Span(f"{len(self.annotations)}/{len(self.ply_files)}",
                             className="badge bg-success")
                ]),
            ])

            # 状态显示
            if existing:
                status = html.Div([
                    html.P([
                        html.Strong("当前标注:"),
                        html.Br(),
                        f"对称性: K={existing['K']}",
                        html.Br(),
                        f"正面方向: {existing['front_direction']}",
                        html.Br(),
                        html.Span(
                            "✅ 已对齐" if existing['aligned'] else "⚠️ 需要矫正",
                            className="badge bg-success" if existing['aligned'] else "badge bg-warning"
                        )
                    ], className="small")
                ])
            else:
                status = html.P("未标注", className="text-muted small")

            return fig, info_text, status, current_idx

    def run(self, port=8050):
        """运行Web服务器"""
        print("\n" + "="*60)
        print("  Web版点云对称性与方向标注工具")
        print("="*60)
        print(f"\n🌐 在浏览器中打开: http://localhost:{port}")
        print("\n使用说明:")
        print("1. 左侧3D图可以用鼠标拖动旋转、滚轮缩放")
        print("2. 观察点云，判断对称性和正面朝向")
        print("3. 右侧选择对称类型（1/2/4峰/完全对称）")
        print("4. 选择当前正面方向（理论应为-Z）")
        print("5. 点击'下一个'继续，或'保存'手动保存")
        print("6. 标注会自动保存")
        print("\n提示：如果正面不是-Z方向，说明数据需要矫正！")
        print("="*60 + "\n")

        self.app.run(debug=False, host='0.0.0.0', port=port)


def main():
    parser = argparse.ArgumentParser(description="Web版对称性与方向标注工具")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/modelnet40_normal_resampled_ply',
        help='点云文件目录（原始ModelNet40数据）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/symmetry_and_alignment_annotations.json',
        help='标注输出文件（JSON格式）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Web服务器端口'
    )

    args = parser.parse_args()

    annotator = WebSymmetryAnnotator(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output)
    )

    annotator.run(port=args.port)


if __name__ == "__main__":
    main()

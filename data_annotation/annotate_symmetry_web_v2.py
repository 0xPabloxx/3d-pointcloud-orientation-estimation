#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web版点云对称性与方向标注工具 v2

修复：
- 修改数据集为 full_mn40_normal_resampled_ply
- 修复按钮点击反馈问题
- 添加"没有正面"类型（5个对称类型）
- 添加明确的保存提示
- 自动保存标注

使用方法：
    python annotate_symmetry_web_v2.py
    然后在浏览器中打开 http://localhost:8050

作者: Claude
创建日期: 2025-11-25
版本: 2.0
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


class WebSymmetryAnnotatorV2:
    """Web版对称性标注工具 v2"""

    def __init__(self, data_dir: Path, output_file: Path, start_category: str = None):
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)

        # 加载已有标注
        self.annotations = self._load_annotations()

        # 扫描所有类别
        self.categories = self._scan_categories()
        if len(self.categories) == 0:
            raise RuntimeError(f"No categories found in {data_dir}")

        print(f"[Categories] Found {len(self.categories)} categories")

        # 计算所有类别的进度
        self.category_progress = self._calculate_all_category_progress()

        # 确定起始类别（优先选择有未标注样本的）
        if start_category and start_category in self.categories:
            self.current_category = start_category
        else:
            # 选择第一个有未标注样本的类别
            self.current_category = self._find_first_incomplete_category()

        print(f"[Category] Starting with: {self.current_category}")

        # 加载当前类别的文件
        self._load_category_files(self.current_category)

        # 初始化Dash app
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._build_layout()
        self._setup_callbacks()

    def _scan_categories(self):
        """扫描所有类别目录"""
        categories = {}
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir():
                ply_files = list(item.glob("*.ply"))
                if ply_files:
                    categories[item.name] = {
                        'path': item,
                        'total': len(ply_files)
                    }
        return categories

    def _calculate_all_category_progress(self):
        """计算所有类别的标注进度"""
        progress = {}
        for cat_name, cat_info in self.categories.items():
            total = cat_info['total']
            annotated = 0

            # 统计该类别已标注数量
            for file_path, ann in self.annotations.items():
                if file_path.startswith(cat_name + '/'):
                    # 检查是否完整标注（有对称性和方向）
                    if ann.get('K') is not None and ann.get('front_direction'):
                        annotated += 1

            progress[cat_name] = {
                'total': total,
                'annotated': annotated,
                'remaining': total - annotated,
                'progress': (annotated / total * 100) if total > 0 else 0
            }

        return progress

    def _find_first_incomplete_category(self):
        """找到第一个有未标注样本的类别"""
        # 按剩余数量降序排序，选择剩余最多的
        sorted_cats = sorted(
            self.category_progress.items(),
            key=lambda x: x[1]['remaining'],
            reverse=True
        )
        for cat_name, prog in sorted_cats:
            if prog['remaining'] > 0:
                return cat_name
        # 如果全部标注完成，返回第一个类别
        return list(self.categories.keys())[0] if self.categories else None

    def _load_category_files(self, category_name):
        """加载指定类别的所有PLY文件"""
        cat_path = self.categories[category_name]['path']
        self.ply_files = sorted(list(cat_path.glob("*.ply")))
        print(f"[Category] Loaded {len(self.ply_files)} files from '{category_name}'")

        # 找到该类别第一个未标注的样本
        self.start_index = self._find_first_unannotated()

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

    def _find_first_unannotated(self):
        """找到第一个未标注的样本索引"""
        for idx, ply_path in enumerate(self.ply_files):
            rel_path = str(ply_path.relative_to(self.data_dir))
            ann = self.annotations.get(rel_path)
            # 如果没有标注，或者标注不完整（缺少方向）
            if not ann or not ann.get('front_direction'):
                print(f"[Resume] Starting from sample {idx + 1}/{len(self.ply_files)}: {ply_path.name}")
                return idx

        # 全部标注完成
        print(f"[Resume] All samples annotated! Starting from beginning.")
        return 0

    def _save_annotations(self):
        """保存标注到JSON + CSV + Markdown"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. 保存JSON（完整数据）
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)

        # 2. 保存CSV（表格格式，Excel可打开）
        import csv
        csv_file = self.output_file.with_suffix('.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.annotations:
                writer = csv.DictWriter(f, fieldnames=[
                    'file', 'K', 'symmetry_name', 'front_direction',
                    'aligned', 'needs_correction', 'rotation_needed'
                ])
                writer.writeheader()
                for ann in self.annotations.values():
                    # 计算需要的旋转
                    rotation_needed = self._calculate_rotation(ann.get('front_direction'))
                    writer.writerow({
                        'file': ann['file'],
                        'K': ann['K'],
                        'symmetry_name': ann.get('symmetry_name', ''),
                        'front_direction': ann.get('front_direction', ''),
                        'aligned': 'YES' if ann.get('aligned', False) else 'NO',
                        'needs_correction': 'NO' if ann.get('aligned', False) else 'YES',
                        'rotation_needed': rotation_needed
                    })

        # 3. 生成Markdown报告
        md_file = self.output_file.with_suffix('.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 点云对称性与方向标注报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据集**: {self.data_dir}\n\n")
            f.write("---\n\n")

            # 统计
            stats = {}
            direction_stats = {}
            need_correction = []

            for ann in self.annotations.values():
                k_name = ann.get('symmetry_name', '')
                stats[k_name] = stats.get(k_name, 0) + 1

                direction = ann.get('front_direction', 'unknown')
                direction_stats[direction] = direction_stats.get(direction, 0) + 1

                if not ann.get('aligned', False):
                    need_correction.append(ann)

            f.write("## 📊 标注统计\n\n")
            f.write(f"- **总标注数**: {len(self.annotations)}/{len(self.ply_files)}\n")
            f.write(f"- **完成进度**: {len(self.annotations)/len(self.ply_files)*100:.1f}%\n\n")

            f.write("### 对称性分布\n\n")
            f.write("| 对称类型 | 数量 | 占比 |\n")
            f.write("|---------|------|------|\n")
            for k_name, count in sorted(stats.items()):
                pct = count / len(self.annotations) * 100
                f.write(f"| {k_name} | {count} | {pct:.1f}% |\n")

            f.write("\n### 正面方向分布\n\n")
            f.write("| 方向 | 数量 | 占比 | 状态 |\n")
            f.write("|------|------|------|------|\n")
            for direction, count in sorted(direction_stats.items()):
                pct = count / len(self.annotations) * 100
                status = "✅ 已对齐" if direction == "-Z" else "⚠️ 需矫正"
                f.write(f"| {direction} | {count} | {pct:.1f}% | {status} |\n")

            f.write("\n---\n\n")
            f.write(f"## ⚠️ 需要矫正的数据（共{len(need_correction)}个）\n\n")

            if need_correction:
                f.write("| 文件 | 对称类型 | 当前方向 | 需要旋转 |\n")
                f.write("|------|---------|----------|----------|\n")
                for ann in sorted(need_correction, key=lambda x: x['file']):
                    rotation = self._calculate_rotation(ann.get('front_direction'))
                    f.write(f"| `{ann['file']}` | {ann.get('symmetry_name', '')} | "
                           f"{ann.get('front_direction', '')} | {rotation} |\n")
            else:
                f.write("✅ 所有数据已对齐！\n")

            f.write("\n---\n\n")
            f.write("## 📋 完整标注列表\n\n")
            f.write("| 文件 | 对称类型 | 方向 | 对齐状态 |\n")
            f.write("|------|---------|------|----------|\n")
            for ann in sorted(self.annotations.values(), key=lambda x: x['file']):
                status = "✅" if ann.get('aligned', False) else "⚠️"
                f.write(f"| `{ann['file']}` | {ann.get('symmetry_name', '')} | "
                       f"{ann.get('front_direction', '')} | {status} |\n")

        print(f"[Saved] JSON: {self.output_file}")
        print(f"[Saved] CSV: {csv_file}")
        print(f"[Saved] Report: {md_file}")

        return stats, direction_stats

    def _calculate_rotation(self, current_direction):
        """计算需要的旋转角度来对齐到-Z"""
        if not current_direction:
            return "未知"

        rotation_map = {
            '-Z': '无需旋转（已对齐）',
            '+Z': '绕X轴旋转180°',
            '-X': '绕Y轴旋转-90°',
            '+X': '绕Y轴旋转+90°',
            '-Y': '绕X轴旋转+90°',
            '+Y': '绕X轴旋转-90°',
        }

        return rotation_map.get(current_direction, '未知方向')

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
            html.H2("🏷️ 点云对称性与方向标注工具", className="text-center my-3"),

            dbc.Alert([
                html.Strong("📂 数据集: "), "full_mn40_normal_resampled_ply (完整对齐数据)",
                html.Br(),
                html.Strong("💾 标注保存: "), html.Code(str(self.output_file)),
                html.Br(),
                html.Strong("✨ 自动保存: "), "每次选择后自动保存到JSON文件"
            ], color="info", className="mb-3"),

            # 类别选择器
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("📂 选择类别:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='category-selector',
                                options=[{'label': f"{cat} ({self.category_progress[cat]['annotated']}/{self.category_progress[cat]['total']})",
                                         'value': cat}
                                        for cat in sorted(self.categories.keys())],
                                value=self.current_category,
                                clearable=False,
                                className="mb-2"
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Div(id='category-progress-info', className="mt-4", children=self._get_category_progress_display(self.current_category))
                        ], width=6),
                    ])
                ])
            ], className="mb-3"),

            dbc.Row([
                # 左侧：3D可视化
                dbc.Col([
                    dcc.Graph(
                        id='pointcloud-3d',
                        style={'height': '65vh'},
                        config={'displayModeBar': True}
                    ),
                ], width=8),

                # 右侧：控制面板
                dbc.Col([
                    html.H5("📊 进度信息", className="mb-2"),
                    html.Div(id='progress-info', className="mb-3"),

                    html.Hr(),
                    html.H5("1️⃣ 对称类型", className="mb-2"),
                    html.Div([
                        dbc.Button("1个正面", id="btn-k1", color="primary", className="w-100 mb-2", size="lg"),
                        dbc.Button("2个正面", id="btn-k2", color="primary", className="w-100 mb-2", size="lg"),
                        dbc.Button("4个正面", id="btn-k4", color="primary", className="w-100 mb-2", size="lg"),
                        dbc.Button("完全对称", id="btn-k0", color="primary", className="w-100 mb-2", size="lg"),
                        dbc.Button("没有正面", id="btn-k-none", color="primary", className="w-100 mb-2", size="lg"),
                    ]),

                    html.Hr(),
                    html.H5("2️⃣ 正面方向", className="mb-2"),
                    html.P("理论上应为-Z，请标注实际方向：", className="small text-muted mb-2"),
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("-Z", id="btn-dir-z-", color="success", size="sm", className="me-1"),
                            dbc.Button("+Z", id="btn-dir-z+", color="success", size="sm", className="me-1"),
                            dbc.Button("-X", id="btn-dir-x-", color="success", size="sm", className="me-1"),
                        ], className="mb-2 w-100"),
                        dbc.ButtonGroup([
                            dbc.Button("+X", id="btn-dir-x+", color="success", size="sm", className="me-1"),
                            dbc.Button("-Y", id="btn-dir-y-", color="success", size="sm", className="me-1"),
                            dbc.Button("+Y", id="btn-dir-y+", color="success", size="sm"),
                        ], className="mb-2 w-100"),
                    ]),

                    html.Hr(),
                    html.Div(id='annotation-status', className="mb-3"),

                    html.Hr(),
                    html.H5("3️⃣ 导航", className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("⬅️ 上一个", id="btn-prev", color="secondary", className="me-1"),
                        dbc.Button("下一个 ➡️", id="btn-next", color="secondary"),
                    ], className="mb-3 w-100"),

                    dbc.Button("💾 手动保存", id="btn-manual-save", color="warning", className="w-100 mb-2"),
                    html.Div(id='save-message', className="small text-success"),

                ], width=4),
            ]),

            # 隐藏的存储组件
            dcc.Store(id='current-index', data=self.start_index),
            dcc.Store(id='current-category', data=self.current_category),
            dcc.Interval(id='interval', interval=1000, n_intervals=0),  # 每秒更新一次

        ], fluid=True)

    def _create_3d_figure(self, points, title=""):
        """创建3D点云图"""
        fig = go.Figure()

        # 点云（采样显示，加快渲染）
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            points_display = points[indices]
        else:
            points_display = points

        fig.add_trace(go.Scatter3d(
            x=points_display[:, 0],
            y=points_display[:, 1],
            z=points_display[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_display[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z", len=0.5),
            ),
            name='Point Cloud',
            hoverinfo='skip'
        ))

        # 坐标轴
        max_range = np.abs(points).max()
        axis_length = max_range * 0.6

        # X轴（红色）
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=8),
            text=['', '+X'],
            textposition='top center',
            textfont=dict(size=16, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Y轴（绿色）
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=8),
            text=['', '+Y'],
            textposition='top center',
            textfont=dict(size=16, color='green'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Z轴（蓝色）
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+text',
            line=dict(color='blue', width=8),
            text=['', '+Z'],
            textposition='top center',
            textfont=dict(size=16, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # 负轴标记
        fig.add_trace(go.Scatter3d(
            x=[-axis_length*0.4, 0, 0],
            y=[0, -axis_length*0.4, 0],
            z=[0, 0, -axis_length*0.4],
            mode='text',
            text=['-X', '-Y', '-Z'],
            textfont=dict(size=14, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            scene=dict(
                xaxis=dict(title='X', range=[-max_range, max_range], showgrid=True),
                yaxis=dict(title='Y', range=[-max_range, max_range], showgrid=True),
                zaxis=dict(title='Z', range=[-max_range, max_range], showgrid=True),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )

        return fig

    def _get_current_annotation(self, idx):
        """获取当前样本的标注"""
        ply_path = self.ply_files[idx]
        rel_path = str(ply_path.relative_to(self.data_dir))
        return self.annotations.get(rel_path, None)

    def _save_current_annotation(self, idx, k_value, k_name, direction):
        """保存当前标注"""
        ply_path = self.ply_files[idx]
        rel_path = str(ply_path.relative_to(self.data_dir))

        self.annotations[rel_path] = {
            'file': rel_path,
            'K': k_value,
            'symmetry_name': k_name,
            'front_direction': direction,
            'index': idx,
            'aligned': (direction == '-Z')
        }

        # 保存到文件
        stats, dir_stats = self._save_annotations()

        # 重新计算类别进度
        self.category_progress = self._calculate_all_category_progress()

        return stats, dir_stats

    def _get_category_progress_display(self, category_name):
        """生成类别进度显示HTML"""
        prog = self.category_progress[category_name]

        # 创建进度条
        progress_bar_filled = int(prog['progress'] / 10)
        progress_bar = "█" * progress_bar_filled + "░" * (10 - progress_bar_filled)

        return html.Div([
            html.P([
                html.Strong(f"{category_name}"), " 的标注进度:"
            ], className="mb-1"),
            html.P([
                html.Span(f"{prog['annotated']}/{prog['total']}",
                         className="badge bg-primary me-2"),
                html.Span(f"{prog['progress']:.1f}%",
                         className="badge bg-success me-2"),
                html.Span(f"剩余: {prog['remaining']}",
                         className="badge bg-warning")
            ], className="mb-1"),
            html.Code(progress_bar + f" {prog['progress']:.1f}%", style={'fontSize': '12px'})
        ])

    def _setup_callbacks(self):
        """设置回调函数"""

        # 类别切换回调
        @self.app.callback(
            [Output('current-index', 'data', allow_duplicate=True),
             Output('current-category', 'data'),
             Output('category-progress-info', 'children')],
            [Input('category-selector', 'value')],
            [State('current-category', 'data')],
            prevent_initial_call=True
        )
        def switch_category(selected_category, prev_category):
            if selected_category != prev_category:
                # 切换类别
                self.current_category = selected_category
                self._load_category_files(selected_category)
                print(f"[Switch] Category changed to: {selected_category}")

            # 生成进度显示
            progress_display = self._get_category_progress_display(selected_category)

            return self.start_index, selected_category, progress_display

        # 主回调：处理所有交互
        @self.app.callback(
            [Output('pointcloud-3d', 'figure'),
             Output('progress-info', 'children'),
             Output('annotation-status', 'children'),
             Output('save-message', 'children'),
             Output('current-index', 'data', allow_duplicate=True),
             Output('category-selector', 'options'),
             Output('category-progress-info', 'children', allow_duplicate=True)],
            [Input('btn-prev', 'n_clicks'),
             Input('btn-next', 'n_clicks'),
             Input('btn-k1', 'n_clicks'),
             Input('btn-k2', 'n_clicks'),
             Input('btn-k4', 'n_clicks'),
             Input('btn-k0', 'n_clicks'),
             Input('btn-k-none', 'n_clicks'),
             Input('btn-dir-z-', 'n_clicks'),
             Input('btn-dir-z+', 'n_clicks'),
             Input('btn-dir-x-', 'n_clicks'),
             Input('btn-dir-x+', 'n_clicks'),
             Input('btn-dir-y-', 'n_clicks'),
             Input('btn-dir-y+', 'n_clicks'),
             Input('btn-manual-save', 'n_clicks'),
             Input('interval', 'n_intervals')],
            [State('current-index', 'data')],
            prevent_initial_call=True
        )
        def handle_all_interactions(*args):
            ctx = callback_context
            current_idx = args[-1]  # 最后一个参数是State

            if current_idx is None:
                current_idx = 0

            save_msg = ""

            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                # 导航
                if button_id == 'btn-prev':
                    current_idx = max(0, current_idx - 1)
                elif button_id == 'btn-next':
                    current_idx = min(len(self.ply_files) - 1, current_idx + 1)

                # 对称性标注
                k_map = {
                    'btn-k1': (1, '1个正面'),
                    'btn-k2': (2, '2个正面'),
                    'btn-k4': (4, '4个正面'),
                    'btn-k0': (0, '完全对称'),
                    'btn-k-none': (-1, '没有正面'),
                }

                # 方向标注
                dir_map = {
                    'btn-dir-z-': '-Z', 'btn-dir-z+': '+Z',
                    'btn-dir-x-': '-X', 'btn-dir-x+': '+X',
                    'btn-dir-y-': '-Y', 'btn-dir-y+': '+Y',
                }

                # 获取当前标注
                current_ann = self._get_current_annotation(current_idx)

                # 处理对称性点击
                if button_id in k_map:
                    k_val, k_name = k_map[button_id]
                    direction = current_ann['front_direction'] if current_ann else None

                    if direction:  # 如果已经选择了方向，立即保存
                        stats, dir_stats = self._save_current_annotation(current_idx, k_val, k_name, direction)
                        save_msg = f"✅ 已保存！对称性: {k_name}, 方向: {direction}"
                    else:
                        # 临时保存对称性，等待方向选择
                        self.annotations[str(self.ply_files[current_idx].relative_to(self.data_dir))] = {
                            'K': k_val,
                            'symmetry_name': k_name,
                            'front_direction': None,
                        }
                        save_msg = f"⏳ 已选择: {k_name}，请选择正面方向"

                # 处理方向点击
                elif button_id in dir_map:
                    direction = dir_map[button_id]
                    current_ann = self._get_current_annotation(current_idx)

                    if current_ann and current_ann.get('K') is not None:  # 如果已经选择了对称性
                        k_val = current_ann['K']
                        k_name = current_ann.get('symmetry_name', f'K={k_val}')
                        stats, dir_stats = self._save_current_annotation(current_idx, k_val, k_name, direction)
                        save_msg = f"✅ 已保存！对称性: {k_name}, 方向: {direction}"
                    else:
                        save_msg = "⚠️ 请先选择对称类型！"

                # 手动保存
                elif button_id == 'btn-manual-save':
                    stats, dir_stats = self._save_annotations()
                    save_msg = f"💾 手动保存成功！共{len(self.annotations)}个标注"

            # 加载当前点云
            ply_path = self.ply_files[current_idx]
            points = self._load_ply(ply_path)

            title = f"[{current_idx + 1}/{len(self.ply_files)}] {ply_path.name}"
            fig = self._create_3d_figure(points, title)

            # 进度信息（显示当前类别的进度）
            current_category_prog = self.category_progress[self.current_category]
            progress = html.Div([
                html.P([
                    html.Strong("📁 当前: "),
                    html.Code(ply_path.name, style={'fontSize': '11px'})
                ], className="mb-1"),
                html.P([
                    html.Strong(f"📊 {self.current_category}: "),
                    html.Span(f"{current_category_prog['annotated']}/{current_category_prog['total']}",
                             className="badge bg-primary me-2"),
                    html.Span(f"{current_category_prog['progress']:.1f}%",
                             className="badge bg-success")
                ], className="mb-0"),
            ])

            # 当前标注状态
            current_ann = self._get_current_annotation(current_idx)
            if current_ann and current_ann.get('front_direction'):
                status = dbc.Alert([
                    html.Strong("当前标注:"),
                    html.Hr(className="my-2"),
                    html.P([
                        "🏷️ 对称性: ",
                        html.Strong(current_ann.get('symmetry_name', f"K={current_ann['K']}"))
                    ], className="mb-1"),
                    html.P([
                        "🧭 方向: ",
                        html.Strong(current_ann['front_direction']),
                        html.Span(" ✅ 已对齐" if current_ann['aligned'] else " ⚠️ 需矫正",
                                 className="ms-2 badge " + ("bg-success" if current_ann['aligned'] else "bg-warning"))
                    ], className="mb-0"),
                ], color="success", className="py-2")
            elif current_ann and current_ann.get('K') is not None:
                status = dbc.Alert([
                    html.P([
                        "⏳ 已选择: ",
                        html.Strong(current_ann.get('symmetry_name', f"K={current_ann['K']}")),
                    ], className="mb-0"),
                    html.P("请选择正面方向 ⬇️", className="mb-0 mt-1 small text-muted")
                ], color="warning", className="py-2")
            else:
                status = dbc.Alert("未标注", color="light", className="py-2")

            # 更新类别选择器选项（显示最新进度）
            category_options = [
                {'label': f"{cat} ({self.category_progress[cat]['annotated']}/{self.category_progress[cat]['total']})",
                 'value': cat}
                for cat in sorted(self.categories.keys())
            ]

            # 更新当前类别进度显示
            category_progress_display = self._get_category_progress_display(self.current_category)

            return fig, progress, status, save_msg, current_idx, category_options, category_progress_display

    def run(self, port=8050):
        """运行Web服务器"""
        print("\n" + "="*70)
        print("  🏷️  Web版点云对称性与方向标注工具 v2")
        print("="*70)
        print(f"\n📂 数据集: {self.data_dir}")
        print(f"📊 样本数: {len(self.ply_files)}")
        print(f"💾 输出文件: {self.output_file}")
        print(f"\n🌐 在浏览器中打开: http://localhost:{port}")
        print("\n📖 使用说明:")
        print("  1. 左侧3D图可以用鼠标拖动旋转、滚轮缩放")
        print("  2. 右侧先点击'对称类型'按钮（1个正面/2个正面/4个正面/完全对称/没有正面）")
        print("  3. 再点击'正面方向'按钮（-Z/+Z/-X/+X/-Y/+Y）")
        print("  4. 两个都选择后会自动保存到JSON文件")
        print("  5. 点击'下一个'继续标注")
        print("\n💡 提示：")
        print("  - 如果正面不是-Z方向，说明数据需要矫正！")
        print("  - 标注会实时自动保存到JSON文件")
        print("  - 可以随时按Ctrl+C退出，下次继续")
        print("="*70 + "\n")

        self.app.run(debug=False, host='0.0.0.0', port=port)


def main():
    parser = argparse.ArgumentParser(description="Web版对称性与方向标注工具 v2")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply',
        help='点云文件目录（完整对齐的ModelNet40数据）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations.json',
        help='标注输出文件（JSON格式）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Web服务器端口'
    )

    args = parser.parse_args()

    annotator = WebSymmetryAnnotatorV2(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output)
    )

    annotator.run(port=args.port)


if __name__ == "__main__":
    main()

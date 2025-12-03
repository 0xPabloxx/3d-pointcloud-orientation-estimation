#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云截图展示器 - 用于论文图片截取

功能：
1. 加载PLY点云文件并3D显示
2. 支持多种渲染风格（论文风格、彩色、按高度着色等）
3. 预设多个视角（正面、侧面、45度等）
4. 一键截图保存为高分辨率PNG
5. 支持批量截图
6. 支持自定义背景颜色
7. 支持已标注数据的一键对齐（front/upright/side）

使用方法：
    python pointcloud_screenshot_viewer.py
    然后在浏览器中打开 http://localhost:8051

作者: Claude
创建日期: 2025-11-27
更新日期: 2025-11-27
"""

import argparse
import json
from pathlib import Path
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


def compute_rotation_matrix(front_dir: str, up_dir: str):
    """
    根据当前前向方向和竖直方向，计算将点云旋转到标准姿态的旋转矩阵

    标准姿态：前向=-Z, 上方=+Y, 右侧=+X
    """
    dir_to_vec = {
        '+X': np.array([1, 0, 0], dtype=np.float64),
        '-X': np.array([-1, 0, 0], dtype=np.float64),
        '+Y': np.array([0, 1, 0], dtype=np.float64),
        '-Y': np.array([0, -1, 0], dtype=np.float64),
        '+Z': np.array([0, 0, 1], dtype=np.float64),
        '-Z': np.array([0, 0, -1], dtype=np.float64),
    }

    if front_dir not in dir_to_vec or up_dir not in dir_to_vec:
        return None

    current_front = dir_to_vec[front_dir]
    current_up = dir_to_vec[up_dir]

    if abs(np.dot(current_front, current_up)) > 1e-6:
        return None

    current_right = np.cross(current_front, current_up)
    current_basis = np.column_stack([current_right, current_up, -current_front])
    rotation_matrix = current_basis.T

    return rotation_matrix


class PointCloudScreenshotViewer:
    """点云截图展示器"""

    def __init__(self, data_dir: Path, output_dir: Path, annotation_file: Path = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载标注数据（用于一键对齐）
        self.annotations = {}
        if annotation_file and Path(annotation_file).exists():
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"[Annotations] Loaded {len(self.annotations)} annotations")

        # 扫描所有类别和文件
        self.categories = self._scan_categories()
        if len(self.categories) == 0:
            raise RuntimeError(f"No categories found in {data_dir}")

        print(f"[Categories] Found {len(self.categories)} categories")

        # 当前状态
        self.current_category = list(self.categories.keys())[0]
        self.current_files = self._get_category_files(self.current_category)
        self.current_index = 0

        # 背景颜色选项
        self.bg_colors = {
            'white': '#FFFFFF',
            'light_gray': '#F5F5F5',
            'gray': '#E0E0E0',
            'dark_gray': '#808080',
            'black': '#000000',
            'light_blue': '#E3F2FD',
            'cream': '#FFFAF0',
            'transparent': 'rgba(0,0,0,0)',
        }

        # 点云颜色选项（纯色）
        self.solid_colors = {
            'blue': '#1f77b4',
            'black': '#333333',
            'white': '#FFFFFF',
            'red': '#E53935',
            'green': '#43A047',
            'orange': '#FF9800',
            'yellow': '#FFEB3B',
            'purple': '#9C27B0',
            'cyan': '#00BCD4',
            'pink': '#E91E63',
            'brown': '#795548',
            'gray': '#9E9E9E',
        }

        # 渲染设置（着色方案）
        self.color_schemes = {
            'height_blues': {'colorscale': 'Blues', 'description': 'Height (Blues)'},
            'height_grays': {'colorscale': 'Greys', 'description': 'Height (Grays)'},
            'height_viridis': {'colorscale': 'Viridis', 'description': 'Height (Viridis)'},
            'height_jet': {'colorscale': 'Jet', 'description': 'Height (Jet)'},
            'height_plasma': {'colorscale': 'Plasma', 'description': 'Height (Plasma)'},
            'height_turbo': {'colorscale': 'Turbo', 'description': 'Height (Turbo)'},
            'solid': {'colorscale': None, 'description': 'Solid Color'},
        }

        # 预设视角 (eye position + up vector)
        # 标准姿态: Front=-Z (物体正面朝向-Z), Up=+Y, Right=+X
        # 要看到正面，相机需要在-Z方向看向原点
        # eye: 相机位置, up: 相机的上方向
        self.camera_presets = {
            # 正面视图：相机在-Z位置，看向原点，能看到物体的正面（-Z方向）
            'front': {'eye': {'x': 0, 'y': 0, 'z': -2.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 正面45度仰视
            'front_45_up': {'eye': {'x': 0, 'y': 1.2, 'z': -2.0}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 右侧视图：从+X看向原点，Y轴朝上
            'side_right': {'eye': {'x': 2.5, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 左侧视图：从-X看向原点，Y轴朝上
            'side_left': {'eye': {'x': -2.5, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 俯视图：从+Y看向原点，-Z朝上（前方朝上）
            'top': {'eye': {'x': 0, 'y': 2.5, 'z': 0.01}, 'up': {'x': 0, 'y': 0, 'z': -1}},
            # 仰视图：从-Y看向原点，-Z朝上
            'bottom': {'eye': {'x': 0, 'y': -2.5, 'z': 0.01}, 'up': {'x': 0, 'y': 0, 'z': -1}},
            # 等轴测视角（从前方看的角度），Y轴朝上
            'isometric': {'eye': {'x': 1.5, 'y': 1.5, 'z': -1.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            'isometric_2': {'eye': {'x': -1.5, 'y': 1.5, 'z': -1.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            'isometric_3': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            'isometric_4': {'eye': {'x': -1.5, 'y': 1.5, 'z': 1.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 背面视图：相机在+Z位置，看向原点，看到物体背面
            'back': {'eye': {'x': 0, 'y': 0, 'z': 2.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
            # 低角度视图（从前方）
            'low_angle': {'eye': {'x': 0, 'y': -0.5, 'z': -2.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
        }

        # 初始化Dash app
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
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

    def _get_category_files(self, category_name):
        """获取类别下的所有PLY文件"""
        cat_path = self.categories[category_name]['path']
        return sorted(list(cat_path.glob("*.ply")))

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

    def _get_annotation(self, file_path):
        """获取文件的标注信息"""
        rel_path = str(Path(file_path).relative_to(self.data_dir))
        return self.annotations.get(rel_path, None)

    def _align_points(self, points, annotation):
        """根据标注对齐点云"""
        if not annotation:
            return points

        front_dir = annotation.get('front_direction')
        up_dir = annotation.get('up_direction')

        if not front_dir or not up_dir:
            return points

        R = compute_rotation_matrix(front_dir, up_dir)
        if R is None:
            return points

        return points @ R.T

    def _create_3d_figure(self, points, color_scheme='height_blues', solid_color='blue',
                          bg_color='white', point_size=2, camera_preset='isometric',
                          show_axes=True, show_grid=False, title="",
                          zoom=2.5):
        """创建3D点云图

        Args:
            zoom: 缩放级别，值越小越近（默认2.5）
        """
        scheme = self.color_schemes.get(color_scheme, self.color_schemes['height_blues'])

        # 获取预设相机，然后根据zoom调整距离
        base_camera = self.camera_presets.get(camera_preset, self.camera_presets['isometric'])

        # 计算相机方向向量并根据zoom调整距离
        eye = base_camera['eye']
        # 计算原始距离
        orig_dist = np.sqrt(eye['x']**2 + eye['y']**2 + eye['z']**2)
        # 根据zoom缩放（预设距离是2.5，所以用zoom/2.5作为缩放因子，但直接用zoom作为新距离更直观）
        if orig_dist > 0:
            scale = zoom / orig_dist
            camera = {
                'eye': {'x': eye['x'] * scale, 'y': eye['y'] * scale, 'z': eye['z'] * scale},
                'up': base_camera['up']
            }
        else:
            camera = base_camera

        bg = self.bg_colors.get(bg_color, '#FFFFFF')

        fig = go.Figure()

        # 准备颜色
        if scheme.get('colorscale'):
            marker_config = dict(
                size=point_size,
                color=points[:, 1],  # Y是高度
                colorscale=scheme['colorscale'],
                showscale=False,
            )
        else:
            marker_config = dict(
                size=point_size,
                color=self.solid_colors.get(solid_color, '#1f77b4'),
            )

        # 添加点云
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=marker_config,
            hoverinfo='skip'
        ))

        # 坐标轴
        if show_axes:
            max_range = np.abs(points).max()
            axis_length = max_range * 0.5

            # X轴（红色）
            fig.add_trace(go.Scatter3d(
                x=[0, axis_length], y=[0, 0], z=[0, 0],
                mode='lines+text',
                line=dict(color='red', width=5),
                text=['', 'X'],
                textposition='top center',
                textfont=dict(size=14, color='red'),
                showlegend=False, hoverinfo='skip'
            ))
            # Y轴（绿色）
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, axis_length], z=[0, 0],
                mode='lines+text',
                line=dict(color='green', width=5),
                text=['', 'Y'],
                textposition='top center',
                textfont=dict(size=14, color='green'),
                showlegend=False, hoverinfo='skip'
            ))
            # Z轴（蓝色）
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, axis_length],
                mode='lines+text',
                line=dict(color='blue', width=5),
                text=['', 'Z'],
                textposition='top center',
                textfont=dict(size=14, color='blue'),
                showlegend=False, hoverinfo='skip'
            ))

        # 计算范围
        max_range = np.abs(points).max() * 1.1

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)) if title else None,
            scene=dict(
                xaxis=dict(
                    title='', range=[-max_range, max_range],
                    showgrid=show_grid, showticklabels=False,
                    showbackground=False, visible=show_grid
                ),
                yaxis=dict(
                    title='', range=[-max_range, max_range],
                    showgrid=show_grid, showticklabels=False,
                    showbackground=False, visible=show_grid
                ),
                zaxis=dict(
                    title='', range=[-max_range, max_range],
                    showgrid=show_grid, showticklabels=False,
                    showbackground=False, visible=show_grid
                ),
                aspectmode='cube',
                camera=dict(
                    eye=camera['eye'],
                    up=camera['up'],
                    center=dict(x=0, y=0, z=0)
                )
            ),
            margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            showlegend=False,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
        )

        return fig

    def _build_layout(self):
        """构建Web界面"""
        self.app.layout = dbc.Container([
            html.H2("Point Cloud Screenshot Tool", className="text-center my-3"),

            dbc.Alert([
                html.Strong("Output: "), html.Code(str(self.output_dir)),
                html.Span(" | ", className="mx-2"),
                html.Strong("Annotations: "),
                html.Span(f"{len(self.annotations)} loaded" if self.annotations else "None",
                         className="text-success" if self.annotations else "text-muted"),
            ], color="info", className="mb-3"),

            dbc.Row([
                # 左侧：3D可视化
                dbc.Col([
                    dcc.Graph(
                        id='pointcloud-3d',
                        style={'height': '75vh'},
                        config={
                            'displayModeBar': True,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'pointcloud',
                                'height': 1200,
                                'width': 1200,
                                'scale': 3
                            }
                        }
                    ),
                ], width=8),

                # 右侧：控制面板
                dbc.Col([
                    # 文件选择
                    dbc.Card([
                        dbc.CardHeader("File Selection"),
                        dbc.CardBody([
                            html.Label("Category:", className="fw-bold"),
                            dcc.Dropdown(
                                id='category-selector',
                                options=[{'label': f"{cat} ({info['total']})", 'value': cat}
                                        for cat, info in self.categories.items()],
                                value=self.current_category,
                                clearable=False,
                                className="mb-2"
                            ),
                            html.Label("File:", className="fw-bold"),
                            dcc.Dropdown(
                                id='file-selector',
                                options=[],
                                value=None,
                                clearable=False,
                                className="mb-2"
                            ),
                            dbc.ButtonGroup([
                                dbc.Button("Prev", id="btn-prev", color="secondary", size="sm"),
                                dbc.Button("Next", id="btn-next", color="secondary", size="sm"),
                            ], className="w-100"),
                        ])
                    ], className="mb-2"),

                    # 一键对齐（如果有标注）
                    dbc.Card([
                        dbc.CardHeader("Auto-Align (requires annotation)"),
                        dbc.CardBody([
                            dbc.Checklist(
                                id='align-checkbox',
                                options=[{'label': ' Apply alignment rotation', 'value': 'align'}],
                                value=[],
                                className="mb-2"
                            ),
                            html.Div(id='align-status', className="small"),
                            html.Hr(className="my-2"),
                            html.Label("Quick Views (aligned):", className="fw-bold small"),
                            dbc.ButtonGroup([
                                dbc.Button("Front", id="btn-align-front", color="info", size="sm", className="me-1"),
                                dbc.Button("Side", id="btn-align-side", color="info", size="sm", className="me-1"),
                                dbc.Button("Top", id="btn-align-top", color="info", size="sm"),
                            ], className="w-100"),
                        ])
                    ], className="mb-2"),

                    # 颜色设置
                    dbc.Card([
                        dbc.CardHeader("Colors"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Color Scheme:", className="fw-bold small"),
                                    dcc.Dropdown(
                                        id='color-scheme-selector',
                                        options=[
                                            {'label': 'Height (Blues)', 'value': 'height_blues'},
                                            {'label': 'Height (Grays)', 'value': 'height_grays'},
                                            {'label': 'Height (Viridis)', 'value': 'height_viridis'},
                                            {'label': 'Height (Jet)', 'value': 'height_jet'},
                                            {'label': 'Height (Plasma)', 'value': 'height_plasma'},
                                            {'label': 'Height (Turbo)', 'value': 'height_turbo'},
                                            {'label': 'Solid Color', 'value': 'solid'},
                                        ],
                                        value='solid',
                                        clearable=False,
                                        className="mb-2"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Solid Color:", className="fw-bold small"),
                                    dcc.Dropdown(
                                        id='solid-color-selector',
                                        options=[
                                            {'label': 'Blue', 'value': 'blue'},
                                            {'label': 'Black', 'value': 'black'},
                                            {'label': 'White', 'value': 'white'},
                                            {'label': 'Red', 'value': 'red'},
                                            {'label': 'Green', 'value': 'green'},
                                            {'label': 'Orange', 'value': 'orange'},
                                            {'label': 'Yellow', 'value': 'yellow'},
                                            {'label': 'Purple', 'value': 'purple'},
                                            {'label': 'Cyan', 'value': 'cyan'},
                                            {'label': 'Pink', 'value': 'pink'},
                                            {'label': 'Brown', 'value': 'brown'},
                                            {'label': 'Gray', 'value': 'gray'},
                                        ],
                                        value='white',
                                        clearable=False,
                                        className="mb-2"
                                    ),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Background:", className="fw-bold small"),
                                    dcc.Dropdown(
                                        id='bg-color-selector',
                                        options=[
                                            {'label': 'White', 'value': 'white'},
                                            {'label': 'Light Gray', 'value': 'light_gray'},
                                            {'label': 'Gray', 'value': 'gray'},
                                            {'label': 'Dark Gray', 'value': 'dark_gray'},
                                            {'label': 'Black', 'value': 'black'},
                                            {'label': 'Light Blue', 'value': 'light_blue'},
                                            {'label': 'Cream', 'value': 'cream'},
                                        ],
                                        value='black',
                                        clearable=False,
                                        className="mb-2"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Point Size:", className="fw-bold small"),
                                    dcc.Slider(
                                        id='point-size-slider',
                                        min=1, max=6, step=0.5, value=2,
                                        marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'},
                                    ),
                                ], width=6),
                            ]),
                        ])
                    ], className="mb-2"),

                    # 相机和显示选项
                    dbc.Card([
                        dbc.CardHeader("View"),
                        dbc.CardBody([
                            html.Label("Camera Preset:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='camera-selector',
                                options=[
                                    {'label': 'Front (-Z)', 'value': 'front'},
                                    {'label': 'Front 45 Up', 'value': 'front_45_up'},
                                    {'label': 'Side Right (+X)', 'value': 'side_right'},
                                    {'label': 'Side Left (-X)', 'value': 'side_left'},
                                    {'label': 'Top (+Y)', 'value': 'top'},
                                    {'label': 'Bottom (-Y)', 'value': 'bottom'},
                                    {'label': 'Isometric 1', 'value': 'isometric'},
                                    {'label': 'Isometric 2', 'value': 'isometric_2'},
                                    {'label': 'Isometric 3', 'value': 'isometric_3'},
                                    {'label': 'Isometric 4', 'value': 'isometric_4'},
                                    {'label': 'Back (+Z)', 'value': 'back'},
                                    {'label': 'Low Angle', 'value': 'low_angle'},
                                ],
                                value='isometric',
                                clearable=False,
                                className="mb-2"
                            ),
                            dbc.Checklist(
                                id='render-options',
                                options=[
                                    {'label': ' Show Axes', 'value': 'axes'},
                                    {'label': ' Show Grid', 'value': 'grid'},
                                ],
                                value=['axes'],
                                inline=True,
                            ),
                        ])
                    ], className="mb-2"),

                    # Zoom 控制
                    dbc.Card([
                        dbc.CardHeader("Zoom Control"),
                        dbc.CardBody([
                            html.Label("Zoom Level:", className="fw-bold small"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Input(id='zoom-value', type='number', value=1.5, step=0.1,
                                             min=0.5, max=10.0,
                                             className="form-control form-control-sm"),
                                ], width=6),
                                dbc.Col([
                                    html.Span("(smaller = closer)", className="small text-muted"),
                                ], width=6, className="d-flex align-items-center"),
                            ], className="mb-2"),
                            dcc.Slider(
                                id='zoom-slider',
                                min=0.5, max=5.0, step=0.1, value=1.5,
                                marks={0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2', 3: '3', 4: '4', 5: '5'},
                            ),
                            html.Div(id='current-camera-info', className="small text-muted mt-2"),
                        ])
                    ], className="mb-2"),

                    # 截图
                    dbc.Card([
                        dbc.CardHeader("Screenshot"),
                        dbc.CardBody([
                            dbc.Button("Save Current View", id="btn-save", color="primary",
                                       className="w-100 mb-2", size="sm"),
                            dbc.Button("Batch: 3 Views (F/S/T)", id="btn-batch-3", color="warning",
                                       className="w-100 mb-2", size="sm"),
                            dbc.Button("Batch: 4 Isometric", id="btn-batch-iso", color="warning",
                                       className="w-100", size="sm"),
                            html.Div(id='save-status', className="mt-2 small"),
                        ])
                    ], className="mb-2"),

                    # 文件信息
                    html.Div(id='file-info', className="small text-muted"),

                ], width=4),
            ]),

            # 隐藏存储
            dcc.Store(id='current-points', data=None),

        ], fluid=True)

    def _setup_callbacks(self):
        """设置回调函数"""

        # 类别变化 -> 更新文件列表
        @self.app.callback(
            [Output('file-selector', 'options'),
             Output('file-selector', 'value')],
            [Input('category-selector', 'value')]
        )
        def update_file_list(category):
            if not category:
                return [], None
            files = self._get_category_files(category)
            options = [{'label': f.name, 'value': str(f)} for f in files]
            return options, str(files[0]) if files else None

        # 同步zoom slider和input
        @self.app.callback(
            [Output('zoom-value', 'value'),
             Output('zoom-slider', 'value')],
            [Input('zoom-value', 'value'),
             Input('zoom-slider', 'value')],
            prevent_initial_call=True
        )
        def sync_zoom(input_val, slider_val):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'zoom-value':
                return no_update, input_val if input_val else 2.5
            else:
                return slider_val if slider_val else 2.5, no_update

        # 主更新回调
        @self.app.callback(
            [Output('pointcloud-3d', 'figure'),
             Output('pointcloud-3d', 'config'),
             Output('file-info', 'children'),
             Output('file-selector', 'value', allow_duplicate=True),
             Output('align-status', 'children'),
             Output('camera-selector', 'value', allow_duplicate=True),
             Output('current-camera-info', 'children')],
            [Input('file-selector', 'value'),
             Input('color-scheme-selector', 'value'),
             Input('solid-color-selector', 'value'),
             Input('bg-color-selector', 'value'),
             Input('point-size-slider', 'value'),
             Input('camera-selector', 'value'),
             Input('render-options', 'value'),
             Input('align-checkbox', 'value'),
             Input('btn-prev', 'n_clicks'),
             Input('btn-next', 'n_clicks'),
             Input('btn-align-front', 'n_clicks'),
             Input('btn-align-side', 'n_clicks'),
             Input('btn-align-top', 'n_clicks'),
             Input('zoom-value', 'value')],
            [State('category-selector', 'value')],
            prevent_initial_call=True
        )
        def update_display(file_path, color_scheme, solid_color, bg_color, point_size,
                          camera, render_opts, align_opts, prev_clicks, next_clicks,
                          align_front, align_side, align_top, zoom_value,
                          category):
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update

            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            # 处理导航
            files = self._get_category_files(category) if category else []
            current_file = file_path
            new_camera = camera

            if trigger == 'btn-prev' and files:
                try:
                    idx = next(i for i, f in enumerate(files) if str(f) == file_path)
                    idx = max(0, idx - 1)
                    current_file = str(files[idx])
                except StopIteration:
                    current_file = str(files[0])
            elif trigger == 'btn-next' and files:
                try:
                    idx = next(i for i, f in enumerate(files) if str(f) == file_path)
                    idx = min(len(files) - 1, idx + 1)
                    current_file = str(files[idx])
                except StopIteration:
                    current_file = str(files[0])

            # 快捷视角按钮
            if trigger == 'btn-align-front':
                new_camera = 'front'
            elif trigger == 'btn-align-side':
                new_camera = 'side_right'
            elif trigger == 'btn-align-top':
                new_camera = 'top'

            if not current_file:
                return go.Figure(), no_update, "No file selected", no_update, "", no_update

            # 加载点云
            points = self._load_ply(Path(current_file))

            # 获取标注并决定是否对齐
            annotation = self._get_annotation(current_file)
            do_align = 'align' in (align_opts or [])

            if do_align and annotation:
                points = self._align_points(points, annotation)
                align_status = html.Span([
                    html.Span("Aligned ", className="text-success fw-bold"),
                    html.Span(f"(Front={annotation.get('front_direction')}, Up={annotation.get('up_direction')})",
                             className="text-muted")
                ])
            elif annotation:
                align_status = html.Span([
                    html.Span("Has annotation ", className="text-info"),
                    html.Span("(check box to align)", className="text-muted")
                ])
            else:
                align_status = html.Span("No annotation", className="text-muted")

            # 渲染设置
            show_axes = 'axes' in (render_opts or [])
            show_grid = 'grid' in (render_opts or [])

            # Zoom值
            zoom = zoom_value if zoom_value else 1.5

            # 创建图形
            fig = self._create_3d_figure(
                points,
                color_scheme=color_scheme or 'height_blues',
                solid_color=solid_color or 'blue',
                bg_color=bg_color or 'white',
                point_size=point_size or 2,
                camera_preset=new_camera or 'isometric',
                show_axes=show_axes,
                show_grid=show_grid,
                zoom=zoom
            )

            # 文件信息
            file_path_obj = Path(current_file)
            file_name = file_path_obj.name
            file_stem = file_path_obj.stem
            category_name = file_path_obj.parent.name
            try:
                idx = next(i for i, f in enumerate(files) if str(f) == current_file) + 1
            except StopIteration:
                idx = 1
            info = f"{file_name} | {len(points)} points | [{idx}/{len(files)}]"

            # 动态更新下载文件名
            align_suffix = "_aligned" if do_align and annotation else ""
            download_filename = f"{category_name}_{file_stem}{align_suffix}"
            graph_config = {
                'displayModeBar': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': download_filename,
                    'height': 1200,
                    'width': 1200,
                    'scale': 3
                }
            }

            # 显示当前相机位置信息
            camera_info = f"Zoom: {zoom:.1f} (camera distance)"

            return fig, graph_config, info, current_file, align_status, new_camera, camera_info

        # 保存截图
        @self.app.callback(
            Output('save-status', 'children'),
            [Input('btn-save', 'n_clicks'),
             Input('btn-batch-3', 'n_clicks'),
             Input('btn-batch-iso', 'n_clicks')],
            [State('file-selector', 'value'),
             State('color-scheme-selector', 'value'),
             State('solid-color-selector', 'value'),
             State('bg-color-selector', 'value'),
             State('point-size-slider', 'value'),
             State('camera-selector', 'value'),
             State('render-options', 'value'),
             State('align-checkbox', 'value'),
             State('zoom-value', 'value')],
            prevent_initial_call=True
        )
        def save_screenshot(save_clicks, batch3_clicks, batch_iso_clicks,
                           file_path, color_scheme, solid_color, bg_color, point_size,
                           camera, render_opts, align_opts, zoom_value):
            ctx = callback_context
            if not ctx.triggered or not file_path:
                return ""

            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            # 加载点云
            points = self._load_ply(Path(file_path))

            # 对齐处理
            annotation = self._get_annotation(file_path)
            do_align = 'align' in (align_opts or [])
            if do_align and annotation:
                points = self._align_points(points, annotation)

            show_axes = 'axes' in (render_opts or [])
            show_grid = 'grid' in (render_opts or [])
            zoom = zoom_value if zoom_value else 1.5

            # 提取类别名和文件名
            file_path_obj = Path(file_path)
            file_stem = file_path_obj.stem  # 如 glass_box_0001
            category = file_path_obj.parent.name  # 如 glass_box

            # 构建文件名前缀：类别_文件名_对齐状态
            align_suffix = "_aligned" if do_align and annotation else ""
            file_prefix = f"{category}_{file_stem}{align_suffix}"

            saved_files = []

            if trigger == 'btn-save':
                # 单视角保存（使用当前zoom）
                fig = self._create_3d_figure(points, color_scheme, solid_color, bg_color,
                                            point_size, camera, show_axes, show_grid,
                                            zoom=zoom)
                out_path = self.output_dir / f"{file_prefix}_{camera}_z{zoom}.png"
                fig.write_image(str(out_path), width=1200, height=1200, scale=2)
                saved_files.append(out_path.name)

            elif trigger == 'btn-batch-3':
                # Front/Side/Top 三视图（使用当前zoom）
                for cam_name in ['front', 'side_right', 'top']:
                    fig = self._create_3d_figure(points, color_scheme, solid_color, bg_color,
                                                point_size, cam_name, show_axes, show_grid,
                                                zoom=zoom)
                    out_path = self.output_dir / f"{file_prefix}_{cam_name}.png"
                    fig.write_image(str(out_path), width=1200, height=1200, scale=2)
                    saved_files.append(out_path.name)

            elif trigger == 'btn-batch-iso':
                # 4个等轴测视角（使用当前zoom）
                for cam_name in ['isometric', 'isometric_2', 'isometric_3', 'isometric_4']:
                    fig = self._create_3d_figure(points, color_scheme, solid_color, bg_color,
                                                point_size, cam_name, show_axes, show_grid,
                                                zoom=zoom)
                    out_path = self.output_dir / f"{file_prefix}_{cam_name}.png"
                    fig.write_image(str(out_path), width=1200, height=1200, scale=2)
                    saved_files.append(out_path.name)

            return html.Div([
                html.Span("Saved: ", className="text-success fw-bold"),
                html.Br(),
                *[html.Span([f, html.Br()]) for f in saved_files]
            ])

    def run(self, port=8051):
        """运行Web服务器"""
        print("\n" + "="*70)
        print("  Point Cloud Screenshot Tool v2")
        print("="*70)
        print(f"\n  Data: {self.data_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Annotations: {len(self.annotations)} loaded")
        print(f"\n  Open in browser: http://localhost:{port}")
        print("\n  Usage:")
        print("    1. Select category and file")
        print("    2. Choose colors, background, point size")
        print("    3. Enable 'Apply alignment' if annotation exists")
        print("    4. Use quick view buttons (Front/Side/Top)")
        print("    5. Save single view or batch")
        print("="*70 + "\n")

        self.app.run(debug=False, host='0.0.0.0', port=port)


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Screenshot Tool")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply',
        help='Point cloud directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/pablo/ForwardNet-claude/screenshots',
        help='Screenshot output directory'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='/home/pablo/ForwardNet-claude/data_annotation/symmetry_annotations_v3.json',
        help='Annotation file for auto-alignment'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8051,
        help='Web server port'
    )

    args = parser.parse_args()

    viewer = PointCloudScreenshotViewer(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        annotation_file=Path(args.annotations) if args.annotations else None
    )

    viewer.run(port=args.port)


if __name__ == "__main__":
    main()

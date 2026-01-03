#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web版点云对称性与方向标注工具 v3

基于v2，新增功能：
- 添加 "After Correction Preview" 视图，显示矫正后的点云效果
- 端口改为8052

保留v2所有功能：
- 修改数据集为 full_mn40_normal_resampled_ply
- 修复按钮点击反馈问题
- 添加"没有正面"类型（5个对称类型）
- 添加明确的保存提示
- 自动保存标注
- 读取已有的 symmetry_annotations.json

使用方法：
    python annotate_symmetry_web_v3.py
    然后在浏览器中打开 http://localhost:8052

作者: Claude
创建日期: 2025-11-25
更新日期: 2025-11-27
版本: 3.0
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


def compute_correction_rotation(front_direction):
    """
    计算将点云从当前前向方向旋转到标准姿态(-Z为前)的旋转矩阵

    假设 upright 方向始终是 +Y（不变）
    只需要绕 Y 轴旋转来矫正前向方向
    """
    if not front_direction or front_direction == '-Z':
        return None  # 已对齐，无需旋转

    # 绕Y轴旋转的角度（从当前方向旋转到-Z）
    # 如果正面朝+X，需要绕Y轴旋转+90度使其朝向-Z
    # 如果正面朝-X，需要绕Y轴旋转-90度使其朝向-Z
    rotation_angles = {
        '+Z': np.pi,       # 180度
        '+X': np.pi/2,     # +90度（修正：原来是-90度，错误）
        '-X': -np.pi/2,    # -90度（修正：原来是+90度，错误）
        # +Y/-Y 作为前向方向是特殊情况，需要绕X轴旋转
    }

    if front_direction in rotation_angles:
        angle = rotation_angles[front_direction]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # 绕Y轴旋转矩阵
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float64)
        return R
    elif front_direction == '+Y':
        # 绕X轴旋转-90度
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        return R
    elif front_direction == '-Y':
        # 绕X轴旋转+90度
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64)
        return R

    return None


class WebSymmetryAnnotatorV3:
    """Web版对称性标注工具 v3 - 基于v2，添加矫正预览"""

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

        # 延迟保存标记
        self._pending_changes = False
        # 追踪当前会话中修改的文件（避免覆盖外部修改）
        self._modified_in_session = set()

        # 初始化Dash app (深色主题)
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

        # 添加深色Dropdown样式和键盘快捷键
        self.app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown深色样式 */
            .Select-control, .Select-menu-outer {
                background-color: #303030 !important;
                color: #fff !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #fff !important;
            }
            .Select-option {
                background-color: #303030 !important;
                color: #fff !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #444 !important;
            }
            .VirtualizedSelectOption {
                background-color: #303030 !important;
                color: #fff !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #444 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // CLI风格键盘输入状态机
            var inputState = 'idle';  // idle -> symmetry -> direction
            var selectedSymmetry = null;
            var symmetryNames = {1: '1个正面', 2: '2个正面', 3: '4个正面', 4: '旋转对称', 5: '无正面'};
            var directionNames = {1: '-Z', 2: '+Z', 3: '-X', 4: '+X', 5: '-Y', 6: '+Y', 7: '斜向', 8: '多正面'};

            function updatePrompt(text, color) {
                var prompt = document.getElementById('keyboard-prompt');
                if (prompt) {
                    prompt.innerHTML = '<code style="color: ' + (color || '#0f0') + '">' + text + '</code>';
                }
            }

            function resetState() {
                inputState = 'idle';
                selectedSymmetry = null;
                updatePrompt('等待输入... (按1-5选择对称类型)', '#888');
            }

            // 初始化提示
            setTimeout(resetState, 500);

            document.addEventListener('keydown', function(e) {
                // 如果在输入框中，不处理快捷键
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                var key = e.key.toLowerCase();
                var btn = null;

                // Esc取消当前输入
                if (key === 'escape') {
                    e.preventDefault();
                    resetState();
                    return;
                }

                // 导航键（任何状态下都可用）
                if (key === 'arrowleft' || key === 'a') {
                    e.preventDefault();
                    document.getElementById('btn-prev').click();
                    resetState();
                    return;
                }
                if (key === 'arrowright' || key === 'd') {
                    e.preventDefault();
                    document.getElementById('btn-next').click();
                    resetState();
                    return;
                }

                // S键保存
                if (key === 's') {
                    e.preventDefault();
                    document.getElementById('btn-manual-save').click();
                    updatePrompt('💾 已保存!', '#ff0');
                    setTimeout(resetState, 1000);
                    return;
                }

                // 状态机逻辑
                if (inputState === 'idle' || inputState === 'symmetry') {
                    // 选择对称类型 (1-5)
                    var symMap = {
                        '1': {btn: 'btn-k1', name: '1个正面'},
                        '2': {btn: 'btn-k2', name: '2个正面'},
                        '3': {btn: 'btn-k4', name: '4个正面'},
                        '4': {btn: 'btn-k0', name: '旋转对称'},
                        '5': {btn: 'btn-k-none', name: '无正面'}
                    };

                    if (symMap[key]) {
                        e.preventDefault();
                        selectedSymmetry = symMap[key];
                        document.getElementById(selectedSymmetry.btn).click();
                        inputState = 'direction';
                        updatePrompt('对称: ' + selectedSymmetry.name + ' ✓\\n→ 请选择方向 (1-8):', '#0ff');
                        return;
                    }
                }

                if (inputState === 'direction') {
                    // 选择方向 (1-8)
                    var dirMap = {
                        '1': 'btn-dir-z-',   // -Z
                        '2': 'btn-dir-z+',   // +Z
                        '3': 'btn-dir-x-',   // -X
                        '4': 'btn-dir-x+',   // +X
                        '5': 'btn-dir-y-',   // -Y
                        '6': 'btn-dir-y+',   // +Y
                        '7': 'btn-dir-oblique',  // 斜向
                        '8': 'btn-dir-multi'     // 多正面
                    };

                    if (dirMap[key]) {
                        e.preventDefault();
                        document.getElementById(dirMap[key]).click();
                        var dirName = directionNames[parseInt(key)];
                        updatePrompt('✅ 已标注: ' + selectedSymmetry.name + ' + ' + dirName, '#0f0');
                        setTimeout(function() {
                            // 自动跳转下一个
                            document.getElementById('btn-next').click();
                            resetState();
                        }, 300);
                        return;
                    }
                }
            });
        </script>
    </body>
</html>
'''
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

        # 先读取现有文件，智能合并
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing = json.load(f)
            # 智能合并：外部文件为基础，只覆盖当前会话中修改的标注
            merged = existing.copy()
            for file_path in self._modified_in_session:
                if file_path in self.annotations:
                    merged[file_path] = self.annotations[file_path]
            self.annotations = merged

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
            f.write(f"- **总标注数**: {len(self.annotations)}\n")
            f.write(f"- **需矫正数**: {len(need_correction)}\n\n")

            f.write("### 对称性分布\n\n")
            f.write("| 对称类型 | 数量 | 占比 |\n")
            f.write("|---------|------|------|\n")
            for k_name, count in sorted(stats.items()):
                pct = count / len(self.annotations) * 100 if self.annotations else 0
                f.write(f"| {k_name} | {count} | {pct:.1f}% |\n")

            f.write("\n### 正面方向分布\n\n")
            f.write("| 方向 | 数量 | 占比 | 状态 |\n")
            f.write("|------|------|------|------|\n")
            for direction, count in sorted(direction_stats.items()):
                pct = count / len(self.annotations) * 100 if self.annotations else 0
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
            '+Z': '绕Y轴旋转180°',
            '-X': '绕Y轴旋转+90°',
            '+X': '绕Y轴旋转-90°',
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

    def _get_global_stats(self):
        """计算全局标注统计"""
        stats = {
            -1: 0,  # 旋转对称
            0: 0,   # 无正面
            1: 0,   # 1个正面
            2: 0,   # 2个正面
            4: 0,   # 4个正面
        }
        for ann in self.annotations.values():
            k = ann.get('K')
            if k in stats:
                stats[k] += 1
        return stats

    def _build_stats_display(self):
        """构建统计显示组件"""
        stats = self._get_global_stats()
        total = len(self.annotations)

        labels = {
            -1: ('旋转对称', 'info'),
            0: ('无正面', 'secondary'),
            1: ('1个正面', 'primary'),
            2: ('2个正面', 'success'),
            4: ('4个正面', 'warning'),
        }

        badges = []
        for k in [-1, 0, 1, 2, 4]:
            name, color = labels[k]
            count = stats[k]
            badges.append(
                html.Span([
                    html.Strong(f"{name}: ", style={'fontSize': '11px'}),
                    dbc.Badge(f"{count}", color=color, className="me-2")
                ])
            )

        return html.Div([
            html.Span([html.Strong("总计: "), dbc.Badge(f"{total}", color="dark", className="me-3")]),
            *badges
        ], style={'fontSize': '12px'})

    def _build_layout(self):
        """构建Web界面"""
        self.app.layout = dbc.Container([
            html.H2("🏷️ 点云对称性与方向标注工具 v3", className="text-center my-3"),

            # 全局统计面板
            dbc.Card([
                dbc.CardBody([
                    html.H6("📊 全局标注统计", className="mb-2"),
                    html.Div(id='global-stats', children=self._build_stats_display())
                ], className="py-2")
            ], className="mb-2", color="light"),

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
                                className="mb-2",
                                style={'backgroundColor': '#303030', 'color': '#fff'}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Div(id='category-progress-info', className="mt-4", children=self._get_category_progress_display(self.current_category))
                        ], width=6),
                    ])
                ])
            ], className="mb-3"),

            dbc.Row([
                # 左侧：原始3D可视化
                dbc.Col([
                    html.H5("📍 原始点云", className="text-center mb-1"),
                    dcc.Graph(
                        id='pointcloud-3d',
                        style={'height': '50vh'},
                        config={'displayModeBar': True}
                    ),
                ], width=5),

                # 中间：矫正后预览
                dbc.Col([
                    html.H5("✅ 矫正后预览", className="text-center mb-1"),
                    dcc.Graph(
                        id='pointcloud-corrected',
                        style={'height': '50vh'},
                        config={'displayModeBar': True}
                    ),
                ], width=4),

                # 右侧：控制面板
                dbc.Col([
                    html.H5("📊 进度信息", className="mb-2"),
                    html.Div(id='progress-info', className="mb-2"),

                    html.Hr(),
                    html.H5("1️⃣ 对称类型", className="mb-2"),
                    html.Div([
                        dbc.Button("1个正面", id="btn-k1", color="primary", className="w-100 mb-1", size="sm"),
                        dbc.Button("2个正面", id="btn-k2", color="primary", className="w-100 mb-1", size="sm"),
                        dbc.Button("4个正面", id="btn-k4", color="primary", className="w-100 mb-1", size="sm"),
                        dbc.Button("旋转对称", id="btn-k0", color="info", className="w-100 mb-1", size="sm"),
                        dbc.Button("无正面", id="btn-k-none", color="secondary", className="w-100 mb-1", size="sm"),
                    ]),

                    html.Hr(),
                    html.H5("2️⃣ 正面方向", className="mb-2"),
                    html.P("理论上应为-Z:", className="small text-muted mb-1"),
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("-Z", id="btn-dir-z-", color="success", size="sm", className="me-1"),
                            dbc.Button("+Z", id="btn-dir-z+", color="success", size="sm", className="me-1"),
                            dbc.Button("-X", id="btn-dir-x-", color="success", size="sm"),
                        ], className="mb-1 w-100"),
                        dbc.ButtonGroup([
                            dbc.Button("+X", id="btn-dir-x+", color="success", size="sm", className="me-1"),
                            dbc.Button("-Y", id="btn-dir-y-", color="success", size="sm", className="me-1"),
                            dbc.Button("+Y", id="btn-dir-y+", color="success", size="sm"),
                        ], className="mb-1 w-100"),
                        dbc.ButtonGroup([
                            dbc.Button("斜向", id="btn-dir-oblique", color="info", size="sm", className="me-1"),
                            dbc.Button("多正面", id="btn-dir-multi", color="info", size="sm"),
                        ], className="mb-1 w-100"),
                    ]),

                    html.Hr(),
                    html.Div(id='annotation-status', className="mb-2"),

                    html.Hr(),
                    html.H5("3️⃣ 导航", className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("⬅️ 上一个", id="btn-prev", color="secondary", size="sm", className="me-1"),
                        dbc.Button("下一个 ➡️", id="btn-next", color="secondary", size="sm"),
                    ], className="mb-2 w-100"),

                    # 跳转功能
                    dbc.InputGroup([
                        dbc.Input(id='jump-input', type='number', placeholder='编号', size='sm', min=1),
                        dbc.Button("跳转", id="btn-jump", color="info", size="sm"),
                    ], className="mb-2", size="sm"),

                    dbc.Button("💾 手动保存", id="btn-manual-save", color="warning", className="w-100 mb-1", size="sm"),
                    html.Div(id='save-message', className="small text-success"),

                    # CLI风格键盘输入提示
                    html.Hr(),
                    html.Div([
                        html.H6("⌨️ 键盘输入", className="mb-2"),
                        html.Div(id='keyboard-prompt', children=[
                            html.Code("等待输入...", style={'color': '#888'})
                        ], className="mb-2 p-2", style={
                            'backgroundColor': '#1a1a1a',
                            'borderRadius': '4px',
                            'fontFamily': 'monospace',
                            'minHeight': '60px'
                        }),
                        html.Div([
                            html.Small("对称: 1=1面 2=2面 3=4面 4=旋转 5=无面", className="text-muted d-block"),
                            html.Small("方向: 1=-Z 2=+Z 3=-X 4=+X 5=-Y 6=+Y 7=斜 8=多", className="text-muted d-block"),
                            html.Small("导航: A/D或←/→  保存: S  取消: Esc", className="text-muted d-block"),
                        ], className="small"),
                    ], className="mb-2"),

                ], width=3),
            ]),

            # 隐藏的存储组件
            dcc.Store(id='current-index', data=self.start_index),
            dcc.Store(id='current-category', data=self.current_category),
            dcc.Store(id='pending-save', data=None),  # 延迟保存
            dcc.Interval(id='save-interval', interval=5000, n_intervals=0),  # 5秒自动保存

            # 键盘事件监听
            html.Div(id='keyboard-target', tabIndex='0', style={'outline': 'none'}),

        ], fluid=True)

    def _create_3d_figure(self, points, title="", show_corrected=False, front_direction=None):
        """创建3D点云图"""
        fig = go.Figure()

        # 如果需要显示矫正后的效果
        display_points = points.copy()
        if show_corrected and front_direction:
            R = compute_correction_rotation(front_direction)
            if R is not None:
                display_points = points @ R.T

        # 点云（采样显示，加快渲染）
        if len(display_points) > 5000:
            indices = np.random.choice(len(display_points), 5000, replace=False)
            points_display = display_points[indices]
        else:
            points_display = display_points

        fig.add_trace(go.Scatter3d(
            x=points_display[:, 0],
            y=points_display[:, 1],
            z=points_display[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_display[:, 1],  # Y是高度
                colorscale='Viridis',
                showscale=False,
            ),
            name='Point Cloud',
            hoverinfo='skip'
        ))

        # 坐标轴
        max_range = np.abs(display_points).max()
        axis_length = max_range * 0.6

        # X轴（红色）
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=8),
            text=['', '+X'],
            textposition='top center',
            textfont=dict(size=14, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Y轴（绿色）
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=8),
            text=['', '+Y (UP)'],
            textposition='top center',
            textfont=dict(size=14, color='green'),
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
            textfont=dict(size=14, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # -Z方向标记（前方）
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[-axis_length * 0.5],
            mode='text',
            text=['-Z (FRONT)'],
            textfont=dict(size=12, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # 负轴标记
        fig.add_trace(go.Scatter3d(
            x=[-axis_length*0.4, 0],
            y=[0, -axis_length*0.4],
            z=[0, 0],
            mode='text',
            text=['-X', '-Y'],
            textfont=dict(size=12, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color='#fff')),
            scene=dict(
                xaxis=dict(title='X', range=[-max_range, max_range], showgrid=True,
                          gridcolor='#444', color='#aaa', backgroundcolor='#1a1a1a'),
                yaxis=dict(title='Y', range=[-max_range, max_range], showgrid=True,
                          gridcolor='#444', color='#aaa', backgroundcolor='#1a1a1a'),
                zaxis=dict(title='Z', range=[-max_range, max_range], showgrid=True,
                          gridcolor='#444', color='#aaa', backgroundcolor='#1a1a1a'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=1, z=0)
                ),
                bgcolor='#1a1a1a'
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            paper_bgcolor='#222',
            plot_bgcolor='#222',
        )

        return fig

    def _get_current_annotation(self, idx):
        """获取当前样本的标注"""
        ply_path = self.ply_files[idx]
        rel_path = str(ply_path.relative_to(self.data_dir))
        return self.annotations.get(rel_path, None)

    def _save_current_annotation(self, idx, k_value, k_name, direction, save_to_disk=False):
        """保存当前标注到内存（可选保存到磁盘）"""
        ply_path = self.ply_files[idx]
        rel_path = str(ply_path.relative_to(self.data_dir))

        self.annotations[rel_path] = {
            'file': rel_path,
            'K': k_value,
            'symmetry_name': k_name,
            'front_direction': direction,
            'index': idx,
            'aligned': (direction == '-Z') or (k_value == -1)  # 旋转对称也视为已对齐
        }

        # 追踪当前会话修改的文件
        self._modified_in_session.add(rel_path)
        # 标记有待保存的更改
        self._pending_changes = True

        # 只在需要时保存到磁盘（减少IO）
        if save_to_disk:
            stats, dir_stats = self._save_annotations()
            self._pending_changes = False
        else:
            stats, dir_stats = None, None

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
             Output('pointcloud-corrected', 'figure'),
             Output('progress-info', 'children'),
             Output('annotation-status', 'children'),
             Output('save-message', 'children'),
             Output('current-index', 'data', allow_duplicate=True),
             Output('category-selector', 'options'),
             Output('category-progress-info', 'children', allow_duplicate=True),
             Output('global-stats', 'children', allow_duplicate=True)],
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
             Input('btn-dir-oblique', 'n_clicks'),
             Input('btn-dir-multi', 'n_clicks'),
             Input('btn-manual-save', 'n_clicks'),
             Input('btn-jump', 'n_clicks'),
             Input('save-interval', 'n_intervals')],
            [State('current-index', 'data'),
             State('jump-input', 'value')],
            prevent_initial_call=True
        )
        def handle_all_interactions(*args):
            ctx = callback_context
            current_idx = args[-2]  # State: current-index
            jump_value = args[-1]   # State: jump-input

            if current_idx is None:
                current_idx = 0

            save_msg = ""
            need_save_to_disk = False  # 是否需要保存到磁盘

            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                # 定时保存（后台）
                if button_id == 'save-interval':
                    if self._pending_changes:
                        self._save_annotations()
                        self._pending_changes = False
                        save_msg = "✅ 自动保存完成"

                # 导航
                elif button_id == 'btn-prev':
                    current_idx = max(0, current_idx - 1)
                elif button_id == 'btn-next':
                    current_idx = min(len(self.ply_files) - 1, current_idx + 1)
                elif button_id == 'btn-jump' and jump_value:
                    # 跳转到指定编号（1-based）
                    target_idx = int(jump_value) - 1
                    if 0 <= target_idx < len(self.ply_files):
                        current_idx = target_idx
                        save_msg = f"跳转到 #{jump_value}"
                    else:
                        save_msg = f"⚠️ 编号超出范围 (1-{len(self.ply_files)})"

                # 对称性标注
                k_map = {
                    'btn-k1': (1, '1个正面'),
                    'btn-k2': (2, '2个正面'),
                    'btn-k4': (4, '4个正面'),
                    'btn-k0': (-1, '旋转对称'),  # 旋转对称（含完全对称）
                    'btn-k-none': (0, '无正面'),  # 无正面（不规则物体）
                }

                # 方向标注
                dir_map = {
                    'btn-dir-z-': '-Z', 'btn-dir-z+': '+Z',
                    'btn-dir-x-': '-X', 'btn-dir-x+': '+X',
                    'btn-dir-y-': '-Y', 'btn-dir-y+': '+Y',
                    'btn-dir-oblique': 'OBLIQUE',  # 斜向（不与xyz轴对齐）
                    'btn-dir-multi': 'MULTI',      # 多正面（不相反）
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
                    self._save_annotations()
                    self._pending_changes = False
                    save_msg = f"💾 手动保存成功！共{len(self.annotations)}个标注"

            # 加载当前点云
            ply_path = self.ply_files[current_idx]
            points = self._load_ply(ply_path)

            # 获取当前标注
            current_ann = self._get_current_annotation(current_idx)
            front_direction = current_ann.get('front_direction') if current_ann else None

            # 原始点云图
            title = f"[{current_idx + 1}/{len(self.ply_files)}] {ply_path.name}"
            fig_original = self._create_3d_figure(points, title)

            # 矫正后预览图
            if front_direction and front_direction != '-Z':
                corrected_title = f"矫正后 (Front → -Z)"
                fig_corrected = self._create_3d_figure(points, corrected_title,
                                                        show_corrected=True,
                                                        front_direction=front_direction)
            elif front_direction == '-Z':
                fig_corrected = self._create_3d_figure(points, "✅ 已对齐 (无需矫正)")
            else:
                # 空白图，等待选择方向
                fig_corrected = go.Figure()
                fig_corrected.add_annotation(
                    text="请先选择正面方向",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="#888")
                )
                fig_corrected.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(text="等待标注...", font=dict(size=12, color='#fff')),
                    paper_bgcolor='#222',
                    plot_bgcolor='#222',
                )

            # 进度信息
            current_category_prog = self.category_progress[self.current_category]
            progress = html.Div([
                html.P([
                    html.Strong("📁 当前: "),
                    html.Code(ply_path.name, style={'fontSize': '10px'})
                ], className="mb-1"),
                html.P([
                    html.Strong(f"📊 {self.current_category}: "),
                    html.Span(f"{current_category_prog['annotated']}/{current_category_prog['total']}",
                             className="badge bg-primary me-1"),
                    html.Span(f"{current_category_prog['progress']:.1f}%",
                             className="badge bg-success")
                ], className="mb-0"),
            ])

            # 当前标注状态
            if current_ann and current_ann.get('front_direction'):
                status = dbc.Alert([
                    html.Strong("当前标注:"),
                    html.Hr(className="my-1"),
                    html.P([
                        "🏷️ ", html.Strong(current_ann.get('symmetry_name', f"K={current_ann['K']}"))
                    ], className="mb-1 small"),
                    html.P([
                        "🧭 ", html.Strong(current_ann['front_direction']),
                        html.Span(" ✅" if current_ann.get('aligned') else " ⚠️",
                                 className="ms-1")
                    ], className="mb-0 small"),
                ], color="success" if current_ann.get('aligned') else "warning", className="py-2")
            elif current_ann and current_ann.get('K') is not None:
                status = dbc.Alert([
                    html.P([
                        "⏳ ", html.Strong(current_ann.get('symmetry_name', f"K={current_ann['K']}")),
                    ], className="mb-0 small"),
                    html.P("请选择正面方向 ⬇️", className="mb-0 small text-muted")
                ], color="warning", className="py-2")
            else:
                status = dbc.Alert("未标注", color="light", className="py-2")

            # 更新类别选择器选项
            category_options = [
                {'label': f"{cat} ({self.category_progress[cat]['annotated']}/{self.category_progress[cat]['total']})",
                 'value': cat}
                for cat in sorted(self.categories.keys())
            ]

            # 更新当前类别进度显示
            category_progress_display = self._get_category_progress_display(self.current_category)

            # 更新全局统计
            global_stats_display = self._build_stats_display()

            return (fig_original, fig_corrected, progress, status, save_msg,
                    current_idx, category_options, category_progress_display, global_stats_display)

    def run(self, port=8052, host='127.0.0.1'):
        """运行Web服务器"""
        print("\n" + "="*70)
        print("  🏷️  Web版点云对称性与方向标注工具 v3")
        print("="*70)
        print(f"\n📂 数据集: {self.data_dir}")
        print(f"📊 样本数: {len(self.ply_files)}")
        print(f"💾 输出文件: {self.output_file}")
        print(f"\n🌐 在浏览器中打开: http://{host}:{port}")
        print("\n📖 使用说明:")
        print("  1. 左侧3D图显示原始点云")
        print("  2. 中间显示矫正后预览（选择方向后自动更新）")
        print("  3. 右侧先点击'对称类型'按钮")
        print("  4. 再点击'正面方向'按钮")
        print("  5. 两个都选择后会自动保存")
        print("\n💡 提示：")
        print("  - 矫正后预览显示应用旋转后的效果")
        print("  - 如果正面不是-Z方向，说明数据需要矫正！")
        print("="*70 + "\n")

        self.app.run(debug=False, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Web版对称性与方向标注工具 v3")
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
        default=8052,
        help='Web服务器端口'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='绑定地址 (使用 0.0.0.0 允许远程访问)'
    )

    args = parser.parse_args()

    annotator = WebSymmetryAnnotatorV3(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output)
    )

    annotator.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()

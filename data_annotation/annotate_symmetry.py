#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式点云对称性标注工具

功能：
- 3D可视化点云（matplotlib）
- 手动标注对称类型：1峰/2峰/4峰/完全对称
- 支持前进/后退浏览
- 自动保存标注结果到JSON
- 支持断点续标

使用方法：
    python annotate_symmetry.py --data_dir <点云目录>

快捷键：
    - 1: 标注为1峰（单正面）
    - 2: 标注为2峰（双正面，180°对称）
    - 4: 标注为4峰（四正面，90°对称）
    - S: 标注为完全对称
    - N: 下一个样本
    - P: 上一个样本
    - Q: 退出并保存

作者: Claude
创建日期: 2025-11-25
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用Tk后端，在WSL中更稳定
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D


class SymmetryAnnotator:
    """交互式点云对称性标注工具"""

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
        self.current_label = None

        # 对称类型映射
        self.symmetry_types = {
            '1': {'name': '1-peak (单正面)', 'K': 1},
            '2': {'name': '2-peak (双正面,180°)', 'K': 2},
            '4': {'name': '4-peak (四正面,90°)', 'K': 4},
            's': {'name': 'Fully Symmetric (完全对称)', 'K': 0},
        }

        # 初始化GUI
        self._init_gui()

    def _load_annotations(self):
        """加载已有标注"""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                annotations = json.load(f)
            print(f"[Annotation] Loaded {len(annotations)} existing annotations")
            return annotations
        else:
            print("[Annotation] No existing annotations found, starting fresh")
            return {}

    def _save_annotations(self):
        """保存标注到JSON"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        # 统计标注数量
        stats = {}
        for ann in self.annotations.values():
            k = ann['K']
            stats[k] = stats.get(k, 0) + 1

        print(f"[Saved] {len(self.annotations)}/{len(self.ply_files)} samples annotated")
        print(f"  Stats: K=1: {stats.get(1,0)}, K=2: {stats.get(2,0)}, "
              f"K=4: {stats.get(4,0)}, Symmetric: {stats.get(0,0)}")

    def _load_ply(self, ply_path: Path):
        """加载PLY点云文件"""
        try:
            # 简单的PLY加载器（假设是ASCII格式）
            with open(ply_path, 'r') as f:
                lines = f.readlines()

            # 查找header结束位置
            header_end = 0
            num_vertices = 0
            for i, line in enumerate(lines):
                if 'element vertex' in line:
                    num_vertices = int(line.split()[-1])
                if 'end_header' in line:
                    header_end = i + 1
                    break

            # 读取点云数据
            points = []
            for line in lines[header_end:header_end + num_vertices]:
                coords = line.strip().split()[:3]
                points.append([float(x) for x in coords])

            return np.array(points)

        except Exception as e:
            print(f"[Error] Failed to load {ply_path.name}: {e}")
            # 返回随机点云作为fallback
            return np.random.randn(1024, 3)

    def _init_gui(self):
        """初始化GUI界面"""
        self.fig = plt.figure(figsize=(14, 8))

        # 3D子图
        self.ax = self.fig.add_subplot(121, projection='3d')

        # 信息面板
        self.info_ax = self.fig.add_subplot(122)
        self.info_ax.axis('off')

        # 按钮位置 (left, bottom, width, height)
        btn_width = 0.15
        btn_height = 0.05
        btn_x_start = 0.55
        btn_y_start = 0.7
        btn_spacing = 0.08

        # 创建标注按钮
        self.btn_1peak = Button(
            plt.axes([btn_x_start, btn_y_start, btn_width, btn_height]),
            '1 - 单峰'
        )
        self.btn_1peak.on_clicked(lambda event: self._annotate('1'))

        self.btn_2peak = Button(
            plt.axes([btn_x_start, btn_y_start - btn_spacing, btn_width, btn_height]),
            '2 - 双峰(180°)'
        )
        self.btn_2peak.on_clicked(lambda event: self._annotate('2'))

        self.btn_4peak = Button(
            plt.axes([btn_x_start, btn_y_start - 2*btn_spacing, btn_width, btn_height]),
            '4 - 四峰(90°)'
        )
        self.btn_4peak.on_clicked(lambda event: self._annotate('4'))

        self.btn_symmetric = Button(
            plt.axes([btn_x_start, btn_y_start - 3*btn_spacing, btn_width, btn_height]),
            'S - 完全对称'
        )
        self.btn_symmetric.on_clicked(lambda event: self._annotate('s'))

        # 导航按钮
        self.btn_prev = Button(
            plt.axes([btn_x_start, btn_y_start - 5*btn_spacing, btn_width/2 - 0.01, btn_height]),
            '← Prev (P)'
        )
        self.btn_prev.on_clicked(lambda event: self._navigate(-1))

        self.btn_next = Button(
            plt.axes([btn_x_start + btn_width/2 + 0.01, btn_y_start - 5*btn_spacing,
                     btn_width/2 - 0.01, btn_height]),
            'Next (N) →'
        )
        self.btn_next.on_clicked(lambda event: self._navigate(1))

        # 保存并退出按钮
        self.btn_save = Button(
            plt.axes([btn_x_start, btn_y_start - 6*btn_spacing, btn_width, btn_height]),
            'Save & Quit (Q)'
        )
        self.btn_save.on_clicked(lambda event: self._quit())

        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # 显示第一个样本
        self._show_current()

    def _show_current(self):
        """显示当前点云"""
        if self.current_idx < 0:
            self.current_idx = 0
        if self.current_idx >= len(self.ply_files):
            self.current_idx = len(self.ply_files) - 1

        ply_path = self.ply_files[self.current_idx]
        rel_path = str(ply_path.relative_to(self.data_dir))

        # 加载点云
        points = self._load_ply(ply_path)

        # 清空3D子图
        self.ax.clear()

        # 绘制点云
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

        # 设置等比例和标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"Sample {self.current_idx + 1}/{len(self.ply_files)}\n{ply_path.name}")

        # 设置等比例显示
        max_range = np.abs(points).max()
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        # 更新信息面板
        self.info_ax.clear()
        self.info_ax.axis('off')

        # 检查当前是否已标注
        existing_label = self.annotations.get(rel_path, None)

        info_text = f"📁 文件: {ply_path.name}\n\n"
        info_text += f"📊 进度: {self.current_idx + 1}/{len(self.ply_files)}\n"
        info_text += f"✅ 已标注: {len(self.annotations)}\n\n"

        if existing_label:
            info_text += f"🏷️  当前标注: K={existing_label['K']}\n"
            info_text += f"   ({existing_label['name']})\n\n"
        else:
            info_text += "🏷️  当前标注: 未标注\n\n"

        info_text += "━━━━━━━━━━━━━━━━━\n\n"
        info_text += "🎯 对称性定义:\n\n"
        info_text += "1️⃣  1峰（单正面）\n"
        info_text += "   例：椅子、显示器、飞机\n\n"
        info_text += "2️⃣  2峰（180°对称）\n"
        info_text += "   例：门、电视柜\n\n"
        info_text += "4️⃣  4峰（90°对称）\n"
        info_text += "   例：玻璃盒、床头柜\n\n"
        info_text += "🔄 完全对称\n"
        info_text += "   例：球、圆锥、碗\n"

        self.info_ax.text(0.05, 0.95, info_text,
                         transform=self.info_ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         family='monospace')

        plt.draw()

    def _annotate(self, sym_type: str):
        """标注当前样本"""
        ply_path = self.ply_files[self.current_idx]
        rel_path = str(ply_path.relative_to(self.data_dir))

        sym_info = self.symmetry_types[sym_type]

        self.annotations[rel_path] = {
            'file': rel_path,
            'K': sym_info['K'],
            'name': sym_info['name'],
            'index': self.current_idx
        }

        print(f"[Annotated] {ply_path.name} → K={sym_info['K']} ({sym_info['name']})")

        # 自动保存
        self._save_annotations()

        # 自动跳转到下一个
        self._navigate(1)

    def _navigate(self, direction: int):
        """导航到上一个/下一个样本"""
        self.current_idx += direction

        # 循环处理
        if self.current_idx < 0:
            self.current_idx = len(self.ply_files) - 1
        elif self.current_idx >= len(self.ply_files):
            self.current_idx = 0

        self._show_current()

    def _on_key_press(self, event):
        """键盘快捷键"""
        if event.key == '1':
            self._annotate('1')
        elif event.key == '2':
            self._annotate('2')
        elif event.key == '4':
            self._annotate('4')
        elif event.key == 's':
            self._annotate('s')
        elif event.key == 'n':
            self._navigate(1)
        elif event.key == 'p':
            self._navigate(-1)
        elif event.key == 'q':
            self._quit()

    def _quit(self):
        """保存并退出"""
        self._save_annotations()
        print("\n[Exit] Annotation saved. Goodbye!")
        plt.close(self.fig)

    def run(self):
        """运行标注工具"""
        print("\n" + "="*60)
        print("  点云对称性标注工具")
        print("="*60)
        print("\n快捷键:")
        print("  1 - 标注为1峰（单正面）")
        print("  2 - 标注为2峰（双正面，180°对称）")
        print("  4 - 标注为4峰（四正面，90°对称）")
        print("  S - 标注为完全对称")
        print("  N - 下一个样本")
        print("  P - 上一个样本")
        print("  Q - 保存并退出")
        print("\n提示：鼠标拖动可以旋转点云查看不同角度")
        print("="*60 + "\n")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="点云对称性标注工具")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_2d_rotated_ply',
        help='点云文件目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/pablo/ForwardNet-claude/data/symmetry_annotations.json',
        help='标注输出文件（JSON格式）'
    )

    args = parser.parse_args()

    annotator = SymmetryAnnotator(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output)
    )

    annotator.run()


if __name__ == "__main__":
    main()

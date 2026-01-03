"""
D-Series Network Architecture - 完整训练流程
包含 Softmax 和 KL Divergence Loss
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle
import numpy as np
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'


class Block3D:
    """3D块"""
    def __init__(self, ax, x, y, w, h, d, color, alpha=1.0):
        self.ax = ax
        self.x, self.y = x, y
        self.w, self.h, self.d = w, h, d
        self.color = np.array(plt.cm.colors.to_rgb(color))
        self.alpha = alpha
        self._draw()

    def _draw(self):
        front = FancyBboxPatch(
            (self.x, self.y), self.w, self.h,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=self.color, edgecolor='#2c2c2c',
            linewidth=0.7, alpha=self.alpha, zorder=3
        )
        self.ax.add_patch(front)

        top_pts = [
            [self.x, self.y + self.h],
            [self.x + self.d * 0.6, self.y + self.h + self.d * 0.6],
            [self.x + self.w + self.d * 0.6, self.y + self.h + self.d * 0.6],
            [self.x + self.w, self.y + self.h]
        ]
        top = Polygon(top_pts, facecolor=self.color * 0.85,
                      edgecolor='#2c2c2c', linewidth=0.5, alpha=self.alpha, zorder=2)
        self.ax.add_patch(top)

        right_pts = [
            [self.x + self.w, self.y],
            [self.x + self.w + self.d * 0.6, self.y + self.d * 0.6],
            [self.x + self.w + self.d * 0.6, self.y + self.h + self.d * 0.6],
            [self.x + self.w, self.y + self.h]
        ]
        right = Polygon(right_pts, facecolor=self.color * 0.65,
                        edgecolor='#2c2c2c', linewidth=0.5, alpha=self.alpha, zorder=2)
        self.ax.add_patch(right)

        self.right_x = self.x + self.w + self.d * 0.6
        self.center_y = self.y + self.h / 2 + self.d * 0.3
        self.top_y = self.y + self.h + self.d * 0.6


def draw_arrow(ax, x1, y1, x2, y2, color='#4a4a4a', style='->', lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1, shrinkB=1),
                zorder=1)


def draw_rounded_box(ax, x, y, w, h, color, text, fontsize=8, text_color='white'):
    """圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                          facecolor=color, edgecolor='#333', linewidth=0.8, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
    return x + w


def draw_circle_op(ax, x, y, r, color, text, fontsize=8):
    """圆形操作符"""
    circle = Circle((x, y), r, facecolor=color, edgecolor='#333', linewidth=0.8, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    return x + r


def create_diagram():
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.2, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # 配色
    C = {
        'input': '#43A047',
        'sa': '#1E88E5',
        'global': '#FB8C00',
        'fc': '#E53935',
        'logits': '#7B1FA2',
        'softmax': '#00ACC1',
        'gt': '#558B2F',
        'loss': '#D32F2F',
        'prob': '#AB47BC'
    }

    # ========== 主网络流程 (y=1.5) ==========
    y0 = 1.5
    d3d = 0.12

    # INPUT
    x = 0
    b = Block3D(ax, x, y0, 0.35, 1.4, d3d, C['input'])
    ax.text(x + 0.175, y0 + 0.7, 'Input', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.175, y0 - 0.15, r'$N{\times}3$', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.25
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # SA1
    b = Block3D(ax, x, y0 + 0.1, 0.45, 1.15, d3d, C['sa'])
    ax.text(x + 0.225, y0 + 0.68, 'SA1', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.225, y0 - 0.05, '128', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.25
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # SA2
    b = Block3D(ax, x, y0 + 0.2, 0.5, 0.95, d3d, C['sa'])
    ax.text(x + 0.25, y0 + 0.68, 'SA2', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.25, y0 + 0.05, '32', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.25
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # SA3 (Global)
    b = Block3D(ax, x, y0 + 0.3, 0.55, 0.75, d3d, C['global'])
    ax.text(x + 0.275, y0 + 0.68, 'SA3', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.275, y0 + 0.15, 'global', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.3
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # FC1
    b = Block3D(ax, x, y0 + 0.38, 0.4, 0.6, d3d, C['fc'])
    ax.text(x + 0.2, y0 + 0.68, 'FC', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.2, y0 + 0.25, '512', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.22
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # FC2
    b = Block3D(ax, x, y0 + 0.42, 0.35, 0.52, d3d, C['fc'])
    ax.text(x + 0.175, y0 + 0.68, 'FC', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.175, y0 + 0.32, '256', ha='center', fontsize=6, color='#555')
    x = b.right_x + 0.22
    draw_arrow(ax, b.right_x + 0.03, b.center_y, x - 0.03, b.center_y)

    # Logits (output)
    b = Block3D(ax, x, y0 + 0.48, 0.3, 0.4, d3d, C['logits'])
    ax.text(x + 0.15, y0 + 0.68, '8', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=5)
    ax.text(x + 0.15, y0 + 0.35, 'logits', ha='center', fontsize=6, color='#555')
    logits_x = b.right_x
    logits_y = b.center_y

    # ========== Softmax 分支 ==========
    softmax_x = logits_x + 0.6
    softmax_y = y0 + 0.68

    draw_arrow(ax, logits_x + 0.05, logits_y, softmax_x - 0.25, softmax_y, C['softmax'])
    draw_rounded_box(ax, softmax_x - 0.25, softmax_y - 0.18, 0.65, 0.36,
                     C['softmax'], 'Softmax', fontsize=7)

    # 预测概率分布
    pred_x = softmax_x + 0.6
    draw_arrow(ax, softmax_x + 0.4, softmax_y, pred_x + 0.05, softmax_y, C['prob'])
    draw_rounded_box(ax, pred_x + 0.05, softmax_y - 0.22, 0.55, 0.44,
                     C['prob'], r'$\hat{p}$', fontsize=10)
    ax.text(pred_x + 0.32, softmax_y - 0.35, 'pred probs', ha='center', fontsize=6, color='#666')

    # ========== GT 分支 (下方) ==========
    gt_y = 0.3

    # GT 方向输入
    gt_input_x = 5.5
    draw_rounded_box(ax, gt_input_x, gt_y - 0.15, 0.8, 0.4, C['gt'], r'GT $\theta$', fontsize=8)
    ax.text(gt_input_x + 0.4, gt_y - 0.3, 'direction', ha='center', fontsize=6, color='#666')

    # cos 投影
    proj_x = gt_input_x + 1.1
    draw_arrow(ax, gt_input_x + 0.8, gt_y + 0.05, proj_x, gt_y + 0.05, '#666')
    draw_rounded_box(ax, proj_x, gt_y - 0.18, 0.9, 0.46, '#78909C',
                     r'cos proj', fontsize=7, text_color='white')
    ax.text(proj_x + 0.45, gt_y - 0.35, r'$\cos(\theta - \theta_i)$', ha='center', fontsize=6, color='#666')

    # GT Softmax
    gt_softmax_x = proj_x + 1.15
    draw_arrow(ax, proj_x + 0.9, gt_y + 0.05, gt_softmax_x, gt_y + 0.05, '#666')
    draw_rounded_box(ax, gt_softmax_x, gt_y - 0.18, 0.75, 0.46,
                     C['softmax'], r'Softmax', fontsize=7)
    ax.text(gt_softmax_x + 0.375, gt_y - 0.35, r'$\tau$=' + '5.0', ha='center', fontsize=6, color='#666')

    # GT 概率分布
    gt_prob_x = gt_softmax_x + 1.0
    draw_arrow(ax, gt_softmax_x + 0.75, gt_y + 0.05, gt_prob_x, gt_y + 0.05, C['gt'])
    draw_rounded_box(ax, gt_prob_x, gt_y - 0.22, 0.55, 0.54,
                     C['gt'], r'$p$', fontsize=10)
    ax.text(gt_prob_x + 0.275, gt_y - 0.4, 'GT probs', ha='center', fontsize=6, color='#666')

    # ========== KL Divergence Loss ==========
    loss_x = 11.8
    loss_y = 1.2

    # 从 pred probs 到 loss
    draw_arrow(ax, pred_x + 0.6, softmax_y - 0.1, loss_x - 0.05, loss_y + 0.25, C['loss'], lw=1.2)

    # 从 GT probs 到 loss
    draw_arrow(ax, gt_prob_x + 0.55, gt_y + 0.15, loss_x - 0.05, loss_y - 0.1, C['loss'], lw=1.2)

    # Loss 圆圈
    loss_circle = Circle((loss_x + 0.35, loss_y), 0.4, facecolor=C['loss'],
                          edgecolor='#8B0000', linewidth=1.5, zorder=5)
    ax.add_patch(loss_circle)
    ax.text(loss_x + 0.35, loss_y + 0.05, 'KL', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)
    ax.text(loss_x + 0.35, loss_y - 0.12, 'Loss', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=6)

    # Loss 公式
    ax.text(loss_x + 0.35, loss_y - 0.65, r'$\mathcal{L} = D_{KL}(p \| \hat{p})$',
            ha='center', fontsize=8, color='#666')

    # ========== Backbone 和 Head 标注 ==========
    ax.plot([0, 4.2], [3.15, 3.15], color=C['sa'], lw=2)
    ax.text(2.1, 3.25, 'PointNet++ Backbone', ha='center', fontsize=8, color=C['sa'], fontweight='bold')

    ax.plot([4.5, 7.2], [3.15, 3.15], color=C['fc'], lw=2)
    ax.text(5.85, 3.25, 'Classification Head', ha='center', fontsize=8, color=C['fc'], fontweight='bold')

    # ========== 标题 ==========
    ax.text(6, 4.1, 'D-Series: Discrete Direction Classification', fontsize=14,
            fontweight='bold', ha='center', color='#222')
    ax.text(6, 3.7, 'Training Pipeline with KL Divergence Loss', fontsize=10,
            ha='center', color='#666', style='italic')

    # ========== 图例 ==========
    leg_y = -0.85
    leg_items = [
        (C['input'], 'Point Cloud'),
        (C['sa'], 'Set Abstraction'),
        (C['global'], 'Global Pool'),
        (C['fc'], 'FC+BN+ReLU'),
        (C['softmax'], 'Softmax'),
        (C['gt'], 'Ground Truth'),
        (C['loss'], 'Loss'),
    ]
    for i, (c, t) in enumerate(leg_items):
        lx = 1.0 + i * 2.0
        rect = FancyBboxPatch((lx, leg_y), 0.18, 0.14, boxstyle="round,pad=0.01",
                               facecolor=c, edgecolor='#333', lw=0.5)
        ax.add_patch(rect)
        ax.text(lx + 0.25, leg_y + 0.07, t, va='center', fontsize=6.5, color='#333')

    plt.tight_layout(pad=0.3)
    return fig


if __name__ == '__main__':
    fig = create_diagram()
    out_dir = os.path.dirname(__file__)

    fig.savefig(os.path.join(out_dir, 'network_arch.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out_dir, 'network_arch.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(out_dir, 'network_arch.png'),
                format='png', bbox_inches='tight', dpi=300, facecolor='white')

    print("Generated: network_arch.pdf, network_arch.svg, network_arch.png")
    plt.close()

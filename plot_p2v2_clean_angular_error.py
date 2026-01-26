#!/usr/bin/env python3
"""
Generate publication-quality Angular Error curve for P2v2_Clean training.
Follows deep learning top-venue style (CVPR/ICCV/NeurIPS/ICLR).
"""

import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# Publication-quality settings (IEEE/CVPR style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (6, 4.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,  # Set to True if LaTeX is available
})


def parse_log_file(log_path, start_epoch=1):
    """Parse training log file and extract metrics."""
    data = {
        'epochs': [],
        'val_1f_error': [],
        'val_2f_error': [],
        'val_4f_error': [],
        'train_loss': [],
        'kappa': [],
        'lt10_pct': [],
    }

    current_epoch = None
    train_loss = None

    with open(log_path, 'r') as f:
        for line in f:
            # Match: Epoch X/Y
            epoch_match = re.match(r'Epoch (\d+)/\d+', line.strip())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Match train loss: Train loss: X.XXXX
            train_match = re.match(r'\s*Train loss: ([-\d.]+)', line.strip())
            if train_match:
                train_loss = float(train_match.group(1))
                continue

            # Match: Val 1f: XX.XX° (median=YY.YY°, <10°=ZZ.Z%, κ=W.W)
            val_1f_match = re.match(
                r'\s*Val 1f: ([\d.]+)°.*<10°=([\d.]+)%.*κ=([\d.]+)',
                line.strip()
            )
            if val_1f_match and current_epoch is not None and current_epoch >= start_epoch:
                if current_epoch not in data['epochs']:
                    data['epochs'].append(current_epoch)
                    data['val_1f_error'].append(float(val_1f_match.group(1)))
                    data['lt10_pct'].append(float(val_1f_match.group(2)))
                    data['kappa'].append(float(val_1f_match.group(3)))
                    if train_loss is not None:
                        data['train_loss'].append(train_loss)
                continue

            # Match: Val 2f: XX.XX°
            val_2f_match = re.match(r'\s*Val 2f: ([\d.]+)°', line.strip())
            if val_2f_match and current_epoch is not None and current_epoch >= start_epoch:
                if len(data['val_2f_error']) < len(data['epochs']):
                    data['val_2f_error'].append(float(val_2f_match.group(1)))
                continue

            # Match: Val 4f: XX.XX°
            val_4f_match = re.match(r'\s*Val 4f: ([\d.]+)°', line.strip())
            if val_4f_match and current_epoch is not None and current_epoch >= start_epoch:
                if len(data['val_4f_error']) < len(data['epochs']):
                    data['val_4f_error'].append(float(val_4f_match.group(1)))
                continue

    return data


def merge_data(data1, data2):
    """Merge two data dictionaries, preferring data1 for overlapping epochs."""
    merged = {key: list(data1[key]) for key in data1}

    for i, epoch in enumerate(data2['epochs']):
        if epoch not in merged['epochs']:
            merged['epochs'].append(epoch)
            for key in ['val_1f_error', 'val_2f_error', 'val_4f_error', 'train_loss', 'kappa', 'lt10_pct']:
                if i < len(data2[key]):
                    merged[key].append(data2[key][i])

    # Sort by epoch
    sorted_indices = np.argsort(merged['epochs'])
    for key in merged:
        merged[key] = [merged[key][i] for i in sorted_indices]

    return merged


def plot_angular_error(data, save_path):
    """Plot Angular Error curves in publication-quality style."""
    fig, ax = plt.subplots()

    epochs = np.array(data['epochs'])
    val_1f = np.array(data['val_1f_error'])
    val_2f = np.array(data['val_2f_error'][:len(epochs)])
    val_4f = np.array(data['val_4f_error'][:len(epochs)])

    # Color palette (colorblind-friendly)
    colors = {
        '1f': '#D55E00',  # vermillion
        '2f': '#0072B2',  # blue
        '4f': '#009E73',  # green
    }

    # Plot curves
    ax.plot(epochs, val_1f, color=colors['1f'], linewidth=2, label='1-Front', zorder=3)
    ax.plot(epochs, val_2f, color=colors['2f'], linewidth=2, label='2-Front', zorder=2)
    ax.plot(epochs, val_4f, color=colors['4f'], linewidth=2, label='4-Front', zorder=2)

    # Mark best points
    best_1f_idx = np.argmin(val_1f)
    best_1f_val = val_1f[best_1f_idx]
    ax.scatter([epochs[best_1f_idx]], [best_1f_val], color=colors['1f'], s=80,
               zorder=4, marker='*', edgecolors='black', linewidths=0.5)

    # Add annotation for best 1-front error
    ax.annotate(f'{best_1f_val:.2f}°',
                xy=(epochs[best_1f_idx], best_1f_val),
                xytext=(epochs[best_1f_idx] + 5, best_1f_val + 5),
                fontsize=9, color=colors['1f'],
                arrowprops=dict(arrowstyle='->', color=colors['1f'], lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Angular Error (°)')
    ax.set_xlim(0, max(epochs) + 2)
    ax.set_ylim(0, max(val_1f) * 1.1)

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))  # Also save PDF
    plt.close()
    print(f"Saved: {save_path}")


def plot_1f_error_only(data, save_path):
    """Plot only 1-Front Angular Error with training details."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = np.array(data['epochs'])
    val_1f = np.array(data['val_1f_error'])

    # Main color
    main_color = '#0072B2'  # blue

    # Plot with light fill
    ax.fill_between(epochs, 0, val_1f, alpha=0.15, color=main_color)
    ax.plot(epochs, val_1f, color=main_color, linewidth=2.2, label='Validation Angular Error', zorder=3)

    # Mark best point
    best_idx = np.argmin(val_1f)
    best_val = val_1f[best_idx]
    best_epoch = epochs[best_idx]

    ax.scatter([best_epoch], [best_val], color='#D55E00', s=120,
               zorder=5, marker='*', edgecolors='black', linewidths=0.8,
               label=f'Best: {best_val:.2f}° (Epoch {best_epoch})')

    # Add horizontal line at final value
    final_val = val_1f[-1]
    ax.axhline(y=final_val, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(max(epochs) + 1, final_val, f'{final_val:.1f}°',
            va='center', fontsize=9, color='gray')

    # Add vertical line at convergence
    # Find where error first drops below 15°
    converge_idx = np.where(val_1f < 15)[0]
    if len(converge_idx) > 0:
        converge_epoch = epochs[converge_idx[0]]
        ax.axvline(x=converge_epoch, color='#009E73', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(converge_epoch + 1, max(val_1f) * 0.85, f'<15° at Epoch {converge_epoch}',
                fontsize=9, color='#009E73', rotation=0)

    ax.set_xlabel('Epoch', fontweight='medium')
    ax.set_ylabel('Angular Error (°)', fontweight='medium')
    ax.set_xlim(0, max(epochs) + 5)
    ax.set_ylim(0, max(val_1f) * 1.05)

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {save_path}")


def plot_step_based(data, save_path, steps_per_epoch=794):
    """Plot Angular Error vs Training Steps (iterations)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = np.array(data['epochs'])
    val_1f = np.array(data['val_1f_error'])

    # Convert epochs to steps
    steps = epochs * steps_per_epoch

    main_color = '#0072B2'

    ax.fill_between(steps, 0, val_1f, alpha=0.12, color=main_color)
    ax.plot(steps, val_1f, color=main_color, linewidth=2, zorder=3)

    # Mark best
    best_idx = np.argmin(val_1f)
    best_val = val_1f[best_idx]
    best_step = steps[best_idx]

    ax.scatter([best_step], [best_val], color='#D55E00', s=100,
               zorder=5, marker='*', edgecolors='black', linewidths=0.7)

    ax.annotate(f'Best: {best_val:.2f}°\n({best_step//1000}k steps)',
                xy=(best_step, best_val),
                xytext=(best_step + 5000, best_val + 8),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Training Steps', fontweight='medium')
    ax.set_ylabel('Angular Error (°)', fontweight='medium')
    ax.set_xlim(0, max(steps) + 2000)
    ax.set_ylim(0, max(val_1f) * 1.05)

    # Format x-axis with K notation
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {save_path}")


def plot_weighted_error(data, save_path):
    """Plot weighted average Angular Error across all symmetry types."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = np.array(data['epochs'])
    val_1f = np.array(data['val_1f_error'])
    val_2f = np.array(data['val_2f_error'][:len(epochs)])
    val_4f = np.array(data['val_4f_error'][:len(epochs)])

    # Weights based on sample distribution (1f:180, 2f:60, 4f:40)
    total = 180 + 60 + 40
    w1, w2, w4 = 180/total, 60/total, 40/total

    # Weighted average
    weighted_error = w1 * val_1f + w2 * val_2f + w4 * val_4f

    main_color = '#0072B2'

    ax.fill_between(epochs, 0, weighted_error, alpha=0.15, color=main_color)
    ax.plot(epochs, weighted_error, color=main_color, linewidth=2.2, zorder=3)

    # Mark best
    best_idx = np.argmin(weighted_error)
    best_val = weighted_error[best_idx]
    best_epoch = epochs[best_idx]

    ax.scatter([best_epoch], [best_val], color='#D55E00', s=120,
               zorder=5, marker='*', edgecolors='black', linewidths=0.8,
               label=f'Best: {best_val:.2f}° (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontweight='medium')
    ax.set_ylabel('Weighted Angular Error (°)', fontweight='medium')
    ax.set_xlim(0, max(epochs) + 2)
    ax.set_ylim(0, max(weighted_error) * 1.05)

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    import os

    # Log file paths
    wandb_log = "/home/pablo/ForwardNet-claude/wandb/run-20251230_165848-yjthu57d/files/output.log"
    resume_log = "/home/pablo/ForwardNet-claude/training_clean_pipeline.log"

    # Output directory
    output_dir = "/home/pablo/ForwardNet-claude/paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing training logs...")

    # Parse both log files
    data1 = parse_log_file(wandb_log, start_epoch=1)
    data2 = parse_log_file(resume_log, start_epoch=40)

    # Merge data
    data = merge_data(data1, data2)

    # Limit to first 50 epochs
    max_epoch = 50
    indices = [i for i, e in enumerate(data['epochs']) if e <= max_epoch]
    for key in data:
        data[key] = [data[key][i] for i in indices if i < len(data[key])]

    print(f"Total epochs: {len(data['epochs'])}")
    print(f"Epoch range: {min(data['epochs'])} - {max(data['epochs'])}")
    print(f"Best 1-Front Error: {min(data['val_1f_error']):.2f}° at Epoch {data['epochs'][np.argmin(data['val_1f_error'])]}")

    # Generate plots
    print("\nGenerating publication-quality figures...")

    # 1. All symmetry types
    plot_angular_error(data, os.path.join(output_dir, "p2v2_clean_angular_error_all.png"))

    # 2. 1-Front only (main figure)
    plot_1f_error_only(data, os.path.join(output_dir, "p2v2_clean_angular_error_1f.png"))

    # 3. Step-based plot
    plot_step_based(data, os.path.join(output_dir, "p2v2_clean_angular_error_steps.png"))

    # 4. Weighted error
    plot_weighted_error(data, os.path.join(output_dir, "p2v2_clean_weighted_error.png"))

    print("\nDone! Figures saved to:", output_dir)

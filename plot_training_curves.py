#!/usr/bin/env python3
"""
Generate publication-quality training curves for D, DR, MF, and P2v2 series experiments.
Includes comprehensive metrics: loss, validation error, learning rate curves.
"""

import os
import re
import glob
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

# Use non-interactive backend
matplotlib.use('Agg')

# Publication-quality settings (IEEE/CVPR style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (5, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-friendly palette (Wong 2011)
COLORS = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
]


def parse_training_log(log_path):
    """Parse a training_log.txt file and extract metrics per epoch."""
    if not os.path.exists(log_path):
        return None

    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_error': [],
        'lr': [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(
                r'Epoch (\d+): train_loss=([\d.]+), val_loss=([\d.]+), val_error=([\d.]+)°, lr=([\d.]+)',
                line.strip()
            )
            if match:
                data['epochs'].append(int(match.group(1)))
                data['train_loss'].append(float(match.group(2)))
                data['val_loss'].append(float(match.group(3)))
                data['val_error'].append(float(match.group(4)))
                data['lr'].append(float(match.group(5)))

    if len(data['epochs']) == 0:
        return None

    return data


def parse_p2v2_output_log(log_path):
    """Parse a P2v2 output.log file from wandb."""
    if not os.path.exists(log_path):
        return None

    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_error_1f': [],
        'gate_acc': [],
        'kappa': [],
    }

    current_epoch = None
    train_loss = None

    with open(log_path, 'r') as f:
        for line in f:
            epoch_match = re.match(r'Epoch (\d+)/\d+', line.strip())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            train_match = re.match(r'\s*Train: loss=([-\d.]+)', line.strip())
            if train_match:
                train_loss = float(train_match.group(1))
                continue

            # Val:   loss=X.XXXX, gate_acc=Y.YYY, 1f_err=Z.Z°, 1f_κ=W.WW
            val_match = re.match(
                r'\s*Val:\s+loss=([-\d.]+), gate_acc=([\d.]+), 1f_err=([\d.]+)°, 1f_κ=([\d.]+)',
                line.strip()
            )
            if val_match and current_epoch is not None and train_loss is not None:
                data['epochs'].append(current_epoch)
                data['train_loss'].append(train_loss)
                data['val_loss'].append(float(val_match.group(1)))
                data['gate_acc'].append(float(val_match.group(2)))
                data['val_error_1f'].append(float(val_match.group(3)))
                data['kappa'].append(float(val_match.group(4)))
                train_loss = None

    if len(data['epochs']) == 0:
        return None

    return data


def get_experiment_label(dir_name):
    """Extract clean experiment label for legends."""
    # Map experiment names to clean labels
    label_map = {
        'D_8a': 'D-8 (baseline)',
        'D_8b': 'D-8b (variant)',
        'D_8c': 'D-8c (improved)',
        'D_8d': 'D-8d (final)',
        'D_16a': 'D-16a',
        'D_16b': 'D-16b',
        'DR_8a': 'DR-8a',
        'DR_8b': 'DR-8b',
        'DR_16a': 'DR-16a',
        'MF_1a_combined': 'Combined Loss',
        'MF_1b_reverse_kl': 'Reverse KL',
        'MF_1c_mu_only': 'Mean Only',
        'MF_1d_heavy_kappa': 'Heavy κ',
        'MF_1e_kappa_mu_only': 'κ + Mean',
        'P2v2_SoftGate': 'SoftGate',
        'P2v2_Clean': 'Clean',
    }

    parts = dir_name.rsplit('_', 2)
    exp_name = parts[0] if len(parts) >= 2 else dir_name
    return label_map.get(exp_name, exp_name)


def get_experiment_name(dir_name):
    """Extract experiment name from directory name."""
    parts = dir_name.rsplit('_', 2)
    return parts[0] if len(parts) >= 2 else dir_name


def plot_single(data, x_key, y_key, xlabel, ylabel, title, save_path, color=COLORS[0]):
    """Plot a single experiment curve."""
    fig, ax = plt.subplots()

    x = np.array(data[x_key])
    y = np.array(data[y_key])

    ax.plot(x, y, color=color, linewidth=1.5, zorder=2)

    # Mark best point
    if 'error' in y_key.lower() or 'loss' in y_key.lower():
        best_idx = np.argmin(y)
        ax.scatter([x[best_idx]], [y[best_idx]], color='red', s=50, zorder=3,
                   marker='*', label=f'Best: {y[best_idx]:.2f}')
        ax.legend(loc='upper right')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_comparison(datasets, x_key, y_key, xlabel, ylabel, title, save_path,
                   ylim=None, show_best=True):
    """Plot comparison of multiple experiments."""
    fig, ax = plt.subplots()

    for i, (name, data) in enumerate(datasets.items()):
        if data is None:
            continue
        x = np.array(data[x_key])
        y = np.array(data[y_key])
        label = get_experiment_label(data.get('full_name', name))
        ax.plot(x, y, color=COLORS[i % len(COLORS)], linewidth=1.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(loc='best', framealpha=0.9, edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_train_val_loss(data, name, save_path):
    """Plot training and validation loss on same figure."""
    fig, ax = plt.subplots()

    epochs = np.array(data['epochs'])
    train = np.array(data['train_loss'])
    val = np.array(data['val_loss'])

    ax.plot(epochs, train, color=COLORS[0], linewidth=1.5, label='Train Loss')
    ax.plot(epochs, val, color=COLORS[1], linewidth=1.5, label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{get_experiment_label(name)} - Train vs Val Loss')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_lr_schedule(data, name, save_path):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots()

    epochs = np.array(data['epochs'])
    lr = np.array(data['lr'])

    ax.plot(epochs, lr, color=COLORS[2], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{get_experiment_label(name)} - Learning Rate Schedule')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def process_d_series():
    """Process D series experiments."""
    print("\n=== D Series (Discrete Direction) ===")
    base_dir = "/home/pablo/ForwardNet-claude/checkpoints"
    results_dir = "/home/pablo/ForwardNet-claude/results/D_series"

    experiments = {}
    for d in sorted(glob.glob(os.path.join(base_dir, "D_*"))):
        name = os.path.basename(d)
        if name.startswith("DR_"):
            continue
        log_path = os.path.join(d, "training_log.txt")
        data = parse_training_log(log_path)
        if data:
            exp_name = get_experiment_name(name)
            # Keep the one with more epochs or better results
            if exp_name not in experiments or len(data['epochs']) > len(experiments[exp_name]['epochs']):
                experiments[exp_name] = data
                experiments[exp_name]['full_name'] = name

    # Individual plots
    for exp_name, data in experiments.items():
        # Validation Error
        plot_single(data, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                   f'{get_experiment_label(data["full_name"])} - Validation Error',
                   os.path.join(results_dir, f'{exp_name}_val_error.png'))

        # Train vs Val Loss
        plot_train_val_loss(data, data['full_name'],
                           os.path.join(results_dir, f'{exp_name}_loss.png'))

        # Learning Rate
        plot_lr_schedule(data, data['full_name'],
                        os.path.join(results_dir, f'{exp_name}_lr.png'))

    # Comparison plots
    if len(experiments) > 1:
        # Filter to main experiments (8c, 8d, 16a, 16b)
        main_exps = {k: v for k, v in experiments.items()
                     if k in ['D_8c', 'D_8d', 'D_16a', 'D_16b', 'D_8a']}

        if main_exps:
            plot_comparison(main_exps, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                           'D Series - Validation Error Comparison',
                           os.path.join(results_dir, 'D_comparison_val_error.png'))

            plot_comparison(main_exps, 'epochs', 'train_loss', 'Epoch', 'Training Loss',
                           'D Series - Training Loss Comparison',
                           os.path.join(results_dir, 'D_comparison_train_loss.png'))

    return experiments


def process_dr_series():
    """Process DR series experiments."""
    print("\n=== DR Series (Discrete + Regression) ===")
    base_dir = "/home/pablo/ForwardNet-claude/checkpoints"
    results_dir = "/home/pablo/ForwardNet-claude/results/DR_series"

    experiments = {}
    for d in sorted(glob.glob(os.path.join(base_dir, "DR_*"))):
        name = os.path.basename(d)
        if 'test' in name.lower():
            continue
        log_path = os.path.join(d, "training_log.txt")
        data = parse_training_log(log_path)
        if data:
            exp_name = get_experiment_name(name)
            if exp_name not in experiments or len(data['epochs']) > len(experiments[exp_name]['epochs']):
                experiments[exp_name] = data
                experiments[exp_name]['full_name'] = name

    for exp_name, data in experiments.items():
        plot_single(data, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                   f'{get_experiment_label(data["full_name"])} - Validation Error',
                   os.path.join(results_dir, f'{exp_name}_val_error.png'))

        plot_train_val_loss(data, data['full_name'],
                           os.path.join(results_dir, f'{exp_name}_loss.png'))

    if len(experiments) > 1:
        plot_comparison(experiments, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                       'DR Series - Validation Error Comparison',
                       os.path.join(results_dir, 'DR_comparison_val_error.png'))

    return experiments


def process_mf_series():
    """Process MF series experiments."""
    print("\n=== MF Series (von Mises Fisher) ===")
    base_dir = "/home/pablo/ForwardNet-claude/checkpoints"
    results_dir = "/home/pablo/ForwardNet-claude/results/MF_series"

    experiments = {}
    for d in sorted(glob.glob(os.path.join(base_dir, "MF_*"))):
        name = os.path.basename(d)
        log_path = os.path.join(d, "training_log.txt")
        data = parse_training_log(log_path)
        if data:
            exp_name = get_experiment_name(name)
            if exp_name not in experiments or len(data['epochs']) > len(experiments[exp_name]['epochs']):
                experiments[exp_name] = data
                experiments[exp_name]['full_name'] = name

    for exp_name, data in experiments.items():
        plot_single(data, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                   f'{get_experiment_label(data["full_name"])} - Validation Error',
                   os.path.join(results_dir, f'{exp_name}_val_error.png'))

        plot_train_val_loss(data, data['full_name'],
                           os.path.join(results_dir, f'{exp_name}_loss.png'))

    if len(experiments) > 1:
        plot_comparison(experiments, 'epochs', 'val_error', 'Epoch', 'Validation Error (°)',
                       'MF Series - Validation Error Comparison',
                       os.path.join(results_dir, 'MF_comparison_val_error.png'))

        plot_comparison(experiments, 'epochs', 'train_loss', 'Epoch', 'Training Loss',
                       'MF Series - Training Loss Comparison',
                       os.path.join(results_dir, 'MF_comparison_train_loss.png'))

    return experiments


def parse_p2v2_clean_log(log_path):
    """Parse P2v2_Clean output.log with different format."""
    if not os.path.exists(log_path):
        return None

    data = {
        'epochs': [],
        'val_error': [],
        'val_error_1f': [],
        'val_error_2f': [],
        'val_error_4f': [],
        'kappa': [],
        'lt10_1f': [],
    }

    current_epoch = None

    with open(log_path, 'r') as f:
        for line in f:
            # Match: Epoch X/Y
            epoch_match = re.match(r'Epoch (\d+)/\d+', line.strip())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Match: Val 1f: XX.XX° (median=YY.YY°, <10°=ZZ.Z%, κ=W.W)
            val_1f_match = re.match(
                r'\s*Val 1f: ([\d.]+)°.*<10°=([\d.]+)%.*κ=([\d.]+)',
                line.strip()
            )
            if val_1f_match and current_epoch is not None:
                data['epochs'].append(current_epoch)
                data['val_error_1f'].append(float(val_1f_match.group(1)))
                data['lt10_1f'].append(float(val_1f_match.group(2)))
                data['kappa'].append(float(val_1f_match.group(3)))
                continue

            # Match: Val 2f: XX.XX°
            val_2f_match = re.match(r'\s*Val 2f: ([\d.]+)°', line.strip())
            if val_2f_match:
                data['val_error_2f'].append(float(val_2f_match.group(1)))
                continue

            # Match: Val 4f: XX.XX°
            val_4f_match = re.match(r'\s*Val 4f: ([\d.]+)°', line.strip())
            if val_4f_match:
                data['val_error_4f'].append(float(val_4f_match.group(1)))
                continue

            # Match: [*] New best! val_error=XX.XX°
            best_match = re.match(r'\s*\[\*\] New best! val_error=([\d.]+)°', line.strip())
            if best_match:
                data['val_error'].append(float(best_match.group(1)))

    if len(data['epochs']) == 0:
        return None

    # Ensure all arrays are same length
    min_len = len(data['epochs'])
    for key in data:
        if len(data[key]) > min_len:
            data[key] = data[key][:min_len]

    return data


def process_p2v2_series():
    """Process P2v2 series experiments."""
    print("\n=== P2v2 Series (Pipeline v2) ===")
    base_dir = "/home/pablo/ForwardNet-claude/checkpoints"
    wandb_base = "/home/pablo/ForwardNet-claude/wandb"
    results_dir = "/home/pablo/ForwardNet-claude/results/P2v2_series"

    experiments = {}

    # Check checkpoint directories for wandb logs
    for d in sorted(glob.glob(os.path.join(base_dir, "P2v2_*")) +
                    glob.glob(os.path.join(base_dir, "P0_v2*"))):
        name = os.path.basename(d)

        wandb_logs = glob.glob(os.path.join(d, "wandb/*/files/output.log"))
        if wandb_logs:
            data = parse_p2v2_output_log(wandb_logs[0])
            if data:
                exp_name = get_experiment_name(name)
                experiments[exp_name] = data
                experiments[exp_name]['full_name'] = name

    # Also check main wandb directory for P2v2_Clean (run-20251230_165848)
    p2v2_clean_log = os.path.join(wandb_base, "run-20251230_165848-yjthu57d/files/output.log")
    if os.path.exists(p2v2_clean_log):
        data = parse_p2v2_clean_log(p2v2_clean_log)
        if data:
            experiments['P2v2_Clean'] = data
            experiments['P2v2_Clean']['full_name'] = 'P2v2_Clean_20251230_165848'

    for exp_name, data in experiments.items():
        if 'val_error_1f' in data and len(data['val_error_1f']) > 0:
            # 1-Front Error
            plot_single(data, 'epochs', 'val_error_1f', 'Epoch', '1-Front Error (°)',
                       f'{get_experiment_label(data["full_name"])} - 1-Front Error',
                       os.path.join(results_dir, f'{exp_name}_val_error_1f.png'))

            # Training Loss (if available)
            if 'train_loss' in data and len(data['train_loss']) > 0:
                plot_single(data, 'epochs', 'train_loss', 'Epoch', 'Training Loss',
                           f'{get_experiment_label(data["full_name"])} - Training Loss',
                           os.path.join(results_dir, f'{exp_name}_train_loss.png'))

            # 2-Front Error (if available)
            if 'val_error_2f' in data and len(data['val_error_2f']) > 0:
                plot_single(data, 'epochs', 'val_error_2f', 'Epoch', '2-Front Error (°)',
                           f'{get_experiment_label(data["full_name"])} - 2-Front Error',
                           os.path.join(results_dir, f'{exp_name}_val_error_2f.png'),
                           color=COLORS[1])

            # 4-Front Error (if available)
            if 'val_error_4f' in data and len(data['val_error_4f']) > 0:
                plot_single(data, 'epochs', 'val_error_4f', 'Epoch', '4-Front Error (°)',
                           f'{get_experiment_label(data["full_name"])} - 4-Front Error',
                           os.path.join(results_dir, f'{exp_name}_val_error_4f.png'),
                           color=COLORS[2])

            # Kappa (concentration parameter)
            if 'kappa' in data and len(data['kappa']) > 0:
                plot_single(data, 'epochs', 'kappa', 'Epoch', 'κ (Concentration)',
                           f'{get_experiment_label(data["full_name"])} - Concentration κ',
                           os.path.join(results_dir, f'{exp_name}_kappa.png'),
                           color=COLORS[3])

            # Gate Accuracy
            if 'gate_acc' in data and len(data['gate_acc']) > 0:
                plot_single(data, 'epochs', 'gate_acc', 'Epoch', 'Gate Accuracy',
                           f'{get_experiment_label(data["full_name"])} - Gate Accuracy',
                           os.path.join(results_dir, f'{exp_name}_gate_acc.png'),
                           color=COLORS[4])

            # <10° accuracy for 1-front (if available)
            if 'lt10_1f' in data and len(data['lt10_1f']) > 0:
                plot_single(data, 'epochs', 'lt10_1f', 'Epoch', '1-Front <10° Acc (%)',
                           f'{get_experiment_label(data["full_name"])} - 1-Front <10° Accuracy',
                           os.path.join(results_dir, f'{exp_name}_lt10_1f.png'),
                           color=COLORS[5])

    # Comparison plot for 1-front error
    if len(experiments) > 1:
        filtered = {k: v for k, v in experiments.items()
                   if 'val_error_1f' in v and len(v['val_error_1f']) > 0}
        if filtered:
            plot_comparison(filtered, 'epochs', 'val_error_1f', 'Epoch', '1-Front Error (°)',
                           'P2v2 Series - 1-Front Error Comparison',
                           os.path.join(results_dir, 'P2v2_comparison_val_error_1f.png'))

    return experiments


def create_summary_table(d_exp, dr_exp, mf_exp, p2v2_exp):
    """Create a summary of best results."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY - BEST RESULTS")
    print("="*70)

    print("\n{:<25} {:>12} {:>10} {:>10}".format(
        "Experiment", "Best Error", "Best Loss", "Epoch"))
    print("-"*60)

    print("\n--- D Series (Discrete Direction) ---")
    for name, data in sorted(d_exp.items()):
        best_err_idx = np.argmin(data['val_error'])
        best_loss_idx = np.argmin(data['val_loss'])
        print("{:<25} {:>11.2f}° {:>10.4f} {:>10}".format(
            name,
            data['val_error'][best_err_idx],
            data['val_loss'][best_loss_idx],
            data['epochs'][best_err_idx]))

    print("\n--- DR Series (Discrete + Regression) ---")
    for name, data in sorted(dr_exp.items()):
        best_err_idx = np.argmin(data['val_error'])
        best_loss_idx = np.argmin(data['val_loss'])
        print("{:<25} {:>11.2f}° {:>10.4f} {:>10}".format(
            name,
            data['val_error'][best_err_idx],
            data['val_loss'][best_loss_idx],
            data['epochs'][best_err_idx]))

    print("\n--- MF Series (von Mises Fisher) ---")
    for name, data in sorted(mf_exp.items()):
        best_err_idx = np.argmin(data['val_error'])
        best_loss_idx = np.argmin(data['val_loss'])
        print("{:<25} {:>11.2f}° {:>10.4f} {:>10}".format(
            name,
            data['val_error'][best_err_idx],
            data['val_loss'][best_loss_idx],
            data['epochs'][best_err_idx]))

    print("\n--- P2v2 Series (Pipeline v2) ---")
    for name, data in sorted(p2v2_exp.items()):
        if 'val_error_1f' in data and len(data['val_error_1f']) > 0:
            best_err_idx = np.argmin(data['val_error_1f'])
            print("{:<25} {:>11.2f}° {:>10} {:>10}".format(
                name,
                data['val_error_1f'][best_err_idx],
                "N/A",
                data['epochs'][best_err_idx]))


if __name__ == "__main__":
    print("="*70)
    print("Generating Publication-Quality Training Curves")
    print("="*70)

    d_exp = process_d_series()
    dr_exp = process_dr_series()
    mf_exp = process_mf_series()
    p2v2_exp = process_p2v2_series()

    create_summary_table(d_exp, dr_exp, mf_exp, p2v2_exp)

    print("\n" + "="*70)
    print("All plots saved to /home/pablo/ForwardNet-claude/results/")
    print("="*70)

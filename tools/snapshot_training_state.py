#!/usr/bin/env python3
"""
Snapshot Training State

从正在运行的训练中提取状态信息，保存到 JSON 文件。
重启后可以用这个信息恢复训练。

用法：
    python tools/snapshot_training_state.py checkpoints/P2v2_Clean_20251230_165848

这个脚本只读取文件，不会影响正在运行的训练。
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

import torch


def get_current_epoch_from_wandb(checkpoint_dir: Path) -> int:
    """从 wandb output.log 读取当前 epoch"""

    # 找到对应的 wandb run 目录
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        return -1

    # 从 checkpoint_dir 名字提取时间戳
    dir_name = checkpoint_dir.name  # e.g., P2v2_Clean_20251230_165848
    match = re.search(r'(\d{8}_\d{6})', dir_name)
    if not match:
        return -1

    timestamp = match.group(1)  # e.g., 20251230_165848

    # 找到匹配的 wandb run
    for run_dir in wandb_dir.iterdir():
        if timestamp.replace('_', '') in run_dir.name.replace('-', '').replace('_', ''):
            output_log = run_dir / "files" / "output.log"
            if output_log.exists():
                return parse_epoch_from_log(output_log)

    # 备选：直接搜索最新的 wandb run
    run_dirs = sorted(wandb_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for run_dir in run_dirs[:3]:
        output_log = run_dir / "files" / "output.log"
        if output_log.exists():
            epoch = parse_epoch_from_log(output_log)
            if epoch > 0:
                return epoch

    return -1


def parse_epoch_from_log(log_path: Path) -> int:
    """从 output.log 解析最新的 epoch"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # 找所有 "Epoch X/Y" 模式
        matches = re.findall(r'Epoch (\d+)/(\d+)', content)
        if matches:
            last_epoch = int(matches[-1][0])
            total_epochs = int(matches[-1][1])
            return last_epoch
    except Exception as e:
        print(f"  Warning: Could not parse {log_path}: {e}")

    return -1


def get_best_metrics_from_checkpoint(checkpoint_dir: Path) -> dict:
    """从 best.pth 读取 metrics"""
    best_pth = checkpoint_dir / "best.pth"
    if not best_pth.exists():
        return {}

    try:
        ckpt = torch.load(best_pth, map_location='cpu', weights_only=False)
        return {
            'best_epoch': ckpt.get('epoch', -1),
            'best_metrics': ckpt.get('metrics', {}),
            'has_optimizer': 'optimizer_state_dict' in ckpt,
            'has_model': 'model_state_dict' in ckpt,
        }
    except Exception as e:
        print(f"  Warning: Could not load {best_pth}: {e}")
        return {}


def get_config(checkpoint_dir: Path) -> dict:
    """读取训练配置"""
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def compute_scheduler_info(current_epoch: int, config: dict) -> dict:
    """计算 scheduler 相关信息"""
    total_epochs = config.get('p2v2_epochs', 100)
    lr_init = config.get('p2v2_lr', 1e-3)
    lr_min = 1e-6  # eta_min in code

    # CosineAnnealingLR 公式
    import math
    if current_epoch >= 0:
        lr_current = lr_min + (lr_init - lr_min) * (1 + math.cos(math.pi * current_epoch / total_epochs)) / 2
    else:
        lr_current = lr_init

    return {
        'total_epochs': total_epochs,
        'lr_init': lr_init,
        'lr_min': lr_min,
        'lr_current_estimated': lr_current,
        'scheduler_type': 'CosineAnnealingLR',
        'note': 'CosineAnnealingLR is deterministic - can be restored by calling step() current_epoch times'
    }


def get_loss_schedule_info(current_epoch: int, config: dict) -> dict:
    """计算 loss schedule 相关信息"""
    return {
        'cosine_weight': config.get('cosine_weight_init', 0.2) if current_epoch < config.get('cosine_schedule_epoch', 10) else config.get('cosine_weight_final', 0.1),
        'kappa_reg_start_epoch': config.get('kappa_reg_start_epoch', 6),
        'kappa_reg_ramp_epochs': config.get('kappa_reg_ramp_epochs', 10),
        'kappa_reg_weight_final': config.get('kappa_reg_weight_final', 0.02),
        'kappa_target_final': config.get('kappa_target_final', 5.0),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/snapshot_training_state.py <checkpoint_dir>")
        print("Example: python tools/snapshot_training_state.py checkpoints/P2v2_Clean_20251230_165848")
        sys.exit(1)

    checkpoint_dir = Path(sys.argv[1])
    if not checkpoint_dir.exists():
        print(f"Error: {checkpoint_dir} does not exist")
        sys.exit(1)

    print(f"Snapshotting training state from: {checkpoint_dir}")
    print("=" * 60)

    # 收集信息
    config = get_config(checkpoint_dir)
    current_epoch = get_current_epoch_from_wandb(checkpoint_dir)
    best_info = get_best_metrics_from_checkpoint(checkpoint_dir)
    scheduler_info = compute_scheduler_info(current_epoch, config)
    loss_schedule_info = get_loss_schedule_info(current_epoch, config)

    # 组装状态
    state = {
        'snapshot_time': datetime.now().isoformat(),
        'checkpoint_dir': str(checkpoint_dir),
        'current_epoch': current_epoch,
        'best_epoch': best_info.get('best_epoch', -1),
        'best_val_error': best_info.get('best_metrics', {}).get('val_error', None),
        'best_metrics': best_info.get('best_metrics', {}),
        'scheduler': scheduler_info,
        'loss_schedule': loss_schedule_info,
        'config': config,
        'checkpoint_contents': {
            'has_model': best_info.get('has_model', False),
            'has_optimizer': best_info.get('has_optimizer', False),
        },
        'resume_instructions': {
            'step1': 'Load model_state_dict from best.pth',
            'step2': 'Load optimizer_state_dict from best.pth',
            'step3': f'Create scheduler and call step() {best_info.get("best_epoch", 0)} times',
            'step4': f'Set start_epoch = {best_info.get("best_epoch", 0) + 1}',
            'step5': f'Set best_val_error = {best_info.get("best_metrics", {}).get("val_error", "inf")}',
        }
    }

    # 打印摘要
    print(f"Current epoch: {current_epoch}")
    print(f"Best epoch: {best_info.get('best_epoch', 'N/A')}")
    print(f"Best val_error: {best_info.get('best_metrics', {}).get('val_error', 'N/A')}")
    print(f"Estimated LR: {scheduler_info.get('lr_current_estimated', 'N/A'):.6f}")
    print()

    # 保存到文件
    output_path = checkpoint_dir / "training_state_snapshot.json"
    with open(output_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    print(f"State saved to: {output_path}")
    print()
    print("Resume instructions:")
    for step, instruction in state['resume_instructions'].items():
        print(f"  {step}: {instruction}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create a complete checkpoint for resuming training.
This script reconstructs the scheduler state based on the saved epoch.
"""

import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g., checkpoints/P2v2_Clean_20251230_165848)')
    parser.add_argument('--wandb_run_id', type=str, required=True,
                        help='WandB run ID to resume (e.g., yjthu57d)')
    parser.add_argument('--total_epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / 'best.pth'

    if not best_path.exists():
        print(f"ERROR: {best_path} not found!")
        return

    print(f"Loading checkpoint from: {best_path}")
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)

    epoch = ckpt['epoch']
    print(f"Checkpoint epoch: {epoch}")

    # Reconstruct scheduler state
    # CosineAnnealingLR state after `epoch` steps
    # The scheduler.step() is called epoch times (0 to epoch-1), so last_epoch = epoch
    scheduler_state = {
        'T_max': args.total_epochs,
        'eta_min': 1e-6,
        'base_lrs': [args.lr],
        'last_epoch': epoch,  # After epoch steps
        '_step_count': epoch + 1,
        '_last_lr': [args.lr],  # Will be recalculated
    }

    # Calculate the actual last_lr using cosine annealing formula
    import math
    T_max = args.total_epochs
    eta_min = 1e-6
    # After epoch steps, we're at position epoch in the cosine schedule
    if epoch >= T_max:
        last_lr = eta_min
    else:
        last_lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    scheduler_state['_last_lr'] = [last_lr]

    print(f"Reconstructed scheduler state:")
    print(f"  last_epoch: {epoch}")
    print(f"  last_lr: {last_lr:.6f}")

    # Get best_val_error from metrics
    best_val_error = ckpt.get('best_val_error', ckpt['metrics'].get('val_error', float('inf')))
    print(f"Best val error: {best_val_error:.2f}°")

    # Create complete checkpoint
    complete_ckpt = {
        'epoch': epoch,
        'model_state_dict': ckpt['model_state_dict'],
        'optimizer_state_dict': ckpt['optimizer_state_dict'],
        'scheduler_state_dict': scheduler_state,
        'best_val_error': best_val_error,
        'wandb_run_id': args.wandb_run_id,
        'metrics': ckpt['metrics'],
    }

    # Save as latest.pth
    latest_path = checkpoint_dir / 'latest.pth'
    torch.save(complete_ckpt, latest_path)
    print(f"\nSaved complete checkpoint to: {latest_path}")
    print(f"Resume will start from epoch {epoch + 1}")

    # Also update best.pth with complete info
    torch.save(complete_ckpt, best_path)
    print(f"Updated best.pth with complete info")

    print("\n" + "=" * 60)
    print("Done! You can now restart training with:")
    print(f"  bash resume_training.sh {checkpoint_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()

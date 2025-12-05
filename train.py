#!/usr/bin/env python
"""
Unified training script for orientation prediction.

Supports:
- Multiple backbones (PointNet++, DGCNN)
- Multiple prediction heads (Direct, VM_1peak, VM_2peak, VM_4peak, VM_4peak_learnable)
- Multiple loss functions (MSE, Cosine, NLL)
- WandB logging
- Checkpoint saving with full config

Usage:
    python train.py --config configs/baseline_direct.yaml
    python train.py --config configs/vm_4peak.yaml --backbone dgcnn

Author: Claude
Created: 2025-12-05
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Local imports
from core import build_model, get_loss
from core.losses import DirectRegressionLoss, VonMisesNLLLoss, MixtureVonMisesNLLLoss
from datasets import get_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Override config values from command line arguments."""
    if args.backbone:
        config['backbone']['type'] = args.backbone
    if args.head:
        config['head']['type'] = args.head
    if args.lr:
        config['training']['lr'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.k_filter:
        config['data']['k_filter'] = [int(k) for k in args.k_filter.split(',')]
    return config


def compute_loss(model_output: Dict, target: torch.Tensor, loss_fn: nn.Module,
                 head_type: str) -> torch.Tensor:
    """
    Compute loss based on head type.

    Args:
        model_output: Output dict from model
        target: GT direction (B, 2)
        loss_fn: Loss function module
        head_type: Type of prediction head

    Returns:
        Loss tensor
    """
    if head_type == 'direct':
        return loss_fn(model_output['direction'], target)
    elif head_type == 'vm_1peak':
        mu = model_output['mu'].squeeze(1)  # (B, 2)
        kappa = model_output['kappa'].squeeze(1)  # (B,)
        return loss_fn(mu, kappa, target)
    elif head_type in ['vm_2peak', 'vm_4peak', 'vm_4peak_learnable'] or 'peak' in head_type:
        mu = model_output['mu']  # (B, K, 2)
        kappa = model_output['kappa']  # (B, K)
        weights = model_output['weights']  # (B, K)
        return loss_fn(mu, kappa, weights, target)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def compute_angle_error(model_output: Dict, target: torch.Tensor,
                        head_type: str) -> torch.Tensor:
    """
    Compute angle error in degrees.

    For multi-peak heads, use the peak with minimum error.

    Args:
        model_output: Output dict from model
        target: GT direction (B, 2)
        head_type: Type of prediction head

    Returns:
        Angle error in degrees (B,)
    """
    if head_type == 'direct':
        pred = model_output['direction']  # (B, 2)
        # Angle between pred and target
        cos_angle = (pred * target).sum(dim=-1).clamp(-1, 1)
        angle_rad = torch.acos(cos_angle)
    else:
        # Multi-peak: find minimum error
        mu = model_output['mu']  # (B, K, 2)
        B, K, _ = mu.shape
        target_expanded = target.unsqueeze(1).expand(-1, K, -1)  # (B, K, 2)
        cos_angles = (mu * target_expanded).sum(dim=-1).clamp(-1, 1)  # (B, K)
        angles_rad = torch.acos(cos_angles)  # (B, K)
        angle_rad = angles_rad.min(dim=-1)[0]  # (B,)

    return torch.rad2deg(angle_rad)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                loss_fn: nn.Module, device: str, head_type: str) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_angle_error = 0.0
    num_samples = 0

    for batch in loader:
        points = batch['points'].to(device)
        target = batch['direction'].to(device)

        optimizer.zero_grad()
        output = model(points)

        loss = compute_loss(output, target, loss_fn, head_type)
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            angle_error = compute_angle_error(output, target, head_type)

        batch_size = points.size(0)
        total_loss += loss.item() * batch_size
        total_angle_error += angle_error.sum().item()
        num_samples += batch_size

    return {
        'loss': total_loss / num_samples,
        'angle_error': total_angle_error / num_samples
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: str, head_type: str) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_angle_error = 0.0
    num_samples = 0

    for batch in loader:
        points = batch['points'].to(device)
        target = batch['direction'].to(device)

        output = model(points)
        loss = compute_loss(output, target, loss_fn, head_type)
        angle_error = compute_angle_error(output, target, head_type)

        batch_size = points.size(0)
        total_loss += loss.item() * batch_size
        total_angle_error += angle_error.sum().item()
        num_samples += batch_size

    return {
        'loss': total_loss / num_samples,
        'angle_error': total_angle_error / num_samples
    }


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, metrics: Dict, config: Dict,
                    save_path: str, wandb_run_id: Optional[str] = None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': config,
        'metrics': metrics,
        'wandb_run_id': wandb_run_id,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train orientation prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--backbone', type=str, help='Override backbone type')
    parser.add_argument('--head', type=str, help='Override head type')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--k_filter', type=str, help='Filter by K values (comma-separated)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Load and update config
    config = load_config(args.config)
    config = update_config_from_args(config, args)

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create experiment directory
    exp_name = config.get('name', 'exp')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path('checkpoints') / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Initialize wandb
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb and config.get('wandb', {}).get('enabled', True):
        wandb_config = config.get('wandb', {})
        wandb_run = wandb.init(
            project=wandb_config.get('project', 'forwardnet'),
            name=f"{exp_name}_{timestamp}",
            config=config,
            dir=str(exp_dir),
        )
        print(f"WandB run: {wandb_run.url}")

    # Create dataloaders
    data_config = config.get('data', {})
    train_loader, val_loader, test_loader = get_dataloaders(
        annotation_file=data_config.get('annotation_file', 'data_annotation/symmetry_annotations.json'),
        data_dir=data_config.get('data_dir', 'data/full_mn40_normal_resampled_ply'),
        batch_size=config['training'].get('batch_size', 16),
        num_points=data_config.get('num_points', 1024),
        k_filter=data_config.get('k_filter'),
        align_pointcloud=data_config.get('align_pointcloud', True),
        augment_train=data_config.get('augment', True),
        num_rotations=data_config.get('num_rotations', 12),
        num_workers=data_config.get('num_workers', 4),
        seed=seed,
    )

    # Build model
    model = build_model(config)
    model = model.to(device)

    # Get head type for loss computation
    head_type = config['head'].get('type', 'direct')

    # Create loss function
    loss_config = config.get('loss', {'type': 'direct_mse'})
    loss_fn = get_loss(loss_config)

    # Create optimizer
    training_config = config.get('training', {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('lr', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4)
    )

    # Learning rate scheduler
    scheduler = None
    if training_config.get('scheduler', {}).get('enabled', False):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['scheduler'].get('step_size', 50),
            gamma=training_config['scheduler'].get('gamma', 0.5)
        )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('metrics', {}).get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    epochs = training_config.get('epochs', 100)
    print(f"\n{'='*60}")
    print(f"Training: {exp_name}")
    print(f"Backbone: {config['backbone'].get('type', 'pointnet++')}")
    print(f"Head: {head_type}")
    print(f"Loss: {loss_config.get('type', 'direct_mse')}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, head_type)

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, head_type)

        # Step scheduler
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - t0

        # Log
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_metrics['loss']:.4f}, Angle: {train_metrics['angle_error']:.2f}° | "
              f"Val Loss: {val_metrics['loss']:.4f}, Angle: {val_metrics['angle_error']:.2f}°")

        if wandb_run:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/angle_error': train_metrics['angle_error'],
                'val/loss': val_metrics['loss'],
                'val/angle_error': val_metrics['angle_error'],
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch,
                {'best_val_loss': best_val_loss, **val_metrics},
                config, str(exp_dir / 'best_model.pth'),
                wandb_run.id if wandb_run else None
            )
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % training_config.get('save_every', 50) == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {'val_loss': val_metrics['loss'], **val_metrics},
                config, str(exp_dir / f'checkpoint_epoch{epoch+1}.pth'),
                wandb_run.id if wandb_run else None
            )

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    # Load best model
    best_ckpt = torch.load(exp_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = evaluate(model, test_loader, loss_fn, device, head_type)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Angle Error: {test_metrics['angle_error']:.2f}°")

    if wandb_run:
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/angle_error': test_metrics['angle_error'],
        })
        wandb.finish()

    # Save final results
    results = {
        'best_val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'test_angle_error': test_metrics['angle_error'],
        'config': config,
    }
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {exp_dir}")


if __name__ == '__main__':
    main()

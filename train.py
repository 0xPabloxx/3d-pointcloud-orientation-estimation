#!/usr/bin/env python
"""
Unified training script for orientation prediction.

Supports:
- Multiple backbones (PointNet++, DGCNN)
- Multiple prediction heads (Direct, VM_1peak, VM_2peak, VM_4peak, VM_4peak_learnable, sCVAE)
- Multiple loss functions (MSE, Cosine, NLL, sCVAE Monte Carlo)
- WandB logging with visualization
- Checkpoint saving with full config and README

Usage:
    python train.py --config configs/baseline_direct.yaml
    python train.py --config configs/vm_4peak.yaml --backbone dgcnn
    python train.py --config configs/scvae_glassbox.yaml

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
    elif head_type in ['vm_2peak', 'vm_4peak', 'vm_4peak_learnable', 'scvae'] or 'peak' in head_type:
        # Multi-peak and sCVAE: both use (mu, kappa, weights) format
        mu = model_output['mu']  # (B, K, 2) or (B, N, 2) for sCVAE
        kappa = model_output['kappa']  # (B, K) or (B, N)
        weights = model_output['weights']  # (B, K) or (B, N)
        return loss_fn(mu, kappa, weights, target)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def compute_angle_error(model_output: Dict, target: torch.Tensor,
                        head_type: str, K_sym: torch.Tensor = None) -> torch.Tensor:
    """
    Compute symmetry-aware angle error in degrees.

    For symmetric objects (K_sym > 1), the GT direction can be rotated by
    360°/K_sym to get equivalent valid directions. We compute minimum error
    across all valid GT directions.

    For multi-peak heads, we also take minimum across all predicted peaks.

    Args:
        model_output: Output dict from model
        target: GT direction (B, 2)
        head_type: Type of prediction head
        K_sym: Symmetry order per sample (B,). If None, assumes K=1 (no symmetry).

    Returns:
        Angle error in degrees (B,)
    """
    B = target.shape[0]
    device = target.device

    # Get predictions
    if head_type == 'direct':
        pred = model_output['direction']  # (B, 2)
        pred = pred.unsqueeze(1)  # (B, 1, 2) for unified handling
    else:
        # Multi-peak heads output mu: (B, num_peaks, 2)
        pred = model_output['mu']  # (B, num_peaks, 2)

    num_peaks = pred.shape[1]

    # Generate all symmetric GT directions
    # For K_sym=4: GT, GT+90°, GT+180°, GT+270°
    if K_sym is None:
        K_sym = torch.ones(B, dtype=torch.long, device=device)

    # Find maximum K to create rotation matrices
    max_K = K_sym.max().item()

    # Compute GT angle
    gt_angle = torch.atan2(target[:, 1], target[:, 0])  # (B,)

    # Generate all symmetric angles: (B, max_K)
    k_offsets = torch.arange(max_K, device=device).float()  # (max_K,)
    angle_offsets = 2 * 3.14159265359 * k_offsets.unsqueeze(0) / K_sym.unsqueeze(1).float()  # (B, max_K)
    all_gt_angles = gt_angle.unsqueeze(1) + angle_offsets  # (B, max_K)

    # Convert to direction vectors
    all_gt_cos = torch.cos(all_gt_angles)  # (B, max_K)
    all_gt_sin = torch.sin(all_gt_angles)  # (B, max_K)
    all_gt = torch.stack([all_gt_cos, all_gt_sin], dim=-1)  # (B, max_K, 2)

    # Compute angle error between all (pred, gt) pairs
    # pred: (B, num_peaks, 2), all_gt: (B, max_K, 2)
    # Expand for broadcasting: pred (B, num_peaks, 1, 2), all_gt (B, 1, max_K, 2)
    pred_exp = pred.unsqueeze(2)  # (B, num_peaks, 1, 2)
    gt_exp = all_gt.unsqueeze(1)  # (B, 1, max_K, 2)

    # Cosine similarity
    cos_angles = (pred_exp * gt_exp).sum(dim=-1).clamp(-1, 1)  # (B, num_peaks, max_K)
    angles_rad = torch.acos(cos_angles)  # (B, num_peaks, max_K)

    # Mask out invalid K offsets (for samples with K < max_K)
    # Create mask: (B, max_K) where True means valid
    k_indices = torch.arange(max_K, device=device).unsqueeze(0).expand(B, -1)  # (B, max_K)
    valid_mask = k_indices < K_sym.unsqueeze(1)  # (B, max_K)
    valid_mask = valid_mask.unsqueeze(1).expand(-1, num_peaks, -1)  # (B, num_peaks, max_K)

    # Set invalid angles to large value
    angles_rad = torch.where(valid_mask, angles_rad, torch.tensor(float('inf'), device=device))

    # Take minimum across all peaks and all symmetric GTs
    angle_rad = angles_rad.min(dim=-1)[0].min(dim=-1)[0]  # (B,)

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
        K_sym = batch['K'].to(device) if 'K' in batch else None

        optimizer.zero_grad()
        output = model(points)

        loss = compute_loss(output, target, loss_fn, head_type)
        loss.backward()
        optimizer.step()

        # Metrics (symmetry-aware angle error)
        with torch.no_grad():
            angle_error = compute_angle_error(output, target, head_type, K_sym)

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
             device: str, head_type: str, collect_stats: bool = False) -> Dict[str, float]:
    """Evaluate model.

    Args:
        collect_stats: If True, collect kappa/mu statistics for debugging
    """
    model.eval()
    total_loss = 0.0
    total_angle_error = 0.0
    num_samples = 0

    # For collecting kappa/mu stats
    all_kappas = []
    all_mu_norms = []

    for batch in loader:
        points = batch['points'].to(device)
        target = batch['direction'].to(device)
        K_sym = batch['K'].to(device) if 'K' in batch else None

        output = model(points)
        loss = compute_loss(output, target, loss_fn, head_type)
        angle_error = compute_angle_error(output, target, head_type, K_sym)

        batch_size = points.size(0)
        total_loss += loss.item() * batch_size
        total_angle_error += angle_error.sum().item()
        num_samples += batch_size

        # Collect kappa/mu stats for multi-peak heads
        if collect_stats and 'kappa' in output:
            all_kappas.append(output['kappa'].cpu())
            all_mu_norms.append(output['mu'].norm(dim=-1).cpu())

    result = {
        'loss': total_loss / num_samples,
        'angle_error': total_angle_error / num_samples
    }

    # Add kappa/mu statistics
    if collect_stats and all_kappas:
        kappas = torch.cat(all_kappas, dim=0)  # (N, num_peaks) or (N,)
        mu_norms = torch.cat(all_mu_norms, dim=0)
        result['kappa_mean'] = kappas.mean().item()
        result['kappa_std'] = kappas.std().item()
        result['kappa_min'] = kappas.min().item()
        result['kappa_max'] = kappas.max().item()
        result['mu_norm_mean'] = mu_norms.mean().item()
        result['mu_norm_std'] = mu_norms.std().item()

    return result


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


def write_experiment_readme(exp_dir: Path, config: Dict, final_metrics: Dict = None):
    """
    Write a README file describing the experiment.

    Args:
        exp_dir: Experiment directory
        config: Training configuration
        final_metrics: Optional final test metrics
    """
    head_type = config.get('head', {}).get('type', 'unknown')
    backbone_type = config.get('backbone', {}).get('type', 'unknown')
    loss_type = config.get('loss', {}).get('type', 'unknown')
    k_filter = config.get('data', {}).get('k_filter', 'all')

    # Head-specific description
    head_descriptions = {
        'direct': 'Direct regression of (cos θ, sin θ) direction vector',
        'vm_1peak': 'Single-peak von Mises distribution (μ, κ)',
        'vm_2peak': '2-peak mixture of von Mises (180° symmetry)',
        'vm_4peak': '4-peak mixture of von Mises (90° symmetry, fixed equal weights)',
        'vm_4peak_learnable': '4-peak mixture of von Mises (90° symmetry, learnable weights)',
        'scvae': 'sCVAE: Stochastic CVAE for implicit multi-peak prediction via noise sampling',
    }

    readme_content = f"""# Experiment: {config.get('name', 'unnamed')}

## Model Architecture
- **Backbone**: {backbone_type}
- **Head**: {head_type}
- **Description**: {head_descriptions.get(head_type, 'Custom head')}

## Head Configuration
"""

    head_config = config.get('head', {})
    for k, v in head_config.items():
        if k != 'type':
            readme_content += f"- {k}: {v}\n"

    readme_content += f"""
## Loss Function
- **Type**: {loss_type}
"""

    loss_config = config.get('loss', {})
    for k, v in loss_config.items():
        if k != 'type':
            readme_content += f"- {k}: {v}\n"

    readme_content += f"""
## Data
- **K filter**: {k_filter}
- **Num points**: {config.get('data', {}).get('num_points', 1024)}
- **Augmentation**: {config.get('data', {}).get('augment', True)}
- **Num rotations**: {config.get('data', {}).get('num_rotations', 12)}

## Training
- **Epochs**: {config.get('training', {}).get('epochs', 100)}
- **Batch size**: {config.get('training', {}).get('batch_size', 16)}
- **Learning rate**: {config.get('training', {}).get('lr', 0.001)}
- **Weight decay**: {config.get('training', {}).get('weight_decay', 0.0001)}
"""

    if final_metrics:
        readme_content += f"""
## Results
- **Test Loss**: {final_metrics.get('test_loss', 'N/A'):.4f}
- **Test Angle Error**: {final_metrics.get('test_angle_error', 'N/A'):.2f}°
"""

    readme_content += f"""
## Files
- `config.yaml`: Full training configuration
- `best_model.pth`: Best model checkpoint (by validation loss)
- `results.json`: Final metrics and configuration
- `visualizations/`: (if applicable) Prediction visualizations

## Usage
```python
from core import load_model
model = load_model('{exp_dir}/best_model.pth')
output = model(points)  # points: (B, N, 3)
# output['mu']: (B, K, 2) predicted directions
# output['kappa']: (B, K) concentration parameters
# output['weights']: (B, K) mixture weights
```
"""

    with open(exp_dir / 'README.md', 'w') as f:
        f.write(readme_content)


def visualize_predictions_for_wandb(model, val_loader, device, head_type, num_samples=8):
    """
    Create visualization for wandb logging.

    Returns wandb.Image objects for logging.
    """
    try:
        import wandb
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
    except ImportError:
        return {}

    model.eval()

    # Get a batch
    batch = next(iter(val_loader))
    points = batch['points'][:num_samples].to(device)
    targets = batch['direction'][:num_samples]

    with torch.no_grad():
        outputs = model(points)

    # Create polar plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    for i in range(min(num_samples, 8)):
        ax = axes[i]

        gt_angle = np.arctan2(targets[i, 1].item(), targets[i, 0].item())

        if head_type == 'direct':
            pred = outputs['direction'][i].cpu()
            pred_angle = np.arctan2(pred[1].item(), pred[0].item())
            ax.scatter([pred_angle], [1], s=100, c='blue', label='Pred')
        else:
            mu = outputs['mu'][i].cpu()  # (K, 2) or (N, 2)
            kappa = outputs['kappa'][i].cpu()
            pred_angles = np.arctan2(mu[:, 1].numpy(), mu[:, 0].numpy())
            sizes = np.clip(kappa.numpy() * 10, 20, 200)
            ax.scatter(pred_angles, np.ones_like(pred_angles), s=sizes, c='blue', alpha=0.6, label='Pred')

        ax.scatter([gt_angle], [1], s=100, c='red', marker='*', label='GT')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_title(f'Sample {i}', fontsize=10)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'{head_type} Predictions', fontsize=14)
    plt.tight_layout()

    # Convert to wandb image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    wandb_img = wandb.Image(Image.open(buf), caption=f'{head_type} predictions')
    plt.close(fig)

    return {'predictions': wandb_img}


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

    # Collect training history for CSV export
    training_history = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, head_type)

        # Validate (with stats collection for multi-peak heads)
        collect_stats = head_type in ['vm_1peak', 'vm_2peak', 'vm_4peak', 'vm_4peak_learnable', 'scvae']
        val_metrics = evaluate(model, val_loader, loss_fn, device, head_type, collect_stats=collect_stats)

        # Step scheduler
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        log_msg = (f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                   f"Train Loss: {train_metrics['loss']:.4f}, Angle: {train_metrics['angle_error']:.2f}° | "
                   f"Val Loss: {val_metrics['loss']:.4f}, Angle: {val_metrics['angle_error']:.2f}°")

        # Add kappa/mu stats if available
        if 'kappa_mean' in val_metrics:
            log_msg += f" | κ: {val_metrics['kappa_mean']:.1f}±{val_metrics['kappa_std']:.1f}"
            log_msg += f" (min:{val_metrics['kappa_min']:.1f}, max:{val_metrics['kappa_max']:.1f})"

        print(log_msg)

        # Save to history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_angle_error': train_metrics['angle_error'],
            'val_loss': val_metrics['loss'],
            'val_angle_error': val_metrics['angle_error'],
            'lr': current_lr,
            'epoch_time': epoch_time,
        }
        if 'kappa_mean' in val_metrics:
            epoch_data.update({
                'kappa_mean': val_metrics['kappa_mean'],
                'kappa_std': val_metrics['kappa_std'],
                'kappa_min': val_metrics['kappa_min'],
                'kappa_max': val_metrics['kappa_max'],
                'mu_norm_mean': val_metrics['mu_norm_mean'],
                'mu_norm_std': val_metrics['mu_norm_std'],
            })
        training_history.append(epoch_data)

        # Save training history to CSV periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            import pandas as pd
            df = pd.DataFrame(training_history)
            df.to_csv(exp_dir / 'training_history.csv', index=False)

        if wandb_run:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/angle_error': train_metrics['angle_error'],
                'val/loss': val_metrics['loss'],
                'val/angle_error': val_metrics['angle_error'],
                'lr': current_lr,
            }
            if 'kappa_mean' in val_metrics:
                log_dict.update({
                    'val/kappa_mean': val_metrics['kappa_mean'],
                    'val/kappa_std': val_metrics['kappa_std'],
                    'val/kappa_min': val_metrics['kappa_min'],
                    'val/kappa_max': val_metrics['kappa_max'],
                })
            wandb.log(log_dict)

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
        # Log final visualization before finishing
        viz_imgs = visualize_predictions_for_wandb(model, val_loader, device, head_type)
        if viz_imgs:
            wandb.log(viz_imgs)
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

    # Write README with experiment description
    write_experiment_readme(exp_dir, config, {
        'test_loss': test_metrics['loss'],
        'test_angle_error': test_metrics['angle_error']
    })

    print(f"\nResults saved to: {exp_dir}")
    print(f"README written to: {exp_dir / 'README.md'}")


if __name__ == '__main__':
    main()

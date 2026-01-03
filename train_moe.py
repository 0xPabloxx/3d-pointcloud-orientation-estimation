#!/usr/bin/env python3
"""
Training script for ProbabilisticOrientationNet (Mixture of Experts)

This is the 2-step approach:
    Step 1: Pre-trained SymmetryClassifier (97.9% accuracy) - already done
    Step 2: Train direction experts with frozen classifier as gate

Features:
    - Loads pre-trained classifier checkpoint
    - Freezes classifier (gate) during training
    - Uses class-balanced sampling for minority classes (2-front, 4-front)
    - 12x rotation augmentation
    - Logs to WandB

Usage:
    python train_moe.py --exp_name 2-step_MoE_v1
    python train_moe.py --epochs 100 --batch_size 32

Author: Claude
Created: 2025-12-28
"""

import os
import sys
import argparse
import json
import time
import math
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.moe_dataset import MoEDataset, get_dataloaders, collate_fn, LABEL_NAMES
from models import (
    ProbabilisticOrientationNet,
    MaskedExpertLoss,
    get_final_pdf,
    get_peak_predictions
)


def angular_error(pred_angle: float, gt_angle: float, symmetry_order: int = 1) -> float:
    """
    Compute angular error considering symmetry.

    Uses cosine distance which naturally handles circular angle arithmetic.
    For K-fold symmetry, checks all K equivalent directions.

    Args:
        pred_angle: Predicted angle in radians
        gt_angle: Ground truth angle in radians
        symmetry_order: 1 for 1-front, 2 for 2-front, 4 for 4-front

    Returns:
        Angular error in range [0, 2] where 0=perfect, 2=opposite direction
    """
    if symmetry_order == 1:
        return 1 - math.cos(pred_angle - gt_angle)
    elif symmetry_order == 2:
        # Check 0° and 180° offsets
        return min(
            1 - math.cos(pred_angle - gt_angle),
            1 - math.cos(pred_angle - gt_angle - math.pi)
        )
    elif symmetry_order == 4:
        # Check 0°, 90°, 180°, 270° offsets
        return min(
            1 - math.cos(pred_angle - gt_angle - offset)
            for offset in [0, math.pi/2, math.pi, 3*math.pi/2]
        )
    else:
        return 0.0  # Non-directional class

# Import the classifier class that matches the saved checkpoint
# This is from train_symmetry_classifier.py
import train_symmetry_classifier
from train_symmetry_classifier import SymmetryClassifier


def _deterministic_fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Deterministic FPS - always starts from point 0."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    # Fixed starting point for determinism
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


# Monkey-patch the classifier module's FPS to be deterministic
# This ensures the frozen classifier produces consistent results
train_symmetry_classifier.farthest_point_sample = _deterministic_fps


class ClassifierWrapper(nn.Module):
    """Wrapper to make SymmetryClassifier compatible with ProbabilisticOrientationNet.

    The original classifier only takes (points) as input, but
    ProbabilisticOrientationNet calls classifier(x, upright_vec).
    This wrapper ignores the upright_vec argument.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, points, upright_vec=None):
        # Ignore upright_vec, just pass points to the classifier
        return self.classifier(points)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.exp_name:
            self.exp_name = f"{args.exp_name}_{timestamp}"
        else:
            self.exp_name = f"2-step_MoE_{timestamp}"

        self.output_dir = Path('checkpoints') / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_points=args.num_points,
            num_rotations=args.num_rotations,
            val_num_rotations=args.val_num_rotations,
            num_workers=args.num_workers,
            use_balanced_sampler=args.balanced_sampler,
            align_pointcloud=args.align_pointcloud,
            seed=args.seed
        )

        # Load pre-trained classifier
        print(f"\nLoading classifier from: {args.classifier_checkpoint}")
        base_classifier = SymmetryClassifier(num_classes=5)
        ckpt = torch.load(args.classifier_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            base_classifier.load_state_dict(ckpt['model_state_dict'])
        else:
            base_classifier.load_state_dict(ckpt)

        # Wrap classifier to match ProbabilisticOrientationNet interface
        classifier = ClassifierWrapper(base_classifier).to(self.device)
        print("Classifier loaded successfully")

        # Create MoE model
        self.model = ProbabilisticOrientationNet(
            classifier=classifier,
            backbone_dim=args.backbone_dim,
            expert_hidden_dim=args.expert_hidden_dim,
            kappa_min=args.kappa_min,
            kappa_max=args.kappa_max,
            freeze_classifier=args.freeze_classifier
        ).to(self.device)

        # Loss function with Soft Gate-Aware training (P2v2)
        self.criterion = MaskedExpertLoss(
            classification_weight=args.classification_weight,
            use_gate_weighting=not args.no_gate_weighting,
            p_dir_threshold=args.p_dir_threshold,
            gamma=args.gamma,
            cosine_weight_init=args.cosine_weight_init,
            cosine_weight_final=args.cosine_weight_final,
            cosine_schedule_epoch=args.cosine_schedule_epoch,
            kappa_reg_weight_final=args.kappa_reg_weight_final,
            kappa_target_final=args.kappa_target_final,
            kappa_reg_start_epoch=args.kappa_reg_start_epoch,
            kappa_reg_ramp_epochs=args.kappa_reg_ramp_epochs
        )

        # Optimizer (only train non-frozen params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Resume from checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)

        # WandB
        self.use_wandb = args.wandb
        if self.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=self.exp_name,
                config=vars(args),
                dir=str(self.output_dir),
                resume="allow" if args.resume else None
            )
            wandb.watch(self.model, log='gradients', log_freq=100)

        # Print summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Classifier frozen: {args.freeze_classifier}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Balanced sampler: {args.balanced_sampler}")
        print(f"Output dir: {self.output_dir}")
        if self.use_wandb:
            import wandb
            print(f"WandB: {wandb.run.url}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_angles = batch['gt_angle'].to(self.device)
            gt_labels = batch['gt_label'].to(self.device)
            upright_vec = batch['upright_vec'].to(self.device)

            # Forward
            output = self.model(points, upright_vec)

            # Loss
            loss_dict = self.criterion(output, gt_angles, gt_labels)

            # Backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Record metrics
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    metrics[k].append(v.item())
                else:
                    metrics[k].append(v)

            # Classification accuracy (from frozen gate)
            pred_class = output['weights'].argmax(dim=1)
            acc = (pred_class == gt_labels).float().mean().item()
            metrics['gate_acc'].append(acc)

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"loss={loss_dict['loss'].item():.4f}, "
                      f"gate_acc={acc:.3f}")

        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        per_class_metrics = defaultdict(lambda: defaultdict(list))

        # Use weighted accumulation for loss metrics (weight by sample count)
        # This prevents outlier batches with few samples from dominating the average
        total_loss_sum = 0.0
        total_count = 0
        loss_1front_sum, count_1front_total = 0.0, 0
        loss_2front_sum, count_2front_total = 0.0, 0
        loss_4front_sum, count_4front_total = 0.0, 0

        # Gate accuracy: weight by batch size
        gate_acc_sum = 0.0
        gate_total = 0

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_angles = batch['gt_angle'].to(self.device)
            gt_labels = batch['gt_label'].to(self.device)
            upright_vec = batch['upright_vec'].to(self.device)

            batch_size = points.size(0)

            # Forward
            output = self.model(points, upright_vec)

            # Loss - accumulate weighted by sample count
            loss_dict = self.criterion(output, gt_angles, gt_labels)

            # Accumulate total loss (weighted by directional sample count)
            batch_dir_count = loss_dict['count_1front'] + loss_dict['count_2front'] + loss_dict['count_4front']
            if batch_dir_count > 0:
                total_loss_sum += loss_dict['loss'].item() * batch_dir_count
                total_count += batch_dir_count

            # Accumulate per-class losses (weighted by per-class sample count)
            if loss_dict['count_1front'] > 0:
                loss_1front_sum += loss_dict['loss_1front'].item() * loss_dict['count_1front']
                count_1front_total += loss_dict['count_1front']
            if loss_dict['count_2front'] > 0:
                loss_2front_sum += loss_dict['loss_2front'].item() * loss_dict['count_2front']
                count_2front_total += loss_dict['count_2front']
            if loss_dict['count_4front'] > 0:
                loss_4front_sum += loss_dict['loss_4front'].item() * loss_dict['count_4front']
                count_4front_total += loss_dict['count_4front']

            # Gate accuracy (weighted by batch size)
            pred_class = output['weights'].argmax(dim=1)
            acc = (pred_class == gt_labels).float().sum().item()
            gate_acc_sum += acc
            gate_total += batch_size

            # Direction accuracy for directional classes (0, 1, 2)
            predictions = get_peak_predictions(output, gt_labels)
            for i in range(len(gt_labels)):
                label = gt_labels[i].item()
                if label < 3:  # Directional class
                    pred_angle = predictions['predicted_angles'][i].item()
                    gt_angle = gt_angles[i].item()

                    # Angular error considering symmetry
                    symmetry_order = [1, 2, 4][label]
                    error = angular_error(pred_angle, gt_angle, symmetry_order)
                    per_class_metrics[label]['angle_error'].append(error)

                    # Also track kappa for each class
                    pred_kappa = predictions['confidence'][i].item()
                    per_class_metrics[label]['kappa'].append(pred_kappa)

        # Compute weighted averages
        avg_metrics = {}
        avg_metrics['loss'] = total_loss_sum / total_count if total_count > 0 else 0.0
        avg_metrics['loss_1front'] = loss_1front_sum / count_1front_total if count_1front_total > 0 else 0.0
        avg_metrics['loss_2front'] = loss_2front_sum / count_2front_total if count_2front_total > 0 else 0.0
        avg_metrics['loss_4front'] = loss_4front_sum / count_4front_total if count_4front_total > 0 else 0.0
        avg_metrics['gate_acc'] = gate_acc_sum / gate_total if gate_total > 0 else 0.0

        # Include sample counts for debugging
        avg_metrics['count_1front'] = count_1front_total
        avg_metrics['count_2front'] = count_2front_total
        avg_metrics['count_4front'] = count_4front_total

        # Per-class angle errors and kappa stats
        all_errors = []
        for label in range(3):
            if per_class_metrics[label]['angle_error']:
                errors = per_class_metrics[label]['angle_error']
                all_errors.extend(errors)
                avg_metrics[f'angle_error_{LABEL_NAMES[label]}'] = np.mean(errors)
                avg_metrics[f'angle_error_{LABEL_NAMES[label]}_median'] = np.median(errors)
            if per_class_metrics[label]['kappa']:
                kappas = per_class_metrics[label]['kappa']
                avg_metrics[f'kappa_{LABEL_NAMES[label]}_mean'] = np.mean(kappas)
                avg_metrics[f'kappa_{LABEL_NAMES[label]}_std'] = np.std(kappas)

        # Overall angle error
        if all_errors:
            avg_metrics['angle_error_mean'] = np.mean(all_errors)
            avg_metrics['angle_error_median'] = np.median(all_errors)

        return avg_metrics

    @torch.no_grad()
    def test(self) -> dict:
        """Run final evaluation on test set."""
        self.model.eval()
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        # Use weighted accumulation for loss metrics (same as validate)
        total_loss_sum = 0.0
        total_count = 0
        loss_1front_sum, count_1front_total = 0.0, 0
        loss_2front_sum, count_2front_total = 0.0, 0
        loss_4front_sum, count_4front_total = 0.0, 0

        for batch in self.test_loader:
            points = batch['points'].to(self.device)
            gt_angles = batch['gt_angle'].to(self.device)
            gt_labels = batch['gt_label'].to(self.device)
            upright_vec = batch['upright_vec'].to(self.device)

            output = self.model(points, upright_vec)
            loss_dict = self.criterion(output, gt_angles, gt_labels)

            # Accumulate weighted losses
            batch_dir_count = loss_dict['count_1front'] + loss_dict['count_2front'] + loss_dict['count_4front']
            if batch_dir_count > 0:
                total_loss_sum += loss_dict['loss'].item() * batch_dir_count
                total_count += batch_dir_count

            if loss_dict['count_1front'] > 0:
                loss_1front_sum += loss_dict['loss_1front'].item() * loss_dict['count_1front']
                count_1front_total += loss_dict['count_1front']
            if loss_dict['count_2front'] > 0:
                loss_2front_sum += loss_dict['loss_2front'].item() * loss_dict['count_2front']
                count_2front_total += loss_dict['count_2front']
            if loss_dict['count_4front'] > 0:
                loss_4front_sum += loss_dict['loss_4front'].item() * loss_dict['count_4front']
                count_4front_total += loss_dict['count_4front']

            # Gate accuracy per class
            pred_class = output['weights'].argmax(dim=1)
            for i in range(len(gt_labels)):
                label = gt_labels[i].item()
                per_class_total[label] += 1
                if pred_class[i].item() == label:
                    per_class_correct[label] += 1

        # Compute weighted averages
        avg_metrics = {}
        avg_metrics['loss'] = total_loss_sum / total_count if total_count > 0 else 0.0
        avg_metrics['loss_1front'] = loss_1front_sum / count_1front_total if count_1front_total > 0 else 0.0
        avg_metrics['loss_2front'] = loss_2front_sum / count_2front_total if count_2front_total > 0 else 0.0
        avg_metrics['loss_4front'] = loss_4front_sum / count_4front_total if count_4front_total > 0 else 0.0

        # Per-class gate accuracy
        print("\nTest Results - Gate Accuracy by Class:")
        for label in range(5):
            if per_class_total[label] > 0:
                acc = per_class_correct[label] / per_class_total[label]
                avg_metrics[f'test_gate_acc_{LABEL_NAMES[label]}'] = acc
                print(f"  {LABEL_NAMES[label]}: {acc:.3f} ({per_class_correct[label]}/{per_class_total[label]})")

        overall_acc = sum(per_class_correct.values()) / sum(per_class_total.values())
        avg_metrics['test_gate_acc'] = overall_acc
        print(f"  Overall: {overall_acc:.3f}")

        return avg_metrics

    def train(self):
        print("\nStarting training...")
        print("=" * 60)

        epoch_times = []
        training_start = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 40)

            # Update loss function epoch for scheduled parameters
            self.criterion.set_epoch(epoch)
            cos_w = self.criterion.get_cosine_weight()
            kappa_reg_w, kappa_tgt = self.criterion.get_kappa_reg_params()
            print(f"  [Schedule] cosine_w={cos_w:.2f}, κ_reg_w={kappa_reg_w:.3f}, κ_target={kappa_tgt:.1f}")

            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            train_time = time.time() - t0

            # Validate
            val_metrics = self.validate()
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

            # Update scheduler
            self.scheduler.step()

            # ETA calculation
            avg_epoch_time = np.mean(epoch_times[-10:])
            remaining = self.args.epochs - epoch - 1
            eta_seconds = remaining * avg_epoch_time
            eta_h, eta_m = divmod(int(eta_seconds) // 60, 60)
            eta_s = int(eta_seconds) % 60

            elapsed = time.time() - training_start
            elapsed_h, elapsed_m = divmod(int(elapsed) // 60, 60)

            # Log
            lr = self.scheduler.get_last_lr()[0]
            print(f"\n  Train: loss={train_metrics['loss']:.4f}, "
                  f"1f={train_metrics.get('loss_1front', 0):.4f}, "
                  f"2f={train_metrics.get('loss_2front', 0):.4f}, "
                  f"4f={train_metrics.get('loss_4front', 0):.4f}")
            angle_err_1f = val_metrics.get('angle_error_1-front', 0)
            kappa_1f = val_metrics.get('kappa_1-front_mean', 0)
            print(f"  Val:   loss={val_metrics['loss']:.4f}, gate_acc={val_metrics['gate_acc']:.3f}, "
                  f"1f_err={np.degrees(angle_err_1f):.1f}°, 1f_κ={kappa_1f:.2f}")
            print(f"  LR: {lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed_h}h{elapsed_m:02d}m | ETA: {eta_h}h{eta_m:02d}m{eta_s:02d}s")

            # WandB logging
            if self.use_wandb:
                import wandb
                log_dict = {
                    'epoch': epoch + 1,
                    'lr': lr,
                    'train/loss': train_metrics['loss'],
                    'train/loss_1front': train_metrics.get('loss_1front', 0),
                    'train/loss_2front': train_metrics.get('loss_2front', 0),
                    'train/loss_4front': train_metrics.get('loss_4front', 0),
                    'train/loss_kappa_reg': train_metrics.get('loss_kappa_reg', 0),
                    'train/avg_sample_weight': train_metrics.get('avg_sample_weight', 1.0),
                    'train/gate_acc': train_metrics.get('gate_acc', 0),
                    'val/loss': val_metrics['loss'],
                    'val/gate_acc': val_metrics['gate_acc'],
                    # Scheduled parameters
                    'schedule/cosine_weight': train_metrics.get('cosine_weight', 0.1),
                    'schedule/kappa_reg_weight': train_metrics.get('kappa_reg_weight', 0),
                    'schedule/kappa_target': train_metrics.get('kappa_target', 0),
                }
                for k, v in val_metrics.items():
                    if k.startswith('angle_error') or k.startswith('kappa'):
                        log_dict[f'val/{k}'] = v
                # Add per-class val losses
                log_dict['val/loss_1front'] = val_metrics.get('loss_1front', 0)
                log_dict['val/loss_2front'] = val_metrics.get('loss_2front', 0)
                log_dict['val/loss_4front'] = val_metrics.get('loss_4front', 0)
                wandb.log(log_dict)

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self._save_checkpoint(epoch, is_best)

        # Final test
        print("\n" + "=" * 60)
        print("Training completed! Running final test...")
        test_metrics = self.test()

        if self.use_wandb:
            import wandb
            wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
            wandb.finish()

        print(f"\nBest val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }

        torch.save(checkpoint, self.output_dir / 'latest.pth')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pth')
            print("  [*] New best model saved!")

        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, self.output_dir / f'epoch_{epoch+1}.pth')

    def _load_checkpoint(self, path: str):
        print(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed from epoch {ckpt.get('epoch', 'unknown')}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train MoE Orientation Model (2-step)')

    # Data
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_rotations', type=int, default=4,
                        help='Training rotation augmentation factor (random)')
    parser.add_argument('--val_num_rotations', type=int, default=4,
                        help='Val/test rotation factor (deterministic uniform)')
    parser.add_argument('--balanced_sampler', action='store_true', default=True,
                        help='Use class-balanced sampling')
    parser.add_argument('--no_balanced_sampler', dest='balanced_sampler', action='store_false')
    parser.add_argument('--align_pointcloud', action='store_true', default=False,
                        help='Align front to -Z before augmentation (default: False to match classifier)')
    parser.add_argument('--no_align_pointcloud', dest='align_pointcloud', action='store_false')

    # Model
    parser.add_argument('--classifier_checkpoint', type=str,
                        default='checkpoints/SymClassifier_20251216_035345/best.pth',
                        help='Path to pre-trained classifier')
    parser.add_argument('--backbone_dim', type=int, default=1024)
    parser.add_argument('--expert_hidden_dim', type=int, default=256)
    parser.add_argument('--kappa_min', type=float, default=1e-4)
    parser.add_argument('--kappa_max', type=float, default=100.0)
    parser.add_argument('--freeze_classifier', action='store_true', default=True,
                        help='Freeze classifier (gate) during training')
    parser.add_argument('--no_freeze_classifier', dest='freeze_classifier', action='store_false')
    parser.add_argument('--classification_weight', type=float, default=0.0,
                        help='Weight for classification loss (0 = no joint training)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    # P2v2: Soft Gate-Aware training with schedules
    parser.add_argument('--no_gate_weighting', action='store_true',
                        help='Disable soft gate weighting of losses')
    parser.add_argument('--p_dir_threshold', type=float, default=0.4,
                        help='p_dir threshold for soft gate weight ramp (default 0.4)')
    parser.add_argument('--gamma', type=float, default=1.5,
                        help='Power for p_gt weighting (higher = favor high-confidence)')

    # Cosine loss schedule (always active for 1-front)
    parser.add_argument('--cosine_weight_init', type=float, default=0.2,
                        help='Initial cosine loss weight (epochs 0 to schedule_epoch)')
    parser.add_argument('--cosine_weight_final', type=float, default=0.1,
                        help='Final cosine loss weight (after schedule)')
    parser.add_argument('--cosine_schedule_epoch', type=int, default=10,
                        help='Epoch to switch from init to final cosine weight')

    # Kappa regularization schedule (only 1-front, staged)
    parser.add_argument('--kappa_reg_weight_final', type=float, default=0.02,
                        help='Final κ regularization weight')
    parser.add_argument('--kappa_target_final', type=float, default=5.0,
                        help='Final target κ value for regularization')
    parser.add_argument('--kappa_reg_start_epoch', type=int, default=6,
                        help='Epoch to start κ regularization')
    parser.add_argument('--kappa_reg_ramp_epochs', type=int, default=10,
                        help='Epochs to ramp κ reg from 0 to final')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

    # Logging
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Use WandB logging')
    parser.add_argument('--no_wandb', dest='wandb', action='store_false')
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation',
                        help='WandB project name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

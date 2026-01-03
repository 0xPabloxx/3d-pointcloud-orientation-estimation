#!/usr/bin/env python3
"""
Resume P2v2 Training

从 checkpoint 恢复训练，使用之前保存的状态。

用法：
    python tools/resume_p2v2_training.py checkpoints/P2v2_Clean_20251230_165848

会自动读取:
    - best.pth (模型和优化器状态)
    - training_state_snapshot.json (epoch, best_val_error 等)
    - config.json (训练配置)
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ProbabilisticOrientationNet, MaskedExpertLoss
from train_symmetry_classifier import SymmetryClassifier
from train_clean_pipeline import CleanMoEDataset, collate_fn, ClassifierWrapper

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_snapshot(checkpoint_dir: Path) -> dict:
    """加载训练状态快照"""
    snapshot_path = checkpoint_dir / "training_state_snapshot.json"
    if snapshot_path.exists():
        with open(snapshot_path, 'r') as f:
            return json.load(f)
    return {}


def load_config(checkpoint_dir: Path) -> dict:
    """加载训练配置"""
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    raise FileNotFoundError(f"Config not found: {config_path}")


class ResumedP2v2Trainer:
    """从 checkpoint 恢复的 P2v2 训练器"""

    def __init__(self, checkpoint_dir: Path, new_wandb_run: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载配置和状态
        self.config = load_config(checkpoint_dir)
        self.snapshot = load_snapshot(checkpoint_dir)

        # 确定起始 epoch
        if self.snapshot:
            self.start_epoch = self.snapshot.get('best_epoch', 0) + 1
            self.best_val_error = self.snapshot.get('best_val_error', float('inf'))
        else:
            # 从 best.pth 读取
            best_pth = checkpoint_dir / "best.pth"
            if best_pth.exists():
                ckpt = torch.load(best_pth, map_location='cpu', weights_only=False)
                self.start_epoch = ckpt.get('epoch', 0) + 1
                self.best_val_error = ckpt.get('metrics', {}).get('val_error', float('inf'))
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        self.total_epochs = self.config.get('p2v2_epochs', 100)

        print(f"{'='*60}")
        print(f"Resuming P2v2 Training")
        print(f"{'='*60}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Start epoch: {self.start_epoch}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Best val_error: {self.best_val_error:.4f}")
        print(f"{'='*60}")

        # WandB
        self.use_wandb = self.config.get('wandb', False) and WANDB_AVAILABLE and new_wandb_run
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'ForwardNet-LossAblation'),
                name=f"{checkpoint_dir.name}_resumed",
                config={**self.config, 'resumed_from_epoch': self.start_epoch}
            )

        # 数据
        self._setup_data()

        # 模型
        self._setup_model()

        # 优化器和调度器
        self._setup_optimizer()

        # 加载 checkpoint
        self._load_checkpoint()

    def _setup_data(self):
        """设置数据加载器"""
        print("\nLoading data...")

        self.train_dataset = CleanMoEDataset(
            annotation_file=self.config.get('annotation_file', 'data_annotation/symmetry_annotations.json'),
            data_dir=self.config.get('data_dir', 'data/full_mn40_normal_resampled_ply'),
            split='train',
            num_points=self.config.get('num_points', 2048),
            augment=True,
            num_rotations=self.config.get('p2v2_num_rotations', 12),
            seed=self.config.get('seed', 42),
            allowed_1front_categories=self.config.get('allowed_1front_categories'),
            outlier_json=self.config.get('outlier_json'),
            outlier_threshold=self.config.get('outlier_threshold', 'severe')
        )

        self.val_dataset = CleanMoEDataset(
            annotation_file=self.config.get('annotation_file', 'data_annotation/symmetry_annotations.json'),
            data_dir=self.config.get('data_dir', 'data/full_mn40_normal_resampled_ply'),
            split='val',
            num_points=self.config.get('num_points', 2048),
            augment=False,
            num_rotations=4,
            seed=self.config.get('seed', 42),
            allowed_1front_categories=self.config.get('allowed_1front_categories'),
            outlier_json=self.config.get('outlier_json'),
            outlier_threshold=self.config.get('outlier_threshold', 'severe')
        )

        self.test_dataset = CleanMoEDataset(
            annotation_file=self.config.get('annotation_file', 'data_annotation/symmetry_annotations.json'),
            data_dir=self.config.get('data_dir', 'data/full_mn40_normal_resampled_ply'),
            split='test',
            num_points=self.config.get('num_points', 2048),
            augment=False,
            num_rotations=4,
            seed=self.config.get('seed', 42),
            allowed_1front_categories=self.config.get('allowed_1front_categories'),
            outlier_json=self.config.get('outlier_json'),
            outlier_threshold=self.config.get('outlier_threshold', 'severe')
        )

        # Balanced sampler
        sample_weights = self.train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(self.config.get('p2v2_num_rotations', 12))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )

        print(f"Train: {len(self.train_dataset)} samples")
        print(f"Val: {len(self.val_dataset)} samples")
        print(f"Test: {len(self.test_dataset)} samples")

    def _setup_model(self):
        """设置模型"""
        print("\nSetting up model...")

        # 加载分类器
        classifier_path = self.config.get('classifier_path') or self.config.get('classifier_checkpoint')
        base_classifier = SymmetryClassifier(num_classes=5)
        ckpt = torch.load(classifier_path, map_location='cpu', weights_only=False)
        base_classifier.load_state_dict(ckpt['model_state_dict'])
        classifier = ClassifierWrapper(base_classifier).to(self.device)
        print(f"Classifier loaded from: {classifier_path}")

        # 创建模型
        self.model = ProbabilisticOrientationNet(
            classifier=classifier,
            backbone_dim=self.config.get('backbone_dim', 1024),
            expert_hidden_dim=self.config.get('expert_hidden_dim', 256),
            kappa_min=self.config.get('kappa_min', 1e-4),
            kappa_max=self.config.get('kappa_max', 100.0),
            freeze_classifier=True
        ).to(self.device)

        # 损失函数
        self.criterion = MaskedExpertLoss(
            classification_weight=0.0,
            use_gate_weighting=True,
            p_dir_threshold=self.config.get('p_dir_threshold', 0.4),
            gamma=self.config.get('gamma', 1.5),
            cosine_weight_init=self.config.get('cosine_weight_init', 0.2),
            cosine_weight_final=self.config.get('cosine_weight_final', 0.1),
            cosine_schedule_epoch=self.config.get('cosine_schedule_epoch', 10),
            kappa_reg_weight_final=self.config.get('kappa_reg_weight_final', 0.02),
            kappa_target_final=self.config.get('kappa_target_final', 5.0),
            kappa_reg_start_epoch=self.config.get('kappa_reg_start_epoch', 6),
            kappa_reg_ramp_epochs=self.config.get('kappa_reg_ramp_epochs', 10)
        )

    def _setup_optimizer(self):
        """设置优化器和调度器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('p2v2_lr', 1e-3),
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_epochs,
            eta_min=1e-6
        )

    def _load_checkpoint(self):
        """加载 checkpoint"""
        best_pth = self.checkpoint_dir / "best.pth"
        print(f"\nLoading checkpoint from: {best_pth}")

        ckpt = torch.load(best_pth, map_location=self.device, weights_only=False)

        # 加载模型
        self.model.load_state_dict(ckpt['model_state_dict'])
        print("  Model loaded")

        # 加载优化器
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Optimizer loaded")

        # 恢复 scheduler 状态（通过 step）
        for _ in range(self.start_epoch):
            self.scheduler.step()
        print(f"  Scheduler stepped to epoch {self.start_epoch}, LR = {self.scheduler.get_last_lr()[0]:.6f}")

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self.criterion.set_epoch(epoch)
        total_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_angle = batch['gt_angle'].to(self.device)
            gt_label = batch['gt_label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss_dict = self.criterion(outputs, gt_angle, gt_label)
            loss = loss_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}")

        return {'train_loss': total_loss / batch_count}

    @torch.no_grad()
    def validate(self, loader, name='val') -> Dict:
        self.model.eval()
        errors_1f, errors_2f, errors_4f = [], [], []
        kappas_1f, kappas_2f, kappas_4f = [], [], []

        for batch in loader:
            points = batch['points'].to(self.device)
            gt_angles = batch['gt_angle'].numpy()
            gt_labels = batch['gt_label'].numpy()

            outputs = self.model(points)
            mu_1f = outputs['head_1front']['mu'].cpu().numpy()
            kappa_1f = outputs['head_1front']['kappa'].cpu().numpy()
            mu_2f = outputs['head_2front']['mu'].cpu().numpy()
            kappa_2f = outputs['head_2front']['kappa'].cpu().numpy()
            mu_4f = outputs['head_4front']['mu'].cpu().numpy()
            kappa_4f = outputs['head_4front']['kappa'].cpu().numpy()

            for i in range(len(gt_labels)):
                label = gt_labels[i]
                gt = gt_angles[i]

                if label == 0:
                    err = self._circular_error(mu_1f[i, 0], gt)
                    errors_1f.append(err)
                    kappas_1f.append(kappa_1f[i, 0])
                elif label == 1:
                    gt_peaks = [gt, (gt + np.pi) % (2 * np.pi)]
                    pred_peaks = [mu_2f[i, j] for j in range(2)]
                    errors_2f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_2f.extend([kappa_2f[i, j] for j in range(2)])
                elif label == 2:
                    gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    pred_peaks = [mu_4f[i, j] for j in range(4)]
                    errors_4f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_4f.extend([kappa_4f[i, j] for j in range(4)])

        results = {}

        if errors_1f:
            errors_1f = np.array(errors_1f) * 180 / np.pi
            results[f'{name}_1f_error'] = float(np.mean(errors_1f))
            results[f'{name}_1f_median'] = float(np.median(errors_1f))
            results[f'{name}_1f_lt10'] = float(100 * np.mean(errors_1f < 10))
            results[f'{name}_1f_kappa'] = float(np.mean(kappas_1f))

        if errors_2f:
            errors_2f = np.array(errors_2f) * 180 / np.pi
            results[f'{name}_2f_error'] = float(np.mean(errors_2f))
            results[f'{name}_2f_lt10'] = float(100 * np.mean(errors_2f < 10))
            results[f'{name}_2f_kappa'] = float(np.mean(kappas_2f))

        if errors_4f:
            errors_4f = np.array(errors_4f) * 180 / np.pi
            results[f'{name}_4f_error'] = float(np.mean(errors_4f))
            results[f'{name}_4f_lt10'] = float(100 * np.mean(errors_4f < 10))
            results[f'{name}_4f_kappa'] = float(np.mean(kappas_4f))

        # Overall
        all_errors = []
        if len(errors_1f) > 0:
            all_errors.extend(errors_1f.tolist())
        if len(errors_2f) > 0:
            all_errors.extend(errors_2f.tolist())
        if len(errors_4f) > 0:
            all_errors.extend(errors_4f.tolist())

        if all_errors:
            results[f'{name}_error'] = float(np.mean(all_errors))
            results[f'{name}_lt10'] = float(100 * np.mean(np.array(all_errors) < 10))

        return results

    def _circular_error(self, pred: float, gt: float) -> float:
        pred = pred % (2 * np.pi)
        gt = gt % (2 * np.pi)
        diff = abs(pred - gt)
        return min(diff, 2 * np.pi - diff)

    def _hungarian_errors(self, pred_peaks, gt_peaks):
        n = len(gt_peaks)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = self._circular_error(pred_peaks[i], gt_peaks[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        return [cost[i, j] for i, j in zip(row_ind, col_ind)]

    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {self.start_epoch}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.total_epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.total_epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'val')
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining_epochs = self.total_epochs - epoch - 1
            eta = (elapsed / (epoch - self.start_epoch + 1)) * remaining_epochs if epoch > self.start_epoch else 0

            print(f"\n  Train loss: {train_metrics['train_loss']:.4f}")
            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° (<10°={val_metrics['val_1f_lt10']:.1f}%, κ={val_metrics['val_1f_kappa']:.1f})")
            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}° (<10°={val_metrics['val_2f_lt10']:.1f}%)")
            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}° (<10°={val_metrics['val_4f_lt10']:.1f}%)")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"  Time: {epoch_time:.0f}s | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

            if self.use_wandb:
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.scheduler.get_last_lr()[0]
                })

            val_error = val_metrics.get('val_error', float('inf'))
            if val_error < self.best_val_error:
                self.best_val_error = val_error
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                }, self.checkpoint_dir / 'best.pth')
                print(f"  [*] New best! val_error={val_error:.2f}°")

        # Final test
        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        test_metrics = self.validate(self.test_loader, 'test')
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}")

        torch.save({
            'epoch': self.total_epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'test_metrics': test_metrics,
        }, self.checkpoint_dir / 'final.pth')

        print(f"\nTraining complete!")
        print(f"Best val error: {self.best_val_error:.2f}°")

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Resume P2v2 Training')
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: {checkpoint_dir} does not exist")
        sys.exit(1)

    trainer = ResumedP2v2Trainer(checkpoint_dir, new_wandb_run=not args.no_wandb)
    trainer.train()


if __name__ == '__main__':
    main()

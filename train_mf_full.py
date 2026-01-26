#!/usr/bin/env python3
"""
MF Full Training - 包含对称性识别的MF模型训练

包含 1_front, 4_fronts, no_front 三种类别
no_front 类别的 GT kappa = 0，用于学习对称性识别

用法:
    python train_mf_full.py --exp_name MF_Full_C
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent))

from datasets.multi_category_dataset import MultiCategoryDataset, collate_fn
from test_direction_models import DirectionModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not available")


class MixtureVonMisesLoss(nn.Module):
    """
    Mixture von Mises Loss - 支持对称性识别

    对于 no_front 类别，GT kappa = 0
    """
    def __init__(
        self,
        lambda_kl=0.5,
        lambda_kappa=5.0,
        lambda_mu=3.0,
        gt_kappa=10.0,
        eps=1e-8,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_kappa = lambda_kappa
        self.lambda_mu = lambda_mu
        self.gt_kappa = gt_kappa
        self.eps = eps

        # 预计算角度采样点
        self.register_buffer('angles', torch.linspace(0, 2*np.pi, 360))

    def _von_mises_pdf(self, theta, mu, kappa):
        """计算 von Mises PDF"""
        # theta: (360,), mu: (B, 4), kappa: (B, 4)
        # 输出: (B, 4, 360)
        theta = theta.to(mu.device).view(1, 1, -1)  # (1, 1, 360)
        mu = mu.unsqueeze(-1)  # (B, 4, 1)
        kappa = kappa.unsqueeze(-1)  # (B, 4, 1)

        log_pdf = kappa * torch.cos(theta - mu)
        # 归一化
        pdf = F.softmax(log_pdf, dim=-1)
        return pdf

    def _compute_mixture_pdf(self, mu, kappa, weights):
        """计算混合 von Mises PDF"""
        # mu: (B, 4), kappa: (B, 4), weights: (B, 4)
        pdf = self._von_mises_pdf(self.angles, mu, kappa)  # (B, 4, 360)
        weights = weights.unsqueeze(-1)  # (B, 4, 1)
        mixture_pdf = (weights * pdf).sum(dim=1)  # (B, 360)
        return mixture_pdf

    def forward(self, pred_mu, pred_kappa, gt_mu, gt_kappa, categories):
        """
        Args:
            pred_mu: (B, 4, 2) 预测方向向量 [cos, sin]
            pred_kappa: (B, 4) 预测集中度
            gt_mu: (B, 4, 2) GT方向向量 [cos, sin]
            gt_kappa: (B, 4) GT集中度
            categories: list of str, 类别列表
        """
        device = pred_mu.device
        B = pred_mu.shape[0]

        # 转换pred_mu从向量到角度
        pred_mu_angle = torch.atan2(pred_mu[:, :, 1], pred_mu[:, :, 0])  # (B, 4)
        pred_mu_angle = (pred_mu_angle + 2 * np.pi) % (2 * np.pi)

        # 转换GT mu从向量到角度
        gt_mu_angle = torch.atan2(gt_mu[:, :, 1], gt_mu[:, :, 0])  # (B, 4)
        gt_mu_angle = (gt_mu_angle + 2 * np.pi) % (2 * np.pi)

        losses = {}
        weights = torch.full((B, 4), 0.25, device=device)

        # === KL Loss ===
        if self.lambda_kl > 0:
            pred_pdf = self._compute_mixture_pdf(pred_mu_angle, pred_kappa, weights)
            gt_pdf = self._compute_mixture_pdf(gt_mu_angle, gt_kappa, weights)
            kl_loss = (gt_pdf * (torch.log(gt_pdf + self.eps) - torch.log(pred_pdf + self.eps))).sum(dim=-1).mean()
            losses['kl_loss'] = kl_loss

        # === Kappa Loss (分类别) ===
        # 创建类别mask
        mask_1f = torch.tensor([c == '1_front' for c in categories], device=device)
        mask_2f = torch.tensor([c == '2_fronts' for c in categories], device=device)
        mask_4f = torch.tensor([c == '4_fronts' for c in categories], device=device)
        mask_nf = torch.tensor([c in ['no_front', 'symmetric'] for c in categories], device=device)

        kappa_loss = torch.tensor(0.0, device=device)

        # 1_front, 2_fronts, 4_fronts: kappa 应该接近 gt_kappa
        directional_mask = mask_1f | mask_2f | mask_4f
        if directional_mask.any():
            kappa_diff = (pred_kappa[directional_mask] - gt_kappa[directional_mask]).pow(2)
            kappa_loss = kappa_loss + kappa_diff.mean()

        # no_front: kappa 应该接近 0
        if mask_nf.any():
            kappa_nf_loss = pred_kappa[mask_nf].pow(2).mean()
            kappa_loss = kappa_loss + kappa_nf_loss

        losses['kappa_loss'] = kappa_loss

        # === Mu Loss (只对有方向的类别) ===
        mu_loss = torch.tensor(0.0, device=device)
        if directional_mask.any():
            pred_mu_dir = pred_mu_angle[directional_mask]
            gt_mu_dir = gt_mu_angle[directional_mask]

            # Cosine distance for circular
            cos_dist = 1 - torch.cos(pred_mu_dir - gt_mu_dir)
            mu_loss = cos_dist.mean()

        losses['mu_loss'] = mu_loss

        # === Total Loss ===
        total_loss = torch.tensor(0.0, device=device)
        if self.lambda_kl > 0:
            total_loss = total_loss + self.lambda_kl * losses['kl_loss']
        total_loss = total_loss + self.lambda_kappa * losses['kappa_loss']
        total_loss = total_loss + self.lambda_mu * losses['mu_loss']

        losses['total_loss'] = total_loss

        # === 额外统计 ===
        with torch.no_grad():
            if mask_1f.any():
                losses['kappa_1f_mean'] = pred_kappa[mask_1f].mean()
            if mask_4f.any():
                losses['kappa_4f_mean'] = pred_kappa[mask_4f].mean()
            if mask_nf.any():
                losses['kappa_nf_mean'] = pred_kappa[mask_nf].mean()
                losses['kappa_nf_max'] = pred_kappa[mask_nf].max()

        return losses


class MFFullTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{args.exp_name}_{timestamp}"
        self.output_dir = Path('checkpoints') / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

        # Datasets - 包含所有4种类别
        self.train_dataset = MultiCategoryDataset(
            annotation_file=args.annotation_file,
            data_root=args.data_root,
            split='train',
            categories=['1_front', '2_fronts', '4_fronts', 'no_front'],
            num_points=args.num_points,
            augment=True,
            augment_factor=args.augment_factor,
            gt_kappa=args.gt_kappa,
        )

        self.val_dataset = MultiCategoryDataset(
            annotation_file=args.annotation_file,
            data_root=args.data_root,
            split='val',
            categories=['1_front', '2_fronts', '4_fronts', 'no_front'],
            num_points=args.num_points,
            augment=False,
            gt_kappa=args.gt_kappa,
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        # Model
        self.model = DirectionModel(mode='mf', num_peaks=4).to(self.device)

        # Loss
        self.criterion = MixtureVonMisesLoss(
            lambda_kl=args.lambda_kl,
            lambda_kappa=args.lambda_kappa,
            lambda_mu=args.lambda_mu,
            gt_kappa=args.gt_kappa,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # Best tracking
        self.best_val_error = float('inf')
        self.best_val_loss = float('inf')

        # WandB
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=self.exp_name,
                config=vars(args),
            )

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_kl = 0.0
        total_kappa = 0.0
        total_mu = 0.0
        kappa_1f_sum, kappa_4f_sum, kappa_nf_sum = 0.0, 0.0, 0.0
        count_1f, count_4f, count_nf = 0, 0, 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_mu = batch['gt_mu'].to(self.device)
            gt_kappa = batch['gt_kappa'].to(self.device)
            categories = batch['category']

            self.optimizer.zero_grad()

            mu, kappa = self.model(points)
            loss_dict = self.criterion(mu, kappa, gt_mu, gt_kappa, categories)

            loss = loss_dict['total_loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            if 'kl_loss' in loss_dict:
                total_kl += loss_dict['kl_loss'].item()
            total_kappa += loss_dict['kappa_loss'].item()
            total_mu += loss_dict['mu_loss'].item()

            if 'kappa_1f_mean' in loss_dict:
                kappa_1f_sum += loss_dict['kappa_1f_mean'].item()
                count_1f += 1
            if 'kappa_4f_mean' in loss_dict:
                kappa_4f_sum += loss_dict['kappa_4f_mean'].item()
                count_4f += 1
            if 'kappa_nf_mean' in loss_dict:
                kappa_nf_sum += loss_dict['kappa_nf_mean'].item()
                count_nf += 1

            batch_count += 1

        results = {
            'train_loss': total_loss / batch_count,
            'train_kl': total_kl / batch_count if total_kl > 0 else 0,
            'train_kappa_loss': total_kappa / batch_count,
            'train_mu_loss': total_mu / batch_count,
        }

        if count_1f > 0:
            results['train_kappa_1f'] = kappa_1f_sum / count_1f
        if count_4f > 0:
            results['train_kappa_4f'] = kappa_4f_sum / count_4f
        if count_nf > 0:
            results['train_kappa_nf'] = kappa_nf_sum / count_nf

        return results

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        # 分类别收集
        errors_1f, errors_2f, errors_4f = [], [], []
        kappas_1f, kappas_2f, kappas_4f, kappas_nf = [], [], [], []
        total_loss = 0.0
        batch_count = 0

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_mu = batch['gt_mu'].to(self.device)
            gt_kappa = batch['gt_kappa'].to(self.device)
            gt_angle = batch['gt_angle'].numpy()
            categories = batch['category']

            mu, kappa = self.model(points)
            loss_dict = self.criterion(mu, kappa, gt_mu, gt_kappa, categories)
            total_loss += loss_dict['total_loss'].item()
            batch_count += 1

            # 转换mu从向量到角度
            mu_angle = torch.atan2(mu[:, :, 1], mu[:, :, 0])  # (B, 4)
            mu_angle = (mu_angle + 2 * np.pi) % (2 * np.pi)
            mu_np = mu_angle.cpu().numpy()
            kappa_np = kappa.cpu().numpy()

            for i, cat in enumerate(categories):
                if cat == '1_front':
                    # 取最接近GT的peak
                    gt = gt_angle[i]
                    errs = [self._circular_error(mu_np[i, j], gt) for j in range(4)]
                    best_idx = np.argmin(errs)
                    errors_1f.append(errs[best_idx])
                    kappas_1f.append(kappa_np[i, best_idx])

                elif cat == '2_fronts':
                    # 2-front: GT peaks at gt and gt+π
                    gt = gt_angle[i]
                    gt_peaks = [(gt + j * np.pi) % (2 * np.pi) for j in range(2)]
                    # 取top-2 kappa的peaks
                    top2_idx = np.argsort(kappa_np[i])[-2:]
                    pred_peaks = [mu_np[i, j] for j in top2_idx]
                    errors_2f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_2f.extend([kappa_np[i, j] for j in top2_idx])

                elif cat == '4_fronts':
                    gt = gt_angle[i]
                    gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    pred_peaks = [mu_np[i, j] for j in range(4)]
                    errors_4f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_4f.extend([kappa_np[i, j] for j in range(4)])

                elif cat in ['no_front', 'symmetric']:
                    kappas_nf.extend([kappa_np[i, j] for j in range(4)])

        results = {'val_loss': total_loss / batch_count}

        # 1-front metrics
        if errors_1f:
            errors_1f = np.array(errors_1f) * 180 / np.pi
            results['val_1f_error'] = float(np.mean(errors_1f))
            results['val_1f_median'] = float(np.median(errors_1f))
            results['val_1f_std'] = float(np.std(errors_1f))
            results['val_1f_lt5'] = float(100 * np.mean(errors_1f < 5))
            results['val_1f_lt10'] = float(100 * np.mean(errors_1f < 10))
            results['val_1f_lt15'] = float(100 * np.mean(errors_1f < 15))
            results['val_1f_gt45'] = float(100 * np.mean(errors_1f > 45))

        if kappas_1f:
            results['val_1f_kappa'] = float(np.mean(kappas_1f))
            results['val_1f_kappa_std'] = float(np.std(kappas_1f))

        # 2-front metrics
        if errors_2f:
            errors_2f = np.array(errors_2f) * 180 / np.pi
            results['val_2f_error'] = float(np.mean(errors_2f))
            results['val_2f_median'] = float(np.median(errors_2f))
            results['val_2f_lt10'] = float(100 * np.mean(errors_2f < 10))

        if kappas_2f:
            results['val_2f_kappa'] = float(np.mean(kappas_2f))

        # 4-front metrics
        if errors_4f:
            errors_4f = np.array(errors_4f) * 180 / np.pi
            results['val_4f_error'] = float(np.mean(errors_4f))
            results['val_4f_median'] = float(np.median(errors_4f))
            results['val_4f_lt5'] = float(100 * np.mean(errors_4f < 5))
            results['val_4f_lt10'] = float(100 * np.mean(errors_4f < 10))

        if kappas_4f:
            results['val_4f_kappa'] = float(np.mean(kappas_4f))

        # no_front metrics (对称性识别)
        if kappas_nf:
            kappas_nf = np.array(kappas_nf)
            results['val_nf_kappa_mean'] = float(np.mean(kappas_nf))
            results['val_nf_kappa_std'] = float(np.std(kappas_nf))
            results['val_nf_kappa_max'] = float(np.max(kappas_nf))
            # 对称性识别率：kappa < 阈值的比例
            results['val_nf_kappa_lt0.1'] = float(100 * np.mean(kappas_nf < 0.1))
            results['val_nf_kappa_lt0.5'] = float(100 * np.mean(kappas_nf < 0.5))
            results['val_nf_kappa_lt1.0'] = float(100 * np.mean(kappas_nf < 1.0))

        # Overall directional error
        all_errors = []
        if errors_1f is not None and len(errors_1f) > 0:
            all_errors.extend(errors_1f.tolist())
        if errors_2f is not None and len(errors_2f) > 0:
            all_errors.extend(errors_2f.tolist())
        if errors_4f is not None and len(errors_4f) > 0:
            all_errors.extend(errors_4f.tolist())

        if all_errors:
            all_errors = np.array(all_errors)
            results['val_error'] = float(np.mean(all_errors))
            results['val_median'] = float(np.median(all_errors))
            results['val_lt10'] = float(100 * np.mean(all_errors < 10))

        return results

    def _circular_error(self, pred, gt):
        pred = pred % (2 * np.pi)
        gt = gt % (2 * np.pi)
        diff = abs(pred - gt)
        return min(diff, 2 * np.pi - diff)

    def _hungarian_errors(self, pred_peaks, gt_peaks):
        from scipy.optimize import linear_sum_assignment
        n = len(gt_peaks)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = self._circular_error(pred_peaks[i], gt_peaks[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        return [cost[i, j] for i, j in zip(row_ind, col_ind)]

    def train(self):
        print("\n" + "=" * 70)
        print(f"MF Full Training: {self.exp_name}")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Lambda: kl={self.args.lambda_kl}, kappa={self.args.lambda_kappa}, mu={self.args.lambda_mu}")
        print(f"Categories: 1_front, 2_fronts, 4_fronts, no_front")
        print(f"Augment factor: {self.args.augment_factor}")
        print(f"Epochs: {self.args.epochs}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.get_url()}")
        print("=" * 70 + "\n")

        start_time = time.time()

        for epoch in range(self.args.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Time estimation
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.args.epochs - epoch - 1)

            # Print
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 50)
            print(f"  Train: loss={train_metrics['train_loss']:.4f}, "
                  f"kl={train_metrics.get('train_kl', 0):.4f}, "
                  f"kappa={train_metrics['train_kappa_loss']:.4f}, "
                  f"mu={train_metrics['train_mu_loss']:.4f}")

            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° "
                      f"(median={val_metrics['val_1f_median']:.2f}°, "
                      f"<10°={val_metrics['val_1f_lt10']:.1f}%, "
                      f"κ={val_metrics['val_1f_kappa']:.2f})")

            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}° "
                      f"(<10°={val_metrics['val_2f_lt10']:.1f}%, "
                      f"κ={val_metrics['val_2f_kappa']:.2f})")

            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}° "
                      f"(<10°={val_metrics['val_4f_lt10']:.1f}%, "
                      f"κ={val_metrics['val_4f_kappa']:.2f})")

            if 'val_nf_kappa_mean' in val_metrics:
                print(f"  Val nf: κ_mean={val_metrics['val_nf_kappa_mean']:.3f}, "
                      f"κ_max={val_metrics['val_nf_kappa_max']:.3f}, "
                      f"<0.1={val_metrics['val_nf_kappa_lt0.1']:.1f}%, "
                      f"<1.0={val_metrics['val_nf_kappa_lt1.0']:.1f}%")

            if 'val_error' in val_metrics:
                print(f"  Val overall: {val_metrics['val_error']:.2f}° (<10°={val_metrics['val_lt10']:.1f}%)")

            print(f"  Time: {epoch_time:.0f}s | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

            # WandB logging
            if self.use_wandb:
                log_dict = {
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.scheduler.get_last_lr()[0],
                }
                wandb.log(log_dict)

            # Save best by error
            if 'val_error' in val_metrics and val_metrics['val_error'] < self.best_val_error:
                self.best_val_error = val_metrics['val_error']
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'head_state_dict': self.model.head.state_dict(),
                    'best_val_error': self.best_val_error,
                }, self.output_dir / 'best_error.pth')
                print(f"  [*] New best error: {self.best_val_error:.2f}°")

            # Save best by loss
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'head_state_dict': self.model.head.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, self.output_dir / 'best_loss.pth')

            # Log to file
            with open(self.output_dir / 'training_log.txt', 'a') as f:
                f.write(f"Epoch {epoch+1}: train_loss={train_metrics['train_loss']:.4f}, "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"val_error={val_metrics.get('val_error', 0):.2f}°, "
                        f"val_nf_kappa={val_metrics.get('val_nf_kappa_mean', 0):.3f}, "
                        f"lr={self.scheduler.get_last_lr()[0]:.6f}\n")

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total time: {total_time/60:.1f} min")
        print(f"Best val error: {self.best_val_error:.2f}°")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoint: {self.output_dir}")

        with open(self.output_dir / 'training_log.txt', 'a') as f:
            f.write(f"\nBest val error: {self.best_val_error:.2f}°\n")
            f.write(f"Best val loss: {self.best_val_loss:.4f}\n")

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='MF Full Training')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='MF_Full_C')

    # Data
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_root', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--augment_factor', type=int, default=10)

    # Loss weights (方案C: 保方向)
    parser.add_argument('--lambda_kl', type=float, default=0.5)
    parser.add_argument('--lambda_kappa', type=float, default=5.0)
    parser.add_argument('--lambda_mu', type=float, default=3.0)
    parser.add_argument('--gt_kappa', type=float, default=10.0)

    # WandB
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation')
    parser.add_argument('--wandb_entity', type=str, default='augustuschen00-university-of-tokyo')

    args = parser.parse_args()

    trainer = MFFullTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

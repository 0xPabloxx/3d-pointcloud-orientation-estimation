#!/usr/bin/env python3
"""
Train MuOnly model on FULL training set (no filtering)
- 8x random rotations (truly random per sample)
- 50 epochs
- Log to wandb ForwardNet-LossAblation
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_symmetry_classifier import PointNetPlusPlusEncoder
from datasets.moe_dataset import read_ply, LABEL_NAMES, SYMMETRY_TO_LABEL

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Full Dataset (No Filtering)
# =============================================================================

class FullMoEDataset(Dataset):
    """
    Full MoE Dataset - uses ALL data without filtering
    """

    DIRECTION_TO_ANGLE = {
        '+X': 0.0,
        '+Z': np.pi / 2,
        '-X': np.pi,
        '-Z': 3 * np.pi / 2,
    }

    def __init__(self,
                 annotation_file: str = 'data_annotation/symmetry_annotations.json',
                 data_dir: str = 'data/full_mn40_normal_resampled_ply',
                 split: str = 'train',
                 num_points: int = 2048,
                 augment: bool = True,
                 num_rotations: int = 8,
                 seed: int = 42):
        """
        Args:
            split: 'train', 'val', 'test', or 'all'
            num_rotations: Number of rotation augmentations
            augment: If True, use random rotations; if False, use deterministic rotations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.num_rotations = num_rotations

        self.samples = self._load_annotations(annotation_file, split, seed)
        self._print_stats()

    def _get_object_category(self, file_path: str) -> str:
        parts = file_path.split('/')
        if len(parts) >= 1:
            return parts[0]
        return file_path.split('_')[0]

    def _get_sample_name(self, file_path: str) -> str:
        fname = file_path.split('/')[-1] if '/' in file_path else file_path
        return fname.replace('.ply', '')

    def _load_annotations(self, annotation_file: str, split: str, seed: int) -> List[Dict]:
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        for file_path, ann in all_annotations.items():
            symmetry_name = ann.get('symmetry_name')
            direction = ann.get('front_direction')

            if not symmetry_name or direction in excluded_directions:
                continue

            label = SYMMETRY_TO_LABEL.get(symmetry_name)
            if label is None:
                continue

            sample_name = self._get_sample_name(file_path)
            obj_category = self._get_object_category(file_path)

            ply_path = self.data_dir / file_path
            if not ply_path.exists():
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'symmetry_name': symmetry_name,
                'label': label,
                'direction': direction,
                'object_category': obj_category,
                'sample_name': sample_name,
            })

        # Stratified split
        np.random.seed(seed)
        by_label = {i: [] for i in range(5)}
        for i, s in enumerate(samples):
            by_label[s['label']].append(i)

        selected_indices = []
        for label, indices in by_label.items():
            indices = np.array(indices)
            np.random.shuffle(indices)
            n = len(indices)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)

            if split == 'train':
                selected_indices.extend(indices[:n_train])
            elif split == 'val':
                selected_indices.extend(indices[n_train:n_train + n_val])
            elif split == 'test':
                selected_indices.extend(indices[n_train + n_val:])
            elif split == 'all':
                selected_indices.extend(indices)

        return [samples[i] for i in selected_indices]

    def _print_stats(self):
        label_counts = Counter(s['label'] for s in self.samples)
        rot_type = "random" if self.augment else "deterministic"
        aug_str = f" x{self.num_rotations} ({rot_type})" if self.num_rotations > 1 else ""

        print(f"[FullMoEDataset {self.split}] {len(self.samples)} samples{aug_str} = {len(self)} total")
        for label in range(5):
            count = label_counts.get(label, 0)
            print(f"  {label} ({LABEL_NAMES[label]}): {count}")

        # Show 1-front categories distribution
        one_front_cats = Counter(s['object_category'] for s in self.samples if s['label'] == 0)
        print(f"  1-front categories: {dict(one_front_cats)}")

    def __len__(self):
        return len(self.samples) * self.num_rotations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = idx // self.num_rotations
        sample = self.samples[sample_idx]

        points = read_ply(sample['ply_path'])

        # Subsample - use random sampling for augment mode
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]

        # Normalize
        centroid = points.mean(axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        # Get base angle
        base_angle = self.DIRECTION_TO_ANGLE.get(sample['direction'], 0.0)

        # Apply TRULY random rotation (completely random, no seed)
        if self.num_rotations > 1:
            # Truly random rotation angle
            rotation_angle = np.random.uniform(0, 2 * np.pi)

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            rotation_matrix = np.array([
                [cos_r, 0, sin_r],
                [0, 1, 0],
                [-sin_r, 0, cos_r]
            ])
            points = points @ rotation_matrix.T
            gt_angle = (base_angle + rotation_angle) % (2 * np.pi)
        else:
            gt_angle = base_angle

        return {
            'points': torch.from_numpy(points).float(),
            'gt_angle': torch.tensor(gt_angle, dtype=torch.float32),
            'gt_label': torch.tensor(sample['label'], dtype=torch.long),
            'file': sample['file'],
        }

    def get_sample_weights(self) -> torch.Tensor:
        label_counts = Counter(s['label'] for s in self.samples)
        class_weights = {label: 1.0 / count for label, count in label_counts.items()}
        sample_weights = [class_weights[s['label']] for s in self.samples]
        return torch.tensor(sample_weights)


def collate_fn(batch):
    return {
        'points': torch.stack([b['points'] for b in batch]),
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'gt_label': torch.stack([b['gt_label'] for b in batch]),
        'file': [b['file'] for b in batch],
    }


# =============================================================================
# Model (same as in train_clean_pipeline.py)
# =============================================================================

class MixturePeakHead(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512, num_peaks: int = 4):
        super().__init__()
        self.num_peaks = num_peaks

        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        mu_vec = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu_vec = F.normalize(mu_vec, dim=-1)
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])
        mu = (mu + 2 * np.pi) % (2 * np.pi)
        kappa = F.softplus(self.kappa_head(h))
        return mu, kappa


class BaselineDirectionModel(nn.Module):
    def __init__(self, backbone_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=backbone_dim)
        self.head = MixturePeakHead(in_channels=backbone_dim, hidden_channels=hidden_dim)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(points)
        mu, kappa = self.head(features)
        return {'mu': mu, 'kappa': kappa}


# =============================================================================
# Loss (same as in train_clean_pipeline.py)
# =============================================================================

class BaselineMuOnlyLoss(nn.Module):
    def __init__(self, lambda_mu: float = 2.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, outputs: Dict, gt_angle: torch.Tensor, gt_label: torch.Tensor,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        device = gt_angle.device
        B = gt_angle.shape[0]

        pred_mu = outputs['mu']

        total_loss = torch.tensor(0.0, device=device)
        loss_1f = torch.tensor(0.0, device=device)
        loss_2f = torch.tensor(0.0, device=device)
        loss_4f = torch.tensor(0.0, device=device)
        count_1f, count_2f, count_4f = 0, 0, 0

        for i in range(B):
            label = gt_label[i].item()
            gt = gt_angle[i]
            pred = pred_mu[i]

            if label == 0:  # 1-front
                loss = torch.mean(1 - torch.cos(pred - gt))
                loss_1f = loss_1f + loss
                count_1f += 1

            elif label == 1:  # 2-front
                gt_peaks = torch.stack([gt, (gt + np.pi) % (2 * np.pi)])
                pred_peaks = pred[:2]
                loss = self._hungarian_cosine_loss(pred_peaks, gt_peaks)
                loss_2f = loss_2f + loss
                count_2f += 1

            elif label == 2:  # 4-front
                gt_peaks = torch.stack([(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)])
                loss = self._hungarian_cosine_loss(pred, gt_peaks)
                loss_4f = loss_4f + loss
                count_4f += 1

        if count_1f > 0:
            loss_1f = loss_1f / count_1f
        if count_2f > 0:
            loss_2f = loss_2f / count_2f
        if count_4f > 0:
            loss_4f = loss_4f / count_4f

        active_losses = []
        if count_1f > 0:
            active_losses.append(loss_1f)
        if count_2f > 0:
            active_losses.append(loss_2f)
        if count_4f > 0:
            active_losses.append(loss_4f)

        if active_losses:
            total_loss = self.lambda_mu * sum(active_losses) / len(active_losses)
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'loss': total_loss,
            'loss_1front': loss_1f,
            'loss_2front': loss_2f,
            'loss_4front': loss_4f,
            'count_1front': count_1f,
            'count_2front': count_2f,
            'count_4front': count_4f,
        }

    def _hungarian_cosine_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        n = len(gt)
        m = len(pred)
        cost = torch.zeros(m, n, device=pred.device)
        for i in range(m):
            for j in range(n):
                cost[i, j] = 1 - torch.cos(pred[i] - gt[j])

        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        loss = torch.tensor(0.0, device=pred.device)
        for i, j in zip(row_ind, col_ind):
            loss = loss + cost[i, j]
        return loss / len(row_ind)


# =============================================================================
# Trainer
# =============================================================================

class MuOnlyFullTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'checkpoints/MuOnly_Full_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            'model_type': 'muonly_baseline',
            'loss_type': 'mu_only',
            'dataset': 'full (no filtering)',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_rotations': args.num_rotations,
            'num_points': args.num_points,
            'lambda_mu': args.lambda_mu,
            'backbone_dim': args.backbone_dim,
            'hidden_dim': args.hidden_dim,
            'seed': args.seed,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # WandB
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f'MuOnly_Full_8xRandRot_{timestamp}',
                config=config
            )

        # Data - FULL dataset (no filtering)
        print("\n" + "=" * 60)
        print("Loading FULL dataset (NO filtering)")
        print(f"Rotations: {args.num_rotations}x (truly random)")
        print("=" * 60)

        self.train_dataset = FullMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='train',
            num_points=args.num_points,
            augment=True,  # Truly random rotations
            num_rotations=args.num_rotations,
            seed=args.seed,
        )

        self.val_dataset = FullMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='val',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
        )

        self.test_dataset = FullMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='test',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
        )

        # Balanced sampler
        sample_weights = self.train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(args.num_rotations)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )

        # Model
        self.model = BaselineDirectionModel(
            backbone_dim=args.backbone_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        # Loss
        self.criterion = BaselineMuOnlyLoss(lambda_mu=args.lambda_mu)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )

        self.best_val_error = float('inf')

        # Print info
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"\n{'='*60}")
        print(f"MuOnly Full Training Configuration")
        print(f"{'='*60}")
        print(f"Model: BaselineDirectionModel (NO MoE)")
        print(f"Parameters: {total_params:,}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Epochs: {args.epochs}")
        print(f"Rotations: {args.num_rotations}x (random)")
        print(f"Lambda_mu: {args.lambda_mu}")
        print(f"Output dir: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.get_url()}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_loss_1f = 0.0
        total_loss_2f = 0.0
        total_loss_4f = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_angle = batch['gt_angle'].to(self.device)
            gt_label = batch['gt_label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss_dict = self.criterion(outputs, gt_angle, gt_label, epoch)
            loss = loss_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_1f += loss_dict['loss_1front'].item() if isinstance(loss_dict['loss_1front'], torch.Tensor) else loss_dict['loss_1front']
            total_loss_2f += loss_dict['loss_2front'].item() if isinstance(loss_dict['loss_2front'], torch.Tensor) else loss_dict['loss_2front']
            total_loss_4f += loss_dict['loss_4front'].item() if isinstance(loss_dict['loss_4front'], torch.Tensor) else loss_dict['loss_4front']
            batch_count += 1

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}")

        return {
            'train_loss': total_loss / batch_count,
            'train_loss_1f': total_loss_1f / batch_count,
            'train_loss_2f': total_loss_2f / batch_count,
            'train_loss_4f': total_loss_4f / batch_count,
        }

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
            pred_mu = outputs['mu'].cpu().numpy()
            pred_kappa = outputs['kappa'].cpu().numpy()

            for i in range(len(gt_labels)):
                label = gt_labels[i]
                gt = gt_angles[i]

                if label == 0:  # 1-front
                    errs = [self._circular_error(pred_mu[i, j], gt) for j in range(4)]
                    best_idx = np.argmin(errs)
                    errors_1f.append(errs[best_idx])
                    kappas_1f.append(pred_kappa[i, best_idx])

                elif label == 1:  # 2-front
                    gt_peaks = [gt, (gt + np.pi) % (2 * np.pi)]
                    pred_peaks = [pred_mu[i, j] for j in range(2)]
                    errors_2f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_2f.extend([pred_kappa[i, j] for j in range(2)])

                elif label == 2:  # 4-front
                    gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    pred_peaks = [pred_mu[i, j] for j in range(4)]
                    errors_4f.extend(self._hungarian_errors(pred_peaks, gt_peaks))
                    kappas_4f.extend([pred_kappa[i, j] for j in range(4)])

        results = {}

        if errors_1f:
            errors_1f = np.array(errors_1f) * 180 / np.pi
            results[f'{name}_1f_error'] = float(np.mean(errors_1f))
            results[f'{name}_1f_median'] = float(np.median(errors_1f))
            results[f'{name}_1f_std'] = float(np.std(errors_1f))
            results[f'{name}_1f_lt5'] = float(100 * np.mean(errors_1f < 5))
            results[f'{name}_1f_lt10'] = float(100 * np.mean(errors_1f < 10))
            results[f'{name}_1f_lt15'] = float(100 * np.mean(errors_1f < 15))
            results[f'{name}_1f_gt45'] = float(100 * np.mean(errors_1f > 45))
            results[f'{name}_1f_gt90'] = float(100 * np.mean(errors_1f > 90))
            results[f'{name}_1f_kappa'] = float(np.mean(kappas_1f))

        if errors_2f:
            errors_2f = np.array(errors_2f) * 180 / np.pi
            results[f'{name}_2f_error'] = float(np.mean(errors_2f))
            results[f'{name}_2f_median'] = float(np.median(errors_2f))
            results[f'{name}_2f_lt5'] = float(100 * np.mean(errors_2f < 5))
            results[f'{name}_2f_lt10'] = float(100 * np.mean(errors_2f < 10))
            results[f'{name}_2f_kappa'] = float(np.mean(kappas_2f))

        if errors_4f:
            errors_4f = np.array(errors_4f) * 180 / np.pi
            results[f'{name}_4f_error'] = float(np.mean(errors_4f))
            results[f'{name}_4f_median'] = float(np.median(errors_4f))
            results[f'{name}_4f_lt5'] = float(100 * np.mean(errors_4f < 5))
            results[f'{name}_4f_lt10'] = float(100 * np.mean(errors_4f < 10))
            results[f'{name}_4f_kappa'] = float(np.mean(kappas_4f))

        # Overall metrics
        all_errors = []
        if errors_1f is not None and len(errors_1f) > 0:
            all_errors.extend(errors_1f.tolist())
        if errors_2f is not None and len(errors_2f) > 0:
            all_errors.extend(errors_2f.tolist())
        if errors_4f is not None and len(errors_4f) > 0:
            all_errors.extend(errors_4f.tolist())

        if all_errors:
            all_errors = np.array(all_errors)
            results[f'{name}_error'] = float(np.mean(all_errors))
            results[f'{name}_median'] = float(np.median(all_errors))
            results[f'{name}_lt10'] = float(100 * np.mean(all_errors < 10))

        return results

    def _circular_error(self, pred: float, gt: float) -> float:
        pred = pred % (2 * np.pi)
        gt = gt % (2 * np.pi)
        diff = abs(pred - gt)
        return min(diff, 2 * np.pi - diff)

    def _hungarian_errors(self, pred_peaks: List[float], gt_peaks: List[float]) -> List[float]:
        n = len(gt_peaks)
        m = len(pred_peaks)
        cost = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                cost[i, j] = self._circular_error(pred_peaks[i], gt_peaks[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        return [cost[i, j] for i, j in zip(row_ind, col_ind)]

    def train(self):
        print("\n" + "=" * 60)
        print("Starting MuOnly Full Training...")
        print(f"Training on FULL dataset with {self.args.num_rotations}x random rotations")
        print("=" * 60)

        import time
        start_time = time.time()

        for epoch in range(self.args.epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'val')
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.args.epochs - epoch - 1)

            print(f"\n  Train loss: {train_metrics['train_loss']:.4f}")
            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° (median={val_metrics['val_1f_median']:.2f}°)")
            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}°")
            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}°")
            if 'val_error' in val_metrics:
                print(f"  Val overall: {val_metrics['val_error']:.2f}°")
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
                    'metrics': val_metrics,
                }, self.output_dir / 'best.pth')
                print(f"  [*] New best! val_error={val_error:.2f}°")

        # Test
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        test_metrics = self.validate(self.test_loader, 'test')
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}")

        torch.save({
            'epoch': self.args.epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'test_metrics': test_metrics,
        }, self.output_dir / 'final.pth')

        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f}h")
        print(f"Best val error: {self.best_val_error:.2f}°")

        if self.use_wandb:
            wandb.finish()

        return self.best_val_error


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train MuOnly on Full Dataset')

    # Data
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_rotations', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--backbone_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lambda_mu', type=float, default=2.0)

    # Logging
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation')

    args = parser.parse_args()

    if args.no_wandb:
        args.wandb = False

    print("=" * 70)
    print("MuOnly Full Training")
    print("=" * 70)
    print(f"Dataset: FULL (no filtering)")
    print(f"Rotations: {args.num_rotations}x (truly random)")
    print(f"Epochs: {args.epochs}")
    print(f"WandB project: {args.wandb_project}")
    print("=" * 70)

    trainer = MuOnlyFullTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

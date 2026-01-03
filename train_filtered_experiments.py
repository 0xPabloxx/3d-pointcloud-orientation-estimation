#!/usr/bin/env python3
"""
Filtered Experiments: mu_only vs P2v2_SoftGate

训练两个实验:
1. mu_only: 纯角度回归，不使用von Mises NLL
2. P2v2_SoftGate: 当前最佳方法

数据过滤:
- 1-front: 只使用 airplane 和 chair (排除 bookshelf, wardrobe, bathtub, bench)
- 其他类别: 正常使用所有数据
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ProbabilisticOrientationNet, MaskedExpertLoss
from train_symmetry_classifier import SymmetryClassifier
from datasets.moe_dataset import read_ply, LABEL_NAMES, SYMMETRY_TO_LABEL

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Filtered Dataset
# =============================================================================

class FilteredMoEDataset(Dataset):
    """
    MoE Dataset with category filtering for 1-front.

    对于 1-front 类别，只使用指定的物体类别（如 airplane, chair）
    其他类别正常使用所有数据
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
                 num_rotations: int = 4,
                 align_pointcloud: bool = False,
                 seed: int = 42,
                 # Filtering options
                 allowed_1front_categories: Optional[List[str]] = None):
        """
        Args:
            allowed_1front_categories: List of allowed object categories for 1-front.
                                       e.g., ['airplane', 'chair']
                                       If None, use all categories.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.num_rotations = num_rotations
        self.align_pointcloud = align_pointcloud
        self.allowed_1front_categories = allowed_1front_categories

        self.samples = self._load_annotations(annotation_file, split, seed)
        self._print_stats()

    def _get_object_category(self, file_path: str) -> str:
        """Extract object category from file path like 'airplane/airplane_0001.ply'"""
        # file_path is like "airplane/airplane_0001.ply"
        parts = file_path.split('/')
        if len(parts) >= 1:
            return parts[0]
        return file_path.split('_')[0]

    def _load_annotations(self, annotation_file: str, split: str, seed: int) -> List[Dict]:
        """Load annotations with filtering."""
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        # Stats for filtering
        filtered_count = 0

        for file_path, ann in all_annotations.items():
            symmetry_name = ann.get('symmetry_name')
            direction = ann.get('front_direction')

            if not symmetry_name:
                continue
            if direction in excluded_directions:
                continue

            label = SYMMETRY_TO_LABEL.get(symmetry_name)
            if label is None:
                continue

            # Apply 1-front category filter
            if label == 0 and self.allowed_1front_categories is not None:
                obj_category = self._get_object_category(file_path)
                if obj_category not in self.allowed_1front_categories:
                    filtered_count += 1
                    continue

            ply_path = self.data_dir / file_path
            if not ply_path.exists():
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'symmetry_name': symmetry_name,
                'label': label,
                'direction': direction,
                'object_category': self._get_object_category(file_path),
            })

        if filtered_count > 0:
            print(f"  [Filter] Excluded {filtered_count} 1-front samples (not in {self.allowed_1front_categories})")

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
        """Print dataset statistics."""
        label_counts = Counter(s['label'] for s in self.samples)

        if self.num_rotations > 1:
            rot_type = "random" if self.augment else "deterministic"
            aug_str = f" x{self.num_rotations} ({rot_type})"
        else:
            aug_str = ""

        print(f"[FilteredMoEDataset {self.split}] {len(self.samples)} samples{aug_str} = {len(self)} total")
        for label in range(5):
            count = label_counts.get(label, 0)
            print(f"  {label} ({LABEL_NAMES[label]}): {count}")

        # Print 1-front category breakdown
        if self.allowed_1front_categories:
            one_front_cats = Counter(s['object_category'] for s in self.samples if s['label'] == 0)
            print(f"  1-front categories: {dict(one_front_cats)}")

    def __len__(self):
        return len(self.samples) * self.num_rotations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = idx // self.num_rotations
        rotation_idx = idx % self.num_rotations
        sample = self.samples[sample_idx]

        points = read_ply(sample['ply_path'])

        # Subsample
        if len(points) > self.num_points:
            if self.split == 'train':
                indices = np.random.choice(len(points), self.num_points, replace=False)
            else:
                np.random.seed(idx)
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

        # Apply rotation augmentation
        if self.num_rotations > 1:
            if self.augment:
                rotation_angle = np.random.uniform(0, 2 * np.pi)
            else:
                rotation_angle = rotation_idx * (2 * np.pi / self.num_rotations)

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
        """Get sample weights for balanced sampling."""
        label_counts = Counter(s['label'] for s in self.samples)
        class_weights = {label: 1.0 / count for label, count in label_counts.items()}

        sample_weights = []
        for s in self.samples:
            sample_weights.append(class_weights[s['label']])
        return torch.tensor(sample_weights)


def collate_fn(batch):
    return {
        'points': torch.stack([b['points'] for b in batch]),
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'gt_label': torch.stack([b['gt_label'] for b in batch]),
        'file': [b['file'] for b in batch],
    }


def get_filtered_dataloaders(
    annotation_file: str = 'data_annotation/symmetry_annotations.json',
    data_dir: str = 'data/full_mn40_normal_resampled_ply',
    batch_size: int = 32,
    num_points: int = 2048,
    num_rotations: int = 4,
    val_num_rotations: int = 4,
    num_workers: int = 4,
    use_balanced_sampler: bool = True,
    seed: int = 42,
    allowed_1front_categories: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create filtered dataloaders."""

    train_dataset = FilteredMoEDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='train',
        num_points=num_points,
        augment=True,
        num_rotations=num_rotations,
        seed=seed,
        allowed_1front_categories=allowed_1front_categories
    )

    val_dataset = FilteredMoEDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='val',
        num_points=num_points,
        augment=False,
        num_rotations=val_num_rotations,
        seed=seed,
        allowed_1front_categories=allowed_1front_categories
    )

    test_dataset = FilteredMoEDataset(
        annotation_file=annotation_file,
        data_dir=data_dir,
        split='test',
        num_points=num_points,
        augment=False,
        num_rotations=val_num_rotations,
        seed=seed,
        allowed_1front_categories=allowed_1front_categories
    )

    # Balanced sampler for training
    if use_balanced_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(num_rotations)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Mu-Only Loss (MF_1c style)
# =============================================================================

class MuOnlyLoss(nn.Module):
    """
    Mu-Only Loss: 纯角度回归，不使用von Mises NLL

    与MF_1c相同的思想，但适配MoE架构：
    - 1-front: cosine loss
    - 2-front: Hungarian matching + cosine loss
    - 4-front: Hungarian matching + cosine loss
    """

    def __init__(self, lambda_mu: float = 2.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, outputs: Dict, gt_angle: torch.Tensor, gt_label: torch.Tensor,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        device = gt_angle.device
        B = gt_angle.shape[0]

        # Extract predictions
        mu_1f = outputs['head_1front']['mu']  # [B, 1]
        mu_2f = outputs['head_2front']['mu']  # [B, 2]
        mu_4f = outputs['head_4front']['mu']  # [B, 4]

        total_loss = torch.tensor(0.0, device=device)
        loss_1f = torch.tensor(0.0, device=device)
        loss_2f = torch.tensor(0.0, device=device)
        loss_4f = torch.tensor(0.0, device=device)
        count_1f, count_2f, count_4f = 0, 0, 0

        for i in range(B):
            label = gt_label[i].item()
            gt = gt_angle[i]

            if label == 0:  # 1-front
                pred = mu_1f[i, 0]
                loss = 1 - torch.cos(pred - gt)
                loss_1f = loss_1f + loss
                count_1f += 1

            elif label == 1:  # 2-front
                gt_peaks = torch.stack([gt, (gt + np.pi) % (2 * np.pi)])
                pred_peaks = mu_2f[i]
                loss = self._hungarian_cosine_loss(pred_peaks, gt_peaks)
                loss_2f = loss_2f + loss
                count_2f += 1

            elif label == 2:  # 4-front
                gt_peaks = torch.stack([(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)])
                pred_peaks = mu_4f[i]
                loss = self._hungarian_cosine_loss(pred_peaks, gt_peaks)
                loss_4f = loss_4f + loss
                count_4f += 1

        # Average losses
        if count_1f > 0:
            loss_1f = loss_1f / count_1f
        if count_2f > 0:
            loss_2f = loss_2f / count_2f
        if count_4f > 0:
            loss_4f = loss_4f / count_4f

        total_loss = self.lambda_mu * (loss_1f + loss_2f + loss_4f)

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
        """Hungarian matching with cosine loss."""
        n = len(gt)

        # Build cost matrix
        cost = torch.zeros(n, n, device=pred.device)
        for i in range(n):
            for j in range(n):
                cost[i, j] = 1 - torch.cos(pred[i] - gt[j])

        # Hungarian matching
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Compute matched loss
        loss = torch.tensor(0.0, device=pred.device)
        for i, j in zip(row_ind, col_ind):
            loss = loss + cost[i, j]

        return loss / n


# =============================================================================
# Classifier Wrapper
# =============================================================================

class ClassifierWrapper(nn.Module):
    """Wrapper for SymmetryClassifier to match ProbabilisticOrientationNet interface."""

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, points, upright_vec=None):
        return self.classifier(points)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, args, experiment_name: str, loss_type: str = 'p2v2'):
        self.args = args
        self.experiment_name = experiment_name
        self.loss_type = loss_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'checkpoints/{experiment_name}_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = vars(args).copy()
        config['experiment_name'] = experiment_name
        config['loss_type'] = loss_type
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize wandb
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f'{experiment_name}_{timestamp}',
                config=config
            )

        # Data loaders (filtered)
        print(f"\nLoading filtered dataset (1-front: {args.allowed_1front_categories})...")
        self.train_loader, self.val_loader, self.test_loader = get_filtered_dataloaders(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_points=args.num_points,
            num_rotations=args.num_rotations,
            val_num_rotations=args.val_num_rotations,
            num_workers=args.num_workers,
            use_balanced_sampler=args.balanced_sampler,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories
        )

        # Load classifier
        print(f"\nLoading classifier from: {args.classifier_checkpoint}")
        base_classifier = SymmetryClassifier(num_classes=5)
        ckpt = torch.load(args.classifier_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            base_classifier.load_state_dict(ckpt['model_state_dict'])
        else:
            base_classifier.load_state_dict(ckpt)
        classifier = ClassifierWrapper(base_classifier).to(self.device)
        print("Classifier loaded successfully")

        # Create model
        self.model = ProbabilisticOrientationNet(
            classifier=classifier,
            backbone_dim=args.backbone_dim,
            expert_hidden_dim=args.expert_hidden_dim,
            kappa_min=args.kappa_min,
            kappa_max=args.kappa_max,
            freeze_classifier=args.freeze_classifier
        ).to(self.device)

        # Loss function
        if loss_type == 'mu_only':
            self.criterion = MuOnlyLoss(lambda_mu=args.lambda_mu)
        else:  # p2v2
            self.criterion = MaskedExpertLoss(
                classification_weight=0.0,
                use_gate_weighting=True,
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

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # Training state
        self.best_val_loss = float('inf')
        self.best_val_error = float('inf')

        # Print info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}_{timestamp}")
        print(f"Loss type: {loss_type}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Output dir: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.get_url()}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_angle = batch['gt_angle'].to(self.device)
            gt_label = batch['gt_label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)

            if self.loss_type == 'mu_only':
                loss_dict = self.criterion(outputs, gt_angle, gt_label, epoch)
            else:
                loss_dict = self.criterion(outputs, gt_angle, gt_label, epoch)

            loss = loss_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}")

        return {'train_loss': total_loss / batch_count}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        self.model.eval()

        errors_1f, errors_2f, errors_4f = [], [], []
        kappas_1f = []

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_angles = batch['gt_angle'].numpy()
            gt_labels = batch['gt_label'].numpy()

            outputs = self.model(points)

            mu_1f = outputs['head_1front']['mu'].cpu().numpy()
            kappa_1f = outputs['head_1front']['kappa'].cpu().numpy()
            mu_2f = outputs['head_2front']['mu'].cpu().numpy()
            mu_4f = outputs['head_4front']['mu'].cpu().numpy()

            for i in range(len(gt_labels)):
                label = gt_labels[i]
                gt = gt_angles[i]

                if label == 0:  # 1-front
                    pred = mu_1f[i, 0]
                    err = self._circular_error(pred, gt)
                    errors_1f.append(err)
                    kappas_1f.append(kappa_1f[i, 0])

                elif label == 1:  # 2-front
                    gt_peaks = [gt, (gt + np.pi) % (2 * np.pi)]
                    pred_peaks = [mu_2f[i, j] for j in range(2)]
                    errs = self._hungarian_errors(pred_peaks, gt_peaks)
                    errors_2f.extend(errs)

                elif label == 2:  # 4-front
                    gt_peaks = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
                    pred_peaks = [mu_4f[i, j] for j in range(4)]
                    errs = self._hungarian_errors(pred_peaks, gt_peaks)
                    errors_4f.extend(errs)

        results = {}

        if errors_1f:
            errors_1f = np.array(errors_1f) * 180 / np.pi
            results['val_1f_error'] = float(np.mean(errors_1f))
            results['val_1f_median'] = float(np.median(errors_1f))
            results['val_1f_lt10'] = float(100 * np.mean(errors_1f < 10))
            results['val_1f_kappa'] = float(np.mean(kappas_1f))

        if errors_2f:
            errors_2f = np.array(errors_2f) * 180 / np.pi
            results['val_2f_error'] = float(np.mean(errors_2f))
            results['val_2f_lt10'] = float(100 * np.mean(errors_2f < 10))

        if errors_4f:
            errors_4f = np.array(errors_4f) * 180 / np.pi
            results['val_4f_error'] = float(np.mean(errors_4f))
            results['val_4f_lt10'] = float(100 * np.mean(errors_4f < 10))

        # Overall error (weighted by sample count)
        all_errors = []
        if errors_1f is not None and len(errors_1f) > 0:
            all_errors.extend(errors_1f.tolist())
        if errors_2f is not None and len(errors_2f) > 0:
            all_errors.extend(errors_2f.tolist())
        if errors_4f is not None and len(errors_4f) > 0:
            all_errors.extend(errors_4f.tolist())

        if all_errors:
            results['val_error'] = float(np.mean(all_errors))

        return results

    def _circular_error(self, pred: float, gt: float) -> float:
        """Compute circular angular error in radians."""
        pred = pred % (2 * np.pi)
        gt = gt % (2 * np.pi)
        diff = abs(pred - gt)
        return min(diff, 2 * np.pi - diff)

    def _hungarian_errors(self, pred_peaks: List[float], gt_peaks: List[float]) -> List[float]:
        """Compute Hungarian-matched errors."""
        n = len(gt_peaks)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = self._circular_error(pred_peaks[i], gt_peaks[j])

        row_ind, col_ind = linear_sum_assignment(cost)
        return [cost[i, j] for i, j in zip(row_ind, col_ind)]

    def train(self):
        print("\nStarting training...")
        print("=" * 60)

        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 40)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log
            lr = self.scheduler.get_last_lr()[0]
            print(f"\n  Train loss: {train_metrics['train_loss']:.4f}")
            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° (κ={val_metrics.get('val_1f_kappa', 0):.1f})")
            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}°")
            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}°")
            print(f"  LR: {lr:.6f}")

            # WandB logging
            if self.use_wandb:
                log_dict = {**train_metrics, **val_metrics, 'lr': lr, 'epoch': epoch}
                wandb.log(log_dict)

            # Save best model
            val_error = val_metrics.get('val_error', float('inf'))
            if val_error < self.best_val_error:
                self.best_val_error = val_error
                self.save_checkpoint('best.pth', epoch, val_metrics)
                print(f"  [*] New best model saved! (error={val_error:.2f}°)")

        # Save final model
        self.save_checkpoint('final.pth', self.args.epochs - 1, val_metrics)

        print("\n" + "=" * 60)
        print(f"Training complete! Best val error: {self.best_val_error:.2f}°")
        print(f"Checkpoints saved to: {self.output_dir}")

        if self.use_wandb:
            wandb.finish()

        return self.best_val_error

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_error': self.best_val_error,
        }, self.output_dir / filename)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Filtered Experiments: mu_only vs P2v2')

    # Data
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--classifier_checkpoint', type=str,
                        default='checkpoints/SymClassifier_20251216_035345/best.pth')

    # Filtering
    parser.add_argument('--allowed_1front_categories', type=str, nargs='+',
                        default=['airplane', 'chair'],
                        help='Allowed object categories for 1-front')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_rotations', type=int, default=4)
    parser.add_argument('--val_num_rotations', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--balanced_sampler', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--backbone_dim', type=int, default=1024)
    parser.add_argument('--expert_hidden_dim', type=int, default=256)
    parser.add_argument('--kappa_min', type=float, default=1e-4)
    parser.add_argument('--kappa_max', type=float, default=100.0)
    parser.add_argument('--freeze_classifier', type=bool, default=True)

    # Mu-only loss
    parser.add_argument('--lambda_mu', type=float, default=2.0)

    # P2v2 loss
    parser.add_argument('--p_dir_threshold', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--cosine_weight_init', type=float, default=0.2)
    parser.add_argument('--cosine_weight_final', type=float, default=0.1)
    parser.add_argument('--cosine_schedule_epoch', type=int, default=10)
    parser.add_argument('--kappa_reg_weight_final', type=float, default=0.02)
    parser.add_argument('--kappa_target_final', type=float, default=5.0)
    parser.add_argument('--kappa_reg_start_epoch', type=int, default=6)
    parser.add_argument('--kappa_reg_ramp_epochs', type=int, default=10)

    # Logging
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-FilteredExp')

    # Experiment selection
    parser.add_argument('--exp', type=str, choices=['mu_only', 'p2v2', 'both'],
                        default='both', help='Which experiment to run')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Filtered Experiments: mu_only vs P2v2_SoftGate")
    print("=" * 70)
    print(f"1-front filter: only {args.allowed_1front_categories}")
    print("=" * 70)

    results = {}

    if args.exp in ['mu_only', 'both']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: mu_only (MF_1c style)")
        print("=" * 70)

        trainer = Trainer(args, 'MuOnly_Filtered', loss_type='mu_only')
        results['mu_only'] = trainer.train()

    if args.exp in ['p2v2', 'both']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: P2v2_SoftGate")
        print("=" * 70)

        trainer = Trainer(args, 'P2v2_Filtered', loss_type='p2v2')
        results['p2v2'] = trainer.train()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, error in results.items():
        print(f"  {name}: {error:.2f}°")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Clean Training Pipeline: Classifier + P2v2

完整训练流程:
1. 训练新的分类器（使用过滤后的干净数据）
2. 训练P2v2（排除异常值，强旋转增强）

数据过滤:
- 1-front: 只使用 airplane 和 chair
- 排除 1front_outliers.json 中的异常样本
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
from typing import Dict, List, Tuple, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ProbabilisticOrientationNet, MaskedExpertLoss
from train_symmetry_classifier import SymmetryClassifier, PointNetPlusPlusEncoder
from datasets.moe_dataset import read_ply, LABEL_NAMES, SYMMETRY_TO_LABEL

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Clean Dataset with Outlier Exclusion
# =============================================================================

class CleanMoEDataset(Dataset):
    """
    Clean MoE Dataset:
    - 1-front只用指定类别（airplane, chair）
    - 排除异常值列表中的样本
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
                 seed: int = 42,
                 # Filtering
                 allowed_1front_categories: Optional[List[str]] = None,
                 exclude_samples: Optional[Set[str]] = None,
                 outlier_json: Optional[str] = None,
                 outlier_threshold: str = 'severe'):  # 'severe', 'major', 'moderate'
        """
        Args:
            allowed_1front_categories: 1-front允许的物体类别
            exclude_samples: 要排除的样本名集合
            outlier_json: 异常值JSON文件路径
            outlier_threshold: 排除阈值 ('severe'>=90°, 'major'>=45°, 'moderate'>=15°)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.num_rotations = num_rotations
        self.allowed_1front_categories = allowed_1front_categories

        # Load outlier exclusion list
        self.exclude_samples = exclude_samples or set()
        if outlier_json and os.path.exists(outlier_json):
            with open(outlier_json, 'r') as f:
                outlier_data = json.load(f)
            key = f'exclude_list_{outlier_threshold}'
            if key in outlier_data:
                self.exclude_samples = self.exclude_samples | set(outlier_data[key])
                print(f"  [Outlier] Loaded {len(outlier_data[key])} samples to exclude ({outlier_threshold})")

        self.samples = self._load_annotations(annotation_file, split, seed)
        self._print_stats()

    def _get_object_category(self, file_path: str) -> str:
        """Extract object category from file path"""
        parts = file_path.split('/')
        if len(parts) >= 1:
            return parts[0]
        return file_path.split('_')[0]

    def _get_sample_name(self, file_path: str) -> str:
        """Extract sample name (e.g., 'airplane_0001' from 'airplane/airplane_0001.ply')"""
        fname = file_path.split('/')[-1] if '/' in file_path else file_path
        return fname.replace('.ply', '')

    def _load_annotations(self, annotation_file: str, split: str, seed: int) -> List[Dict]:
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}
        filter_stats = {'category': 0, 'outlier': 0}

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

            # Filter 1-front by category
            if label == 0 and self.allowed_1front_categories is not None:
                if obj_category not in self.allowed_1front_categories:
                    filter_stats['category'] += 1
                    continue

            # Filter by outlier list
            if sample_name in self.exclude_samples:
                filter_stats['outlier'] += 1
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
                'object_category': obj_category,
                'sample_name': sample_name,
            })

        if filter_stats['category'] > 0:
            print(f"  [Filter] Excluded {filter_stats['category']} samples by category")
        if filter_stats['outlier'] > 0:
            print(f"  [Filter] Excluded {filter_stats['outlier']} outlier samples")

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

        print(f"[CleanMoEDataset {self.split}] {len(self.samples)} samples{aug_str} = {len(self)} total")
        for label in range(5):
            count = label_counts.get(label, 0)
            print(f"  {label} ({LABEL_NAMES[label]}): {count}")

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
            if self.augment:
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

        # Apply rotation
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
# Classifier Training
# =============================================================================

class ClassifierTrainer:
    """Train a new SymmetryClassifier on clean data."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'checkpoints/CleanClassifier_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = vars(args).copy()
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # WandB
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f'CleanClassifier_{timestamp}',
                config=config
            )

        # Data
        print("\n" + "=" * 60)
        print("Loading clean data for classifier training...")
        print("=" * 60)

        self.train_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='train',
            num_points=args.num_points,
            augment=True,
            num_rotations=args.classifier_num_rotations,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        self.val_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='val',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        # Balanced sampler
        sample_weights = self.train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(args.classifier_num_rotations)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )

        # Model
        self.model = SymmetryClassifier(num_classes=5).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.classifier_lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.classifier_epochs, eta_min=1e-6
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc = 0.0

        print(f"\nClassifier output dir: {self.output_dir}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.get_url()}")

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            points = batch['points'].to(self.device)
            labels = batch['gt_label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(points)  # SymmetryClassifier returns logits directly

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100.0 * correct / total
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            labels = batch['gt_label'].to(self.device)

            logits = self.model(points)  # SymmetryClassifier returns logits directly

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Overall accuracy
        val_acc = 100.0 * np.mean(all_preds == all_labels)

        # Per-class accuracy
        results = {
            'val_acc': val_acc,
            'val_loss': total_loss / len(self.val_loader)
        }

        for label_idx, label_name in enumerate(LABEL_NAMES):
            mask = all_labels == label_idx
            if mask.sum() > 0:
                class_acc = 100.0 * np.mean(all_preds[mask] == all_labels[mask])
                results[f'val_acc_{label_name}'] = class_acc

        return results

    def train(self) -> str:
        print("\n" + "=" * 60)
        print("Training Classifier...")
        print("=" * 60)

        for epoch in range(self.args.classifier_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.args.classifier_epochs}: "
                  f"train_acc={train_metrics['train_acc']:.1f}%, "
                  f"val_acc={val_metrics['val_acc']:.1f}%")

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.scheduler.get_last_lr()[0]
                })

            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': self.best_val_acc,
                }, self.output_dir / 'best.pth')
                print(f"  [*] New best! val_acc={self.best_val_acc:.2f}%")

        print(f"\nClassifier training complete! Best val_acc: {self.best_val_acc:.2f}%")

        if self.use_wandb:
            wandb.finish()

        return str(self.output_dir / 'best.pth')


# =============================================================================
# Baseline Model (MF_1c style - PointNet++ + MixturePeakHead)
# =============================================================================

class MixturePeakHead(nn.Module):
    """
    混合 von Mises 预测头 (原始MF系列架构)
    输出 4 个 (mu, kappa) 对
    """
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512, num_peaks: int = 4):
        super().__init__()
        self.num_peaks = num_peaks

        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # mu: 方向向量 (会被归一化为角度)
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)

        # kappa: 集中度 (通过softplus保证非负)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_channels) 编码器特征
        Returns:
            mu: (B, 4) 角度值 [0, 2π)
            kappa: (B, 4) 集中度参数
        """
        h = self.shared(x)

        # mu: 预测方向向量，归一化后转为角度
        mu_vec = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu_vec = F.normalize(mu_vec, dim=-1)
        # 转换为角度
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])  # (B, 4)
        mu = (mu + 2 * np.pi) % (2 * np.pi)  # 确保在 [0, 2π) 范围

        # kappa: 集中度，使用softplus保证非负
        kappa = F.softplus(self.kappa_head(h))

        return mu, kappa


class BaselineDirectionModel(nn.Module):
    """
    原始MF系列基线模型: PointNet++ + MixturePeakHead
    不使用分类器，不使用MoE架构
    统一输出4个峰，根据GT label解释
    """
    def __init__(self, backbone_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=backbone_dim)
        self.head = MixturePeakHead(in_channels=backbone_dim, hidden_channels=hidden_dim)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) 点云
        Returns:
            dict with 'mu' (B, 4) and 'kappa' (B, 4)
        """
        features = self.encoder(points)
        mu, kappa = self.head(features)
        return {'mu': mu, 'kappa': kappa}


# =============================================================================
# Baseline Mu-Only Loss (MF_1c style - pure cosine loss)
# =============================================================================

class BaselineMuOnlyLoss(nn.Module):
    """
    原始MF_1c风格的Mu-Only Loss: 纯角度回归
    - 不使用von Mises NLL
    - 使用cosine距离 + Hungarian匹配

    GT生成规则:
    - 1-front: 4峰同方向
    - 2-front: 2峰间隔180°
    - 4-front: 4峰间隔90°
    - symmetric/no_front: 跳过
    """

    def __init__(self, lambda_mu: float = 2.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, outputs: Dict, gt_angle: torch.Tensor, gt_label: torch.Tensor,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        device = gt_angle.device
        B = gt_angle.shape[0]

        pred_mu = outputs['mu']  # (B, 4)

        total_loss = torch.tensor(0.0, device=device)
        loss_1f = torch.tensor(0.0, device=device)
        loss_2f = torch.tensor(0.0, device=device)
        loss_4f = torch.tensor(0.0, device=device)
        count_1f, count_2f, count_4f = 0, 0, 0

        for i in range(B):
            label = gt_label[i].item()
            gt = gt_angle[i]
            pred = pred_mu[i]  # (4,)

            if label == 0:  # 1-front: 4峰同方向
                # 计算所有4个预测峰与GT的cosine损失，取平均
                loss = torch.mean(1 - torch.cos(pred - gt))
                loss_1f = loss_1f + loss
                count_1f += 1

            elif label == 1:  # 2-front: 2峰间隔180°
                gt_peaks = torch.stack([gt, (gt + np.pi) % (2 * np.pi)])
                # 只用前2个峰
                pred_peaks = pred[:2]
                loss = self._hungarian_cosine_loss(pred_peaks, gt_peaks)
                loss_2f = loss_2f + loss
                count_2f += 1

            elif label == 2:  # 4-front: 4峰间隔90°
                gt_peaks = torch.stack([(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)])
                loss = self._hungarian_cosine_loss(pred, gt_peaks)
                loss_4f = loss_4f + loss
                count_4f += 1
            # label == 3 (symmetric) 和 label == 4 (no_front) 跳过

        if count_1f > 0:
            loss_1f = loss_1f / count_1f
        if count_2f > 0:
            loss_2f = loss_2f / count_2f
        if count_4f > 0:
            loss_4f = loss_4f / count_4f

        # 只计算有样本的类别
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
# P2v2 Training
# =============================================================================

class ClassifierWrapper(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, points, upright_vec=None):
        return self.classifier(points)


class P2v2Trainer:
    """Train P2v2 with clean data and strong augmentation."""

    def __init__(self, args, classifier_path: str, resume_dir: str = None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.resume_checkpoint = None

        if resume_dir:
            self.output_dir = Path(resume_dir)
            # Try latest.pth first, then best.pth
            checkpoint_path = self.output_dir / 'latest.pth'
            if not checkpoint_path.exists():
                checkpoint_path = self.output_dir / 'best.pth'
            if checkpoint_path.exists():
                self.resume_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                self.start_epoch = self.resume_checkpoint['epoch'] + 1
                print(f"\n[Resume] Loading from {checkpoint_path}")
                print(f"[Resume] Will continue from epoch {self.start_epoch}")
            else:
                print(f"\n[Resume] Warning: No checkpoint found in {resume_dir}, starting fresh")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'checkpoints/P2v2_Clean_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config (only if not resuming)
        config = vars(args).copy()
        config['classifier_path'] = classifier_path
        if not resume_dir:
            with open(self.output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

        # WandB - support resume
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        self.wandb_run_id = None
        if self.use_wandb:
            if self.resume_checkpoint and 'wandb_run_id' in self.resume_checkpoint:
                # Resume existing wandb run
                self.wandb_run_id = self.resume_checkpoint['wandb_run_id']
                wandb.init(
                    project=args.wandb_project,
                    id=self.wandb_run_id,
                    resume="must",
                    config=config
                )
                print(f"[Resume] Continuing wandb run: {self.wandb_run_id}")
            else:
                # New wandb run
                run_name = self.output_dir.name if resume_dir else f'P2v2_Clean_{timestamp}'
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=config
                )
                self.wandb_run_id = wandb.run.id
                print(f"[WandB] New run ID: {self.wandb_run_id}")

        # Data with strong augmentation
        print("\n" + "=" * 60)
        print("Loading clean data for P2v2 training...")
        print(f"Training rotations: {args.p2v2_num_rotations}")
        print("=" * 60)

        self.train_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='train',
            num_points=args.num_points,
            augment=True,
            num_rotations=args.p2v2_num_rotations,  # Strong augmentation
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        self.val_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='val',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        self.test_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='test',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        # Balanced sampler
        sample_weights = self.train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(args.p2v2_num_rotations)
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

        # Load classifier
        print(f"\nLoading classifier from: {classifier_path}")
        base_classifier = SymmetryClassifier(num_classes=5)
        ckpt = torch.load(classifier_path, map_location='cpu', weights_only=False)
        base_classifier.load_state_dict(ckpt['model_state_dict'])
        classifier = ClassifierWrapper(base_classifier).to(self.device)
        print("Classifier loaded successfully")

        # Model
        self.model = ProbabilisticOrientationNet(
            classifier=classifier,
            backbone_dim=args.backbone_dim,
            expert_hidden_dim=args.expert_hidden_dim,
            kappa_min=args.kappa_min,
            kappa_max=args.kappa_max,
            freeze_classifier=True
        ).to(self.device)

        # Loss
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
            self.model.parameters(), lr=args.p2v2_lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.p2v2_epochs, eta_min=1e-6
        )

        self.best_val_error = float('inf')

        # Load checkpoint if resuming
        if self.resume_checkpoint is not None:
            self.model.load_state_dict(self.resume_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in self.resume_checkpoint:
                self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
            if 'best_val_error' in self.resume_checkpoint:
                self.best_val_error = self.resume_checkpoint['best_val_error']
            print(f"[Resume] Loaded model, optimizer, scheduler states")
            print(f"[Resume] Best val error so far: {self.best_val_error:.2f}°")

        # Print info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"P2v2 Training Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Epochs: {args.p2v2_epochs}")
        print(f"Output dir: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.get_url()}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self.criterion.set_epoch(epoch)  # Set epoch for scheduled parameters
        total_loss = 0.0
        total_nll = 0.0
        total_cosine = 0.0
        total_kappa_reg = 0.0
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
            # Collect loss components if available
            if 'nll_loss' in loss_dict:
                total_nll += loss_dict['nll_loss'].item() if isinstance(loss_dict['nll_loss'], torch.Tensor) else loss_dict['nll_loss']
            if 'cosine_loss' in loss_dict:
                total_cosine += loss_dict['cosine_loss'].item() if isinstance(loss_dict['cosine_loss'], torch.Tensor) else loss_dict['cosine_loss']
            if 'kappa_reg' in loss_dict:
                total_kappa_reg += loss_dict['kappa_reg'].item() if isinstance(loss_dict['kappa_reg'], torch.Tensor) else loss_dict['kappa_reg']
            batch_count += 1

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}")

        results = {'train_loss': total_loss / batch_count}
        if total_nll > 0:
            results['train_nll'] = total_nll / batch_count
        if total_cosine > 0:
            results['train_cosine'] = total_cosine / batch_count
        if total_kappa_reg > 0:
            results['train_kappa_reg'] = total_kappa_reg / batch_count

        return results

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
            results[f'{name}_1f_std'] = float(np.std(errors_1f))
            results[f'{name}_1f_lt5'] = float(100 * np.mean(errors_1f < 5))
            results[f'{name}_1f_lt10'] = float(100 * np.mean(errors_1f < 10))
            results[f'{name}_1f_lt15'] = float(100 * np.mean(errors_1f < 15))
            results[f'{name}_1f_gt45'] = float(100 * np.mean(errors_1f > 45))
            results[f'{name}_1f_gt90'] = float(100 * np.mean(errors_1f > 90))
            results[f'{name}_1f_kappa'] = float(np.mean(kappas_1f))
            results[f'{name}_1f_kappa_std'] = float(np.std(kappas_1f))

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
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = self._circular_error(pred_peaks[i], gt_peaks[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        return [cost[i, j] for i, j in zip(row_ind, col_ind)]

    def train(self):
        print("\n" + "=" * 60)
        print("Starting P2v2 Training...")
        if self.start_epoch > 0:
            print(f"[Resume] Continuing from epoch {self.start_epoch + 1}")
        print("=" * 60)

        import time
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.p2v2_epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.args.p2v2_epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'val')
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.args.p2v2_epochs - epoch - 1)

            # Log
            print(f"\n  Train loss: {train_metrics['train_loss']:.4f}")
            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° "
                      f"(median={val_metrics['val_1f_median']:.2f}°, "
                      f"<10°={val_metrics['val_1f_lt10']:.1f}%, "
                      f"κ={val_metrics['val_1f_kappa']:.1f})")
            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}° (<10°={val_metrics['val_2f_lt10']:.1f}%)")
            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}° (<10°={val_metrics['val_4f_lt10']:.1f}%)")
            print(f"  Time: {epoch_time:.0f}s | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

            # WandB
            if self.use_wandb:
                log_dict = {
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.scheduler.get_last_lr()[0]
                }
                wandb.log(log_dict)

            # Save latest (for resume) - complete state
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_error': self.best_val_error,
                'wandb_run_id': self.wandb_run_id,
                'metrics': val_metrics,
            }, self.output_dir / 'latest.pth')

            # Save best
            val_error = val_metrics.get('val_error', float('inf'))
            if val_error < self.best_val_error:
                self.best_val_error = val_error
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_error': self.best_val_error,
                    'wandb_run_id': self.wandb_run_id,
                    'metrics': val_metrics,
                }, self.output_dir / 'best.pth')
                print(f"  [*] New best! val_error={val_error:.2f}°")

        # Final test evaluation
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        test_metrics = self.validate(self.test_loader, 'test')
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}")

        # Save final
        torch.save({
            'epoch': self.args.p2v2_epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'test_metrics': test_metrics,
        }, self.output_dir / 'final.pth')

        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f}h")
        print(f"Best val error: {self.best_val_error:.2f}°")
        print(f"Checkpoints saved to: {self.output_dir}")

        if self.use_wandb:
            wandb.finish()

        return self.best_val_error


# =============================================================================
# Mu-Only Trainer (Baseline - NO MoE)
# =============================================================================

class MuOnlyTrainer:
    """
    原始MF_1c风格的Baseline训练器
    - 使用 PointNet++ + MixturePeakHead (不使用MoE架构)
    - 纯cosine损失，不使用von Mises NLL
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'checkpoints/MuOnly_Baseline_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = vars(args).copy()
        config['model_type'] = 'baseline'
        config['loss_type'] = 'mu_only'
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # WandB
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f'MuOnly_Baseline_{timestamp}',
                config=config
            )

        # Data
        print("\n" + "=" * 60)
        print("Loading clean data for MuOnly Baseline training...")
        print(f"Model: PointNet++ + MixturePeakHead (NO MoE)")
        print(f"Training rotations: {args.muonly_num_rotations}")
        print("=" * 60)

        self.train_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='train',
            num_points=args.num_points,
            augment=True,
            num_rotations=args.muonly_num_rotations,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        self.val_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='val',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        self.test_dataset = CleanMoEDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_dir,
            split='test',
            num_points=args.num_points,
            augment=False,
            num_rotations=4,
            seed=args.seed,
            allowed_1front_categories=args.allowed_1front_categories,
            outlier_json=args.outlier_json,
            outlier_threshold=args.outlier_threshold
        )

        # Balanced sampler
        sample_weights = self.train_dataset.get_sample_weights()
        sample_weights = sample_weights.repeat_interleave(args.muonly_num_rotations)
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

        # Baseline Model (PointNet++ + MixturePeakHead, NO classifier, NO MoE)
        self.model = BaselineDirectionModel(
            backbone_dim=args.backbone_dim,
            hidden_dim=args.expert_hidden_dim
        ).to(self.device)

        # Baseline Mu-Only Loss
        self.criterion = BaselineMuOnlyLoss(lambda_mu=args.lambda_mu)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.muonly_lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.muonly_epochs, eta_min=1e-6
        )

        self.best_val_error = float('inf')

        # Print info
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"\n{'='*60}")
        print(f"MuOnly Baseline Training Configuration")
        print(f"{'='*60}")
        print(f"Model: BaselineDirectionModel (NO MoE)")
        print(f"Parameters: {total_params:,}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Epochs: {args.muonly_epochs}")
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
            pred_mu = outputs['mu'].cpu().numpy()  # (B, 4)
            pred_kappa = outputs['kappa'].cpu().numpy()  # (B, 4)

            for i in range(len(gt_labels)):
                label = gt_labels[i]
                gt = gt_angles[i]

                if label == 0:  # 1-front
                    # 对于1-front，所有4个峰应该指向同一方向，取最接近GT的峰
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
        print("Starting MuOnly Baseline Training...")
        print("=" * 60)

        import time
        start_time = time.time()

        for epoch in range(self.args.muonly_epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.args.muonly_epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'val')
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.args.muonly_epochs - epoch - 1)

            print(f"\n  Train loss: {train_metrics['train_loss']:.4f}")
            if 'val_1f_error' in val_metrics:
                print(f"  Val 1f: {val_metrics['val_1f_error']:.2f}° (median={val_metrics['val_1f_median']:.2f}°)")
            if 'val_2f_error' in val_metrics:
                print(f"  Val 2f: {val_metrics['val_2f_error']:.2f}°")
            if 'val_4f_error' in val_metrics:
                print(f"  Val 4f: {val_metrics['val_4f_error']:.2f}°")
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
            'epoch': self.args.muonly_epochs - 1,
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

def parse_args():
    parser = argparse.ArgumentParser(description='Clean Training Pipeline')

    # Data
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply')
    parser.add_argument('--outlier_json', type=str,
                        default='data_annotation/1front_outliers.json')
    parser.add_argument('--outlier_threshold', type=str, default='severe',
                        choices=['severe', 'major', 'moderate'])

    # Filtering
    parser.add_argument('--allowed_1front_categories', type=str, nargs='+',
                        default=['airplane', 'chair'])

    # Common
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Classifier training
    parser.add_argument('--classifier_epochs', type=int, default=30)
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--classifier_num_rotations', type=int, default=10)

    # P2v2 training
    parser.add_argument('--p2v2_epochs', type=int, default=100)
    parser.add_argument('--p2v2_lr', type=float, default=1e-3)
    parser.add_argument('--p2v2_num_rotations', type=int, default=12)  # Strong augmentation

    # MuOnly training (baseline - NO MoE)
    parser.add_argument('--muonly_epochs', type=int, default=50)
    parser.add_argument('--muonly_lr', type=float, default=1e-3)
    parser.add_argument('--muonly_num_rotations', type=int, default=10)
    parser.add_argument('--lambda_mu', type=float, default=2.0)

    # Model
    parser.add_argument('--backbone_dim', type=int, default=1024)
    parser.add_argument('--expert_hidden_dim', type=int, default=256)
    parser.add_argument('--kappa_min', type=float, default=1e-4)
    parser.add_argument('--kappa_max', type=float, default=100.0)

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
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation')

    # Skip/run options
    parser.add_argument('--skip_classifier', action='store_true',
                        help='Skip classifier training, use existing checkpoint')
    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to existing classifier checkpoint')
    parser.add_argument('--skip_p2v2', action='store_true',
                        help='Skip P2v2 training')
    parser.add_argument('--run_muonly', action='store_true',
                        help='Run MuOnly baseline training')

    # Resume training
    parser.add_argument('--resume_p2v2', type=str, default=None,
                        help='Path to P2v2 checkpoint directory to resume from (e.g., checkpoints/P2v2_Clean_xxx)')
    parser.add_argument('--resume_muonly', type=str, default=None,
                        help='Path to MuOnly checkpoint directory to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Clean Training Pipeline")
    print("=" * 70)
    print(f"1-front filter: only {args.allowed_1front_categories}")
    print(f"Outlier exclusion: {args.outlier_threshold} (from {args.outlier_json})")
    print("\nPipeline steps:")
    print(f"  1. Classifier: {'skip' if args.skip_classifier else f'{args.classifier_epochs} epochs'}")
    print(f"  2. P2v2: {'skip' if args.skip_p2v2 else f'{args.p2v2_epochs} epochs, {args.p2v2_num_rotations}x rotations'}")
    print(f"  3. MuOnly Baseline: {'run' if args.run_muonly else 'skip'} ({args.muonly_epochs} epochs)")
    print("=" * 70)

    classifier_path = None

    # Step 1: Train classifier (or use existing)
    if args.skip_classifier and args.classifier_checkpoint:
        classifier_path = args.classifier_checkpoint
        print(f"\nSkipping classifier training, using: {classifier_path}")
    elif not args.skip_classifier:
        print("\n" + "=" * 70)
        print("STEP 1: Training New Classifier")
        print("=" * 70)
        classifier_trainer = ClassifierTrainer(args)
        classifier_path = classifier_trainer.train()

    # Step 2: Train P2v2 (requires classifier)
    if not args.skip_p2v2:
        if classifier_path is None and not args.resume_p2v2:
            print("\nERROR: P2v2 requires a classifier. Provide --classifier_checkpoint or don't --skip_classifier")
        else:
            print("\n" + "=" * 70)
            print("STEP 2: Training P2v2")
            if args.resume_p2v2:
                print(f"[Resume mode] Resuming from: {args.resume_p2v2}")
            print("=" * 70)
            p2v2_trainer = P2v2Trainer(args, classifier_path, resume_dir=args.resume_p2v2)
            p2v2_trainer.train()
    else:
        print("\nSkipping P2v2 training")

    # Step 3: Train MuOnly Baseline (NO classifier needed)
    if args.run_muonly:
        print("\n" + "=" * 70)
        print("STEP 3: Training MuOnly Baseline")
        print("=" * 70)
        muonly_trainer = MuOnlyTrainer(args)
        muonly_trainer.train()
    else:
        print("\nSkipping MuOnly baseline training (use --run_muonly to enable)")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

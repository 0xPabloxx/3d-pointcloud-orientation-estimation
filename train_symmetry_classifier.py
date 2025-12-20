"""
Symmetry Classifier Training - 对称性类型分类器训练脚本

使用 PointNet++ 预测5种对称性类型:
0: 1个正面
1: 2个正面
2: 4个正面
3: 旋转对称
4: 无正面

排除 OBLIQUE 和 MULTI 方向的数据

用法:
    python train_symmetry_classifier.py --exp_name SymClassifier
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.symmetry_classifier_dataset import SymmetryClassifierDataset, collate_fn


# ============== Model ==============

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape

        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=2)[0].permute(0, 2, 1)

        return new_xyz, new_points


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1024):
        super().__init__()

        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(32, 0.8, 128, 256 + 3, [256, 512, out_channels])

        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, points):
        xyz = points
        feat = None

        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        xyz, feat = self.sa3(xyz, feat)

        x = feat.max(dim=1)[0]
        x = self.fc(x)

        return x


class SymmetryClassifier(nn.Module):
    """对称性类型分类器"""

    def __init__(self, encoder_dim: int = 1024, num_classes: int = 5):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=encoder_dim)

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, points):
        features = self.encoder(points)
        logits = self.classifier(features)
        return logits


# ============== Trainer ==============

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f"{args.exp_name}_{timestamp}"
        self.output_dir = os.path.join('checkpoints', self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置
        config = vars(args).copy()
        config['script'] = 'train_symmetry_classifier.py'
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 数据集
        self.train_dataset = SymmetryClassifierDataset(
            split='train',
            num_points=args.num_points,
            augment=True,
            augment_factor=args.augment_factor,
        )
        self.val_dataset = SymmetryClassifierDataset(
            split='val',
            num_points=args.num_points,
            augment=False,
        )
        self.test_dataset = SymmetryClassifierDataset(
            split='test',
            num_points=args.num_points,
            augment=False,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

        # 类别名称
        self.class_names = SymmetryClassifierDataset.CLASS_NAMES

        # 模型
        self.model = SymmetryClassifier(
            encoder_dim=args.encoder_dim,
            num_classes=len(self.class_names)
        ).to(self.device)

        # Loss (可选类别权重)
        if args.use_class_weights:
            class_weights = self.train_dataset.get_class_weights().to(self.device)
            print(f"Class weights: {class_weights}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        self.best_val_acc = 0.0

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # WandB
        self.use_wandb = args.wandb
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=self.exp_name,
                config=config,
                dir=self.output_dir,
            )
            wandb.watch(self.model, log='all', log_freq=100)

        print(f"=" * 60)
        print(f"Symmetry Classifier Training")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_dataset)} (online augment each epoch)")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Classes: {self.class_names}")
        print(f"Output: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.url}")
        print(f"=" * 60)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(points)

            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 计算准确率
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

            metrics['loss'].append(loss.item())
            metrics['accuracy'].append(acc.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self, loader, prefix='val') -> dict:
        self.model.eval()
        metrics = defaultdict(list)

        all_preds = []
        all_labels = []

        for batch in loader:
            points = batch['points'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(points)
            loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

            metrics['loss'].append(loss.item())
            metrics['accuracy'].append(acc.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算每类准确率
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        per_class_acc = {}
        for i, name in enumerate(self.class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[name] = (all_preds[mask] == i).mean()
            else:
                per_class_acc[name] = 0.0

        result = {k: np.mean(v) for k, v in metrics.items()}
        result['per_class_acc'] = per_class_acc

        return result

    def train(self):
        for epoch in range(self.args.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'val')

            self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]['lr']

            per_class_str = " | ".join([f"{k[:4]}:{v:.2f}" for k, v in val_metrics['per_class_acc'].items()])

            print(f"Epoch {epoch+1}/{self.args.epochs} ({time.time()-t0:.1f}s) | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f}")
            print(f"  Per-class: {per_class_str}")

            # TensorBoard
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('lr', lr, epoch)

            for name, acc in val_metrics['per_class_acc'].items():
                self.writer.add_scalar(f'val/acc_{name}', acc, epoch)

            # WandB
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'lr': lr,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                }
                for name, acc in val_metrics['per_class_acc'].items():
                    log_dict[f'val/acc_{name}'] = acc
                wandb.log(log_dict)

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                }, os.path.join(self.output_dir, 'best.pth'))
                print(f"  -> New best model saved! Acc: {self.best_val_acc:.3f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                }, os.path.join(self.output_dir, f'epoch_{epoch+1}.pth'))

        # Save final model
        torch.save({
            'epoch': self.args.epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
        }, os.path.join(self.output_dir, 'latest.pth'))

        # Final test evaluation
        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        test_metrics = self.validate(self.test_loader, 'test')
        print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.3f}")
        for name, acc in test_metrics['per_class_acc'].items():
            print(f"  {name}: {acc:.3f}")

        if self.use_wandb:
            wandb.log({
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
                **{f'test/acc_{name}': acc for name, acc in test_metrics['per_class_acc'].items()}
            })
            wandb.finish()

        print(f"\nTraining complete! Best val acc: {self.best_val_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Symmetry Classifier Training')

    # 数据
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augment_factor', type=int, default=10)

    # 模型
    parser.add_argument('--encoder_dim', type=int, default=1024)

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')

    # 实验
    parser.add_argument('--exp_name', type=str, default='SymmetryClassifier')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

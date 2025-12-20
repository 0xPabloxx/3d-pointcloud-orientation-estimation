"""
Single Front Direction Training - 单正面方向预测训练脚本

只使用 "1个正面" 类型的数据训练4峰von Mises模型
排除 OBLIQUE 和 MULTI 方向的数据

用法:
    python train_single_front.py --exp_name SF_1a_pureKL --lambda_kl 1.0 --lambda_kappa 0 --lambda_mu 0
    python train_single_front.py --exp_name SF_1b_angleLoss --lambda_kl 0 --lambda_kappa 0 --lambda_mu 2.0
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

from datasets.single_front_dataset import SingleFrontDataset, collate_fn

# 从 train_fixed_4peak.py 导入模型和 Loss
# 为了代码独立性，这里重新定义关键组件
from scipy.optimize import linear_sum_assignment


# ============== Loss Functions ==============

class SingleFrontLoss(nn.Module):
    """
    单正面方向预测的Loss函数

    支持多种Loss组合:
    - KL Divergence (分布匹配)
    - κ 监督 (防止坍塌)
    - μ 监督 (角度监督)
    """

    def __init__(
        self,
        n_bins: int = 360,
        eps: float = 1e-10,
        lambda_kl: float = 1.0,
        lambda_kappa: float = 5.0,
        lambda_mu: float = 2.0,
        reverse_kl: bool = False
    ):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps
        self.weight = 0.25  # 固定权重
        self.lambda_kl = lambda_kl
        self.lambda_kappa = lambda_kappa
        self.lambda_mu = lambda_mu
        self.reverse_kl = reverse_kl

        # 预计算网格点 [0, 2π)
        grid = torch.linspace(0, 2 * np.pi, n_bins + 1)[:-1]
        self.register_buffer('grid', grid)
        self.bin_width = 2 * np.pi / n_bins

    def _mu_to_angle(self, mu: torch.Tensor) -> torch.Tensor:
        """将 (cos, sin) 转换为角度 [0, 2π)"""
        angles = torch.atan2(mu[..., 1], mu[..., 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        return angles

    def _von_mises_pdf(self, angles: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """计算von Mises PDF"""
        mu_angle = self._mu_to_angle(mu)

        if angles.dim() == 1:
            angles = angles.unsqueeze(0).unsqueeze(0)

        angle_diff = angles - mu_angle.unsqueeze(-1)
        log_pdf = kappa.unsqueeze(-1) * torch.cos(angle_diff)

        # 数值稳定的归一化
        log_pdf_max = log_pdf.max(dim=-1, keepdim=True)[0]
        pdf = torch.exp(log_pdf - log_pdf_max)
        pdf = pdf / (pdf.sum(dim=-1, keepdim=True) + self.eps)

        return pdf

    def _compute_mixture_pdf(self, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """计算混合von Mises的PDF"""
        B = mu.shape[0]
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, 4, -1)

        component_pdfs = self._von_mises_pdf(grid, mu, kappa)
        mixture_pdf = self.weight * component_pdfs.sum(dim=1)
        mixture_pdf = mixture_pdf / (mixture_pdf.sum(dim=-1, keepdim=True) + self.eps)

        return mixture_pdf

    def _hungarian_match(self, pred_mus, pred_kappas, gt_mus, gt_kappas):
        """匈牙利匹配"""
        device = pred_mus.device
        B = pred_mus.shape[0]

        pred_angles = self._mu_to_angle(pred_mus)
        gt_angles = self._mu_to_angle(gt_mus)

        pred_indices = []
        gt_indices = []

        for b in range(B):
            pred_ang = pred_angles[b]
            gt_ang = gt_angles[b]

            angle_diff = pred_ang.unsqueeze(1) - gt_ang.unsqueeze(0)
            angle_cost = 1 - torch.cos(angle_diff)
            kappa_diff = torch.abs(pred_kappas[b].unsqueeze(1) - gt_kappas[b].unsqueeze(0)) / 10.0
            cost_matrix = angle_cost + 0.1 * kappa_diff

            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            pred_indices.append(torch.tensor(row_ind, device=device))
            gt_indices.append(torch.tensor(col_ind, device=device))

        pred_indices = torch.stack(pred_indices)
        gt_indices = torch.stack(gt_indices)

        matched_pred_kappas = torch.gather(pred_kappas, 1, pred_indices)
        matched_gt_kappas = torch.gather(gt_kappas, 1, gt_indices)
        matched_pred_angles = torch.gather(pred_angles, 1, pred_indices)
        matched_gt_angles = torch.gather(gt_angles, 1, gt_indices)

        return matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles

    def forward(self, pred_mus, pred_kappas, gt_mus, gt_kappas):
        """计算Loss"""
        # KL Loss
        if self.lambda_kl > 0:
            pred_pdf = self._compute_mixture_pdf(pred_mus, pred_kappas)
            with torch.no_grad():
                gt_pdf = self._compute_mixture_pdf(gt_mus, gt_kappas)

            pred_pdf = pred_pdf + self.eps
            gt_pdf = gt_pdf + self.eps

            if self.reverse_kl:
                kl_per_bin = pred_pdf * (torch.log(pred_pdf) - torch.log(gt_pdf))
            else:
                kl_per_bin = gt_pdf * (torch.log(gt_pdf) - torch.log(pred_pdf))

            kl_div = kl_per_bin.sum(dim=1) * self.bin_width
            kl_loss = kl_div.mean()
        else:
            kl_loss = torch.tensor(0.0, device=pred_mus.device)

        # 匈牙利匹配
        matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles = \
            self._hungarian_match(pred_mus, pred_kappas, gt_mus, gt_kappas)

        # κ Loss
        kappa_loss = F.smooth_l1_loss(matched_pred_kappas / 10.0, matched_gt_kappas / 10.0)

        # μ Loss (只对有效峰)
        angle_diff = matched_pred_angles - matched_gt_angles
        angle_loss_per_peak = 1 - torch.cos(angle_diff)
        valid_mask = (matched_gt_kappas > 0).float()

        if valid_mask.sum() > 0:
            mu_loss = (angle_loss_per_peak * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            mu_loss = torch.tensor(0.0, device=pred_mus.device)

        # Total Loss
        total_loss = self.lambda_kl * kl_loss + self.lambda_kappa * kappa_loss + self.lambda_mu * mu_loss

        return {
            'loss': total_loss,
            'kl_div': kl_loss,
            'kappa_loss': kappa_loss,
            'mu_loss': mu_loss
        }


# ============== Model (复用 PointNet++) ==============

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


class Fixed4PeakHead(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.3)

        self.mu_head = nn.Linear(hidden_channels, 8)
        self.kappa_head = nn.Linear(hidden_channels, 4)

        self._init_mu_bias()

    def _init_mu_bias(self):
        with torch.no_grad():
            angles_deg = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float32)
            angles_rad = torch.deg2rad(angles_deg)
            cos_vals = torch.cos(angles_rad)
            sin_vals = torch.sin(angles_rad)

            bias = torch.zeros(8)
            for i in range(4):
                bias[i * 2] = cos_vals[i]
                bias[i * 2 + 1] = sin_vals[i]

            self.mu_head.bias.data = bias
            nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)

        self._init_kappa_bias()

    def _init_kappa_bias(self):
        with torch.no_grad():
            self.kappa_head.bias.data.fill_(5.0)
            nn.init.xavier_uniform_(self.kappa_head.weight, gain=0.1)

    def forward(self, x, min_kappa: float = 0.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))

        mu = self.mu_head(x).view(-1, 4, 2)
        mu = F.normalize(mu, dim=-1)

        kappa = F.softplus(self.kappa_head(x)) + min_kappa

        return mu, kappa


class SingleFrontModel(nn.Module):
    """单正面方向预测模型 (输出4峰，但只有1个有效)"""

    def __init__(self, encoder_dim: int = 1024, min_kappa: float = 0.0):
        super().__init__()
        self.min_kappa = min_kappa
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=encoder_dim)
        self.head = Fixed4PeakHead(in_channels=encoder_dim)

    def forward(self, points):
        features = self.encoder(points)
        mu, kappa = self.head(features, min_kappa=self.min_kappa)
        return mu, kappa


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
        config['script'] = 'train_single_front.py'
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 数据集
        self.train_dataset = SingleFrontDataset(
            split='train',
            num_points=args.num_points,
            augment=True,
            augment_factor=args.augment_factor,
        )
        self.val_dataset = SingleFrontDataset(
            split='val',
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

        # 模型
        self.model = SingleFrontModel(
            encoder_dim=args.encoder_dim,
            min_kappa=args.min_kappa
        ).to(self.device)

        # Loss
        self.criterion = SingleFrontLoss(
            n_bins=360,
            lambda_kl=args.lambda_kl,
            lambda_kappa=args.lambda_kappa,
            lambda_mu=args.lambda_mu,
            reverse_kl=args.reverse_kl
        ).to(self.device)

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

        self.best_val_loss = float('inf')

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
        print(f"Single Front Direction Training")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_dataset)} (online augment each epoch)")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Loss: λ_KL={args.lambda_kl}, λ_κ={args.lambda_kappa}, λ_μ={args.lambda_mu}")
        print(f"Output: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.url}")
        print(f"=" * 60)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            gt_mu = batch['gt_mu'].to(self.device)
            gt_kappa = batch['gt_kappa'].to(self.device)

            self.optimizer.zero_grad()
            pred_mu, pred_kappa = self.model(points)

            loss_dict = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa)
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                metrics[k].append(v.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        metrics = defaultdict(list)

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_mu = batch['gt_mu'].to(self.device)
            gt_kappa = batch['gt_kappa'].to(self.device)

            pred_mu, pred_kappa = self.model(points)
            loss_dict = self.criterion(pred_mu, pred_kappa, gt_mu, gt_kappa)

            for k, v in loss_dict.items():
                metrics[k].append(v.item())

            # 计算角度误差 (所有4峰，因为GT是4峰同向)
            gt_angle = batch['gt_angle'].to(self.device)  # (B,)

            # 计算每个峰的角度
            pred_angles = torch.atan2(pred_mu[:, :, 1], pred_mu[:, :, 0])  # (B, 4)

            # 每个峰与GT的角度误差
            angle_errors = torch.abs(pred_angles - gt_angle.unsqueeze(1))  # (B, 4)
            angle_errors = torch.min(angle_errors, 2 * np.pi - angle_errors)  # 处理周期性

            # 平均角度误差 (所有4峰)
            mean_error = angle_errors.mean(dim=1)  # (B,)
            metrics['angle_error_deg'].extend((mean_error * 180 / np.pi).cpu().numpy().tolist())

            # 最小角度误差 (最好的那个峰)
            min_error = angle_errors.min(dim=1)[0]  # (B,)
            metrics['angle_error_min_deg'].extend((min_error * 180 / np.pi).cpu().numpy().tolist())

            # 最大角度误差 (最差的那个峰)
            max_error = angle_errors.max(dim=1)[0]  # (B,)
            metrics['angle_error_max_deg'].extend((max_error * 180 / np.pi).cpu().numpy().tolist())

        return {k: np.mean(v) for k, v in metrics.items()}

    def train(self):
        for epoch in range(self.args.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{self.args.epochs} ({time.time()-t0:.1f}s) | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Angle Err: {val_metrics.get('angle_error_deg', 0):.1f}° "
                  f"(min:{val_metrics.get('angle_error_min_deg', 0):.1f}° max:{val_metrics.get('angle_error_max_deg', 0):.1f}°)")

            # TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
            self.writer.add_scalar('lr', lr, epoch)

            # WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': lr,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                })

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, os.path.join(self.output_dir, 'best.pth'))
                print(f"  -> New best model saved!")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, os.path.join(self.output_dir, f'epoch_{epoch+1}.pth'))

        # Save final model
        torch.save({
            'epoch': self.args.epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
        }, os.path.join(self.output_dir, 'latest.pth'))

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Single Front Direction Training')

    # 数据
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augment_factor', type=int, default=10)

    # 模型
    parser.add_argument('--encoder_dim', type=int, default=1024)
    parser.add_argument('--min_kappa', type=float, default=0.0)

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # Loss权重
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_kappa', type=float, default=5.0)
    parser.add_argument('--lambda_mu', type=float, default=2.0)
    parser.add_argument('--reverse_kl', action='store_true')

    # 实验
    parser.add_argument('--exp_name', type=str, default='SingleFront')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LossAblation')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

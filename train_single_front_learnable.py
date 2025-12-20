"""
Single Front Direction Training with Learnable Weights

可学习权重 vs 固定权重对比实验

方案1 (Baseline): 固定权重 = 0.25
方案2 (Learnable): 权重通过网络学习

用法:
    # 固定权重 (baseline)
    python train_single_front_learnable.py --exp_name LW_1a_baseline

    # 可学习权重 (无监督)
    python train_single_front_learnable.py --exp_name LW_1b_learnable_free --learnable_weights

    # 可学习权重 + 熵正则
    python train_single_front_learnable.py --exp_name LW_1c_learnable_entropy --learnable_weights --weight_loss_type entropy --lambda_weight 0.1

    # 可学习权重 + GT监督
    python train_single_front_learnable.py --exp_name LW_1d_learnable_gt --learnable_weights --weight_loss_type gt --lambda_weight 1.0
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.multi_category_dataset import MultiCategoryDataset, collate_fn
from scipy.optimize import linear_sum_assignment


# ============== Loss Functions ==============

class LearnableWeightLoss(nn.Module):
    """
    支持可学习权重的Loss函数

    Loss = λ_KL * KL_div + λ_κ * kappa_loss + λ_μ * mu_loss + λ_w * weight_loss
    """

    def __init__(
        self,
        n_bins: int = 360,
        eps: float = 1e-10,
        lambda_kl: float = 1.0,
        lambda_kappa: float = 5.0,
        lambda_mu: float = 2.0,
        lambda_weight: float = 0.0,
        reverse_kl: bool = False,
        weight_loss_type: str = 'none',  # 'none', 'entropy', 'gt'
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps
        self.lambda_kl = lambda_kl
        self.lambda_kappa = lambda_kappa
        self.lambda_mu = lambda_mu
        self.lambda_weight = lambda_weight
        self.reverse_kl = reverse_kl
        self.weight_loss_type = weight_loss_type
        self.learnable_weights = learnable_weights

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

    def _compute_mixture_pdf(self, mu: torch.Tensor, kappa: torch.Tensor,
                            weights: torch.Tensor = None) -> torch.Tensor:
        """计算混合von Mises的PDF，支持可变权重"""
        B = mu.shape[0]
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, 4, -1)

        component_pdfs = self._von_mises_pdf(grid, mu, kappa)  # (B, 4, n_bins)

        if weights is None:
            # 固定权重
            mixture_pdf = 0.25 * component_pdfs.sum(dim=1)
        else:
            # 可变权重
            weighted_pdf = weights.unsqueeze(-1) * component_pdfs  # (B, 4, n_bins)
            mixture_pdf = weighted_pdf.sum(dim=1)  # (B, n_bins)

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

        return matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles, pred_indices, gt_indices

    def forward(self, pred_mus, pred_kappas, gt_mus, gt_kappas,
                pred_weights=None, gt_weights=None):
        """计算Loss"""
        # KL Loss
        if self.lambda_kl > 0:
            pred_pdf = self._compute_mixture_pdf(pred_mus, pred_kappas, pred_weights)
            with torch.no_grad():
                gt_pdf = self._compute_mixture_pdf(gt_mus, gt_kappas, gt_weights)

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
        matched_pred_kappas, matched_gt_kappas, matched_pred_angles, matched_gt_angles, pred_idx, gt_idx = \
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

        # Weight Loss
        weight_loss = torch.tensor(0.0, device=pred_mus.device)
        weight_entropy = torch.tensor(0.0, device=pred_mus.device)

        if self.learnable_weights and pred_weights is not None:
            # 计算权重熵 (用于监控)
            weight_entropy = -(pred_weights * torch.log(pred_weights + self.eps)).sum(dim=-1).mean()

            if self.lambda_weight > 0:
                if self.weight_loss_type == 'entropy':
                    # 最小化熵 → 鼓励稀疏
                    weight_loss = weight_entropy
                elif self.weight_loss_type == 'gt':
                    # GT 监督
                    if gt_weights is not None:
                        # 需要根据匈牙利匹配重排gt_weights
                        matched_gt_weights = torch.gather(gt_weights, 1, gt_idx)
                        matched_pred_weights = torch.gather(pred_weights, 1, pred_idx)
                        weight_loss = F.mse_loss(matched_pred_weights, matched_gt_weights)
                    else:
                        # 默认GT权重为均匀
                        uniform_weights = torch.ones_like(pred_weights) / 4
                        weight_loss = F.mse_loss(pred_weights, uniform_weights)

        # Total Loss
        total_loss = (self.lambda_kl * kl_loss +
                     self.lambda_kappa * kappa_loss +
                     self.lambda_mu * mu_loss +
                     self.lambda_weight * weight_loss)

        return {
            'loss': total_loss,
            'kl_div': kl_loss,
            'kappa_loss': kappa_loss,
            'mu_loss': mu_loss,
            'weight_loss': weight_loss,
            'weight_entropy': weight_entropy,
        }


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


class LearnableWeightHead(nn.Module):
    """支持可学习权重的Head"""

    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512,
                 learnable_weights: bool = False):
        super().__init__()
        self.learnable_weights = learnable_weights

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.3)

        self.mu_head = nn.Linear(hidden_channels, 8)
        self.kappa_head = nn.Linear(hidden_channels, 4)

        if learnable_weights:
            self.weight_head = nn.Linear(hidden_channels, 4)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # Mu初始化: 四个方向
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

            # Kappa初始化
            self.kappa_head.bias.data.fill_(5.0)
            nn.init.xavier_uniform_(self.kappa_head.weight, gain=0.1)

            # Weight初始化 (如果可学习)
            if self.learnable_weights:
                self.weight_head.bias.data.fill_(0.0)  # softmax后为均匀分布
                nn.init.xavier_uniform_(self.weight_head.weight, gain=0.1)

    def forward(self, x, min_kappa: float = 0.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))

        mu = self.mu_head(x).view(-1, 4, 2)
        mu = F.normalize(mu, dim=-1)

        kappa = F.softplus(self.kappa_head(x)) + min_kappa

        if self.learnable_weights:
            weights = F.softmax(self.weight_head(x), dim=-1)
        else:
            weights = None

        return mu, kappa, weights


class LearnableWeightModel(nn.Module):
    """支持可学习权重的模型"""

    def __init__(self, encoder_dim: int = 1024, min_kappa: float = 0.0,
                 learnable_weights: bool = False):
        super().__init__()
        self.min_kappa = min_kappa
        self.learnable_weights = learnable_weights
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=encoder_dim)
        self.head = LearnableWeightHead(in_channels=encoder_dim, learnable_weights=learnable_weights)

    def forward(self, points):
        features = self.encoder(points)
        mu, kappa, weights = self.head(features, min_kappa=self.min_kappa)
        return mu, kappa, weights


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
        config['script'] = 'train_single_front_learnable.py'
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 数据集 (支持多类别)
        categories = args.categories.split(',') if args.categories else ['1_front', '4_fronts', 'no_front']
        self.categories = categories

        self.train_dataset = MultiCategoryDataset(
            split='train',
            categories=categories,
            num_points=args.num_points,
            augment=True,
            augment_factor=args.augment_factor,
        )
        self.val_dataset = MultiCategoryDataset(
            split='val',
            categories=categories,
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
        self.model = LearnableWeightModel(
            encoder_dim=args.encoder_dim,
            min_kappa=args.min_kappa,
            learnable_weights=args.learnable_weights
        ).to(self.device)

        # Loss
        self.criterion = LearnableWeightLoss(
            n_bins=360,
            lambda_kl=args.lambda_kl,
            lambda_kappa=args.lambda_kappa,
            lambda_mu=args.lambda_mu,
            lambda_weight=args.lambda_weight,
            reverse_kl=args.reverse_kl,
            weight_loss_type=args.weight_loss_type,
            learnable_weights=args.learnable_weights,
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
        print(f"Learnable Weight Training")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Learnable weights: {args.learnable_weights}")
        print(f"Weight loss type: {args.weight_loss_type}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Loss: λ_KL={args.lambda_kl}, λ_κ={args.lambda_kappa}, λ_μ={args.lambda_mu}, λ_w={args.lambda_weight}")
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
            gt_weights = batch['gt_weights'].to(self.device)  # 从数据集获取

            self.optimizer.zero_grad()
            pred_mu, pred_kappa, pred_weights = self.model(points)

            loss_dict = self.criterion(
                pred_mu, pred_kappa, gt_mu, gt_kappa,
                pred_weights=pred_weights, gt_weights=gt_weights
            )
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                metrics[k].append(v.item())

            # 记录权重统计
            if pred_weights is not None:
                metrics['weight_max'].append(pred_weights.max(dim=-1)[0].mean().item())
                metrics['weight_min'].append(pred_weights.min(dim=-1)[0].mean().item())

        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        metrics = defaultdict(list)
        category_metrics = defaultdict(lambda: defaultdict(list))

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            gt_mu = batch['gt_mu'].to(self.device)
            gt_kappa = batch['gt_kappa'].to(self.device)
            gt_weights = batch['gt_weights'].to(self.device)
            categories = batch['category']

            pred_mu, pred_kappa, pred_weights = self.model(points)
            loss_dict = self.criterion(
                pred_mu, pred_kappa, gt_mu, gt_kappa,
                pred_weights=pred_weights, gt_weights=gt_weights
            )

            for k, v in loss_dict.items():
                metrics[k].append(v.item())

            # 计算角度误差
            gt_angle = batch['gt_angle'].to(self.device)
            pred_angles = torch.atan2(pred_mu[:, :, 1], pred_mu[:, :, 0])

            angle_errors = torch.abs(pred_angles - gt_angle.unsqueeze(1))
            angle_errors = torch.min(angle_errors, 2 * np.pi - angle_errors)

            mean_error = angle_errors.mean(dim=1)
            min_error = angle_errors.min(dim=1)[0]

            # 整体角度误差
            metrics['angle_error_deg'].extend((mean_error * 180 / np.pi).cpu().numpy().tolist())
            metrics['angle_error_min_deg'].extend((min_error * 180 / np.pi).cpu().numpy().tolist())

            # 按类别统计角度误差
            for i, cat in enumerate(categories):
                if gt_kappa[i].sum() > 0:  # 只对有方向的类别计算
                    err_deg = (min_error[i] * 180 / np.pi).cpu().item()
                    category_metrics[cat]['angle_error_deg'].append(err_deg)

            # 权重统计
            if pred_weights is not None:
                metrics['weight_max'].append(pred_weights.max(dim=-1)[0].mean().item())
                metrics['weight_min'].append(pred_weights.min(dim=-1)[0].mean().item())

                # 按类别统计权重
                for i, cat in enumerate(categories):
                    w_max = pred_weights[i].max().cpu().item()
                    w_entropy = -(pred_weights[i] * torch.log(pred_weights[i] + 1e-10)).sum().cpu().item()
                    category_metrics[cat]['weight_max'].append(w_max)
                    category_metrics[cat]['weight_entropy'].append(w_entropy)

        # 汇总结果
        result = {k: np.mean(v) for k, v in metrics.items()}

        # 添加按类别的结果
        for cat, cat_metrics in category_metrics.items():
            for k, v in cat_metrics.items():
                if len(v) > 0:
                    result[f'{cat}/{k}'] = np.mean(v)

        return result

    def train(self):
        for epoch in range(self.args.epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            weight_info = ""
            if self.args.learnable_weights:
                weight_info = f" | W_max: {val_metrics.get('weight_max', 0):.3f}"

            print(f"Epoch {epoch+1}/{self.args.epochs} ({time.time()-t0:.1f}s) | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Angle Err: {val_metrics.get('angle_error_deg', 0):.1f}°"
                  f"{weight_info}")

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

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, os.path.join(self.output_dir, f'epoch_{epoch+1}.pth'))

        # Final model
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
    parser = argparse.ArgumentParser(description='Learnable Weight Training')

    # 数据
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augment_factor', type=int, default=10)
    parser.add_argument('--categories', type=str, default='1_front,4_fronts,no_front',
                        help='Comma-separated list of categories: 1_front,4_fronts,no_front')

    # 模型
    parser.add_argument('--encoder_dim', type=int, default=1024)
    parser.add_argument('--min_kappa', type=float, default=0.0)
    parser.add_argument('--learnable_weights', action='store_true',
                        help='Enable learnable weights')

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # Loss权重
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_kappa', type=float, default=5.0)
    parser.add_argument('--lambda_mu', type=float, default=2.0)
    parser.add_argument('--lambda_weight', type=float, default=0.0,
                        help='Weight for weight loss')
    parser.add_argument('--reverse_kl', action='store_true')
    parser.add_argument('--weight_loss_type', type=str, default='none',
                        choices=['none', 'entropy', 'gt'],
                        help='Type of weight regularization')

    # 实验
    parser.add_argument('--exp_name', type=str, default='LearnableWeight')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='ForwardNet-LearnableWeight')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

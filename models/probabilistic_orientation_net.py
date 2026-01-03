#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probabilistic Orientation Network with Topology-Aware Mixture of Experts

This module implements a probabilistic approach to orientation prediction
that leverages a pre-trained symmetry classifier as a gating mechanism.

Architecture:
    1. Classifier (Gate): Predicts symmetry class weights [B, 5]
       - Can be frozen (default) or fine-tuned
    2. Shared Backbone: PointNet++ extracts global features [B, 1024]
    3. Expert Heads: 3 specialized heads for 1-front, 2-front, 4-front cases
       Each head outputs (mu, kappa) pairs for von Mises distributions

Label Mapping (MoE convention):
    0: 1-front (1个正面) - uses head_1front
    1: 2-front (2个正面) - uses head_2front
    2: 4-front (4个正面) - uses head_4front
    3: Rotational symmetric (旋转对称) - no direction
    4: No-front (无正面) - no direction

Author: Claude
Created: 2025-12-28
Updated: 2025-12-28 - Fixed issues from code review
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import index_points, query_ball_point


# ==============================================================================
# Utility Functions
# ==============================================================================

def _fps_core(xyz: torch.Tensor, npoint: int, start_idx: torch.Tensor) -> torch.Tensor:
    """
    Core FPS loop - separated for torch.compile optimization.

    Args:
        xyz: (B, N, 3) point cloud
        npoint: number of points to sample
        start_idx: (B,) starting point indices

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = start_idx
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


# Compile the core FPS loop for ~30x speedup
# Note: mode='reduce-overhead' is best for small kernels with loops
try:
    _fps_core_compiled = torch.compile(_fps_core, mode='reduce-overhead')
    _USE_COMPILED_FPS = True
except Exception:
    _fps_core_compiled = _fps_core
    _USE_COMPILED_FPS = False


def farthest_point_sample(xyz: torch.Tensor, npoint: int,
                          random_start: bool = True) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) with optional torch.compile acceleration.

    Uses torch.compile for ~30x speedup on CUDA.

    Args:
        xyz: (B, N, 3) point cloud
        npoint: number of points to sample
        random_start: if True, start from random point (training);
                      if False, start from point 0 (inference, deterministic)

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Starting point: random for training, fixed for eval (deterministic)
    if random_start:
        start_idx = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

    # Use compiled version if available
    if _USE_COMPILED_FPS and xyz.is_cuda:
        return _fps_core_compiled(xyz, npoint, start_idx)
    else:
        return _fps_core(xyz, npoint, start_idx)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer with deterministic FPS.

    Uses Farthest Point Sampling instead of random sampling for reproducible results.
    """
    def __init__(self, npoint, nsample, in_channel, mlp_channels, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all

        last_ch = in_channel + 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in mlp_channels:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)
            new_points = grouped_xyz if points is None else torch.cat([grouped_xyz, points.unsqueeze(1)], -1)
        else:
            # FPS: random start during training for diversity, fixed start during eval for determinism
            fps_idx = farthest_point_sample(xyz, self.npoint, random_start=self.training)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(new_xyz, xyz, self.nsample)
            grouped_xyz = index_points(xyz, idx)
            normed = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                new_points = torch.cat([normed, grouped_pts], -1)
            else:
                new_points = normed

        x = new_points.permute(0, 3, 1, 2)  # (B, 3+D, npoint, nsample)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = torch.max(x, 3)[0]  # (B, mlp[-1], npoint)
        return new_xyz, x.permute(0, 2, 1)


class ExpertHead(nn.Module):
    """
    Expert head for predicting von Mises distribution parameters.

    Architecture (P1 fix for kappa collapse):
        Uses SEPARATE output layers for μ and κ to prevent training μ from
        affecting κ output. This matches the working design in core/heads.py.

        Shared hidden layers:  features -> hidden -> hidden
        Separate output layers: hidden -> fc_mu (cos, sin)
                                hidden -> fc_kappa (raw_kappa)

    Outputs K pairs of (mu, kappa) where:
        - mu is represented as (cos, sin) and converted to angle via atan2
        - kappa is computed via softplus for positivity
    """

    def __init__(self, in_dim: int, num_peaks: int, hidden_dim: int = 256,
                 kappa_min: float = 1e-4, kappa_max: float = 100.0,
                 kappa_init: float = 5.0):
        """
        Args:
            in_dim: Input feature dimension (1024 from backbone)
            num_peaks: Number of von Mises peaks (1, 2, or 4)
            hidden_dim: Hidden layer dimension
            kappa_min: Minimum kappa value for numerical stability
            kappa_max: Maximum kappa value to prevent numerical overflow
            kappa_init: Initial kappa value (via softplus bias initialization)
        """
        super().__init__()
        self.num_peaks = num_peaks
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        # Shared hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # SEPARATE output layers (P1 fix: prevents μ training from affecting κ)
        self.fc_mu = nn.Linear(hidden_dim, num_peaks * 2)  # cos, sin for each peak
        self.fc_kappa = nn.Linear(hidden_dim, num_peaks)   # raw_kappa for each peak

        # Initialize kappa bias so softplus(bias) ≈ kappa_init
        # softplus(x) = log(1 + exp(x)), inverse: x = log(exp(k) - 1)
        if kappa_init > 0:
            init_bias = math.log(math.exp(kappa_init) - 1) if kappa_init > 0.5 else kappa_init
            self.fc_kappa.bias.data.fill_(init_bias)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, in_dim] global features from backbone

        Returns:
            dict with keys:
                - 'mu': [B, num_peaks] angles in radians
                - 'kappa': [B, num_peaks] concentration parameters (clamped)
        """
        B = features.size(0)

        # Shared hidden representation
        hidden = self.hidden(features)  # [B, hidden_dim]

        # Separate output paths
        mu_raw = self.fc_mu(hidden)  # [B, num_peaks * 2]
        mu_raw = mu_raw.view(B, self.num_peaks, 2)  # [B, num_peaks, 2]

        raw_kappa = self.fc_kappa(hidden)  # [B, num_peaks]

        # Convert (cos, sin) to angle via atan2
        cos_val = mu_raw[:, :, 0]  # [B, num_peaks]
        sin_val = mu_raw[:, :, 1]  # [B, num_peaks]
        mu = torch.atan2(sin_val, cos_val)  # [B, num_peaks]

        # Convert raw_kappa to positive via softplus
        kappa = F.softplus(raw_kappa)  # [B, num_peaks]

        # Apply clamp for numerical stability
        kappa = kappa.clamp(min=self.kappa_min, max=self.kappa_max)  # [B, num_peaks]

        return {'mu': mu, 'kappa': kappa}


class ProbabilisticOrientationNet(nn.Module):
    """
    Topology-Aware Mixture of Experts for Probabilistic Orientation Prediction.

    This network combines:
        1. A pre-trained classifier as a gating mechanism (frozen by default)
        2. A shared PointNet++ backbone for feature extraction
        3. Three expert heads for 1-front, 2-front, and 4-front cases

    The classifier provides soft routing weights during inference,
    while training uses hard GT routing via MaskedExpertLoss.
    """

    def __init__(self,
                 classifier: nn.Module,
                 backbone_dim: int = 1024,
                 expert_hidden_dim: int = 256,
                 kappa_min: float = 1e-4,
                 kappa_max: float = 100.0,
                 freeze_classifier: bool = True):
        """
        Args:
            classifier: Pre-trained symmetry classifier (e.g., SymmetryClassifier_GlobalConcat)
            backbone_dim: Output dimension of the shared backbone (default 1024)
            expert_hidden_dim: Hidden dimension for expert heads
            kappa_min: Minimum kappa value for numerical stability
            kappa_max: Maximum kappa value to prevent numerical overflow
            freeze_classifier: Whether to freeze classifier weights (default True)
        """
        super().__init__()

        # Store classifier and freeze setting
        self.classifier = classifier
        self.freeze_classifier = freeze_classifier

        if freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.eval()

        # Shared PointNet++ Backbone (uses deterministic FPS)
        self.sa1 = PointNetSetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, 256, [256, 512, backbone_dim], group_all=True)

        # Expert Heads
        # Head 1: 1-front -> 1 peak
        self.head_1front = ExpertHead(backbone_dim, num_peaks=1, hidden_dim=expert_hidden_dim,
                                       kappa_min=kappa_min, kappa_max=kappa_max)

        # Head 2: 2-front -> 2 peaks (fixed uniform weights [0.5, 0.5])
        self.head_2front = ExpertHead(backbone_dim, num_peaks=2, hidden_dim=expert_hidden_dim,
                                       kappa_min=kappa_min, kappa_max=kappa_max)

        # Head 4: 4-front -> 4 peaks (fixed uniform weights [0.25, 0.25, 0.25, 0.25])
        self.head_4front = ExpertHead(backbone_dim, num_peaks=4, hidden_dim=expert_hidden_dim,
                                       kappa_min=kappa_min, kappa_max=kappa_max)

        # Fixed mixture weights (not learnable)
        self.register_buffer('weights_2front', torch.tensor([0.5, 0.5]))
        self.register_buffer('weights_4front', torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def train(self, mode: bool = True):
        """Override train to keep classifier in eval mode if frozen."""
        super().train(mode)
        if self.freeze_classifier:
            self.classifier.eval()
        return self

    def _ensure_correct_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input is in [B, N, 3] format for PointNet++ processing.

        Args:
            x: Input point cloud, either [B, N, 3] or [B, 3, N]

        Returns:
            Point cloud in [B, N, 3] format
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")

        # Check if input is [B, 3, N] format (3 channels dimension)
        if x.size(1) == 3 and x.size(2) != 3:
            # Transpose from [B, 3, N] to [B, N, 3]
            x = x.transpose(1, 2)

        return x

    def forward(self, x: torch.Tensor, upright_vec: torch.Tensor = None) -> dict:
        """
        Forward pass.

        Args:
            x: Point cloud [B, N, 3] or [B, 3, N]
            upright_vec: Upright direction vector [B, 3], default [0, 1, 0]

        Returns:
            dict containing:
                - 'weights': [B, 5] classifier softmax probabilities
                - 'head_1front': dict with 'mu' [B, 1], 'kappa' [B, 1]
                - 'head_2front': dict with 'mu' [B, 2], 'kappa' [B, 2], 'mix_weights' [2]
                - 'head_4front': dict with 'mu' [B, 4], 'kappa' [B, 4], 'mix_weights' [4]
        """
        # Ensure correct format
        x = self._ensure_correct_format(x)
        B = x.size(0)

        # Default upright vector (must match classifier training)
        if upright_vec is None:
            upright_vec = torch.zeros(B, 3, device=x.device)
            upright_vec[:, 1] = 1.0  # [0, 1, 0]

        # 1. Get classifier weights
        # Conditionally use no_grad based on freeze_classifier
        if self.freeze_classifier:
            with torch.no_grad():
                logits = self.classifier(x, upright_vec)
            logits = logits.detach()  # Detach to prevent gradients
            weights = F.softmax(logits, dim=1)  # [B, 5]
        else:
            # Allow gradient flow for fine-tuning
            logits = self.classifier(x, upright_vec)
            weights = F.softmax(logits, dim=1)  # [B, 5]

        # 2. Shared backbone feature extraction
        l1_xyz, l1_pts = self.sa1(x, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.view(B, -1)  # [B, 1024]

        # 3. Expert heads
        out_1front = self.head_1front(global_feat)
        out_2front = self.head_2front(global_feat)
        out_4front = self.head_4front(global_feat)

        # Add fixed mixture weights
        out_2front['mix_weights'] = self.weights_2front
        out_4front['mix_weights'] = self.weights_4front

        return {
            'weights': weights,
            'logits': logits,  # Raw logits for CE loss
            'head_1front': out_1front,
            'head_2front': out_2front,
            'head_4front': out_4front,
            'global_feat': global_feat  # Optional: for downstream use
        }


# ==============================================================================
# Loss Functions
# ==============================================================================

def log_bessel_i0(kappa: torch.Tensor) -> torch.Tensor:
    """
    Compute log(I0(kappa)) in a numerically stable way.

    Uses: log(I0(kappa)) = kappa + log(I0e(kappa))
    where I0e(kappa) = I0(kappa) * exp(-kappa) is exponentially scaled.
    """
    from torch.special import i0e
    return kappa + torch.log(i0e(kappa) + 1e-10)


def von_mises_nll(theta_gt: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for von Mises distribution.

    p(θ|μ,κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))
    NLL = -κ * cos(θ_gt - μ) + log(2π) + log(I₀(κ))

    Args:
        theta_gt: Ground truth angle in radians
        mu: Predicted mean angle in radians
        kappa: Predicted concentration parameter

    Returns:
        NLL loss (same shape as input)
    """
    cos_diff = torch.cos(theta_gt - mu)
    nll = -kappa * cos_diff + math.log(2 * math.pi) + log_bessel_i0(kappa)
    return nll


def batched_hungarian_match_k2(pred_mu: torch.Tensor, gt_angles: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Hungarian matching for K=2 (only 2 permutations).

    Args:
        pred_mu: Predicted angles [B, 2]
        gt_angles: GT angles [B, 2]

    Returns:
        matched_gt: [B, 2] GT angles reordered to match predictions
    """
    # Cost for identity permutation
    cost_identity = (1 - torch.cos(pred_mu - gt_angles)).sum(dim=1)  # [B]
    # Cost for swap permutation
    gt_swapped = torch.stack([gt_angles[:, 1], gt_angles[:, 0]], dim=1)  # [B, 2]
    cost_swap = (1 - torch.cos(pred_mu - gt_swapped)).sum(dim=1)  # [B]

    # Choose better permutation
    use_swap = (cost_swap < cost_identity).unsqueeze(1)  # [B, 1]
    return torch.where(use_swap, gt_swapped, gt_angles)


def batched_hungarian_match_k4(pred_mu: torch.Tensor, gt_angles: torch.Tensor,
                                perms_4: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Hungarian matching for K=4 using pre-cached permutations.

    Args:
        pred_mu: Predicted angles [B, 4]
        gt_angles: GT angles [B, 4]
        perms_4: Cached permutation tensor [24, 4]

    Returns:
        matched_gt: [B, 4] GT angles reordered to match predictions
    """
    # Compute cost for each permutation: [B, 24]
    gt_permuted = gt_angles[:, perms_4]  # [B, 24, 4]
    pred_expanded = pred_mu.unsqueeze(1)  # [B, 1, 4]

    costs = (1 - torch.cos(pred_expanded - gt_permuted)).sum(dim=2)  # [B, 24]

    # Find best permutation for each sample
    best_perm_idx = costs.argmin(dim=1)  # [B]
    best_perms = perms_4[best_perm_idx]  # [B, 4]

    # Gather matched GT
    return torch.gather(gt_angles, 1, best_perms)  # [B, 4]


class MaskedExpertLoss(nn.Module):
    """
    Loss function with GT-based routing and Soft Gate-Aware weighting (P2v2).

    Label Mapping (MoE convention):
        - Label 0 (1-front): Train head_1front with single-peak von Mises NLL
        - Label 1 (2-front): Train head_2front with Hungarian-matched 2-peak NLL
        - Label 2 (4-front): Train head_4front with Hungarian-matched 4-peak NLL
        - Label 3, 4 (Rot, No-front): No direction loss (skip)

    P2v2 Gate-Aware Training (improved stability):
        1. Soft gate weighting (no hard skip):
           - w_gate = clamp((p_dir - threshold) / (1 - threshold), 0, 1)
           - w_cls = p_gt^gamma
           - w = w_gate * w_cls
        2. Weight-normalized loss: L = sum(w*L) / (sum(w) + eps)
        3. Always-on cosine loss for 1-front (scheduled: 0.2 -> 0.1)
        4. Staged κ regularization (only 1-front, epoch 6+)
    """

    def __init__(self, classification_weight: float = 0.0,
                 # Soft gate weighting
                 use_gate_weighting: bool = True,
                 p_dir_threshold: float = 0.4,
                 gamma: float = 1.5,
                 # Cosine loss (always active for 1-front)
                 cosine_weight_init: float = 0.2,
                 cosine_weight_final: float = 0.1,
                 cosine_schedule_epoch: int = 10,
                 # Kappa regularization (only 1-front, staged)
                 kappa_reg_weight_final: float = 0.02,
                 kappa_target_final: float = 5.0,
                 kappa_reg_start_epoch: int = 6,
                 kappa_reg_ramp_epochs: int = 10):
        """
        Args:
            classification_weight: Weight for classification loss (default 0 = frozen gate).
            use_gate_weighting: Use soft gate weighting. Default True.
            p_dir_threshold: Threshold for p_dir where weight starts ramping up. Default 0.4.
            gamma: Power for p_gt weighting (higher = favor high-confidence). Default 1.5.
            cosine_weight_init: Initial cosine loss weight (epochs 0 to cosine_schedule_epoch).
            cosine_weight_final: Final cosine loss weight (after schedule).
            cosine_schedule_epoch: Epoch to switch from init to final cosine weight.
            kappa_reg_weight_final: Final κ regularization weight. Default 0.02.
            kappa_target_final: Final target κ value. Default 5.0.
            kappa_reg_start_epoch: Epoch to start κ regularization. Default 6.
            kappa_reg_ramp_epochs: Epochs to ramp κ reg from 0 to final. Default 10.
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.use_gate_weighting = use_gate_weighting
        self.p_dir_threshold = p_dir_threshold
        self.gamma = gamma

        # Cosine loss schedule
        self.cosine_weight_init = cosine_weight_init
        self.cosine_weight_final = cosine_weight_final
        self.cosine_schedule_epoch = cosine_schedule_epoch

        # Kappa regularization schedule (only for 1-front)
        self.kappa_reg_weight_final = kappa_reg_weight_final
        self.kappa_target_final = kappa_target_final
        self.kappa_reg_start_epoch = kappa_reg_start_epoch
        self.kappa_reg_ramp_epochs = kappa_reg_ramp_epochs

        self.current_epoch = 0

        if classification_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss()

        # Cache Hungarian permutations as buffers
        import itertools
        perms_4 = list(itertools.permutations(range(4)))
        self.register_buffer('perms_4', torch.tensor(perms_4, dtype=torch.long))

    def set_epoch(self, epoch: int):
        """Set current epoch for scheduled parameters."""
        self.current_epoch = epoch

    def get_cosine_weight(self) -> float:
        """Get current cosine loss weight based on epoch schedule."""
        if self.current_epoch < self.cosine_schedule_epoch:
            return self.cosine_weight_init
        return self.cosine_weight_final

    def get_kappa_reg_params(self) -> tuple:
        """
        Get current κ regularization parameters based on epoch schedule.

        Returns:
            (kappa_reg_weight, kappa_target) tuple
        """
        epoch = self.current_epoch
        start = self.kappa_reg_start_epoch
        ramp = self.kappa_reg_ramp_epochs

        if epoch < start:
            return 0.0, 0.0

        # Linear ramp from start to start+ramp
        progress = min((epoch - start) / ramp, 1.0)
        weight = progress * self.kappa_reg_weight_final
        target = progress * self.kappa_target_final
        return weight, target

    def compute_sample_weights(self, gate_weights: torch.Tensor,
                                gt_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample soft gate weights.

        Args:
            gate_weights: [B, 5] softmax probabilities from classifier
            gt_labels: [B] ground truth labels

        Returns:
            [B] per-sample weights
        """
        if not self.use_gate_weighting:
            return torch.ones(gt_labels.size(0), device=gt_labels.device)

        # p_dir = p_1f + p_2f + p_4f
        p_dir = gate_weights[:, 0] + gate_weights[:, 1] + gate_weights[:, 2]

        # Soft gate weight: linear ramp from threshold to 1.0
        # w_gate = clamp((p_dir - threshold) / (1 - threshold), 0, 1)
        denom = 1.0 - self.p_dir_threshold
        w_gate = torch.clamp((p_dir - self.p_dir_threshold) / (denom + 1e-8), 0, 1)

        # Class confidence weight: p_gt^gamma
        p_gt = gate_weights.gather(1, gt_labels.unsqueeze(1)).squeeze(1)
        w_cls = torch.pow(p_gt, self.gamma)

        # Combined weight
        return w_gate * w_cls

    def forward(self,
                model_output: dict,
                gt_angles: torch.Tensor,
                gt_labels: torch.Tensor) -> dict:
        """
        Compute masked expert loss with Soft Gate-Aware weighting (P2v2).

        Args:
            model_output: Output from ProbabilisticOrientationNet (includes 'weights')
            gt_angles: Ground truth angles in radians [B]
            gt_labels: Ground truth labels (0=1front, 1=2front, 2=4front, 3=rot, 4=nofront) [B]

        Returns:
            dict with loss components and metrics
        """
        device = gt_angles.device
        B = gt_angles.size(0)

        # Get gate probabilities and compute soft sample weights
        gate_weights = model_output['weights']  # [B, 5] softmax probs
        sample_weights = self.compute_sample_weights(gate_weights, gt_labels)  # [B]

        # Get scheduled parameters
        cosine_weight = self.get_cosine_weight()
        kappa_reg_weight, kappa_target = self.get_kappa_reg_params()

        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        loss_1front = torch.tensor(0.0, device=device)
        loss_2front = torch.tensor(0.0, device=device)
        loss_4front = torch.tensor(0.0, device=device)
        loss_kappa_reg = torch.tensor(0.0, device=device)

        # Create masks for each directional category
        mask_1front = (gt_labels == 0)
        mask_2front = (gt_labels == 1)
        mask_4front = (gt_labels == 2)

        count_1front = mask_1front.sum().item()
        count_2front = mask_2front.sum().item()
        count_4front = mask_4front.sum().item()

        # Accumulate weighted losses for micro-average
        total_weight_sum = torch.tensor(0.0, device=device)

        # =====================================================================
        # 1-front loss (label == 0) with Soft Gate weighting
        # Key: ALWAYS include cosine loss to ensure μ has gradient
        # =====================================================================
        if count_1front > 0:
            head_out = model_output['head_1front']
            mu = head_out['mu'][mask_1front, 0]  # [N]
            kappa = head_out['kappa'][mask_1front, 0]  # [N]
            gt = gt_angles[mask_1front]  # [N]
            w = sample_weights[mask_1front]  # [N]

            # Cosine loss: gradient doesn't depend on κ
            cosine_loss_per_sample = 1 - torch.cos(mu - gt)  # [N]

            # NLL loss
            nll_per_sample = von_mises_nll(gt, mu, kappa)  # [N]

            # Combined loss: NLL + cosine (always)
            loss_per_sample = nll_per_sample + cosine_weight * cosine_loss_per_sample

            # Accumulate for micro-average (weighted by sample count)
            w_sum = w.sum()
            loss_1front = (w * loss_per_sample).sum() / (w_sum + 1e-6)  # For logging
            total_loss = total_loss + (w * loss_per_sample).sum()
            total_weight_sum = total_weight_sum + w_sum

            # κ regularization (only for 1-front, staged)
            # Use w_gate only (not w_cls) to ensure κ reg is effective even with low p_gt
            if kappa_reg_weight > 0:
                # Compute w_gate separately for κ reg (more stable)
                p_dir = gate_weights[mask_1front, 0] + gate_weights[mask_1front, 1] + gate_weights[mask_1front, 2]
                denom = 1.0 - self.p_dir_threshold
                w_gate_only = torch.clamp((p_dir - self.p_dir_threshold) / (denom + 1e-8), 0.1, 1)  # min=0.1

                kappa_gap = F.relu(kappa_target - kappa)  # [N]
                loss_kappa_reg = kappa_reg_weight * (w_gate_only * kappa_gap).mean()

        # =====================================================================
        # 2-front loss (label == 1) - Hungarian matching + Soft Gate weighting
        # =====================================================================
        if count_2front > 0:
            head_out = model_output['head_2front']
            mu = head_out['mu'][mask_2front]  # [N, 2]
            kappa = head_out['kappa'][mask_2front]  # [N, 2]
            gt = gt_angles[mask_2front]  # [N]
            w = sample_weights[mask_2front]  # [N]

            # Generate 2 symmetric GT angles: [gt, gt + π]
            gt_sym = torch.stack([gt, gt + math.pi], dim=1)  # [N, 2]

            # Hungarian matching
            gt_matched = batched_hungarian_match_k2(mu, gt_sym)  # [N, 2]

            # NLL for matched pairs
            nll = von_mises_nll(gt_matched, mu, kappa)  # [N, 2]
            nll_per_sample = nll.mean(dim=1)  # [N]

            # Accumulate for micro-average
            w_sum = w.sum()
            loss_2front = (w * nll_per_sample).sum() / (w_sum + 1e-6)  # For logging
            total_loss = total_loss + (w * nll_per_sample).sum()
            total_weight_sum = total_weight_sum + w_sum

        # =====================================================================
        # 4-front loss (label == 2) - Hungarian matching + Soft Gate weighting
        # =====================================================================
        if count_4front > 0:
            head_out = model_output['head_4front']
            mu = head_out['mu'][mask_4front]  # [N, 4]
            kappa = head_out['kappa'][mask_4front]  # [N, 4]
            gt = gt_angles[mask_4front]  # [N]
            w = sample_weights[mask_4front]  # [N]

            # Generate 4 symmetric GT angles
            offsets = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2], device=device)
            gt_sym = gt.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 4]

            # Hungarian matching
            perms_4 = self.perms_4.to(device)
            gt_permuted = gt_sym[:, perms_4]  # [N, 24, 4]
            pred_expanded = mu.unsqueeze(1)  # [N, 1, 4]
            costs = (1 - torch.cos(pred_expanded - gt_permuted)).sum(dim=2)  # [N, 24]
            best_perm_idx = costs.argmin(dim=1)  # [N]
            best_perms = perms_4[best_perm_idx]  # [N, 4]
            gt_matched = torch.gather(gt_sym, 1, best_perms)  # [N, 4]

            # NLL for matched pairs
            nll = von_mises_nll(gt_matched, mu, kappa)  # [N, 4]
            nll_per_sample = nll.mean(dim=1)  # [N]

            # Accumulate for micro-average
            w_sum = w.sum()
            loss_4front = (w * nll_per_sample).sum() / (w_sum + 1e-6)  # For logging
            total_loss = total_loss + (w * nll_per_sample).sum()
            total_weight_sum = total_weight_sum + w_sum

        # Micro-average: normalize by total weight sum
        if total_weight_sum > 0:
            total_loss = total_loss / total_weight_sum

        # Add κ regularization (already normalized)
        total_loss = total_loss + loss_kappa_reg

        # =====================================================================
        # Optional: Classification loss for joint fine-tuning
        # =====================================================================
        classification_loss = torch.tensor(0.0, device=device)
        if self.classification_weight > 0:
            logits = model_output['logits']  # [B, 5] raw logits

            if not logits.requires_grad and self.training:
                raise RuntimeError(
                    "classification_weight > 0 but logits have no gradient. "
                    "Set freeze_classifier=False when using classification_weight > 0 "
                    "to enable classifier fine-tuning."
                )

            classification_loss = self.ce_loss(logits, gt_labels)
            total_loss = total_loss + self.classification_weight * classification_loss

        # Compute average sample weight for logging
        avg_sample_weight = sample_weights[gt_labels < 3].mean().item() if (gt_labels < 3).any() else 0.0

        return {
            'loss': total_loss,
            'loss_1front': loss_1front,
            'loss_2front': loss_2front,
            'loss_4front': loss_4front,
            'loss_kappa_reg': loss_kappa_reg,
            'loss_classification': classification_loss,
            'count_1front': count_1front,
            'count_2front': count_2front,
            'count_4front': count_4front,
            'avg_sample_weight': avg_sample_weight,
            'cosine_weight': cosine_weight,
            'kappa_reg_weight': kappa_reg_weight,
            'kappa_target': kappa_target,
        }


# ==============================================================================
# Inference Helper Functions
# ==============================================================================

def get_final_pdf(model_output: dict,
                  theta_range: torch.Tensor = None,
                  num_points: int = 360) -> torch.Tensor:
    """
    Compute the final probability density function for inference using soft routing.

    The final PDF is a weighted combination of expert PDFs:
        P(θ) = w_0 * P_head1(θ) + w_1 * P_head2(θ) + w_2 * P_head4(θ)
               + (w_3 + w_4) * Uniform(1/2π)

    where w_i are the classifier softmax outputs.

    Uses numerically stable log-space computation for von Mises PDF.

    Args:
        model_output: Output from ProbabilisticOrientationNet forward()
        theta_range: Angles at which to evaluate PDF [num_points], default [-π, π]
        num_points: Number of points for theta_range if not provided

    Returns:
        pdf: [B, num_points] probability density values
    """
    device = model_output['weights'].device
    B = model_output['weights'].size(0)

    # Default theta range
    if theta_range is None:
        theta_range = torch.linspace(-math.pi, math.pi, num_points, device=device)

    num_pts = theta_range.size(0)
    weights = model_output['weights']  # [B, 5]

    # Initialize PDF
    pdf = torch.zeros(B, num_pts, device=device)

    # Helper: von Mises PDF using log-space for numerical stability
    def vm_pdf_stable(theta, mu, kappa):
        """
        Compute von Mises PDF using log-space for numerical stability.

        Args:
            theta: [B, num_pts] angles to evaluate
            mu: [B, num_pts] mean angles
            kappa: [B, num_pts] concentration parameters

        Returns:
            pdf: [B, num_pts] probability density values
        """
        cos_diff = torch.cos(theta - mu)
        # Use log_bessel_i0 for numerical stability
        log_prob = kappa * cos_diff - math.log(2 * math.pi) - log_bessel_i0(kappa)
        return torch.exp(log_prob)

    # Expand theta for broadcasting: [B, num_pts]
    theta_exp = theta_range.unsqueeze(0).expand(B, -1)

    # 1. Head 1 (1-front) contribution
    head1 = model_output['head_1front']
    mu1 = head1['mu']  # [B, 1]
    kappa1 = head1['kappa']  # [B, 1]
    mu1_exp = mu1.expand(-1, num_pts)  # [B, num_pts]
    kappa1_exp = kappa1.expand(-1, num_pts)  # [B, num_pts]

    pdf_1front = vm_pdf_stable(theta_exp, mu1_exp, kappa1_exp)  # [B, num_pts]
    pdf = pdf + weights[:, 0:1] * pdf_1front

    # 2. Head 2 (2-front) contribution - mixture of 2 von Mises
    head2 = model_output['head_2front']
    mu2 = head2['mu']  # [B, 2]
    kappa2 = head2['kappa']  # [B, 2]
    mix_w2 = head2['mix_weights']  # [2]

    pdf_2front = torch.zeros(B, num_pts, device=device)
    for k in range(2):
        mu_k = mu2[:, k:k+1].expand(-1, num_pts)  # [B, num_pts]
        kappa_k = kappa2[:, k:k+1].expand(-1, num_pts)  # [B, num_pts]
        pdf_2front = pdf_2front + mix_w2[k] * vm_pdf_stable(theta_exp, mu_k, kappa_k)

    pdf = pdf + weights[:, 1:2] * pdf_2front

    # 3. Head 4 (4-front) contribution - mixture of 4 von Mises
    head4 = model_output['head_4front']
    mu4 = head4['mu']  # [B, 4]
    kappa4 = head4['kappa']  # [B, 4]
    mix_w4 = head4['mix_weights']  # [4]

    pdf_4front = torch.zeros(B, num_pts, device=device)
    for k in range(4):
        mu_k = mu4[:, k:k+1].expand(-1, num_pts)  # [B, num_pts]
        kappa_k = kappa4[:, k:k+1].expand(-1, num_pts)  # [B, num_pts]
        pdf_4front = pdf_4front + mix_w4[k] * vm_pdf_stable(theta_exp, mu_k, kappa_k)

    pdf = pdf + weights[:, 2:3] * pdf_4front

    # 4. Rotational symmetric + No-front -> Uniform distribution
    uniform_prob = 1.0 / (2 * math.pi)
    pdf = pdf + (weights[:, 3:4] + weights[:, 4:5]) * uniform_prob

    return pdf


def get_peak_predictions(model_output: dict, gt_labels: torch.Tensor = None) -> dict:
    """
    Extract peak predictions based on classifier weights or GT labels.

    For multi-front cases (2-front, 4-front), returns the first peak (index 0)
    since all peaks are equivalent directions. Also returns all peaks for
    detailed analysis.

    Args:
        model_output: Output from ProbabilisticOrientationNet
        gt_labels: Optional GT labels for hard routing. If None, uses soft routing.

    Returns:
        dict with:
            - 'predicted_angles': [B] primary angle (first peak for multi-front)
            - 'confidence': [B] confidence (kappa of the primary peak)
            - 'predicted_class': [B] predicted symmetry class (0-4)
            - 'all_mu': dict with 'head_1front', 'head_2front', 'head_4front' raw outputs
            - 'all_kappa': dict with kappa values for each head
    """
    weights = model_output['weights']  # [B, 5]
    B = weights.size(0)
    device = weights.device

    if gt_labels is not None:
        # Hard routing based on GT
        predicted_class = gt_labels
    else:
        # Soft routing: use argmax of classifier
        predicted_class = weights.argmax(dim=1)  # [B]

    predicted_angles = torch.zeros(B, device=device)
    confidence = torch.zeros(B, device=device)

    for i in range(B):
        cls = predicted_class[i].item()

        if cls == 0:  # 1-front
            head = model_output['head_1front']
            predicted_angles[i] = head['mu'][i, 0]
            confidence[i] = head['kappa'][i, 0]

        elif cls == 1:  # 2-front
            head = model_output['head_2front']
            # Return the first peak (all peaks are equivalent)
            predicted_angles[i] = head['mu'][i, 0]
            confidence[i] = head['kappa'][i, 0]

        elif cls == 2:  # 4-front
            head = model_output['head_4front']
            # Return the first peak (all peaks are equivalent)
            predicted_angles[i] = head['mu'][i, 0]
            confidence[i] = head['kappa'][i, 0]

        else:  # 3 (Rot) or 4 (No-front)
            # No meaningful direction prediction
            predicted_angles[i] = 0.0
            confidence[i] = 0.0

    return {
        'predicted_angles': predicted_angles,
        'confidence': confidence,
        'predicted_class': predicted_class,
        # Include all peaks for detailed analysis
        'all_mu': {
            'head_1front': model_output['head_1front']['mu'],
            'head_2front': model_output['head_2front']['mu'],
            'head_4front': model_output['head_4front']['mu'],
        },
        'all_kappa': {
            'head_1front': model_output['head_1front']['kappa'],
            'head_2front': model_output['head_2front']['kappa'],
            'head_4front': model_output['head_4front']['kappa'],
        }
    }


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Testing ProbabilisticOrientationNet")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a dummy classifier for testing
    from .symmetry_classifier import SymmetryClassifier_GlobalConcat
    classifier = SymmetryClassifier_GlobalConcat(num_classes=5).to(device)

    # Test with freeze_classifier=True (default)
    print("\n--- Testing with freeze_classifier=True ---")
    model = ProbabilisticOrientationNet(
        classifier=classifier,
        backbone_dim=1024,
        expert_hidden_dim=256,
        freeze_classifier=True
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    B, N = 4, 2048
    x = torch.randn(B, N, 3).to(device)
    upright_vec = torch.tensor([[0, 1, 0]] * B, dtype=torch.float32).to(device)

    print(f"\nInput shape: {x.shape}")

    # Test eval mode determinism
    model.eval()
    with torch.no_grad():
        output1 = model(x, upright_vec)
        output2 = model(x, upright_vec)

    # Check determinism
    mu_diff = (output1['head_1front']['mu'] - output2['head_1front']['mu']).abs().max()
    print(f"Eval mode determinism check - mu diff: {mu_diff.item():.6f}")
    assert mu_diff < 1e-5, "Model is not deterministic in eval mode!"
    print("✅ Eval mode is deterministic")

    print(f"\nOutput keys: {output1.keys()}")
    print(f"  weights shape: {output1['weights'].shape}")
    print(f"  head_1front mu shape: {output1['head_1front']['mu'].shape}")
    print(f"  head_2front mu shape: {output1['head_2front']['mu'].shape}")
    print(f"  head_4front mu shape: {output1['head_4front']['mu'].shape}")

    # Test with [B, 3, N] format
    x_transposed = x.transpose(1, 2)
    print(f"\nTesting with transposed input: {x_transposed.shape}")
    with torch.no_grad():
        output3 = model(x_transposed, upright_vec)
    print(f"  Output weights shape: {output3['weights'].shape}")

    # Test loss function
    print("\n" + "=" * 80)
    print("Testing MaskedExpertLoss (with vectorized Hungarian matching)")
    print("=" * 80)

    loss_fn = MaskedExpertLoss()

    # Create dummy GT
    gt_angles = torch.randn(B).to(device)
    gt_labels = torch.tensor([0, 1, 2, 3]).to(device)  # Mix of all classes

    model.train()
    output = model(x, upright_vec)
    loss_dict = loss_fn(output, gt_angles, gt_labels)

    print(f"Loss: {loss_dict['loss'].item():.4f}")
    print(f"  1-front loss: {loss_dict['loss_1front'].item():.4f} (count: {loss_dict['count_1front']})")
    print(f"  2-front loss: {loss_dict['loss_2front'].item():.4f} (count: {loss_dict['count_2front']})")
    print(f"  4-front loss: {loss_dict['loss_4front'].item():.4f} (count: {loss_dict['count_4front']})")

    # Test get_final_pdf
    print("\n" + "=" * 80)
    print("Testing get_final_pdf (with stable log-space computation)")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        output = model(x, upright_vec)
        pdf = get_final_pdf(output, num_points=360)

    print(f"PDF shape: {pdf.shape}")
    print(f"PDF sum (should be ~1): {pdf.sum(dim=1) * (2 * math.pi / 360)}")

    # Test with freeze_classifier=False and classification_weight > 0
    print("\n" + "=" * 80)
    print("Testing with freeze_classifier=False + classification_weight=1.0")
    print("=" * 80)

    model_finetune = ProbabilisticOrientationNet(
        classifier=SymmetryClassifier_GlobalConcat(num_classes=5).to(device),
        freeze_classifier=False
    ).to(device)

    trainable_ft = sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)
    print(f"Trainable parameters (with classifier): {trainable_ft:,}")

    # Use loss with classification_weight > 0 to enable joint fine-tuning
    loss_fn_ft = MaskedExpertLoss(classification_weight=1.0)

    # Check that classifier gradients flow
    model_finetune.train()
    output_ft = model_finetune(x, upright_vec)
    loss_dict_ft = loss_fn_ft(output_ft, gt_angles, gt_labels)
    print(f"Classification loss: {loss_dict_ft['loss_classification'].item():.4f}")
    loss_dict_ft['loss'].backward()

    classifier_grad = model_finetune.classifier.fc1.weight.grad
    print(f"Classifier gradient exists: {classifier_grad is not None}")
    if classifier_grad is not None:
        print(f"Classifier gradient norm: {classifier_grad.norm().item():.6f}")
    assert classifier_grad is not None, "Classifier gradients should flow with classification_weight > 0"
    print("✅ Classifier fine-tuning works")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

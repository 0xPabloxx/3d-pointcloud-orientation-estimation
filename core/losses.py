"""
Loss functions for orientation prediction.

Supported losses:
- MSE: Mean Squared Error for direct (cos, sin) regression
- Cosine: Cosine similarity loss
- NLL_VM: Negative Log-Likelihood for von Mises distribution
- NLL_MVM: Negative Log-Likelihood for mixture of von Mises

Author: Claude
Created: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import i0e  # exponentially scaled I0 Bessel function


def log_bessel_i0(kappa: torch.Tensor) -> torch.Tensor:
    """
    Compute log(I0(kappa)) in a numerically stable way.

    For large kappa, I0(kappa) ≈ exp(kappa) / sqrt(2*pi*kappa)
    So log(I0(kappa)) ≈ kappa - 0.5*log(2*pi*kappa)

    We use: log(I0(kappa)) = kappa + log(I0e(kappa))
    where I0e(kappa) = I0(kappa) * exp(-kappa) is the exponentially scaled version

    Args:
        kappa: concentration parameter, shape (*, )

    Returns:
        log(I0(kappa)), same shape as input
    """
    # i0e is numerically stable for large kappa
    return kappa + torch.log(i0e(kappa) + 1e-10)


def von_mises_nll(theta_gt: torch.Tensor,
                  mu: torch.Tensor,
                  kappa: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for von Mises distribution.

    p(θ|μ,κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))
    NLL = -κ * cos(θ_gt - μ) + log(2π) + log(I₀(κ))

    Args:
        theta_gt: Ground truth angle in radians, shape (B,)
        mu: Predicted mean angle in radians, shape (B,)
        kappa: Predicted concentration parameter, shape (B,)

    Returns:
        NLL loss, shape (B,)
    """
    cos_diff = torch.cos(theta_gt - mu)
    nll = -kappa * cos_diff + torch.log(torch.tensor(2 * 3.14159265359)) + log_bessel_i0(kappa)
    return nll


def mixture_von_mises_nll(theta_gt: torch.Tensor,
                          mu: torch.Tensor,
                          kappa: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for mixture of von Mises distributions.

    p(θ|params) = Σᵢ wᵢ * vM(θ|μᵢ, κᵢ)
    NLL = -log(Σᵢ wᵢ * exp(κᵢ * cos(θ_gt - μᵢ)) / (2π * I₀(κᵢ)))

    Using log-sum-exp trick for numerical stability.

    Args:
        theta_gt: Ground truth angle in radians, shape (B,)
        mu: Predicted mean angles, shape (B, K) where K is number of peaks
        kappa: Predicted concentration parameters, shape (B, K)
        weights: Mixture weights (should sum to 1), shape (B, K)

    Returns:
        NLL loss, shape (B,)
    """
    B, K = mu.shape

    # Expand theta_gt to (B, K)
    theta_gt_expanded = theta_gt.unsqueeze(1).expand(-1, K)

    # Compute log probability for each component
    # log p_i = κᵢ * cos(θ_gt - μᵢ) - log(2π) - log(I₀(κᵢ))
    cos_diff = torch.cos(theta_gt_expanded - mu)
    log_prob_components = kappa * cos_diff - torch.log(torch.tensor(2 * 3.14159265359)) - log_bessel_i0(kappa)

    # Add log weights: log(wᵢ * p_i) = log(wᵢ) + log(p_i)
    log_weights = torch.log(weights + 1e-10)
    log_weighted_probs = log_weights + log_prob_components

    # Log-sum-exp to get log(Σᵢ wᵢ * p_i)
    log_mixture_prob = torch.logsumexp(log_weighted_probs, dim=1)

    # NLL is negative of log probability
    nll = -log_mixture_prob

    return nll


class DirectRegressionLoss(nn.Module):
    """
    Loss for direct (cos θ, sin θ) regression.

    Supports:
    - MSE: Mean squared error
    - Cosine: 1 - cosine similarity
    """

    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted (cos, sin), shape (B, 2), should be normalized
            target: Ground truth (cos, sin), shape (B, 2), normalized

        Returns:
            Loss scalar
        """
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'cosine':
            # 1 - cosine similarity (cosine is dot product for unit vectors)
            cos_sim = (pred * target).sum(dim=1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class VonMisesNLLLoss(nn.Module):
    """
    NLL loss for single-peak von Mises distribution.
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                mu: torch.Tensor,
                kappa: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Predicted direction as (cos, sin), shape (B, 2), normalized
            kappa: Predicted concentration, shape (B,)
            target: Ground truth (cos, sin), shape (B, 2), normalized

        Returns:
            Loss scalar
        """
        # Convert (cos, sin) to angle
        theta_gt = torch.atan2(target[:, 1], target[:, 0])
        mu_angle = torch.atan2(mu[:, 1], mu[:, 0])

        nll = von_mises_nll(theta_gt, mu_angle, kappa)
        return nll.mean()


class MixtureVonMisesNLLLoss(nn.Module):
    """
    NLL loss for mixture of von Mises distributions.

    For K-peak prediction, the GT is a single direction.
    The mixture NLL allows any of the K peaks to explain the GT.
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                mu: torch.Tensor,
                kappa: torch.Tensor,
                weights: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Predicted directions as (cos, sin), shape (B, K, 2), normalized
            kappa: Predicted concentrations, shape (B, K)
            weights: Mixture weights, shape (B, K), should sum to 1
            target: Ground truth (cos, sin), shape (B, 2), normalized

        Returns:
            Loss scalar
        """
        B, K, _ = mu.shape

        # Convert (cos, sin) to angles
        theta_gt = torch.atan2(target[:, 1], target[:, 0])  # (B,)
        mu_angles = torch.atan2(mu[:, :, 1], mu[:, :, 0])   # (B, K)

        nll = mixture_von_mises_nll(theta_gt, mu_angles, kappa, weights)
        return nll.mean()


def get_loss(loss_config: dict) -> nn.Module:
    """
    Factory function to get loss module based on config.

    Args:
        loss_config: dict with keys:
            - type: 'direct_mse', 'direct_cosine', 'vm_nll', 'mvm_nll'

    Returns:
        Loss module
    """
    loss_type = loss_config.get('type', 'direct_mse')

    if loss_type == 'direct_mse':
        return DirectRegressionLoss(loss_type='mse')
    elif loss_type == 'direct_cosine':
        return DirectRegressionLoss(loss_type='cosine')
    elif loss_type == 'vm_nll':
        return VonMisesNLLLoss()
    elif loss_type == 'mvm_nll':
        return MixtureVonMisesNLLLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

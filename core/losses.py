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


class sCVAEMonteCarleLoss(nn.Module):
    """
    Monte Carlo NLL loss for sCVAE-style models.

    Based on "Deep Directional Statistics" (Prokudin et al., ECCV 2018).

    The key idea is to estimate the mixture likelihood by averaging
    likelihoods from multiple stochastic samples:

        p(θ|x) ≈ (1/N) Σᵢ vM(θ|μᵢ, κᵢ)

    where (μᵢ, κᵢ) are decoded from [features, zᵢ] with random noise zᵢ.

    An additional uniform distribution is mixed in for robustness:

        p(θ|x) = γ * U(θ) + (1-γ) * (1/N) Σᵢ vM(θ|μᵢ, κᵢ)

    where U(θ) = 1/(2π) is the uniform distribution on the circle.
    """

    def __init__(self, gamma: float = 0.1):
        """
        Args:
            gamma: Weight of uniform distribution for robustness (default 0.1).
                   Higher gamma = more regularization, less overfitting.
                   gamma=0 means pure von Mises mixture.
        """
        super().__init__()
        self.gamma = gamma
        self.log_uniform = -torch.log(torch.tensor(2 * 3.14159265359))  # log(1/2π)

    def forward(self,
                mu: torch.Tensor,
                kappa: torch.Tensor,
                weights: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute Monte Carlo NLL loss.

        Args:
            mu: Predicted directions as (cos, sin), shape (B, N, 2), normalized
            kappa: Predicted concentrations, shape (B, N)
            weights: Sample weights (usually uniform 1/N), shape (B, N)
            target: Ground truth (cos, sin), shape (B, 2), normalized

        Returns:
            Loss scalar
        """
        B, N, _ = mu.shape
        device = mu.device

        # Convert to angles
        theta_gt = torch.atan2(target[:, 1], target[:, 0])  # (B,)
        mu_angles = torch.atan2(mu[:, :, 1], mu[:, :, 0])   # (B, N)

        # Expand theta_gt for broadcasting: (B,) -> (B, N)
        theta_gt_expanded = theta_gt.unsqueeze(1).expand(-1, N)

        # Compute von Mises log-likelihood for each sample
        # log p_i = κᵢ * cos(θ_gt - μᵢ) - log(2π) - log(I₀(κᵢ))
        cos_diff = torch.cos(theta_gt_expanded - mu_angles)
        log_prob = kappa * cos_diff - torch.log(torch.tensor(2 * 3.14159265359, device=device)) - log_bessel_i0(kappa)

        # Convert to likelihood (not log)
        likelihood = torch.exp(log_prob)  # (B, N)

        # Average likelihood across samples (Monte Carlo estimate)
        mean_likelihood = (likelihood * weights).sum(dim=1)  # (B,)

        # Mix with uniform distribution for robustness
        # p(θ|x) = γ * U(θ) + (1-γ) * mean_likelihood
        # where U(θ) = 1/(2π)
        uniform_prob = 1.0 / (2 * 3.14159265359)
        mixed_likelihood = self.gamma * uniform_prob + (1 - self.gamma) * mean_likelihood

        # NLL
        nll = -torch.log(mixed_likelihood + 1e-10)

        return nll.mean()


class sCVAEMonteCarloLossMultiGT(nn.Module):
    """
    Monte Carlo NLL loss for sCVAE with multiple valid GT directions.

    For symmetric objects like glassbox (K=4), there are 4 equivalent correct
    directions: θ, θ+90°, θ+180°, θ+270°.

    The likelihood is computed as:
        p(θ|x) = max over all K GTs of: (1/N) Σᵢ vM(θ_k|μᵢ, κᵢ)

    This allows any of the K symmetric directions to be considered correct.

    Alternatively, we can use a "soft" version that sums likelihoods:
        p(θ|x) = (1/K) Σ_k (1/N) Σᵢ vM(θ_k|μᵢ, κᵢ)
    """

    def __init__(self, gamma: float = 0.1, K: int = 4, mode: str = 'max'):
        """
        Args:
            gamma: Weight of uniform distribution for robustness
            K: Number of symmetric directions (e.g., 4 for 90° symmetry)
            mode: 'max' (take best GT) or 'sum' (average over all GTs)
        """
        super().__init__()
        self.gamma = gamma
        self.K = K
        self.mode = mode

        # Precompute symmetric offsets: 0, π/2, π, 3π/2 for K=4
        self.register_buffer(
            'angle_offsets',
            torch.tensor([2 * 3.14159265359 * k / K for k in range(K)])
        )

    def forward(self,
                mu: torch.Tensor,
                kappa: torch.Tensor,
                weights: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute Monte Carlo NLL loss with multiple valid GTs.

        Args:
            mu: Predicted directions (cos, sin), shape (B, N, 2)
            kappa: Predicted concentrations, shape (B, N)
            weights: Sample weights, shape (B, N)
            target: Ground truth (cos, sin), shape (B, 2) - ONE of the K valid directions

        Returns:
            Loss scalar
        """
        B, N, _ = mu.shape
        device = mu.device
        K = self.K

        # Convert predictions to angles
        mu_angles = torch.atan2(mu[:, :, 1], mu[:, :, 0])  # (B, N)

        # Get base GT angle
        theta_gt_base = torch.atan2(target[:, 1], target[:, 0])  # (B,)

        # Generate all K valid GT angles: (B, K)
        angle_offsets = self.angle_offsets.to(device)  # (K,)
        theta_gt_all = theta_gt_base.unsqueeze(1) + angle_offsets.unsqueeze(0)  # (B, K)

        # Compute likelihood for each (sample, GT) pair
        # mu_angles: (B, N), theta_gt_all: (B, K)
        # We need (B, N, K) for all combinations
        mu_angles_exp = mu_angles.unsqueeze(2)  # (B, N, 1)
        theta_gt_exp = theta_gt_all.unsqueeze(1)  # (B, 1, K)

        cos_diff = torch.cos(mu_angles_exp - theta_gt_exp)  # (B, N, K)

        # von Mises log-likelihood: κ * cos(θ - μ) - log(2π) - log(I₀(κ))
        kappa_exp = kappa.unsqueeze(2)  # (B, N, 1)
        log_prob = kappa_exp * cos_diff - torch.log(torch.tensor(2 * 3.14159265359, device=device)) - log_bessel_i0(kappa_exp)
        # log_prob: (B, N, K)

        # Convert to likelihood
        likelihood = torch.exp(log_prob)  # (B, N, K)

        # Weighted average over samples for each GT
        weights_exp = weights.unsqueeze(2)  # (B, N, 1)
        mean_likelihood_per_gt = (likelihood * weights_exp).sum(dim=1)  # (B, K)

        # Combine likelihoods across GTs
        if self.mode == 'max':
            # Take the best matching GT
            mean_likelihood = mean_likelihood_per_gt.max(dim=1)[0]  # (B,)
        else:  # 'sum' or 'avg'
            # Average over all GTs (softer version)
            mean_likelihood = mean_likelihood_per_gt.mean(dim=1)  # (B,)

        # Mix with uniform distribution
        uniform_prob = 1.0 / (2 * 3.14159265359)
        mixed_likelihood = self.gamma * uniform_prob + (1 - self.gamma) * mean_likelihood

        # NLL
        nll = -torch.log(mixed_likelihood + 1e-10)

        return nll.mean()


class sCVAEMonteCarleLossWithEntropy(nn.Module):
    """
    sCVAE loss with optional entropy regularization.

    Adds entropy term to encourage diversity in predictions:
        L = NLL - β * H(samples)

    where H(samples) measures how spread out the N samples are.
    """

    def __init__(self, gamma: float = 0.1, entropy_weight: float = 0.0):
        """
        Args:
            gamma: Weight of uniform distribution
            entropy_weight: Weight for entropy regularization (0 = no regularization)
        """
        super().__init__()
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.base_loss = sCVAEMonteCarleLoss(gamma=gamma)

    def forward(self,
                mu: torch.Tensor,
                kappa: torch.Tensor,
                weights: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: (B, N, 2)
            kappa: (B, N)
            weights: (B, N)
            target: (B, 2)

        Returns:
            Loss scalar
        """
        # Base NLL loss
        nll_loss = self.base_loss(mu, kappa, weights, target)

        if self.entropy_weight > 0:
            # Compute sample diversity (entropy-like measure)
            # Simple approach: variance of predicted angles
            mu_angles = torch.atan2(mu[:, :, 1], mu[:, :, 0])  # (B, N)

            # Circular variance (using complex mean)
            # High variance = high diversity = good
            cos_mu = torch.cos(mu_angles)
            sin_mu = torch.sin(mu_angles)
            mean_cos = cos_mu.mean(dim=1)
            mean_sin = sin_mu.mean(dim=1)
            R = torch.sqrt(mean_cos**2 + mean_sin**2)  # R close to 0 = high spread
            circular_variance = 1 - R  # High = diverse

            # We want to maximize diversity, so subtract
            diversity_bonus = self.entropy_weight * circular_variance.mean()
            return nll_loss - diversity_bonus
        else:
            return nll_loss


def get_loss(loss_config: dict) -> nn.Module:
    """
    Factory function to get loss module based on config.

    Args:
        loss_config: dict with keys:
            - type: 'direct_mse', 'direct_cosine', 'vm_nll', 'mvm_nll',
                    'scvae_mc', 'scvae_mc_multi_gt', 'scvae_mc_entropy'
            - gamma: (sCVAE only) uniform mixture weight (default 0.1)
            - K: (scvae_mc_multi_gt only) number of symmetric directions (default 4)
            - mode: (scvae_mc_multi_gt only) 'max' or 'sum' (default 'max')
            - entropy_weight: (sCVAE entropy only) diversity bonus weight (default 0.01)

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
    elif loss_type == 'scvae_mc':
        gamma = loss_config.get('gamma', 0.1)
        return sCVAEMonteCarleLoss(gamma=gamma)
    elif loss_type == 'scvae_mc_multi_gt':
        # For symmetric objects with K valid GT directions
        gamma = loss_config.get('gamma', 0.1)
        K = loss_config.get('K', 4)
        mode = loss_config.get('mode', 'max')
        return sCVAEMonteCarloLossMultiGT(gamma=gamma, K=K, mode=mode)
    elif loss_type == 'scvae_mc_entropy':
        gamma = loss_config.get('gamma', 0.1)
        entropy_weight = loss_config.get('entropy_weight', 0.01)
        return sCVAEMonteCarleLossWithEntropy(gamma=gamma, entropy_weight=entropy_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

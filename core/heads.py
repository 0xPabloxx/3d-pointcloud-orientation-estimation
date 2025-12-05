"""
Prediction heads for orientation prediction.

Supported heads:
- DirectHead: Direct regression of (cos θ, sin θ)
- VM1PeakHead: Single-peak von Mises (μ, κ)
- VM2PeakHead: 2-peak von Mises with fixed equal weights
- VM4PeakHead: 4-peak von Mises with fixed equal weights
- VM4PeakLearnableHead: 4-peak von Mises with learnable weights

All heads output normalized (cos θ, sin θ) for direction.
κ is learnable and output via softplus to ensure positivity.

Author: Claude
Created: 2025-12-05
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectHead(nn.Module):
    """
    Direct regression head for (cos θ, sin θ).

    Output: (B, 2) normalized direction vector
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head_type = 'direct'

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, input_dim) global features from backbone

        Returns:
            dict with:
                - 'direction': (B, 2) normalized (cos, sin)
        """
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.fc2(x)

        # Normalize to unit vector
        direction = F.normalize(x, dim=-1)

        return {'direction': direction}


class VM1PeakHead(nn.Module):
    """
    Single-peak von Mises head.

    Output:
        - μ: (B, 2) normalized direction (cos, sin)
        - κ: (B,) concentration parameter (positive)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, kappa_init: float = 1.0):
        super().__init__()
        self.head_type = 'vm_1peak'
        self.kappa_init = kappa_init

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Direction output (2D for cos, sin)
        self.fc_mu = nn.Linear(hidden_dim, 2)

        # Kappa output (1D, will use softplus)
        self.fc_kappa = nn.Linear(hidden_dim, 1)

        # Initialize kappa bias so that softplus(bias) ≈ kappa_init
        # softplus(x) = log(1 + exp(x)), inverse: x = log(exp(kappa_init) - 1)
        if kappa_init > 0:
            init_bias = math.log(math.exp(kappa_init) - 1) if kappa_init > 0.5 else kappa_init
            self.fc_kappa.bias.data.fill_(init_bias)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, input_dim) global features from backbone

        Returns:
            dict with:
                - 'mu': (B, 2) normalized direction
                - 'kappa': (B,) concentration parameter
                - 'weights': (B, 1) all ones (for API consistency)
        """
        x = F.relu(self.bn1(self.fc1(features)))

        # Direction (normalized)
        mu_raw = self.fc_mu(x)
        mu = F.normalize(mu_raw, dim=-1)

        # Kappa (positive via softplus)
        kappa = F.softplus(self.fc_kappa(x)).squeeze(-1)

        # Weights (fixed to 1 for single peak)
        weights = torch.ones(features.size(0), 1, device=features.device)

        return {
            'mu': mu.unsqueeze(1),      # (B, 1, 2) for consistency
            'kappa': kappa.unsqueeze(1), # (B, 1)
            'weights': weights           # (B, 1)
        }


class VMMultiPeakHead(nn.Module):
    """
    Multi-peak von Mises head with fixed or learnable weights.

    Output:
        - μ: (B, K, 2) normalized directions
        - κ: (B, K) concentration parameters
        - weights: (B, K) mixture weights (sum to 1)
    """

    def __init__(self,
                 input_dim: int,
                 num_peaks: int,
                 hidden_dim: int = 256,
                 kappa_init: float = 5.0,
                 learnable_weights: bool = False,
                 init_spread: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            num_peaks: Number of peaks (K)
            hidden_dim: Hidden layer dimension
            kappa_init: Initial kappa value
            learnable_weights: If True, weights are predicted; if False, fixed to 1/K
            init_spread: If True, initialize mu directions spread evenly
        """
        super().__init__()
        self.head_type = f'vm_{num_peaks}peak' + ('_learnable' if learnable_weights else '')
        self.num_peaks = num_peaks
        self.kappa_init = kappa_init
        self.learnable_weights = learnable_weights
        self.init_spread = init_spread

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Direction outputs: K * 2 values
        self.fc_mu = nn.Linear(hidden_dim, num_peaks * 2)

        # Kappa outputs: K values
        self.fc_kappa = nn.Linear(hidden_dim, num_peaks)

        # Weight outputs (optional)
        if learnable_weights:
            self.fc_weights = nn.Linear(hidden_dim, num_peaks)
        else:
            self.fc_weights = None

        # Initialize
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to encourage spread directions."""
        # Kappa initialization
        if self.kappa_init > 0:
            init_bias = math.log(math.exp(self.kappa_init) - 1) if self.kappa_init > 0.5 else self.kappa_init
            self.fc_kappa.bias.data.fill_(init_bias)

        # Mu initialization: spread evenly around the circle
        if self.init_spread:
            # Initialize bias to spread peaks evenly
            angles = [2 * math.pi * i / self.num_peaks for i in range(self.num_peaks)]
            init_mu = []
            for angle in angles:
                init_mu.extend([math.cos(angle), math.sin(angle)])
            self.fc_mu.bias.data = torch.tensor(init_mu, dtype=torch.float32)
            # Small weights to let bias dominate initially
            self.fc_mu.weight.data *= 0.01

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, input_dim) global features from backbone

        Returns:
            dict with:
                - 'mu': (B, K, 2) normalized directions
                - 'kappa': (B, K) concentration parameters
                - 'weights': (B, K) mixture weights
        """
        B = features.size(0)
        K = self.num_peaks

        x = F.relu(self.bn1(self.fc1(features)))

        # Directions: (B, K*2) -> (B, K, 2) -> normalize
        mu_raw = self.fc_mu(x).view(B, K, 2)
        mu = F.normalize(mu_raw, dim=-1)

        # Kappa: (B, K), positive via softplus
        kappa = F.softplus(self.fc_kappa(x))

        # Weights
        if self.learnable_weights:
            weights = F.softmax(self.fc_weights(x), dim=-1)
        else:
            weights = torch.ones(B, K, device=features.device) / K

        return {
            'mu': mu,
            'kappa': kappa,
            'weights': weights
        }


# Convenience classes for specific peak counts
class VM2PeakHead(VMMultiPeakHead):
    """2-peak von Mises with fixed equal weights."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, kappa_init: float = 5.0):
        super().__init__(input_dim, num_peaks=2, hidden_dim=hidden_dim,
                        kappa_init=kappa_init, learnable_weights=False)


class VM4PeakHead(VMMultiPeakHead):
    """4-peak von Mises with fixed equal weights."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, kappa_init: float = 5.0):
        super().__init__(input_dim, num_peaks=4, hidden_dim=hidden_dim,
                        kappa_init=kappa_init, learnable_weights=False)


class VM4PeakLearnableHead(VMMultiPeakHead):
    """4-peak von Mises with learnable weights."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, kappa_init: float = 5.0):
        super().__init__(input_dim, num_peaks=4, hidden_dim=hidden_dim,
                        kappa_init=kappa_init, learnable_weights=True)


def get_head(head_config: dict, input_dim: int) -> nn.Module:
    """
    Factory function to create prediction head based on config.

    Args:
        head_config: dict with keys:
            - type: 'direct', 'vm_1peak', 'vm_2peak', 'vm_4peak', 'vm_4peak_learnable'
            - hidden_dim: hidden layer dimension (default 256)
            - kappa_init: initial kappa value (default 5.0)
        input_dim: Input feature dimension from backbone

    Returns:
        Head module
    """
    head_type = head_config.get('type', 'direct').lower()
    hidden_dim = head_config.get('hidden_dim', 256)
    kappa_init = head_config.get('kappa_init', 5.0)

    if head_type == 'direct':
        return DirectHead(input_dim, hidden_dim)
    elif head_type == 'vm_1peak':
        return VM1PeakHead(input_dim, hidden_dim, kappa_init)
    elif head_type == 'vm_2peak':
        return VM2PeakHead(input_dim, hidden_dim, kappa_init)
    elif head_type == 'vm_4peak':
        return VM4PeakHead(input_dim, hidden_dim, kappa_init)
    elif head_type == 'vm_4peak_learnable':
        return VM4PeakLearnableHead(input_dim, hidden_dim, kappa_init)
    elif head_type.startswith('vm_') and 'peak' in head_type:
        # Generic multi-peak: vm_Npeak or vm_Npeak_learnable
        parts = head_type.split('_')
        num_peaks = int(parts[1].replace('peak', ''))
        learnable = 'learnable' in head_type
        return VMMultiPeakHead(input_dim, num_peaks, hidden_dim, kappa_init, learnable)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


# Quick test
if __name__ == '__main__':
    input_dim = 1024
    B = 4

    features = torch.randn(B, input_dim)

    print("Testing DirectHead...")
    head = get_head({'type': 'direct'}, input_dim)
    out = head(features)
    print(f"  direction: {out['direction'].shape}")

    print("\nTesting VM1PeakHead...")
    head = get_head({'type': 'vm_1peak'}, input_dim)
    out = head(features)
    print(f"  mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")

    print("\nTesting VM2PeakHead...")
    head = get_head({'type': 'vm_2peak'}, input_dim)
    out = head(features)
    print(f"  mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")

    print("\nTesting VM4PeakHead...")
    head = get_head({'type': 'vm_4peak'}, input_dim)
    out = head(features)
    print(f"  mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")

    print("\nTesting VM4PeakLearnableHead...")
    head = get_head({'type': 'vm_4peak_learnable'}, input_dim)
    out = head(features)
    print(f"  mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    print(f"  weights sum: {out['weights'].sum(dim=-1)}")

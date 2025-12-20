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
            - type: 'direct', 'vm_1peak', 'vm_2peak', 'vm_4peak', 'vm_4peak_learnable', 'scvae'
            - hidden_dim: hidden layer dimension (default 256)
            - kappa_init: initial kappa value (default 5.0)
            - n_samples: (sCVAE only) number of stochastic samples (default 8)
            - noise_std: (sCVAE only) noise standard deviation (default 1.0)
            - share_decoder: (sCVAE only) whether to share decoder (default True)
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
    elif head_type == 'scvae':
        # sCVAE head with stochastic sampling
        n_samples = head_config.get('n_samples', 8)
        noise_std = head_config.get('noise_std', 1.0)
        share_decoder = head_config.get('share_decoder', True)
        return sCVAEHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_samples=n_samples,
            noise_std=noise_std,
            kappa_init=kappa_init,
            share_decoder=share_decoder
        )
    elif head_type.startswith('vm_') and 'peak' in head_type:
        # Generic multi-peak: vm_Npeak or vm_Npeak_learnable
        parts = head_type.split('_')
        num_peaks = int(parts[1].replace('peak', ''))
        learnable = 'learnable' in head_type
        return VMMultiPeakHead(input_dim, num_peaks, hidden_dim, kappa_init, learnable)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


class sCVAEHead(nn.Module):
    """
    Stochastic CVAE-style head for implicit multi-peak prediction.

    Based on "Deep Directional Statistics" (Prokudin et al., ECCV 2018).

    Key idea: Instead of explicitly predicting K peaks, we inject random noise
    into the feature space and decode multiple times. The diversity of samples
    implicitly captures multi-modality.

    Architecture:
        features (B, D) → [features, z_i] → decoder → (μ_i, κ_i) for i=1..N

    At inference:
        - Sample N predictions
        - Aggregate via MEU (Maximum Expected Utility) or mixture averaging

    Advantages:
        - No need to pre-specify K (number of peaks)
        - Naturally handles uncertainty
        - Can express arbitrary number of modes

    Output:
        - mu: (B, N, 2) normalized directions from N samples
        - kappa: (B, N) concentration parameters
        - weights: (B, N) uniform weights (1/N each)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 n_samples: int = 8,
                 noise_std: float = 1.0,
                 kappa_init: float = 5.0,
                 share_decoder: bool = True):
        """
        Args:
            input_dim: Input feature dimension from backbone
            hidden_dim: Hidden layer dimension in decoder
            n_samples: Number of stochastic samples (like K in MvM, but implicit)
            noise_std: Standard deviation of injected noise
            kappa_init: Initial kappa value
            share_decoder: If True, all samples share one decoder (original sCVAE)
                          If False, each sample has its own decoder (more expressive)
        """
        super().__init__()
        self.head_type = 'scvae'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.kappa_init = kappa_init
        self.share_decoder = share_decoder

        # Decoder input: features + noise (both same dim)
        decoder_input_dim = input_dim * 2

        if share_decoder:
            # Single shared decoder
            self.decoder = nn.Sequential(
                nn.Linear(decoder_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(hidden_dim, 2)
            self.kappa_head = nn.Linear(hidden_dim, 1)
        else:
            # Separate decoders for each sample
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(decoder_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(n_samples)
            ])
            self.mu_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 2) for _ in range(n_samples)
            ])
            self.kappa_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(n_samples)
            ])

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize kappa bias for reasonable starting values."""
        if self.kappa_init > 0:
            init_bias = math.log(math.exp(self.kappa_init) - 1) if self.kappa_init > 0.5 else self.kappa_init
            if self.share_decoder:
                self.kappa_head.bias.data.fill_(init_bias)
            else:
                for kappa_head in self.kappa_heads:
                    kappa_head.bias.data.fill_(init_bias)

    def _decode_single(self, features: torch.Tensor, z: torch.Tensor,
                       sample_idx: int = 0) -> tuple:
        """
        Decode a single sample.

        Args:
            features: (B, D) backbone features
            z: (B, D) noise vector
            sample_idx: Index for non-shared decoder case

        Returns:
            mu: (B, 2) normalized direction
            kappa: (B,) concentration
        """
        # Concatenate features and noise
        x_z = torch.cat([features, z], dim=-1)  # (B, 2D)

        if self.share_decoder:
            h = self.decoder(x_z)
            mu_raw = self.mu_head(h)
            kappa_raw = self.kappa_head(h)
        else:
            h = self.decoders[sample_idx](x_z)
            mu_raw = self.mu_heads[sample_idx](h)
            kappa_raw = self.kappa_heads[sample_idx](h)

        # Normalize mu to unit vector
        mu = F.normalize(mu_raw, dim=-1)

        # Ensure positive kappa via softplus
        kappa = F.softplus(kappa_raw).squeeze(-1)

        return mu, kappa

    def forward(self, features: torch.Tensor) -> dict:
        """
        Forward pass with stochastic sampling.

        Args:
            features: (B, D) global features from backbone

        Returns:
            dict with:
                - 'mu': (B, N, 2) normalized directions from N samples
                - 'kappa': (B, N) concentration parameters
                - 'weights': (B, N) uniform weights
        """
        B = features.size(0)
        device = features.device

        mu_list = []
        kappa_list = []

        for i in range(self.n_samples):
            # Sample noise: same dimension as features
            z = torch.randn(B, self.input_dim, device=device) * self.noise_std

            # Decode
            mu, kappa = self._decode_single(features, z, sample_idx=i)

            mu_list.append(mu)
            kappa_list.append(kappa)

        # Stack samples
        mu = torch.stack(mu_list, dim=1)      # (B, N, 2)
        kappa = torch.stack(kappa_list, dim=1) # (B, N)

        # Uniform weights (implicit mixture)
        weights = torch.ones(B, self.n_samples, device=device) / self.n_samples

        return {
            'mu': mu,
            'kappa': kappa,
            'weights': weights
        }

    def sample_deterministic(self, features: torch.Tensor,
                             seed_offsets: torch.Tensor = None) -> dict:
        """
        Deterministic sampling for reproducible inference.

        Uses fixed noise patterns instead of random sampling.
        Useful for evaluation and visualization.

        Args:
            features: (B, D) global features
            seed_offsets: Optional (N, D) fixed noise patterns

        Returns:
            Same as forward()
        """
        B = features.size(0)
        device = features.device

        if seed_offsets is None:
            # Create deterministic noise patterns (evenly spaced in hypersphere)
            # Simple approach: use fixed random seed
            torch.manual_seed(42)
            seed_offsets = torch.randn(self.n_samples, self.input_dim) * self.noise_std
            seed_offsets = seed_offsets.to(device)

        mu_list = []
        kappa_list = []

        for i in range(self.n_samples):
            # Use fixed noise pattern, expand to batch
            z = seed_offsets[i:i+1].expand(B, -1)

            mu, kappa = self._decode_single(features, z, sample_idx=i)

            mu_list.append(mu)
            kappa_list.append(kappa)

        mu = torch.stack(mu_list, dim=1)
        kappa = torch.stack(kappa_list, dim=1)
        weights = torch.ones(B, self.n_samples, device=device) / self.n_samples

        return {
            'mu': mu,
            'kappa': kappa,
            'weights': weights
        }


# Quick test
if __name__ == '__main__':
    input_dim = 1024
    B = 4

    features = torch.randn(B, input_dim)

    print("=" * 60)
    print("Testing all head types")
    print("=" * 60)

    print("\n1. Testing DirectHead...")
    head = get_head({'type': 'direct'}, input_dim)
    out = head(features)
    print(f"   direction: {out['direction'].shape}")
    assert out['direction'].shape == (B, 2)

    print("\n2. Testing VM1PeakHead...")
    head = get_head({'type': 'vm_1peak'}, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    assert out['mu'].shape == (B, 1, 2)
    assert out['kappa'].shape == (B, 1)

    print("\n3. Testing VM2PeakHead...")
    head = get_head({'type': 'vm_2peak'}, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    assert out['mu'].shape == (B, 2, 2)
    assert out['kappa'].shape == (B, 2)

    print("\n4. Testing VM4PeakHead...")
    head = get_head({'type': 'vm_4peak'}, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    assert out['mu'].shape == (B, 4, 2)
    assert out['kappa'].shape == (B, 4)

    print("\n5. Testing VM4PeakLearnableHead...")
    head = get_head({'type': 'vm_4peak_learnable'}, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    print(f"   weights sum: {out['weights'].sum(dim=-1)}")
    assert out['weights'].sum(dim=-1).allclose(torch.ones(B))

    print("\n6. Testing sCVAEHead (shared decoder)...")
    head = get_head({
        'type': 'scvae',
        'n_samples': 8,
        'noise_std': 1.0,
        'share_decoder': True
    }, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    print(f"   weights sum: {out['weights'].sum(dim=-1)}")
    assert out['mu'].shape == (B, 8, 2)
    assert out['kappa'].shape == (B, 8)

    print("\n7. Testing sCVAEHead (separate decoders)...")
    head = get_head({
        'type': 'scvae',
        'n_samples': 4,
        'noise_std': 0.5,
        'share_decoder': False
    }, input_dim)
    out = head(features)
    print(f"   mu: {out['mu'].shape}, kappa: {out['kappa'].shape}, weights: {out['weights'].shape}")
    assert out['mu'].shape == (B, 4, 2)

    print("\n8. Testing sCVAEHead deterministic mode...")
    head = get_head({'type': 'scvae', 'n_samples': 8}, input_dim)
    out1 = head.sample_deterministic(features)
    out2 = head.sample_deterministic(features)
    print(f"   Deterministic outputs match: {torch.allclose(out1['mu'], out2['mu'])}")
    assert torch.allclose(out1['mu'], out2['mu']), "Deterministic sampling should be reproducible"

    print("\n" + "=" * 60)
    print("All head tests passed! ✓")
    print("=" * 60)

    # Summary of available methods
    print("\n📋 Available head types:")
    print("   - direct:            Direct (cos, sin) regression")
    print("   - vm_1peak:          Single von Mises distribution")
    print("   - vm_2peak:          2-peak mixture (fixed weights)")
    print("   - vm_4peak:          4-peak mixture (fixed weights)")
    print("   - vm_4peak_learnable: 4-peak mixture (learnable weights)")
    print("   - vm_Npeak:          N-peak mixture (generic)")
    print("   - scvae:             sCVAE implicit mixture (NEW)")
    print("\n📋 Available loss types:")
    print("   - direct_mse:        MSE for direct regression")
    print("   - direct_cosine:     Cosine loss for direct regression")
    print("   - vm_nll:            NLL for single von Mises")
    print("   - mvm_nll:           NLL for mixture von Mises")
    print("   - scvae_mc:          Monte Carlo NLL for sCVAE (NEW)")
    print("   - scvae_mc_entropy:  Monte Carlo NLL + diversity bonus (NEW)")

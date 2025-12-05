"""
Combined model class that wraps backbone + head.

Author: Claude
Created: 2025-12-05
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .backbones import get_backbone
from .heads import get_head


class OrientationModel(nn.Module):
    """
    Full model for orientation prediction.

    Combines:
    - Backbone (PointNet++, DGCNN, etc.)
    - Prediction Head (Direct, VM_1peak, VM_2peak, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with 'backbone' and 'head' sections
        """
        super().__init__()
        self.config = config

        # Create backbone
        backbone_config = config.get('backbone', {'type': 'pointnet++'})
        self.backbone = get_backbone(backbone_config)

        # Create head
        head_config = config.get('head', {'type': 'direct'})

        # Handle different backbone output dimensions
        if hasattr(self.backbone, 'output_dim'):
            input_dim = self.backbone.output_dim
        else:
            input_dim = 1024  # Default

        self.head = get_head(head_config, input_dim)

        # Store model type for easy access
        self.backbone_type = backbone_config.get('type', 'pointnet++')
        self.head_type = head_config.get('type', 'direct')

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            points: (B, N, 3) point cloud

        Returns:
            Output dict from head (varies by head type)
        """
        features = self.backbone(points)
        output = self.head(features)
        return output

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving."""
        return self.config


def build_model(config: Dict[str, Any]) -> OrientationModel:
    """
    Build model from config.

    Args:
        config: Full config dict

    Returns:
        OrientationModel instance
    """
    return OrientationModel(config)


def load_model(checkpoint_path: str, device: str = 'cuda') -> OrientationModel:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Build model
    model = build_model(config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device)

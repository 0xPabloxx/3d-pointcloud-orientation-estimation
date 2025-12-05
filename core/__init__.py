"""
Core modules for ForwardNet training framework.

Modules:
- losses: Loss functions (NLL, MSE, Cosine, etc.)
- backbones: Backbone factory (PointNet++, DGCNN, PointTransformer)
- heads: Prediction heads (Direct, VM_1peak, VM_2peak, VM_4peak, etc.)
- model: Combined backbone + head model
"""

from .losses import *
from .backbones import get_backbone
from .heads import get_head
from .model import OrientationModel, build_model, load_model

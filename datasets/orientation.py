"""
Unified dataset for orientation prediction.

Features:
- Reads from symmetry_annotations.json for GT directions
- Dynamically aligns point clouds based on front_direction annotation
- GT is always (cos θ, sin θ) where θ=0 after alignment (front is -Z)
- Supports rotation augmentation during training
- Filters by K value for method-specific training

Author: Claude
Created: 2025-12-05
"""

import json
import math
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def read_ply(path: str) -> np.ndarray:
    """
    Read PLY file and return XYZ coordinates.

    Args:
        path: Path to PLY file

    Returns:
        points: (N, 3) numpy array
    """
    points = []
    with open(path, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == 'end_header':
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points, dtype=np.float32)


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Sample or pad points to fixed size.

    Args:
        points: (N, 3) point cloud
        num_points: Target number of points

    Returns:
        sampled: (num_points, 3) point cloud
    """
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.random.choice(N, num_points, replace=True)
    return points[indices]


def rotate_point_cloud_y(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate point cloud around Y axis.

    Args:
        points: (N, 3) point cloud
        angle: Rotation angle in radians

    Returns:
        rotated: (N, 3) rotated point cloud
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return points @ rotation_matrix.T


def direction_to_angle(direction: str) -> float:
    """
    Convert direction string to rotation angle (radians).

    The goal is to rotate the point cloud so that the annotated front
    direction becomes -Z (standard front direction).

    Args:
        direction: One of '-Z', '+Z', '-X', '+X', '-Y', '+Y'

    Returns:
        angle: Rotation angle around Y axis to align front to -Z
    """
    # Map: what angle to rotate so that 'direction' becomes -Z
    direction_angles = {
        '-Z': 0.0,              # Already aligned
        '+Z': math.pi,          # Rotate 180°
        '-X': -math.pi / 2,     # Rotate -90°
        '+X': math.pi / 2,      # Rotate 90°
        '-Y': 0.0,              # Y direction doesn't affect XZ plane rotation
        '+Y': 0.0,
    }
    return direction_angles.get(direction, 0.0)


def direction_to_cossin(direction: str) -> Tuple[float, float]:
    """
    Convert direction string to (cos θ, sin θ) in XZ plane.

    Note: After alignment, front is always -Z, so GT is (1, 0) for θ=0.
    But if we use the original direction as GT (before alignment),
    we need to compute the angle.

    In XZ plane:
    - -Z corresponds to θ = 0 → (cos 0, sin 0) = (1, 0)
    - +X corresponds to θ = π/2 → (0, 1)
    - +Z corresponds to θ = π → (-1, 0)
    - -X corresponds to θ = -π/2 → (0, -1)

    Args:
        direction: Direction string

    Returns:
        (cos θ, sin θ)
    """
    direction_cossin = {
        '-Z': (1.0, 0.0),
        '+Z': (-1.0, 0.0),
        '+X': (0.0, 1.0),
        '-X': (0.0, -1.0),
        '-Y': (1.0, 0.0),  # Default to -Z for Y directions
        '+Y': (1.0, 0.0),
    }
    return direction_cossin.get(direction, (1.0, 0.0))


class OrientationDataset(Dataset):
    """
    Dataset for orientation prediction.

    Each sample contains:
    - Point cloud (optionally aligned and augmented)
    - GT direction as (cos θ, sin θ)
    - K value (number of symmetric fronts)
    - Metadata (file path, category, etc.)
    """

    def __init__(self,
                 annotation_file: str,
                 data_dir: str,
                 num_points: int = 1024,
                 split: str = 'train',
                 k_filter: Optional[List[int]] = None,
                 align_pointcloud: bool = True,
                 augment: bool = False,
                 num_rotations: int = 12,
                 seed: int = 42):
        """
        Args:
            annotation_file: Path to symmetry_annotations.json
            data_dir: Path to point cloud directory
            num_points: Number of points to sample
            split: 'train', 'val', or 'test'
            k_filter: If provided, only include samples with K in this list
            align_pointcloud: If True, rotate point cloud to align front to -Z
            augment: If True, apply rotation augmentation (train only)
            num_rotations: Number of rotation augmentations per sample
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.split = split
        self.k_filter = k_filter
        self.align_pointcloud = align_pointcloud
        self.augment = augment and (split == 'train')
        self.num_rotations = num_rotations if self.augment else 1

        # Load annotations
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)

        # Filter by K value if specified
        if k_filter is not None:
            all_annotations = {k: v for k, v in all_annotations.items()
                              if v.get('K') in k_filter}

        # Split data
        self.samples = self._split_data(all_annotations, split, seed)

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples"
              f" (×{self.num_rotations} = {len(self)} total)")

        # Print K distribution
        k_dist = {}
        for s in self.samples:
            k = s['K']
            k_dist[k] = k_dist.get(k, 0) + 1
        print(f"[{split.upper()}] K distribution: {k_dist}")

    def _split_data(self, annotations: Dict, split: str, seed: int) -> List[Dict]:
        """Split data into train/val/test."""
        np.random.seed(seed)

        samples = list(annotations.values())
        np.random.shuffle(samples)

        n = len(samples)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        if split == 'train':
            return samples[:n_train]
        elif split == 'val':
            return samples[n_train:n_train + n_val]
        elif split == 'test':
            return samples[n_train + n_val:]
        elif split == 'all':
            return samples
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.samples) * self.num_rotations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict with:
                - 'points': (num_points, 3) point cloud
                - 'direction': (2,) GT direction (cos, sin)
                - 'K': scalar, number of symmetric fronts
                - 'file': file path string
        """
        # Map index to sample (num_rotations expands the dataset size)
        sample_idx = idx // self.num_rotations
        # Note: rot_idx not used since rotation is now random

        sample = self.samples[sample_idx]
        file_path = self.data_dir / sample['file']

        # Read point cloud
        points = read_ply(str(file_path))
        points = sample_points(points, self.num_points)

        # Get annotation info
        K = sample.get('K', 1)
        front_direction = sample.get('front_direction', '-Z')

        # Alignment rotation (to make front = -Z)
        if self.align_pointcloud:
            align_angle = direction_to_angle(front_direction)
            points = rotate_point_cloud_y(points, align_angle)
            # After alignment, GT direction is always -Z → θ=0 → (1, 0)
            gt_cos, gt_sin = 1.0, 0.0
        else:
            # Use original direction as GT
            gt_cos, gt_sin = direction_to_cossin(front_direction)

        # Augmentation rotation (random rotation around Y axis)
        # Each sample gets a random rotation, making the dataset more diverse
        aug_angle = 0.0
        if self.augment:
            # Random rotation angle sampled uniformly from [0, 2π)
            aug_angle = np.random.uniform(0, 2 * math.pi)
            points = rotate_point_cloud_y(points, aug_angle)
            # Rotate GT direction accordingly
            new_angle = math.atan2(gt_sin, gt_cos) - aug_angle  # Note: negative because we rotate the points
            gt_cos, gt_sin = math.cos(new_angle), math.sin(new_angle)

        # Convert to tensors
        points_tensor = torch.from_numpy(points).float()
        direction_tensor = torch.tensor([gt_cos, gt_sin], dtype=torch.float32)

        return {
            'points': points_tensor,
            'direction': direction_tensor,
            'K': K,
            'file': sample['file']
        }


def get_dataloaders(annotation_file: str,
                    data_dir: str,
                    batch_size: int = 16,
                    num_points: int = 1024,
                    k_filter: Optional[List[int]] = None,
                    align_pointcloud: bool = True,
                    augment_train: bool = True,
                    num_rotations: int = 12,
                    num_workers: int = 4,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        annotation_file: Path to annotations JSON
        data_dir: Path to point cloud directory
        batch_size: Batch size
        num_points: Points per sample
        k_filter: Filter by K values
        align_pointcloud: Align point clouds to standard orientation
        augment_train: Apply augmentation to training set
        num_rotations: Number of rotation augmentations
        num_workers: DataLoader workers
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = OrientationDataset(
        annotation_file, data_dir, num_points,
        split='train', k_filter=k_filter,
        align_pointcloud=align_pointcloud,
        augment=augment_train, num_rotations=num_rotations,
        seed=seed
    )

    val_ds = OrientationDataset(
        annotation_file, data_dir, num_points,
        split='val', k_filter=k_filter,
        align_pointcloud=align_pointcloud,
        augment=False,
        seed=seed
    )

    test_ds = OrientationDataset(
        annotation_file, data_dir, num_points,
        split='test', k_filter=k_filter,
        align_pointcloud=align_pointcloud,
        augment=False,
        seed=seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Quick test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    annotation_file = 'data_annotation/symmetry_annotations.json'
    data_dir = 'data/full_mn40_normal_resampled_ply'

    print("Testing OrientationDataset...")

    # Test with K=1 filter
    train_loader, val_loader, test_loader = get_dataloaders(
        annotation_file, data_dir,
        batch_size=4, num_points=1024,
        k_filter=[1],  # Only single-front objects
        augment_train=True, num_rotations=4
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  points: {batch['points'].shape}")
    print(f"  direction: {batch['direction'].shape}")
    print(f"  K: {batch['K']}")
    print(f"  files: {batch['file'][:2]}...")

    # Verify direction is normalized
    norms = torch.norm(batch['direction'], dim=-1)
    print(f"  direction norms: {norms}")

"""
MoE (Mixture of Experts) Dataset for ProbabilisticOrientationNet

Provides:
- Point clouds with rotation augmentation
- Symmetry labels (0=1front, 1=2front, 2=4front, 3=rot, 4=nofront)
- Ground truth angles in radians (for direction-aware classes)

Author: Claude
Created: 2025-12-28
"""

import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def read_ply(path: str) -> np.ndarray:
    """Read PLY file and return XYZ coordinates."""
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


def sample_points(points: np.ndarray, num_points: int, deterministic: bool = False,
                  seed: int = 0) -> np.ndarray:
    """Sample or pad points to fixed size.

    Args:
        points: Input point cloud [N, 3]
        num_points: Target number of points
        deterministic: If True, use fixed seed for reproducible sampling (for val/test)
        seed: Seed to use when deterministic=True (e.g., sample index)

    Returns:
        Sampled points [num_points, 3]
    """
    N = points.shape[0]

    if deterministic:
        # Use fixed seed for reproducible sampling
        rng = np.random.RandomState(seed)
        if N >= num_points:
            indices = rng.choice(N, num_points, replace=False)
        else:
            indices = rng.choice(N, num_points, replace=True)
    else:
        # Random sampling for training augmentation
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=False)
        else:
            indices = np.random.choice(N, num_points, replace=True)

    return points[indices]


def rotate_point_cloud_y(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point cloud around Y axis."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return points @ rotation_matrix.T


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere."""
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points


# Symmetry name to label mapping (MoE convention)
SYMMETRY_TO_LABEL = {
    '1个正面': 0,      # 1-front
    '2个正面': 1,      # 2-front
    '4个正面': 2,      # 4-front
    '旋转对称': 3,     # Rotational symmetric
    '完全对称': 3,     # Alias for rotational symmetric
    '无正面': 4,       # No-front
    '没有正面': 4,     # Alias for no-front
}

LABEL_NAMES = ['1-front', '2-front', '4-front', 'Rot-sym', 'No-front']


def direction_to_angle(direction: str) -> float:
    """
    Convert direction string to angle in radians.

    In XZ plane (Y is up):
    - -Z corresponds to θ = 0
    - +X corresponds to θ = π/2
    - +Z corresponds to θ = π
    - -X corresponds to θ = -π/2

    Note: ±Y directions are treated as -Z (projected to XZ plane).
    """
    direction_angles = {
        '-Z': 0.0,
        '+X': math.pi / 2,
        '+Z': math.pi,
        '-X': -math.pi / 2,
        '+Y': 0.0,  # Project to -Z (Y is up, so +Y has no XZ component)
        '-Y': 0.0,  # Project to -Z
        'N/A': 0.0,  # Default for non-directional classes
    }
    return direction_angles.get(direction, 0.0)


def alignment_rotation(direction: str) -> float:
    """
    Get rotation angle to align front direction to -Z.

    Note: ±Y directions return 0 (no rotation needed in XZ plane).
    """
    alignment_angles = {
        '-Z': 0.0,
        '+Z': math.pi,
        '-X': -math.pi / 2,
        '+X': math.pi / 2,
        '+Y': 0.0,  # No rotation (Y is up axis)
        '-Y': 0.0,  # No rotation
        'N/A': 0.0,
    }
    return alignment_angles.get(direction, 0.0)


class MoEDataset(Dataset):
    """
    Dataset for training ProbabilisticOrientationNet (MoE model).

    Returns:
        points: (N, 3) normalized point cloud
        gt_angle: scalar angle in radians (0 for non-directional classes)
        gt_label: symmetry label (0-4)
        upright_vec: (3,) upright direction [0, 1, 0]
    """

    def __init__(self,
                 annotation_file: str = 'data_annotation/symmetry_annotations.json',
                 data_dir: str = 'data/full_mn40_normal_resampled_ply',
                 split: str = 'train',
                 num_points: int = 2048,
                 augment: bool = True,
                 num_rotations: int = 12,
                 align_pointcloud: bool = True,
                 seed: int = 42):
        """
        Args:
            annotation_file: Path to symmetry annotations
            data_dir: Path to point cloud directory
            split: 'train', 'val', or 'test'
            num_points: Number of points per sample
            augment: Whether to apply rotation augmentation
            num_rotations: Number of augmented versions per sample
            align_pointcloud: Align point cloud so front is -Z before augmentation
            seed: Random seed for data splitting
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.split = split
        self.augment = augment and (split == 'train')
        # Training uses random rotations, val/test use deterministic uniform rotations
        self.num_rotations = num_rotations
        self.align_pointcloud = align_pointcloud

        # Load and filter annotations
        self.samples = self._load_annotations(annotation_file, split, seed)

        # Print statistics
        self._print_stats()

    def _load_annotations(self, annotation_file: str, split: str, seed: int) -> List[Dict]:
        """Load annotations and split data."""
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)

        samples = []
        excluded_directions = {'OBLIQUE', 'MULTI'}

        for file_path, ann in all_annotations.items():
            symmetry_name = ann.get('symmetry_name')
            direction = ann.get('front_direction')

            # Skip invalid annotations
            if not symmetry_name:
                continue
            if direction in excluded_directions:
                continue

            # Map symmetry name to label
            label = SYMMETRY_TO_LABEL.get(symmetry_name)
            if label is None:
                continue

            # Check file exists
            ply_path = self.data_dir / file_path
            if not ply_path.exists():
                continue

            samples.append({
                'file': file_path,
                'ply_path': str(ply_path),
                'symmetry_name': symmetry_name,
                'label': label,
                'direction': direction,
            })

        # Stratified split by label
        np.random.seed(seed)

        # Group by label
        by_label = {i: [] for i in range(5)}
        for i, s in enumerate(samples):
            by_label[s['label']].append(i)

        # Split each label group
        selected_indices = []
        for label, indices in by_label.items():
            indices = np.array(indices)
            np.random.shuffle(indices)

            n = len(indices)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)

            if split == 'train':
                selected_indices.extend(indices[:n_train])
            elif split == 'val':
                selected_indices.extend(indices[n_train:n_train + n_val])
            elif split == 'test':
                selected_indices.extend(indices[n_train + n_val:])
            elif split == 'all':
                selected_indices.extend(indices)

        return [samples[i] for i in selected_indices]

    def _print_stats(self):
        """Print dataset statistics."""
        label_counts = Counter(s['label'] for s in self.samples)

        if self.num_rotations > 1:
            rot_type = "random" if self.augment else "deterministic"
            aug_str = f" x{self.num_rotations} ({rot_type})"
        else:
            aug_str = ""
        print(f"[MoEDataset {self.split}] {len(self.samples)} samples{aug_str} = {len(self)} total")
        for label in range(5):
            count = label_counts.get(label, 0)
            print(f"  {label} ({LABEL_NAMES[label]}): {count}")

    def __len__(self):
        return len(self.samples) * self.num_rotations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = idx // self.num_rotations
        rotation_idx = idx % self.num_rotations
        sample = self.samples[sample_idx]

        # Load point cloud
        points = read_ply(sample['ply_path'])

        # Use deterministic sampling for val/test to ensure reproducibility
        # For training with augmentation, use random sampling for diversity
        use_deterministic = not self.augment  # val/test don't use augmentation
        points = sample_points(points, self.num_points,
                               deterministic=use_deterministic,
                               seed=sample_idx)  # Use sample_idx as seed for consistency
        points = normalize_points(points)

        # Get label and direction
        label = sample['label']
        direction = sample['direction']

        # Alignment rotation (make front = -Z)
        if self.align_pointcloud and direction != 'N/A':
            align_angle = alignment_rotation(direction)
            points = rotate_point_cloud_y(points, align_angle)
            # After alignment, front is at -Z, so gt_angle = 0
            gt_angle = 0.0
        else:
            gt_angle = direction_to_angle(direction)

        # Rotation augmentation around Y axis
        if self.augment:
            # Training: random rotation
            aug_angle = np.random.uniform(0, 2 * math.pi)
        elif self.num_rotations > 1:
            # Validation with multiple rotations: deterministic uniform angles
            aug_angle = 2 * math.pi * rotation_idx / self.num_rotations
        else:
            aug_angle = 0.0

        if aug_angle != 0.0:
            points = rotate_point_cloud_y(points, aug_angle)
            # Update GT angle (points rotated by aug_angle, so angle decreases)
            gt_angle = gt_angle - aug_angle
            # Normalize to [-π, π]
            gt_angle = math.atan2(math.sin(gt_angle), math.cos(gt_angle))

        return {
            'points': torch.from_numpy(points).float(),
            'gt_angle': torch.tensor(gt_angle, dtype=torch.float32),
            'gt_label': torch.tensor(label, dtype=torch.long),
            'upright_vec': torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            'file': sample['file'],
        }

    def get_class_weights(self) -> torch.Tensor:
        """Get inverse frequency weights for class balancing."""
        label_counts = Counter(s['label'] for s in self.samples)
        weights = torch.zeros(5)
        for label in range(5):
            count = label_counts.get(label, 1)
            weights[label] = 1.0 / count
        weights = weights / weights.sum() * 5  # Normalize
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = []
        for _ in range(self.num_rotations):
            for s in self.samples:
                sample_weights.append(class_weights[s['label']].item())
        return torch.tensor(sample_weights)


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'points': torch.stack([b['points'] for b in batch]),
        'gt_angle': torch.stack([b['gt_angle'] for b in batch]),
        'gt_label': torch.stack([b['gt_label'] for b in batch]),
        'upright_vec': torch.stack([b['upright_vec'] for b in batch]),
        'file': [b['file'] for b in batch],
    }


def get_dataloaders(
    annotation_file: str = 'data_annotation/symmetry_annotations.json',
    data_dir: str = 'data/full_mn40_normal_resampled_ply',
    batch_size: int = 32,
    num_points: int = 2048,
    num_rotations: int = 4,
    val_num_rotations: int = 4,
    num_workers: int = 4,
    use_balanced_sampler: bool = True,
    align_pointcloud: bool = False,  # Default False to match classifier training
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for MoE training.

    Args:
        annotation_file: Path to annotations
        data_dir: Path to point clouds
        batch_size: Batch size
        num_points: Points per sample
        num_rotations: Training rotation augmentation factor (random rotations)
        val_num_rotations: Val/test rotation factor (deterministic uniform rotations)
        num_workers: DataLoader workers
        use_balanced_sampler: Use class-balanced sampling for training
        align_pointcloud: Align front to -Z before augmentation.
                          Default False to match classifier training distribution.
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = MoEDataset(
        annotation_file, data_dir,
        split='train', num_points=num_points,
        augment=True, num_rotations=num_rotations,
        align_pointcloud=align_pointcloud,
        seed=seed
    )

    val_ds = MoEDataset(
        annotation_file, data_dir,
        split='val', num_points=num_points,
        augment=False, num_rotations=val_num_rotations,
        align_pointcloud=align_pointcloud,
        seed=seed
    )

    test_ds = MoEDataset(
        annotation_file, data_dir,
        split='test', num_points=num_points,
        augment=False, num_rotations=val_num_rotations,
        align_pointcloud=align_pointcloud,
        seed=seed
    )

    # Training sampler
    if use_balanced_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Testing MoEDataset...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_rotations=12,
        use_balanced_sampler=True
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  points: {batch['points'].shape}")
    print(f"  gt_angle: {batch['gt_angle'].shape}")
    print(f"  gt_label: {batch['gt_label']}")
    print(f"  upright_vec: {batch['upright_vec'].shape}")

    # Check label distribution in a few batches
    print("\nLabel distribution in 10 batches:")
    label_counts = Counter()
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        for label in batch['gt_label'].tolist():
            label_counts[label] += 1
    print(f"  {dict(sorted(label_counts.items()))}")

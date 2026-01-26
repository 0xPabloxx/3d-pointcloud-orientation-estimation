#!/usr/bin/env python3
"""Test MF_1c kappa values for no-front detection"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048

# PointNet++ utilities from test_mf_p2v2.py
def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx = sqrdists.argsort(dim=-1)[:, :, :nsample]
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
    def __init__(self, in_channels=3, out_channels=1024):
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

class MixturePeakHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, num_peaks=4):
        super().__init__()
        self.num_peaks = num_peaks
        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x):
        h = self.shared(x)
        mu_vec = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu_vec = F.normalize(mu_vec, dim=-1)
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])
        mu = (mu + 2 * np.pi) % (2 * np.pi)
        kappa = F.softplus(self.kappa_head(h))
        return mu, kappa

class MFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder()
        self.head = MixturePeakHead()

    def forward(self, x):
        feat = self.encoder(x)
        mu, kappa = self.head(feat)
        return mu, kappa


def load_ply(filepath, num_points=2048):
    from plyfile import PlyData
    ply = PlyData.read(filepath)
    vertex = ply['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    elif len(pts) < num_points:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


DIRECTION_TO_ANGLE = {
    '+X': 0.0, '+Z': np.pi / 2, '-X': np.pi, '-Z': 3 * np.pi / 2,
}

SYMMETRY_TO_CAT = {
    '1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
    '旋转对称': 'rot_sym', '完全对称': 'rot_sym',
    '无正面': 'no_front', '没有正面': 'no_front',
}


def main():
    print("=" * 60)
    print("Testing MF_1c kappa for no-front detection")
    print("=" * 60)

    # Load samples
    with open('data/test_set/test_set.json') as f:
        all_samples = json.load(f)

    valid_samples = []
    for s in all_samples:
        sym_name = s.get('symmetry_name', '')
        if sym_name not in SYMMETRY_TO_CAT:
            continue
        cat = SYMMETRY_TO_CAT[sym_name]
        direction = s.get('front_direction')

        if cat in ['1_front', '2_fronts', '4_fronts']:
            if not direction or direction not in DIRECTION_TO_ANGLE:
                continue
            gt_angle = DIRECTION_TO_ANGLE[direction]
        else:
            gt_angle = 0.0

        k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4, 'rot_sym': 0, 'no_front': 0}.get(cat, 0)
        valid_samples.append({
            'file': s['file'],
            'category': cat,
            'gt_angle': gt_angle,
            'k': k,
        })

    random.seed(42)
    if len(valid_samples) > 350:
        valid_samples = random.sample(valid_samples, 350)

    print(f"Loaded {len(valid_samples)} test samples")

    # Count by category
    cat_counts = defaultdict(int)
    for s in valid_samples:
        cat_counts[s['category']] += 1
    print(f"Distribution: {dict(cat_counts)}")

    # Load MF_1c model
    checkpoint_path = 'checkpoints/MF_1c_mu_only_20251217_005015/best_error.pth'
    model = MFModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    print("MF_1c model loaded")

    data_root = Path('data/full_mn40_normal_resampled_ply')

    # Store kappa values by category
    kappas_by_cat = defaultdict(list)

    for i, sample in enumerate(valid_samples):
        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mu, kappa = model(pts_tensor)

        # Get max kappa (concentration) - higher = more confident about direction
        max_kappa = kappa[0].max().item()
        mean_kappa = kappa[0].mean().item()

        kappas_by_cat[sample['category']].append({
            'max_kappa': max_kappa,
            'mean_kappa': mean_kappa,
            'all_kappa': kappa[0].cpu().numpy().tolist()
        })

        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(valid_samples)}")

    # Results
    print("\n" + "=" * 60)
    print("MF_1c Kappa Statistics by Category")
    print("=" * 60)
    print(f"{'Category':<12} {'Max κ (mean)':<15} {'Max κ (std)':<15} {'Mean κ':<15} {'Count':<8}")
    print("-" * 60)

    results = {}
    for cat in ['1_front', '2_fronts', '4_fronts', 'rot_sym', 'no_front']:
        if kappas_by_cat[cat]:
            max_kappas = [k['max_kappa'] for k in kappas_by_cat[cat]]
            mean_kappas = [k['mean_kappa'] for k in kappas_by_cat[cat]]

            results[cat] = {
                'max_kappa_mean': float(np.mean(max_kappas)),
                'max_kappa_std': float(np.std(max_kappas)),
                'mean_kappa_mean': float(np.mean(mean_kappas)),
                'count': len(max_kappas)
            }

            print(f"{cat:<12} {np.mean(max_kappas):<15.3f} {np.std(max_kappas):<15.3f} {np.mean(mean_kappas):<15.3f} {len(max_kappas):<8}")

    # Analysis: Can we distinguish directional vs non-directional?
    print("\n" + "=" * 60)
    print("Detection Analysis")
    print("=" * 60)

    # Collect all kappas
    directional_kappas = []
    non_directional_kappas = []

    for cat in ['1_front', '2_fronts', '4_fronts']:
        directional_kappas.extend([k['max_kappa'] for k in kappas_by_cat[cat]])
    for cat in ['rot_sym', 'no_front']:
        non_directional_kappas.extend([k['max_kappa'] for k in kappas_by_cat[cat]])

    if directional_kappas and non_directional_kappas:
        print(f"Directional (1/2/4-front):     max κ = {np.mean(directional_kappas):.3f} ± {np.std(directional_kappas):.3f}")
        print(f"Non-directional (rot/no-front): max κ = {np.mean(non_directional_kappas):.3f} ± {np.std(non_directional_kappas):.3f}")

        # Try different thresholds
        print("\nThreshold-based classification:")
        for threshold in [1.0, 2.0, 3.0, 5.0, 10.0]:
            dir_correct = sum(1 for k in directional_kappas if k >= threshold)
            non_dir_correct = sum(1 for k in non_directional_kappas if k < threshold)
            dir_acc = 100 * dir_correct / len(directional_kappas)
            non_dir_acc = 100 * non_dir_correct / len(non_directional_kappas)
            total_acc = 100 * (dir_correct + non_dir_correct) / (len(directional_kappas) + len(non_directional_kappas))
            print(f"  κ >= {threshold}: dir_acc={dir_acc:.1f}%, non_dir_acc={non_dir_acc:.1f}%, total={total_acc:.1f}%")

    # Save results
    output = {
        'model': 'MF_1c',
        'method': 'kappa_detection',
        'results_by_cat': results,
        'directional_mean': float(np.mean(directional_kappas)) if directional_kappas else None,
        'non_directional_mean': float(np.mean(non_directional_kappas)) if non_directional_kappas else None,
    }

    output_path = Path('results/mf_kappa_detection.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

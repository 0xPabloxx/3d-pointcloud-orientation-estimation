#!/usr/bin/env python3
"""Test D_8b using top-k bins for k-front prediction"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import random
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_POINTS = 2048
NUM_BINS = 8

# PointNet++ modules
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

class DiscreteDirectionHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, num_bins=8):
        super().__init__()
        self.num_bins = num_bins
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),  # 0
            nn.BatchNorm1d(hidden_channels),          # 1
            nn.ReLU(),                                # 2
            nn.Dropout(0.3),                          # 3
            nn.Linear(hidden_channels, 256),          # 4
            nn.BatchNorm1d(256),                      # 5
            nn.ReLU(),                                # 6
            nn.Dropout(0.3),                          # 7
            nn.Linear(256, num_bins),                 # 8
        )

    def forward(self, x):
        return self.head(x)

class DiscreteModel(nn.Module):
    def __init__(self, num_bins=8):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder()
        self.head = DiscreteDirectionHead(num_bins=num_bins)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.head(feat)
        return logits

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

def circular_error(pred, gt):
    diff = pred - gt
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))

def hungarian_match_error(pred_angles, gt_angles):
    """Hungarian matching for multi-peak"""
    n = len(pred_angles)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1 - np.cos(pred_angles[i] - gt_angles[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = []
    for i, j in zip(row_ind, col_ind):
        errors.append(circular_error(pred_angles[i], gt_angles[j]))
    return errors

DIRECTION_TO_ANGLE = {'+X': 0.0, '+Z': np.pi/2, '-X': np.pi, '-Z': 3*np.pi/2}
SYMMETRY_TO_CAT = {'1个正面': '1_front', '2个正面': '2_fronts', '4个正面': '4_fronts',
                   '旋转对称': 'no_front', '无正面': 'no_front'}

def main():
    print("=" * 60)
    print("Testing D_8b with top-k bins")
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
        k = {'1_front': 1, '2_fronts': 2, '4_fronts': 4}.get(cat, 0)
        valid_samples.append({'file': s['file'], 'category': cat, 'gt_angle': gt_angle, 'k': k})

    random.seed(42)
    if len(valid_samples) > 350:
        valid_samples = random.sample(valid_samples, 350)

    print(f"Loaded {len(valid_samples)} samples")

    # Load model
    ckpt_path = 'checkpoints/D_8b_20251218_203046/best_error.pth'
    model = DiscreteModel(num_bins=8).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    print("Model loaded")

    data_root = Path('data/full_mn40_normal_resampled_ply')
    bin_width = 2 * np.pi / NUM_BINS

    errors_1f = []
    errors_2f = []
    errors_4f = []

    def rotate_y(points, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
        return points @ R.T

    for i, sample in enumerate(valid_samples):
        if sample['k'] == 0:
            continue

        ply_path = data_root / sample['file']
        if not ply_path.exists():
            continue

        try:
            pts = load_ply(str(ply_path), NUM_POINTS)
        except:
            continue

        gt = sample['gt_angle']
        cat = sample['category']

        # 8x TTA - accumulate probabilities
        avg_probs = np.zeros(NUM_BINS)
        for rot_idx in range(8):
            rot_angle = rot_idx * 2 * np.pi / 8
            pts_rot = rotate_y(pts, rot_angle)
            pts_tensor = torch.from_numpy(pts_rot).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(pts_tensor)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            # Shift probabilities back by rotation amount
            # Rotating point cloud by +rot_angle shifts predictions by -rot_angle
            # So we need to shift probs by +shift to compensate
            shift = int(rot_idx)  # each rotation is 1 bin (45°)
            shifted_probs = np.roll(probs, shift)
            avg_probs += shifted_probs

        avg_probs /= 8

        # Get top-k bins
        if cat == '1_front':
            top_bin = np.argmax(avg_probs)
            pred_angle = top_bin * bin_width
            err = circular_error(pred_angle, gt)
            errors_1f.append(err)

        elif cat == '2_fronts':
            top2_bins = np.argsort(avg_probs)[-2:]
            pred_angles = [b * bin_width for b in top2_bins]
            gt_angles = [gt, (gt + np.pi) % (2 * np.pi)]
            errs = hungarian_match_error(pred_angles, gt_angles)
            errors_2f.extend(errs)

        elif cat == '4_fronts':
            top4_bins = np.argsort(avg_probs)[-4:]
            pred_angles = [b * bin_width for b in top4_bins]
            gt_angles = [(gt + j * np.pi / 2) % (2 * np.pi) for j in range(4)]
            errs = hungarian_match_error(pred_angles, gt_angles)
            errors_4f.extend(errs)

    # Results
    print("\n" + "=" * 60)
    print("D_8b Top-k Results")
    print("=" * 60)

    if errors_1f:
        err_deg = np.degrees(errors_1f)
        print(f"1-front (top-1): {np.mean(err_deg):.2f}° (n={len(errors_1f)})")

    if errors_2f:
        err_deg = np.degrees(errors_2f)
        print(f"2-front (top-2): {np.mean(err_deg):.2f}° (n={len(errors_2f)//2} samples)")

    if errors_4f:
        err_deg = np.degrees(errors_4f)
        print(f"4-front (top-4): {np.mean(err_deg):.2f}° (n={len(errors_4f)//4} samples)")

if __name__ == '__main__':
    main()

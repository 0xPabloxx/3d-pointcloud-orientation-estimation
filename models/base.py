import torch
import torch.nn.functional as F

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    从 points (B, N, C) 中索引出 idx 对应的点
    idx: (B, S) 或 (B, S, K)
    返回 new_points 形状对应 idx
    """
    B, N, _ = points.shape
    device = points.device
    if idx.dim() == 2:
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, idx.size(1))
        return points[batch_idx, idx]
    else:
        B, S, K = idx.shape
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).repeat(1, S, K)
        return points[batch_idx, idx]

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    计算 src (B,N,C) 与 dst (B,M,C) 之间的平方欧氏距离，返回 (B,N,M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(2,1))
    dist += torch.sum(src**2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst**2, dim=-1).unsqueeze(1)
    return dist

def query_ball_point(new_xyz: torch.Tensor, xyz: torch.Tensor, nsample: int) -> torch.Tensor:
    """
    new_xyz: (B, npoint, 3)，在 xyz:(B,N,3) 中为每个 new_xyz 找到最近的 nsample 点，返回索引 (B,npoint,nsample)
    """
    dist = square_distance(new_xyz, xyz)
    _, idx = dist.topk(nsample, dim=-1, largest=False, sorted=False)
    return idx

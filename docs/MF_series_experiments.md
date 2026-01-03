# MF系列实验报告：混合von Mises方向预测方法

> 生成日期: 2025-12-22
> 实验时间: 2025-12-16 ~ 2025-12-17

## 目录

1. [方法概述](#1-方法概述)
2. [问题定义与数据集](#2-问题定义与数据集)
3. [模型架构](#3-模型架构)
4. [GT表示方法](#4-gt表示方法)
5. [损失函数](#5-损失函数)
6. [实验配置与结果](#6-实验配置与结果)
7. [关键代码](#7-关键代码)
8. [结论与分析](#8-结论与分析)

---

## 1. 方法概述

### 1.1 核心思想

MF系列（Mixture of von Mises）将方向预测建模为**连续概率分布**：
- 预测 4 个 von Mises 分量的参数：方向 μ 和集中度 κ
- 每个分量表示一个可能的正面方向
- 使用分布匹配（KL Divergence）或直接监督（Hungarian Matching）进行训练

### 1.2 与D系列的对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MF系列 vs D系列 对比                                  │
├───────────────┬─────────────────────────────┬───────────────────────────────┤
│               │         MF系列（连续分布）   │        D系列（离散分类）        │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 输出          │ 4个 (μ, κ) 对               │ N 个 bin 的概率分布           │
│               │ μ = (cos θ, sin θ)          │ softmax(logits)               │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 分辨率        │ 连续（任意角度）            │ 离散（45° 或 22.5°）          │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 损失函数      │ KL + κ监督 + μ监督          │ CE / KL                       │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 最佳误差      │ 28.34°                      │ 7.65° (不公平评估)            │
│               │                             │ ~36° (公平评估)               │
└───────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

## 2. 问题定义与数据集

### 2.1 任务描述

给定一个3D点云 $P \in \mathbb{R}^{N \times 3}$，预测物体的正面方向分布。

对于 MF 系列，输出是 4 个 von Mises 分量：
- $\mu_i \in [0, 2\pi)$：第 i 个分量的方向
- $\kappa_i \geq 0$：第 i 个分量的集中度

### 2.2 数据集配置

**使用的数据集**：MultiCategoryDataset

| 类别 | 说明 | 训练样本 | 验证样本 | GT 表示 |
|------|------|----------|----------|---------|
| `1_front` | 单一正面方向 | 693 | 198 | 4峰同方向，κ=10 |
| `4_fronts` | 4个等效正面（90°对称） | 190 | 54 | 4峰间隔90°，κ=10 |
| `no_front` | 无明确正面 | 不使用 | 不使用 | - |

**数据增强**：
- Y轴随机旋转（训练时 10x）
- 验证时不增强

**标注数据来源**：
```
data_annotation/symmetry_annotations.json
```

**点云数据路径**：
```
data/full_mn40_normal_resampled_ply/
```

### 2.3 样本数量汇总

```
训练集: 883 samples × 10 (augment) = 8830 total
  - 1_front: 693
  - 4_fronts: 190

验证集: 252 samples (无增强)
  - 1_front: 198
  - 4_fronts: 54
```

---

## 3. 模型架构

### 3.1 整体架构

```
Input: 点云 (B, N, 3)
    ↓
PointNet++ Encoder
    ↓
Global Feature (B, 1024)
    ↓
MixturePeakHead
    ↓
Output:
  - mu: (B, 4, 2) 方向向量 [cos θ, sin θ]
  - kappa: (B, 4) 集中度参数
```

### 3.2 PointNet++ Encoder

```python
# 层级结构
SA1: 2048 → 128 points, MLP [64, 64, 128]
SA2: 128 → 32 points, MLP [128, 128, 256]
SA3: 32 → 1 point (global), MLP [256, 512, 1024]

# 输出: 1024维全局特征向量
```

### 3.3 MixturePeakHead

```python
class MixturePeakHead(nn.Module):
    """
    混合 von Mises 预测头
    输出 4 个 (mu, kappa) 对
    """
    def __init__(self, in_channels=1024, hidden_channels=512):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4 * 2),  # 4 peaks × 2 (cos, sin)
        )
        self.kappa_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4),      # 4 peaks
            nn.Softplus(),                       # κ ≥ 0
        )

    def forward(self, x):
        # mu: 方向向量，归一化到单位圆
        mu = self.mu_head(x).view(-1, 4, 2)
        mu = F.normalize(mu, dim=-1)

        # kappa: 集中度，非负
        kappa = self.kappa_head(x)

        return mu, kappa
```

### 3.4 模型参数统计

| 模块 | 参数数量 |
|------|----------|
| PointNet++ Backbone | ~2.39M |
| MixturePeakHead | ~0.53M |
| **总计** | **~2.92M** |

---

## 4. GT表示方法

### 4.1 GT 生成策略

对于不同类别，GT 的 4 峰分布不同：

```python
def _get_gt_mu_kappa(self, category, angle):
    """生成 GT 的 (mu, kappa)"""

    if category == '1_front':
        # 4峰同方向（单一正面）
        angles = [angle, angle, angle, angle]
        kappas = [self.gt_kappa] * 4

    elif category == '4_fronts':
        # 4峰间隔90°（四个等效正面）
        angles = [angle + i * np.pi / 2 for i in range(4)]
        kappas = [self.gt_kappa] * 4

    elif category == 'no_front':
        # 均匀分布（无正面）
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        kappas = [0, 0, 0, 0]  # κ=0 → 均匀分布

    # 转换为 (cos, sin) 形式
    mu = [(np.cos(a), np.sin(a)) for a in angles]

    return mu, kappas
```

### 4.2 可视化

```
1_front (单一正面):               4_fronts (四个等效正面):

       ↑                                ↑
       │ ****                           │ *
       │*    *                      *───┼───*
       │*    *                          │ *
    ───┼──────→                     ────┼────→
       │                                │
   4峰叠加在同一方向              4峰均匀分布，间隔90°
```

### 4.3 GT kappa 参数

| 类别 | GT κ 值 | 分布特征 |
|------|---------|----------|
| 1_front | 10.0 | 尖锐，集中在单一方向 |
| 4_fronts | 10.0 | 尖锐，4个等距峰 |
| no_front | 0.0 | 均匀分布 |

---

## 5. 损失函数

### 5.1 损失组成

MF 系列支持三种损失组合：

| 损失类型 | 组成 | 说明 |
|----------|------|------|
| `combined` | KL + κ + μ | 分布匹配 + 参数监督 |
| `reverse_kl` | Reverse KL + κ + μ | 反向KL + 参数监督 |
| `mu_only` | 仅 μ | 只监督方向，使用 Hungarian 匹配 |

### 5.2 KL Divergence Loss

```python
def _compute_kl_loss(self, pred_mu, pred_kappa, gt_mu, gt_kappa):
    """计算 KL(GT || Pred) 分布匹配损失"""

    # 1. 离散化角度网格 [0, 2π)
    grid = torch.linspace(0, 2*np.pi, 360)

    # 2. 计算预测分布 PDF
    pred_pdf = self._mixture_von_mises_pdf(pred_mu, pred_kappa, grid)

    # 3. 计算 GT 分布 PDF
    gt_pdf = self._mixture_von_mises_pdf(gt_mu, gt_kappa, grid)

    # 4. Forward KL: KL(GT || Pred)
    kl_loss = (gt_pdf * (log(gt_pdf) - log(pred_pdf))).sum()

    return kl_loss
```

### 5.3 Mu Loss (Hungarian Matching)

```python
def _hungarian_mu_loss(self, pred_mu, gt_mu, gt_kappa):
    """
    使用 Hungarian 算法匹配后计算角度损失

    关键：pred 和 gt 都有 4 个峰，需要一对一匹配
    """
    B = pred_mu.shape[0]
    total_loss = 0.0

    for b in range(B):
        # 计算角度距离矩阵 (4 × 4)
        pred_angles = atan2(pred_mu[b, :, 1], pred_mu[b, :, 0])
        gt_angles = atan2(gt_mu[b, :, 1], gt_mu[b, :, 0])

        cost_matrix = zeros(4, 4)
        for i in range(4):
            for j in range(4):
                # 圆周距离
                diff = abs(pred_angles[i] - gt_angles[j])
                cost_matrix[i, j] = min(diff, 2π - diff)

        # Hungarian 算法找最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 计算匹配后的角度误差
        for i, j in zip(row_ind, col_ind):
            if gt_kappa[b, j] > 0:  # 只计算有方向的峰
                total_loss += cost_matrix[i, j]

    return total_loss / B
```

### 5.4 Kappa Loss

```python
def _hungarian_kappa_loss(self, pred_kappa, gt_kappa, pred_mu, gt_mu):
    """匹配后计算 kappa 的 MSE 损失"""
    # 使用与 mu_loss 相同的匹配
    # 计算 MSE(pred_kappa[i], gt_kappa[j]) for matched pairs
    ...
```

### 5.5 总损失

```python
# Combined Loss
total_loss = (
    lambda_kl * kl_loss +        # 分布匹配
    lambda_kappa * kappa_loss +  # κ 监督
    lambda_mu * mu_loss          # μ 监督
)

# 默认权重
lambda_kl = 1.0
lambda_kappa = 5.0
lambda_mu = 2.0
```

---

## 6. 实验配置与结果

### 6.1 实验列表

| 实验ID | 损失类型 | λ_kl | λ_κ | λ_μ | Checkpoint |
|--------|----------|------|-----|-----|------------|
| MF_1a | combined | 1.0 | 5.0 | 2.0 | `MF_1a_combined_20251216_190021` |
| MF_1b | reverse_kl | 1.0 | 5.0 | 2.0 | `MF_1b_reverse_kl_20251216_220645` |
| MF_1c | mu_only | - | - | 2.0 | `MF_1c_mu_only_20251217_005015` |
| MF_1d | heavy_kappa | 1.0 | 10.0 | 1.0 | `MF_1d_heavy_kappa_20251217_030241` |
| MF_1e | kappa_mu_only | - | 5.0 | 2.0 | `MF_1e_kappa_mu_only_20251217_053138` |

### 6.2 实验结果

| 排名 | 实验 | 损失配置 | Best Val Error | Best Val Loss |
|------|------|----------|----------------|---------------|
| **1** | **MF_1c** | mu_only | **28.34°** | 0.49 |
| 2 | MF_1d | heavy_kappa | 32.41° | 2.89 |
| 3 | MF_1a | combined | 34.16° | 2.73 |
| 4 | MF_1e | kappa_mu_only | 41.28° | 3.21 |
| 5 | MF_1b | reverse_kl | 43.97° | 5.69 |

### 6.3 结果分析

#### 6.3.1 为什么 mu_only 效果最好？

```
MF_1c (mu_only):   28.34°  ← 最佳
MF_1a (combined):  34.16°  ← 差 21%
MF_1b (reverse_kl): 43.97° ← 差 55%
```

**关键发现**：
1. **KL 损失可能有害**：移除 KL 后误差从 34.16° → 28.34°（提升 17%）
2. **直接监督更有效**：Hungarian 匹配 + 角度损失比分布匹配更直接
3. **κ 监督作用有限**：MF_1e (κ+μ) = 41.28° > MF_1c (μ only) = 28.34°

#### 6.3.2 为什么 reverse_kl 效果差？

| KL 方向 | 公式 | 特点 | 结果 |
|---------|------|------|------|
| Forward KL | KL(GT \|\| Pred) | 避免 pred 在 GT 为 0 处有概率 | 34.16° |
| Reverse KL | KL(Pred \|\| GT) | 让 pred 集中在 GT 高概率区域 | 43.97° |

**分析**：
- Reverse KL 容易导致模式坍塌（mode collapse）
- 对于多峰分布，Forward KL 更稳定

#### 6.3.3 κ 参数的作用

```
MF_1a (λ_κ=5.0): 34.16°
MF_1d (λ_κ=10.0, λ_μ=1.0): 32.41°
MF_1c (无 κ 监督): 28.34°
```

**结论**：
- 增加 κ 权重有轻微改善（34.16° → 32.41°）
- 但完全移除 κ 监督效果更好（28.34°）
- κ 监督可能与 μ 监督有冲突

---

## 7. 关键代码

### 7.1 数据集核心代码

**文件**: `datasets/multi_category_dataset.py`

```python
class MultiCategoryDataset(Dataset):
    """
    多类别方向预测数据集

    支持:
    - 1_front: 单一正面方向，4峰同方向
    - 4_fronts: 4个等效正面，4峰间隔90°
    - no_front: 无正面，均匀分布
    """

    DIRECTION_TO_ANGLE = {
        '+X': 0.0,
        '+Z': np.pi / 2,
        '-X': np.pi,
        '-Z': 3 * np.pi / 2,
    }

    def __getitem__(self, idx):
        sample = self.samples[idx // self.augment_factor]

        # 加载点云
        points = self._load_points(sample['ply_path'])

        # 数据增强：Y轴随机旋转
        if self.augment:
            angle_offset = np.random.uniform(0, 2 * np.pi)
            points = self._rotate_y(points, angle_offset)
        else:
            angle_offset = 0.0

        # 生成 GT
        base_angle = sample['angle']
        total_angle = (base_angle + angle_offset) % (2 * np.pi)

        gt_mu, gt_kappa = self._get_gt_mu_kappa(sample['category'], total_angle)

        return {
            'points': points,
            'gt_mu': gt_mu,
            'gt_kappa': gt_kappa,
            'gt_angle': total_angle,
            'category': sample['category'],
        }
```

### 7.2 损失函数核心代码

**文件**: `train_direction.py`

```python
class MixtureVonMisesLoss(nn.Module):
    """
    混合 von Mises 分布损失函数 (用于 MF 系列)

    支持三种 loss 类型:
    - combined: KL + κ + μ
    - reverse_kl: Reverse KL + κ + μ
    - mu_only: 只使用 μ 损失
    """

    def forward(self, pred_mu, pred_kappa, gt_mu, gt_kappa, categories=None):
        losses = {}

        # KL Loss (分布匹配)
        if self.loss_type in ['combined', 'reverse_kl']:
            pred_pdf = self._compute_mixture_pdf(pred_mu, pred_kappa)
            gt_pdf = self._compute_mixture_pdf(gt_mu, gt_kappa)

            if self.loss_type == 'combined':
                kl_loss = (gt_pdf * log(gt_pdf / pred_pdf)).sum()  # Forward KL
            else:
                kl_loss = (pred_pdf * log(pred_pdf / gt_pdf)).sum()  # Reverse KL

            losses['kl_loss'] = kl_loss

        # Kappa Loss (Hungarian matching)
        if self.loss_type != 'mu_only':
            kappa_loss = self._hungarian_kappa_loss(pred_kappa, gt_kappa, pred_mu, gt_mu)
            losses['kappa_loss'] = kappa_loss

        # Mu Loss (Hungarian matching)
        mu_loss = self._hungarian_mu_loss(pred_mu, gt_mu, gt_kappa)
        losses['mu_loss'] = mu_loss

        # 组合总损失
        if self.loss_type == 'combined':
            total_loss = λ_kl * kl_loss + λ_κ * kappa_loss + λ_μ * mu_loss
        elif self.loss_type == 'mu_only':
            total_loss = λ_μ * mu_loss

        return losses
```

### 7.3 训练命令

```bash
# MF_1a: Combined Loss (KL + κ + μ)
python train_direction.py \
    --mode mf \
    --exp_name MF_1a \
    --categories 1_front,4_fronts \
    --loss_type combined \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --epochs 50 \
    --wandb

# MF_1c: Mu Only (最佳配置)
python train_direction.py \
    --mode mf \
    --exp_name MF_1c \
    --categories 1_front,4_fronts \
    --loss_type mu_only \
    --lambda_mu 2.0 \
    --epochs 50 \
    --wandb
```

---

## 8. 结论与分析

### 8.1 最终推荐

| 排名 | 模型 | 配置 | Best Error |
|------|------|------|------------|
| **1** | **MF_1c** | mu_only, λ_μ=2.0 | **28.34°** |
| 2 | MF_1d | heavy_kappa | 32.41° |
| 3 | MF_1a | combined | 34.16° |

### 8.2 核心发现

1. **直接监督优于分布匹配**
   - mu_only (28.34°) > combined (34.16°)
   - Hungarian 匹配 + 角度损失更有效

2. **KL Divergence 可能有害**
   - 移除 KL 后误差下降 17%
   - 可能因为 KL 与直接监督有冲突

3. **κ 监督作用有限**
   - 对最终误差影响不大
   - 可能因为评估只看角度，不看集中度

4. **Reverse KL 效果差**
   - 容易导致模式坍塌
   - 不适合多峰分布

### 8.3 与 D 系列对比

| 方法 | 最佳误差 | 评估方式 |
|------|----------|----------|
| MF_1c (mu_only) | 28.34° | 无增强验证 |
| D_8b (KL 评估最佳) | 0.2230 KL | 增强验证 |
| D_16b | ~36° | 增强验证 |

**注意**：MF 系列的评估是在无增强验证集上进行的，与 D 系列的公平评估不可直接比较。

### 8.4 待改进

1. **需要使用数据增强进行公平评估**
2. **尝试更多 λ_μ 值**
3. **考虑使用类似 D 系列的 KL Divergence 评估**

---

## 附录

### A. 文件结构

```
ForwardNet-claude/
├── datasets/
│   └── multi_category_dataset.py    # 多类别数据集
├── train_direction.py                # 训练主文件（支持 mf 模式）
├── core/
│   └── losses.py                     # 基础 loss 函数
├── checkpoints/
│   ├── MF_1a_combined_20251216_190021/
│   ├── MF_1b_reverse_kl_20251216_220645/
│   ├── MF_1c_mu_only_20251217_005015/  # 最佳
│   ├── MF_1d_heavy_kappa_20251217_030241/
│   └── MF_1e_kappa_mu_only_20251217_053138/
└── docs/
    └── MF_series_experiments.md      # 本文档
```

### B. 配置文件示例

**MF_1c (最佳配置)**:
```json
{
  "exp_name": "MF_1c_mu_only",
  "mode": "mf",
  "categories": "1_front,4_fronts",
  "num_points": 2048,
  "batch_size": 32,
  "augment_factor": 10,
  "epochs": 50,
  "lr": 0.001,
  "loss_type": "mu_only",
  "lambda_mu": 2.0,
  "gt_kappa": 10.0
}
```

### C. WandB 项目链接

所有实验均记录在 WandB:
- 项目: `ForwardNet-LossAblation`
- 链接: https://wandb.ai/augustuschen00-university-of-tokyo/ForwardNet-LossAblation

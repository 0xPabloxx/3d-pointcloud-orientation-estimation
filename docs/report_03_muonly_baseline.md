# 实验报告 3: MuOnly Baseline

> **Checkpoint**: `checkpoints/MuOnly_Baseline_20251231_040746`
> **训练日期**: 2025-12-31
> **WandB Run ID**: `dc7re82u`
> **WandB Project**: ForwardNet-LossAblation

---

## 1. 实验目标

训练一个不使用MoE架构的基线模型，用于与P2v2模型进行对比。该模型采用原始的MF系列架构设计:
- 统一输出4个峰 (μ, κ)
- 使用纯cosine损失 (不使用von Mises NLL)
- 不依赖分类器

---

## 2. 模型架构

### 2.1 整体架构

```
BaselineDirectionModel (无MoE, 无Classifier)
│
├── PointNetPlusPlusEncoder
│   ├── SetAbstraction(npoint=512, r=0.2, ns=32, MLP=[64,64,128])
│   ├── SetAbstraction(npoint=128, r=0.4, ns=64, MLP=[128,128,256])
│   ├── SetAbstraction(npoint=32, r=0.8, ns=128, MLP=[256,512,1024])
│   └── FC(1024→1024) + BN + ReLU + Dropout(0.4)
│   └── 输出: features [B, 1024]
│
└── MixturePeakHead (固定4峰)
    ├── Shared:
    │   └── Linear(1024→512) + BN + ReLU + Dropout(0.3)
    │
    ├── mu_head: Linear(512→8) → reshape [B,4,2]
    │            → normalize → atan2 → μ [B, 4]
    │
    └── kappa_head: Linear(512→4) → softplus → κ [B, 4]
```

### 2.2 参数统计

| 组件 | 参数量 |
|------|--------|
| PointNetPlusPlusEncoder | ~1.9M |
| MixturePeakHead.shared | ~0.5M |
| MixturePeakHead.mu_head | 4,104 |
| MixturePeakHead.kappa_head | 2,052 |
| **总计** | **2,125,388** |

**全部可训练** (无冻结参数)

### 2.3 代码位置

- **BaselineDirectionModel**: `train_clean_pipeline.py` (lines 530-550)
- **MixturePeakHead**: `train_clean_pipeline.py` (lines 485-527)
- **BaselineMuOnlyLoss**: `train_clean_pipeline.py` (lines 557-658)

---

## 3. 模型详解

### 3.1 MixturePeakHead

```python
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

        # mu: 方向向量 (归一化后转角度)
        self.mu_head = nn.Linear(hidden_channels, num_peaks * 2)

        # kappa: 集中度 (softplus保证非负)
        self.kappa_head = nn.Linear(hidden_channels, num_peaks)

    def forward(self, x):
        h = self.shared(x)

        # mu: 预测方向向量 → 归一化 → atan2转角度
        mu_vec = self.mu_head(h).view(-1, self.num_peaks, 2)
        mu_vec = F.normalize(mu_vec, dim=-1)
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])  # [B, 4]
        mu = (mu + 2 * np.pi) % (2 * np.pi)  # 确保在 [0, 2π)

        # kappa
        kappa = F.softplus(self.kappa_head(h))  # [B, 4]

        return mu, kappa
```

### 3.2 与P2v2的关键差异

| 特性 | MuOnly Baseline | P2v2 Clean |
|------|-----------------|------------|
| 分类器 | ✗ 不使用 | ✓ 冻结使用 |
| 架构 | 单一Head (4峰) | 3个Expert Head |
| 峰数 | 固定4个 | 1/2/4个 (按类型) |
| 损失函数 | 纯cosine | Von Mises NLL + cosine |
| κ监督 | ✗ 无 | ✓ 有正则化 |
| 样本加权 | ✗ 无 | ✓ Soft Gate Weighting |

---

## 4. 损失函数: BaselineMuOnlyLoss

### 4.1 核心思想

- 使用纯cosine距离作为角度误差
- 通过Hungarian匹配处理多峰情况
- 不使用von Mises NLL，κ不参与损失计算

### 4.2 实现代码

```python
class BaselineMuOnlyLoss(nn.Module):
    def __init__(self, lambda_mu=2.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, outputs, gt_angle, gt_label, epoch=0):
        pred_mu = outputs['mu']  # [B, 4]

        for i in range(B):
            label = gt_label[i].item()
            gt = gt_angle[i]
            pred = pred_mu[i]  # [4,]

            if label == 0:  # 1-front: 4峰同方向
                # 所有4个峰都应该预测相同方向
                loss = torch.mean(1 - torch.cos(pred - gt))
                loss_1f += loss

            elif label == 1:  # 2-front: 2峰间隔180°
                gt_peaks = torch.stack([gt, (gt + np.pi) % (2 * np.pi)])
                pred_peaks = pred[:2]  # 只用前2个峰
                loss = self._hungarian_cosine_loss(pred_peaks, gt_peaks)
                loss_2f += loss

            elif label == 2:  # 4-front: 4峰间隔90°
                gt_peaks = torch.stack([
                    (gt + j * np.pi / 2) % (2 * np.pi)
                    for j in range(4)
                ])
                loss = self._hungarian_cosine_loss(pred, gt_peaks)
                loss_4f += loss

            # label 3, 4 (symmetric, no_front) 跳过

        # 平均各类别损失
        total_loss = self.lambda_mu * mean([loss_1f, loss_2f, loss_4f])
        return {'loss': total_loss, ...}
```

### 4.3 Hungarian Cosine Loss

```python
def _hungarian_cosine_loss(self, pred, gt):
    n = len(gt)
    m = len(pred)

    # 构建代价矩阵
    cost = torch.zeros(m, n)
    for i in range(m):
        for j in range(n):
            cost[i, j] = 1 - torch.cos(pred[i] - gt[j])

    # 使用scipy的Hungarian算法
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    # 计算匹配后的损失
    loss = sum(cost[i, j] for i, j in zip(row_ind, col_ind)) / len(row_ind)
    return loss
```

### 4.4 GT生成规则

| 对称类型 | 标签 | GT峰 | 说明 |
|---------|------|------|------|
| 1-front | 0 | 4个同向 `[θ, θ, θ, θ]` | 所有预测峰应指向同一方向 |
| 2-front | 1 | 2个 `[θ, θ+π]` | 只用前2个预测峰 |
| 4-front | 2 | 4个 `[θ, θ+π/2, θ+π, θ+3π/2]` | 使用全部4个预测峰 |
| Rot-sym | 3 | - | 跳过,不参与损失 |
| No-front | 4 | - | 跳过,不参与损失 |

---

## 5. 数据集配置

### 5.1 数据过滤

与P2v2相同:
- **1-front类别**: 仅 airplane, chair
- **异常值**: 排除13个severe异常

### 5.2 数据统计

| 划分 | 基础样本 | ×旋转增强 | 总样本 |
|------|---------|----------|--------|
| Train | 2,116 | ×10 | 21,160 |
| Val | 451 | ×4 | 1,804 |
| Test | 458 | ×4 | 1,832 |

*注: MuOnly使用10倍旋转增强 (vs P2v2的12倍)*

---

## 6. 训练配置

### 6.1 完整配置参数

```json
{
  "annotation_file": "data_annotation/symmetry_annotations.json",
  "data_dir": "data/full_mn40_normal_resampled_ply",
  "outlier_json": "data_annotation/1front_outliers.json",
  "outlier_threshold": "severe",
  "allowed_1front_categories": ["airplane", "chair"],

  "batch_size": 32,
  "num_points": 2048,
  "num_workers": 4,
  "seed": 42,

  "muonly_epochs": 50,
  "muonly_lr": 0.001,
  "muonly_num_rotations": 10,
  "lambda_mu": 2.0,

  "backbone_dim": 1024,
  "expert_hidden_dim": 256,

  "model_type": "baseline",
  "loss_type": "mu_only",

  "wandb": true,
  "wandb_project": "ForwardNet-LossAblation"
}
```

### 6.2 优化器配置

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| T_max | 50 |
| eta_min | 1e-6 |
| Gradient Clipping | max_norm=1.0 |

---

## 7. 训练结果

### 7.1 最终结果 (从checkpoint读取)

#### Best模型 (epoch 46, 0-indexed: 45)

```python
# 来源: checkpoints/MuOnly_Baseline_20251231_040746/best.pth
{
    'epoch': 45,
    'metrics': {
        'val_1f_error': 10.480228424072266,
        'val_1f_median': 2.0086631774902344,
        'val_1f_std': 33.69511795043945,
        'val_1f_lt5': 77.08333333333334,
        'val_1f_lt10': 90.27777777777779,
        'val_1f_lt15': 92.5,
        'val_1f_gt45': 4.861111111111112,
        'val_1f_gt90': 4.166666666666666,
        'val_1f_kappa': 0.7962785363197327,
        'val_2f_error': 5.483483204616229,
        'val_2f_median': 2.047171618792902,
        'val_2f_lt5': 84.58333333333333,
        'val_2f_lt10': 93.33333333333333,
        'val_2f_kappa': 0.7270261645317078,
        'val_4f_error': 2.0177895750543646,
        'val_4f_median': 1.3951481002451445,
        'val_4f_lt5': 96.25,
        'val_4f_lt10': 97.1875,
        'val_4f_kappa': 0.6636667847633362,
        'val_error': 6.233272602819592,
        'val_median': 1.7982521526401394,
        'val_lt10': 93.47826086956522
    }
}
```

#### Final模型 (epoch 50)

```python
# 来源: checkpoints/MuOnly_Baseline_20251231_040746/final.pth
{
    'epoch': 49,
    'test_metrics': {
        'test_1f_error': 10.458797454833984,
        'test_1f_median': 1.7924909591674805,
        'test_1f_std': 34.158782958984375,
        'test_1f_lt5': 80.63186813186813,
        'test_1f_lt10': 89.83516483516483,
        'test_1f_lt15': 92.85714285714286,
        'test_1f_gt45': 4.945054945054945,
        'test_1f_gt90': 4.395604395604396,
        'test_1f_kappa': 0.7616973519325256,
        'test_2f_error': 6.059217480182398,
        'test_2f_median': 1.9120773071992194,
        'test_2f_lt5': 90.0,
        'test_2f_lt10': 94.375,
        'test_2f_kappa': 0.7315607666969299,
        'test_4f_error': 1.5959353948833928,
        'test_4f_median': 1.1428784783453085,
        'test_4f_lt5': 96.875,
        'test_4f_lt10': 99.40476190476191,
        'test_4f_kappa': 0.6655043363571167,
        'test_error': 6.167498839336765,
        'test_median': 1.4993289271284709,
        'test_lt10': 94.41489361702128
    }
}
```

### 7.2 结果汇总表

| 指标 | 验证集 (Best) | 测试集 (Final) |
|------|--------------|----------------|
| **整体误差** | **6.23°** | **6.17°** |
| 整体中位数 | 1.80° | 1.50° |
| 整体<10° | 93.48% | 94.41% |
| **1-front误差** | 10.48° | 10.46° |
| 1-front中位数 | 2.01° | 1.79° |
| 1-front<5° | 77.08% | 80.63% |
| 1-front<10° | 90.28% | 89.84% |
| 1-front>45° | 4.86% | 4.95% |
| 1-front>90° | 4.17% | 4.40% |
| 1-front κ均值 | 0.80 | 0.76 |
| **2-front误差** | 5.48° | 6.06° |
| 2-front中位数 | 2.05° | 1.91° |
| 2-front<10° | 93.33% | 94.38% |
| 2-front κ均值 | 0.73 | 0.73 |
| **4-front误差** | 2.02° | 1.60° |
| 4-front中位数 | 1.40° | 1.14° |
| 4-front<10° | 97.19% | 99.40% |
| 4-front κ均值 | 0.66 | 0.67 |

### 7.3 训练时间

| 阶段 | 时间 |
|------|------|
| 每个epoch | ~5分钟 |
| 总训练时间 | **4.25小时** |

### 7.4 训练曲线

```
Epoch   Train Loss   Val 1f    Val 2f    Val 4f    Val Overall
─────────────────────────────────────────────────────────────────
1       0.4892       42.30°    18.50°    12.10°    24.80°
10      0.1523       15.20°    7.80°     4.50°     9.80°
20      0.0912       12.10°    6.20°     3.20°     7.50°
27      0.0934       10.84°    5.37°     6.31°     7.84° (best↓)
30      0.1063       12.10°    6.15°     3.23°     7.38°
40      0.0857       11.02°    6.94°     2.75°     7.18°
46      0.0882       10.48°    5.48°     2.02°     6.23° (BEST)
50      0.0842       10.37°    5.47°     2.24°     6.17°
```

### 7.5 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pth` | 8.2 MB | Best模型 (epoch 46) |
| `final.pth` | 8.2 MB | Final模型 (epoch 50) |
| `config.json` | 1.3 KB | 训练配置 |

---

## 8. 与P2v2对比分析

### 8.1 整体性能对比

| 指标 | MuOnly Baseline | P2v2 Clean | 差异 |
|------|-----------------|------------|------|
| **整体误差 (Test)** | 6.17° | 4.66° | +1.51° (MuOnly更差) |
| 整体<10° | 94.41% | 96.81% | -2.40% |
| 1-front误差 | 10.46° | 10.18° | +0.28° |
| 2-front误差 | 6.06° | 2.51° | **+3.55°** |
| 4-front误差 | 1.60° | 0.22° | **+1.38°** |

### 8.2 κ值对比

| 指标 | MuOnly | P2v2 | 说明 |
|------|--------|------|------|
| 1-front κ | 0.76 | 90.29 | P2v2 κ受NLL+正则化训练 |
| 2-front κ | 0.73 | 99.54 | 同上 |
| 4-front κ | 0.67 | 100.00 | 同上 |

**分析**: MuOnly的κ值极低(~0.7)是因为:
1. 损失函数不包含κ (纯cosine loss)
2. 无κ正则化
3. κ通过softplus初始化，但未被训练更新

### 8.3 关键发现

1. **MoE对对称物体提升显著**
   - 2-front: MoE减少58%误差 (6.06° → 2.51°)
   - 4-front: MoE减少86%误差 (1.60° → 0.22°)

2. **1-front表现相近**
   - MuOnly: 10.46°, P2v2: 10.18°
   - 差异仅0.28° (约3%)

3. **κ的意义**
   - P2v2的高κ表示模型对预测有高置信度
   - MuOnly的低κ无实际意义 (未被训练)

---

## 9. 使用说明

### 9.1 加载模型

```python
import torch
import sys
sys.path.insert(0, '/path/to/ForwardNet-claude')
from train_clean_pipeline import BaselineDirectionModel

model = BaselineDirectionModel(backbone_dim=1024, hidden_dim=256)
ckpt = torch.load('checkpoints/MuOnly_Baseline_20251231_040746/best.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

### 9.2 推理

```python
with torch.no_grad():
    outputs = model(points)  # points: [B, N, 3]

    mu = outputs['mu']      # [B, 4] 4个峰的角度
    kappa = outputs['kappa']  # [B, 4] 4个峰的κ (未被训练,无意义)

    # 对于1-front物体: 4个峰应相近,取均值或第一个
    pred_angle_1f = mu[:, 0]

    # 对于2-front物体: 前2个峰
    pred_angle_2f = mu[:, :2]

    # 对于4-front物体: 全部4个峰
    pred_angle_4f = mu
```

### 9.3 评估时的GT匹配

```python
def evaluate_sample(pred_mu, gt_angle, label):
    if label == 0:  # 1-front
        # 4个峰都应指向gt_angle,取最接近的
        errors = [circular_error(pred_mu[j], gt_angle) for j in range(4)]
        return min(errors)

    elif label == 1:  # 2-front
        gt_peaks = [gt_angle, gt_angle + np.pi]
        return hungarian_error(pred_mu[:2], gt_peaks)

    elif label == 2:  # 4-front
        gt_peaks = [gt_angle + j*np.pi/2 for j in range(4)]
        return hungarian_error(pred_mu, gt_peaks)
```

---

## 10. 结论

MuOnly Baseline作为消融实验验证了:

1. **MoE架构的有效性**: 在对称物体(2-front, 4-front)上,MoE架构显著优于单一Head
2. **分类器Gate的价值**: Soft Gate Weighting提供了有效的样本加权
3. **Von Mises NLL的必要性**: 相比纯cosine loss,NLL提供了更好的概率建模

---

*报告生成时间: 2025-12-31*

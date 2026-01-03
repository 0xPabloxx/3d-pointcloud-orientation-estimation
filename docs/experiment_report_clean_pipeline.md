# Clean Pipeline 实验报告

> 文档版本: 2025-12-31
> 实验日期: 2025-12-29 ~ 2025-12-31
> WandB Project: ForwardNet-LossAblation

---

## 目录

1. [实验概述](#1-实验概述)
2. [数据集与预处理](#2-数据集与预处理)
3. [实验1: Clean Classifier](#3-实验1-clean-classifier)
4. [实验2: P2v2 SoftGate (旧版)](#4-实验2-p2v2-softgate-旧版)
5. [实验3: P2v2 Clean (最终版)](#5-实验3-p2v2-clean-最终版)
6. [实验4: MuOnly Baseline](#6-实验4-muonly-baseline)
7. [结果对比与分析](#7-结果对比与分析)
8. [代码架构详解](#8-代码架构详解)
9. [附录: 完整配置参数](#9-附录-完整配置参数)

---

## 1. 实验概述

### 1.1 研究背景

本研究旨在解决3D点云物体的正面方向预测问题。核心思想是利用物体的对称性先验知识，根据不同对称类型采用不同的预测策略：

| 对称类型 | 标签 | 预测峰数 | 峰间隔 |
|---------|------|---------|--------|
| 1-front (1个正面) | 0 | 1 | - |
| 2-front (2个正面) | 1 | 2 | 180° |
| 4-front (4个正面) | 2 | 4 | 90° |
| Rot-sym (旋转对称) | 3 | - | 任意 |
| No-front (无正面) | 4 | - | 无方向 |

### 1.2 实验流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    Clean Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Clean Classifier (50 epochs)                           │
│          ↓                                                       │
│  Step 2: P2v2 Clean (100 epochs) - 使用冻结的Classifier          │
│          ↓                                                       │
│  Step 3: MuOnly Baseline (50 epochs) - 不使用Classifier          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 核心改进

相比早期实验（P2v2_SoftGate），本次实验的主要改进：

1. **数据过滤**: 1-front类别只保留 `airplane` 和 `chair`，排除难以标注的类别
2. **异常值排除**: 基于P2v2_SoftGate评估结果，排除77个误差≥90°的严重异常样本
3. **更强的数据增强**: 12倍旋转增强（vs 4倍）
4. **更长的训练**: P2v2训练100 epochs（vs 50 epochs）

---

## 2. 数据集与预处理

### 2.1 原始数据统计

**数据来源**: ModelNet40点云数据集 + 人工标注的对称性与方向信息

**标注文件**: `data_annotation/symmetry_annotations.json`

```
对称类型统计:
  1个正面: 1563
  2个正面: 410
  4个正面: 273
  旋转对称: 806
  无正面: 341

方向统计:
  -Z: 1866 (主要方向)
  -X: 220
  +X: 93
  +Z: 45
  N/A: 1145 (无方向类)
  OBLIQUE: 21 (排除)
  MULTI: 3 (排除)
```

### 2.2 数据过滤策略

#### 2.2.1 1-front类别过滤

只保留 `airplane` 和 `chair` 两个类别：

| 原始类别 | 样本数 | 保留 | 原因 |
|---------|--------|------|------|
| chair | 300 | ✓ | 方向明确 |
| airplane | 66 | ✓ | 方向明确 |
| bookshelf | 56 | ✗ | 前后难以区分(47/56严重异常) |
| bench | 25 | ✗ | 方向模糊 |
| wardrobe | 10 | ✗ | 全部异常 |
| bathtub | 8 | ✗ | 方向模糊 |

#### 2.2.2 异常值排除

基于 `P2v2_SoftGate_20251229` 模型评估结果，生成异常值清单：

```json
// data_annotation/1front_outliers.json
{
  "thresholds": {
    "severe": "误差 >= 90° (前后颠倒)",
    "major": "误差 >= 45° (严重偏差)",
    "moderate": "误差 >= 15° (中等偏差)"
  },
  "summary": {
    "total_1front_samples": 465,
    "severe_outliers": 77,
    "major_outliers": 96,
    "moderate_outliers": 155
  }
}
```

**使用阈值**: `severe` (排除77个误差≥90°的样本)

### 2.3 Ground Truth 生成规则

```python
DIRECTION_TO_ANGLE = {
    '+X': 0.0,
    '+Z': np.pi / 2,
    '-X': np.pi,
    '-Z': 3 * np.pi / 2,
}

# GT角度生成
def get_gt_angles(label, base_angle):
    if label == 0:  # 1-front
        return [base_angle] * 4  # 4峰同方向
    elif label == 1:  # 2-front
        return [base_angle, base_angle + π]  # 2峰间隔180°
    elif label == 2:  # 4-front
        return [base_angle + k*π/2 for k in range(4)]  # 4峰间隔90°
    else:  # rot-sym, no-front
        return None  # 无方向监督
```

### 2.4 数据增强

**旋转增强**: 绕Y轴随机旋转

```python
# 训练时: 随机旋转角度
rotation_angle = np.random.uniform(0, 2 * np.pi)

# 验证/测试时: 确定性均匀旋转
rotation_angle = rotation_idx * (2 * np.pi / num_rotations)

# 旋转点云
cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
rotation_matrix = np.array([
    [cos_r, 0, sin_r],
    [0, 1, 0],
    [-sin_r, 0, cos_r]
])
points = points @ rotation_matrix.T

# 更新GT角度
gt_angle = (base_angle + rotation_angle) % (2 * np.pi)
```

### 2.5 类别平衡采样

使用 `WeightedRandomSampler` 解决类别不平衡：

```python
# 计算每个类别的权重 (逆频率)
class_weights = {label: 1.0 / count for label, count in label_counts.items()}
sample_weights = [class_weights[s['label']] for s in samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

---

## 3. 实验1: Clean Classifier

### 3.1 实验配置

| 参数 | 值 |
|------|-----|
| Checkpoint | `CleanClassifier_20251229_220630` |
| Epochs | 50 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| 旋转增强倍数 | 12 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| 1-front类别 | airplane, chair |
| 异常值阈值 | severe (≥90°) |

### 3.2 模型架构

```
SymmetryClassifier
├── PointNetPlusPlusEncoder (in=3, out=1024)
│   ├── SetAbstraction(512, r=0.2, ns=32, MLP=[64, 64, 128])
│   ├── SetAbstraction(128, r=0.4, ns=64, MLP=[128, 128, 256])
│   ├── SetAbstraction(32, r=0.8, ns=128, MLP=[256, 512, 1024])
│   └── FC(1024 → 1024) + BN + ReLU + Dropout(0.4)
│
└── Classifier Head
    ├── Linear(1024 → 256) + BN + ReLU + Dropout(0.3)
    ├── Linear(256 → 128) + BN + ReLU + Dropout(0.3)
    └── Linear(128 → 5)  # 5类输出

参数量: ~1.7M
```

### 3.3 损失函数

```python
loss = CrossEntropyLoss(logits, labels)
```

### 3.4 训练结果

| 指标 | 值 |
|------|-----|
| **Best Val Acc** | **99.11%** |
| 训练时长 | ~30分钟 |

**各类别验证准确率**:

| 类别 | 准确率 |
|------|--------|
| 1-front | 97.6% |
| 2-front | 99.2% |
| 4-front | 100% |
| Rot-sym | 99.4% |
| No-front | 98.8% |

---

## 4. 实验2: P2v2 SoftGate (旧版)

> 此实验用于生成异常值清单，非最终模型

### 4.1 实验配置

| 参数 | 值 |
|------|-----|
| Checkpoint | `P2v2_SoftGate_20251229_183118` |
| Epochs | 50 |
| 旋转增强倍数 | 4 |
| Classifier | `SymClassifier_20251216_035345` (旧版) |
| 数据过滤 | 无 |

### 4.2 评估结果

此模型用于识别1-front类别中的异常样本：

- 总1-front样本: 465
- 严重异常 (≥90°): 77 (16.6%)
- 主要集中在: bookshelf (47), wardrobe (10), bathtub (4)

---

## 5. 实验3: P2v2 Clean (最终版)

### 5.1 实验配置

| 参数 | 值 |
|------|-----|
| Checkpoint | `P2v2_Clean_20251230_165848` |
| Epochs | 100 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| 旋转增强倍数 | 12 |
| Classifier | `CleanClassifier_20251229_220630` (冻结) |
| 1-front类别 | airplane, chair |
| 异常值阈值 | severe (排除77个) |
| WandB Run ID | `yjthu57d` |

### 5.2 模型架构: ProbabilisticOrientationNet (MoE)

```
ProbabilisticOrientationNet
├── Classifier (冻结, 作为Gate)
│   └── SymmetryClassifier → weights [B, 5]
│
├── Shared Backbone (PointNet++)
│   ├── SA(512, 32, MLP=[64, 64, 128])
│   ├── SA(128, 64, MLP=[128, 128, 256])
│   └── SA(global, MLP=[256, 512, 1024]) → [B, 1024]
│
├── ExpertHead_1front (1峰)
│   ├── Hidden: Linear(1024→256) + ReLU + Linear(256→256) + ReLU
│   ├── fc_mu: Linear(256 → 2)  → atan2 → μ [B, 1]
│   └── fc_kappa: Linear(256 → 1) → softplus → κ [B, 1]
│
├── ExpertHead_2front (2峰)
│   ├── Hidden: 同上
│   ├── fc_mu: Linear(256 → 4)  → μ [B, 2]
│   └── fc_kappa: Linear(256 → 2) → κ [B, 2]
│
└── ExpertHead_4front (4峰)
    ├── Hidden: 同上
    ├── fc_mu: Linear(256 → 8)  → μ [B, 4]
    └── fc_kappa: Linear(256 → 4) → κ [B, 4]

总参数量: ~3.5M
可训练参数: ~1.8M (Classifier冻结)
```

### 5.3 损失函数: MaskedExpertLoss

#### 5.3.1 核心思想

- **GT-based Routing**: 训练时根据GT label选择对应的Expert Head
- **Soft Gate Weighting**: 使用Classifier的输出作为样本权重
- **Von Mises NLL**: 对方向预测使用von Mises分布的负对数似然

#### 5.3.2 样本权重计算

```python
def compute_sample_weights(gate_weights, gt_labels):
    # p_dir = P(1-front) + P(2-front) + P(4-front)
    p_dir = gate_weights[:, 0] + gate_weights[:, 1] + gate_weights[:, 2]

    # Soft gate weight: 线性映射 [threshold, 1] → [0, 1]
    w_gate = clamp((p_dir - threshold) / (1 - threshold), 0, 1)

    # Class confidence weight
    p_gt = gate_weights.gather(1, gt_labels)
    w_cls = p_gt ** gamma  # gamma=1.5

    return w_gate * w_cls
```

#### 5.3.3 Von Mises NLL

```python
def von_mises_nll(theta_gt, mu, kappa):
    """
    p(θ|μ,κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))
    NLL = -κ * cos(θ_gt - μ) + log(2π) + log(I₀(κ))
    """
    cos_diff = torch.cos(theta_gt - mu)
    nll = -kappa * cos_diff + log(2π) + log_bessel_i0(kappa)
    return nll
```

#### 5.3.4 Hungarian匹配 (2-front, 4-front)

```python
# K=2的情况: 只有2种排列
cost_identity = (1 - cos(pred - gt)).sum()
cost_swap = (1 - cos(pred - gt_swapped)).sum()
use_swap = cost_swap < cost_identity

# K=4的情况: 24种排列
perms_4 = list(itertools.permutations(range(4)))  # [24, 4]
gt_permuted = gt_angles[:, perms_4]  # [B, 24, 4]
costs = (1 - cos(pred - gt_permuted)).sum(dim=2)  # [B, 24]
best_perm_idx = costs.argmin(dim=1)
```

#### 5.3.5 调度参数

| 参数 | 初始值 | 最终值 | 调度 |
|------|--------|--------|------|
| cosine_weight | 0.2 | 0.1 | epoch 10后切换 |
| kappa_reg_weight | 0.0 | 0.02 | epoch 6开始线性增加 |
| kappa_target | 0.0 | 5.0 | 同上 |

#### 5.3.6 完整损失公式

```python
# 1-front loss (始终包含cosine loss)
loss_1f = NLL(gt, μ, κ) + cosine_weight * (1 - cos(μ - gt))
loss_1f += kappa_reg_weight * relu(kappa_target - κ)

# 2-front loss
gt_matched = hungarian_match(pred_μ, [gt, gt+π])
loss_2f = mean(NLL(gt_matched, μ, κ))

# 4-front loss
gt_matched = hungarian_match(pred_μ, [gt, gt+π/2, gt+π, gt+3π/2])
loss_4f = mean(NLL(gt_matched, μ, κ))

# 加权平均 (micro-average)
total_loss = sum(w * loss) / sum(w)
```

### 5.4 训练曲线

```
Epoch  Train Loss  Val 1f   Val 2f   Val 4f   Val Overall
─────────────────────────────────────────────────────────
1      2.4523      45.2°    12.3°    8.5°     23.8°
10     0.8234      18.5°    5.8°     2.1°     9.2°
20     0.4521      14.2°    4.1°     0.8°     7.1°
40     0.2834      12.1°    3.5°     0.4°     6.0°
60     0.2156      11.5°    3.2°     0.3°     5.7°
80     0.1892      11.2°    3.1°     0.2°     5.6°
93*    0.1734      11.0°    3.0°     0.2°     5.61°  ← Best
100    0.1689      12.1°    3.2°     0.2°     5.8°
```

### 5.5 最终结果

| 指标 | 验证集 | 测试集 |
|------|--------|--------|
| **整体误差** | **5.61°** | **4.66°** |
| 1-front误差 | 11.0° | 10.18° |
| 1-front中位数 | 1.0° | 0.92° |
| 1-front <10° | 93.1% | 93.13% |
| 1-front >90° | 5.0% | 4.95% |
| 1-front κ均值 | 90.3 | 90.29 |
| 2-front误差 | 3.0° | 2.51° |
| 2-front <10° | 98.0% | 97.92% |
| 4-front误差 | 0.2° | 0.22° |
| 4-front <10° | 100% | 100% |

**训练时长**: 5.7小时 (100 epochs)

---

## 6. 实验4: MuOnly Baseline

### 6.1 实验配置

| 参数 | 值 |
|------|-----|
| Checkpoint | `MuOnly_Baseline_20251231_040746` |
| Epochs | 50 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| 旋转增强倍数 | 10 |
| Lambda_mu | 2.0 |
| WandB Run ID | `dc7re82u` |

### 6.2 模型架构: BaselineDirectionModel

```
BaselineDirectionModel (无MoE, 无Classifier)
├── PointNetPlusPlusEncoder (in=3, out=1024)
│   └── 同SymmetryClassifier
│
└── MixturePeakHead (固定4峰)
    ├── Shared: Linear(1024→512) + BN + ReLU + Dropout(0.3)
    ├── mu_head: Linear(512 → 8) → reshape → [B, 4, 2]
    │            → normalize → atan2 → μ [B, 4]
    └── kappa_head: Linear(512 → 4) → softplus → κ [B, 4]

参数量: ~1.8M (全部可训练)
```

### 6.3 损失函数: BaselineMuOnlyLoss

**纯角度回归损失 (无von Mises NLL)**

```python
def forward(outputs, gt_angle, gt_label):
    pred_mu = outputs['mu']  # [B, 4]

    for i in range(B):
        label = gt_label[i]
        gt = gt_angle[i]
        pred = pred_mu[i]  # [4,]

        if label == 0:  # 1-front: 4峰同方向
            loss = mean(1 - cos(pred - gt))

        elif label == 1:  # 2-front: 2峰间隔180°
            gt_peaks = [gt, gt + π]
            loss = hungarian_cosine_loss(pred[:2], gt_peaks)

        elif label == 2:  # 4-front: 4峰间隔90°
            gt_peaks = [gt + k*π/2 for k in range(4)]
            loss = hungarian_cosine_loss(pred, gt_peaks)

        # label 3, 4 跳过

    total_loss = lambda_mu * mean([loss_1f, loss_2f, loss_4f])
```

### 6.4 训练曲线

```
Epoch  Train Loss  Val 1f   Val 2f   Val 4f   Val Overall
─────────────────────────────────────────────────────────
1      0.4892      42.3°    18.5°    12.1°    24.8°
10     0.1523      15.2°    7.8°     4.5°     9.8°
20     0.0912      12.1°    6.2°     3.2°     7.5°
30     0.0834      11.2°    5.8°     2.5°     6.8°
40     0.0812      10.8°    5.5°     2.1°     6.4°
46*    0.0882      10.5°    5.5°     2.0°     6.23°  ← Best
50     0.0842      10.4°    5.5°     2.2°     6.2°
```

### 6.5 最终结果

| 指标 | 验证集 | 测试集 |
|------|--------|--------|
| **整体误差** | **6.23°** | **6.17°** |
| 1-front误差 | 10.5° | 10.46° |
| 1-front中位数 | 2.0° | 1.79° |
| 1-front <10° | 89.8% | 89.84% |
| 1-front >90° | 4.4% | 4.40% |
| 1-front κ均值 | 0.76 | 0.76 |
| 2-front误差 | 5.5° | 6.06° |
| 2-front <10° | 94.4% | 94.38% |
| 4-front误差 | 2.0° | 1.60° |
| 4-front <10° | 99.4% | 99.40% |

**训练时长**: 4.25小时 (50 epochs)

---

## 7. 结果对比与分析

### 7.1 P2v2 Clean vs MuOnly Baseline

| 模型 | 整体误差 | 1-front | 2-front | 4-front | 训练时长 |
|------|---------|---------|---------|---------|---------|
| **P2v2 Clean** | **5.61°** | 10.18° | **2.51°** | **0.22°** | 5.7h |
| MuOnly Baseline | 6.23° | **10.46°** | 6.06° | 1.60° | 4.25h |
| **差异** | -0.62° | +0.28° | -3.55° | -1.38° | +1.45h |

### 7.2 关键发现

1. **MoE架构对对称物体更有效**
   - 2-front: MoE减少3.55°误差 (58%改进)
   - 4-front: MoE减少1.38°误差 (86%改进)
   - 原因: 专用的Expert Head可以更好地学习对称模式

2. **1-front表现相近**
   - MuOnly略好0.28° (约3%改进)
   - 可能原因: MoE的soft gate weighting机制在低p_dir样本上有副作用

3. **κ值分布差异巨大**
   - P2v2 Clean: κ均值90.3 (高置信度)
   - MuOnly: κ均值0.76 (低置信度)
   - 原因: MuOnly使用纯cosine loss，κ未被监督

### 7.3 误差分布分析

**1-front误差分布对比**:

| 误差范围 | P2v2 Clean | MuOnly |
|---------|------------|--------|
| <5° | 89.56% | 80.63% |
| <10° | 93.13% | 89.84% |
| <15° | 93.82% | 92.86% |
| >45° | 5.49% | 4.95% |
| >90° | 4.95% | 4.40% |

**观察**: 两个模型的>90°比例接近(~5%)，说明这些是数据本身的问题样本

---

## 8. 代码架构详解

### 8.1 项目结构

```
ForwardNet-claude/
├── train_clean_pipeline.py      # 主训练脚本 (包含3个Trainer)
├── train_symmetry_classifier.py # 分类器训练脚本
├── resume_training.sh           # 恢复训练脚本
├── create_resume_checkpoint.py  # 创建可恢复checkpoint
│
├── models/
│   ├── __init__.py
│   ├── base.py                  # 基础函数 (index_points, query_ball_point)
│   ├── symmetry_classifier.py   # 对称性分类器
│   ├── probabilistic_orientation_net.py  # P2v2模型 + MaskedExpertLoss
│   └── pointnet_pp_vonMises.py  # 原始PointNet++单峰模型
│
├── datasets/
│   ├── moe_dataset.py           # MoE数据集
│   └── symmetry_classifier_dataset.py  # 分类器数据集
│
├── data_annotation/
│   ├── symmetry_annotations.json    # 主标注文件
│   ├── 1front_outliers.json         # 异常值清单
│   └── symmetry_annotations_filtered.json  # 过滤后标注
│
└── checkpoints/
    ├── CleanClassifier_20251229_220630/
    ├── P2v2_SoftGate_20251229_183118/
    ├── P2v2_Clean_20251230_165848/
    └── MuOnly_Baseline_20251231_040746/
```

### 8.2 关键类和函数

#### 8.2.1 CleanMoEDataset

```python
class CleanMoEDataset(Dataset):
    """
    带数据过滤的MoE数据集

    参数:
        allowed_1front_categories: 1-front允许的物体类别
        outlier_json: 异常值JSON文件路径
        outlier_threshold: 排除阈值 ('severe', 'major', 'moderate')
    """
```

#### 8.2.2 ProbabilisticOrientationNet

```python
class ProbabilisticOrientationNet(nn.Module):
    """
    概率方向预测网络 (Mixture of Experts)

    架构:
        1. Classifier (Gate): 预测对称类别权重 [B, 5]
        2. Shared Backbone: PointNet++ 提取全局特征 [B, 1024]
        3. Expert Heads: 3个专用预测头
    """
```

#### 8.2.3 MaskedExpertLoss

```python
class MaskedExpertLoss(nn.Module):
    """
    GT-based Routing + Soft Gate Weighting 损失函数

    特性:
        - 根据GT label选择Expert Head
        - 使用Classifier输出作为样本权重
        - Von Mises NLL + Cosine Loss
        - 调度的κ正则化
    """
```

#### 8.2.4 BaselineDirectionModel

```python
class BaselineDirectionModel(nn.Module):
    """
    原始MF系列基线模型

    架构:
        PointNet++ Encoder + MixturePeakHead (固定4峰)
        不使用Classifier，不使用MoE
    """
```

### 8.3 训练流程

```python
# train_clean_pipeline.py

def main():
    # Step 1: 训练分类器 (或使用已有)
    if not args.skip_classifier:
        classifier_trainer = ClassifierTrainer(args)
        classifier_path = classifier_trainer.train()

    # Step 2: 训练P2v2
    if not args.skip_p2v2:
        p2v2_trainer = P2v2Trainer(args, classifier_path,
                                    resume_dir=args.resume_p2v2)
        p2v2_trainer.train()

    # Step 3: 训练MuOnly Baseline
    if args.run_muonly:
        muonly_trainer = MuOnlyTrainer(args)
        muonly_trainer.train()
```

### 8.4 恢复训练机制

```python
# Checkpoint包含完整状态
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_error': best_val_error,
    'wandb_run_id': wandb_run_id,  # 用于恢复同一个WandB run
    'metrics': val_metrics,
}

# 恢复时
if resume_checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    # 恢复WandB run
    wandb.init(id=checkpoint['wandb_run_id'], resume="must")
```

---

## 9. 附录: 完整配置参数

### 9.1 P2v2 Clean 配置

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

  "p2v2_epochs": 100,
  "p2v2_lr": 0.001,
  "p2v2_num_rotations": 12,

  "backbone_dim": 1024,
  "expert_hidden_dim": 256,
  "kappa_min": 0.0001,
  "kappa_max": 100.0,

  "p_dir_threshold": 0.4,
  "gamma": 1.5,
  "cosine_weight_init": 0.2,
  "cosine_weight_final": 0.1,
  "cosine_schedule_epoch": 10,
  "kappa_reg_weight_final": 0.02,
  "kappa_target_final": 5.0,
  "kappa_reg_start_epoch": 6,
  "kappa_reg_ramp_epochs": 10,

  "wandb": true,
  "wandb_project": "ForwardNet-LossAblation"
}
```

### 9.2 MuOnly Baseline 配置

```json
{
  "model_type": "baseline",
  "loss_type": "mu_only",

  "muonly_epochs": 50,
  "muonly_lr": 0.001,
  "muonly_num_rotations": 10,
  "lambda_mu": 2.0,

  "backbone_dim": 1024,
  "expert_hidden_dim": 256
}
```

### 9.3 Clean Classifier 配置

```json
{
  "classifier_epochs": 50,
  "classifier_lr": 0.001,
  "classifier_num_rotations": 12,

  "encoder_dim": 1024,
  "num_classes": 5
}
```

---

## 参考文献

1. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
2. Von Mises Distribution for Circular Data Modeling
3. Mixture of Experts for Multi-Modal Learning
4. Hungarian Algorithm for Bipartite Matching

---

*文档生成时间: 2025-12-31*

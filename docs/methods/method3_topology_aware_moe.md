# Method 3: Topology-Aware Mixture of Experts (MoE)

---

## 1. 概述

### 1.1 方法定位

这是继 **Method 1 (Discrete)** 和 **Method 2 (Fixed 4-Peak MvM)** 之后的第三种方向预测方法。

| 方法 | 输出形式 | 多峰处理 | 对称性感知 |
|------|----------|----------|------------|
| Method 1: Discrete | 8-bin分类 | 无 | 无 |
| Method 2: Fixed 4-Peak | 固定4峰MvM | 全部4峰 | 通过κ→0 |
| **Method 3: MoE** | 动态K峰MvM | 按类别路由 | 显式分类器 |

### 1.2 核心思想

**两阶段策略**：
1. **Stage 1**: 预训练分类器判断对称性类型（已完成，97.9%准确率）
2. **Stage 2**: 根据对称性类型路由到对应专家头

```
Point Cloud
     ↓
┌─────────────────────────────────────────────────┐
│  Frozen Classifier (Gate)                       │
│  → 输出: [w0, w1, w2, w3, w4] (5类权重)         │
└─────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────┐
│  Shared PointNet++ Backbone                     │
│  → 输出: [B, 1024] 全局特征                     │
└─────────────────────────────────────────────────┘
     ↓
┌──────────┬──────────┬──────────┐
│ Expert 1 │ Expert 2 │ Expert 4 │
│ (1-peak) │ (2-peak) │ (4-peak) │
└──────────┴──────────┴──────────┘
     ↓
Final PDF = Σ wi × Expert_i(θ)
```

---

## 2. 标签映射

### 2.1 MoE 标签约定 (Method 3)

```python
# models/probabilistic_orientation_net.py
# datasets/moe_dataset.py

LABEL_TO_NAME = {
    0: '1-front',      # 1个正面 → head_1front (1 peak)
    1: '2-front',      # 2个正面 → head_2front (2 peaks)
    2: '4-front',      # 4个正面 → head_4front (4 peaks)
    3: 'Rot-sym',      # 旋转对称 → Uniform distribution
    4: 'No-front',     # 无正面   → Uniform distribution
}
```

### 2.2 旧标签约定对比 (Method 1 & 2)

```python
# 旧约定 (data/symmetry_classification_gt/)
OLD_LABEL_TO_NAME = {
    0: 'no_front',     # ⚠️ 注意：旧约定中 0 是 no_front
    1: '1_front',
    2: '2_fronts',
    3: '4_fronts',
    4: 'symmetric',
}
```

### 2.3 映射关系

| MoE Label | MoE Name | Old Label | Old Name |
|-----------|----------|-----------|----------|
| 0 | 1-front | 1 | 1_front |
| 1 | 2-front | 2 | 2_fronts |
| 2 | 4-front | 3 | 4_fronts |
| 3 | Rot-sym | 4 | symmetric |
| 4 | No-front | 0 | no_front |

---

## 3. 模型架构

### 3.1 核心组件

```python
class ProbabilisticOrientationNet(nn.Module):
    def __init__(self,
                 classifier,              # 预训练分类器
                 backbone_dim=1024,       # Backbone输出维度
                 expert_hidden_dim=256,   # Expert隐藏层
                 kappa_min=1e-4,          # κ下限
                 kappa_max=100.0,         # κ上限
                 freeze_classifier=True): # 是否冻结分类器
```

### 3.2 参数量统计

| 组件 | 参数量 | 可训练 (freeze=True) |
|------|--------|---------------------|
| Classifier | ~2.16M | 否 |
| Backbone (SA1-SA3) | ~1.55M | 是 |
| Expert Heads | ~0.24M | 是 |
| **Total** | **~3.95M** | **~1.80M** |

### 3.3 Expert Head 输出

每个 Expert Head 输出 K 个 von Mises 分布参数：

```python
class ExpertHead(nn.Module):
    # 输出: K × (cos, sin, raw_kappa)
    # 转换:
    #   mu = atan2(sin, cos)           # 角度
    #   kappa = softplus(raw_kappa)    # 集中度 (clamped)
```

| Expert | K | 输出 | 混合权重 |
|--------|---|------|---------|
| head_1front | 1 | (μ, κ) | [1.0] |
| head_2front | 2 | [(μ₁, κ₁), (μ₂, κ₂)] | [0.5, 0.5] |
| head_4front | 4 | [(μ₁, κ₁), ..., (μ₄, κ₄)] | [0.25, 0.25, 0.25, 0.25] |

---

## 4. 训练配置

### 4.1 数据管线

```python
from datasets.moe_dataset import MoEDataset, get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    annotation_file='data_annotation/symmetry_annotations.json',
    data_dir='data/full_mn40_normal_resampled_ply',
    batch_size=32,
    num_points=2048,
    num_rotations=12,        # 12× 旋转增强
    use_balanced_sampler=True,  # 类别均衡采样
    seed=42
)
```

### 4.2 损失函数

```python
from models.probabilistic_orientation_net import MaskedExpertLoss

loss_fn = MaskedExpertLoss(
    classification_weight=0.0  # 0 = 冻结gate, >0 = 联合微调
)
```

**GT Hard Routing**: 训练时使用 GT 标签决定路由，只训练对应的 Expert Head。

| GT Label | 训练的 Expert | 损失函数 |
|----------|--------------|----------|
| 0 (1-front) | head_1front | von Mises NLL |
| 1 (2-front) | head_2front | Hungarian + NLL |
| 2 (4-front) | head_4front | Hungarian + NLL |
| 3, 4 | 无 | 跳过 |

### 4.3 训练命令

```bash
python train_moe.py \
    --exp_name 2-step_MoE_v1 \
    --epochs 100 \
    --batch_size 32 \
    --num_rotations 12 \
    --lr 1e-3 \
    --balanced_sampler \
    --wandb \
    --wandb_project ForwardNet-LossAblation
```

### 4.4 Gate 冻结 vs 联合微调

| 配置 | freeze_classifier | classification_weight | 说明 |
|------|-------------------|----------------------|------|
| 冻结 (默认) | True | 0.0 | 只训练 Expert Heads |
| 联合微调 | False | > 0 | 同时微调分类器 |

**注意**: 如果 `classification_weight > 0` 但 `freeze_classifier=True`，代码会抛出 `RuntimeError`。

---

## 5. 推理策略

### 5.1 Soft Routing

推理时使用分类器的 softmax 输出作为权重：

```python
from models.probabilistic_orientation_net import get_final_pdf

model.eval()
with torch.no_grad():
    output = model(points)
    pdf = get_final_pdf(output, num_points=360)  # [B, 360]
```

**最终 PDF**:
```
P(θ) = w₀ × P_1front(θ) + w₁ × P_2front(θ) + w₂ × P_4front(θ)
     + (w₃ + w₄) × Uniform(1/2π)
```

### 5.2 Peak Extraction

```python
from models.probabilistic_orientation_net import get_peak_predictions

predictions = get_peak_predictions(output)
# predictions['predicted_angles']: [B] 最可能的角度
# predictions['confidence']: [B] 置信度 (kappa)
# predictions['predicted_class']: [B] 预测类别
```

---

## 6. 代码位置

| 文件 | 内容 |
|------|------|
| `models/probabilistic_orientation_net.py` | 模型定义、损失函数、推理函数 |
| `datasets/moe_dataset.py` | 数据集、DataLoader |
| `train_moe.py` | 训练脚本 |
| `checkpoints/SymClassifier_20251216_035345/` | 预训练分类器 (97.9% acc) |

---

## 7. 与 Method 2 的关键区别

| 特性 | Method 2 (Fixed 4-Peak) | Method 3 (MoE) |
|------|------------------------|----------------|
| 峰数 | 固定 4 峰 | 动态 (1/2/4) |
| 对称性处理 | κ→0 表示无效峰 | 显式路由到不同 Head |
| 训练效率 | 所有样本训练所有峰 | GT 路由只训练对应 Head |
| 推理 | 直接输出 4 峰 | 加权混合多个 Head |
| 分类器 | 无 | 预训练冻结/可微调 |

---

## 8. 实验记录

### 2-step_MoE_v1 (2025-12-28)

**配置**:
- epochs: 100
- batch_size: 32
- num_rotations: 12
- lr: 1e-3
- balanced_sampler: True
- freeze_classifier: True

**数据**:
- Train: 1,858 samples × 12 = 22,296
- Val: 396 samples
- Test: 403 samples

**WandB**: ForwardNet-LossAblation

---

## 附录: Hungarian 匹配

对于 2-front 和 4-front 类别，GT 只给出一个角度 θ，需要生成对称的多个角度：

- 2-front: [θ, θ+π]
- 4-front: [θ, θ+π/2, θ+π, θ+3π/2]

然后使用 Hungarian 匹配将预测的 K 个峰与 GT 的 K 个角度配对，最小化 cosine 距离。

```python
# 2-front: 枚举 2 种排列
# 4-front: 枚举 24 种排列 (缓存为 buffer)
```

# Issue Report: MoE 方向预测模型问题汇总

本报告包含四个问题：

1. **[严重] 1-Front Expert Head Kappa Collapse** - 1-front head 的 kappa 坍塌到 0 ❌ 未解决
2. **[工程 Bug] Validation Loss 统计方式错误** - 导致 val loss 出现虚假 spike ✅ 已修复
3. **[工程 Bug] 验证集随机点云采样** - 导致同一模型每个 epoch 结果不同 ✅ 已修复
4. **[设计问题] 有效样本数少** - 只有 directional 类别参与 loss 计算 ⚠️ 需注意

---

## Issue #1: 1-Front Expert Head Kappa Collapse Problem

### TL;DR

在 MoE (Mixture of Experts) 方向预测模型中，**1-front expert head 的 kappa 参数会坍塌到 0**，导致该 head 无法学习方向预测。2-front 和 4-front heads 不受影响。

核心原因是 **von Mises NLL 的梯度方向问题**：当预测方向错误时，减小 kappa 可以降低 loss，模型会陷入 kappa=0 的局部最优。

---

## 1. 问题描述

### 1.1 现象

训练 MoE 模型后，观察到：

```
1-front loss: 1.8379 (= log(2π)) ← 不变，说明 kappa ≈ 0
2-front loss: -2.7              ← 负值，说明 kappa 正常
4-front loss: -3.9              ← 负值，说明 kappa 正常
```

### 1.2 诊断结果

检查训练后的模型参数：

```python
# head_1front 的 raw_kappa 输出
raw_kappa: [-72.14, -59.49, -36.36, -61.42, -66.85]  # 极端负值！

# 经过 softplus 后
softplus(raw_kappa): [4.6e-32, 1.4e-26, ...]  # 基本是 0

# 经过 clamp 后
kappa (final): [0.0001, 0.0001, ...]  # 卡在下限
```

### 1.3 为什么只有 1-front 有问题？

**对称性带来的"容错度"不同**：

| 类别 | 对称方向数 | 最大角度差 | cos(最大Δθ) | 初始预测容易？ |
|------|----------|----------|------------|--------------|
| 1-front | 1 | 180° | -1 | 难 |
| 2-front | 2 | 90° | 0 | 中 |
| 4-front | 4 | 45° | +0.707 | 易 |

- **1-front**：只有 1 个正确方向，初始预测可能完全相反（180°）
- **2-front**：有 2 个对称方向（相差 180°），最坏误差只有 90°
- **4-front**：有 4 个对称方向（相差 90°），最坏误差只有 45°

---

## 2. 数学分析

### 2.1 Von Mises 分布

```
p(θ|μ,κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))
```

其中：
- μ: 均值方向
- κ: 集中度参数（越大分布越集中）
- I₀(κ): 第一类修正 Bessel 函数

### 2.2 Negative Log-Likelihood (NLL)

```
NLL = -κ * cos(Δθ) + log(2π) + log(I₀(κ))
```

其中 Δθ = θ_gt - μ 是预测误差。

### 2.3 Loss 对 κ 的梯度

```
∂NLL/∂κ = -cos(Δθ) + I₁(κ)/I₀(κ)
```

其中 I₁(κ)/I₀(κ) ≈ 1 当 κ 较大时。

**关键观察**：
- 当 cos(Δθ) > 0（预测正确）：增大 κ 可以减小 loss
- 当 cos(Δθ) < 0（预测错误）：**增大 κ 反而增大 loss**

### 2.4 1-front 陷入陷阱的过程

```
初始状态:
  - μ 随机，可能与 GT 相差 180°
  - κ ≈ 1 (初始化)
  - cos(Δθ) ≈ -1 (预测完全错误)

梯度方向:
  ∂NLL/∂κ = -cos(Δθ) + I₁(κ)/I₀(κ)
          = -(-1) + 0.45  (假设 κ=1)
          = 1.45 > 0

所以 κ 会减小！

陷入陷阱:
  - κ 减小 → NLL 减小 (因为 -κ*cos(Δθ) = -κ*(-1) = κ 减小)
  - 但 κ 减小 → μ 的梯度 ∝ κ*sin(Δθ) 也减小
  - 最终 κ → 0，μ 无法学习
  - NLL → log(2π) ≈ 1.8379 (均匀分布)
```

### 2.5 为什么 2-front 和 4-front 不受影响？

**2-front** (有 2 个对称方向 [θ, θ+π])：
- 使用 Hungarian matching 匹配最近的 GT
- 最大误差只有 90°，cos(90°) = 0
- 梯度 ∂NLL/∂κ = 0 + 0.45 = 0.45，κ 不会快速下降

**4-front** (有 4 个对称方向 [θ, θ+π/2, θ+π, θ+3π/2])：
- 最大误差只有 45°，cos(45°) = 0.707 > 0
- 梯度 ∂NLL/∂κ = -0.707 + 0.45 = -0.26 < 0
- κ 会增大！

---

## 3. 代码分析

### 3.1 模型结构

```python
# models/probabilistic_orientation_net.py

class ExpertHead(nn.Module):
    """每个 expert head 预测 (mu, kappa) 对"""

    def __init__(self, in_dim, num_peaks, hidden_dim=256,
                 kappa_min=1e-4, kappa_max=100.0):
        super().__init__()
        self.num_peaks = num_peaks
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        # MLP: in_dim -> hidden_dim -> hidden_dim -> num_peaks * 3
        # 输出: num_peaks 个 (cos, sin, raw_kappa)
        output_dim = num_peaks * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        raw = self.mlp(features)  # [B, num_peaks * 3]
        raw = raw.view(B, self.num_peaks, 3)

        cos_val = raw[:, :, 0]
        sin_val = raw[:, :, 1]
        raw_kappa = raw[:, :, 2]

        mu = torch.atan2(sin_val, cos_val)

        # 问题在这里：当 raw_kappa 很负时，softplus 梯度接近 0
        kappa = F.softplus(raw_kappa)
        kappa = kappa.clamp(min=self.kappa_min, max=self.kappa_max)

        return {'mu': mu, 'kappa': kappa}
```

### 3.2 Loss 函数

```python
def von_mises_nll(theta_gt, mu, kappa):
    """Von Mises NLL"""
    cos_diff = torch.cos(theta_gt - mu)
    nll = -kappa * cos_diff + math.log(2 * math.pi) + log_bessel_i0(kappa)
    return nll


class MaskedExpertLoss(nn.Module):
    """基于 GT label 的 hard routing loss"""

    def forward(self, model_output, gt_angles, gt_labels):
        # 1-front (label == 0)
        if count_1front > 0:
            mu = model_output['head_1front']['mu'][mask_1front, 0]
            kappa = model_output['head_1front']['kappa'][mask_1front, 0]
            gt = gt_angles[mask_1front]

            nll = von_mises_nll(gt, mu, kappa)
            loss_1front = nll.mean()

        # 2-front (label == 1) - 使用 Hungarian matching
        if count_2front > 0:
            mu = model_output['head_2front']['mu'][mask_2front]  # [B, 2]
            kappa = model_output['head_2front']['kappa'][mask_2front]
            gt = gt_angles[mask_2front]

            # GT 对称方向: [gt, gt + π]
            gt_sym = torch.stack([gt, gt + math.pi], dim=1)

            # Hungarian matching
            gt_matched = batched_hungarian_match_k2(mu, gt_sym)

            nll = von_mises_nll(gt_matched, mu, kappa)
            loss_2front = nll.mean()

        # 4-front 类似，使用 4 个对称方向
```

---

## 4. 实验验证

### 4.1 实验 1：观察 kappa 坍塌

```python
# 训练 10 个 epoch，只在 1-front 样本上
for epoch in range(10):
    # ... 训练代码 ...
    print(f"Epoch {epoch+1}:")
    print(f"  1-front loss: {loss_1front:.4f}")
    print(f"  1-front kappa: {kappa_1front:.4f}")
    print(f"  raw_kappa range: [{raw_kappa_min:.2f}, {raw_kappa_max:.2f}]")
```

**结果**：

```
Epoch 1:
  1-front loss: 1.9046
  1-front kappa: 0.1677
  raw_kappa range: [-18.95, -1.14]

Epoch 2:
  1-front loss: 1.8377
  1-front kappa: 0.0067
  raw_kappa range: [-46.79, -10.11]

...

Epoch 10:
  1-front loss: 1.8372  ← 稳定在 log(2π)
  1-front kappa: 0.0067  ← 卡在下限
  raw_kappa range: [-71.96, -15.65]  ← 继续变负
```

### 4.2 实验 2：固定 kappa=10 训练

```python
# 使用固定 kappa=10，只学习 mu
FIXED_KAPPA = 10.0

def fixed_kappa_loss(output, gt_angles, gt_labels):
    mu = output['head_1front']['mu'][mask_1front, 0]
    gt = gt_angles[mask_1front]
    kappa = torch.full_like(mu, FIXED_KAPPA)  # 固定 kappa
    nll = von_mises_nll(gt, mu, kappa)
    return nll.mean()
```

**结果**：

```
Epoch 1:  loss=9.27, error=87.6°
Epoch 2:  loss=10.47, error=94.5°
...
Epoch 10: loss=8.16, error=75.4°

最终结果:
  平均角度误差: 72.5°
  误差 < 30°: 28.5%
  误差 < 15°: 14.0%
```

**结论**：即使固定 kappa，1-front 仍然难以学习，说明问题可能不仅仅是 kappa 坍塌。

---

## 5. 尝试过的修复方案

### 5.1 方案 1：改变 kappa 激活函数

**原始**：`softplus(x)` → 当 x << 0 时，梯度 → 0

**修改**：`exp(clamp(x, -5, 5))` → 输出范围 [0.0067, 148.4]

```python
# 修改后的 ExpertHead.forward()
raw_kappa_clamped = torch.clamp(raw_kappa, min=-5.0, max=5.0)
kappa = torch.exp(raw_kappa_clamped)
```

**结果**：❌ 无效

raw_kappa 仍然学到了极端负值（-72），说明问题在于梯度方向，不是激活函数。

### 5.2 方案 2：固定 kappa

直接使用固定的 kappa=10，只学习 mu。

**结果**：❌ 部分有效

1-front 仍然难以学习，平均误差 72.5°。可能需要更多训练或更好的初始化。

---

## 6. 数据统计

### 6.1 数据集分布

```
训练集 (1704 samples, x12 rotations = 20448 total):
  0 (1-front): 450 (26.4%)
  1 (2-front): 263 (15.4%)
  2 (4-front): 190 (11.1%)
  3 (Rot-sym): 564 (33.1%)  ← 不需要方向预测
  4 (No-front): 237 (13.9%) ← 不需要方向预测

验证集 (148 samples):
  0 (1-front): 37
  1 (2-front): 35
  2 (4-front): 18
  3 (Rot-sym): 43
  4 (No-front): 15
```

### 6.2 物体类别

1-front 类别示例：
- wardrobe (衣柜)
- chair (椅子)
- bookshelf (书架)
- desk (桌子)

---

## 7. 可能的解决方案

### 7.1 方案 A：使用 Cosine Loss 而非 Von Mises NLL

```python
def cosine_loss(mu, theta_gt):
    """不依赖 kappa 的角度 loss"""
    return 1 - torch.cos(mu - theta_gt)
```

**优点**：
- μ 的梯度始终存在
- 不会陷入 kappa=0 的陷阱

**缺点**：
- 无法输出置信度（kappa）
- 对于多模态分布不适用

### 7.2 方案 B：两阶段训练

1. **阶段 1**：固定 kappa=10，只学习 mu
2. **阶段 2**：解冻 kappa，联合学习

### 7.3 方案 C：初始化 mu 接近 GT

使用数据增强或其他方式，确保初始预测不会完全错误。

### 7.4 方案 D：Loss 正则化

添加一个正则项，防止 kappa 过小：

```python
kappa_reg = -torch.log(kappa + 1e-6)  # 惩罚小 kappa
total_loss = nll + lambda_reg * kappa_reg
```

### 7.5 方案 E：温度退火

类似于模拟退火，逐渐增大 kappa：

```python
# 第 1 epoch: kappa_min = 0.1
# 第 10 epoch: kappa_min = 1.0
# 第 50 epoch: kappa_min = 10.0
```

---

## 8. 相关代码位置

- **模型定义**: `models/probabilistic_orientation_net.py`
  - `ExpertHead` class (line 159-240)
  - `ProbabilisticOrientationNet` class (line 243-370)
  - `MaskedExpertLoss` class (line 457-626)

- **训练脚本**: `train_moe.py`

- **数据集**: `datasets/moe_dataset.py`

- **标注文件**: `data_annotation/symmetry_annotations.json`

---

## 9. 复现步骤

```bash
# 1. 训练 MoE 模型
python train_moe.py \
    --exp_name "2-step_MoE_v2" \
    --classifier_checkpoint "checkpoints/SymClassifier_20251216_035345/best.pth" \
    --epochs 100 \
    --freeze_classifier \
    --wandb

# 2. 观察训练日志中的 1-front loss
# 如果 1-front loss 稳定在 1.8379，说明问题复现

# 3. 检查训练后的模型
python -c "
import torch
from models.probabilistic_orientation_net import ProbabilisticOrientationNet
# ... 加载模型并检查 head_1front 的 kappa
"
```

---

## 10. 期望的帮助

1. **理论分析**：是否有更好的方法处理这个 chicken-and-egg 问题？
2. **Loss 设计**：是否有其他 probabilistic 方向预测的 loss 函数？
3. **训练策略**：如何避免陷入 kappa=0 的局部最优？
4. **替代方案**：是否应该放弃 MoE 方法，使用其他架构？

---

## 附录：完整诊断脚本

```python
"""
诊断脚本：检查 1-front head 的 kappa 坍塌问题
"""
import torch
import torch.nn.functional as F

# 加载训练好的模型
model = ...  # 加载模型代码

# 检查 head_1front 的参数
model.eval()
with torch.no_grad():
    # 创建测试输入
    test_input = torch.randn(5, 2048, 3).cuda()
    output = model(test_input)

    # 获取 raw_kappa
    features = output['global_feat']
    raw = model.head_1front.mlp(features)
    raw = raw.view(-1, 1, 3)
    raw_kappa = raw[:, :, 2]

    print("raw_kappa:", raw_kappa.flatten().tolist())
    print("softplus(raw_kappa):", F.softplus(raw_kappa).flatten().tolist())
    print("final kappa:", output['head_1front']['kappa'].flatten().tolist())

# 检查梯度
model.train()
test_input = torch.randn(5, 2048, 3).cuda()
output = model(test_input)
loss = output['head_1front']['kappa'].sum()
loss.backward()

for name, param in model.head_1front.mlp.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
```

---
---

## Issue #2: Validation Loss 统计方式错误 (工程 Bug)

### TL;DR

Validation loss 出现剧烈波动（0.05～0.91，24% epoch spike），主要原因是 **统计方式的工程 bug**：
- 当前实现按 batch 等权平均
- 应该按 directional sample 数量加权平均

### 问题描述

观察到的 validation loss 行为：
```
Epoch 1:  val_loss = 0.0561
Epoch 2:  val_loss = 0.8912  ← spike!
Epoch 3:  val_loss = 0.0523
Epoch 4:  val_loss = 0.9134  ← spike!
...
24% of epochs have abnormal spikes (>0.3)
```

### 问题代码

```python
# train_moe.py: validate() 函数

@torch.no_grad()
def validate(self) -> dict:
    self.model.eval()
    metrics = defaultdict(list)

    for batch in self.val_loader:
        # ... forward pass ...
        loss_dict = self.criterion(output, gt_angles, gt_labels)

        # 问题在这里：把每个 batch 的 loss 直接 append
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics[k].append(v.item())  # batch loss = mean(directional samples in batch)

    # 最后对 batch 做等权平均
    return {k: np.mean(v) for k, v in metrics.items()}  # ← 问题：按 batch 等权
```

### 问题分析

**MaskedExpertLoss 的输出**：
```python
# MaskedExpertLoss.forward()
total_count = count_1front + count_2front + count_4front
if total_count > 0:
    total_loss = total_loss / total_count  # ← 该 batch 中 directional samples 的平均
```

**叠加效果**：

假设 validation set 有 10 个 batch：
- Batch 1: 20 个 directional samples，loss = 0.05
- Batch 2: 1 个 directional sample（恰好是 outlier），loss = 8.8
- ...
- Batch 10: 15 个 directional samples，loss = 0.06

**当前方式（按 batch 等权）**：
```
val_loss = mean([0.05, 8.8, 0.05, 0.06, ...]) ≈ 0.9+
```

**正确方式（按样本数加权）**：
```
val_loss = (0.05*20 + 8.8*1 + ...) / (20 + 1 + ...) ≈ 0.1
```

### 修复方案

```python
@torch.no_grad()
def validate(self) -> dict:
    self.model.eval()

    # 改用加权累加
    total_loss = 0.0
    total_count = 0
    loss_1front_sum, count_1front_total = 0.0, 0
    loss_2front_sum, count_2front_total = 0.0, 0
    loss_4front_sum, count_4front_total = 0.0, 0

    for batch in self.val_loader:
        # ... forward pass ...
        loss_dict = self.criterion(output, gt_angles, gt_labels)

        # 累加加权 loss
        batch_count = loss_dict['count_1front'] + loss_dict['count_2front'] + loss_dict['count_4front']
        if batch_count > 0:
            total_loss += loss_dict['loss'].item() * batch_count
            total_count += batch_count

        # 分类别累加
        if loss_dict['count_1front'] > 0:
            loss_1front_sum += loss_dict['loss_1front'].item() * loss_dict['count_1front']
            count_1front_total += loss_dict['count_1front']
        # 类似处理 2front, 4front...

    # 计算加权平均
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    avg_loss_1front = loss_1front_sum / count_1front_total if count_1front_total > 0 else 0.0
    # ...

    return {
        'loss': avg_loss,
        'loss_1front': avg_loss_1front,
        # ...
    }
```

### 为什么这个 bug 会放大 spike

1. **Validation set 类别分布不均**：
   - 总共 148 samples
   - 只有 37 + 35 + 18 = 90 个 directional samples
   - 分布在多个 batch 中，每个 batch 的 directional sample 数量不同

2. **某些 batch 可能 directional sample 很少**：
   - 假设 batch_size = 32，某个 batch 可能只有 1-2 个 directional samples
   - 如果这 1-2 个恰好是 outlier（高 κ + 错误预测），会得到 loss ≈ 8-10

3. **等权平均放大 outlier batch 的影响**：
   - 1 个 outlier batch 的贡献 = 1/10（假设 10 个 batch）
   - 但按样本数加权的话，1 个 outlier sample 的贡献应该只有 1/90

### 影响评估

- **Best checkpoint 选择错误**：如果 spike epoch 恰好选了错误的 checkpoint
- **Wandb 曲线误导**：看起来模型在剧烈震荡，实际上可能是稳定的
- **Early stopping 失效**：可能因为虚假 spike 导致过早停止

### 优先级

**高优先级**：这是一个工程 bug，修复后可以：
1. 消除大部分虚假 spike
2. 正确选择 best checkpoint
3. 更准确地评估模型性能

---

## 总结

| Issue | 问题 | 根因 | 优先级 | 难度 |
|-------|-----|------|-------|-----|
| #1 | 1-front kappa collapse | von Mises 梯度方向问题 | 高 | 难 |
| #2 | Val loss spike | 统计方式 bug | 高 | 易 |

**建议修复顺序**：
1. 先修复 Issue #2（工程 bug，容易修复）✅ 已完成
2. 修复 Issue #3（工程 bug，容易修复）✅ 已完成
3. 再分析 Issue #1 是否仍然存在

---
---

## Issue #3: 验证集随机点云采样 (已修复)

### TL;DR

验证集每次采样点云时使用随机采样，导致同一模型在不同 epoch 看到不同的点云子采样，增加了 loss 噪声。

### 问题代码

```python
# datasets/moe_dataset.py:40-47

def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Sample or pad points to fixed size."""
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)  # ← 随机！
    else:
        indices = np.random.choice(N, num_points, replace=True)
    return points[indices]
```

### 修复方案 (已实施)

```python
# datasets/moe_dataset.py

def sample_points(points: np.ndarray, num_points: int, deterministic: bool = False,
                  seed: int = 0) -> np.ndarray:
    """Sample or pad points to fixed size.

    Args:
        deterministic: If True, use fixed seed for reproducible sampling (for val/test)
        seed: Seed to use when deterministic=True (e.g., sample index)
    """
    N = points.shape[0]

    if deterministic:
        rng = np.random.RandomState(seed)
        if N >= num_points:
            indices = rng.choice(N, num_points, replace=False)
        else:
            indices = rng.choice(N, num_points, replace=True)
    else:
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=False)
        else:
            indices = np.random.choice(N, num_points, replace=True)

    return points[indices]

# __getitem__ 中使用:
use_deterministic = not self.augment  # val/test 不做增强
points = sample_points(points, self.num_points,
                       deterministic=use_deterministic,
                       seed=sample_idx)
```

---

## Issue #4: 有效样本数少 (设计问题)

### 描述

MaskedExpertLoss 只对 1-front/2-front/4-front 类别计算方向损失，Rot-sym 和 No-front 不参与 loss 计算。

验证集分布：
- 总样本数: 148 (或 319，取决于是否用旋转增强)
- 有效 directional 样本: 37 (1-front) + 35 (2-front) + 18 (4-front) = 90

### 影响

1. 有效样本数少 → 统计方差大
2. 每个 batch 的 directional 样本数量不均匀
3. 配合 Issue #2 的 bug，某些只有 1-2 个 directional 样本的 batch 会导致 spike

### 建议

1. 已通过修复 Issue #2 和 #3 缓解
2. 可以考虑增加标注数据
3. 使用更大的验证集

---

## 修复验证

运行以下命令验证修复效果：

```bash
# 训练新模型，观察 val loss 是否还有剧烈 spike
python train_moe.py \
    --exp_name "2-step_MoE_v3_fixed" \
    --classifier_checkpoint "checkpoints/SymClassifier_20251216_035345/best.pth" \
    --epochs 20 \
    --freeze_classifier \
    --wandb

# 预期结果:
# 1. Val loss 应该更稳定，不再有 0.05 → 0.9 的跳变
# 2. 1-front loss 可能仍然是 1.8379（这是 Issue #1，未修复）
```

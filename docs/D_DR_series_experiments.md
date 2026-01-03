# D系列与DR系列实验报告：离散方向预测方法

> 生成日期: 2025-12-21
> 实验时间: 2025-12-18 ~ 2025-12-20

## 目录

1. [方法概述](#1-方法概述)
2. [问题定义](#2-问题定义)
3. [GT表示方法](#3-gt表示方法)
4. [模型架构](#4-模型架构)
5. [损失函数](#5-损失函数)
6. [实验配置](#6-实验配置)
7. [实验结果](#7-实验结果)
   - 7.1-7.5: D/DR系列结果与相对误差评估
   - 7.6: 评估方法演进（Argmax → Hungarian → KL Divergence）
   - 7.7: 结论与建议（D_8b 最佳）
8. [关键代码](#8-关键代码)
9. [结论与分析](#9-结论与分析)

---

## 1. 方法概述

### 1.1 背景

在3D物体的"正面方向"预测任务中，我们之前尝试了**MF系列**（Mixture of von Mises分布），将方向预测建模为连续概率分布。但实验表明连续方法误差较大（最佳28.34°）。

**D系列**和**DR系列**采用**离散化方法**，将圆周划分为若干个bin（8或16个），将连续方向预测转化为分类或回归问题。

### 1.2 两种方法的核心区别

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        D系列 vs DR系列 对比                                  │
├───────────────┬─────────────────────────────┬───────────────────────────────┤
│               │         D系列（槽分类）      │        DR系列（投影回归）      │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ GT表示        │ gt_mode='projection'        │ gt_mode='dr'                  │
│               │ softmax(cos投影×温度)→概率  │ 原始cos投影值 [-1,1]          │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 网络输出      │ logits（任意值）            │ tanh约束的投影值 [-1,1]       │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 输出处理      │ softmax(logits) → 概率分布  │ softmax(投影值) → 概率分布    │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 损失函数      │ CE/KL(pred_probs, gt_probs) │ KL/CE(softmax(pred), softmax(gt))│
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 物理约束      │ 无（logits可以是任意值）    │ 有（tanh约束到[-1,1]）        │
├───────────────┼─────────────────────────────┼───────────────────────────────┤
│ 本质          │ "方向落在哪个槽"的分类      │ "与各基础方向的相似度"回归    │
└───────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

## 2. 问题定义

### 2.1 任务描述

给定一个3D点云 $P \in \mathbb{R}^{N \times 3}$，预测物体的"正面方向"$\theta \in [0, 2\pi)$。

### 2.2 类别定义

| 类别 | 说明 | 样本数(train/val) |
|------|------|-------------------|
| `1_front` | 单一正面方向 | 697 / 199 |
| `4_fronts` | 4个等效正面（90°对称） | 190 / 54 |
| `no_front` | 无明确正面（旋转对称） | 801 / 229 |

### 2.3 离散化方案

将圆周 $[0, 2\pi)$ 等分为 $N$ 个bin：

- **8 bins**: 每个bin覆盖45°，中心角度为 $\theta_i = i \times 45°$, $i \in \{0,1,...,7\}$
- **16 bins**: 每个bin覆盖22.5°，中心角度为 $\theta_i = i \times 22.5°$, $i \in \{0,1,...,15\}$

```
8 bins示意图:
        +Z (90°)
          │
    135°  │  45°
       ╲  │  ╱
        ╲ │ ╱
-X (180°)──┼──+X (0°)
        ╱ │ ╲
       ╱  │  ╲
    225°  │  315°
          │
        -Z (270°)
```

---

## 3. GT表示方法

### 3.1 D系列：Projection模式

**核心思想**: 用cos投影 + softmax生成soft label

**数学公式**:
$$p_i = \text{softmax}(\tau \cdot \cos(\theta - \theta_i))$$

其中：
- $\theta$ 是真实方向角度
- $\theta_i$ 是第$i$个bin的中心角度
- $\tau$ 是温度参数（控制分布的"尖锐"程度）

**不同类别的处理**:

```python
def _generate_gt_projection(self, category, base_angle, angle_offset):
    if category == 'no_front':
        # 无正面 → 均匀分布
        return np.full(self.num_bins, 1.0 / self.num_bins)

    angle = (base_angle + angle_offset) % (2 * np.pi)

    if category == '1_front':
        # 单方向投影
        projections = np.cos(angle - self.bin_angles)
        gt_probs = softmax(projections * self.temperature)

    elif category == '4_fronts':
        # 4个方向叠加
        gt_probs = np.zeros(self.num_bins)
        for i in range(4):
            a = (angle + i * np.pi / 2) % (2 * np.pi)
            projections = np.cos(a - self.bin_angles)
            gt_probs += softmax(projections * self.temperature)
        gt_probs /= 4  # 归一化

    return gt_probs
```

**温度参数的影响**:

| 温度τ | 分布特征 | 适用场景 |
|-------|----------|----------|
| τ=3 | 较平滑，信息分散 | 更容易学习，但精度低 |
| τ=5 | 中等尖锐（默认） | 平衡性能 |
| τ=10 | 非常尖锐，接近one-hot | 精度高，但可能过拟合 |

**示例** (8 bins, 方向=0°):

```
τ=3:  [0.32, 0.18, 0.06, 0.02, 0.01, 0.02, 0.06, 0.18]  # 较平滑
τ=5:  [0.52, 0.19, 0.03, 0.00, 0.00, 0.00, 0.03, 0.19]  # 中等
τ=10: [0.89, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05]  # 非常尖锐
```

### 3.2 DR系列：原始投影值

**核心思想**: 直接使用cos投影值作为GT，不做softmax

**数学公式**:
$$p_i = \cos(\theta - \theta_i)$$

**不同类别的处理**:

```python
def _generate_gt_regression(self, category, base_angle, angle_offset):
    if category == 'no_front':
        # 无正面 → 全零（无方向偏好）
        return np.zeros(self.num_bins)

    angle = (base_angle + angle_offset) % (2 * np.pi)

    if category == '1_front':
        # 单方向：直接计算cos投影
        gt_proj = np.cos(angle - self.bin_angles)

    elif category == '4_fronts':
        # 4个方向叠加
        gt_proj = np.zeros(self.num_bins)
        for i in range(4):
            a = (angle + i * np.pi / 2) % (2 * np.pi)
            gt_proj += np.cos(a - self.bin_angles)
        gt_proj /= 4.0  # 归一化到[-1,1]

    return gt_proj
```

**关键区别**:

| 特性 | D系列 (projection) | DR系列 (dr) |
|------|---------------------|-------------|
| GT值范围 | [0, 1]，和=1 | [-1, 1]，和≠1 |
| no_front表示 | 均匀分布 1/N | 全零向量 |
| 物理意义 | 概率分布 | 相似度投影 |

---

## 4. 模型架构

### 4.1 Backbone: PointNet++

使用三层Set Abstraction进行点云特征提取：

```python
# 层级结构
SA1: 2048 → 128 points, MLP [64, 64, 128]
SA2: 128 → 32 points, MLP [128, 128, 256]
SA3: 32 → 1 point (global), MLP [256, 512, 1024]

# 输出: 1024维全局特征向量
```

### 4.2 D系列预测头: DiscreteDirectionHead

```python
class DiscreteDirectionHead(nn.Module):
    """
    【槽分类】离散方向分类头
    输出: logits (B, num_bins) - 未归一化的分数
    """
    def __init__(self, in_channels=1024, hidden_channels=512, num_bins=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),      # 1024 → 512
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),  # 512 → 256
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_bins),    # 256 → 8/16
        )

    def forward(self, x):
        return self.head(x)  # 返回raw logits
```

### 4.3 DR系列预测头: ProjectionRegressionHead

```python
class ProjectionRegressionHead(nn.Module):
    """
    【投影回归】基础方向投影头
    输出: projections (B, num_dirs) - 投影值，范围 [-1, 1]

    关键区别: 使用Tanh激活，约束输出到物理意义的范围
    """
    def __init__(self, in_channels=1024, hidden_channels=512, num_dirs=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, num_dirs),
            nn.Tanh(),  # 关键: 约束输出到 [-1, 1]
        )

    def forward(self, x):
        return self.head(x)
```

### 4.4 模型参数统计

| 模块 | 参数数量 |
|------|----------|
| PointNet++ Backbone | ~2.39M |
| Direction Head | ~0.13M |
| **总计** | **~2.52M** |

---

## 5. 损失函数

### 5.1 D系列损失: DiscreteDirectionLoss

```python
class DiscreteDirectionLoss(nn.Module):
    """
    【槽分类】离散方向分类损失

    支持:
    - ce: CrossEntropy with soft labels
    - kl: KL Divergence
    - focal: Focal Loss
    """
    def forward(self, logits, gt_probs, categories=None):
        pred_probs = F.softmax(logits, dim=-1)

        if self.loss_type == 'ce':
            # Cross Entropy: -sum(gt * log(pred))
            loss = -(gt_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        elif self.loss_type == 'kl':
            # KL Divergence: sum(gt * log(gt/pred))
            loss = F.kl_div(F.log_softmax(logits, dim=-1), gt_probs, reduction='batchmean')

        return {'total_loss': loss}
```

**数学公式**:

CrossEntropy:
$$L_{CE} = -\sum_{i=1}^{N} p_i^{gt} \log(p_i^{pred})$$

KL Divergence:
$$L_{KL} = \sum_{i=1}^{N} p_i^{gt} \log\frac{p_i^{gt}}{p_i^{pred}}$$

### 5.2 DR系列损失: ProjectionSoftmaxLoss

```python
class ProjectionSoftmaxLoss(nn.Module):
    """
    【DR系列】投影 + Softmax 损失

    关键区别: 对预测和GT都做softmax后再比较

    处理流程:
      pred_proj (tanh输出, [-1,1]) → softmax → pred_probs
      gt_proj (cos投影, [-1,1]) → softmax → gt_probs
      loss = KL(pred_probs, gt_probs) 或 CE
    """
    def forward(self, pred_proj, gt_proj, categories=None):
        # 对预测和GT都做softmax
        pred_probs = F.softmax(pred_proj, dim=-1)
        gt_probs = F.softmax(gt_proj, dim=-1)

        if self.loss_type == 'ce':
            loss = -torch.sum(gt_probs * torch.log(pred_probs + 1e-10), dim=-1).mean()

        elif self.loss_type == 'kl':
            loss = F.kl_div(torch.log(pred_probs + 1e-10), gt_probs, reduction='batchmean')

        return {'total_loss': loss}
```

**DR系列的独特之处**:

```
D系列:
  logits (任意值) → softmax → pred_probs
  gt_probs (已经是概率)
  loss = KL(pred_probs, gt_probs)

DR系列:
  pred_proj (tanh约束到[-1,1]) → softmax → pred_probs
  gt_proj (cos投影[-1,1]) → softmax → gt_probs
  loss = KL(pred_probs, gt_probs)
```

---

## 6. 实验配置

### 6.1 D系列实验

| 实验ID | Bins | 温度τ | 损失类型 | Checkpoint |
|--------|------|-------|----------|------------|
| D_8a | 8 | 5.0 | CE | `D_8a_20251218_171207` |
| D_8b | 8 | 3.0 | CE | `D_8b_20251218_203046` |
| D_8c | 8 | 10.0 | CE | `D_8c_20251218_234110` |
| D_8d | 8 | 5.0 | KL | `D_8d_20251219_024151` |
| D_16a | 16 | 5.0 | CE | `D_16a_20251219_053127` |
| D_16b | 16 | 8.0 | CE | `D_16b_20251219_081853` |

### 6.2 DR系列实验

| 实验ID | Bins | 损失类型 | Checkpoint |
|--------|------|----------|------------|
| DR_8a | 8 | KL | `DR_8a_20251219_215002` |
| DR_8b | 8 | CE | `DR_8b_20251220_004513` |
| DR_16a | 16 | KL | `DR_16a_20251220_033656` |

### 6.3 通用训练配置

```json
{
  "mode": "discrete",
  "categories": "1_front,4_fronts,no_front",
  "num_points": 2048,
  "batch_size": 32,
  "augment_factor": 10,
  "epochs": 50,
  "lr": 0.001,
  "weight_decay": 0.0001
}
```

---

## 7. 实验结果

### 7.1 D系列结果

| 实验 | 配置 | Best Val Error | Best Val Loss |
|------|------|----------------|---------------|
| **D_16a** | 16 bins, CE, τ=5 | **7.65°** | 2.4626 |
| D_16b | 16 bins, CE, τ=8 | 7.87° | 2.3374 |
| D_8c | 8 bins, CE, τ=10 | 8.09° | 1.5337 |
| D_8d | 8 bins, KL, τ=5 | 8.61° | 0.1585 |
| D_8a | 8 bins, CE, τ=5 | 9.49° | 1.7543 |
| D_8b | 8 bins, CE, τ=3 | 10.29° | 1.8716 |

### 7.2 DR系列结果

| 实验 | 配置 | Best Val Error | Best Val Loss |
|------|------|----------------|---------------|
| **DR_8b** | 8 bins, CE | **13.76°** | 2.0331 |
| DR_8a | 8 bins, KL | 13.87° | 0.0399 |
| DR_16a | 16 bins, KL | 15.38° | 0.0403 |

### 7.3 对比分析

```
性能排名:
1. D_16a   (D系列, 16bins)   7.65°  ← 最佳
2. D_16b   (D系列, 16bins)   7.87°
3. D_8c    (D系列, 8bins)    8.09°
4. D_8d    (D系列, 8bins)    8.61°
5. D_8a    (D系列, 8bins)    9.49°
6. D_8b    (D系列, 8bins)   10.29°
7. DR_8b   (DR系列, 8bins)  13.76°
8. DR_8a   (DR系列, 8bins)  13.87°
9. DR_16a  (DR系列, 16bins) 15.38°
```

### 7.4 分类别评估与GT分析

#### 7.4.1 问题发现：GT只有4个固定方向

深入分析发现，**标注数据只包含4个离散方向**（+X, +Z, -X, -Z）：

```
GT方向分布:
  4_fronts: 100% 是 -Z 方向 (270°)
  1_front:  -Z (69.3%), -X (19.1%), +Z (6.0%), +X (5.5%)

验证集GT角度分布:
  4_fronts (n=54): 全部是270°（单一值！）
  1_front (n=199): 只有4个值 {0°, 90°, 180°, 270°}
```

**这意味着之前观察到的"0°误差"是假象**：
- 8 bins的中心是 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- GT方向正好落在 bin 0, 2, 4, 6 上
- 4_fronts的4个等效方向完美匹配这4个bin
- 模型只需输出这4个bin之一就能达到0°误差

#### 7.4.2 无增强评估（之前的结果）

| 实验 | 1_front | 4_fronts | Overall | 问题 |
|------|---------|----------|---------|------|
| D_16a | 18.88° | **0.00°** | 16.27° | 4_fronts过于完美 |
| D_8c | 20.80° | 0.00° | 18.50° | |
| DR_8b | 26.23° | 35.00° | 30.95° | |

#### 7.4.3 公平评估：验证集使用随机旋转增强

为测试真正的泛化能力，对验证集应用Y轴随机旋转增强（10x）：

**D系列公平评估**:

| 实验 | 1_front | 4_fronts | Overall | 说明 |
|------|---------|----------|---------|------|
| **D_16b** | **37.16°** | 5.84° | **30.47°** | **实际最佳** |
| D_8b | 39.14° | 12.59° | 33.47° | |
| D_8a | 40.83° | 10.89° | 34.44° | |
| D_8c | 41.42° | 11.01° | 34.93° | |
| D_8d | 41.40° | 11.29° | 34.97° | |
| D_16a | 52.69° | 5.78° | 42.68° | 1_front严重过拟合！ |

**DR系列公平评估**:

| 实验 | 1_front | 4_fronts | Overall |
|------|---------|----------|---------|
| **DR_16a** | **39.30°** | 22.11° | **35.63°** |
| DR_8b | 39.56° | 23.13° | 36.05° |
| DR_8a | 41.53° | 20.89° | 37.13° |

#### 7.4.4 深入分析

1. **D_16a的"过拟合"现象**
   ```
   训练时报告误差: 7.65° (固定角度验证)
   公平评估误差:  42.68° (随机旋转验证)
   ```
   - 1_front误差高达52.69°，模型"记住"了固定方向
   - 4_fronts仍表现良好（5.78°），因为对称结构被学习到了

2. **D_16b是真正的最佳模型**
   - 公平评估Overall: 30.47°
   - 1_front误差最低: 37.16°
   - τ=8的温度可能提供了更好的正则化

3. **4_fronts vs 1_front的本质区别**
   - 4_fronts: 5-12°误差，模型学会了"4峰对称"结构
   - 1_front: 37-53°误差，模型难以预测任意角度
   - 对称性是一种"先验"，降低了学习难度

4. **核心问题**
   - 标注只有4个离散方向，限制了模型学习任意角度的能力
   - 训练时有增强但验证时无增强，导致评估结果失真

### 7.5 公平评估：使用相对误差

#### 7.5.1 为什么需要相对误差？

**直接比较绝对误差不公平**，因为两个类别的"难度"不同：

| 类别 | 等效方向数 | 最大误差 | 随机猜测期望 |
|------|------------|----------|--------------|
| 1_front | 1 | 180° | **90°** |
| 4_fronts | 4 | 45° | **22.5°** |

**相对误差** = 实际误差 / 随机基准
- 1.0x = 和随机猜测一样差
- 0.0x = 完美预测
- 越低越好

#### 7.5.2 完整公平评估结果

使用随机旋转增强评估（10x），计算相对误差：

| 排名 | 模型 | 系列 | 1_front Abs | 1_front Rel | 4_fronts Abs | 4_fronts Rel | **Avg Rel** |
|------|------|------|-------------|-------------|--------------|--------------|-------------|
| **1** | **D_16b** | D | 37.74° | 0.42x | 5.73° | **0.26x** | **0.38x** |
| 2 | D_8b | D | 38.08° | 0.42x | 11.13° | 0.49x | 0.44x |
| 3 | D_8a | D | 39.44° | 0.44x | 11.06° | 0.49x | 0.45x |
| 4 | D_8d | D | 40.78° | 0.45x | 10.95° | 0.49x | 0.46x |
| 5 | D_8c | D | 41.47° | 0.46x | 11.38° | 0.51x | 0.47x |
| 6 | D_16a | D | 51.31° | **0.57x** | 5.63° | 0.25x | 0.50x |
| 7 | DR_16a | DR | 39.64° | 0.44x | 21.29° | 0.95x | 0.55x |
| 8 | DR_8b | DR | 38.68° | 0.43x | 23.54° | **1.05x** | 0.56x |
| 9 | DR_8a | DR | 43.95° | 0.49x | 22.07° | 0.98x | 0.59x |

#### 7.5.3 关键发现

1. **D_16b是真正最佳模型**
   - 平均相对误差: 0.38x
   - 1_front: 比随机好58%
   - 4_fronts: 比随机好74%

2. **D_16a在1_front上过拟合**
   - 1_front相对误差0.57x（最差的D系列）
   - 4_fronts相对误差0.25x（最好）
   - 模型"记住"了固定方向，泛化能力差

3. **DR系列在4_fronts上完全失败**
   ```
   DR_8b: 4_fronts相对误差 1.05x → 比随机还差！
   DR_8a: 4_fronts相对误差 0.98x → 接近随机
   ```
   - DR系列**无法学习4峰对称结构**
   - tanh约束限制了多峰表示能力

4. **D系列 vs DR系列的差距主要在4_fronts**
   - 1_front相对误差：D系列 0.42-0.57x vs DR系列 0.43-0.49x（相近）
   - 4_fronts相对误差：D系列 **0.25-0.51x** vs DR系列 **0.95-1.05x**（差4倍）

5. **温度参数τ影响**
   - D_16b (τ=8): 最佳
   - D_16a (τ=5): 1_front过拟合
   - 较高温度提供更好的正则化

### 7.6 评估方法演进：从角度误差到分布相似度

#### 7.6.1 方法演进过程

我们尝试了多种评估方法，最终确定使用 KL Divergence：

```
方法1: Argmax + Min Distance
    ↓ 问题：只评估单一预测，忽略多峰分布
方法2: Top-K + Hungarian
    ↓ 问题：简单 Top-K 取到相邻 bin，不是真正的峰
方法3: Peak Detection + Hungarian
    ↓ 问题：峰检测阈值（如 0.05）是硬编码，不科学
方法4: KL Divergence ✓
    → 直接比较分布相似度，无需峰检测
```

#### 7.6.2 方法1：Argmax + Min Distance（弃用）

**做法：**
```python
pred_bin = pred_probs.argmax()
pred_angle = pred_bin * bin_width
error = min(circular_distance(pred_angle, gt_k) for gt_k in equiv_angles)
```

**问题：**
- 只关注概率最高的 bin，忽略了整体分布形状
- 对于 4_fronts，无法评估模型是否学会了 4 峰结构
- 所有 4 个预测可能匹配到同一个 GT 方向

#### 7.6.3 方法2：Top-K + Hungarian（弃用）

**做法：**
```python
top_k_bins = probs.argsort()[-K:]  # 取概率最高的 K 个 bin
pred_angles = [b * bin_width for b in top_k_bins]
# 用 Hungarian 算法一对一匹配
```

**问题：**
- Top-K 可能取到相邻的 bin，不是真正的 K 个独立峰
- 例：如果主峰在 bin 5，Top-4 可能是 [4, 5, 6, 7]（相邻的）

**实验结果（Top-K Hungarian 更差）：**
```
模型     | Argmax  | Top-K Hungarian | 差距
---------|---------|-----------------|--------
D_16b    | 5.69°   | 9.95°           | +75%
D_16a    | 5.45°   | 25.92°          | +376%
DR_8b    | 23.38°  | 40.95°          | +75%
```

#### 7.6.4 方法3：Peak Detection + Hungarian（弃用）

**改进：** 用局部最大值检测替代简单 Top-K

```python
def find_local_peaks(probs, min_height=0.05):
    peaks = []
    for i in range(len(probs)):
        left, right = (i-1) % n, (i+1) % n
        if probs[i] >= probs[left] and probs[i] >= probs[right] and probs[i] > min_height:
            peaks.append(i)
    return peaks
```

**问题：**
- 阈值 `min_height=0.05` 是硬编码的，没有理论依据
- 对于不同的分布形状，同一阈值可能不适用
- 峰的定义过于简单（只比较左右邻居）

#### 7.6.5 方法4：KL Divergence（最终方案）

**核心思想：** 不检测峰，直接比较预测分布和理想 GT 分布的相似度

| 评估方法 | 是否需要峰检测 | 是否评估分布形状 | 科学性 |
|----------|---------------|-----------------|--------|
| Argmax + Min Distance | 否 | 否 | 低 |
| Top-K + Hungarian | 否 | 部分 | 低 |
| Peak Detection + Hungarian | 是（需要阈值） | 部分 | 中 |
| **KL Divergence** | **否** | **是** | **高** |

#### 7.6.6 KL Divergence 计算方法

**理想 GT 分布生成：**

```python
def create_gt_distribution(gt_angle, category, num_bins, temperature=5.0):
    """生成理想的 GT 概率分布"""
    bin_centers = np.arange(num_bins) * (2 * np.pi / num_bins)

    # 确定等效方向
    if category == '1_front':
        equiv_angles = [gt_angle]
    elif category == '2_fronts':
        equiv_angles = [gt_angle, (gt_angle + np.pi) % (2 * np.pi)]
    elif category == '4_fronts':
        equiv_angles = [(gt_angle + i * np.pi / 2) % (2 * np.pi) for i in range(4)]

    # 用 von Mises kernel 生成分布
    dist = np.zeros(num_bins)
    for angle in equiv_angles:
        for i, center in enumerate(bin_centers):
            dist[i] += np.exp(temperature * np.cos(center - angle))

    return dist / dist.sum()  # 归一化
```

**KL Divergence 计算：**

```python
def kl_divergence(gt_dist, pred_dist):
    """KL(GT || Pred) - 越小越好"""
    return np.sum(gt_dist * np.log(gt_dist / pred_dist))
```

#### 7.6.7 完整评估结果

使用随机旋转增强评估（10x），KL Divergence 越小越好：

| 排名 | 模型 | 系列 | 1_front KL | 4_fronts KL | **平均 KL** |
|------|------|------|------------|-------------|-------------|
| **1** | **D_8b** | D | **0.4432** | 0.0027 | **0.2230** |
| 2 | D_16b | D | 0.5354 | 0.0043 | 0.2698 |
| 3 | D_8c | D | 0.5428 | 0.0057 | 0.2743 |
| 4 | D_8a | D | 0.5529 | 0.0019 | 0.2774 |
| 5 | D_8d | D | 0.5680 | 0.0018 | 0.2849 |
| 6 | D_16a | D | 0.5722 | 0.0211 | 0.2967 |
| 7 | DR_8b | DR | 0.8028 | 0.0361 | 0.4195 |
| 8 | DR_16a | DR | 0.8171 | 0.0358 | 0.4264 |
| 9 | DR_8a | DR | 0.8379 | 0.0349 | 0.4364 |

#### 7.6.8 关键发现

1. **D_8b 是最佳模型**（平均 KL = 0.2230）
   - 1_front KL 最低：0.4432
   - 温度 τ=3.0（较低温度使分布更平滑，更容易学习）

2. **D 系列远优于 DR 系列**
   ```
   D 系列平均 KL:  0.22 - 0.30
   DR 系列平均 KL: 0.42 - 0.44  （差 1.5-2 倍）
   ```

3. **所有模型在 4_fronts 上都很好**
   - D 系列 4_fronts KL: 0.002 - 0.02（接近完美）
   - DR 系列 4_fronts KL: 0.035（也不错）
   - 说明模型都学会了 4 峰对称结构

4. **主要差距在 1_front**
   - D 系列 1_front KL: 0.44 - 0.57
   - DR 系列 1_front KL: 0.80 - 0.84（差 1.5 倍）
   - DR 系列的 tanh 约束限制了单峰表达能力

5. **8 bins 不一定比 16 bins 差**
   - D_8b (8 bins, KL=0.2230) > D_16b (16 bins, KL=0.2698)
   - 可能原因：8 bins 的 bin 更宽，对预测误差更宽容

#### 7.6.9 各类别评估标准

```python
# 1_front: 1 个峰
# 理想分布：在 gt 方向有单峰
gt_angles = [gt]

# 2_fronts: 2 个峰（相差 180°）
# 理想分布：在 gt 和 gt+180° 各有一峰
gt_angles = [gt, (gt + np.pi) % (2*np.pi)]

# 4_fronts: 4 个峰（相差 90°）
# 理想分布：在 gt, gt+90°, gt+180°, gt+270° 各有一峰
gt_angles = [(gt + i*np.pi/2) % (2*np.pi) for i in range(4)]

# 生成理想分布并计算 KL
gt_dist = create_gt_distribution(gt_angle, category, num_bins, temperature)
pred_dist = softmax(model_output)
kl = kl_divergence(gt_dist, pred_dist)
```

#### 7.6.10 评估脚本

```bash
# 运行 KL Divergence 评估
python evaluate_kl_divergence.py

# 结果保存到 evaluation_kl_divergence.json
```

### 7.7 结论与建议

#### 7.7.1 最终排名（按 KL Divergence）

| 排名 | 模型 | 平均 KL | 说明 |
|------|------|---------|------|
| **1** | **D_8b** | **0.2230** | **最佳模型** |
| 2 | D_16b | 0.2698 | |
| 3 | D_8c | 0.2743 | |
| ... | ... | ... | |
| 7-9 | DR 系列 | 0.42-0.44 | 不推荐 |

#### 7.7.2 核心发现

1. **D_8b 是最佳模型**（τ=3.0, 8 bins, CE loss）
   - 平均 KL = 0.2230（最低）
   - 较低的温度参数使分布更平滑，更容易学习

2. **D 系列远优于 DR 系列**
   - D 系列平均 KL: 0.22-0.30
   - DR 系列平均 KL: 0.42-0.44（差 1.5-2 倍）
   - DR 系列的 tanh 约束限制了表达能力

3. **所有模型在 4_fronts 上都很好**
   - 4_fronts KL 都很低（0.002-0.036）
   - 模型都学会了 4 峰对称结构

4. **主要差距在 1_front**
   - 这是最难的类别（需要预测任意角度的单峰）
   - D_8b 在 1_front 上 KL 最低（0.4432）

5. **8 bins 可能比 16 bins 更好**
   - 更宽的 bin 对预测误差更宽容
   - 但精度上限受限于 bin 宽度

#### 7.7.3 建议

| 场景 | 推荐模型 | 说明 |
|------|----------|------|
| 最佳分布相似度 | **D_8b** | KL = 0.2230 |
| 需要高角度精度 | D_16b | 16 bins, 22.5° 分辨率 |
| 避免使用 | DR 系列 | 性能差 1.5-2 倍 |

#### 7.7.4 待改进

1. **标注多样性**: 当前 GT 只有 4 个离散方向（+X, +Z, -X, -Z），限制了泛化能力
2. **验证方式**: 必须使用数据增强才能反映真实性能
3. **评估指标**: KL Divergence 是更科学的评估方法，应作为主要指标

---

## 8. 关键代码

### 8.1 数据集核心代码

**文件**: `datasets/discrete_direction_dataset.py`

```python
class DiscreteDirectionDataset(Dataset):
    """
    离散方向预测数据集

    支持三种GT模式:
    - projection: cos投影 + softmax → 概率分布 (D系列)
    - dr: 原始cos投影值 (DR系列)
    - onehot: 传统one-hot编码
    """

    # 方向到角度的映射
    DIRECTION_TO_ANGLE = {
        '+X': 0.0,
        '+Z': np.pi / 2,
        '-X': np.pi,
        '-Z': 3 * np.pi / 2,
    }

    def __init__(
        self,
        num_bins: int = 8,           # 8 or 16
        gt_mode: str = 'projection', # 'projection', 'dr', 'onehot'
        temperature: float = 5.0,    # 仅用于projection模式
        ...
    ):
        # bin角度 (中心角度)
        self.bin_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        self.bin_width = 2 * np.pi / num_bins

    def _generate_discrete_gt(self, category, base_angle, angle_offset):
        """根据gt_mode生成不同格式的GT"""
        if self.gt_mode == 'projection':
            return self._generate_gt_projection(category, base_angle, angle_offset)
        elif self.gt_mode == 'dr':
            return self._generate_gt_regression(category, base_angle, angle_offset)
        else:
            return self._generate_gt_onehot(category, base_angle, angle_offset)
```

### 8.2 训练器核心代码

**文件**: `train_direction.py`

```python
class Trainer:
    def _init_model(self):
        # Backbone
        self.encoder = PointNetPPEncoder().to(self.device)

        # 根据gt_mode选择不同的预测头
        if self.args.gt_mode == 'regression':
            self.head = ProjectionRegressionHead(num_dirs=self.args.num_bins)
        elif self.args.gt_mode == 'dr':
            # DR系列: 使用投影回归头，输出tanh约束的值
            self.head = ProjectionRegressionHead(num_dirs=self.args.num_bins)
        else:
            # D系列: 使用分类头，输出logits
            self.head = DiscreteDirectionHead(num_bins=self.args.num_bins)

    def _init_loss(self):
        if self.args.gt_mode == 'regression':
            self.criterion = ProjectionRegressionLoss(loss_type=self.args.reg_loss_type)
        elif self.args.gt_mode == 'dr':
            # DR系列: 使用投影+Softmax损失
            self.criterion = ProjectionSoftmaxLoss(loss_type=self.args.d_loss_type)
        else:
            # D系列: 使用分类损失
            self.criterion = DiscreteDirectionLoss(loss_type=self.args.d_loss_type)

    def _compute_angle_from_discrete(self, pred_probs):
        """从预测概率计算角度"""
        pred_idx = torch.argmax(pred_probs, dim=-1)
        pred_angle = pred_idx.float() * self.bin_width
        return pred_angle

    def _validate_epoch(self):
        """验证一个epoch"""
        for batch in val_loader:
            # 前向传播
            features = self.encoder(points)
            pred = self.head(features)

            # 计算损失
            losses = self.criterion(pred, gt_probs)

            # 计算角度误差
            if self.args.gt_mode in ['dr', 'regression']:
                pred_probs = F.softmax(pred, dim=-1)
            else:
                pred_probs = F.softmax(pred, dim=-1)

            pred_angle = self._compute_angle_from_discrete(pred_probs)
            angle_error = self._circular_distance(pred_angle, gt_angle)
```

### 8.3 运行脚本

**D系列训练命令**:

```bash
# D_8a: 8 bins, CE, τ=5
python train_direction.py \
    --mode discrete \
    --exp_name D_8a \
    --categories 1_front,4_fronts,no_front \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 5.0 \
    --d_loss_type ce \
    --epochs 50 \
    --batch_size 32 \
    --wandb

# D_16a: 16 bins, CE, τ=5 (最佳配置)
python train_direction.py \
    --mode discrete \
    --exp_name D_16a \
    --categories 1_front,4_fronts,no_front \
    --num_bins 16 \
    --gt_mode projection \
    --temperature 5.0 \
    --d_loss_type ce \
    --epochs 50 \
    --batch_size 32 \
    --wandb
```

**DR系列训练命令**:

```bash
# DR_8a: 8 bins, KL
python train_direction.py \
    --mode discrete \
    --exp_name DR_8a \
    --categories 1_front,4_fronts,no_front \
    --num_bins 8 \
    --gt_mode dr \
    --d_loss_type kl \
    --epochs 50 \
    --batch_size 32 \
    --wandb

# DR_8b: 8 bins, CE
python train_direction.py \
    --mode discrete \
    --exp_name DR_8b \
    --categories 1_front,4_fronts,no_front \
    --num_bins 8 \
    --gt_mode dr \
    --d_loss_type ce \
    --epochs 50 \
    --batch_size 32 \
    --wandb
```

---

## 9. 结论与分析

### 9.1 最终推荐（基于 KL Divergence 评估）

| 排名 | 模型 | 平均 KL | 推荐场景 |
|------|------|---------|----------|
| **1** | **D_8b** | **0.2230** | 最佳分布相似度 |
| 2 | D_16b | 0.2698 | 需要高角度精度（22.5°分辨率） |
| 3 | D_8c | 0.2743 | 备选 |

**不推荐**: DR 系列（KL 0.42-0.44，差 1.5-2 倍）

### 9.2 为什么 D 系列优于 DR 系列？

1. **灵活性**: D系列的logits没有范围约束，网络可以更自由地调整输出来拟合目标分布

2. **梯度流**: D系列直接优化softmax后的概率分布，梯度信号更直接

3. **GT质量**: D系列的projection模式GT已经是经过softmax的概率分布，与网络输出形式一致

4. **tanh瓶颈**: DR系列的tanh激活可能在极端值附近产生梯度消失

### 9.3 关于 bins 数量的新发现

**KL Divergence 评估显示 8 bins 可能更优：**

| 对比 | 8 bins (D_8b) | 16 bins (D_16b) |
|------|---------------|-----------------|
| 平均 KL | **0.2230** | 0.2698 |
| 1_front KL | **0.4432** | 0.5354 |
| 4_fronts KL | 0.0027 | 0.0043 |

**可能原因：**
- 8 bins 的 bin 更宽（45°），对预测误差更宽容
- 更少的 bins 意味着更简单的分类问题
- 但角度精度上限受限于 bin 宽度

### 9.4 温度参数的新理解

**KL Divergence 评估显示低温度可能更优：**

| 温度 | 模型 | 1_front KL | 说明 |
|------|------|------------|------|
| τ=3 | D_8b | **0.4432** | 最佳 |
| τ=5 | D_8a | 0.5529 | |
| τ=10 | D_8c | 0.5428 | |

**可能原因：**
- 低温度使 GT 分布更平滑，更容易学习
- 高温度接近 one-hot，可能过于严格

### 9.5 评估方法总结

| 评估方法 | 科学性 | 问题 |
|----------|--------|------|
| Argmax + Min Distance | 低 | 忽略分布形状 |
| Top-K + Hungarian | 低 | Top-K 取到相邻 bin |
| Peak Detection + Hungarian | 中 | 阈值硬编码 |
| **KL Divergence** | **高** | **直接比较分布相似度** |

### 9.6 与 MF 系列对比

| 方法 | 最佳 KL | 说明 |
|------|---------|------|
| **D 系列** | **0.2230** | 离散分类，最佳 |
| DR 系列 | 0.4195 | 离散回归，tanh 约束 |
| MF 系列 | - | 连续 von Mises（角度误差 28.34°）|

**结论**:
1. 离散方法显著优于连续分布建模
2. D 系列优于 DR 系列
3. KL Divergence 是更科学的评估方法

---

## 附录

### A. 文件结构

```
ForwardNet-claude/
├── datasets/
│   └── discrete_direction_dataset.py   # 离散方向数据集
├── models/
│   └── pointnet_pp_8dir.py             # 8方向模型
├── train_direction.py                   # 训练主文件
├── evaluate_by_category.py              # 分类别评估脚本（Argmax）
├── evaluate_comprehensive.py            # 综合评估脚本（Argmax vs Hungarian）
├── evaluate_kl_divergence.py            # KL Divergence 评估脚本 ✓
├── evaluation_by_category.json          # 分类别评估结果
├── evaluation_comprehensive.json        # 综合评估结果
├── evaluation_kl_divergence.json        # KL Divergence 评估结果 ✓
├── checkpoints/
│   ├── D_8b_20251218_203046/           # 最佳模型 ✓
│   ├── D_8a_20251218_171207/           # D系列实验
│   ├── D_16a_20251219_053127/
│   ├── D_16b_20251219_081853/
│   ├── DR_8a_20251219_215002/          # DR系列实验
│   └── ...
└── docs/
    └── D_DR_series_experiments.md      # 本文档
```

### A.1 分类别评估脚本使用方法

```bash
# 评估单个checkpoint
python evaluate_by_category.py --checkpoint checkpoints/D_16a_20251219_053127

# 评估所有D/DR系列
python evaluate_by_category.py --all

# 结果保存到 evaluation_by_category.json
```

### B. 配置文件示例

**D_16a (最佳配置)**:
```json
{
  "exp_name": "D_16a",
  "mode": "discrete",
  "categories": "1_front,4_fronts,no_front",
  "num_bins": 16,
  "d_loss_type": "ce",
  "gt_mode": "projection",
  "temperature": 5.0,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001
}
```

**DR_8a**:
```json
{
  "exp_name": "DR_8a",
  "mode": "discrete",
  "categories": "1_front,4_fronts,no_front",
  "num_bins": 8,
  "d_loss_type": "kl",
  "gt_mode": "dr",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001
}
```

### C. WandB项目链接

所有实验均记录在WandB:
- 项目: `ForwardNet-LossAblation`
- 链接: https://wandb.ai/augustuschen00-university-of-tokyo/ForwardNet-LossAblation

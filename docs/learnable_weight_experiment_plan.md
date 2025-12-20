处理# 可学习权重 vs 固定权重 实验计划

**日期**: 2025-12-16
**目的**: 对比可学习权重方案与固定权重(Ground Truth)方案的效果

---

## 实验背景

当前 Single Front 实验使用**固定权重** `w = 0.25` (每个峰权重相同)。
本实验将对比两种方案:
- **方案1 (Baseline)**: 固定权重 = 0.25
- **方案2 (Learnable)**: 权重通过网络学习，使用 softmax 归一化

---

## 数据集设计 (多类别)

本实验使用三种对称类型的数据:

| 类别 | 中文名 | GT配置 | 说明 |
|------|--------|--------|------|
| **1_front** | 1个正面 | 4峰同向, κ=(10,10,10,10), w=(0.25,0.25,0.25,0.25) | 有明确方向 |
| **4_fronts** | 4个正面 | 4峰90°间隔, κ=(10,10,10,10), w=(0.25,0.25,0.25,0.25) | 4个等价方向 |
| **no_front** | 无正面 | 4峰任意, κ=(0,0,0,0), w=(0.25,0.25,0.25,0.25) | 无方向(均匀分布) |

### GT生成规则

```python
# 1个正面: 4峰都指向同一方向
gt_mu = [θ, θ, θ, θ]  # 4个相同角度
gt_kappa = [10, 10, 10, 10]

# 4个正面: 4峰间隔90°
gt_mu = [θ, θ+90°, θ+180°, θ+270°]
gt_kappa = [10, 10, 10, 10]

# 无正面: κ=0 表示均匀分布，mu无意义
gt_mu = [any, any, any, any]  # 不重要
gt_kappa = [0, 0, 0, 0]  # 关键：κ=0
```

### 权重可学习的潜在优势

| 场景 | 固定权重问题 | 可学习权重优势 |
|------|-------------|---------------|
| 1_front | 4峰同向但等权重，KL可能不够精确 | 可以学到一个峰主导，更集中 |
| 4_fronts | 固定等权重是最优的 | 应该学到 ≈0.25 |
| no_front | κ=0时权重无意义 | 可能学到随机分布 |

---

## 方案设计

### 方案1: Ground Truth 固定权重 (现有方案)

```
模型输出: mu (4, 2), kappa (4,)
权重: 固定 [0.25, 0.25, 0.25, 0.25]

Loss = λ_KL * KL_div + λ_κ * kappa_loss + λ_μ * mu_loss
```

**特点**:
- 所有峰等权重贡献
- 对于单正面任务，GT是4峰同方向 (κ=10, 10, 10, 10)
- 网络只需学习方向和置信度

### 方案2: 可学习权重

```
模型输出: mu (4, 2), kappa (4,), weights (4,)
weights = softmax(fc_weights(features))  # 和为1

Loss = λ_KL * KL_div + λ_κ * kappa_loss + λ_μ * mu_loss + λ_w * weight_loss
```

**Weight Loss 设计** (可选):
```python
# 方案A: 鼓励稀疏 (让一个峰主导)
entropy_loss = -sum(w * log(w))  # 最小化熵 → 稀疏

# 方案B: GT监督 (对于单正面，GT权重应该是 [0.25, 0.25, 0.25, 0.25] 或者 [1, 0, 0, 0])
weight_loss = MSE(pred_weights, gt_weights)

# 方案C: 无监督 (让网络自由学习)
weight_loss = 0
```

---

## 实验配置

### 公共设置
| 参数 | 值 |
|------|-----|
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| 数据集 | Single Front (1个正面) |

### 实验列表

| 实验ID | 方案 | 权重类型 | Weight Loss | λ_KL | λ_κ | λ_μ | λ_w |
|--------|------|----------|-------------|------|-----|-----|-----|
| LW_1a | Baseline | Fixed (0.25) | - | 1.0 | 5.0 | 2.0 | - |
| LW_1b | Learnable | Softmax | None | 1.0 | 5.0 | 2.0 | 0 |
| LW_1c | Learnable | Softmax | Entropy | 1.0 | 5.0 | 2.0 | 0.1 |
| LW_1d | Learnable | Softmax | GT Supervision | 1.0 | 5.0 | 2.0 | 1.0 |

### GT 权重定义 (LW_1d)

对于单正面数据，GT的4峰是同方向的，理论上:
- **方案A**: 均匀权重 [0.25, 0.25, 0.25, 0.25] (当前GT)
- **方案B**: 集中权重 [1.0, 0, 0, 0] (只有主峰有权重)

实验中我们使用 **方案A** 作为GT权重。

---

## 关键代码修改

### 1. 模型输出增加 weights

```python
class LearnableWeightHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512):
        super().__init__()
        # ... existing layers ...
        self.weight_head = nn.Linear(hidden_channels, 4)  # 新增

    def forward(self, x):
        # ... existing code ...
        mu = ...      # (B, 4, 2)
        kappa = ...   # (B, 4)
        weights = F.softmax(self.weight_head(x), dim=-1)  # (B, 4)
        return mu, kappa, weights
```

### 2. Loss 函数修改

```python
def _compute_mixture_pdf(self, mu, kappa, weights):
    """使用可变权重计算混合PDF"""
    component_pdfs = self._von_mises_pdf(grid, mu, kappa)  # (B, 4, 360)

    # 使用可变权重而非固定 0.25
    weighted_pdf = weights.unsqueeze(-1) * component_pdfs  # (B, 4, 360)
    mixture_pdf = weighted_pdf.sum(dim=1)  # (B, 360)

    return mixture_pdf

def forward(self, pred_mus, pred_kappas, pred_weights, gt_mus, gt_kappas, gt_weights):
    # KL Loss (使用可变权重)
    pred_pdf = self._compute_mixture_pdf(pred_mus, pred_kappas, pred_weights)
    gt_pdf = self._compute_mixture_pdf(gt_mus, gt_kappas, gt_weights)
    kl_loss = ...

    # Weight Loss (可选)
    if self.lambda_weight > 0:
        if self.weight_loss_type == 'entropy':
            # 最小化熵 → 鼓励稀疏
            weight_loss = -(pred_weights * torch.log(pred_weights + eps)).sum(dim=-1).mean()
        elif self.weight_loss_type == 'gt':
            # GT 监督
            weight_loss = F.mse_loss(pred_weights, gt_weights)

    total_loss = ... + self.lambda_weight * weight_loss
```

---

## 预期结果

| 实验 | 预期效果 |
|------|----------|
| LW_1a (Baseline) | 基准，角度误差 ~32° |
| LW_1b (Free) | 可能更灵活，但权重可能学偏 |
| LW_1c (Entropy) | 鼓励稀疏，可能收敛到单峰 |
| LW_1d (GT) | 类似 Baseline，但有额外学习负担 |

**关键观察点**:
1. 可学习权重是否能提升角度预测精度
2. 网络学到的权重分布是怎样的 (均匀 vs 稀疏)
3. Weight Loss 对训练稳定性的影响

---

## 评估指标

1. **val/angle_error_deg**: 平均角度误差
2. **val/kl_div**: KL 散度
3. **val/kappa_loss**: κ 监督 loss
4. **val/weight_entropy**: 权重熵 (衡量权重分布集中程度)
5. **val/weight_max**: 最大权重值 (衡量主峰是否突出)

---

## 运行命令

```bash
# 运行全部实验
bash scripts/run_learnable_weight_exp.sh

# 或单独运行
python train_single_front_learnable.py \
    --exp_name LW_1b_learnable_free \
    --learnable_weights \
    --lambda_weight 0.0 \
    --wandb
```

---

## 文档版本

- **v1.0** (2025-12-16): 初始设计
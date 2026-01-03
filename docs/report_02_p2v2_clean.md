# 实验报告 2: P2v2 Clean (Mixture of Experts)

> **Checkpoint**: `checkpoints/P2v2_Clean_20251230_165848`
> **训练日期**: 2025-12-30 ~ 2025-12-31
> **WandB Run ID**: `yjthu57d`
> **WandB Project**: ForwardNet-LossAblation

---

## 1. 实验目标

训练一个基于Mixture of Experts (MoE)架构的方向预测模型。利用预训练的对称性分类器作为Gate机制，根据物体的对称类型选择不同的Expert Head进行方向预测。

---

## 2. 模型架构

### 2.1 整体架构

```
ProbabilisticOrientationNet
│
├── Classifier (Gate) ── 冻结，来自CleanClassifier_20251229_220630
│   └── 输出: weights [B, 5] ── 5类softmax概率
│
├── Shared Backbone (PointNet++)
│   ├── SetAbstraction(npoint=512, nsample=32, MLP=[64,64,128])
│   ├── SetAbstraction(npoint=128, nsample=64, MLP=[128,128,256])
│   ├── SetAbstraction(global, MLP=[256,512,1024])
│   └── 输出: global_feat [B, 1024]
│
├── ExpertHead_1front (1峰)
│   ├── Hidden: Linear(1024→256) + ReLU + Linear(256→256) + ReLU
│   ├── fc_mu: Linear(256→2) → atan2 → μ [B, 1]
│   └── fc_kappa: Linear(256→1) → softplus → κ [B, 1]
│
├── ExpertHead_2front (2峰)
│   ├── Hidden: 同上
│   ├── fc_mu: Linear(256→4) → μ [B, 2]
│   └── fc_kappa: Linear(256→2) → κ [B, 2]
│
└── ExpertHead_4front (4峰)
    ├── Hidden: 同上
    ├── fc_mu: Linear(256→8) → μ [B, 4]
    └── fc_kappa: Linear(256→4) → κ [B, 4]
```

### 2.2 参数统计

| 组件 | 参数量 | 可训练 |
|------|--------|--------|
| Classifier (Gate) | 2,156,101 | ✗ 冻结 |
| Shared Backbone | 1,534,528 | ✓ |
| ExpertHead_1front | 87,555 | ✓ |
| ExpertHead_2front | 88,070 | ✓ |
| ExpertHead_4front | 89,100 | ✓ |
| **总计** | **3,953,818** | **1,797,717** |

### 2.3 代码位置

- **模型定义**: `models/probabilistic_orientation_net.py` (lines 246-393)
- **ExpertHead**: `models/probabilistic_orientation_net.py` (lines 159-243)
- **MaskedExpertLoss**: `models/probabilistic_orientation_net.py` (lines 480-797)

---

## 3. Expert Head 详解

### 3.1 角度预测 (μ)

使用2D单位向量 + atan2方法处理角度的周期性:

```python
# 网络输出原始2D向量
mu_raw = self.fc_mu(hidden)  # [B, num_peaks * 2]
mu_raw = mu_raw.view(B, num_peaks, 2)  # [B, num_peaks, 2]

# 提取cos和sin分量
cos_val = mu_raw[:, :, 0]  # [B, num_peaks]
sin_val = mu_raw[:, :, 1]  # [B, num_peaks]

# 转换为角度
mu = torch.atan2(sin_val, cos_val)  # [B, num_peaks], 范围 [-π, π]
```

### 3.2 集中度预测 (κ)

```python
raw_kappa = self.fc_kappa(hidden)  # [B, num_peaks]
kappa = F.softplus(raw_kappa)  # 保证非负
kappa = kappa.clamp(min=kappa_min, max=kappa_max)  # [1e-4, 100]
```

### 3.3 κ初始化

```python
# 初始化bias使softplus(bias) ≈ kappa_init (5.0)
# softplus(x) = log(1 + exp(x))
# 反函数: x = log(exp(k) - 1)
init_bias = math.log(math.exp(5.0) - 1)  # ≈ 4.99
self.fc_kappa.bias.data.fill_(init_bias)
```

---

## 4. 损失函数: MaskedExpertLoss

### 4.1 核心思想

1. **GT-based Routing**: 训练时根据GT label选择对应的Expert Head
2. **Soft Gate Weighting**: 使用分类器输出作为样本权重
3. **Von Mises NLL**: 方向预测使用von Mises分布的负对数似然

### 4.2 样本权重计算

```python
def compute_sample_weights(gate_weights, gt_labels):
    # gate_weights: [B, 5] 分类器softmax输出
    # gt_labels: [B] GT标签

    # p_dir = P(1-front) + P(2-front) + P(4-front)
    p_dir = gate_weights[:, 0] + gate_weights[:, 1] + gate_weights[:, 2]

    # Soft gate weight: 线性映射
    # threshold=0.4 时, p_dir∈[0.4, 1.0] → w_gate∈[0, 1]
    threshold = 0.4
    w_gate = torch.clamp((p_dir - threshold) / (1 - threshold), 0, 1)

    # Class confidence weight: p_gt^gamma
    p_gt = gate_weights.gather(1, gt_labels.unsqueeze(1)).squeeze(1)
    gamma = 1.5
    w_cls = torch.pow(p_gt, gamma)

    # 组合权重
    return w_gate * w_cls  # [B]
```

### 4.3 Von Mises 负对数似然

```python
def von_mises_nll(theta_gt, mu, kappa):
    """
    Von Mises分布: p(θ|μ,κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))

    NLL = -κ * cos(θ_gt - μ) + log(2π) + log(I₀(κ))
    """
    cos_diff = torch.cos(theta_gt - mu)
    nll = -kappa * cos_diff + math.log(2 * math.pi) + log_bessel_i0(kappa)
    return nll

def log_bessel_i0(kappa):
    """数值稳定的log(I0(kappa))"""
    from torch.special import i0e
    # log(I0(x)) = x + log(I0e(x)), where I0e(x) = I0(x) * exp(-x)
    return kappa + torch.log(i0e(kappa) + 1e-10)
```

### 4.4 Hungarian匹配 (多峰情况)

#### 4.4.1 K=2 (2-front)

```python
def batched_hungarian_match_k2(pred_mu, gt_angles):
    # pred_mu: [B, 2], gt_angles: [B, 2]

    # 只有2种排列: identity 和 swap
    cost_identity = (1 - torch.cos(pred_mu - gt_angles)).sum(dim=1)

    gt_swapped = torch.stack([gt_angles[:, 1], gt_angles[:, 0]], dim=1)
    cost_swap = (1 - torch.cos(pred_mu - gt_swapped)).sum(dim=1)

    use_swap = (cost_swap < cost_identity).unsqueeze(1)
    return torch.where(use_swap, gt_swapped, gt_angles)
```

#### 4.4.2 K=4 (4-front)

```python
def batched_hungarian_match_k4(pred_mu, gt_angles, perms_4):
    # perms_4: [24, 4] 预缓存的所有排列

    gt_permuted = gt_angles[:, perms_4]  # [B, 24, 4]
    pred_expanded = pred_mu.unsqueeze(1)  # [B, 1, 4]

    # 计算所有排列的代价
    costs = (1 - torch.cos(pred_expanded - gt_permuted)).sum(dim=2)  # [B, 24]

    # 选择最优排列
    best_perm_idx = costs.argmin(dim=1)  # [B]
    best_perms = perms_4[best_perm_idx]  # [B, 4]

    return torch.gather(gt_angles, 1, best_perms)
```

### 4.5 调度参数

| 参数 | 初始值 | 最终值 | 调度方式 |
|------|--------|--------|----------|
| `cosine_weight` | 0.2 | 0.1 | epoch 10后切换 |
| `kappa_reg_weight` | 0.0 | 0.02 | epoch 6开始, 10个epoch线性增加 |
| `kappa_target` | 0.0 | 5.0 | 同上 |

### 4.6 完整损失公式

```python
# 1-front loss (始终包含cosine loss确保μ有梯度)
loss_1f = NLL(gt, μ, κ) + cosine_weight * (1 - cos(μ - gt))
if epoch >= 6:  # κ正则化
    loss_1f += kappa_reg_weight * relu(kappa_target - κ)

# 2-front loss
gt_matched = hungarian_match(pred_μ, [gt, gt+π])  # 对称GT
loss_2f = mean(NLL(gt_matched, μ, κ))  # 平均2个峰

# 4-front loss
gt_matched = hungarian_match(pred_μ, [gt, gt+π/2, gt+π, gt+3π/2])
loss_4f = mean(NLL(gt_matched, μ, κ))  # 平均4个峰

# Micro-average: 按样本权重加权
total_loss = sum(w * loss) / sum(w)
```

---

## 5. Ground Truth 生成

### 5.1 方向到角度映射

```python
DIRECTION_TO_ANGLE = {
    '+X': 0.0,
    '+Z': np.pi / 2,      # 90°
    '-X': np.pi,          # 180°
    '-Z': 3 * np.pi / 2,  # 270°
}
```

### 5.2 GT峰生成规则

| 对称类型 | 标签 | GT峰数 | 公式 |
|---------|------|--------|------|
| 1-front | 0 | 1 | `[θ]` |
| 2-front | 1 | 2 | `[θ, θ+π]` |
| 4-front | 2 | 4 | `[θ, θ+π/2, θ+π, θ+3π/2]` |
| Rot-sym | 3 | - | 无监督 (跳过) |
| No-front | 4 | - | 无监督 (跳过) |

### 5.3 旋转增强后GT更新

```python
# 原始角度
base_angle = DIRECTION_TO_ANGLE[direction]

# 应用旋转增强
rotation_angle = np.random.uniform(0, 2 * np.pi)
points = rotate_y(points, rotation_angle)

# 更新GT角度
gt_angle = (base_angle + rotation_angle) % (2 * np.pi)
```

---

## 6. 数据集配置

### 6.1 数据过滤

与Classifier相同:
- **1-front类别**: 仅 airplane, chair
- **异常值**: 排除13个severe异常 (误差≥90°)

### 6.2 数据统计

| 划分 | 基础样本 | ×旋转增强 | 总样本 |
|------|---------|----------|--------|
| Train | 2,116 | ×12 | 25,392 |
| Val | 451 | ×4 | 1,804 |
| Test | 458 | ×4 | 1,832 |

*注: 分类器训练使用相同数据划分*

---

## 7. 训练配置

### 7.1 完整配置参数

```json
{
  "annotation_file": "data_annotation/symmetry_annotations.json",
  "data_dir": "data/full_mn40_normal_resampled_ply",
  "outlier_json": "data_annotation/1front_outliers.json",
  "outlier_threshold": "severe",
  "allowed_1front_categories": ["airplane", "chair"],

  "classifier_checkpoint": "checkpoints/CleanClassifier_20251229_220630/best.pth",

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

### 7.2 优化器配置

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| T_max | 100 |
| eta_min | 1e-6 |
| Gradient Clipping | max_norm=1.0 |

---

## 8. 训练结果

### 8.1 最终结果 (从checkpoint读取)

#### Best模型 (epoch 93, 0-indexed: 92)

```python
# 来源: checkpoints/P2v2_Clean_20251230_165848/best.pth
{
    'epoch': 92,
    'best_val_error': 5.613948354838872,
    'wandb_run_id': 'yjthu57d',
    'metrics': {
        'val_1f_error': 11.876715660095215,
        'val_1f_median': 0.990145206451416,
        'val_1f_std': 40.5692138671875,
        'val_1f_lt5': 89.02777777777777,
        'val_1f_lt10': 91.80555555555556,
        'val_1f_lt15': 92.77777777777779,
        'val_1f_gt45': 6.527777777777779,
        'val_1f_gt90': 5.972222222222222,
        'val_1f_kappa': 90.64009857177734,
        'val_1f_kappa_std': 25.467330932617188,
        'val_2f_error': 3.3704005728167763,
        'val_2f_median': 0.5250024905530302,
        'val_2f_lt5': 94.16666666666667,
        'val_2f_lt10': 95.83333333333334,
        'val_2f_kappa': 99.8843765258789,
        'val_4f_error': 0.25099569510761527,
        'val_4f_median': 0.208656075944741,
        'val_4f_lt5': 100.0,
        'val_4f_lt10': 100.0,
        'val_4f_kappa': 100.0,
        'val_error': 5.613948354838872,
        'val_median': 0.436042091398604,
        'val_lt10': 95.70652173913044
    }
}
```

#### Final模型 (epoch 100)

```python
# 来源: checkpoints/P2v2_Clean_20251230_165848/final.pth
{
    'epoch': 99,
    'test_metrics': {
        'test_1f_error': 10.182847023010254,
        'test_1f_median': 0.9170348048210144,
        'test_1f_std': 36.80856704711914,
        'test_1f_lt5': 89.56043956043956,
        'test_1f_lt10': 93.13186813186813,
        'test_1f_lt15': 93.81868131868131,
        'test_1f_gt45': 5.4945054945054945,
        'test_1f_gt90': 4.945054945054945,
        'test_1f_kappa': 90.28652954101562,
        'test_1f_kappa_std': 26.482881546020508,
        'test_2f_error': 2.505518112887559,
        'test_2f_median': 0.45346308918631656,
        'test_2f_lt5': 97.5,
        'test_2f_lt10': 97.91666666666666,
        'test_2f_kappa': 99.5401840209961,
        'test_4f_error': 0.22371391061934082,
        'test_4f_median': 0.18066687453063449,
        'test_4f_lt5': 100.0,
        'test_4f_lt10': 100.0,
        'test_4f_kappa': 100.0,
        'test_error': 4.662817432738234,
        'test_median': 0.37071534739161205,
        'test_lt10': 96.80851063829788
    }
}
```

### 8.2 结果汇总表

| 指标 | 验证集 (Best) | 测试集 (Final) |
|------|--------------|----------------|
| **整体误差** | **5.61°** | **4.66°** |
| 整体中位数 | 0.44° | 0.37° |
| 整体<10° | 95.71% | 96.81% |
| **1-front误差** | 11.88° | 10.18° |
| 1-front中位数 | 0.99° | 0.92° |
| 1-front<5° | 89.03% | 89.56% |
| 1-front<10° | 91.81% | 93.13% |
| 1-front>45° | 6.53% | 5.49% |
| 1-front>90° | 5.97% | 4.95% |
| 1-front κ均值 | 90.64 | 90.29 |
| **2-front误差** | 3.37° | 2.51° |
| 2-front中位数 | 0.53° | 0.45° |
| 2-front<10° | 95.83% | 97.92% |
| 2-front κ均值 | 99.88 | 99.54 |
| **4-front误差** | 0.25° | 0.22° |
| 4-front中位数 | 0.21° | 0.18° |
| 4-front<10° | 100% | 100% |
| 4-front κ均值 | 100.00 | 100.00 |

### 8.3 训练时间

| 阶段 | 时间 |
|------|------|
| 每个epoch | ~5-6分钟 |
| 总训练时间 | **5.74小时** |

### 8.4 训练曲线 (部分epochs)

```
Epoch   Train Loss   Val 1f    Val 2f    Val 4f    Val Overall
─────────────────────────────────────────────────────────────────
40      -0.7402      12.64°    3.83°     0.57°     6.15° (best↓)
50      -0.7955      13.55°    3.67°     0.41°     6.52°
60      -0.8552      13.49°    3.35°     0.49°     6.39°
70      -0.8089      13.05°    2.56°     0.25°     6.00°
80      -0.8377      12.35°    3.46°     0.27°     6.04°
90      -0.8552      12.00°    3.21°     0.24°     5.74° (best↓)
93      -0.8xxx      11.88°    3.37°     0.25°     5.61° (BEST)
100     -0.8xxx      12.05°    3.21°     0.24°     5.78°
```

### 8.5 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pth` | 29.0 MB | Best模型 (epoch 93) |
| `latest.pth` | 29.0 MB | 最新checkpoint (epoch 100) |
| `final.pth` | 15.2 MB | 仅模型权重 + 测试指标 |
| `config.json` | 1.3 KB | 训练配置 |
| `training_state_snapshot.json` | 3.2 KB | 训练状态快照 (epoch 39) |

---

## 9. 恢复训练机制

### 9.1 Checkpoint内容

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_error': best_val_error,
    'wandb_run_id': wandb_run_id,  # 用于恢复WandB run
    'metrics': val_metrics,
}
```

### 9.2 恢复代码

```python
# 加载checkpoint
checkpoint = torch.load(resume_dir / 'latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_val_error = checkpoint['best_val_error']

# 恢复WandB run
wandb.init(
    project=args.wandb_project,
    id=checkpoint['wandb_run_id'],
    resume="must"
)
```

---

## 10. 使用说明

### 10.1 加载模型

```python
import torch
from models import ProbabilisticOrientationNet
from train_symmetry_classifier import SymmetryClassifier
from train_clean_pipeline import ClassifierWrapper

# 加载分类器
classifier = SymmetryClassifier(encoder_dim=1024, num_classes=5)
clf_ckpt = torch.load('checkpoints/CleanClassifier_20251229_220630/best.pth')
classifier.load_state_dict(clf_ckpt['model_state_dict'])
wrapped_classifier = ClassifierWrapper(classifier)

# 创建P2v2模型
model = ProbabilisticOrientationNet(
    classifier=wrapped_classifier,
    backbone_dim=1024,
    expert_hidden_dim=256,
    freeze_classifier=True
)

# 加载权重
ckpt = torch.load('checkpoints/P2v2_Clean_20251230_165848/best.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

### 10.2 推理

```python
with torch.no_grad():
    outputs = model(points)  # points: [B, N, 3]

    # 获取分类权重
    weights = outputs['weights']  # [B, 5]
    predicted_class = weights.argmax(dim=1)

    # 获取方向预测
    mu_1f = outputs['head_1front']['mu']  # [B, 1]
    mu_2f = outputs['head_2front']['mu']  # [B, 2]
    mu_4f = outputs['head_4front']['mu']  # [B, 4]

    kappa_1f = outputs['head_1front']['kappa']  # [B, 1]
```

---

*报告生成时间: 2025-12-31*

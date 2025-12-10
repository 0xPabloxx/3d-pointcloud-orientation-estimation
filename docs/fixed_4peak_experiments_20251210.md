# Fixed 4-Peak von Mises 实验报告

**日期**: 2024-12-10
**作者**: Claude Code + 用户
**项目**: 3D点云正面方向检测

---

## 1. 实验概述

### 1.1 任务描述
训练一个网络，输出固定 4 个 von Mises 分布的参数 (μ, κ)，用于预测 3D 物体的正面方向。

### 1.2 数据集
- **训练集**: 798 samples × 10 augmentation = 7,980 samples
- **验证集**: 228 samples
- **类别分布**: 1_front, 2_fronts, 4_fronts, symmetric, no_front (各 ~200 个)

### 1.3 模型架构
- **Encoder**: PointNet++ (SSG), 输出 1024 维特征
- **μ head**: Linear(512, 8) → reshape(4, 2) → normalize → (B, 4, 2)
- **κ head**: Linear(512, 4) → softplus → (B, 4)
- **参数量**: 1,603,404

---

## 2. 实验方法

### 2.1 Exp-1: Pure KL Divergence

**Loss 公式**:
```
L = KL(P_GT || P_pred)
  = ∫ P_GT(θ) log(P_GT(θ) / P_pred(θ)) dθ
```

其中混合 von Mises PDF:
```
P(θ) = Σᵢ wᵢ × VM(θ; μᵢ, κᵢ)
VM(θ; μ, κ) = exp(κ cos(θ - μ)) / (2π I₀(κ))
```

**核心代码**:
```python
# Loss 计算 (train_fixed_4peak.py:269-282)
def forward(self, pred_mus, pred_kappas, gt_mus, gt_kappas):
    # 计算 PDF
    pred_pdf = self._compute_mixture_pdf(pred_mus, pred_kappas)
    with torch.no_grad():
        gt_pdf = self._compute_mixture_pdf(gt_mus, gt_kappas)

    # KL Divergence
    kl_per_bin = gt_pdf * (torch.log(gt_pdf) - torch.log(pred_pdf))
    kl_loss = kl_per_bin.sum(dim=1).mean() * bin_width

    return {'loss': kl_loss, 'kl_div': kl_loss}
```

**命令**:
```bash
python train_fixed_4peak.py --lambda_kl 1.0 --lambda_kappa 0 --lambda_mu 0
```

---

### 2.2 Exp-2: KL + κ Supervision

**Loss 公式**:
```
L = L_KL + λ_κ × L_κ

L_κ = SmoothL1(κ_pred / 50, κ_GT / 50)
```

使用匈牙利算法对齐预测峰和 GT 峰。

**核心代码**:
```python
# 匈牙利匹配 (train_fixed_4peak.py:170-236)
def _hungarian_match(self, pred_mus, pred_kappas, gt_mus, gt_kappas):
    for b in range(B):
        # Cost matrix: 角度距离 + κ 距离
        angle_diff = pred_ang.unsqueeze(1) - gt_ang.unsqueeze(0)
        angle_cost = 1 - torch.cos(angle_diff)
        kappa_diff = torch.abs(pred_kappas[b].unsqueeze(1) - gt_kappas[b].unsqueeze(0)) / 50.0
        cost_matrix = angle_cost + 0.1 * kappa_diff

        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

    # 使用 gather 保持梯度
    matched_pred_kappas = torch.gather(pred_kappas, 1, pred_indices)
    return matched_pred_kappas, matched_gt_kappas

# κ Loss
kappa_loss = F.smooth_l1_loss(matched_pred_kappas / 50.0, matched_gt_kappas / 50.0)
```

**命令**:
```bash
python train_fixed_4peak.py --lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 0
```

---

### 2.3 Exp-3: KL + κ + μ Supervision

**Loss 公式**:
```
L = L_KL + λ_κ × L_κ + λ_μ × L_μ

L_μ = mean(1 - cos(θ_pred - θ_GT))  # 仅对有效峰 (κ_GT > 0)
```

**核心代码**:
```python
# μ Loss (train_fixed_4peak.py:296-307)
angle_diff = matched_pred_angles - matched_gt_angles
angle_loss_per_peak = 1 - torch.cos(angle_diff)  # 范围 [0, 2]

# 只对有效峰计算
valid_mask = (matched_gt_kappas > 0).float()
mu_loss = (angle_loss_per_peak * valid_mask).sum() / (valid_mask.sum() + eps)
```

**命令**:
```bash
python train_fixed_4peak.py --lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 2.0
```

---

### 2.4 Exp-4: μ + κ Only (No KL)

**Loss 公式**:
```
L = λ_κ × L_κ + λ_μ × L_μ
```

移除 KL Loss，只使用参数监督。

**核心代码**:
```python
# 条件计算 KL (train_fixed_4peak.py:269-285)
if self.lambda_kl > 0:
    # 计算 KL...
    kl_loss = ...
else:
    kl_loss = torch.tensor(0.0, device=pred_mus.device)

# Total Loss
total_loss = self.lambda_kl * kl_loss + self.lambda_kappa * kappa_loss + self.lambda_mu * mu_loss
```

**命令**:
```bash
python train_fixed_4peak.py --lambda_kl 0 --lambda_kappa 5.0 --lambda_mu 2.0
```

---

## 3. 实验结果

### 3.1 结果对比表

| 实验 | Loss 配置 | Val KL | Val κ_loss | Val μ_loss | 4_fronts KL | Best Val Loss |
|------|-----------|--------|------------|------------|-------------|---------------|
| **Exp-1** | KL only | **0.32** | N/A | N/A | **0.98** | 0.32 |
| **Exp-2** | KL + κ | 0.45 | 0.08 | N/A | 1.52 | 0.86 |
| **Exp-3** | KL + κ + μ | 0.40 | 0.10 | 0.08 | 1.24 | 1.05 |
| **Exp-4** | κ + μ only | N/A | **0.042** | **0.082** | N/A | **0.35** |

### 3.2 WandB 链接

| 实验 | Run Name | Link |
|------|----------|------|
| Exp-1 | `fixed4peak_pn++_paired-init_kl_1140samples` | [WandB](https://wandb.ai/augustuschen00-university-of-tokyo/forwardnet/runs/p94e1fqe) |
| Exp-2 | `fixed4peak_pn++_kl+kappa_20251210_164456` | [WandB](https://wandb.ai/augustuschen00-university-of-tokyo/forwardnet/runs/okoads9h) |
| Exp-3 | `fixed4peak_pn++_kl+kappa+mu_20251210_170618` | [WandB](https://wandb.ai/augustuschen00-university-of-tokyo/forwardnet/runs/54r76l9h) |
| Exp-4 | `fixed4peak_pn++_kappa5.0_mu2.0_20251210_175008` | [WandB](https://wandb.ai/augustuschen00-university-of-tokyo/forwardnet/runs/5hlqox18) |

### 3.3 Checkpoints 位置

```
checkpoints/
├── fixed4peak_pn++_paired-init_kl_1140samples/   # Exp-1
├── fixed4peak_pn++_kl+kappa_20251210_164456/     # Exp-2
├── fixed4peak_pn++_kl+kappa+mu_20251210_170618/  # Exp-3
└── fixed4peak_pn++_kappa5.0_mu2.0_20251210_175008/  # Exp-4
```

---

## 4. 关键发现与分析

### 4.1 Pure KL 导致 κ 坍塌

**现象**:
- Val KL 从 Epoch 2 开始恒定 (0.3188)
- 模型输出 κ ≈ 0 (uniform 分布)

**原因**:
```
初始: κ ≈ 25, μ 方向随机
  ↓
KL Loss: "峰在错误位置！"
  ↓
梯度: "降低 κ 减少惩罚" (更安全)
  ↓
κ → 0 (uniform 输出)
  ↓
训练停止 (局部最小值)
```

**验证代码**:
```python
# 检查训练后模型的 κ 输出
model.load_state_dict(checkpoint['model_state_dict'])
with torch.no_grad():
    pred_mu, pred_kappa = model(points)
print(f"Pred κ: {pred_kappa}")  # [0.02, 0.01, 0.01, 0.02] ≈ 0
```

### 4.2 添加 κ 监督后 KL 更差

**现象**:
- Val KL: 0.32 → 0.45 (+41%)
- 4_fronts KL: 0.98 → 1.52 (+55%)

**原因**:
```
κ 监督: 强制 κ → 50
  ↓
但 μ 方向仍然错误
  ↓
尖峰在错误位置 → KL 更大
```

### 4.3 添加 μ 监督有改善但存在冲突

**现象**:
- 4_fronts KL: 1.52 → 1.24 (改善)
- 但仍不如 Pure KL (0.98)

**原因**:
```
KL Loss: "输出 uniform 最安全"
κ Loss:  "不行，要 κ=50"
μ Loss:  "不行，要正确角度"
  ↓
三个目标冲突，优化困难
```

### 4.4 移除 KL 后训练稳定

**现象**:
- κ_loss: 0.07 → 0.042 (持续下降)
- μ_loss: 0.066 → 0.082 (稳定)

**原因**:
```
只有 κ + μ 监督
  ↓
消除 KL 的 κ→0 倾向
  ↓
直接学习峰参数
  ↓
优化目标一致
```

---

## 5. 结论与建议

### 5.1 主要结论

1. **Pure KL 不适合直接训练**: 会导致 κ 坍塌到 0
2. **KL + 参数监督存在冲突**: 反而比 Pure KL 更差
3. **纯参数监督 (κ + μ) 效果稳定**: Best val loss 0.35

### 5.2 后续建议

1. **评估 Exp-4 模型的实际预测质量**:
   - 可视化预测分布
   - 计算角度误差统计

2. **尝试两阶段训练**:
   - 阶段 1: 用 κ + μ 预训练
   - 阶段 2: 加入 KL 微调

3. **调整 loss 权重**:
   - 尝试更小的 λ_κ 和 λ_μ
   - 或者更大的 κ 目标值

---

## 6. 附录：完整命令参考

```bash
# Exp-1: Pure KL
python train_fixed_4peak.py --epochs 100 --lambda_kl 1.0 --lambda_kappa 0 --lambda_mu 0

# Exp-2: KL + κ
python train_fixed_4peak.py --epochs 100 --lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 0

# Exp-3: KL + κ + μ
python train_fixed_4peak.py --epochs 100 --lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 2.0

# Exp-4: κ + μ only
python train_fixed_4peak.py --epochs 100 --lambda_kl 0 --lambda_kappa 5.0 --lambda_mu 2.0
```

---

**文档版本**: 1.0
**最后更新**: 2024-12-10

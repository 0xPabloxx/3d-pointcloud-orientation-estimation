# Loss Ablation 实验计划

**日期**: 2025-12-11
**目的**: 快速验证不同loss配置的效果
**训练轮数**: 15 epochs

---

## 实验配置

### 公共设置
| 参数 | 值 |
|------|-----|
| Epochs | 15 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Model | PointNet++ (SSG) |
| κ初始化 | bias=5 (初始κ≈5) |
| μ初始化 | 固定 [0°, 90°, 180°, 270°] (确保可重现性) |
| Weight | 固定 0.25 |

### GT数据
| 类别 | 峰结构 | κ值 | 说明 |
|------|--------|-----|------|
| 1_front | 4峰同方向 | **全10** | 有方向 |
| 2_fronts | [θ, θ+180°, θ, θ+180°] | **全10** | 有方向 |
| 4_fronts | [θ, θ+90°, θ+180°, θ+270°] | **全10** | 有方向 |
| symmetric | 任意 | **全0** | 无方向(均匀分布) |
| no_front | 任意 | **全0** | 无方向(均匀分布) |

**注意**: 60%样本有方向(κ=10)，40%样本无方向(κ=0)

---

## 实验列表 (9个实验)

### 第一组：基础验证

#### Exp-1: 纯4_fronts数据 (最简单case)
```bash
--lambda_kl 0.0 --lambda_kappa 5.0 --lambda_mu 2.0 --categories 4_fronts
```
**目的**: 在最理想情况下验证模型能否学好

**预期**:
- 应该能很好地学习，因为所有样本都是κ=10的4峰

---

#### Exp-2: 纯κ + μ监督 (预期最稳定)
```bash
--lambda_kl 0.0 --lambda_kappa 5.0 --lambda_mu 2.0
```
**Loss**: $\mathcal{L} = 5.0 \cdot L_\kappa + 2.0 \cdot L_\mu$

**预期**:
- 最稳定的配置
- 直接监督参数，应该能学到正确的κ和μ

---

### 第二组：KL配置对比

#### Exp-3: 纯Forward KL (对照组)
```bash
--lambda_kl 1.0 --lambda_kappa 0.0 --lambda_mu 0.0
```
**Loss**: $\mathcal{L} = \text{KL}(p_{GT} \| p_{pred})$

**预期**:
- 可能出现κ坍塌(κ→0)
- Forward KL倾向于mode-covering，可能导致输出均匀分布

---

#### Exp-3b: Forward KL + min_kappa硬下限
```bash
--lambda_kl 1.0 --lambda_kappa 0.0 --lambda_mu 0.0 --min_kappa 5.0
```
**Loss**: $\mathcal{L} = \text{KL}$, 但 $\kappa \geq 5$ (硬下限)

**预期**:
- 防止κ坍塌到0
- 观察KL单独能否学习方向(μ)
- **关键实验**: 验证κ坍塌是否是KL无法学习的根本原因

---

#### Exp-4: 纯Reverse KL
```bash
--lambda_kl 1.0 --lambda_kappa 0.0 --lambda_mu 0.0 --reverse_kl
```
**Loss**: $\mathcal{L} = \text{KL}(p_{pred} \| p_{GT})$

**预期**:
- Reverse KL倾向于mode-seeking
- 可能产生更尖锐的峰
- **关键实验**: 看是否能解决κ坍塌问题

---

#### Exp-5: Forward KL + κ监督
```bash
--lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 0.0
```
**Loss**: $\mathcal{L} = \text{KL} + 5.0 \cdot L_\kappa$

**预期**:
- κ应该能学到正确值
- μ方向可能不准确(无直接监督)

---

#### Exp-6: Forward KL + μ监督
```bash
--lambda_kl 1.0 --lambda_kappa 0.0 --lambda_mu 2.0
```
**Loss**: $\mathcal{L} = \text{KL} + 2.0 \cdot L_\mu$

**预期**:
- μ方向应该有改善
- κ可能仍会坍塌

---

#### Exp-7: Forward KL + κ + μ监督
```bash
--lambda_kl 1.0 --lambda_kappa 5.0 --lambda_mu 2.0
```
**Loss**: $\mathcal{L} = \text{KL} + 5.0 \cdot L_\kappa + 2.0 \cdot L_\mu$

**预期**:
- 三个目标可能冲突
- 需要观察哪个loss主导

---

### 第三组：高级策略

#### Exp-8: 预热式训练 (先监督5ep→再加KL)
**Stage 1 - 预热** (5 epochs):
```bash
--epochs 5 --lambda_kl 0.0 --lambda_kappa 5.0 --lambda_mu 2.0
```
**Stage 2 - 加入KL** (10 epochs, 从Stage 1 checkpoint继续):
```bash
--epochs 10 --lambda_kl 1.0 --lambda_kappa 3.0 --lambda_mu 2.0 --resume <stage1_ckpt>
```

**预期**:
- Stage 1: 纯监督快速收敛，稳定μ和κ
- Stage 2: 加入KL优化分布形状，降低κ权重避免冲突
- 减少三路loss冲突导致的训练初期抖动

---

#### Exp-9: 平衡权重
```bash
--lambda_kl 0.5 --lambda_kappa 5.0 --lambda_mu 3.0
```
**Loss**: $\mathcal{L} = 0.5 \cdot \text{KL} + 5.0 \cdot L_\kappa + 3.0 \cdot L_\mu$

**预期**:
- 降低KL权重，增加μ权重
- 可能获得更平衡的训练

---

## 评估指标

每个实验记录:
1. **val_loss**: 总验证loss
2. **val_kl**: KL散度 (如有)
3. **val_kappa_loss**: κ监督loss
4. **val_mu_loss**: μ监督loss
5. **val_mean_kappa**: 验证集平均预测κ值

---

## 运行命令

```bash
# 运行全部9个实验 (顺序执行)
bash scripts/exp_loss_ablation.sh

# 或单独运行某个实验
python train_fixed_4peak.py \
    --epochs 15 \
    --lambda_kl 1.0 \
    --lambda_kappa 0.0 \
    --lambda_mu 0.0 \
    --wandb \
    --wandb_project ForwardNet-LossAblation \
    --exp_name Exp3_forwardKL
```

---

## 新增参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--categories` | 只使用指定类别 | `--categories 4_fronts 2_fronts` |
| `--reverse_kl` | 使用Reverse KL | `--reverse_kl` |
| `--resume` | 从checkpoint恢复 | `--resume checkpoints/xxx/best.pth` |

---

## 结果记录

### Exp-1: 纯4_fronts
| Epoch | Val Loss | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|------------|------------|--------------|
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |

### Exp-2: 纯κ + μ监督
| Epoch | Val Loss | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|------------|------------|--------------|
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |

### Exp-3: 纯Forward KL
| Epoch | Val Loss | Val KL | Val κ (mean) |
|-------|----------|--------|--------------|
| 5 | | | |
| 10 | | | |
| 15 | | | |

### Exp-4: 纯Reverse KL
| Epoch | Val Loss | Val KL | Val κ (mean) |
|-------|----------|--------|--------------|
| 5 | | | |
| 10 | | | |
| 15 | | | |

### Exp-5: Forward KL + κ监督
| Epoch | Val Loss | Val KL | Val κ Loss | Val κ (mean) |
|-------|----------|--------|------------|--------------|
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |

### Exp-6: Forward KL + μ监督
| Epoch | Val Loss | Val KL | Val μ Loss | Val κ (mean) |
|-------|----------|--------|------------|--------------|
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |

### Exp-7: Forward KL + κ + μ监督
| Epoch | Val Loss | Val KL | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|--------|------------|------------|--------------|
| 5 | | | | | |
| 10 | | | | | |
| 15 | | | | | |

### Exp-8: 两阶段训练
**Stage 1:**
| Epoch | Val Loss | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|------------|------------|--------------|
| 5 | | | | |
| 10 | | | | |

**Stage 2:**
| Epoch | Val Loss | Val KL | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|--------|------------|------------|--------------|
| 5 | | | | | |
| 10 | | | | | |

### Exp-9: 平衡权重
| Epoch | Val Loss | Val KL | Val κ Loss | Val μ Loss | Val κ (mean) |
|-------|----------|--------|------------|------------|--------------|
| 5 | | | | | |
| 10 | | | | | |
| 15 | | | | | |

---

## 结论

(实验完成后填写)

### 最佳配置
-

### 关键发现
-

### 下一步
-

---

**文档版本**: 1.1
**最后更新**: 2025-12-11

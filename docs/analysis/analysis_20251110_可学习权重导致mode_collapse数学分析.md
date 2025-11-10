# 可学习权重导致Mode Collapse的数学分析

**作者**: Claude
**日期**: 2025-11-10
**关键词**: Mode Collapse, Weight Exploitation, KL散度, 加权Loss, von Mises Mixture

---

## 1. 问题现象

在Fixed 4-peak MvM训练中，我们观察到一个反常现象：

**当weight可学习时**：
- ✅ 训练loss可以降得很低（看似成功）
- ❌ 但预测结果只有**一个peak的weight接近0.99**，其他3个peak的weight接近0
- ❌ 即使预测的4个peak位置（μ）不准确，loss依然很低
- ❌ 这不是我们想要的"4个等权重peak"分布

**当weight固定为0.25时**：
- ✅ 模型被迫学习4个peak的正确位置
- ✅ 预测结果是真正的4峰等权重分布
- ✅ 泛化性能更好

**核心问题**：为什么可学习weight会导致这种"作弊"行为？这背后的数学机制是什么？

---

## 2. Loss函数分析

### 2.1 实际使用的Loss函数

从代码 `train_pointnetpp_mvm_glassbox_augmented.py` 第154-167行可以看到：

```python
# 计算cost矩阵 (K, K)
for i in range(K):
    for j in range(K):
        cost[i, j] = kl_von_mises(μp[i], κp[i], μg[j], κg[j])

# 匈牙利匹配
row, col = linear_sum_assignment(cost_np)

# 计算加权loss
matched_ws = wp[row]                              # 预测的weight
ws_sum = torch.sum(matched_ws) + 1e-8
loss_bc = torch.sum(matched_ws * cost[row, col]) / ws_sum
```

**关键点1**：KL散度的方向
```python
def kl_von_mises(mu_p, kappa_p, mu_q, kappa_q):
    """计算两个von Mises分布的KL散度: KL(P||Q)"""
    KL = torch.log(i0_q / i0_p) + A_p * (kappa_p - kappa_q * torch.cos(delta))
    return KL
```

这里计算的是 **KL(Pred||GT)**，即从预测分布P到GT分布Q的KL散度。

**关键点2**：加权方式
```python
loss = Σᵢ wp[i] * KL(Pred[i]||GT[matched[i]]) / Σ wp[i]
```

Loss是用**预测的weight wp**来加权的！由于weight已归一化（Σ wp = 1），所以：

$$
\mathcal{L} = \sum_{i=1}^{K} w_i^{\text{pred}} \cdot \text{KL}\big(P_i^{\text{pred}} \,\|\, P_{\sigma(i)}^{\text{GT}}\big)
$$

其中 σ(i) 是匈牙利算法找到的最优匹配。

---

## 3. Mode Collapse的数学机制

### 3.1 GT分布

假设GT是4个等权重的sharp peaks（以glassbox为例）：

$$
p_{\text{GT}}(\theta) = \sum_{j=1}^{4} w_j^{\text{GT}} \cdot \text{VM}(\theta \,|\, \mu_j^{\text{GT}}, \kappa_j^{\text{GT}})
$$

其中：
- $w_1^{\text{GT}} = w_2^{\text{GT}} = w_3^{\text{GT}} = w_4^{\text{GT}} = 0.25$
- $\mu^{\text{GT}} \approx [0°, 90°, 180°, 270°]$（4向对称）
- $\kappa^{\text{GT}}$ 较大（sharp peaks）

### 3.2 预测分布（可学习weight时的collapse模式）

模型学会了以下"作弊"策略：

$$
q_{\text{pred}}(\theta) = \sum_{i=1}^{4} w_i^{\text{pred}} \cdot \text{VM}(\theta \,|\, \mu_i^{\text{pred}}, \kappa_i^{\text{pred}})
$$

其中：
- $w_1^{\text{pred}} = 0.99$（只有一个peak有高权重）
- $w_2^{\text{pred}} = w_3^{\text{pred}} = w_4^{\text{pred}} = 0.003$（其他peak权重极低）

### 3.3 为什么这样可以让Loss很小？

**Loss展开**：
$$
\begin{aligned}
\mathcal{L} &= \sum_{i=1}^{4} w_i^{\text{pred}} \cdot \text{KL}\big(\text{VM}(\mu_i^{\text{pred}}, \kappa_i^{\text{pred}}) \,\|\, \text{VM}(\mu_{\sigma(i)}^{\text{GT}}, \kappa_{\sigma(i)}^{\text{GT}})\big) \\
&= 0.99 \cdot \text{KL}(P_1 \| Q_1) + 0.003 \cdot \text{KL}(P_2 \| Q_2) \\
&\quad + 0.003 \cdot \text{KL}(P_3 \| Q_3) + 0.003 \cdot \text{KL}(P_4 \| Q_4)
\end{aligned}
$$

**关键洞察**：
1. **高权重peak可以精确match一个GT peak**：只要 $\mu_1^{\text{pred}}$ 能对齐到某个GT peak（比如0°），并且 $\kappa_1^{\text{pred}}$ 也match好，那么 $\text{KL}(P_1 \| Q_1)$ 可以非常小（比如0.001）

2. **低权重peak的KL散度被忽略**：即使 $\text{KL}(P_2 \| Q_2)$, $\text{KL}(P_3 \| Q_3)$, $\text{KL}(P_4 \| Q_4)$ 很大（比如10.0），因为权重只有0.003，对总loss贡献只有 $0.003 \times 10 = 0.03$

3. **总Loss很小**：
$$
\mathcal{L} \approx 0.99 \times 0.001 + 3 \times (0.003 \times 10) = 0.00099 + 0.09 \approx 0.091
$$

   甚至可以更小，因为低权重peak可以通过调整 $\kappa$ 来进一步降低KL。

### 3.4 为什么这是"作弊"？

**问题本质**：Loss function的目标应该是让**整体分布**接近GT分布，即：

$$
\mathcal{L}_{\text{ideal}} = \text{KL}\left(\sum_j w_j^{\text{GT}} \cdot \text{VM}_j^{\text{GT}} \,\Big\|\, \sum_i w_i^{\text{pred}} \cdot \text{VM}_i^{\text{pred}}\right)
$$

但实际使用的loss是**component-wise KL的加权和**：

$$
\mathcal{L}_{\text{actual}} = \sum_i w_i^{\text{pred}} \cdot \text{KL}(\text{VM}_i^{\text{pred}} \| \text{VM}_{\sigma(i)}^{\text{GT}})
$$

这两者**不等价**！

当weight可学习时，模型会exploit这个gap：
- 用一个高权重peak去match整个GT分布的某一个mode
- 让其他peak的weight趋近于0，这样它们的KL散度（即使很大）对loss贡献也很小
- 结果：**只学到了GT分布的一部分，而不是全部**

---

## 4. 数值示例

### 4.1 场景设定

**GT分布**（4个等权重sharp peaks）：
- Peak 1: $\mu_1^{\text{GT}} = 0°$, $\kappa_1^{\text{GT}} = 50$, $w_1^{\text{GT}} = 0.25$
- Peak 2: $\mu_2^{\text{GT}} = 90°$, $\kappa_2^{\text{GT}} = 50$, $w_2^{\text{GT}} = 0.25$
- Peak 3: $\mu_3^{\text{GT}} = 180°$, $\kappa_3^{\text{GT}} = 50$, $w_3^{\text{GT}} = 0.25$
- Peak 4: $\mu_4^{\text{GT}} = 270°$, $\kappa_4^{\text{GT}} = 50$, $w_4^{\text{GT}} = 0.25$

### 4.2 策略A：可学习weight（Mode Collapse）

**预测分布**：
- Peak 1: $\mu_1 = 0°$, $\kappa_1 = 50$, $w_1 = 0.99$ ✅ Match完美
- Peak 2: $\mu_2 = 45°$, $\kappa_2 = 10$, $w_2 = 0.003$ ❌ 位置不对
- Peak 3: $\mu_3 = 135°$, $\kappa_3 = 10$, $w_3 = 0.003$ ❌ 位置不对
- Peak 4: $\mu_4 = 225°$, $\kappa_4 = 10$, $w_4 = 0.004$ ❌ 位置不对

**计算Loss**（简化，不考虑匈牙利匹配）：

对于Peak 1（精确match）：
$$
\text{KL}(\text{VM}(0°, 50) \| \text{VM}(0°, 50)) \approx 0
$$

对于Peak 2-4（位置偏差大）：假设KL ≈ 5.0

总Loss：
$$
\mathcal{L} = 0.99 \times 0 + 3 \times (0.003 \times 5.0) = 0 + 0.045 = 0.045
$$

**结果**：Loss很小，但预测分布完全不对（只有1个正确的peak）！

### 4.3 策略B：固定weight = 0.25

**预测分布**：
- Peak 1: $\mu_1 = 0°$, $\kappa_1 = 50$, $w_1 = 0.25$ ✅
- Peak 2: $\mu_2 = 45°$, $\kappa_2 = 50$, $w_2 = 0.25$ ❌
- Peak 3: $\mu_3 = 135°$, $\kappa_3 = 50$, $w_3 = 0.25$ ❌
- Peak 4: $\mu_4 = 225°$, $\kappa_4 = 50$, $w_4 = 0.25$ ❌

**计算Loss**：

对于Peak 1：KL ≈ 0
对于Peak 2-4：假设KL ≈ 5.0（因为位置偏差大）

总Loss：
$$
\mathcal{L} = 4 \times 0.25 \times 2.5 = 2.5
$$

（这里取平均值，假设4个peak的平均KL是2.5）

**结果**：Loss较大！模型**必须**降低所有4个peak的KL才能降低loss，因此被迫学习正确的4峰位置。

---

## 5. KL散度方向的影响

### 5.1 KL(P||Q) vs KL(Q||P)

**我们使用的是 KL(Pred||GT)**，即 KL(P||Q)：

$$
\text{KL}(P \| Q) = \int p(\theta) \log \frac{p(\theta)}{q(\theta)} d\theta
$$

**性质**：
- **Zero-avoiding**：当 $p(\theta) > 0$ 但 $q(\theta) \approx 0$ 时，KL → ∞（惩罚很大）
- **Zero-forcing**：当 $q(\theta) > 0$ 但 $p(\theta) \approx 0$ 时，KL 贡献很小

**在我们的场景中**：
- P是预测分布：$p(\theta) = \sum w_i^{\text{pred}} \cdot \text{VM}_i$
- Q是GT分布：$q(\theta) = \sum w_j^{\text{GT}} \cdot \text{VM}_j$

如果某个预测peak的weight很小（$w_i \approx 0$），那么：
- 这个peak对 $p(\theta)$ 的贡献很小
- 即使它的位置完全错误，也不会导致 $p(\theta) > 0$ while $q(\theta) \approx 0$ 的情况
- 因此KL散度的"zero-avoiding"惩罚不会触发

**这就是为什么低权重peak可以"逃避"惩罚！**

### 5.2 如果使用 KL(GT||Pred) 会怎样？

如果反过来，使用 KL(Q||P) = KL(GT||Pred)：

$$
\text{KL}(Q \| P) = \int q(\theta) \log \frac{q(\theta)}{p(\theta)} d\theta
$$

**性质**：
- 当GT在某个位置有peak（$q(\theta) > 0$），但预测在那个位置没有（$p(\theta) \approx 0$）时，KL → ∞

**在我们的场景中**：
- GT在4个位置都有peak
- 如果预测只在1个位置有高权重peak，其他3个位置的 $p(\theta)$ 很小
- 那么 KL(GT||Pred) 会非常大！

**结论**：使用 KL(GT||Pred) 可以避免mode collapse，但：
- 需要修改代码（反转KL的参数顺序）
- 可能带来其他问题（比如预测多余的peak也会被惩罚）

---

## 6. 为什么固定Weight解决了问题？

### 6.1 消除了Weight的自由度

当 $w_i = 0.25$ 固定时：

$$
\mathcal{L} = \sum_{i=1}^{4} 0.25 \cdot \text{KL}(P_i \| Q_{\sigma(i)}) = 0.25 \sum_{i=1}^{4} \text{KL}(P_i \| Q_{\sigma(i)})
$$

**关键变化**：
- 所有4个peak对loss的贡献**平等**（都是0.25）
- 模型无法通过调低某些peak的weight来"作弊"
- 必须让**所有4个peak**的KL都尽量小才能降低总loss

### 6.2 强制学习多峰分布

**优化目标变为**：
$$
\min_{\mu, \kappa} \sum_{i=1}^{4} \text{KL}(\text{VM}(\mu_i, \kappa_i) \| \text{VM}(\mu_{\sigma(i)}^{\text{GT}}, \kappa_{\sigma(i)}^{\text{GT}}))
$$

这要求：
1. 每个预测peak的 $\mu_i$ 要接近某个GT peak的位置
2. 每个预测peak的 $\kappa_i$ 要接近GT的concentration
3. 通过匈牙利匹配，4个预测peak会自动分配到4个GT peak

**结果**：模型被迫学习真正的4峰分布！

---

## 7. 其他可能的解决方案

### 7.1 方案A：使用真正的分布KL散度

**思路**：直接计算mixture分布之间的KL，而不是component-wise KL的加权和。

$$
\mathcal{L} = \text{KL}\left(\sum_j w_j^{\text{GT}} \text{VM}_j^{\text{GT}} \,\Big\|\, \sum_i w_i^{\text{pred}} \text{VM}_i^{\text{pred}}\right)
$$

**优点**：
- 数学上更严格
- 直接优化整体分布的相似度
- 不会被weight exploitation

**缺点**：
- 计算复杂（需要数值积分或采样）
- 没有解析解
- 训练速度可能慢很多

### 7.2 方案B：Weight正则化

**思路**：添加正则项鼓励等权重。

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KL}} + \lambda \cdot \text{Reg}(w)
$$

其中：
$$
\text{Reg}(w) = \sum_{i=1}^{K} \left(w_i - \frac{1}{K}\right)^2
$$

**优点**：
- 简单易实现
- 可以通过 $\lambda$ 控制正则化强度

**缺点**：
- 需要调超参数 $\lambda$
- 无法完全消除mode collapse（只是缓解）
- 对于不等权重的真实分布不适用

### 7.3 方案C：反转KL散度方向

**思路**：使用 KL(GT||Pred) 代替 KL(Pred||GT)。

$$
\mathcal{L} = \sum_i w_i^{\text{GT}} \cdot \text{KL}(Q_i \| P_{\sigma^{-1}(i)})
$$

**优点**：
- 简单（只需反转参数）
- Zero-forcing性质会惩罚missing modes

**缺点**：
- 对多余的modes宽容（可能预测出5个peak）
- 需要修改匈牙利匹配逻辑（从GT → Pred匹配）

### 7.4 方案D：固定Weight（我们采用的方案）

**思路**：对于已知峰数K的问题，直接固定 $w_i = 1/K$。

$$
\mathcal{L} = \frac{1}{K} \sum_{i=1}^{K} \text{KL}(P_i \| Q_{\sigma(i)})
$$

**优点**：
- ✅ 最简单直接
- ✅ 完全消除weight的自由度
- ✅ 强制学习等权重多峰分布
- ✅ 适用于对称物体（4向对称 → 4个等权重peak）
- ✅ 训练稳定，收敛快

**缺点**：
- ❌ 仅适用于已知峰数K的情况
- ❌ 无法处理不等权重的真实分布（如果存在）
- ❌ 灵活性降低

**适用场景**：
- ✅ 多个正面方向已知（如4向对称）
- ✅ 各方向重要性相等（等权重假设合理）
- ✅ 数据标注是等权重的（我们的glassbox GT就是这样）

---

## 8. 实验验证

### 8.1 可学习Weight的实验结果

**观察到的现象**：
- 训练loss降到0.05左右（看似成功）
- 验证集可视化显示：
  - 1个peak的weight = 0.987
  - 其他3个peak的weight ≈ 0.004
  - 4个peak的位置 $\mu$ 分散，但并不在正确的0°/90°/180°/270°
- 泛化性能差：测试集上预测不稳定

**分析**：
- 模型确实学会了降低loss的"捷径"
- 通过weight collapse避免了学习真正的4峰分布
- 证实了我们的数学分析

### 8.2 固定Weight的实验结果

**实验配置**：
- 固定 $w_i = 0.25$ for all i
- 预设角度初始化：[0°, 90°, 180°, 270°]
- 其他超参数相同

**结果**：
- 训练loss降到0.0017（比可学习weight更低！）
- 验证集可视化显示：
  - 4个peak的weight = 0.25（固定）
  - 4个peak的位置 $\mu$ 精确对齐到0°/90°/180°/270°
  - 4个peak的形状均匀一致
- 泛化性能优秀：测试集loss = 0.0055

**分析**：
- 固定weight迫使模型学习正确的多峰分布
- Loss更低说明真正学到了GT分布，而不是"作弊"
- 证明了固定weight是这个问题的正确解决方案

---

## 9. 理论总结

### 9.1 Mode Collapse的本质

**Mode Collapse**是指模型只学到了多模态分布的**部分模态**，而忽略了其他模态。

**在我们的问题中**：
- GT是4个等权重的模态（4个peak）
- 可学习weight允许模型只关注1个模态（1个high-weight peak）
- 通过降低其他模态的权重来降低它们对loss的贡献

**根本原因**：
1. **Loss function设计问题**：Component-wise KL的加权和 ≠ 真正的分布KL
2. **Weight作为可学习参数**：给了模型"作弊"的自由度
3. **KL散度的方向性**：KL(P||Q)的zero-forcing性质允许P忽略Q的某些模态

### 9.2 固定Weight的理论依据

**为什么固定weight有效？**

1. **先验知识的引入**：对称物体的各个正面方向应该等权重
2. **消除冗余自由度**：Weight不应该是需要学习的，而是问题定义的一部分
3. **强制正确的优化目标**：让模型专注于学习位置 $\mu$ 和concentration $\kappa$

**数学上等价于**：
$$
\mathcal{L} = \frac{1}{K} \sum_{i=1}^{K} \text{KL}\left(\text{VM}(\mu_i^{\text{pred}}, \kappa_i^{\text{pred}}) \,\Big\|\, \text{VM}(\mu_{\sigma(i)}^{\text{GT}}, \kappa_{\sigma(i)}^{\text{GT}})\right)
$$

这是一个**纯粹的位置和形状匹配问题**，没有weight的干扰。

### 9.3 推广到其他问题

**什么时候应该固定weight？**

✅ **应该固定的情况**：
1. 多模态数量已知（K已知）
2. 各模态重要性相等（对称性、等权重假设）
3. 训练数据的GT是等权重的
4. 目标是学习模态位置，而不是模态权重

❌ **不应该固定的情况**：
1. 模态数量未知（需要自动推断K）
2. 各模态权重本质上不同（如人脸朝向：正面更常见）
3. GT包含不等权重信息
4. Weight是问题的重要输出（如聚类任务）

**我们的Fixed 4-peak MvM属于第一类**：
- 4向对称 → K=4已知
- 对称性 → 等权重合理
- GT标注就是等权重
- 目标是预测4个正面方向的角度，不是权重

因此**固定weight是正确的选择**！

---

## 10. 经验教训

### 10.1 Loss Function设计的陷阱

**教训1**：加权loss要小心weight的来源
- 如果用模型预测的weight来加权，模型可能exploit这一点
- 应该用固定的或GT的weight，而不是预测的weight

**教训2**：Component-wise loss ≠ Distribution-level loss
- 对于mixture model，单独优化每个component不等于优化整体分布
- 需要确保loss真正反映了目标任务

**教训3**：KL散度的方向很重要
- KL(P||Q) 和 KL(Q||P) 行为完全不同
- 选择方向时要考虑：哪个是"目标"，哪个是"逼近"

### 10.2 引入先验知识的重要性

**教训4**：不是所有参数都需要学习
- 如果某个参数有明确的先验（如等权重），直接固定它
- 减少模型自由度可以避免overfitting和mode collapse

**教训5**：问题建模要符合任务本质
- 对称物体的方向预测本质上是**等权重多峰分布**
- 把weight作为可学习参数是错误的建模
- 正确的建模应该只学习peak位置和concentration

### 10.3 调试深度学习的方法论

**教训6**：Loss低不一定代表学到了正确的东西
- 必须可视化预测结果
- 检查模型是否"作弊"（如weight collapse）
- 理解loss function的数学性质

**教训7**：从数学角度分析异常现象
- 当观察到反常行为时，做数学推导
- 找到loss function可以被exploit的地方
- 设计实验验证假设

**教训8**：简单的解决方案往往是最好的
- 固定weight比复杂的正则化更直接有效
- 先用先验知识简化问题，再考虑复杂模型

---

## 11. 结论

**核心发现**：
1. **可学习weight + component-wise加权KL loss** 会导致mode collapse
2. 模型通过让少数peak获得高权重、多数peak权重趋近0来降低loss
3. 这是loss function设计的问题，不是优化算法或网络结构的问题

**解决方案**：
- 对于等权重多峰分布（如对称物体），**固定weight = 1/K**
- 这迫使模型学习所有peak的正确位置
- 实验证明：固定weight后loss更低(0.0017 vs 0.05)、分布更准确

**理论意义**：
- 揭示了mixture model训练中weight exploitation的风险
- 强调了loss function设计要与任务目标一致
- 证明了引入先验知识（固定weight）的有效性

**实践价值**：
- 为Fixed K-peak MvM问题提供了明确的训练指导
- 避免了mode collapse导致的训练失败
- 为其他多模态学习任务提供了参考

---

## 附录A：von Mises分布的KL散度推导

两个von Mises分布之间的KL散度解析解：

**定义**：
$$
\text{VM}(\theta \,|\, \mu, \kappa) = \frac{\exp(\kappa \cos(\theta - \mu))}{2\pi I_0(\kappa)}
$$

其中 $I_0(\kappa)$ 是修正Bessel函数。

**KL散度**：
$$
\text{KL}(\text{VM}(\mu_p, \kappa_p) \,\|\, \text{VM}(\mu_q, \kappa_q))
$$

$$
= \int_0^{2\pi} p(\theta) \log \frac{p(\theta)}{q(\theta)} d\theta
$$

$$
= \log \frac{I_0(\kappa_q)}{I_0(\kappa_p)} + A(\kappa_p) \left[\kappa_p - \kappa_q \cos(\mu_p - \mu_q)\right]
$$

其中：
$$
A(\kappa) = \frac{I_1(\kappa)}{I_0(\kappa)}
$$

是von Mises分布的mean resultant length。

**性质**：
- 当 $\mu_p = \mu_q$ 且 $\kappa_p = \kappa_q$ 时，KL = 0
- 当 $|\mu_p - \mu_q|$ 增大时，KL增大（位置偏差惩罚）
- 当 $\kappa_p \neq \kappa_q$ 时，即使位置对齐也有KL（形状差异惩罚）

---

## 附录B：数值示例的完整计算

### B.1 KL散度的数值计算

假设：
- Peak A: $\mu_A = 0°$, $\kappa_A = 50$
- Peak B: $\mu_B = 45°$, $\kappa_B = 50$

**计算KL(A||B)**：

1. 计算Bessel函数：
   - $I_0(50) \approx 5.989 \times 10^{19}$
   - $I_1(50) \approx 5.986 \times 10^{19}$
   - $A(50) = I_1(50)/I_0(50) \approx 0.9995$

2. 计算角度差：
   - $\Delta\mu = 0° - 45° = -45° = -\pi/4$
   - $\cos(-\pi/4) = 0.707$

3. KL散度：
   $$
   \text{KL}(A \| B) = \log\frac{I_0(50)}{I_0(50)} + 0.9995 \times [50 - 50 \times 0.707]
   $$

   $$
   = 0 + 0.9995 \times [50 - 35.35] = 0.9995 \times 14.65 \approx 14.64
   $$

**结论**：45°的位置偏差导致KL ≈ 14.64，这是一个很大的值！

### B.2 Mode Collapse场景的Loss计算

**GT**（4个等权重peak）：
- $\mu^{\text{GT}} = [0°, 90°, 180°, 270°]$
- $\kappa^{\text{GT}} = [50, 50, 50, 50]$
- $w^{\text{GT}} = [0.25, 0.25, 0.25, 0.25]$

**预测（mode collapse）**：
- $\mu^{\text{pred}} = [0°, 45°, 135°, 225°]$
- $\kappa^{\text{pred}} = [50, 50, 50, 50]$
- $w^{\text{pred}} = [0.985, 0.005, 0.005, 0.005]$

**匈牙利匹配**（假设最优匹配）：
- Pred[0] → GT[0]: KL ≈ 0 (完美匹配)
- Pred[1] → GT[1]: KL ≈ 14.6 (45°偏差)
- Pred[2] → GT[2]: KL ≈ 14.6 (45°偏差)
- Pred[3] → GT[3]: KL ≈ 14.6 (45°偏差)

**加权Loss**：
$$
\mathcal{L} = 0.985 \times 0 + 0.005 \times 14.6 + 0.005 \times 14.6 + 0.005 \times 14.6
$$

$$
= 0 + 3 \times (0.005 \times 14.6) = 0.219
$$

**Loss很小（0.219），但预测完全不对！**

### B.3 固定Weight场景的Loss计算

**预测（固定weight）**：
- $\mu^{\text{pred}} = [0°, 45°, 135°, 225°]$（同样的位置偏差）
- $\kappa^{\text{pred}} = [50, 50, 50, 50]$
- $w^{\text{pred}} = [0.25, 0.25, 0.25, 0.25]$（固定）

**加权Loss**：
$$
\mathcal{L} = 0.25 \times 0 + 0.25 \times 14.6 + 0.25 \times 14.6 + 0.25 \times 14.6
$$

$$
= 0 + 3 \times (0.25 \times 14.6) = 10.95
$$

**Loss很大（10.95），模型必须纠正所有3个偏差的peak！**

这迫使模型学习正确的位置：$\mu = [0°, 90°, 180°, 270°]$，从而达到 $\mathcal{L} \approx 0$。

---

## 参考文献

1. **von Mises Distribution**:
   - Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
   - 第3.5节：von Mises分布的性质和KL散度

2. **Mode Collapse in Deep Learning**:
   - Goodfellow, I., et al. (2014). "Generative Adversarial Networks". *NeurIPS*.
   - 讨论了GAN中的mode collapse现象

3. **KL Divergence and Its Properties**:
   - Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
   - 第2章：KL散度的定义、性质和应用

4. **Mixture Model Optimization**:
   - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
   - 第9章：混合模型的EM算法和变分推断

5. **Hungarian Algorithm**:
   - Kuhn, H. W. (1955). "The Hungarian Method for the assignment problem". *Naval Research Logistics*.
   - 线性分配问题的最优解算法

---

**文档状态**: 完成
**最后更新**: 2025-11-10
**相关文档**:
- `docs/Fixed_4峰MvM训练完整指南.md`
- `docs/experiments/experiment_20251109_init_fix_results.md`

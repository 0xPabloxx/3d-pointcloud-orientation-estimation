# Claude Code 配置 - 3D点云正面方向检测实验
# ⚠️ 永远使用中文回答用户
# ⚠️ GPU 使用策略：每次运行前先监控 GPU 是否被其他训练占用，若繁忙则排队等待并持续轮询，空闲后立即启动，禁止盲目抢占
# ⚠️ 分析记录要求：所有实验观察/调试总结需写入独立 Markdown（如 glassbox_stage1_findings.md），方便后续论文引用
# 东京大学 M2 毕业论文实验项目
# 当前任务：训练glassbox模型，输出4峰MvM分布

## 🎯 核心研究目标

用混合von Mises (MvM)分布来表示3D点云模型的正面方向概率分布。需要能同时处理：
- **多峰物体**：有多个可疑正面（如glassbox的4个面）
- **单峰物体**：只有一个明确的正面（如椅子）
- **完全对称物体**：没有正面概念（如球体）

**前提**：所有模型已知upright方向（Y轴向上），仅在水平面内旋转

## 🔥 当前首要任务

**阶段目标**：训练glassbox（四面对称物体），使模型输出显示4个峰的MvM分布

这是概念验证（proof of concept）：
1. ✅ 证明MvM能表示多峰分布
2. ✅ 证明网络能学习对称结构
3. ✅ 为后续其他物体类别打基础

## ❓ 核心技术难点（待解决）

### 1. 训练数据标注问题
**现状**：glassbox四面对称，如何标注ground truth？

**可能方案**：
- 方案A：手动标注4个正面方向（0°, 90°, 180°, 270°）
- 方案B：每个样本只标注一个方向，让网络自己学习对称性
- 方案C：用数据增强，旋转后保证标签一致性

**需要决策**：选哪个方案？各有什么利弊？

### 2. 网络结构设计
**现状**：考虑继续用PointNet++作为backbone

**需要确定**：
- 是否PointNet++最合适？要不要试DGCNN？
- 输出层如何设计？
  - 输出什么：N=4个(μ, κ, weight)三元组
  - μ范围：[0, 2π)
  - κ范围：正数，如何约束？
  - weight范围：归一化到和为1

**预测头架构**：
```python
# 伪代码示意
class MvMPredictionHead(nn.Module):
    def forward(self, features):
        # features: (B, D) from PointNet++ backbone
        mu = self.mu_head(features)      # (B, N) -> [0, 2π)
        kappa = self.kappa_head(features) # (B, N) -> 正数
        weight = self.weight_head(features) # (B, N) -> softmax归一化
        return mu, kappa, weight
```

### 3. Kappa参数处理
**问题**：κ控制分布集中度，如何处理？

**方案对比**：
- 固定κ值（如κ=10）：简单但不灵活
- 预测κ值：需要用激活函数保证正数（softplus? exp?）
- κ太小：分布太平，没有峰
- κ太大：数值不稳定

**需要实验**：不同κ范围对训练的影响

### 4. Loss函数选择
**候选方案**：

**方案A：KL散度** (当前倾向)
```python
# 预测分布P和真值分布Q之间的KL散度
loss = KL(Q || P) 
# 需要pairwise matching（匈牙利算法）
```
- 优点：理论上合理，度量分布差异
- 缺点：需要匈牙利匹配，计算复杂

**方案B：Negative Log-Likelihood**
```python
# 直接优化对数似然
loss = -log P(θ_gt | predicted_MvM)
```
- 优点：简单直接
- 缺点：需要明确的θ_gt，多峰情况怎么办？

**需要决策**：先试哪个？

### 5. Overfitting防止
**担心**：glassbox数据量可能不够

**检测方法**：
- 监控train loss vs val loss
- 可视化训练集和验证集的预测结果
- 检查κ值是否过大（记忆具体样本）

**防止措施**：
- Dropout
- 数据增强（旋转、jittering）
- Early stopping
- Weight decay

### 6. 是否一定要用MvM？
**替代方案**：
- 方案A：直接回归多个向量
- 方案B：8方向分类（baseline）
- 方案C：连续角度回归 + 不确定性估计

**MvM的优势**：
- 天然处理周期性（0° = 360°）
- 可以表示多峰
- 输出概率分布，不是单点

**需要验证**：MvM是否真的比其他方法好？

## 📁 项目结构（简化）

```
3d-pointcloud-orientation/
├── data/
│   └── modelnet40/glassbox/     # glassbox点云数据
├── models/
│   ├── pointnet2_backbone.py    # PointNet++主干
│   ├── mvm_head.py               # MvM预测头
│   └── baseline_models.py        # 对比用的基线模型
├── utils/
│   ├── mvm_distribution.py       # MvM分布计算
│   ├── loss_functions.py         # KL loss, NLL loss等
│   ├── visualization.py          # 极坐标图可视化
│   └── data_loader.py            # 数据加载
├── train_glassbox.py             # 训练脚本
├── eval_glassbox.py              # 评估脚本
└── visualize_results.py          # 结果可视化
```

## 🎯 Claude Code 行为准则

### 1. 代码规范（精简版）

**Python基本要求**：
- 遵循PEP 8，最大行长100字符
- 函数加类型提示和docstring
- 变量名要清晰（避免单字母）

**PyTorch要求**：
- 用`@torch.no_grad()`做推理
- 处理好GPU/CPU设备
- 设随机种子保证可重现

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### 2. 实验协议

**命名规范**：
```
exp_YYYYMMDD_<描述>
例如：exp_20250110_glassbox_4peak_kl_loss
```

**每次实验记录**：
- config.yaml：所有超参数
- train.log：训练日志
- checkpoints/：模型保存
- results/：可视化结果

**必须记录的超参数**：
- 学习率、batch size、epochs
- N（MvM组件数量）
- κ的处理方式
- Loss函数类型
- 网络结构细节

### 3. 调试策略

**训练不收敛时检查**：
1. Loss是否在下降？画loss曲线
2. 梯度是否正常？检查梯度范数
3. 预测的μ, κ, weight是否合理？打印看看
4. 可视化早期的预测结果（epoch 10, 50, 100...）

**数值问题排查**：
- NaN/Inf：检查除零、log(0)、exp(过大值)
- κ过大：加上限或用更温和的激活函数
- 梯度爆炸：gradient clipping

**可视化调试**：
```python
# 每50个epoch可视化一次
if epoch % 50 == 0:
    visualize_predictions(model, val_samples)
    save_polar_plot(predictions, f"epoch_{epoch}.png")
```

### 4. 与我的交互方式

**实验建议时**：
- 清楚说明原因和权衡
- 给出具体的代码实现建议
- 估算需要的时间和资源

**遇到技术决策时**：
- 列出多个方案
- 分析每个方案的优缺点
- 推荐一个方案但征求我的意见

**报告进度时**：
- 当前loss值和趋势
- 可视化结果的观察
- 下一步建议

**写代码时**：
- 模块化，方便后续修改
- 关键部分加注释解释为什么这样做
- 先在小数据上测试（10-100个样本）

## 🔬 Glassbox实验计划

### 第一阶段：最简化验证（当前）

**目标**：证明概念可行

**简化假设**：
1. 固定N=4（4个峰）
2. 固定κ=10（先不预测κ）
3. 只用KL divergence loss
4. Ground truth：手动标注4个方向

**预期结果**：
- 可视化显示4个明显的峰
- 峰的位置大致在0°, 90°, 180°, 270°附近
- 训练loss稳定下降

**成功标准**：
- ✅ Loss收敛到<0.1
- ✅ 可视化出现4个峰
- ✅ 验证集效果也ok

### 第二阶段：改进（后续）

**逐步放开限制**：
1. 让网络预测κ值
2. 尝试不同的N值（N=2, 4, 8）
3. 对比KL loss vs NLL loss
4. 添加其他对称物体（如椅子的4个旋转）

### 失败情况应对

**如果出现4个峰但位置不对**：
- 检查ground truth标注
- 检查数据增强是否破坏了对称性
- 可视化网络看到的点云

**如果只出现1-2个峰**：
- κ可能太大，分布太集中
- Loss可能不合适，没有鼓励多峰
- 初始化可能有问题，加随机性

**如果完全不收敛**：
- 降低学习率
- 检查数据预处理
- 简化网络结构
- 从更简单的任务开始（如先做8方向分类）

## 📊 关键评估指标

**定量指标**：
- 训练loss和验证loss
- 峰的数量（应该是4个）
- 峰的位置（应该接近90°间隔）
- 峰的高度（weight应该接近0.25）

**定性指标**：
- 极坐标图是否显示清晰的4个峰
- 不同样本的结果是否一致
- 随机旋转输入后，预测是否跟着旋转

## ⚠️ 重要提醒

1. **先跑通小规模**：
   - 10个glassbox样本
   - 训练10个epoch
   - 确保pipeline没问题

2. **频繁可视化**：
   - 不要盲目训练200个epoch
   - 每10-20个epoch看一次结果
   - 早发现问题早调整

3. **记录一切**：
   - 每个实验的超参数
   - 每次修改的原因
   - 什么有效什么无效

4. **不要急于求成**：
   - glassbox是最简单的情况
   - 跑通了再扩展到其他物体
   - 一步一个脚印

## 🤝 协作约定

**我会主动**：
- 遇到技术决策时询问你的意见
- 实验结果出来后总结发现
- 提出下一步建议

**你可以随时**：
- 让我解释某段代码
- 让我对比不同方案
- 让我画图可视化结果
- 让我修改实验设置

**一起目标**：
让glassbox实验成功，输出漂亮的4峰MvM分布！🎯

---

**版本**: 2.0 (实验专用简化版)  
**更新**: 2025年11月  
**核心任务**: Glassbox 4峰MvM → 成功 → 扩展到其他物体

がんばろう！💪

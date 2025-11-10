# Claude Code 配置 - 3D点云正面方向检测实验
# 东京大学 M2 毕业论文实验项目
# 当前任务：训练glassbox模型，输出4峰MvM分布

## 🚨 强制工作规范（必须遵守）

### 规则1: 工作目录约束
**你必须始终在 claude worktree 中工作**

```bash
工作目录: /home/pablo/ForwardNet-claude/
分支: claude
```

**为什么这条规则存在：**
- 项目有多个worktree（main, claude, codex），各自独立
- claude分支是你的专属实验分支，避免影响主分支
- 所有配置、文档、实验都基于claude worktree路径

**Claude的反思（为什么我刚才违反了这条规则）：**
我在第一次创建文档时，在主worktree `/home/pablo/ForwardNet/` 创建了文档，然后才复制到claude worktree。这是错误的！

**正确做法：**
1. ✅ 始终 `cd /home/pablo/ForwardNet-claude` 后再工作
2. ✅ 创建任何文件前，确认当前目录是claude worktree
3. ✅ 提交和推送都在claude分支进行

**违反此规则的后果：**
- 文件可能被创建在错误的worktree
- git提交到错误的分支
- 路径混乱，难以追踪

---

### 规则2: 文档存储位置约束
**所有分析、实验、方法论文档必须存放在 `docs/` 目录下**

```
/home/pablo/ForwardNet-claude/
├── claude.md              # 仅此文件和project_structure.md放根目录
├── project_structure.md
│
└── docs/                  # 所有其他markdown文档必须放这里
    ├── 离散方向向量预测实现文档.md  # 方法论文档
    ├── analysis/           # 分析文档
    ├── experiments/        # 实验报告
    └── methods/            # 方法论（可选子目录）
```

**为什么这条规则存在：**
- 根目录只放核心配置文件，保持简洁
- docs/统一管理所有知识文档
- 便于后续整理成论文材料
- 避免根目录杂乱

**Claude的反思（为什么我刚才违反了这条规则）：**
我在创建"离散方向向量预测实现文档.md"时，直接放在了根目录，而不是`docs/`目录。虽然claude.md第455-476行明确规定了文档存储规范，但我没有仔细遵守！

**正确做法：**
1. ✅ 创建文档前，先 `mkdir -p docs`
2. ✅ 文档直接创建在 `docs/离散方向向量预测实现文档.md`
3. ✅ 如果误放根目录，立即 `mv xxx.md docs/`

**文档分类：**
- `analysis_YYYYMMDD_*.md` → `docs/analysis/`
- `experiment_YYYYMMDD_*.md` → `docs/experiments/`
- `method_*.md` 或技术文档 → `docs/` 或 `docs/methods/`

**违反此规则的后果：**
- 根目录混乱，难以管理
- 后续整理文档时需要额外工作
- 可能导致文档丢失或重复

---

**这两条规则的优先级高于本文档中的其他所有内容！**

如果Claude违反了这两条规则，必须：
1. 立即停止当前操作
2. 反思为什么违反（写在claude.md中）
3. 修正错误（移动文件、切换目录等）
4. 重新执行正确的操作

---

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

## 🚨 深度学习训练的强制要求

**在运行任何深度学习训练前，你必须：**

1. **明确告知用户**：
   - 说明你要运行什么训练
   - 预计需要多长时间
   - 会使用什么资源（GPU、内存等）
   - **等待用户确认后再开始**

2. **必须有可视化**：
   - 实时或定期输出训练进度
   - Loss曲线图（训练集/验证集）
   - 关键指标的变化
   - 让用户能随时看到当前状态
   - **绝对不允许"黑盒"训练**

3. **监控与汇报**：
   - 定期检查训练状态（如每10个epoch）
   - 发现异常立即停止并报告
   - 训练结束后汇报最终结果

4. **可中断性**：
   - 使用后台运行或可中断的方式
   - 定期保存checkpoint
   - 用户可以随时中止

**示例流程**：
```
Claude: "我准备训练glassbox模型，预计需要90分钟（100 epochs），
        会使用GPU约6GB显存。训练过程中每10个epoch会保存可视化结果。
        可以开始吗？"
User: "可以" / "等一下" / "改成50 epochs"
Claude: [开始训练并定期汇报进度]
```

**违反上述要求的后果**：
- 用户可能中断你的操作
- 浪费计算资源
- 失去用户信任

## 📝 研究分析文档记录要求

**每次重要的分析、调试、实验发现，都必须：**

1. **创建独立的Markdown文档**：
   - 文件名格式：`analysis_YYYYMMDD_<主题>.md`
   - 例如：`analysis_20251109_glassbox_training_failure.md`
   - 存放在项目根目录或`docs/`文件夹

2. **文档内容应包含**：
   - 问题概述（目标、预期、实际结果）
   - 实验设置（数据、模型、超参数）
   - 详细分析（现象、诊断、根本原因）
   - 解决方案（多个方案对比、优先级）
   - 后续计划
   - 代码引用和数据样本

3. **写作要求**：
   - **专业性**：适合作为论文参考材料
   - **完整性**：任何人读了都能复现问题
   - **结构化**：用标题、列表、代码块、公式
   - **量化**：有具体数字、图表、对比

4. **何时创建文档**：
   - 发现训练失败的根本原因
   - 完成重要的消融实验
   - 实现关键的技术突破
   - 遇到反直觉的现象
   - 用户明确要求

5. **文档的用途**：
   - 日后写论文的素材
   - 记录实验失败的教训
   - 团队知识沉淀
   - 复现和debug的参考

**示例情况**：
```
场景1: "训练loss降不下去"
→ 创建 analysis_YYYYMMDD_loss_plateau_diagnosis.md
→ 记录loss曲线、模型预测、梯度分析、根本原因

场景2: "发现新的数据增强方法有效"
→ 创建 analysis_YYYYMMDD_rotation_augmentation_ablation.md
→ 对比有无增强的效果、分析为什么有效

场景3: "修改初始化后训练成功"
→ 创建 analysis_YYYYMMDD_initialization_fix.md
→ 记录before/after对比、可视化、性能提升
```

**不要做**：
- ❌ 重要分析只在聊天中说，不记录
- ❌ 分析文档写得太简略，缺少细节
- ❌ 不记录失败的实验（失败也是宝贵经验）

---

## 📄 Markdown文档命名与存储规范

**为了保持项目清晰，所有分析文档必须遵循统一规范：**

### 文档分类与命名规则

1. **实验分析文档**：
   - **格式**: `analysis_YYYYMMDD_<主题描述>.md`
   - **示例**:
     - `analysis_20251109_glassbox_training_failure.md` (问题诊断)
     - `analysis_20251115_rotation_augmentation_ablation.md` (消融实验)
   - **用途**: 记录问题分析、根因诊断、调试过程

2. **实验结果报告**：
   - **格式**: `experiment_YYYYMMDD_<实验名称>_results.md`
   - **示例**:
     - `experiment_20251109_init_fix_results.md` (实验1结果)
     - `experiment_20251115_chair_multimodal_results.md` (椅子多峰实验)
   - **用途**: 完整记录实验配置、结果、分析、结论

3. **方法论文档**：
   - **格式**: `method_<方法名称>.md`
   - **示例**:
     - `method_mvm_distribution.md` (MvM分布理论)
     - `method_hungarian_matching.md` (匈牙利匹配算法)
   - **用途**: 详细说明技术方法、算法原理

4. **项目管理文档**：
   - **格式**: `<功能>_<描述>.md`
   - **示例**:
     - `project_structure.md` (项目结构说明)
     - `TODO.md` (待办事项)
     - `CHANGELOG.md` (变更记录)
   - **用途**: 项目组织、规划、记录

### 存储位置规范

```
/home/pablo/ForwardNet-claude/
├── claude.md                           # 核心配置文档（本文件）
├── project_structure.md                # 项目结构说明
│
├── docs/                               # 文档主目录
│   ├── analysis/                       # 分析文档
│   │   ├── analysis_20251109_*.md
│   │   └── analysis_20251115_*.md
│   │
│   ├── experiments/                    # 实验报告
│   │   ├── experiment_20251109_*.md
│   │   └── experiment_20251115_*.md
│   │
│   └── methods/                        # 方法论文档
│       ├── method_mvm_distribution.md
│       └── method_hungarian_matching.md
│
└── [临时] 根目录markdown              # 初期可以放根目录，后续整理到docs/
```

**规则**：
- ✅ **新文档**: 直接创建在根目录，便于快速访问
- ✅ **定期整理**: 每周或实验阶段结束后，移动到`docs/`对应子目录
- ✅ **重要文档**: `claude.md`和`project_structure.md`始终保持在根目录

### 文档质量要求

**每份markdown必须包含**：
1. **标题与元数据**：
   ```markdown
   # 标题
   **日期**: YYYY-MM-DD
   **作者**: Claude / 用户名
   **实验ID**: exp_YYYYMMDD (如适用)
   **相关文件**: 列出相关的代码文件
   ```

2. **核心章节**（根据文档类型调整）：
   - 问题概述 / 实验目标
   - 方法/设置
   - 结果/发现
   - 分析/讨论
   - 结论/下一步

3. **量化数据**：
   - 具体数字（loss值、准确率等）
   - 对比表格（before/after）
   - 可视化引用（图片路径）

4. **代码引用规范**：
   ```markdown
   修改了 `models/pointnet_pp_mvM.py:69-82` 中的初始化代码：
   ```python
   # 代码片段
   ```

5. **结论明确**：
   - ✅ 成功 / ❌ 失败 / ⚠️ 部分成功
   - 关键发现（1-3条）
   - 可操作的下一步

---

## 🐍 Python文件命名与注释规范

**为了提高代码可读性和可维护性，所有Python文件必须遵循：**

### 文件命名规范

**格式**: `<功能>_<模型/方法>_<数据/类别>_<其他>.py`

**组成部分**：
1. **功能前缀**（必需）：
   - `train_` - 训练脚本
   - `eval_` - 评估脚本
   - `test_` - 测试脚本
   - `vis_` - 可视化脚本
   - `dataloader_` - 数据加载器
   - `preprocess_` - 数据预处理

2. **模型/方法**（必需）：
   - `pointnetpp` - PointNet++
   - `dgcnn` - DGCNN
   - `mvm` - MvM分布方法
   - `single_vm` - 单峰von Mises
   - `8dir` - 8方向分类

3. **数据/类别**（推荐）：
   - `glassbox` - 仅glassbox类别
   - `chair` - 仅chair类别
   - `modelnet40` - 全ModelNet40数据集
   - `symmetric` - 对称物体

4. **其他标识**（可选）：
   - `augmented` - 带数据增强
   - `debug` - debug版本
   - `baseline` - 基线方法

**示例**：
- ✅ `train_pointnetpp_mvm_glassbox_augmented.py` - 清晰明确
- ✅ `eval_pointnetpp_mvm_modelnet40.py` - 评估全数据集
- ✅ `dataloader_glassbox_augmented.py` - 数据加载器
- ✅ `vis_mvm_predictions_polar.py` - MvM预测的极坐标可视化
- ❌ `train_glassbox_only.py` - 不清楚用什么模型/方法
- ❌ `train.py` - 太笼统
- ❌ `test_new.py` - 无意义的命名

### 文件头注释规范（强制要求）

**每个Python文件开头必须包含**：

```python
"""
<一句话描述这个文件的功能>

详细说明：
- 模型/方法: <PointNet++ + MvM / DGCNN + 8-dir等>
- 数据集: <glassbox / ModelNet40全集等>
- 训练策略: <数据增强方式、loss函数等>
- 输出: <模型保存位置、日志位置等>

使用方法：
    python <filename>.py [--参数]

示例：
    python train_pointnetpp_mvm_glassbox_augmented.py --epochs 100 --lr 0.001

作者: <Claude / 用户名>
创建日期: YYYY-MM-DD
最后修改: YYYY-MM-DD
关联文档: <相关的analysis或experiment markdown文件>
"""
```

**最小示例**：
```python
"""
训练PointNet++ + MvM模型在glassbox类别上（带12旋转增强）

模型: PointNet++ backbone + MvM预测头（K=4峰）
数据: ModelNet40 glassbox (271样本，12旋转增强)
Loss: KL散度 + Hungarian匹配
输出: results/glassbox_YYYYMMDD_HHMMSS/

作者: Claude
创建: 2025-11-09
关联: experiment_20251109_init_fix_results.md
"""
```

### 函数/类注释规范

**重要函数必须有docstring**：
```python
def calculate_kl_divergence(pred_mu, pred_kappa, gt_mu, gt_kappa):
    """
    计算两个von Mises分布之间的KL散度

    Args:
        pred_mu (torch.Tensor): 预测的均值角度，形状(B, K)
        pred_kappa (torch.Tensor): 预测的集中度参数，形状(B, K)
        gt_mu (torch.Tensor): GT均值角度，形状(B, K)
        gt_kappa (torch.Tensor): GT集中度参数，形状(B, K)

    Returns:
        torch.Tensor: KL散度值，形状(B,)

    Notes:
        使用Hungarian算法进行峰的匹配
    """
    pass
```

**复杂逻辑必须有行内注释**：
```python
# 打破对称性：预设4个方向[0°, 90°, 180°, 270°]
# 如果用zeros初始化，4个峰会始终重叠，梯度为0
initial_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
```

### 重要常量/超参数注释

```python
# 超参数
EPOCHS = 50              # 总训练轮数
BATCH_SIZE = 8           # 批大小（受GPU内存限制）
LR = 5e-4                # 学习率（Adam优化器）
NUM_ROTATIONS = 12       # 数据增强旋转数量（30°间隔）

# MvM配置
MAX_K = 4                # 最大峰数量（glassbox是4面对称）
KAPPA_INIT = 0.0         # kappa初始化值（通过bias控制）
```

### 目录内组织规范

```
/home/pablo/ForwardNet-claude/
├── train_*.py                          # 所有训练脚本
├── eval_*.py                           # 所有评估脚本
├── vis_*.py                            # 所有可视化脚本
├── dataloader_*.py                     # 数据加载器
│
├── models/                             # 模型定义
│   ├── pointnet_pp_mvM.py             # PointNet++ + MvM
│   ├── pointnet_pp_8dir.py            # PointNet++ + 8方向分类
│   └── ...
│
├── utils/                              # 工具函数
│   ├── mvm_utils.py                   # MvM分布计算
│   ├── loss_functions.py              # Loss函数
│   └── visualization.py               # 可视化工具
│
└── data_process/                       # 数据预处理脚本
    └── 2d_multi_peak_MvM_gt_1.py      # GT生成
```

---

**版本**: 3.0
**更新**: 2025-11-09
**核心任务**: Glassbox 4峰MvM → 成功 → 扩展到其他物体

がんばろう！💪
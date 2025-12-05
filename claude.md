# Claude Code 配置 - 3D点云正面方向检测实验
# 东京大学 M2 毕业论文实验项目
# 当前任务：训练glassbox模型，输出4峰MvM分布

---

## 📋 精简版速查 (TL;DR)

### 🚨 强制规则（违反必停！）
1. **工作目录**：始终在 `/home/pablo/ForwardNet-claude/`，分支 `claude`
2. **文档位置**：只有 `claude.md` 和 `project_structure.md` 放根目录，其他 markdown 放 `docs/`
3. **数据集划分**：seed=42，train/val/test 严格分离，增强只用于训练集，测试集只评估一次

### 🎯 核心目标
用混合 von Mises (MvM) 分布预测 3D 点云的正面方向概率分布，处理多峰/单峰/完全对称物体。

### 🔧 常用工具
```bash
# 论文截图工具
python pointcloud_screenshot_viewer.py  # http://localhost:8051

# 对称性标注工具
python data_annotation/annotate_symmetry_web_v3.py  # http://localhost:8052
```

### 📁 关键文件
- 标注数据：`data_annotation/symmetry_annotations.json`
- 数据集：`data/full_mn40_normal_resampled_ply/`
- 训练脚本：`train_*.py`

### ⚠️ 训练前必须
1. 告知用户：训练内容、预计时间、资源占用
2. **等待确认后再开始**
3. 有可视化输出（loss曲线、定期checkpoint）
4. 发现异常立即停止报告

### 📝 命名规范
- 文档：`analysis_YYYYMMDD_<topic>.md`、`experiment_YYYYMMDD_<name>_results.md`
- Python：`<功能>_<模型>_<数据>.py`（如 `train_pointnetpp_mvm_glassbox.py`）
- 文件头必须有 docstring（模型/数据/loss/用法）

### 🗣️ 语言
如果用户没有特别要求其他语言，请用中文回答。

---

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

---

### 规则3: 数据集划分与防止数据泄露（强制要求）
**机器学习实验的基本准则：训练集、验证集、测试集必须严格分离**

```
数据集划分规范：
- Train Set:  用于模型训练，可以使用数据增强
- Val Set:    用于超参数调优和early stopping，不使用增强
- Test Set:   只在最终评估时使用一次，不使用增强
```

**为什么这条规则存在：**
- 数据泄露会导致过高估计模型性能
- 在训练集上评估 = 作弊，结果无效
- 在全数据集上验证 = 包含了训练数据，结果不可信
- 测试集必须是"未见过的数据"，只用于最终报告

**Claude的反思（为什么我刚才违反了这条规则）：**
在`verify_weight_fix.py`中，我使用了全部271个样本来验证weight分布，这包括了189个训练样本！这是严重的数据泄露错误。虽然后来用独立测试集(28样本)重新验证了，但暴露了流程问题。

**正确做法：**

1. **✅ 数据集划分（实验开始时）**：
   ```python
   # 必须使用固定的随机种子
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)

   # 划分比例：7:2:1（或根据数据量调整）
   n_total = len(samples)
   n_train = int(0.7 * n_total)
   n_val = int(0.2 * n_total)

   train_samples = samples[:n_train]
   val_samples = samples[n_train:n_train + n_val]
   test_samples = samples[n_train + n_val:]

   # 明确打印数据集大小
   print(f"[Data Split] Train: {len(train_samples)}, "
         f"Val: {len(val_samples)}, Test: {len(test_samples)}")
   ```

2. **✅ 数据增强（只在训练集）**：
   ```python
   # 训练集：使用增强
   train_ds = Dataset(train_samples, augment=True, n_rotations=12)

   # 验证/测试集：不使用增强
   val_ds = Dataset(val_samples, augment=False)
   test_ds = Dataset(test_samples, augment=False)
   ```

3. **✅ 验证/评估脚本（必须明确指定数据集）**：
   ```python
   # ❌ 错误：使用全数据集
   all_samples = load_all_samples()  # 包含训练集！
   verify(model, all_samples)

   # ✅ 正确：明确使用测试集
   test_samples = load_test_split_only()  # 只加载测试集
   verify(model, test_samples)

   # 验证数据集大小
   assert len(test_samples) == 28, "Test set size mismatch!"
   ```

4. **✅ 训练过程中的监控**：
   ```python
   # 训练循环
   for epoch in range(epochs):
       train_loss = train_one_epoch(model, train_loader)  # 只用训练集
       val_loss = validate(model, val_loader)             # 用验证集

       # 保存最佳模型（基于验证集）
       if val_loss < best_val_loss:
           save_checkpoint(model, "best_model.pth")

   # 最终测试（只运行一次！）
   test_loss = test(model, test_loader)
   print(f"Final Test Loss: {test_loss}")
   ```

**检查清单（每次评估前必查）：**
- [ ] 确认使用的是哪个数据集（train/val/test）
- [ ] 验证数据集大小是否正确（打印出来检查）
- [ ] 确认没有加载全数据集
- [ ] 确认测试集没有用于超参数调优
- [ ] 确认训练/验证/测试集使用相同的随机种子划分

**常见数据泄露场景：**

| 错误场景 | 为什么错误 | 正确做法 |
|---------|-----------|---------|
| 在全数据集上评估 | 包含训练集数据 | 只在测试集评估 |
| 用测试集调超参数 | 测试集变成验证集 | 用验证集调参 |
| 数据增强用在测试集 | 改变了数据分布 | 测试集不增强 |
| 多次在测试集上评估 | 间接地"优化"测试集 | 测试集只用一次 |
| 验证脚本没指定数据集 | 默认可能加载全数据 | 显式指定test_samples |

**报告实验结果时：**
```markdown
## 实验结果

**数据集划分**（seed=42）：
- Train: 189 samples (×12 rotation augmentation = 2268)
- Val: 54 samples (no augmentation)
- Test: 28 samples (no augmentation)

**验证集表现**（用于模型选择）：
- Best Val Loss: 0.0012 @ Epoch 47

**测试集表现**（最终评估，仅运行一次）：
- Test Loss: 0.0019
- Weight accuracy: [0.250, 0.250, 0.250, 0.250]
```

**违反此规则的后果：**
- 实验结果不可信，无法发表
- 过度估计模型性能
- 无法正确评估泛化能力
- 可能需要重新训练和评估

---

### 🚨 Phase 2 对称性分类的数据集划分实现（必须遵守！）

**当前实现**（`dataloader_symmetry.py`）：

```python
# 固定随机种子（确保可复现）
np.random.seed(42)

# 划分比例：70% / 15% / 15%
# - Train: 用于训练，使用旋转增强
# - Val:   用于超参数调优和模型选择，不使用增强
# - Test:  最终评估，只运行一次，不使用增强

# 分层采样（按K值类别）
for K, samples in K_to_samples.items():
    np.random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_samples.extend(samples[:n_train])
    val_samples.extend(samples[n_train:n_train+n_val])
    test_samples.extend(samples[n_train+n_val:])
```

**关键点**：
1. ✅ **固定随机种子 seed=42** - 确保每次运行划分相同
2. ✅ **分层采样** - 每个K值类别按相同比例划分（处理不平衡数据）
3. ✅ **训练时增强，验证/测试时不增强**
4. ✅ **自动打印数据集大小** - 每次加载都会显示train/val/test数量

**验证数据集划分正确性**：

```python
# 运行数据集测试
python dataloader_symmetry.py

# 应该看到类似输出：
# [TRAIN] 数据集大小: 75
# [TRAIN] 类别分布:
#   K= 0 (class 1):    1 (  1.3%)
#   K= 1 (class 2):   72 ( 96.0%)
#   K= 4 (class 4):    2 (  2.7%)
#
# [VAL] 数据集大小: 15
# [VAL] 类别分布:
#   K= 1 (class 2):   15 (100.0%)
#
# [TEST] 数据集大小: 20
# [TEST] 类别分布:
#   K= 0 (class 1):    1 (  5.0%)
#   K= 1 (class 2):   17 ( 85.0%)
#   K= 2 (class 3):    1 (  5.0%)
#   K= 4 (class 4):    1 (  5.0%)

# ✅ 验证：75 + 15 + 20 = 110 (总标注数)
# ✅ 验证：没有样本重复出现在多个集合中
```

**训练脚本中的使用**（`train_symmetry_classifier.py`）：

```python
# 自动划分数据集
train_loader, val_loader, test_loader = get_symmetry_dataloaders(
    annotation_file='data_annotation/symmetry_annotations.json',
    data_dir='data/full_mn40_normal_resampled_ply',
    batch_size=16,
    num_workers=4,
    num_points=10000
)

# 训练循环
for epoch in range(epochs):
    # 只在训练集上训练
    train_loss, train_acc = train_epoch(model, train_loader, ...)

    # 只在验证集上评估（用于模型选择）
    val_loss, val_acc, _, _ = evaluate(model, val_loader, ...)

    # 保存最佳模型（基于验证集）
    if val_acc > best_acc:
        save_checkpoint(model, 'best_model.pth')

# 最终测试（只运行一次！加载最佳模型后）
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc, confusion_matrix, class_acc = evaluate(
    model, test_loader, ...
)
```

**⚠️ 绝对禁止的操作**：

```python
# ❌ 错误1：在全数据集上训练
all_data = load_all_annotations()  # 包含测试集！
train(model, all_data)  # 数据泄露！

# ❌ 错误2：在测试集上调参
for lr in [1e-3, 1e-4, 1e-5]:
    train(model, train_loader)
    acc = evaluate(model, test_loader)  # 测试集变验证集了！
    if acc > best_acc:
        best_lr = lr

# ❌ 错误3：测试集多次评估
test_acc_1 = evaluate(model_v1, test_loader)
# 调整模型...
test_acc_2 = evaluate(model_v2, test_loader)  # 间接优化测试集！

# ❌ 错误4：没有固定随机种子
# 每次运行划分不同，无法复现！
```

**✅ 正确的实验流程**：

```python
# 1. 数据集划分（固定seed=42）
train_loader, val_loader, test_loader = get_symmetry_dataloaders(...)

# 2. 训练 + 验证（可以多次）
for experiment in [exp1, exp2, exp3]:
    model = create_model(experiment.config)

    for epoch in range(200):
        train_loss = train(model, train_loader)  # 只用训练集
        val_loss = validate(model, val_loader)   # 只用验证集

        if val_loss < best_val_loss:
            save_best_model(model)

    # 记录验证集最佳性能（用于对比实验）
    best_val_acc = load_and_evaluate(best_model, val_loader)

# 3. 最终测试（只运行一次！）
best_experiment = select_best_from_validation_results()
final_model = load_checkpoint(best_experiment.best_model_path)
test_acc = evaluate(final_model, test_loader)  # ← 只此一次！

# 4. 报告结果
print(f"Test Accuracy: {test_acc}")  # 这是论文中报告的数字
```

**报告实验结果模板**：

```markdown
## Phase 2 对称性分类实验结果

**数据集划分**（seed=42，分层采样）：
- Train: 75 samples (70%)
  - 使用旋转增强（θ ∈ [0, 2π)）
- Val: 15 samples (15%)
  - 不使用增强
- Test: 20 samples (15%)
  - 不使用增强

**验证集表现**（用于模型选择）：
- Exp 1.1 (GlobalConcat): Val Acc = 0.95
- Exp 1.2 (PointConcat):  Val Acc = 0.93
- Exp 1.3 (NoUpright):    Val Acc = 0.90

**最终测试集表现**（仅运行一次）：
- Best Model: Exp 1.1 (GlobalConcat)
- Test Accuracy: 0.92
- Per-class Accuracy: {K=-1: 0.0, K=0: 1.0, K=1: 0.95, K=2: 0.0, K=4: 1.0}
- Confusion Matrix: ...
```

---

**这三条规则的优先级高于本文档中的其他所有内容！**

如果Claude违反了这些规则，必须：
1. 立即停止当前操作
2. 反思为什么违反（写在claude.md中）
3. 修正错误（重新划分数据、重新评估等）
4. 重新执行正确的操作

---

## 🎯 核心研究目标

用混合von Mises (MvM)分布来表示3D点云模型的正面方向概率分布。需要能同时处理：
- **多峰物体**：有多个可疑正面（如glassbox的4个面）
- **单峰物体**：只有一个明确的正面（如椅子）
- **完全对称物体**：没有正面概念（如球体）

**前提**：所有模型已知upright方向（Y轴向上），仅在水平面内旋转

## 🔥 当前任务清单

### 任务1：搭建多种方法的训练代码框架
- 搭建几个不同方法的训练代码
- **注意**：如果 dataloader 等模块有很大不同、导致无法公平对比，必须向用户报告！

### 任务2：直接回归方向向量（Baseline）
- 搭建直接回归 (cos θ, sin θ) 的训练脚本
- 不使用 von Mises 分布，作为最简单的 baseline

### 任务3：单峰 von Mises 回归
- 搭建回归单一 μ 和 κ 的 von Mises 训练脚本
- 适用于单正面物体（如椅子、显示器）

### 任务4：多峰 von Mises 回归（固定峰数）
- 搭建 **2峰** von Mises 训练脚本（180°对称物体）
- 搭建 **4峰** von Mises 训练脚本（90°对称物体）
- 两个独立的训练脚本

### 任务5：可学习权重的4峰 von Mises
- 搭建固定 K=4，但 weight 也可学习的训练脚本
- 让网络学习每个峰的重要性

### 任务6：保持扩展性
- 随时方便搭建新的脚本，包括：
  - 不固定峰个数的脚本
  - 学习预测峰数量的分类器（方便二阶段方法）

### 任务7：代码结构要求（强制）
- **解耦设计**：保证代码结构简单、清晰、模块化
- **Backbone 可替换**：能够将 PointNet++ 换成 DGCNN 或 Point Transformer，且不影响其他部分
- **Loss 可替换**：更换 loss 函数必须简单清晰，不需要大改代码

### 任务8：实验前确认（强制）
- **做任何实验前必须和用户确定细节！**
- 包括：数据集划分、超参数、评估指标、预期结果等

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

### 🚨 输出方向/角度 μ 的表示规范（强制要求！）

**输出方向 μ 必须用 (cos θ, sin θ) 表示，并且必须做归一化！！！！**

**原因**：
1. 角度有周期性问题（0° = 360°），直接回归角度值会导致训练困难
2. 用 (cos θ, sin θ) 可以自然处理周期性边界
3. 归一化确保输出在单位圆上，物理意义明确

**实现方式**：
```python
# ❌ 错误：直接输出角度
mu = self.mu_head(features)  # 输出 θ ∈ [0, 2π)，周期性边界难处理

# ✅ 正确：输出 (cos θ, sin θ) 并归一化
mu_raw = self.mu_head(features)  # (B, N, 2)
mu_normalized = F.normalize(mu_raw, dim=-1)  # 归一化到单位圆
# mu_normalized[:, :, 0] = cos θ
# mu_normalized[:, :, 1] = sin θ
```

**注意事项**：
- 网络输出 2 维向量，不是 1 维角度
- 必须在输出层做 `F.normalize`，不能省略
- Loss 计算时使用 (cos θ, sin θ) 形式，避免角度差的周期性问题

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

### 第二阶段：对称性分类+路由架构（新实验方向）

**核心思路**：训练一个对称性判别网络，根据点云几何特征预测有几个正面，然后路由到专门的MvM网络。

**架构设计**：
```
输入点云 → 对称性分类器（4分类）
              ↓
   [1峰 / 2峰 / 4峰 / 完全对称]
              ↓
         路由决策（硬路由或软路由）
              ↓
   ┌──────────┼──────────┬──────────┐
   ↓          ↓          ↓          ↓
 K=1 MvM   K=2 MvM   K=4 MvM   均匀分布
```

**对称性定义**（基于几何特征，非类别标签）：
- **1峰（单正面）**：chair, monitor, airplane - 只有一个明确正面
- **2峰（双正面）**：door, tv_stand - 有两个相对的正面（180°对称）
- **4峰（四正面）**：glass_box, night_stand - 四个正面（90°对称）
- **完全对称**：ball, cone, bowl - 任意方向都是正面

**实验计划**：
1. **数据标注阶段**（当前首要任务）：
   - 开发交互式标注工具
   - 可视化点云，手动判断对称性类型
   - 标注至少50个样本验证概念

2. **小规模验证**：
   - 先训练3分类：1峰 vs 4峰 vs 完全对称（跳过2峰）
   - 使用50个标注样本
   - 验证分类器是否能学习几何对称性

3. **全量训练**：
   - 扩展到完整4分类
   - 训练专门的MvM网络（共享backbone）
   - 评估级联误差影响

**技术细节**：
- **分类器架构**：
  ```python
  backbone (PTv3/PointNet++) → global features → FC → 4-way softmax
  输出: [P(K=1), P(K=2), P(K=4), P(symmetric)]
  ```

- **路由机制**（两种方案）：
  - 方案A：硬路由（argmax选择一个网络）
  - 方案B：软路由（加权组合多个MvM预测）

- **loss函数**：
  - 分类器：交叉熵loss
  - MvM网络：KL散度 + Hungarian匹配（沿用第一阶段）
  - 联合训练或分阶段训练

**优势**：
- ✅ 每个MvM网络专注一个K值，可能更精确
- ✅ 可解释性强，明确知道为什么选择某个K
- ✅ 灵活性高，可以为特殊对称性设计专门结构
- ✅ 调试友好，可以独立验证每个组件

**挑战**：
- ⚠️ 训练复杂度：需要训练5个网络（1分类+4 MvM）
- ⚠️ 数据标注成本：需要手动标注所有样本的对称性
- ⚠️ 级联误差：分类器错误会直接影响MvM预测
- ⚠️ 数据平衡：某些对称类型样本可能很少

**成功标准**：
- 分类器准确率 > 90%（在测试集上）
- 路由后的MvM预测loss < 单一网络baseline
- 可视化显示：正确K值 + 准确的峰位置

**失败应对**：
- 如果分类器准确率低(<70%)：改用方案C（固定最大K，用weight=0表示无峰）
- 如果级联误差严重：尝试方案D（软路由，概率加权）
- 如果标注成本过高：探索自监督对称性检测

**数据标注规范**（见下方标注工具部分）

---

## 🚀 启动时自动提醒：常用工具

**每次打开此项目时，Claude应主动提醒用户以下工具：**

### 1. 点云截图展示器（论文图片用）

**用途**：为论文截取高质量的点云3D图片

**启动命令**：
```bash
python pointcloud_screenshot_viewer.py
# 浏览器打开 http://localhost:8051
```

**核心功能**：
- 7种背景颜色（白/灰/黑/浅蓝/奶油色等）
- 12种点云颜色（纯色：蓝/黑/白/红/黄/紫等）
- 6种渐变色（Blues/Grays/Viridis/Jet/Plasma/Turbo）
- 12种相机预设（正面/侧面/俯视/等轴测等）
- **一键对齐功能**：勾选"Apply alignment"后根据标注自动旋转到标准姿态
- **快捷视角按钮**：Front / Side / Top 快速切换
- **批量截图**：一键保存3视图（正/侧/顶）或4个等轴测视角
- 输出位置：`paper/figures/pointclouds/`

**依赖标注文件**：`data_annotation/symmetry_annotations.json`（用于一键对齐）

---

### 2. 对称性标注工具 v3（数据标注用）

**用途**：标注点云的对称类型和前向方向

**启动命令**：
```bash
python data_annotation/annotate_symmetry_web_v3.py
# 浏览器打开 http://localhost:8052
```

**核心功能**：
- 5种对称类型：1个正面 / 2个正面 / 4个正面 / 完全对称 / 没有正面
- 6个方向标注：-Z / +Z / -X / +X / -Y / +Y
- **矫正后预览**：选择方向后实时显示旋转矫正效果
- 类别选择和进度追踪
- 自动保存到 JSON / CSV / Markdown
- 断点续标

**输出文件**：
- `data_annotation/symmetry_annotations.json`
- `data_annotation/symmetry_annotations.csv`
- `data_annotation/symmetry_annotations.md`

---

### 提醒模板

当用户打开项目时，Claude可以说：

> **工具提醒**：检测到你在ForwardNet项目中。以下工具可用：
> 1. `python pointcloud_screenshot_viewer.py` → 论文截图工具 (端口8051)
> 2. `python data_annotation/annotate_symmetry_web_v3.py` → 标注工具 (端口8052)
>
> 需要我启动哪个吗？

---

## 🏷️ 对称性数据标注工具

### 最新工具：Web版标注工具 v2（推荐）

**工具文件**：`annotate_symmetry_web_v2.py`

**核心功能**：
- ✅ Web浏览器界面（Dash + Plotly），无需X11转发
- ✅ 3D交互式点云可视化，支持旋转/缩放/平移
- ✅ **类别选择功能**：在Web界面直接切换类别标注
- ✅ **实时进度追踪**：显示每个类别的标注进度
- ✅ **5种对称类型**：1个正面/2个正面/4个正面/完全对称/没有正面
- ✅ **6个方向标注**：-Z/+Z/-X/+X/-Y/+Y（检测数据是否对齐）
- ✅ 自动保存（双重标注后立即保存）
- ✅ 断点续标（自动从第一个未标注样本继续）
- ✅ 三重输出：JSON + CSV + Markdown报告

**启动工具**：
```bash
cd /home/pablo/ForwardNet-claude

# 默认启动（自动扫描所有类别）
python data_annotation/annotate_symmetry_web_v2.py

# 指定端口
python data_annotation/annotate_symmetry_web_v2.py --port 8051

# 浏览器访问
http://localhost:8051
```

**Web界面操作**：
1. **选择类别**：顶部下拉菜单选择要标注的类别（显示进度）
2. **标注对称性**：点击右侧"1个正面/2个正面/4个正面/完全对称/没有正面"按钮
3. **标注方向**：点击方向按钮（-Z/+Z/-X/+X/-Y/+Y）
4. **完成标注**：选择对称性+方向后自动保存
5. **导航**：点击"下一个"继续标注，或切换类别
6. **查看进度**：右上角显示当前类别进度

**类别选择功能**（新增）：
- 下拉菜单显示所有40个类别及其进度
- 自动从剩余样本最多的类别开始
- 切换类别时自动跳转到该类别的第一个未标注样本
- 实时更新每个类别的标注数量

**数据对齐检测**（重要）：
- 数据集理论上应该对齐（正面为-Z方向）
- 如果标注时发现正面不是-Z，说明数据需要矫正
- Markdown报告会列出所有需要矫正的样本及旋转角度

**输出文件**（所有类别共享同一个文件）：
- `data_annotation/symmetry_annotations.json` - 完整标注数据
- `data_annotation/symmetry_annotations.csv` - Excel可读表格
- `data_annotation/symmetry_annotations.md` - 人类可读报告

**JSON格式示例**：
```json
{
  "airplane/airplane_0001.ply": {
    "file": "airplane/airplane_0001.ply",
    "K": 1,
    "symmetry_name": "1个正面",
    "front_direction": "-Z",
    "aligned": true,
    "index": 0
  },
  "glass_box/glass_box_0001.ply": {
    "file": "glass_box/glass_box_0001.ply",
    "K": 4,
    "symmetry_name": "4个正面",
    "front_direction": "+X",
    "aligned": false,
    "index": 15
  }
}
```

**辅助工具**：
1. **类别进度查看器**：`category_progress_viewer.py`
   - 显示所有类别的标注进度表格
   - 可以选择类别启动标注工具
   - 用于快速了解整体进度

2. **标注数据处理器**：`process_annotations.py`
   - 按对称性分类数据
   - 筛选需要矫正的数据
   - 生成统计报告

### 旧版工具（可选）

**工具文件**：`annotate_symmetry.py`（matplotlib GUI版本，需要X11）

**功能特性**：
- ✅ 3D交互式点云可视化（matplotlib）
- ✅ 鼠标拖动旋转查看不同角度
- ✅ 键盘快捷键快速标注
- ✅ 自动保存标注结果（JSON格式）
- ✅ 支持断点续标（重新运行时加载已有标注）
- ✅ 实时显示标注进度统计

**使用方法**：
```bash
cd /home/pablo/ForwardNet-claude

# 方式1: 使用默认路径
python annotate_symmetry.py

# 方式2: 指定数据目录
python annotate_symmetry.py \
  --data_dir data/full_mn40_normal_resampled_ply \
  --output data_annotation/symmetry_annotations.json
```

**快捷键**：
- `1` - 标注为1峰（单正面）
- `2` - 标注为2峰（双正面，180°对称）
- `4` - 标注为4峰（四正面，90°对称）
- `S` - 标注为完全对称
- `N` - 下一个样本
- `P` - 上一个样本
- `Q` - 保存并退出

**标注流程**：
1. 启动工具后会显示第一个点云
2. 鼠标拖动旋转点云，观察几何对称性
3. 按对应数字键（1/2/4/S）标注对称类型
4. 工具会自动保存并跳转到下一个样本
5. 随时按Q保存并退出，下次可以继续

**输出格式**（JSON）：
```json
{
  "glass_box/glass_box_0001.ply": {
    "file": "glass_box/glass_box_0001.ply",
    "K": 4,
    "name": "4-peak (四正面,90°)",
    "index": 0
  },
  "chair/chair_0001.ply": {
    "file": "chair/chair_0001.ply",
    "K": 1,
    "name": "1-peak (单正面)",
    "index": 15
  }
}
```

### 标注指南

**判断对称性的方法**：

1. **观察点云的几何结构**：
   - 旋转点云到不同角度
   - 寻找是否有重复的"正面"视角
   - 不要依赖类别名称，完全基于几何特征

2. **1峰（单正面）**：
   - 只有一个明显的正面方向
   - 从其他角度看明显不同
   - 例子：椅子（背靠方向）、显示器（屏幕朝向）

3. **2峰（180°对称）**：
   - 有两个相对的正面（旋转180°后相同）
   - 例子：门（两面都可以是正面）、长方形桌子

4. **4峰（90°对称）**：
   - 有四个正面（旋转90°后相同）
   - 例子：玻璃盒、正方形桌子、某些床头柜

5. **完全对称**：
   - 任意角度看起来都一样（或非常相似）
   - 例子：球体、圆锥、碗、花瓶

**边界情况处理**：
- 如果不确定是1峰还是2峰：默认选1峰
- 如果接近对称但有细微差异：选择最接近的类型
- 如果物体变形或不规则：基于大致形状判断

### 标注质量控制

**初期验证（前50个样本）**：
1. 每个对称类型至少标注10个样本
2. 标注完后检查分布是否合理
3. 随机抽查5个样本，重新标注验证一致性

**全量标注建议**：
- 每次标注50-100个样本后休息，避免疲劳
- 定期检查已标注样本，确保一致性
- 记录难以判断的样本，后续讨论

### 标注数据使用

**训练分类器时**：
```python
import json

# 加载标注
with open('data_annotation/symmetry_annotations.json', 'r') as f:
    annotations = json.load(f)

# 创建数据集
train_samples = []
for file_path, ann in annotations.items():
    full_path = Path(data_dir) / file_path
    K_label = ann['K']  # 0/1/2/4
    train_samples.append((full_path, K_label))

# 划分训练/验证/测试集
# (参考规则3: 数据集划分规范)
```

---

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
---

## 📝 标注工具开发日志（2025-11-25）

### 本次更新内容

**背景**：为Phase 2实验准备数据，需要对ModelNet40数据集进行对称性标注。

### 1. 文件重组

**问题**：标注相关文件散落在根目录，且data/文件夹包含数据集，不应上传GitHub

**解决方案**：
- 创建`data_annotation/`专用文件夹
- 移动所有标注工具到`data_annotation/`
- 移动标注结果文件到`data_annotation/`
- 更新`.gitignore`排除`data/`目录

**文件结构**：
```
ForwardNet-claude/
├── data_annotation/           # 新增：标注相关文件
│   ├── README.md             # 文件夹说明
│   ├── ANNOTATION_GUIDE.md   # 使用指南
│   │
│   ├── annotate_symmetry_web_v2.py    # 主工具（Web版v2）
│   ├── category_progress_viewer.py    # 进度查看器
│   ├── annotation_stats.py            # 速度统计工具（新增）
│   ├── annotate_by_category.py        # 按类别标注
│   ├── process_annotations.py         # 数据处理
│   │
│   ├── symmetry_annotations.json      # 标注数据
│   ├── symmetry_annotations.csv       # Excel格式
│   └── symmetry_annotations.md        # Markdown报告
│
└── data/                      # 数据集（不上传GitHub）
    ├── full_mn40_normal_resampled_ply/
    └── ...
```

### 2. Web标注工具v2核心功能

**主要改进**：
1. **类别选择器**：
   - 在Web界面顶部添加下拉菜单
   - 显示所有40个类别及其进度（如`airplane (101/726)`）
   - 自动从剩余样本最多的类别开始

2. **实时进度显示**：
   - 当前类别进度（X/Y已标注，百分比，进度条）
   - 全局进度统计
   - 类别进度实时更新

3. **进度显示修复**：
   - **Bug**：之前显示全局已标注数/当前类别总数（如108/84，错误）
   - **修复**：现在正确显示当前类别已标注数/当前类别总数（如2/84）
   - **实现**：使用`self.category_progress[self.current_category]`获取当前类别数据

4. **技术实现**：
   ```python
   # 新增方法
   _scan_categories()              # 扫描所有类别
   _calculate_all_category_progress()  # 计算各类别进度
   _load_category_files(category)  # 加载指定类别文件
   _get_category_progress_display()   # 生成进度显示HTML
   
   # 新增回调
   - 类别切换回调：切换类别时重新加载文件并跳转
   - 主回调增强：保存后更新类别进度显示
   ```

### 3. 标注速度统计工具

**文件**：`data_annotation/annotation_stats.py`

**功能**：
- 显示整体标注进度（已标注/总数/百分比）
- 估算完成时间（基于不同标注速度）
- 显示各类别剩余数量排名
- 轻量级，不占用资源

**使用方法**：
```bash
cd data_annotation
python annotation_stats.py
```

**输出示例**：
```
📊 标注进度与速度统计
══════════════════════════════════════════

🎯 整体进度
  总样本数: 12,311
  已标注: 110 (0.89%)
  剩余: 12,201 (99.11%)

💡 完成时间估算（基于标注速度）:
    @  10 样本/小时 →  1220.1 小时 ( 50.84 天)
    @  20 样本/小时 →   610.0 小时 ( 25.42 天)
    @  30 样本/小时 →   406.7 小时 ( 16.95 天)
    @  50 样本/小时 →   244.0 小时 ( 10.17 天)
    @ 100 样本/小时 →   122.0 小时 (  5.08 天)

📋 类别进度（剩余最多的前10个）
  chair       2/989    (0.2%)
  sofa        0/780    (0.0%)
  airplane  101/726  (13.9%)
  ...
```

### 4. 数据安全保证

**关键点**：
1. **不丢失进度**：
   - 所有类别共享同一个JSON文件
   - 切换类别不会清空已有标注
   - 每次保存都追加而非覆盖

2. **三重备份**：
   - JSON（程序读取）
   - CSV（Excel可读）
   - Markdown（人类可读报告）

3. **断点续标**：
   - 启动时自动加载已有标注
   - 自动从第一个未标注样本继续
   - 可随时关闭，下次继续

### 5. Git提交

**提交内容**：
- 添加`data_annotation/`文件夹及所有工具
- 更新claude.md文件路径
- 更新.gitignore排除data/

**提交信息**：`Add symmetry annotation tools and reorganize data`

**Remote branch**：`claude`

### 关键代码位置

**Web工具v2核心逻辑**：
- 文件：`data_annotation/annotate_symmetry_web_v2.py`
- 类别扫描：`_scan_categories()` (第69-80行)
- 进度计算：`_calculate_all_category_progress()` (第82-103行)
- 类别切换回调：`switch_category()` (第555-565行)
- 进度显示修复：第672-686行（显示当前类别进度而非全局）

**进度统计工具**：
- 文件：`data_annotation/annotation_stats.py`
- 整体进度计算：`display_stats()` (第88-150行)
- 完成时间估算：`estimate_completion_time()` (第73-77行)

### 后续LLM快速上手指南

**如果需要修改标注工具**：
1. 主文件：`data_annotation/annotate_symmetry_web_v2.py`
2. 关键类：`WebSymmetryAnnotatorV2`
3. 布局定义：`_build_layout()` (第296行)
4. 回调逻辑：`_setup_callbacks()` (第543行)

**如果需要处理标注数据**：
1. 标注文件：`data_annotation/symmetry_annotations.json`
2. 处理工具：`data_annotation/process_annotations.py`
3. 数据格式：
   ```json
   {
     "category/filename.ply": {
       "K": -1/0/1/2/4,
       "symmetry_name": "1个正面",
       "front_direction": "-Z",
       "aligned": true/false
     }
   }
   ```

**如果需要查看进度**：
```bash
# 快速查看
python data_annotation/annotation_stats.py

# 详细查看
python data_annotation/category_progress_viewer.py
```

**如果需要启动标注工具**：
```bash
# 推荐：Web版v2（支持类别选择）
python data_annotation/annotate_symmetry_web_v2.py --port 8051
# 浏览器访问 http://localhost:8051
```

---


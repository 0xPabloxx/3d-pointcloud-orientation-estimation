# ForwardNet-Claude 项目状态总结

**最后更新**: 2025-11-09 20:30
**当前分支**: claude (worktree)
**项目状态**: ✅ 阶段性成功 - Glassbox 4峰MvM训练已完成
**目的**: 为LLM提供完整的项目上下文

---

## 📋 快速导航

- [1. 项目概述](#1-项目概述)
- [2. 当前仓库状态](#2-当前仓库状态)
- [3. 核心成果](#3-核心成果)
- [4. 关键文件说明](#4-关键文件说明)
- [5. 实验结果汇总](#5-实验结果汇总)
- [6. 下一步计划](#6-下一步计划)
- [7. 如何继续工作](#7-如何继续工作)

---

## 1. 项目概述

### 1.1 研究背景

**项目名称**: 3D点云正面方向检测 (ForwardNet)
**研究机构**: 东京大学 M2 毕业论文项目
**研究目标**: 用混合von Mises (MvM)分布预测3D物体的正面方向概率分布

**核心问题**:
- 传统方法：单一正面方向预测 → 无法处理多正面物体
- 本项目：输出概率分布 → 可以表示多个可能的正面方向

### 1.2 技术方案

**模型架构**:
```
输入: 3D点云 (10,000个点, xyz坐标)
    ↓
PointNet++ Backbone (特征提取)
    ↓
MvM Prediction Head (预测头)
    ↓
输出: K个峰的MvM分布参数
      - μ (均值角度): K个方向
      - κ (集中度): K个浮点数
      - π (权重): K个浮点数 (和为1)
```

**当前聚焦**: 4向对称物体 (K=4)
- 例如: glassbox (玻璃盒)，正面方向在[0°, 90°, 180°, 270°]

**数据集**: ModelNet40
- 40个物体类别
- 每个类别约200-400个样本
- 当前实验类别: glassbox (271个样本)

---

## 2. 当前仓库状态

### 2.1 Git Worktree结构

```
/home/pablo/ForwardNet/              # 主仓库 (main分支)
├── claude.md                        # (已过时，在worktree中更新)
└── ...

/home/pablo/ForwardNet-claude/       # Claude工作区 (claude分支) ⭐ 当前
├── claude.md                        # 核心配置文档 (v3.0)
├── project_structure.md             # 项目结构说明
├── PROJECT_STATUS_20251109.md       # 本文件
│
├── docs/                            # 文档目录 (新建)
│   ├── README.md
│   ├── analysis/                    # 分析文档
│   │   ├── analysis_20251109_glassbox_training_failure.md
│   │   └── analysis_20251109_4向对称物体数据集合并可行性分析.md
│   ├── experiments/                 # 实验报告
│   │   ├── experiment_20251109_init_fix_results.md
│   │   └── experiment_20251109_data_augmentation_ablation_results.md
│   └── methods/                     # 方法论文档 (待填充)
│
├── train_pointnetpp_mvm_glassbox_augmented.py      # 主训练脚本 (12旋转增强)
├── train_pointnetpp_mvm_glassbox_no_augment.py     # 无增强训练脚本
├── dataloader_glassbox_augmented.py                # 数据加载器
│
├── models/
│   └── pointnet_pp_mvM.py           # ⭐ 核心模型 (已修复初始化)
│
├── results/                         # 训练结果
│   ├── glassbox_only_20251109_183051/              # 实验1: 增强版
│   └── glassbox_no_augment_20251109_201200/        # 实验2: 无增强版
│
└── data/ -> /home/pablo/ForwardNet/data/           # 软链接到主仓库数据

/home/pablo/ForwardNet-codex/        # Codex工作区 (codex分支)
└── (独立开发，暂不涉及)
```

### 2.2 分支说明

| 分支 | 位置 | 用途 | 状态 |
|------|------|------|------|
| `main` | /home/pablo/ForwardNet | 主分支 | 稳定 |
| **`claude`** | /home/pablo/ForwardNet-claude | **Claude开发分支** | **活跃** ⭐ |
| `codex` | /home/pablo/ForwardNet-codex | Codex开发分支 | 独立 |

**当前工作分支**: `claude`

### 2.3 关键提交历史

**最近3次提交**:

1. **c4a121a09** (2025-11-09 19:52) - 最新
   ```
   Add project documentation and enforce naming conventions
   - 添加 claude.md (命名规范)
   - 添加 project_structure.md (项目结构)
   - 重命名训练脚本遵循新规范
   ```

2. **a5d8b907d** (2025-11-09 19:41)
   ```
   Fix MvM training for glassbox by breaking initialization symmetry
   - 修复初始化（预设角度）
   - Val Loss: 0.74 → 0.0017 (435× 改进)
   - 添加分析和实验报告文档
   ```

3. **25e1f74a0** (更早)
   ```
   Add .gitignore to exclude data and cache files
   ```

---

## 3. 核心成果

### 3.1 问题诊断与解决 ✅

**问题**: Glassbox训练loss卡在0.74，无法下降

**根本原因**: Zeros初始化导致4个峰重叠，梯度为0，无法分离

**解决方案**: 预设角度初始化
```python
# models/pointnet_pp_mvM.py:69-82
initial_angles = [0, math.pi/2, math.pi, 3*math.pi/2]  # 0°, 90°, 180°, 270°

with torch.no_grad():
    for i, angle in enumerate(initial_angles):
        self.head_mu.bias[2*i]   = math.cos(angle)
        self.head_mu.bias[2*i+1] = math.sin(angle)
```

**结果**: Val Loss从0.74降至0.0017 (435倍改进)

**详细分析**: 见 `docs/analysis/analysis_20251109_glassbox_training_failure.md`

### 3.2 实验1: 初始化修复 + 数据增强 ✅

**配置**:
- 训练样本: 217 × 12 = 2604 (12旋转增强)
- Epochs: 50
- 初始化: 预设角度 [0°, 90°, 180°, 270°]

**结果**:
- Best Val Loss: **0.0017** @ epoch 45
- Test Loss: 0.0131
- 收敛速度: ~20 epochs
- 训练时间: ~50分钟

**质量**:
- ✅ 4个峰清晰均匀
- ✅ 角度准确 (接近0°/90°/180°/270°)
- ✅ 旋转不变性强

**详细报告**: 见 `docs/experiments/experiment_20251109_init_fix_results.md`

### 3.3 实验2: 数据增强消融实验 ✅

**目的**: 验证数据增强的作用

**配置**:
- 训练样本: 189 (无增强)
- 其他参数与实验1相同

**结果**:
- Best Val Loss: **0.0060** @ epoch 46
- Test Loss: 0.0055 (⚠️ 测试集小，28样本)
- 收敛速度: ~15 epochs
- 训练时间: ~6分钟

**质量**:
- ⚠️ 4个峰不均匀
- ⚠️ 倾向于单峰或主峰
- ⚠️ 缺乏旋转不变性

**结论**:
- ✅ 无增强也能训练成功 (预设初始化的功劳)
- ✅ 增强版质量更好 (loss低3.5倍，可视化更好)
- ✅ 数据增强显著提升旋转不变性

**详细报告**: 见 `docs/experiments/experiment_20251109_data_augmentation_ablation_results.md`

### 3.4 核心洞察 💡

**发现1**: **预设初始化 >> 数据量**
```
189样本 + 预设初始化 → Val Loss = 0.0060 ✅
2604样本 + Zeros初始化 → Val Loss = 0.74 ❌
```

**发现2**: **数据增强提升质量**
```
无增强: Val Loss = 0.0060, 4峰不均
增强版: Val Loss = 0.0017, 4峰完美
→ 3.5倍改进
```

**发现3**: **Glassbox相对简单**
- 完美对称，参数确定
- 少量样本（189）足以学习
- 为更复杂物体提供了信心

---

## 4. 关键文件说明

### 4.1 配置与文档

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `claude.md` | Claude工作规范、命名规范、实验协议 | ⭐⭐⭐ 必读 |
| `project_structure.md` | 项目结构详细说明 | ⭐⭐⭐ 必读 |
| `PROJECT_STATUS_20251109.md` | 本文件，项目总体状态 | ⭐⭐⭐ 最新 |

### 4.2 训练脚本

| 文件 | 功能 | 状态 |
|------|------|------|
| `train_pointnetpp_mvm_glassbox_augmented.py` | 主训练脚本 (12旋转增强) | ✅ 推荐 |
| `train_pointnetpp_mvm_glassbox_no_augment.py` | 无增强版 (消融实验) | ✅ 对比用 |
| `train_multi_peaks_vonMises_KL.py` | 旧版本 | ⚠️ 有bug (zeros初始化) |

### 4.3 数据加载器

| 文件 | 功能 |
|------|------|
| `dataloader_glassbox_augmented.py` | Glassbox专用，支持旋转增强 |
| `dataloader_multi_peak_vonMises.py` | 通用MvM数据加载器 |

### 4.4 模型定义

| 文件 | 内容 | 关键修改 |
|------|------|---------|
| `models/pointnet_pp_mvM.py` | PointNet++ + MvM预测头 | ✅ 已修复初始化 (L69-82) |
| `models/pointnet_pp.py` | PointNet++ backbone | - |

### 4.5 文档系统

```
docs/
├── README.md                        # 文档组织说明
├── analysis/                        # 问题分析
│   ├── analysis_20251109_glassbox_training_failure.md
│   │   → 诊断: Zeros初始化问题
│   │   → 解决方案: 预设角度初始化
│   │
│   └── analysis_20251109_4向对称物体数据集合并可行性分析.md
│       → 评估: 合并多类别的可行性
│       → 风险: 标注不一致、几何对称≠语义对称
│       → 建议: 分阶段执行 (筛选→Pilot→大规模)
│
├── experiments/                     # 实验报告
│   ├── experiment_20251109_init_fix_results.md
│   │   → 实验1: 初始化修复 + 增强
│   │   → 结果: Val Loss 0.0017, 435× 改进
│   │
│   └── experiment_20251109_data_augmentation_ablation_results.md
│       → 实验2: 消融实验
│       → 对比: 增强 vs 无增强
│       → 结论: 增强版质量更好
│
└── methods/                         # 方法论 (待创建)
    └── (计划: MvM理论, Hungarian匹配等)
```

### 4.6 实验结果

```
results/
├── glassbox_only_20251109_183051/   # 实验1: 增强版
│   ├── best_model.pth               # Val Loss = 0.0017
│   ├── checkpoints/
│   ├── figs/
│   │   ├── final_predictions.png   # ⭐ 4峰完美
│   │   ├── loss_curve.png
│   │   └── predictions_epoch_*.png
│   └── config.txt
│
└── glassbox_no_augment_20251109_201200/  # 实验2: 无增强
    ├── best_model.pth               # Val Loss = 0.0060
    ├── figs/
    │   ├── final_predictions.png   # ⚠️ 4峰不均
    │   └── loss_curve.png
    └── config.txt
```

---

## 5. 实验结果汇总

### 5.1 定量对比表

| 实验 | 初始化 | 增强 | 训练样本 | Val Loss | Test Loss | 收敛Epoch | 训练时间 |
|------|-------|------|---------|---------|-----------|---------|---------|
| 0 (baseline) | Zeros | ✅ 12旋转 | 2604 | **0.74** | - | ❌ 不收敛 | - |
| **1 (最优)** | **预设** | ✅ 12旋转 | 2604 | **0.0017** ⭐ | 0.0131 | 20 | 50分钟 |
| 2 (消融) | 预设 | ❌ 无 | 189 | 0.0060 | 0.0055 | 15 | 6分钟 |

**关键发现**:
- Zeros初始化 → 完全失败 (loss=0.74)
- 预设初始化 + 无增强 → 成功但质量一般 (loss=0.0060)
- **预设初始化 + 增强 → 最优** (loss=0.0017)

### 5.2 定性对比

| 维度 | 实验0 (Zeros) | 实验1 (预设+增强) | 实验2 (预设+无增强) |
|------|-------------|----------------|-----------------|
| 4峰清晰度 | ❌ 单峰 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 角度准确性 | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Weight均匀性 | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 旋转不变性 | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 训练稳定性 | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 5.3 可视化对比

**实验1 (预设+增强) - 优秀**:
```
4个峰均匀分布在0°/90°/180°/270°
Weight都接近0.25
κ值合理
旋转不变性强
```

**实验2 (预设+无增强) - 中等**:
```
4个峰不均匀
某些样本退化为单峰
倾向于某个主方向 (如0°或45°)
缺乏旋转不变性
```

**实验0 (Zeros) - 失败**:
```
4个峰重叠
只预测单一方向
完全不收敛
```

---

## 6. 下一步计划

### 6.1 短期计划 (本周)

**计划A**: 深入分析无增强版
- [ ] 在完整271样本测试集上重新评估
- [ ] 可视化更多样本，量化"单峰倾向"
- [ ] 分析模型对不同角度输入的响应

**计划B**: 准备多类别合并 (已完成可行性分析)
- [ ] 创建 `analyze_modelnet40_symmetry.py` 脚本
- [ ] 自动检查所有类别的GT统计
- [ ] 筛选出2-5个候选类别
- [ ] 可视化验证对称性

**计划C**: 论文写作准备
- [x] 整理实验报告 (已完成)
- [ ] 创建高质量可视化对比图
- [ ] 撰写方法论文档 (MvM理论、Hungarian匹配)

### 6.2 中期计划 (下周)

**如果选择多类别合并**:
1. Pilot实验: glassbox + 1-2个候选类别
2. 对比混合训练 vs 单类别训练
3. 分析哪些类别组合效果好

**如果聚焦单类别优化**:
1. 测试不同旋转数量 (6旋转、24旋转)
2. 测试其他数据增强策略
3. 优化模型架构或loss函数

### 6.3 长期目标 (论文前)

1. **方法普适性验证**
   - 扩展到2向对称物体 (chair)
   - 扩展到6/8向对称物体
   - 验证预设初始化的通用性

2. **论文实验完善**
   - 消融实验 (已完成)
   - 对比实验 (vs baseline方法)
   - 泛化实验 (跨类别测试)

3. **代码开源准备**
   - 完善文档
   - 创建复现指南
   - 清理代码

---

## 7. 如何继续工作

### 7.1 LLM接手指南

**你可以立即开始的任务**:

#### 任务1: 创建GT统计分析脚本 ⭐⭐⭐
```python
# 文件: analyze_modelnet40_symmetry.py
# 目的: 自动检查所有类别，筛选4向对称候选

功能:
1. 读取所有ModelNet40类别的GT文件
2. 统计每个类别的K值分布、weight均匀性、μ间隔
3. 输出候选类别列表

输出示例:
  glassbox:  ✅ K4=100%, weight_std=0.02, μ_interval=90°
  table:     ⚠️ K4=65%, weight_std=0.15, μ_interval=85°
  desk:      ✅ K4=95%, weight_std=0.05, μ_interval=88°
```

**参考**: `docs/analysis/analysis_20251109_4向对称物体数据集合并可行性分析.md`

#### 任务2: 标准化测试评估
```python
# 目的: 在相同测试集上对比实验1和实验2

步骤:
1. 加载实验1的best model (glassbox_only_20251109_183051/best_model.pth)
2. 加载实验2的best model (glassbox_no_augment_20251109_201200/best_model.pth)
3. 在相同的271样本测试集上评估
4. 对比Test Loss和可视化质量
```

#### 任务3: 可视化对比图生成
```python
# 目的: 为论文创建对比图

图1: Loss曲线对比 (增强 vs 无增强)
图2: 极坐标预测对比 (并排: GT | 无增强Pred | 增强Pred)
图3: 训练样本数 vs Val Loss 关系图
```

### 7.2 关键命令

**切换到工作目录**:
```bash
cd /home/pablo/ForwardNet-claude
```

**查看当前分支**:
```bash
git branch --show-current
# 应该输出: claude
```

**训练脚本**:
```bash
# 增强版 (推荐)
python3 train_pointnetpp_mvm_glassbox_augmented.py

# 无增强版 (消融实验)
python3 train_pointnetpp_mvm_glassbox_no_augment.py
```

**查看文档**:
```bash
# 核心配置
cat claude.md

# 项目结构
cat project_structure.md

# 最新状态 (本文件)
cat PROJECT_STATUS_20251109.md
```

**查看实验结果**:
```bash
# 实验1结果
ls results/glassbox_only_20251109_183051/figs/

# 实验2结果
ls results/glassbox_no_augment_20251109_201200/figs/
```

### 7.3 重要提醒 ⚠️

**深度学习训练规范** (见 `claude.md`):
1. ✅ **必须先询问用户** 再运行训练
2. ✅ **必须有可视化** (loss曲线、预测图)
3. ✅ **定期汇报进度** (如每10 epochs)
4. ✅ **记录实验配置** (超参数、结果)

**文件命名规范** (见 `claude.md`):
```
训练脚本: train_<模型>_<方法>_<数据>_<其他>.py
数据加载: dataloader_<数据>_<特征>.py
分析文档: analysis_YYYYMMDD_<主题>.md
实验报告: experiment_YYYYMMDD_<实验名>_results.md
```

**Git工作流**:
```bash
# 当前在claude分支
git branch  # 确认

# 提交前检查
git status
git diff

# 提交
git add <files>
git commit -m "..."
git push origin claude
```

---

## 8. 已知问题与限制

### 8.1 当前问题

1. **测试集不统一**
   - 实验1: 271样本
   - 实验2: 28样本
   - 导致Test Loss不可比

2. **仅在单一类别验证**
   - 只测试了glassbox
   - 其他对称物体未验证

3. **未测试不同增强强度**
   - 只测试了12旋转
   - 6/24旋转效果未知

### 8.2 技术债务

1. **工具函数分散**
   - Loss函数、可视化等分散在训练脚本中
   - 需要重构到 `utils/` 目录

2. **旧版脚本未清理**
   - `train_multi_peaks_vonMises_KL.py` 等旧脚本仍在
   - 需要标注状态或删除

3. **缺少自动化测试**
   - 没有单元测试
   - 修改模型后需要手动验证

---

## 9. 资源与参考

### 9.1 关键论文 (待补充)

- PointNet++ (Qi et al., 2017)
- von Mises distribution
- Hungarian algorithm for assignment

### 9.2 代码仓库

- **GitHub**: https://github.com/0xPabloxx/3d-pointcloud-orientation-estimation
- **分支**: `claude` (当前活跃)

### 9.3 数据集

- **ModelNet40**: Princeton大学，40类物体，~12,000个模型
- **本地路径**: `/home/pablo/ForwardNet-claude/data/` (软链接)

---

## 10. 总结

### 当前状态: ✅ 阶段性成功

**已完成**:
- ✅ 诊断并修复训练失败问题
- ✅ 实现Val Loss 0.0017 (435× 改进)
- ✅ 完成数据增强消融实验
- ✅ 建立完整的文档系统
- ✅ 创建规范的代码组织

**核心成果**:
- ✅ 预设角度初始化方法
- ✅ 旋转数据增强策略
- ✅ 完整的实验报告 (2篇)
- ✅ 可行性分析文档 (1篇)

**下一步**:
- 📊 多类别合并 (已完成可行性分析)
- 📊 方法普适性验证
- 📝 论文写作准备

### 给LLM的建议

**优先级排序**:
1. ⭐⭐⭐ 创建GT统计分析脚本 (快速筛选候选类别)
2. ⭐⭐⭐ 标准化测试评估 (公平对比实验1和2)
3. ⭐⭐ 可视化对比图 (论文素材)
4. ⭐⭐ Pilot实验 (如果多类别合并)
5. ⭐ 方法论文档撰写

**工作风格**:
- 严格遵循 `claude.md` 的规范
- 每个任务完成后更新文档
- 重要发现写成markdown
- 运行训练前征求用户同意

---

**文档版本**: 1.0
**创建时间**: 2025-11-09 20:30
**创建者**: Claude
**用途**: 为LLM提供完整项目上下文
**下次更新**: 当有重大进展时

---

**📞 如有疑问，请查阅**:
- 核心配置: `claude.md`
- 项目结构: `project_structure.md`
- 实验报告: `docs/experiments/`
- 问题分析: `docs/analysis/`

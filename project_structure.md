# ForwardNet-Claude 项目结构说明

**最后更新**: 2025-11-09
**维护者**: Claude
**用途**: 详细说明项目中每个文件和文件夹的功能

---

## 📁 目录结构总览

```
/home/pablo/ForwardNet-claude/
├── claude.md                              # 核心配置文档
├── project_structure.md                   # 项目结构说明（本文件）
├── analysis_*.md                          # 实验分析文档
├── experiment_*.md                        # 实验结果报告
│
├── train_*.py                             # 训练脚本
├── dataloader_*.py                        # 数据加载器
├── test.py / eval_*.py                    # 测试/评估脚本
├── vis_*.py                               # 可视化脚本
│
├── models/                                # 神经网络模型定义
├── data/                                  # 数据集（软链接）
├── data_process/                          # 数据预处理脚本
├── visualization/                         # 可视化结果存储
├── results/                               # 训练结果和checkpoints
└── utils/                                 # 工具函数（待创建）
```

---

## 📄 根目录文件详解

### 核心配置文档

| 文件名 | 用途 | 重要性 |
|--------|------|--------|
| `claude.md` | Claude Code配置、研究目标、规范 | ⭐⭐⭐ 必读 |
| `project_structure.md` | 项目结构说明（本文件） | ⭐⭐⭐ 必读 |

### 分析与实验报告

| 文件名 | 用途 | 创建日期 |
|--------|------|---------|
| `analysis_20251109_glassbox_training_failure.md` | Glassbox训练失败根因分析 | 2025-11-09 |
| `experiment_20251109_init_fix_results.md` | 初始化修复实验完整报告 | 2025-11-09 |

**命名规范**:
- 分析文档: `analysis_YYYYMMDD_<主题>.md`
- 实验报告: `experiment_YYYYMMDD_<实验名>_results.md`

### 训练脚本

| 文件名 | 模型/方法 | 数据集 | 状态 | 备注 |
|--------|----------|-------|------|------|
| `train_pointnetpp_mvm_glassbox_augmented.py` | PointNet++ + MvM | Glassbox + 12旋转 | ✅ 推荐 | **当前最佳方案**（待重命名） |
| `train_multi_peaks_vonMises_KL.py` | PointNet++ + MvM | ModelNet40全集 | ⚠️ 有bug | zeros初始化问题 |
| `train_multi_peaks_vonMises_KL_debug.py` | PointNet++ + MvM | ModelNet40全集 | ⚠️ 有bug | debug版本 |
| `train_single_peak_vonMises_KL.py` | PointNet++ + 单峰vM | ModelNet40 | ✅ 可用 | 单峰基线 |
| `train_8dir_KL.py` | PointNet++ + 8方向 | ModelNet40 | ✅ 可用 | KL散度loss |
| `train_8dir_MSE.py` | PointNet++ + 8方向 | ModelNet40 | ✅ 可用 | MSE loss |
| `train_8dir.py` | PointNet++ + 8方向 | ModelNet40 | ✅ 可用 | 原始版本 |
| `train_multi_8dir.py` | PointNet++ + 多目标8方向 | ModelNet40 | ⚠️ 实验性 | - |
| `train.py` | PointNet++ | ModelNet40 | 🔧 旧版 | 通用训练脚本 |
| `PointNet++_train.py` | PointNet++ | - | 🔧 旧版 | Demo训练脚本 |
| `simple_pointnet_train.py` | PointNet | - | 🔧 旧版 | 简单PointNet |

**待重命名**：
- `train_glassbox_only.py` → `train_pointnetpp_mvm_glassbox_augmented.py`

### 数据加载器

| 文件名 | 用途 | 对应训练脚本 |
|--------|------|-------------|
| `dataloader_glassbox_augmented.py` | Glassbox + 12旋转增强 | `train_glassbox_only.py` |
| `dataloader_multi_peak_vonMises.py` | 多峰MvM数据加载 | `train_multi_peaks_vonMises_*.py` |
| `dataloader_single_peak_vonMises.py` | 单峰vM数据加载 | `train_single_peak_vonMises_KL.py` |
| `dataloader_8dir_sampled.py` | 8方向采样数据 | `train_8dir_*.py` |
| `dataloader.py` | 通用数据加载器 | 多个脚本 |

### 测试与演示

| 文件名 | 用途 |
|--------|------|
| `test.py` | 测试脚本 |
| `PointNet++Demo.py` | PointNet++ Demo |
| `PointNetDemo.py` | PointNet Demo |

---

## 📂 子目录详解

### `models/` - 神经网络模型定义

| 文件名 | 模型架构 | 输出 | 备注 |
|--------|---------|------|------|
| `pointnet_pp_mvM.py` | PointNet++ + MvM头 | (μ, κ, π) × K | **核心模型**，已修复初始化 |
| `pointnet_pp_vonMises.py` | PointNet++ + 单峰vM | (μ, κ) | 单峰版本 |
| `pointnet_pp_8dir.py` | PointNet++ + 8方向分类 | 8维softmax | 基线模型 |
| `pointnet_pp_Fwd.py` | PointNet++ + Forward头 | - | 早期版本 |
| `pointnet_pp.py` | PointNet++ backbone | 特征向量 | 通用backbone |
| `Pointnet_pp_xyz.py` | PointNet++ (xyz版本) | - | 变体 |
| `Pointnet_pp_xyz_Schedmit.py` | PointNet++ (Schedmit版本) | - | 变体 |
| `pointnet.py` | PointNet | 特征向量 | 经典PointNet |
| `point_transformer.py` | Point Transformer | 特征向量 | Transformer架构 |
| `base.py` | 基础模块 | - | 公共组件 |
| `__init__.py` | 模型包初始化 | - | - |

**关键修改**：
- `pointnet_pp_mvM.py:69-82`: 预设角度初始化（2025-11-09）

### `data/` - 数据集（软链接）

```
data/ -> /home/pablo/ForwardNet/data/
├── modelnet40/
│   ├── glassbox/
│   │   ├── train/
│   │   └── test/
│   ├── chair/
│   ├── table/
│   └── ...
└── processed/
    └── multi_peak_gt/
```

**说明**：
- 软链接指向主仓库的data目录
- 避免数据重复
- ModelNet40数据集：40个类别的3D点云

### `data_process/` - 数据预处理脚本

| 文件名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `2d_multi_peak_MvM_gt_1.py` | 生成多峰MvM ground truth | 点云 + 类别 | MvM参数(μ,κ,π) |
| `2d_single_peak_vM_gt.py` | 生成单峰vM ground truth | 点云 + 类别 | vM参数(μ,κ) |
| `2d_8dir_sample.py` | 8方向采样 | 点云 | 8方向标签 |
| `rotate.py` | 旋转点云（带法线） | 点云 + 角度 | 旋转后点云 |
| `rotate_without_normals.py` | 旋转点云（无法线） | 点云 + 角度 | 旋转后点云 |
| `2d_rotate_without_normals.py` | 2D旋转（无法线） | 点云 + 角度 | 旋转后点云 |
| `hdf5_process.py` | HDF5数据处理 | HDF5文件 | 处理后数据 |
| `convert_txt_to_ply.py` | 格式转换 | TXT点云 | PLY点云 |
| `DataProcess.py` | 通用数据处理 | 多种格式 | 标准格式 |

**重要**：
- GT生成脚本已验证正确（2025-11-09）
- `2d_multi_peak_MvM_gt_1.py` 为glassbox生成4峰GT

### `results/` - 训练结果存储

```
results/
├── glassbox_only_20251109_183051/      # 实验1：初始化修复（成功）
│   ├── best_model.pth                  # 最佳模型 (Val=0.0017)
│   ├── figs/
│   │   ├── predictions_epoch_010.png
│   │   ├── predictions_epoch_020.png
│   │   ├── predictions_epoch_030.png
│   │   ├── predictions_epoch_040.png
│   │   └── final_predictions.png
│   └── config.yaml (如有)
│
├── multi_peak_vonMises_KL/             # 旧实验（zeros初始化，失败）
├── multi_peak_vonMises_KL_debug/       # Debug版本（失败）
├── single_peak_vonMises_KL_1006_1/     # 单峰实验
├── 8dir_KLdiv_0926/                    # 8方向分类实验
└── [其他历史实验结果]/
```

**命名规范**：
- 格式: `<任务>_<日期>_<时间>` 或 `<任务>_<方法>_<日期>`
- 自动生成时间戳目录

### `visualization/` - 可视化结果

```
visualization/
├── glass_box/                          # Glassbox可视化
├── chair/                              # Chair可视化
├── door/                               # Door可视化
├── bottle/                             # Bottle可视化
└── visualization_MVM.py                # MvM可视化工具
```

**用途**：
- 存储各类别的预测可视化
- 极坐标图、3D点云可视化等

### `utils/` - 工具函数（待创建）

**计划创建的工具模块**：
```
utils/
├── mvm_utils.py                        # MvM分布计算
├── loss_functions.py                   # 各种loss函数
├── visualization.py                    # 可视化工具
├── metrics.py                          # 评估指标
└── data_utils.py                       # 数据处理工具
```

**目前状态**: 工具函数分散在各训练脚本中，需要重构整理

---

## 🎯 核心文件推荐阅读顺序

### 新手入门
1. `claude.md` - 了解项目目标和规范
2. `project_structure.md` - 本文件，了解项目结构
3. `models/pointnet_pp_mvM.py` - 核心模型
4. `train_pointnetpp_mvm_glassbox_augmented.py` - 最新训练脚本
5. `experiment_20251109_init_fix_results.md` - 成功实验报告

### 深入研究
1. `analysis_20251109_glassbox_training_failure.md` - 问题诊断分析
2. `data_process/2d_multi_peak_MvM_gt_1.py` - GT生成逻辑
3. `dataloader_glassbox_augmented.py` - 数据增强实现
4. `models/pointnet_pp.py` - PointNet++ backbone细节

---

## 🔧 待重构/清理项目

### 需要重命名的文件
- [ ] `train_glassbox_only.py` → `train_pointnetpp_mvm_glassbox_augmented.py`
- [ ] `train_multi_peaks_vonMises_KL.py` → `train_pointnetpp_mvm_modelnet40.py`
- [ ] `train_single_peak_vonMises_KL.py` → `train_pointnetpp_single_vm_modelnet40.py`

### 需要添加文件头注释的文件
- [ ] `models/pointnet_pp_mvM.py` - 已修改，需补充文档
- [ ] `train_glassbox_only.py` - 需要完整docstring
- [ ] `dataloader_glassbox_augmented.py` - 需要完整docstring
- [ ] 所有 `data_process/*.py` 文件

### 需要创建的文档
- [ ] `docs/methods/method_mvm_distribution.md` - MvM理论
- [ ] `docs/methods/method_hungarian_matching.md` - 匈牙利匹配
- [ ] `CHANGELOG.md` - 记录重要变更

### 需要整理的目录
- [ ] `results/` 中的旧实验结果（考虑归档或删除）
- [ ] 创建 `docs/` 目录结构
- [ ] 创建 `utils/` 并迁移公共函数

---

## 📊 项目统计

**当前状态** (2025-11-09):
- **训练脚本**: 12个
- **数据加载器**: 5个
- **模型文件**: 11个
- **数据处理脚本**: 9个
- **分析文档**: 2个
- **实验报告**: 1个

**成功实验**:
- ✅ Glassbox 4峰MvM训练（Val Loss: 0.0017）
- ✅ 单峰von Mises训练
- ✅ 8方向分类baseline

**正在进行**:
- 📝 文件命名规范化
- 📝 代码注释完善
- 📝 文档体系建立

---

## 🚀 下一步工作

### 短期 (本周)
1. 重命名核心文件遵循新规范
2. 为所有Python文件添加文件头注释
3. 创建`utils/`目录并重构公共函数
4. 将成功实验推广到其他类别（chair, table等）

### 中期 (本月)
1. 建立完整的`docs/`目录结构
2. 编写方法论文档（MvM理论、Hungarian匹配等）
3. 清理旧实验结果
4. 建立自动化测试流程

### 长期 (论文写作前)
1. 整理所有实验结果为论文章节
2. 创建完整的实验复现指南
3. 代码开源准备（如需要）

---

**维护建议**：
- 每次添加新文件后，更新本文档
- 每周审查一次文件命名规范
- 每次重要实验后，创建分析或报告文档
- 定期清理`results/`中的临时文件

**问题反馈**：
如发现文档有误或需要补充，请在`claude.md`中记录或直接告知维护者。

---

**版本**: 1.0
**创建**: 2025-11-09
**最后更新**: 2025-11-09

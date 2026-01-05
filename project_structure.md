# ForwardNet-Claude 项目结构说明

**最后更新**: 2026-01-05
**维护者**: Claude
**用途**: 详细说明项目中每个文件和文件夹的功能
**项目目标**: 3D点云正面方向检测（东京大学 M2 毕业论文实验）

---

## 目录结构总览

```
/home/pablo/ForwardNet-claude/
│
├── claude.md                    # Claude Code核心配置文档（必读）
├── project_structure.md         # 项目结构说明（本文件）
│
├── core/                        # 核心模块（backbone、heads、losses）
├── models/                      # 神经网络模型定义
├── datasets/                    # 数据集加载器
├── configs/                     # YAML配置文件
│
├── train_*.py                   # 各种训练脚本
├── test_*.py                    # 测试/评估脚本
├── evaluate_*.py                # 评估脚本
│
├── tools/                       # 可视化和标注工具
├── scripts/                     # 实验运行脚本
├── data_annotation/             # 标注数据和工具
├── data_process/                # 数据预处理脚本
│
├── checkpoints/                 # 训练保存的模型（85个实验）
├── results/                     # 训练结果和可视化（56个目录）
├── visualization/               # 可视化结果存储
├── paper_figures/               # 论文图片素材
├── screenshots/                 # 点云截图
│
├── docs/                        # 文档目录
├── paper/                       # 论文LaTeX源码
├── wandb/                       # WandB实验日志
│
├── deep_direct_stat/            # 参考代码库（von Mises工具）
├── legacy/                      # 旧版代码存档
└── data -> /home/pablo/ForwardNet/data  # 数据集软链接
```

---

## 核心代码目录

### core/ - 核心模块

模型组件的核心定义。

| 文件 | 功能 |
|------|------|
| `backbones.py` | 网络骨干架构 |
| `heads.py` | 预测头（von Mises、离散方向等） |
| `losses.py` | 损失函数定义 |
| `model.py` | 模型组装 |
| `__init__.py` | 模块初始化 |

---

### models/ - 神经网络模型

各种点云处理模型的实现。

| 文件 | 模型 | 输出 | 备注 |
|------|------|------|------|
| `probabilistic_orientation_net.py` | 概率方向网络 | 混合分布 | **主力模型** |
| `pointnet_pp_mvM.py` | PointNet++ + MvM | (mu, kappa, pi) | von Mises混合 |
| `pointnet_pp_vonMises.py` | PointNet++ + vM | (mu, kappa) | 单峰版本 |
| `pointnet_pp_8dir.py` | PointNet++ + 8方向 | softmax(8) | 离散分类 |
| `pointnet_pp.py` | PointNet++ backbone | 特征向量 | 基础骨干 |
| `pointnet_pp_Fwd.py` | PointNet++ Forward | - | 早期版本 |
| `dgcnn.py` | DGCNN | 特征向量 | 图神经网络 |
| `dgcnn_mvM.py` | DGCNN + MvM | 混合分布 | - |
| `point_transformer.py` | Point Transformer | - | Transformer架构 |
| `point_transformer_v3.py` | PTv3 | - | 升级版 |
| `ptv3_mvM.py` | PTv3 + MvM | - | - |
| `symmetry_classifier.py` | 对称性分类器 | 类别 | K值预测 |
| `pointnet.py` | PointNet | 特征向量 | 经典模型 |
| `Pointnet_pp_xyz.py` | PointNet++ XYZ | - | 变体 |
| `Pointnet_pp_xyz_Schedmit.py` | Schedmit版本 | - | 变体 |
| `base.py` | 基础模块 | - | 公共组件 |

---

### datasets/ - 数据集加载器

不同任务的数据加载实现。

| 文件 | 用途 | 对应任务 |
|------|------|----------|
| `discrete_direction_dataset.py` | 离散方向数据 | D/DR系列实验 |
| `fixed_4peak_dataset.py` | 固定4峰数据 | Fixed 4-Peak训练 |
| `moe_dataset.py` | MoE数据集 | 混合专家模型 |
| `multi_category_dataset.py` | 多类别数据 | MF系列实验 |
| `single_front_dataset.py` | 单正面数据 | 1-front训练 |
| `symmetry_classifier_dataset.py` | 对称性分类 | K值分类器 |
| `orientation.py` | 方向数据 | 通用方向任务 |

---

### configs/ - 配置文件

YAML格式的训练配置。

| 文件 | 配置内容 |
|------|----------|
| `baseline_direct.yaml` | 直接回归基线 |
| `vm_1peak.yaml` | 单峰von Mises |
| `vm_2peak.yaml` | 双峰von Mises |
| `vm_4peak.yaml` | 四峰von Mises |
| `vm_4peak_learnable.yaml` | 可学习权重版 |
| `vm_all_k.yaml` | 所有K值混合 |
| `scvae_glassbox.yaml` | SCVAE配置 |

---

## 训练与评估脚本

### 主要训练脚本

| 脚本 | 方法 | 状态 |
|------|------|------|
| `train_clean_pipeline.py` | 清洁数据pipeline | 当前使用 |
| `train_direction.py` | D/DR系列离散方向 | 已完成 |
| `train_moe.py` | MoE混合专家 | 已完成 |
| `train_fixed_4peak.py` | Fixed 4-Peak | 已完成 |
| `train_filtered_experiments.py` | 过滤数据实验 | - |
| `train_single_front.py` | 单正面训练 | - |
| `train_single_front_learnable.py` | 可学习权重 | - |
| `train_symmetry_classifier.py` | K值分类器 | - |
| `train.py` | 通用训练 | 旧版 |

### 测试/评估脚本

| 脚本 | 功能 |
|------|------|
| `test_classifier_on_testset.py` | 分类器测试集评估 |
| `test_direction_models.py` | 方向模型评估 |
| `test_d8b_simple.py` | D_8b简单测试 |
| `test_d8b_topk.py` | D_8b Top-K测试 |
| `test_mf_kappa_detection.py` | MF系列kappa检测 |
| `test_mf_p2v2.py` | MF+P2v2测试 |
| `test_mf_topk.py` | MF系列Top-K |
| `test_p2v2_correct.py` | P2v2正确性测试 |
| `test_p2v2_correct_v2.py` | P2v2 v2版本 |
| `test_p2v2_only.py` | 仅P2v2测试 |
| `evaluate_by_category.py` | 按类别评估 |
| `evaluate_comprehensive.py` | 综合评估 |
| `evaluate_hungarian_correct.py` | 匈牙利匹配评估 |
| `evaluate_kl_divergence.py` | KL散度评估 |

---

## 工具目录

### tools/ - 可视化与标注工具

详见 `tools/README.md`。

**Web工具（端口号）**:

| 工具 | 端口 | 功能 |
|------|------|------|
| `vis_checkpoint_web.py` | 8070 | 模型可视化（任意checkpoint） |
| `annotate_symmetry_web.py` | 8052 | 对称性标注工具 |
| `verify_gt_web.py` | 8060 | GT验证（点云+方向对齐） |
| `von_mises_interactive.py` | 8055 | 交互式von Mises演示 |
| `von_mises_web.py` | - | von Mises可视化 |
| `screenshot_viewer.py` | 8051 | 点云截图生成 |

**CLI工具**:

| 工具 | 功能 |
|------|------|
| `vis_fixed_4peak.py` | Fixed 4-Peak静态可视化 |
| `annotation_stats.py` | 标注进度统计 |
| `resume_p2v2_training.py` | 恢复P2v2训练 |
| `snapshot_training_state.py` | 保存训练状态快照 |

**子目录**:
- `PlotNeuralNet/` - 神经网络架构图绘制工具

---

### scripts/ - 实验运行脚本

批量运行实验的Shell脚本。

| 脚本 | 用途 |
|------|------|
| `run_d_series.sh` | D系列实验 |
| `run_dr_series.sh` | DR系列实验 |
| `run_d_series.py` | D系列Python版 |
| `run_dr_series.py` | DR系列Python版 |
| `run_mf_d_experiments.sh` | MF+D实验 |
| `run_learnable_weight_exp.sh` | 可学习权重实验 |
| `exp_loss_ablation.sh` | Loss消融实验 |
| `add_more_data.py` | 添加更多数据 |
| `run_new_experiments.sh` | 新实验 |
| `run_d_regression.sh` | 方向回归实验 |

---

### data_annotation/ - 数据标注

对称性标注数据和相关工具。

**标注数据文件**:

| 文件 | 格式 | 说明 |
|------|------|------|
| `symmetry_annotations.json` | JSON | 主标注数据（639KB） |
| `symmetry_annotations.csv` | CSV | 表格版本（237KB） |
| `symmetry_annotations.md` | Markdown | 可读版本 |
| `symmetry_annotations_filtered.json` | JSON | 过滤后数据 |
| `symmetry_annotations_indexed.json` | JSON | 索引版本 |
| `1front_outliers.json` | JSON | 1-front异常样本 |

**标注工具**:

| 文件 | 功能 |
|------|------|
| `annotate_symmetry_web.py` | Web标注界面 |
| `annotation_stats.py` | 统计进度 |
| `export_annotations.py` | 导出标注 |
| `process_annotations.py` | 处理标注 |
| `rebuild_indexed_annotations.py` | 重建索引 |
| `category_progress_viewer.py` | 类别进度查看 |
| `query_annotations.py` | 查询标注 |

---

### data_process/ - 数据预处理

点云数据预处理脚本。

| 文件 | 功能 |
|------|------|
| `2d_multi_peak_MvM_gt_1.py` | 生成多峰MvM GT |
| `2d_single_peak_vM_gt.py` | 生成单峰vM GT |
| `2d_8dir_sample.py` | 8方向采样 |
| `rotate.py` | 点云旋转（带法线） |
| `rotate_without_normals.py` | 点云旋转（无法线） |
| `2d_rotate_without_normals.py` | 2D旋转 |
| `hdf5_process.py` | HDF5数据处理 |
| `convert_txt_to_ply.py` | TXT转PLY格式 |
| `DataProcess.py` | 通用数据处理 |

---

## 输出目录

### checkpoints/ - 模型检查点

保存的训练模型（共85个实验目录）。

**实验系列命名规范**:

| 前缀 | 含义 | 示例 |
|------|------|------|
| `D_` | 离散方向预测 | `D_8a_*`, `D_16b_*` |
| `DR_` | 离散方向+回归 | `DR_8a_*`, `DR_16a_*` |
| `MF_` | Multi-Front混合 | `MF_1a_*`, `MF_1b_*` |
| `Exp*_` | 编号实验 | `Exp1_*`, `Exp2_*` |
| `CleanClassifier_` | 清洁分类器 | - |
| `2-step_MoE_` | 两阶段MoE | - |
| `fixed4peak_` | Fixed 4-Peak | - |

**目录结构示例**:
```
checkpoints/D_8a_20251218_170150/
├── best.pth           # 最佳模型
├── final.pth          # 最终模型
├── config.json        # 训练配置
└── training_log.txt   # 训练日志
```

---

### results/ - 训练结果

共56个结果目录，包含可视化和评估结果。

**主要目录**:

| 目录 | 内容 |
|------|------|
| `D_series/` | D系列实验结果 |
| `DR_series/` | DR系列实验结果 |
| `MF_series/` | MF系列实验结果 |
| `CleanClassifier_*` | 分类器结果 |
| `figs/` | 通用图片 |
| `direction_experiments_summary.md` | 方向实验总结 |

---

### visualization/ - 可视化结果

按类别存储的可视化图片。

| 子目录 | 内容 |
|--------|------|
| `glass_box/` | Glass box类别 |
| `chair/` | Chair类别 |
| `door/` | Door类别 |
| `bottle/` | Bottle类别 |
| `outputs/` | 工具输出 |

**可视化脚本**:
- `visualization_MVM.py` - MvM可视化
- `vis_scvae_peaks.py` - SCVAE峰值可视化
- `vis_training_progression.py` - 训练进度可视化

---

### paper_figures/ - 论文图片

论文使用的图片和架构图。

**架构图（.drawio）**:
- `three_methods_overview.drawio` - 三种方法概览
- `mf_series_architecture.drawio` - MF系列架构
- `mf_series_detailed.drawio` - MF详细版
- `mf_series_simple.drawio` - MF简化版
- `p2v2_clean_architecture.drawio` - P2v2架构
- `expert_head_architecture.drawio` - 专家头架构
- `moe.drawio` - MoE架构
- `architecture_diagram.drawio` - 通用架构图

**训练曲线**:
- `p2v2_clean_training.png/pdf` - P2v2训练曲线
- `d8b_train_val_loss.png` - D8b训练曲线

**子目录**:
- `discrete_vis/` - 离散可视化图片

---

### screenshots/ - 点云截图

各类点云的截图，用于论文和演示。

---

## 文档目录

### docs/ - 文档

项目相关的分析、实验报告和方法论文档。

**子目录**:

| 目录 | 内容 |
|------|------|
| `analysis/` | 问题分析文档 |
| `experiments/` | 旧实验报告 |
| `methods/` | 方法论文档 |

**主要文档**:

| 文件 | 内容 |
|------|------|
| `D_DR_series_experiments.md` | D/DR系列实验报告 |
| `MF_series_experiments.md` | MF系列实验报告 |
| `method3_moe_experiment_report.md` | MoE实验报告 |
| `experiment_report_clean_pipeline.md` | 清洁pipeline报告 |
| `report_01_clean_classifier.md` | 分类器报告 |
| `report_02_p2v2_clean.md` | P2v2清洁报告 |
| `report_03_muonly_baseline.md` | Mu-only基线报告 |
| `fixed_4peak_experiments_20251210.md` | Fixed 4-Peak实验 |
| `issue_1front_kappa_collapse.md` | 1-front问题分析 |
| `experiments_summary_for_LLM.md` | 实验总结 |
| `exp_loss_ablation_plan.md` | Loss消融计划 |
| `learnable_weight_experiment_plan.md` | 可学习权重计划 |
| `Fixed_4峰MvM训练完整指南.md` | 中文训练指南 |
| `离散方向向量预测实现文档.md` | 离散方向实现 |

---

### paper/ - 论文源码

LaTeX论文源文件。

| 文件 | 内容 |
|------|------|
| `background.tex` | 背景章节 |
| `method.tex` | 方法章节 |
| `results.tex` | 结果章节 |
| `figures/` | 论文图片 |

---

## 参考代码

### deep_direct_stat/ - 参考库

从GitHub克隆的deep_direct_stat库，提供von Mises工具函数。

| 子目录 | 内容 |
|--------|------|
| `datasets/` | 数据集 |
| `models/` | 模型 |
| `utils/` | 工具函数 |
| `scripts/` | 脚本 |
| `view_estimation/` | 视角估计 |
| `notebooks/` | Jupyter笔记本 |
| `training_scripts/` | 训练脚本 |

---

### legacy/ - 旧版代码

已归档的旧版训练脚本和数据加载器。

| 文件 | 内容 |
|------|------|
| `dataloader_*.py` | 旧数据加载器 |
| `train_*.py` | 旧训练脚本 |
| `README.md` | 说明文档 |

---

## 数据目录

### data/ -> /home/pablo/ForwardNet/data

软链接指向主数据目录。

**数据集结构**:
```
data/
├── full_mn40_normal_resampled_ply/   # ModelNet40点云
├── symmetry_classification_gt/        # 对称性分类GT
│   ├── 1_front/                       # 1个正面（200个）
│   ├── 2_fronts/                      # 2个正面（200个）
│   ├── 4_fronts/                      # 4个正面（200个）
│   ├── symmetric/                     # 旋转对称（200个）
│   ├── no_front/                      # 无正面（200个）
│   └── dataset_info.json              # 元信息
└── [其他数据集]/
```

---

## 根目录其他文件

### 日志文件

| 文件 | 内容 |
|------|------|
| `experiment_log.txt` | 通用实验日志 |
| `experiment_log_d_series.txt` | D系列日志 |
| `experiment_log_dr_series.txt` | DR系列日志 |
| `experiment_log_moe.txt` | MoE日志 |
| `experiment_log_moe_v2.txt` | MoE v2日志 |
| `experiment_log_p2v2.txt` | P2v2日志 |
| `experiments.log` | 实验记录 |
| `training_*.log` | 各训练日志 |

### Shell脚本

| 文件 | 用途 |
|------|------|
| `run_clean_pipeline.sh` | 运行清洁pipeline |
| `run_filtered_experiments.sh` | 运行过滤实验 |
| `resume_training.sh` | 恢复训练 |

### 其他

| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python依赖 |
| `codex.md` | Codex配置 |
| `.gitignore` | Git忽略规则 |
| `kl.ipynb` | KL散度笔记本 |
| `*.png` | 调试图片 |

---

## 项目统计

| 类型 | 数量 |
|------|------|
| 训练脚本 | 10+ |
| 测试脚本 | 15+ |
| 模型文件 | 18 |
| 数据加载器 | 7 |
| Web工具 | 6 |
| CLI工具 | 4 |
| 文档 | 20+ |
| 实验检查点 | 85 |
| 结果目录 | 56 |

---

## 核心工作流

### 1. 标注流程
```
annotate_symmetry_web.py -> symmetry_annotations.json
```

### 2. 数据生成
```
symmetry_annotations.json -> data_process/*.py -> GT数据
```

### 3. 训练流程
```
train_*.py + datasets/*.py + models/*.py -> checkpoints/
```

### 4. 评估流程
```
test_*.py / evaluate_*.py -> results/
```

### 5. 可视化
```
tools/vis_checkpoint_web.py -> 浏览器预览
```

---

**版本**: 2.0
**创建**: 2025-11-09
**更新**: 2026-01-05

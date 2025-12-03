# Phase 2: 对称性分类网络实验

## 📋 实验目标

训练一个独立的对称性分类网络，用于判断点云模型的对称性类型（K值）：
- **输入**: 10000点点云 + 竖直方向向量
- **输出**: 5分类（K=-1/0/1/2/4）
  - K=-1: 没有正面（如植物）
  - K=0: 完全对称（如球）
  - K=1: 1个正面（如飞机）
  - K=2: 2个正面（如椅子）
  - K=4: 4个正面（如桌子）

## 🗂️ 项目文件结构

```
ForwardNet-claude/
├── dataloader_symmetry.py              # 对称性数据集（支持旋转增强）
├── models/
│   └── symmetry_classifier.py          # 3种网络架构（消融实验）
├── train_symmetry_classifier.py        # 训练脚本
├── data_annotation/
│   ├── symmetry_annotations.json       # 标注数据
│   └── annotate_symmetry_web_v2.py     # Web标注工具
└── data/
    └── full_mn40_normal_resampled_ply/ # 对齐的ModelNet40数据集
```

## 🧪 消融实验设计

### 实验1: Upright Vector的输入方式

| 实验 | 架构 | 描述 |
|------|------|------|
| Exp 1.1 | `global_concat` | PointNet++ encoder → concat upright_vec到全局特征 |
| Exp 1.2 | `point_concat` | 将upright_vec复制到每个点，作为6维输入 |
| Exp 1.3 | `no_upright` | Baseline，不使用upright_vec |

### 实验2: Backbone对比（后续）

- PointNet++ (baseline)
- DGCNN
- Point Transformer V3 (可选)

## 🚀 快速开始

### 1. 数据准备

当前标注进度：**110/12,311 (约1%)**

```bash
# 查看当前标注进度
python data_annotation/annotation_stats.py

# 继续标注（Web工具）
python data_annotation/annotate_symmetry_web_v2.py --port 8051
# 然后访问: http://localhost:8051
```

### 2. 测试数据加载

```bash
# 测试数据集是否正常加载
python dataloader_symmetry.py
```

### 3. 训练模型

#### 基础训练（Exp 1.1: GlobalConcat）

```bash
python train_symmetry_classifier.py \
    --model global_concat \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_class_weights \
    --save_dir results/symmetry_exp1_global_concat
```

#### Exp 1.2: PointConcat

```bash
python train_symmetry_classifier.py \
    --model point_concat \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_class_weights \
    --save_dir results/symmetry_exp1_point_concat
```

#### Exp 1.3: NoUpright (Baseline)

```bash
python train_symmetry_classifier.py \
    --model no_upright \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --use_class_weights \
    --save_dir results/symmetry_exp1_no_upright
```

### 4. 训练参数说明

```bash
# 数据相关
--annotation_file       # 标注文件路径
--data_dir              # 数据集目录
--num_points 10000      # 采样点数
--batch_size 16         # Batch大小
--num_workers 4         # 数据加载线程数

# 模型相关
--model global_concat   # 模型选择: global_concat / point_concat / no_upright

# 训练相关
--epochs 200            # 训练轮数
--lr 1e-3               # 学习率
--weight_decay 1e-4     # 权重衰减
--use_class_weights     # 使用类别权重（推荐，处理不平衡数据）

# 学习率调度
--scheduler cosine      # cosine / step / none
--step_size 50          # StepLR的step_size
--gamma 0.5             # StepLR的gamma

# 保存相关
--save_dir              # 结果保存目录
--save_freq 50          # 保存checkpoint的频率（epoch）
--resume                # 恢复训练的checkpoint路径
```

## 📊 数据增强策略

**训练时**：
1. 读取对齐的点云（正面通常为-Z方向）
2. **随机旋转**：围绕upright轴（Y轴）旋转 θ ∈ [0, 2π)
3. 同时旋转front_direction向量（用于验证，不作为输入）
4. 归一化到单位球
5. 随机采样10000个点

**验证/测试时**：
- 不使用旋转增强
- 其余步骤相同

## 📈 训练结果

训练完成后，结果保存在指定的`--save_dir`：

```
results/symmetry_exp1_global_concat/
├── config.json              # 实验配置
├── training.log             # 训练日志（JSON格式）
├── best_model.pth           # 最佳模型（验证集准确率最高）
├── checkpoint_epoch_50.pth  # 定期保存的checkpoint
├── checkpoint_epoch_100.pth
└── test_results.json        # 测试集结果
```

### 查看训练日志

```python
import json

# 加载训练日志
with open('results/symmetry_exp1_global_concat/training.log', 'r') as f:
    log = json.load(f)

# 绘制训练曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(log['train_loss'], label='Train Loss')
plt.plot(log['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(log['train_acc'], label='Train Acc')
plt.plot(log['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

## ⚠️ 当前数据限制

**重要提示**：当前标注数据量很少（110个样本），且类别**极度不平衡**：

| K值 | 类别 | 训练集 | 验证集 | 测试集 | 总计 |
|-----|------|--------|--------|--------|------|
| K=-1 | 没有正面 | 0 | 0 | 0 | 0 |
| K=0 | 完全对称 | 1 | 0 | 1 | 2 |
| K=1 | 1个正面 | 72 | 15 | 17 | 104 |
| K=2 | 2个正面 | 0 | 0 | 1 | 1 |
| K=4 | 4个正面 | 2 | 0 | 1 | 3 |

**建议**：
1. **优先标注**: 使用Web标注工具，优先完成某些category（如chair, table, plant）
2. **类别平衡**: 使用 `--use_class_weights` 参数
3. **数据增强**: 默认开启旋转增强
4. **小batch**: 当前数据少，使用小batch_size（如8或16）

## 🔧 后续实验规划

等待更多标注数据后：

### 实验3: 数据增强消融

```bash
# 3.1: 仅旋转增强（默认）
# 3.2: 旋转 + jittering
# 3.3: 旋转 + random scaling
# 3.4: 旋转 + dropout points
```

### 实验4: Backbone对比

```bash
# 4.1: PointNet++ (已实现)
# 4.2: DGCNN
# 4.3: Point Transformer V3
```

## 📝 实验记录表格（用于论文）

| 实验ID | 模型 | Upright输入方式 | Train Acc | Val Acc | Test Acc | Per-class Acc | 备注 |
|--------|------|----------------|-----------|---------|----------|---------------|------|
| Exp 1.1 | PointNet++ | Global Concat | - | - | - | - | 待运行 |
| Exp 1.2 | PointNet++ | Point Concat | - | - | - | - | 待运行 |
| Exp 1.3 | PointNet++ | No Upright | - | - | - | - | Baseline |

## 🎯 下一步

1. **继续标注数据**: 使用Web工具标注更多category
   ```bash
   python data_annotation/annotate_symmetry_web_v2.py --port 8051
   ```

2. **当有足够数据后** (建议每个K值至少100个样本)，运行完整训练：
   ```bash
   # 运行3个消融实验
   bash run_ablation_exp1.sh  # (需要创建)
   ```

3. **对比实验结果**，选择最佳架构

4. **进入Phase 3**: 整合对称性分类网络和MvM网络
   - 对称性网络 → K值预测
   - 根据K值选择对应的MvM网络
   - 端到端训练或两阶段训练

## 💡 Tips

- **GPU内存不足**: 减小batch_size或num_points
- **训练太慢**: 减小num_points（如从10000降到8192或4096）
- **过拟合**: 增强数据增强，增大dropout
- **欠拟合**: 增大模型容量，降低dropout

---

**创建日期**: 2025-11-25
**作者**: Claude
**实验状态**: ✅ 代码准备完成，等待数据标注

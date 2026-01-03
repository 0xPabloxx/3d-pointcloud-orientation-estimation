# 实验报告 1: Clean Classifier

> **Checkpoint**: `checkpoints/CleanClassifier_20251229_220630`
> **训练日期**: 2025-12-29
> **WandB Project**: ForwardNet-LossAblation

---

## 1. 实验目标

训练一个对称性类型分类器，用于后续P2v2模型的Gate机制。分类器需要预测物体属于5种对称类型之一。

---

## 2. 模型架构

### 2.1 网络结构

```
SymmetryClassifier
│
├── PointNetPlusPlusEncoder
│   │
│   ├── SetAbstraction Layer 1
│   │   ├── npoint: 512 (采样点数)
│   │   ├── radius: 0.2 (球查询半径)
│   │   ├── nsample: 32 (每个球内采样数)
│   │   └── MLP: [3] → [64, 64, 128]
│   │
│   ├── SetAbstraction Layer 2
│   │   ├── npoint: 128
│   │   ├── radius: 0.4
│   │   ├── nsample: 64
│   │   └── MLP: [128+3] → [128, 128, 256]
│   │
│   ├── SetAbstraction Layer 3
│   │   ├── npoint: 32
│   │   ├── radius: 0.8
│   │   ├── nsample: 128
│   │   └── MLP: [256+3] → [256, 512, 1024]
│   │
│   └── FC Block
│       ├── Linear(1024 → 1024)
│       ├── BatchNorm1d(1024)
│       ├── ReLU
│       └── Dropout(0.4)
│
└── Classifier Head
    ├── Linear(1024 → 256) + BatchNorm1d + ReLU + Dropout(0.3)
    ├── Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
    └── Linear(128 → 5)
```

### 2.2 参数统计

| 组件 | 参数量 |
|------|--------|
| PointNetPlusPlusEncoder | ~1.9M |
| Classifier Head | ~0.3M |
| **总计** | **2,156,101** |

### 2.3 代码位置

- **模型定义**: `train_symmetry_classifier.py` (lines 131-182)
- **SetAbstraction**: `train_symmetry_classifier.py` (lines 88-128)
- **PointNetPlusPlusEncoder**: `train_symmetry_classifier.py` (lines 131-157)

---

## 3. 数据集

### 3.1 数据来源

- **标注文件**: `data_annotation/symmetry_annotations.json`
- **点云目录**: `data/full_mn40_normal_resampled_ply`
- **异常值清单**: `data_annotation/1front_outliers.json`

### 3.2 数据过滤

#### 3.2.1 类别过滤 (仅1-front)

| 原始类别 | 样本数 | 是否保留 | 原因 |
|---------|--------|----------|------|
| chair | 300 | ✓ | 方向标注明确 |
| airplane | 66 | ✓ | 方向标注明确 |
| bookshelf | 56 | ✗ | 前后难以区分 |
| bench | 25 | ✗ | 方向模糊 |
| wardrobe | 10 | ✗ | 全部为异常值 |
| bathtub | 8 | ✗ | 方向模糊 |

**1-front类别过滤**: 排除331个样本 (bookshelf, bench, wardrobe, bathtub)

#### 3.2.2 异常值过滤

- **阈值**: severe (误差 ≥ 90°)
- **总异常值**: 77个 (来自所有类别)
- **实际排除**: 13个 (在airplane和chair中)
  - airplane: 4个 (`airplane_0022`, `airplane_0104`, `airplane_0193`, `airplane_0123`)
  - chair: 9个 (`chair_0264`, `chair_0709`, `chair_0763`, `chair_0753`, `chair_0974`, `chair_0786`, `chair_0431`, `chair_0198`, `chair_0308`)

### 3.3 过滤后数据统计

| 对称类型 | 标签 | 样本数 |
|---------|------|--------|
| 1个正面 (1-front) | 0 | 353 |
| 2个正面 (2-front) | 1 | 400 |
| 4个正面 (4-front) | 2 | 273 |
| 旋转对称 (Rot-sym) | 3 | 806 |
| 无正面 (No-front) | 4 | 340 |
| **总计** | - | **2,172** |

*注: 1-front = 366 (airplane+chair原始) - 13 (异常值) = 353*

### 3.4 数据划分

| 划分 | 比例 | 基础样本数 | ×旋转增强 | 总样本数 |
|------|------|-----------|----------|---------|
| Train | 70% | ~1,520 | ×12 | ~18,240 |
| Val | 15% | ~326 | ×4 | ~1,304 |
| Test | 15% | ~326 | ×4 | ~1,304 |

### 3.5 数据增强

```python
# 绕Y轴随机旋转
if self.augment:  # 训练时
    rotation_angle = np.random.uniform(0, 2 * np.pi)
else:  # 验证/测试时
    rotation_angle = rotation_idx * (2 * np.pi / num_rotations)

cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
rotation_matrix = np.array([
    [cos_r, 0, sin_r],
    [0, 1, 0],
    [-sin_r, 0, cos_r]
])
points = points @ rotation_matrix.T
```

---

## 4. 训练配置

### 4.1 完整配置参数

```json
{
  "annotation_file": "data_annotation/symmetry_annotations.json",
  "data_dir": "data/full_mn40_normal_resampled_ply",
  "outlier_json": "data_annotation/1front_outliers.json",
  "outlier_threshold": "severe",
  "allowed_1front_categories": ["airplane", "chair"],

  "batch_size": 32,
  "num_points": 2048,
  "num_workers": 4,
  "seed": 42,

  "classifier_epochs": 50,
  "classifier_lr": 0.001,
  "classifier_num_rotations": 12,

  "wandb": true,
  "wandb_project": "ForwardNet-LossAblation"
}
```

### 4.2 优化器配置

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| T_max | 50 (epochs) |
| eta_min | 1e-6 |

### 4.3 损失函数

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

### 4.4 类别平衡

使用 `WeightedRandomSampler` 解决类别不平衡:

```python
# 计算逆频率权重
class_weights = {label: 1.0 / count for label, count in label_counts.items()}
sample_weights = [class_weights[s['label']] for s in samples]

# 扩展到所有旋转增强
sample_weights = sample_weights.repeat_interleave(num_rotations)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

---

## 5. 训练结果

### 5.1 最终结果 (从checkpoint读取)

| 指标 | 值 |
|------|-----|
| **Best Epoch** | **25** (0-indexed: epoch 24) |
| **Best Val Accuracy** | **99.1131%** |

### 5.2 Checkpoint内容

```python
# torch.load('checkpoints/CleanClassifier_20251229_220630/best.pth')
{
    'epoch': 24,
    'model_state_dict': {...},
    'val_acc': 99.1130820399113
}
```

### 5.3 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pth` | 8.3 MB | 最佳模型权重 |
| `config.json` | 1.1 KB | 训练配置 |

---

## 6. 代码实现

### 6.1 模型前向传播

```python
class SymmetryClassifier(nn.Module):
    def __init__(self, encoder_dim: int = 1024, num_classes: int = 5):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=encoder_dim)

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, points):
        features = self.encoder(points)  # [B, 1024]
        logits = self.classifier(features)  # [B, 5]
        return logits
```

### 6.2 训练循环

```python
for epoch in range(args.classifier_epochs):
    # Training
    model.train()
    for batch in train_loader:
        points = batch['points'].to(device)
        labels = batch['gt_label'].to(device)

        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['points'].to(device))
            preds = logits.argmax(dim=1)
            # ... calculate accuracy

    scheduler.step()

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc
        }, output_dir / 'best.pth')
```

---

## 7. 使用说明

### 7.1 加载模型

```python
from train_symmetry_classifier import SymmetryClassifier

model = SymmetryClassifier(encoder_dim=1024, num_classes=5)
checkpoint = torch.load('checkpoints/CleanClassifier_20251229_220630/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 7.2 推理

```python
# 输入: points [B, N, 3]
with torch.no_grad():
    logits = model(points)
    probs = F.softmax(logits, dim=1)
    predicted_class = logits.argmax(dim=1)
```

---

## 8. 后续使用

此分类器作为P2v2模型的Gate机制:

1. **冻结权重**: 在P2v2训练中不更新分类器参数
2. **Soft Routing**: 使用softmax输出作为样本权重
3. **Checkpoint路径**: `checkpoints/CleanClassifier_20251229_220630/best.pth`

---

*报告生成时间: 2025-12-31*

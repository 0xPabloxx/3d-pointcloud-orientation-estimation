# Claude Code 配置 - 3D点云正面方向检测实验
# 东京大学 M2 毕业论文实验项目

---

## 📋 TL;DR

### 🚨 强制规则
1. **工作目录**：始终在 `/home/pablo/ForwardNet-claude/`，分支 `claude`
2. **文档位置**：只有 `claude.md` 和 `project_structure.md` 放根目录，其他 markdown 放 `docs/`
3. **数据集划分**：seed=42，train/val/test 严格分离，测试集只评估一次
4. **Ground Truth 不要动**：训练时当场处理 GT，不要预处理或修改原始标注

### 🎯 当前核心任务

**任务1：Fixed N von Mises 网络**
- 输出固定 N 个 von Mises 分量的 (μ, κ)
- 处理混合数据：1峰、2峰、4峰、无正面(κ=0)、旋转对称(κ=0)
- GT 格式：μ (方向向量) 和 κ (集中度)
- **数据处理先别实现，等用户指示**

**任务2：离散方向采样网络**
- 输出 8 或 16 个离散方向的 softmax 概率
- 1峰/2峰/4峰的 GT 分别处理成离散方向向量
- **GT 不要动，训练时当场处理**

### 🗣️ 语言
如果用户没有特别要求其他语言，请用中文回答。

---

## 🚨 强制工作规范

### 规则1: 工作目录约束

```bash
工作目录: /home/pablo/ForwardNet-claude/
分支: claude
```

### 规则2: 文档存储位置

```
/home/pablo/ForwardNet-claude/
├── claude.md              # 仅此文件和project_structure.md放根目录
├── project_structure.md
└── docs/                  # 所有其他markdown文档放这里
```

### 规则3: 数据集划分（强制）

```python
# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 划分比例：7:2:1
# Train: 训练，可用数据增强
# Val:   超参数调优，不增强
# Test:  最终评估一次，不增强
```

---

## 🎯 任务详细说明

### 任务1：Fixed N von Mises 网络

**目标**：训练一个输出固定 N 个 von Mises 分量的网络

**输出格式**：
- μ: (B, N, 2) — 每个分量的方向 (cos θ, sin θ)，需归一化
- κ: (B, N) — 每个分量的集中度参数

**处理的数据类型**：
| 类型 | K值 | μ | κ |
|------|-----|---|---|
| 1个正面 | 1 | 单一方向 | κ > 0 |
| 2个正面 | 2 | 2个方向(间隔180°) | κ > 0 |
| 4个正面 | 4 | 4个方向(间隔90°) | κ > 0 |
| 没有正面 | 0 | 任意 | κ = 0 |
| 旋转对称 | -1 | 任意 | κ = 0 |

**GT 处理方式**（训练时当场处理）：
- 1峰：μ = 标注方向，κ = 预设值（如10）
- 2峰：μ = [θ, θ+180°]，κ = 预设值
- 4峰：μ = [θ, θ+90°, θ+180°, θ+270°]，κ = 预设值
- 无正面/旋转对称：κ = 0（μ 任意，loss 忽略 μ）

**关键点**：
- N 是固定的超参数（如 N=4）
- 对于 K < N 的情况，用 κ=0 表示"无效分量"

---

### 任务1数据集：Fixed 4-Peak von Mises GT（已完成）

**数据集位置**：`data/symmetry_classification_gt/`

**目录结构**：
```
data/symmetry_classification_gt/
├── 1_front/           # 200个文件（1个正面）
│   ├── xxx.ply        # 旋转后的点云
│   └── xxx_gt.txt     # 对应的GT
├── 2_fronts/          # 200个文件（2个正面）
├── 4_fronts/          # 200个文件（4个正面）
├── symmetric/         # 200个文件（旋转对称）
├── no_front/          # 200个文件（无正面）
├── dataset_info.json  # 元信息（原始方向、旋转角度等）
└── verify_gt_web.py   # 验证工具
```

**GT文件格式** (`xxx_gt.txt`)：
```
# Mixture of 4 von Mises distributions
# Format: weight, mu_cos, mu_sin, kappa (one peak per line)
0.2500 0.123456 0.992345 50.0
0.2500 -0.992345 0.123456 0.0
0.2500 -0.123456 -0.992345 0.0
0.2500 0.992345 -0.123456 0.0
```
- 每行：`weight mu_cos mu_sin kappa`
- weight固定为0.25（4峰等权重混合）
- mu用(cos θ, sin θ)表示，已归一化
- kappa=50表示有峰，kappa=0表示无峰

**各类别GT特征**：
| 类别 | 有效峰数 | kappa分布 |
|------|---------|-----------|
| 1_front | 1 | [50, 0, 0, 0] |
| 2_fronts | 2 | [50, 50, 0, 0] |
| 4_fronts | 4 | [50, 50, 50, 50] |
| symmetric | 0 | [0, 0, 0, 0] |
| no_front | 0 | [0, 0, 0, 0] |

**数据处理流程**：
1. 从`data_annotation/symmetry_annotations.json`读取标注
2. 排除斜向(OBLIQUE)和多正面(MULTI)数据
3. 按类别筛选，每类选200个
4. 对每个物体：
   - 将front_direction转为3D向量（如-Z→[0,0,-1]）
   - 生成随机Y轴旋转角度(0~360°)
   - 用相同旋转矩阵旋转点云和front向量
   - 从旋转后的front向量计算新的mu角度：`atan2(z, x)`
   - 生成4峰GT（根据类别决定哪些峰kappa=50）

**坐标系定义**（XZ平面，Y轴向上）：
```
       +Z (90°)
         ↑
         |
-X (180°)←───→+X (0°)
         |
         ↓
       -Z (270°)
```

**验证工具**：
```bash
python data/symmetry_classification_gt/verify_gt_web.py --port 8060
# 访问 http://localhost:8060
# 显示点云俯视图 + GT方向箭头，验证对齐
```

---

### 任务2：离散方向采样网络

**目标**：训练一个输出离散方向概率的网络

**输出格式**：
- probs: (B, D) — D 个离散方向的 softmax 概率
- D = 8 或 16（对应 45° 或 22.5° 间隔）

**离散方向定义**：
```python
# D=8 的情况（45°间隔）
directions = [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]

# D=16 的情况（22.5°间隔）
directions = [0°, 22.5°, 45°, ..., 337.5°]
```

**GT 处理方式**（训练时当场处理）：
- 1峰：one-hot 向量，最近的离散方向为1
- 2峰：2 个位置各为 0.5（或 soft label）
- 4峰：4 个位置各为 0.25（或 soft label）
- 无正面/旋转对称：均匀分布 1/D

**关键点**：
- **原始 GT 标注不要修改**
- 在 dataloader 或 loss 计算时当场将 GT 转换为离散向量
- 可以考虑 soft label（高斯模糊）让相邻方向也有小概率

---

## 📁 关键文件

**数据标注**：
- `data_annotation/symmetry_annotations.json` — 对称性标注（K值、前向方向）

**数据集**：
- `data/full_mn40_normal_resampled_ply/` — ModelNet40 点云

**现有代码**：
- `train.py` — 主训练脚本
- `core/heads.py` — 预测头定义
- `core/losses.py` — Loss 函数
- `datasets/orientation.py` — 数据加载

---

## 🛠️ 工具列表

所有工具已整理到 `/tools` 目录下，详见 `tools/README.md`

### Web 工具

| 工具 | 端口 | 用途 |
|------|------|------|
| `vis_checkpoint_web.py` | 8070 | **模型可视化** - 任意 .pth 的预测可视化 |
| `annotate_symmetry_web.py` | 8052 | **标注工具** - 对称性类别和前向方向 |
| `verify_gt_web.py` | 8060 | **GT验证** - 点云与GT方向对齐检查 |
| `von_mises_interactive.py` | 8055 | **演示** - 交互式 von Mises 分布 |
| `screenshot_viewer.py` | 8051 | **截图** - 论文图片生成 |

### CLI 工具

| 工具 | 用途 |
|------|------|
| `vis_fixed_4peak.py` | Fixed 4-Peak 模型静态可视化 |
| `annotation_stats.py` | 标注进度统计 |

### 常用命令

```bash
# 模型可视化 (推荐)
python tools/vis_checkpoint_web.py --port 8070

# 对称性标注
python tools/annotate_symmetry_web.py --port 8052

# GT验证
python tools/verify_gt_web.py --port 8060

# 标注统计
python tools/annotation_stats.py

# CLI可视化
python tools/vis_fixed_4peak.py --checkpoint checkpoints/xxx/best.pth --save
```

---

## ⚠️ 训练前必须

1. **告知用户**：训练内容、预计时间、资源占用
2. **等待确认后再开始**
3. **有可视化输出**：loss曲线、定期checkpoint
4. **发现异常立即停止报告**

---

## 📐 输出方向 μ 的表示规范（强制）

**方向 μ 必须用 (cos θ, sin θ) 表示并归一化**

```python
# ❌ 错误：直接输出角度
mu = self.mu_head(features)  # θ ∈ [0, 2π)

# ✅ 正确：输出 (cos θ, sin θ) 并归一化
mu_raw = self.mu_head(features)  # (B, N, 2)
mu = F.normalize(mu_raw, dim=-1)  # 归一化到单位圆
```

---

## 📝 命名规范

**文档**：
- 分析：`analysis_YYYYMMDD_<topic>.md`
- 实验：`experiment_YYYYMMDD_<name>_results.md`

**Python文件**：
- 格式：`<功能>_<模型>_<数据>.py`
- 示例：`train_pointnetpp_mvm.py`、`train_discrete_direction.py`

**文件头必须有 docstring**：说明模型/数据/loss/用法

---

## 🧮 Von Mises 工具函数

### 角度表示转换

```python
def deg2bit(angles_deg: torch.Tensor) -> torch.Tensor:
    """度数 → (cos, sin)"""
    angles_rad = torch.deg2rad(angles_deg)
    return torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=-1)

def bit2deg(angles_bit: torch.Tensor) -> torch.Tensor:
    """(cos, sin) → 度数 [0, 360)"""
    return (torch.rad2deg(torch.atan2(angles_bit[..., 1], angles_bit[..., 0])) + 360) % 360
```

### Kappa 参考值

| kappa | 等效标准差 | 描述 |
|-------|-----------|------|
| 0 | ∞ | 均匀分布（无正面） |
| 1 | ~40° | 很分散 |
| 10 | ~13° | 中等集中 |
| 50 | ~6° | 很集中 |

---

## 🧪 Fixed 4-Peak 实验记录 (2024-12-10)

### 训练脚本
- **文件**: `train_fixed_4peak.py`
- **模型**: PointNet++ (SSG) encoder + 固定 4 峰 von Mises 输出头
- **数据**: 798 base samples × 10 augmentation = 7980 train, 228 val

### Loss 配置说明
通过命令行参数控制 loss 权重：
```bash
python train_fixed_4peak.py \
    --lambda_kl 1.0      # KL Divergence 权重 (设为 0 禁用)
    --lambda_kappa 5.0   # κ 监督权重
    --lambda_mu 2.0      # μ 监督权重
    --epochs 100
```

### 实验列表

| 实验 ID | Loss 配置 | WandB Run | Val KL | 4_fronts KL | 状态 |
|---------|-----------|-----------|--------|-------------|------|
| Exp-1 | Pure KL (λ_KL=1, λ_κ=0, λ_μ=0) | `fixed4peak_pn++_paired-init_kl_1140samples` | **0.32** | **0.98** | ✅ 完成 |
| Exp-2 | KL + κ (λ_KL=1, λ_κ=5, λ_μ=0) | `fixed4peak_pn++_kl+kappa_20251210_164456` | 0.45 | 1.52 | ✅ 完成 |
| Exp-3 | KL + κ + μ (λ_KL=1, λ_κ=5, λ_μ=2) | `fixed4peak_pn++_kl+kappa+mu_20251210_170618` | 0.40 | 1.24 | ✅ 完成 |
| Exp-4 | **μ + κ only** (λ_KL=0, λ_κ=5, λ_μ=2) | TBD | - | - | 📋 待运行 |

### 关键发现

#### 1. Pure KL 导致 κ 坍塌
- 模型输出 κ ≈ 0 (uniform 分布)
- Val KL 从 Epoch 2 开始不再下降 (0.3188 恒定)
- **原因**: 输出 uniform 是"安全"的局部最小值

#### 2. 添加 κ 监督后更差
- 强制 κ=50，但 μ 方向仍然错误
- 尖峰在错误位置 → KL 更大
- 4_fronts KL: 0.98 → 1.52

#### 3. 添加 μ 监督有改善但不够
- 4_fronts KL: 1.52 → 1.24 (比 κ-only 好)
- 但仍不如 Pure KL (0.98)
- **原因**: KL Loss 和参数监督目标冲突

### 下一步计划

**Exp-4: μ + κ only (移除 KL)**
```bash
python train_fixed_4peak.py --epochs 100 --lambda_kl 0 --lambda_kappa 5 --lambda_mu 2
```

**假设**: 移除 KL Loss 后：
1. 消除 κ→0 的倾向
2. 直接监督峰参数
3. 简化优化目标

### Checkpoints 目录结构
```
checkpoints/
├── fixed4peak_pn++_paired-init_kl_1140samples/   # Exp-1
├── fixed4peak_pn++_kl+kappa_20251210_164456/     # Exp-2
├── fixed4peak_pn++_kl+kappa+mu_20251210_170618/  # Exp-3
└── fixed4peak_pn++_kappa5.0_mu2.0_YYYYMMDD_*/    # Exp-4 (待运行)
```

---

**版本**: 6.0
**更新**: 2025-12-10
**核心任务**: Fixed 4-Peak von Mises 训练实验进行中
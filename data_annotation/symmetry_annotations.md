# 点云对称性与方向标注报告

**生成时间**: 2025-11-27 20:02:29

**数据集**: /home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply

---

## 📊 标注统计

- **总标注数**: 381
- **需矫正数**: 44

### 对称性分布

| 对称类型 | 数量 | 占比 |
|---------|------|------|
| 1个正面 | 107 | 28.1% |
| 2个正面 | 1 | 0.3% |
| 4个正面 | 271 | 71.1% |
| 完全对称 | 2 | 0.5% |

### 正面方向分布

| 方向 | 数量 | 占比 | 状态 |
|------|------|------|------|
| +X | 8 | 2.1% | ⚠️ 需矫正 |
| +Z | 5 | 1.3% | ⚠️ 需矫正 |
| -X | 31 | 8.1% | ⚠️ 需矫正 |
| -Z | 337 | 88.5% | ✅ 已对齐 |

---

## ⚠️ 需要矫正的数据（共44个）

| 文件 | 对称类型 | 当前方向 | 需要旋转 |
|------|---------|----------|----------|
| `airplane/airplane_0001.ply` | 1个正面 | +Z | 绕Y轴旋转180° |
| `airplane/airplane_0002.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0004.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0005.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0007.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0009.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0011.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0012.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0016.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0017.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0020.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0023.ply` | 1个正面 | +Z | 绕Y轴旋转180° |
| `airplane/airplane_0024.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0028.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0032.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0037.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0039.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0048.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0052.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0054.ply` | 1个正面 | +Z | 绕Y轴旋转180° |
| `airplane/airplane_0055.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0056.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0058.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0059.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0060.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0064.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0065.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0066.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0067.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0068.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0071.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0075.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0076.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0077.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0082.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0085.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0086.ply` | 1个正面 | +Z | 绕Y轴旋转180° |
| `airplane/airplane_0089.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0097.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0098.ply` | 1个正面 | -X | 绕Y轴旋转+90° |
| `airplane/airplane_0101.ply` | 1个正面 | +Z | 绕Y轴旋转180° |
| `airplane/airplane_0103.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `airplane/airplane_0104.ply` | 1个正面 | +X | 绕Y轴旋转-90° |
| `bench/bench_0002.ply` | 1个正面 | +X | 绕Y轴旋转-90° |

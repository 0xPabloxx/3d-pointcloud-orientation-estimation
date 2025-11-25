# 点云对称性与方向标注报告

**生成时间**: 2025-11-25 11:10:57

**数据集**: /home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply

---

## 📊 标注统计

- **总标注数**: 110/193
- **完成进度**: 57.0%

### 对称性分布

| 对称类型 | 数量 | 占比 |
|---------|------|------|
| 1个正面 | 104 | 94.5% |
| 2个正面 | 1 | 0.9% |
| 4个正面 | 3 | 2.7% |
| 完全对称 | 2 | 1.8% |

### 正面方向分布

| 方向 | 数量 | 占比 | 状态 |
|------|------|------|------|
| +X | 6 | 5.5% | ⚠️ 需矫正 |
| +Z | 5 | 4.5% | ⚠️ 需矫正 |
| -X | 33 | 30.0% | ⚠️ 需矫正 |
| -Z | 66 | 60.0% | ✅ 已对齐 |

---

## ⚠️ 需要矫正的数据（共44个）

| 文件 | 对称类型 | 当前方向 | 需要旋转 |
|------|---------|----------|----------|
| `airplane/airplane_0001.ply` | 1个正面 | +Z | 绕X轴旋转180° |
| `airplane/airplane_0002.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0004.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0005.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0007.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0009.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0011.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `airplane/airplane_0012.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0016.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0017.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `airplane/airplane_0020.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0023.ply` | 1个正面 | +Z | 绕X轴旋转180° |
| `airplane/airplane_0024.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0028.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0032.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0037.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0039.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0048.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0052.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `airplane/airplane_0054.ply` | 1个正面 | +Z | 绕X轴旋转180° |
| `airplane/airplane_0055.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0056.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0058.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `airplane/airplane_0059.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0060.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0064.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0065.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0066.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0067.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0068.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `airplane/airplane_0071.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0075.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0076.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0077.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0082.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0085.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0086.ply` | 1个正面 | +Z | 绕X轴旋转180° |
| `airplane/airplane_0089.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0097.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0098.ply` | 1个正面 | -X | 绕Y轴旋转-90° |
| `airplane/airplane_0101.ply` | 1个正面 | +Z | 绕X轴旋转180° |
| `bench/bench_0002.ply` | 1个正面 | +X | 绕Y轴旋转+90° |
| `glass_box/glass_box_0102.ply` | 4个正面 | -X | 绕Y轴旋转-90° |
| `glass_box/glass_box_0103.ply` | 4个正面 | -X | 绕Y轴旋转-90° |

---

## 📋 完整标注列表

| 文件 | 对称类型 | 方向 | 对齐状态 |
|------|---------|------|----------|
| `airplane/airplane_0001.ply` | 1个正面 | +Z | ⚠️ |
| `airplane/airplane_0002.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0003.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0004.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0005.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0006.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0007.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0008.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0009.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0010.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0011.ply` | 1个正面 | +X | ⚠️ |
| `airplane/airplane_0012.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0013.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0014.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0015.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0016.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0017.ply` | 1个正面 | +X | ⚠️ |
| `airplane/airplane_0018.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0019.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0020.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0021.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0022.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0023.ply` | 1个正面 | +Z | ⚠️ |
| `airplane/airplane_0024.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0025.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0026.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0027.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0028.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0029.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0030.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0031.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0032.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0033.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0034.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0035.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0036.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0037.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0038.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0039.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0040.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0041.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0042.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0043.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0044.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0045.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0046.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0047.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0048.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0049.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0050.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0051.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0052.ply` | 1个正面 | +X | ⚠️ |
| `airplane/airplane_0053.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0054.ply` | 1个正面 | +Z | ⚠️ |
| `airplane/airplane_0055.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0056.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0057.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0058.ply` | 1个正面 | +X | ⚠️ |
| `airplane/airplane_0059.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0060.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0061.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0062.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0063.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0064.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0065.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0066.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0067.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0068.ply` | 1个正面 | +X | ⚠️ |
| `airplane/airplane_0069.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0070.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0071.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0072.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0073.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0074.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0075.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0076.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0077.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0078.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0079.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0080.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0081.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0082.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0083.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0084.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0085.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0086.ply` | 1个正面 | +Z | ⚠️ |
| `airplane/airplane_0087.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0088.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0089.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0090.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0091.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0092.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0093.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0094.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0095.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0096.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0097.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0098.ply` | 1个正面 | -X | ⚠️ |
| `airplane/airplane_0099.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0100.ply` | 1个正面 | -Z | ✅ |
| `airplane/airplane_0101.ply` | 1个正面 | +Z | ⚠️ |
| `bench/bench_0001.ply` | 2个正面 | -Z | ✅ |
| `bench/bench_0002.ply` | 1个正面 | +X | ⚠️ |
| `bowl/bowl_0001.ply` | 完全对称 | -Z | ✅ |
| `bowl/bowl_0002.ply` | 完全对称 | -Z | ✅ |
| `chair/chair_0001.ply` | 1个正面 | -Z | ✅ |
| `chair/chair_0002.ply` | 1个正面 | -Z | ✅ |
| `glass_box/glass_box_0102.ply` | 4个正面 | -X | ⚠️ |
| `glass_box/glass_box_0103.ply` | 4个正面 | -X | ⚠️ |
| `glass_box/glass_box_0104.ply` | 4个正面 | -Z | ✅ |

# Glass Box Stage-1 Debug Notes

## 当前实验设置
- 脚本：`train_glass_box_stage1.py`
- 参数：`--epochs 60 --batch 8 --limit 40 --const-kappa 8.0 --num-workers 0`
- 数据：glass_box 40 个样本（切分后 Train 28 / Val 6 / Test 6）
- 输出目录：`/home/pablo/ForwardNet/results/glass_box_stage1_debug`

## 主要观察
1. **训练/验证/测试 KL 全程恒定在 7.4819**  
   - 每个 epoch 的日志完全不变，`summary.txt` 也记录了同样的最优值。
2. **极坐标可视化显示预测峰分布接近随机**（`figs/sample_00_b*.png`），没有向 4 个 GT 方向收敛。
3. **PyTorch 警告 CUDA 初始化失败**，说明当前运行落在 CPU；性能尚可，但需确认该环境能否正常感知 GPU（后续与其它训练共存时可能要手动指定设备）。

## 初步推测
1. **常量损失来源**：`stage1_loss` 使用固定 `κ_pred=κ_gt=8`，当 μ 差为 90° 时，KL 精确等于 7.4819（推导见内联脚本）。  
   - 目前模型输出似乎始终保持在“错 90°”的状态，因此平均成本固定。
2. **梯度未生效？**  
   - 匈牙利匹配后 `losses[b] = cost[row, col].mean()` 应该对 μ_pred 可导，但仍然 60 个 epoch 无任何下降；需要检查：
     - `linear_sum_assignment` 输出的 row/col numpy 索引在回写 torch tensor 时是否阻断梯度（目前是 torch tensor用 row/col 做 index，应该仍可传播）。
     - PointNet++ 初始化是否导致输出 μ 靠近某组固定方向，且学习率 1e-3 仍不足以推动它离开。
     - `const_kappa` 与 GT 相同，使得 loss 对 μ 的梯度只有 `cos(delta)` 通道；如果 delta 一直接近 ±π/2，梯度非零，就应有更新，说明问题不在 loss 本身，而在 optimizer/input。
3. **CPU-only 运行** 可能隐藏了别的警告；但就算在 CPU，梯度也应更新，因此需要进一步打印 μ_pred 和梯度，确认是否真的没有传播。

## 下一步建议
1. **验证梯度链路**：在一个 batch 上手工 forward/backward，打印 `mu_pred.grad`，确认是否为 0。
2. **尝试随机初始化扰动**：例如在 forward 前强制对 μ_head 权重做轻微噪声，观察 loss 是否跳出 7.48。
3. **放开 κ 预测或增加权重 regularizer**：给网络更多可调节自由度，避免 loss 退化到常数值。
4. **确认 CUDA 可用性**：虽然本次依赖 CPU 也能跑，但为与其他实验保持一致，后续需解决 `cudaGetDeviceCount` 警告（可能是沙箱限制，需要和环境 owner 协调）。

---

## 2025-11-09 实验补充：0.74 vs 7.4 的关系
- 为了验证旧实验（glass_box loss ≈ 0.74）与当前 stage-1 常量 7.4 之间的关系，给 `train_glass_box_stage1.py` 新增 `--loss-scale`，将匈牙利匹配后的 KL 乘以系数。
- 运行设置与前述相同（Train 28 / Val 6 / Test 6，`const_kappa=8`，`--limit 40`，`--num-workers 0`），只改变 `loss_scale`：

| loss_scale | 输出目录 | 最佳 Val/Test KL |
| ---------- | -------- | ---------------- |
| 1.0 | `/home/pablo/ForwardNet/results/glass_box_stage1_scale1` | 7.4819 |
| 0.1 | `/home/pablo/ForwardNet/results/glass_box_stage1_scale0p1` | 0.7482 |

- 结果表明：**0.7482 ≈ 7.4819 × 0.1**，即早期实验报告的 ~0.74 与当前 7.4 仅差一个常数缩放。旧脚本在 `match_loss` 中对 cost 做了“预测权重加权且权重可被压到极小”的处理，相当于对 KL 施加了一个 <1 的缩放；现在通过显式 `loss_scale` 得到同样数值，验证了两者的数量级对应关系。
- 后续若要与旧实验对齐，可：
  1. 保持 `loss_scale=0.1`；
  2. 或者恢复“按预测权重归一化”的操作，但需要额外的均衡正则，避免模型通过调权重逃避损失。

> 每次类似的调试/分析结论将继续追加到该文件或新建条目标注日期，方便论文撰写引用。

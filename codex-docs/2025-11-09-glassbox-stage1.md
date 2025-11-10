# 2025-11-09 Glass Box Stage-1 实验记录

## 概要
- Worktree: `/home/pablo/ForwardNet-codex` （分支 `codex`）
- 目标：跑通 glass_box 专用 Stage-1 训练，验证四峰 MvM 分布，定位 loss 恒定问题
- 结果：loss 始终稳定在 0.7482（或 7.4819，取决于缩放），确认与旧实验的 0.74 量级一致

## 关键变更
1. 新增脚本 `train_glass_box_stage1.py`
   - 仅加载 glass_box 数据，支持 `--limit` 小样本过拟合测试
   - Stage-1 loss 固定 κ 与均匀权重，输出极坐标图
   - 新增 `--loss-scale`，可将匈牙利匹配后的 KL 按常数缩放
2. 新增记录文件 `glassbox_stage1_findings.md`
   - 详细描述 loss 恒定现象、推理以及 0.74 vs 7.4 的关系
3. 更新 `codex.md`
   - 记住“用中文回答”“GPU 监控策略”“分析需写 Markdown”等约定

## 实验摘要
- `loss_scale=1.0`：`/home/pablo/ForwardNet/results/glass_box_stage1_scale1`
  - 训练/验证/测试 KL 恒 7.4819
- `loss_scale=0.1`：`/home/pablo/ForwardNet/results/glass_box_stage1_scale0p1`
  - KL 恒 0.7482（与旧多类别脚本的 0.74 对齐）
- GPU 版本（完整 191/40/40 样本、40 epoch）：`/home/pablo/ForwardNet/results/glass_box_stage1_gpu`
  - 仍为 0.7482，说明问题不在硬件

## 当前结论
- Loss 恒定源于匈牙利匹配后 KL 被均匀平均 + 固定 κ；`loss_scale` 只是线性缩放，不影响梯度问题
- 需要进一步检查梯度链路或引入权重正则，以迫使预测峰脱离常值

## 下一步建议
1. 在 `stage1_loss` 中恢复预测权重并加入均衡正则，观察 loss 是否下降
2. 手动做梯度检查（打印 `mu_pred.grad`）确认匈牙利匹配后的梯度是否畅通
3. 继续监控 GPU，等其它任务完成后再跑改进实验

> 所有关联日志/模型/图表已保存在上述 results 目录，可直接用于论文写作。

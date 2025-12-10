# Codex 快速对齐

- 工作区：`/home/pablo/ForwardNet-claude`，默认中文回复。
- 只在根目录保留 `claude.md`、`project_structure.md`；其他说明放 `docs/`。
- 数据：seed=42，train/val/test 分离（默认 7:2:1），只对 train 做增强，val/test 禁用增强。
- 训练前提醒用户（任务/资源/时长），确认后再跑；异常及时汇报。
- 命名：markdown `analysis_YYYYMMDD_<topic>.md` 等；Python 文件加角色前缀并写明 model/data/loss/usage。

## 当前两项核心任务（待实现数据处理由用户给出后再动）

1) **固定峰数的 von Mises 混合网络**
   - 输出：固定 N 个分量的 `(mu_bit, kappa, weight)`（N 由配置指定，需能覆盖 K=1/2/4 个正面，以及无正面 kappa=0、旋转堆叠 kappa=0）。
   - 输入：Ground truth 提供 mu 和 kappa（数据处理稍后按用户要求做）。
   - 训练：使用混合 von Mises 的 NLL（或等价 Monte Carlo）作为 loss；需能兼容上述混合数据情形。

2) **离散方向采样网络**
   - 输出：8 或 16 个离散方向的 softmax 概率分布。
   - Ground truth：1/2/4 个正面标签处理成对应的离散方向向量（不改原始 GT，训练时在线转换）。
   - 训练：用交叉熵或合适的分布型 loss，让模型在离散方向上学习多峰/对称。

## 实验记录

详见 `claude.md` 中的 "🧪 Fixed 4-Peak 实验记录" 部分。

#!/bin/bash
# =============================================================================
# DR Series: 基础方向投影 + Softmax
# =============================================================================
#
# DR vs D 的根本区别：
#
#   D系列 (槽分类):
#     网络输出: logits (任意值，无物理意义)
#     处理: softmax(logits) → 概率分布
#     含义: "方向落在哪个bin的概率"
#
#   DR系列 (基础方向投影):
#     网络输出: 投影值回归 (tanh约束到[-1,1]，有物理意义)
#     处理: softmax(投影值) → 概率分布
#     含义: "与各基础方向的相似度分布"
#
# GT对比 (方向270°):
#   DR (cos投影 → softmax): [0.10, 0.05, 0.04, 0.05, 0.10, 0.20, 0.27, 0.20]
#   D  (τ=5):               [0.01, 0.00, 0.00, 0.00, 0.01, 0.16, 0.68, 0.16]
#
# =============================================================================

set -e
cd /home/pablo/ForwardNet-claude

echo "=============================================="
echo "  DR Series: Projection + Softmax"
echo "  $(date)"
echo "=============================================="

# Common parameters
EPOCHS=50
BATCH_SIZE=32
CATEGORIES="1_front,4_fronts,no_front"
WANDB_PROJECT="ForwardNet-LossAblation"

# -----------------------------------------------------------------------------
# DR_8a: 8 bins, KL loss
# -----------------------------------------------------------------------------
echo ""
echo "[DR_8a] 8 bins, KL loss"
python train_direction.py \
    --mode discrete \
    --exp_name DR_8a \
    --categories "$CATEGORIES" \
    --num_bins 8 \
    --gt_mode dr \
    --d_loss_type kl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project $WANDB_PROJECT

# -----------------------------------------------------------------------------
# DR_8b: 8 bins, CE loss
# -----------------------------------------------------------------------------
echo ""
echo "[DR_8b] 8 bins, CE loss"
python train_direction.py \
    --mode discrete \
    --exp_name DR_8b \
    --categories "$CATEGORIES" \
    --num_bins 8 \
    --gt_mode dr \
    --d_loss_type ce \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project $WANDB_PROJECT

# -----------------------------------------------------------------------------
# DR_16a: 16 bins, KL loss
# -----------------------------------------------------------------------------
echo ""
echo "[DR_16a] 16 bins, KL loss"
python train_direction.py \
    --mode discrete \
    --exp_name DR_16a \
    --categories "$CATEGORIES" \
    --num_bins 16 \
    --gt_mode dr \
    --d_loss_type kl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project $WANDB_PROJECT

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  DR Series Completed!"
echo "  $(date)"
echo "=============================================="
echo ""
echo "Experiments:"
echo "  DR_8a:  8 bins, KL loss"
echo "  DR_8b:  8 bins, CE loss"
echo "  DR_16a: 16 bins, KL loss"
echo ""
echo "Checkpoints:"
ls -lt checkpoints/ | grep "DR_" | head -5

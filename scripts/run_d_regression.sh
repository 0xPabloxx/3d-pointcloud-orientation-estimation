#!/bin/bash
# =============================================================================
# D Series Regression Experiments - 基础方向投影回归
# =============================================================================
#
# 与槽分类(projection/onehot)的区别：
#   槽分类: 输出概率分布 [0,1], sum=1, 用Softmax+CE
#   投影回归: 输出投影值 [-1,1], 用Tanh+MSE/Cosine
#
# =============================================================================

set -e
cd /home/pablo/ForwardNet-claude

echo "=============================================="
echo "  D Series: Projection Regression Experiments"
echo "  $(date)"
echo "=============================================="

# Common parameters
EPOCHS=50
BATCH_SIZE=32
CATEGORIES="1_front,4_fronts,no_front"

# -----------------------------------------------------------------------------
# D_8r_mse: 8 bins, MSE loss
# -----------------------------------------------------------------------------
echo ""
echo "[D_8r_mse] 8 bins, regression, MSE loss"
python train_direction.py \
    --mode discrete \
    --exp_name D_8r_mse \
    --categories "$CATEGORIES" \
    --num_bins 8 \
    --gt_mode regression \
    --reg_loss_type mse \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project ForwardNet-LossAblation

# -----------------------------------------------------------------------------
# D_8r_cos: 8 bins, Cosine loss
# -----------------------------------------------------------------------------
echo ""
echo "[D_8r_cos] 8 bins, regression, Cosine loss"
python train_direction.py \
    --mode discrete \
    --exp_name D_8r_cos \
    --categories "$CATEGORIES" \
    --num_bins 8 \
    --gt_mode regression \
    --reg_loss_type cosine \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project ForwardNet-LossAblation

# -----------------------------------------------------------------------------
# D_8r_comb: 8 bins, Combined loss (MSE + Cosine)
# -----------------------------------------------------------------------------
echo ""
echo "[D_8r_comb] 8 bins, regression, Combined loss"
python train_direction.py \
    --mode discrete \
    --exp_name D_8r_comb \
    --categories "$CATEGORIES" \
    --num_bins 8 \
    --gt_mode regression \
    --reg_loss_type combined \
    --lambda_mse 1.0 \
    --lambda_cos 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project ForwardNet-LossAblation

# -----------------------------------------------------------------------------
# D_16r_cos: 16 bins, Cosine loss
# -----------------------------------------------------------------------------
echo ""
echo "[D_16r_cos] 16 bins, regression, Cosine loss"
python train_direction.py \
    --mode discrete \
    --exp_name D_16r_cos \
    --categories "$CATEGORIES" \
    --num_bins 16 \
    --gt_mode regression \
    --reg_loss_type cosine \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb \
    --wandb_project ForwardNet-LossAblation

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  D Regression Series Completed!"
echo "  $(date)"
echo "=============================================="
echo ""
echo "Experiments:"
echo "  D_8r_mse:  8 bins, MSE loss"
echo "  D_8r_cos:  8 bins, Cosine loss"
echo "  D_8r_comb: 8 bins, Combined (MSE+Cosine)"
echo "  D_16r_cos: 16 bins, Cosine loss"
echo ""
ls -lt checkpoints/ | grep "D_.*r_" | head -5

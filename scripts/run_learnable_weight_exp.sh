#!/bin/bash
# =============================================================================
# Learnable Weight Experiment Runner
#
# 可学习权重 vs 固定权重 对比实验
#
# 实验配置:
#   LW_1a: Baseline (固定权重)
#   LW_1b: 可学习权重 (无监督)
#   LW_1c: 可学习权重 + 熵正则 (鼓励稀疏)
#   LW_1d: 可学习权重 + GT监督
#
# 数据集: 1_front + 4_fronts + no_front (混合训练)
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  Learnable Weight Experiment"
echo "  可学习权重 vs 固定权重 对比实验"
echo "  $(date)"
echo "=============================================="

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU available:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, using CPU${NC}"
fi

echo ""

# 公共参数
EPOCHS=50
BATCH_SIZE=32
CATEGORIES="1_front,4_fronts,no_front"

# =============================================================================
# 实验组1: 全类别混合训练
# =============================================================================

echo "=============================================="
echo "  Group 1: All Categories Mixed"
echo "  Categories: $CATEGORIES"
echo "=============================================="

# LW_1a: Baseline (固定权重)
echo -e "\n${YELLOW}[1/4] LW_1a: Baseline (Fixed Weights)${NC}"
python train_single_front_learnable.py \
    --exp_name LW_1a_baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# LW_1b: 可学习权重 (无监督)
echo -e "\n${YELLOW}[2/4] LW_1b: Learnable Weights (Free)${NC}"
python train_single_front_learnable.py \
    --exp_name LW_1b_learnable_free \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --learnable_weights \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --lambda_weight 0.0 \
    --weight_loss_type none \
    --wandb

# LW_1c: 可学习权重 + 熵正则
echo -e "\n${YELLOW}[3/4] LW_1c: Learnable Weights + Entropy Regularization${NC}"
python train_single_front_learnable.py \
    --exp_name LW_1c_learnable_entropy \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --learnable_weights \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --lambda_weight 0.1 \
    --weight_loss_type entropy \
    --wandb

# LW_1d: 可学习权重 + GT监督
echo -e "\n${YELLOW}[4/4] LW_1d: Learnable Weights + GT Supervision${NC}"
python train_single_front_learnable.py \
    --exp_name LW_1d_learnable_gt \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --learnable_weights \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --lambda_weight 1.0 \
    --weight_loss_type gt \
    --wandb

# =============================================================================
# 实验组2: 单类别训练 (可选)
# =============================================================================

echo ""
echo "=============================================="
echo "  Group 2: Single Category (1_front only)"
echo "=============================================="

# LW_2a: 只用1_front，固定权重
echo -e "\n${YELLOW}[5/6] LW_2a: 1_front only, Fixed Weights${NC}"
python train_single_front_learnable.py \
    --exp_name LW_2a_1front_baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories "1_front" \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# LW_2b: 只用1_front，可学习权重
echo -e "\n${YELLOW}[6/6] LW_2b: 1_front only, Learnable Weights${NC}"
python train_single_front_learnable.py \
    --exp_name LW_2b_1front_learnable \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --categories "1_front" \
    --learnable_weights \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --lambda_weight 0.0 \
    --weight_loss_type none \
    --wandb

# =============================================================================
# 完成
# =============================================================================

echo ""
echo -e "${GREEN}=============================================="
echo "  All experiments completed!"
echo "  $(date)"
echo "==============================================${NC}"

# 列出生成的checkpoints
echo ""
echo "Generated checkpoints:"
ls -lt checkpoints/ | grep "LW_" | head -10

echo ""
echo -e "${BLUE}Experiment Summary:${NC}"
echo "  LW_1a: Baseline (fixed weights, all categories)"
echo "  LW_1b: Learnable weights, no regularization"
echo "  LW_1c: Learnable weights + entropy (sparse)"
echo "  LW_1d: Learnable weights + GT supervision"
echo "  LW_2a: 1_front only, fixed weights"
echo "  LW_2b: 1_front only, learnable weights"
echo ""
echo "Check WandB for detailed results: https://wandb.ai"

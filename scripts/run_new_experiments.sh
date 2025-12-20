#!/bin/bash
# =============================================================================
# New Experiments Runner
#
# 实验1: 单正面方向预测 (5个sub-experiments)
#   - SF_1a: Pure KL Loss
#   - SF_1b: Angle Loss only (μ监督)
#   - SF_1c: Kappa Loss only (κ监督)
#   - SF_1d: KL + Kappa + Mu (组合)
#   - SF_1e: Reverse KL
#
# 实验2: 对称性分类器
#   - SymClassifier
#
# 每个实验50 epoch
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  New Experiments Runner"
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

# =============================================================================
# 实验1: 单正面方向预测
# =============================================================================

echo "=============================================="
echo "  Experiment 1: Single Front Direction"
echo "=============================================="

# SF_1a: Pure KL Loss
echo -e "\n${YELLOW}[1/6] SF_1a: Pure KL Loss${NC}"
python train_single_front.py \
    --exp_name SF_1a_pureKL \
    --epochs 50 \
    --lambda_kl 1.0 \
    --lambda_kappa 0.0 \
    --lambda_mu 0.0 \
    --wandb

# SF_1b: Angle Loss only
echo -e "\n${YELLOW}[2/6] SF_1b: Angle Loss only (μ监督)${NC}"
python train_single_front.py \
    --exp_name SF_1b_angleLoss \
    --epochs 50 \
    --lambda_kl 0.0 \
    --lambda_kappa 0.0 \
    --lambda_mu 2.0 \
    --wandb

# SF_1c: Kappa Loss only
echo -e "\n${YELLOW}[3/6] SF_1c: Kappa Loss only (κ监督)${NC}"
python train_single_front.py \
    --exp_name SF_1c_kappaLoss \
    --epochs 50 \
    --lambda_kl 0.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 0.0 \
    --wandb

# SF_1d: KL + Kappa + Mu
echo -e "\n${YELLOW}[4/6] SF_1d: KL + Kappa + Mu (组合)${NC}"
python train_single_front.py \
    --exp_name SF_1d_combined \
    --epochs 50 \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# SF_1e: Reverse KL
echo -e "\n${YELLOW}[5/6] SF_1e: Reverse KL${NC}"
python train_single_front.py \
    --exp_name SF_1e_reverseKL \
    --epochs 50 \
    --lambda_kl 1.0 \
    --lambda_kappa 0.0 \
    --lambda_mu 0.0 \
    --reverse_kl \
    --wandb

# =============================================================================
# 实验2: 对称性分类器
# =============================================================================

echo ""
echo "=============================================="
echo "  Experiment 2: Symmetry Classifier"
echo "=============================================="

# SymClassifier
echo -e "\n${YELLOW}[6/6] SymClassifier${NC}"
python train_symmetry_classifier.py \
    --exp_name SymClassifier \
    --epochs 50 \
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
ls -lt checkpoints/ | head -10

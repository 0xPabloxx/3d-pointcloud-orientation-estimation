#!/bin/bash
# =============================================================================
# MF & D Series Experiment Runner
#
# MF系列: Mixture of von Mises (1_front + 4_fronts)
# D系列: Discrete Direction Bins (1_front + 4_fronts + no_front)
#
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=============================================="
echo "  MF & D Series Experiments"
echo "  $(date)"
echo "=============================================="

# GPU检查
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU available:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
fi

# 公共参数
EPOCHS=50
BATCH_SIZE=32

# =============================================================================
# MF系列 (Mixture of von Mises)
# Categories: 1_front, 4_fronts
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  MF Series: Mixture of von Mises"
echo "  Categories: 1_front, 4_fronts"
echo "==============================================${NC}"

# MF_1a: Combined Loss (基准)
echo -e "\n${YELLOW}[MF_1a] Combined Loss (KL + κ + μ)${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_1a_combined \
    --categories "1_front,4_fronts" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss_type combined \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# MF_1b: Reverse KL
echo -e "\n${YELLOW}[MF_1b] Reverse KL${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_1b_reverse_kl \
    --categories "1_front,4_fronts" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss_type reverse_kl \
    --lambda_kl 1.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# MF_1c: Pure Mu Loss
echo -e "\n${YELLOW}[MF_1c] Pure Mu Loss (Hungarian)${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_1c_mu_only \
    --categories "1_front,4_fronts" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss_type mu_only \
    --lambda_mu 2.0 \
    --wandb

# MF_1d: Heavy Kappa
echo -e "\n${YELLOW}[MF_1d] Heavy Kappa (κ=10, μ=1)${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_1d_heavy_kappa \
    --categories "1_front,4_fronts" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss_type combined \
    --lambda_kl 1.0 \
    --lambda_kappa 10.0 \
    --lambda_mu 1.0 \
    --wandb

# MF_1e: No KL, only κ+μ (Exp-2证明最佳)
echo -e "\n${YELLOW}[MF_1e] No KL, κ+μ only (Best from Exp-2)${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_1e_kappa_mu_only \
    --categories "1_front,4_fronts" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss_type combined \
    --lambda_kl 0.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# =============================================================================
# D系列 (Discrete Bins + Projection Softmax)
# Categories: 1_front, 4_fronts, no_front
# GT模式: projection (投影+softmax，保留角度信息)
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  D Series: Discrete Direction (Projection+Softmax)"
echo "  Categories: 1_front, 4_fronts, no_front"
echo "==============================================${NC}"

# D_8a: 8 bins, CE, τ=5 (基准)
echo -e "\n${YELLOW}[D_8a] 8 bins, CE, τ=5 (baseline)${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8a_proj_t5 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 5.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_8b: 8 bins, CE, τ=3 (更软的分布)
echo -e "\n${YELLOW}[D_8b] 8 bins, CE, τ=3 (softer)${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8b_proj_t3 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 3.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_8c: 8 bins, CE, τ=10 (更尖锐的分布)
echo -e "\n${YELLOW}[D_8c] 8 bins, CE, τ=10 (sharper)${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8c_proj_t10 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 10.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_8d: 8 bins, KL Divergence
echo -e "\n${YELLOW}[D_8d] 8 bins, KL Divergence${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8d_proj_kl \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 5.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type kl \
    --wandb

# D_8e: 8 bins, 只用1_front+4_fronts (与MF公平对比)
echo -e "\n${YELLOW}[D_8e] 8 bins, 1_front+4_fronts only (fair comparison with MF)${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8e_proj_no_nofront \
    --categories "1_front,4_fronts" \
    --num_bins 8 \
    --gt_mode projection \
    --temperature 5.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_16a: 16 bins, CE, τ=5
echo -e "\n${YELLOW}[D_16a] 16 bins, CE, τ=5${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_16a_proj_t5 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 16 \
    --gt_mode projection \
    --temperature 5.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_16b: 16 bins, CE, τ=8 (16bins可能需要更尖锐)
echo -e "\n${YELLOW}[D_16b] 16 bins, CE, τ=8${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_16b_proj_t8 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 16 \
    --gt_mode projection \
    --temperature 8.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# D_32a: 32 bins, CE, τ=10 (高精度测试)
echo -e "\n${YELLOW}[D_32a] 32 bins, CE, τ=10 (high precision test)${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_32a_proj_t10 \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 32 \
    --gt_mode projection \
    --temperature 10.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_loss_type ce \
    --wandb

# =============================================================================
# 10000点对比实验
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  10K Points Comparison"
echo "==============================================${NC}"

# MF_10k: MF with 10000 points
echo -e "\n${YELLOW}[MF_10k] MF Combined, 10000 points${NC}"
python train_direction.py \
    --mode mf \
    --exp_name MF_10k_combined \
    --categories "1_front,4_fronts" \
    --num_points 10000 \
    --epochs $EPOCHS \
    --batch_size 16 \
    --loss_type combined \
    --lambda_kl 0.0 \
    --lambda_kappa 5.0 \
    --lambda_mu 2.0 \
    --wandb

# D_8_10k: D8 with 10000 points
echo -e "\n${YELLOW}[D_8_10k] D8 Projection, 10000 points${NC}"
python train_direction.py \
    --mode discrete \
    --exp_name D_8_10k_proj \
    --categories "1_front,4_fronts,no_front" \
    --num_bins 8 \
    --num_points 10000 \
    --gt_mode projection \
    --temperature 5.0 \
    --epochs $EPOCHS \
    --batch_size 16 \
    --d_loss_type ce \
    --wandb

# =============================================================================
# 完成
# =============================================================================

echo ""
echo -e "${GREEN}=============================================="
echo "  All experiments completed!"
echo "  $(date)"
echo "==============================================${NC}"

echo ""
echo "Generated checkpoints:"
ls -lt checkpoints/ | grep -E "MF_|D_" | head -15

echo ""
echo -e "${BLUE}Experiment Summary:${NC}"
echo "  MF Series (Mixture von Mises, 1_front + 4_fronts):"
echo "    MF_1a: Combined (KL + κ + μ)"
echo "    MF_1b: Reverse KL"
echo "    MF_1c: Pure μ Loss"
echo "    MF_1d: Heavy κ"
echo "    MF_1e: No KL, κ+μ only (Best from Exp-2)"
echo ""
echo "  D Series (Discrete Bins + Projection Softmax):"
echo "    D_8a:  8 bins, CE, τ=5 (baseline)"
echo "    D_8b:  8 bins, CE, τ=3 (softer)"
echo "    D_8c:  8 bins, CE, τ=10 (sharper)"
echo "    D_8d:  8 bins, KL Divergence"
echo "    D_8e:  8 bins, CE (1_front+4_fronts only, fair comparison)"
echo "    D_16a: 16 bins, CE, τ=5"
echo "    D_16b: 16 bins, CE, τ=8"
echo "    D_32a: 32 bins, CE, τ=10 (high precision test)"
echo ""
echo "  10K Points Comparison:"
echo "    MF_10k:  MF Combined, 10000 points (vs 2048)"
echo "    D_8_10k: D8 Projection, 10000 points (vs 2048)"
echo ""
echo "Check WandB for detailed results"

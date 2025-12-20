#!/bin/bash
# =============================================================================
# D Series Experiment Runner (Discrete Direction + Projection Softmax)
# Categories: 1_front, 4_fronts, no_front
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=============================================="
echo "  D Series: Discrete Direction Experiments"
echo "  $(date)"
echo "=============================================="

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU available:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
fi

# Common parameters
EPOCHS=50
BATCH_SIZE=32

cd /home/pablo/ForwardNet-claude

# =============================================================================
# D Series: 8 bins experiments
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  D Series: 8 bins experiments"
echo "==============================================${NC}"

# D_8a: 8 bins, CE, τ=5 (baseline)
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

# D_8b: 8 bins, CE, τ=3 (softer)
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

# D_8c: 8 bins, CE, τ=10 (sharper)
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

# D_8e: 8 bins, 1_front+4_fronts only (fair comparison with MF)
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

# =============================================================================
# D Series: 16 bins experiments
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  D Series: 16 bins experiments"
echo "==============================================${NC}"

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

# D_16b: 16 bins, CE, τ=8
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

# =============================================================================
# D Series: 32 bins experiment
# =============================================================================

echo ""
echo -e "${BLUE}=============================================="
echo "  D Series: 32 bins experiment"
echo "==============================================${NC}"

# D_32a: 32 bins, CE, τ=10 (high precision)
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
# Complete
# =============================================================================

echo ""
echo -e "${GREEN}=============================================="
echo "  D Series experiments completed!"
echo "  $(date)"
echo "==============================================${NC}"

echo ""
echo "Generated checkpoints:"
ls -lt checkpoints/ | grep "D_" | head -10

echo ""
echo -e "${BLUE}D Series Summary:${NC}"
echo "  8 bins:"
echo "    D_8a:  CE, τ=5 (baseline)"
echo "    D_8b:  CE, τ=3 (softer)"
echo "    D_8c:  CE, τ=10 (sharper)"
echo "    D_8d:  KL Divergence"
echo "    D_8e:  CE, 1_front+4_fronts only"
echo "  16 bins:"
echo "    D_16a: CE, τ=5"
echo "    D_16b: CE, τ=8"
echo "  32 bins:"
echo "    D_32a: CE, τ=10 (high precision)"
echo ""
echo "Check WandB for detailed results"

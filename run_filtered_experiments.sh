#!/bin/bash
# Run filtered experiments: mu_only vs P2v2_SoftGate
# 1-front only uses airplane and chair

echo "=============================================="
echo "Filtered Experiments"
echo "1-front: airplane + chair only"
echo "=============================================="

# Run both experiments
python train_filtered_experiments.py \
    --exp both \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_rotations 4 \
    --allowed_1front_categories airplane chair \
    --wandb \
    --wandb_project ForwardNet-FilteredExp

echo "=============================================="
echo "Done!"
echo "=============================================="

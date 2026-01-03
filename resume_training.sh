#!/bin/bash
# Resume P2v2 training after computer restart
#
# Usage: bash resume_training.sh [checkpoint_dir]
# Example: bash resume_training.sh checkpoints/P2v2_Clean_20251230_165848

P2V2_DIR=${1:-"checkpoints/P2v2_Clean_20251230_165848"}

echo "=============================================="
echo "Resume Training"
echo "=============================================="
echo "P2v2 checkpoint: $P2V2_DIR"
echo ""

# Check if checkpoint exists
if [ ! -d "$P2V2_DIR" ]; then
    echo "ERROR: Directory $P2V2_DIR does not exist!"
    exit 1
fi

if [ ! -f "$P2V2_DIR/latest.pth" ] && [ ! -f "$P2V2_DIR/best.pth" ]; then
    echo "ERROR: No checkpoint found in $P2V2_DIR"
    exit 1
fi

echo "Starting resume..."
echo ""

python train_clean_pipeline.py \
    --allowed_1front_categories airplane chair \
    --outlier_json data_annotation/1front_outliers.json \
    --outlier_threshold severe \
    --skip_classifier \
    --classifier_checkpoint checkpoints/CleanClassifier_20251229_220630/best.pth \
    --resume_p2v2 "$P2V2_DIR" \
    --p2v2_epochs 100 \
    --p2v2_num_rotations 12 \
    --p2v2_lr 1e-3 \
    --muonly_epochs 50 \
    --muonly_num_rotations 10 \
    --muonly_lr 1e-3 \
    --lambda_mu 2.0 \
    --batch_size 32 \
    --wandb \
    --wandb_project ForwardNet-LossAblation \
    --run_muonly

echo "=============================================="
echo "Done!"
echo "=============================================="

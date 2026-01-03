#!/bin/bash
# Clean Training Pipeline - 3个实验
#
# 实验计划:
# 1. 新分类器 (用干净数据)
# 2. P2v2_SoftGate (用新分类器 + 强增强)
# 3. MuOnly Baseline (原始MF_1c风格，不用MoE)

echo "=============================================="
echo "Clean Training Pipeline - 3 Experiments"
echo "=============================================="
echo ""
echo "数据过滤:"
echo "  - 1-front: 只用 airplane + chair"
echo "  - 排除异常值 (>=90° 误差)"
echo ""
echo "实验计划:"
echo "  1. 新分类器: 50 epochs, 12x旋转增强"
echo "  2. P2v2: 100 epochs, 12x旋转增强 (~10小时)"
echo "  3. MuOnly Baseline: 50 epochs, 10x旋转增强"
echo "=============================================="
echo ""

python train_clean_pipeline.py \
    --allowed_1front_categories airplane chair \
    --outlier_json data_annotation/1front_outliers.json \
    --outlier_threshold severe \
    --skip_classifier \
    --classifier_checkpoint checkpoints/CleanClassifier_20251229_220630/best.pth \
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

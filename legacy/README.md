# Legacy Scripts

This directory contains old training and dataloader scripts that have been replaced by the new unified framework.

## Migration Date: 2025-12-05

## Old Scripts → New Framework Mapping

### Training Scripts

| Old Script | New Equivalent | Notes |
|------------|----------------|-------|
| `train_old.py` | `train.py` | Original generic training script |
| `train_8dir*.py` | - | 8-direction classification (not migrated yet) |
| `train_single_peak_vonMises_KL.py` | `train.py --config configs/vm_1peak.yaml` | Single-peak vM |
| `train_multi_peaks_vonMises_KL.py` | `train.py --config configs/vm_4peak.yaml` | Multi-peak vM |
| `train_pointnetpp_mvm_glassbox_*.py` | `train.py --config configs/vm_4peak.yaml --k_filter 4` | PointNet++ + 4-peak vM |
| `train_dgcnn_mvm_glassbox_*.py` | `train.py --config configs/vm_4peak.yaml --backbone dgcnn` | DGCNN + 4-peak vM |
| `train_ptv3_mvm_glassbox_*.py` | - | Point Transformer V3 (not migrated yet) |
| `train_symmetry_classifier.py` | - | Symmetry K classification (separate task) |

### Dataloader Scripts

| Old Script | New Equivalent | Notes |
|------------|----------------|-------|
| `dataloader.py` | `datasets/orientation.py` | Base dataloader |
| `dataloader_glassbox_augmented.py` | `datasets/orientation.py` with `k_filter=[4]` | GlassBox specific |
| `dataloader_single_peak_vonMises.py` | `datasets/orientation.py` with `k_filter=[1]` | Single-front objects |
| `dataloader_multi_peak_vonMises.py` | `datasets/orientation.py` | Multi-peak vM GT |
| `dataloader_8dir_sampled.py` | - | 8-direction (not migrated) |
| `dataloader_symmetry.py` | - | Symmetry classification (separate task) |

## New Framework Usage

```bash
# Baseline: Direct regression
python train.py --config configs/baseline_direct.yaml

# Single-peak von Mises (K=1 objects only)
python train.py --config configs/vm_1peak.yaml

# 4-peak von Mises (K=4 objects only)
python train.py --config configs/vm_4peak.yaml

# 4-peak with learnable weights
python train.py --config configs/vm_4peak_learnable.yaml

# Override backbone
python train.py --config configs/vm_4peak.yaml --backbone dgcnn

# Override other parameters
python train.py --config configs/vm_4peak.yaml --lr 0.0005 --epochs 200
```

## Why Migrated?

1. **Code Duplication**: Old scripts had significant code repetition
2. **Hard to Maintain**: Each backbone/head combination had its own script
3. **No Config System**: Parameters were hardcoded
4. **No WandB Support**: Manual logging
5. **Inconsistent Interfaces**: Different scripts had different I/O formats

## Keeping Old Scripts

These scripts are kept for reference and to:
- Understand the original implementation details
- Compare results if needed
- Recover any functionality not yet migrated

## DO NOT USE

These scripts are deprecated and should not be used for new experiments.
Use the new `train.py` with appropriate config files instead.

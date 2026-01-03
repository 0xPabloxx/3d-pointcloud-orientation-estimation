# ForwardNet Tools

This directory contains standalone tools for data annotation, visualization, and model evaluation.

## Tool List

| Tool | Type | Port | Description |
|------|------|------|-------------|
| `vis_checkpoint_web.py` | Web | 8070 | Interactive checkpoint visualization |
| `vis_fixed_4peak.py` | CLI | - | Fixed 4-peak model visualization |
| `annotate_symmetry_web.py` | Web | 8052 | Symmetry annotation tool |
| `verify_gt_web.py` | Web | 8060 | Ground truth verification |
| `von_mises_interactive.py` | Web | 8055 | Interactive von Mises demo |
| `screenshot_viewer.py` | Web | 8051 | Point cloud screenshot viewer |
| `annotation_stats.py` | CLI | - | Annotation statistics |

---

## Web Tools

### 1. vis_checkpoint_web.py (NEW)
**Model Checkpoint Visualization Tool**

Interactive web tool for visualizing Fixed 4-Peak model predictions.

```bash
python tools/vis_checkpoint_web.py --port 8070
# Open http://localhost:8070
```

**Features:**
- Load any `.pth` checkpoint file
- Switch between train/val/test splits
- Filter by category (1_front, 2_fronts, 4_fronts, symmetric, no_front)
- Interactive Plotly polar plots
- Side-by-side GT vs Prediction comparison
- Detailed parameter table per sample

---

### 2. annotate_symmetry_web.py
**Symmetry Annotation Tool**

Web-based tool for annotating object symmetry types and front directions.

```bash
python tools/annotate_symmetry_web.py --port 8052
# Open http://localhost:8052
```

**Features:**
- 3D point cloud visualization
- Category annotation (K=1,2,4,0,-1)
- Front direction selection
- Progress tracking
- Auto-save annotations

---

### 3. verify_gt_web.py
**Ground Truth Verification Tool**

Verify the alignment between point clouds and generated GT distributions.

```bash
python tools/verify_gt_web.py --port 8060
# Open http://localhost:8060
```

**Features:**
- Top-down point cloud view
- GT direction arrows overlay
- Von Mises PDF visualization
- Category filtering

---

### 4. von_mises_interactive.py
**Interactive Von Mises Demo**

Live demonstration of von Mises mixture distributions.

```bash
python tools/von_mises_interactive.py --port 8055
# Open http://localhost:8055
```

**Features:**
- Adjustable number of peaks (1-4)
- Interactive kappa slider
- Real-time PDF updates

---

### 5. screenshot_viewer.py
**Point Cloud Screenshot Tool**

Generate publication-quality point cloud screenshots.

```bash
python tools/screenshot_viewer.py --port 8051
# Open http://localhost:8051
```

---

## CLI Tools

### 1. vis_fixed_4peak.py (NEW)
**Fixed 4-Peak Visualization (CLI)**

Generate static visualizations of model predictions.

```bash
# Basic usage
python tools/vis_fixed_4peak.py

# Specify checkpoint
python tools/vis_fixed_4peak.py --checkpoint checkpoints/fixed4peak_xxx/best.pth

# Save figures
python tools/vis_fixed_4peak.py --num_samples 20 --save

# Detailed category comparison
python tools/vis_fixed_4peak.py --detailed --save
```

**Options:**
- `--checkpoint`: Path to .pth file (default: latest best.pth)
- `--split`: Dataset split (train/val/test)
- `--num_samples`: Number of samples to visualize
- `--save`: Save figures to `visualization/outputs/`
- `--detailed`: Show category-by-category comparison grid

---

### 2. annotation_stats.py
**Annotation Statistics**

Display annotation progress and statistics.

```bash
python tools/annotation_stats.py
```

**Output:**
- Total annotated samples
- Per-category distribution
- Completion percentage

---

## Running from Project Root

All tools are designed to be run from the project root directory:

```bash
cd /home/pablo/ForwardNet-claude
python tools/<tool_name>.py [options]
```

---

## Port Summary

| Port | Tool |
|------|------|
| 8051 | screenshot_viewer.py |
| 8052 | annotate_symmetry_web.py |
| 8055 | von_mises_interactive.py |
| 8060 | verify_gt_web.py |
| 8070 | vis_checkpoint_web.py |

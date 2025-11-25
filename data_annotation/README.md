# 对称性数据标注工具与结果

本文件夹包含所有与对称性标注相关的工具和数据。

## 📁 文件结构

### 标注工具
- `annotate_symmetry_web_v2.py` - **主要工具**：Web版标注工具（推荐使用）
- `category_progress_viewer.py` - 类别进度查看器
- `annotate_by_category.py` - 按类别标注工具
- `annotate_symmetry_web.py` - Web版标注工具v1（旧版）
- `annotate_symmetry.py` - GUI版标注工具（需要X11）
- `annotate_priority.sh` - 优先级标注脚本
- `process_annotations.py` - 标注数据处理工具

### 标注数据
- `symmetry_annotations.json` - 标注数据（JSON格式）
- `symmetry_annotations.csv` - 标注数据（CSV格式，Excel可读）
- `symmetry_annotations.md` - 标注报告（Markdown格式）

### 文档
- `ANNOTATION_GUIDE.md` - 标注工具使用指南

## 🚀 快速开始

### 启动Web标注工具
```bash
cd /home/pablo/ForwardNet-claude/data_annotation
python annotate_symmetry_web_v2.py --port 8051
```

然后在浏览器中打开: http://localhost:8051

### 查看标注进度
```bash
python category_progress_viewer.py
```

## 📊 标注数据格式

每个样本的标注包含：
- `K`: 对称性类型（-1=没有正面, 0=完全对称, 1=单正面, 2=双正面, 4=四正面）
- `symmetry_name`: 对称性名称（中文）
- `front_direction`: 正面方向（-Z/+Z/-X/+X/-Y/+Y）
- `aligned`: 是否对齐（正面是否为-Z）
- `file`: 文件路径

## 🔒 注意事项

- 本文件夹**不包含数据集本身**（数据集太大不上传GitHub）
- 标注结果会自动保存到JSON文件
- 支持断点续标
- 多次运行不会丢失已有标注

详细使用说明请参考 `ANNOTATION_GUIDE.md`

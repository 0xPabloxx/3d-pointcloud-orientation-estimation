#!/bin/bash
# 按优先级标注工具 - 简化版
# 使用方法：./annotate_priority.sh airplane chair table

DATA_DIR="/home/pablo/ForwardNet-claude/data/full_mn40_normal_resampled_ply"
OUTPUT_FILE="/home/pablo/ForwardNet-claude/data/symmetry_annotations.json"

echo "=========================================="
echo "🏷️  按类别优先标注"
echo "=========================================="
echo ""

if [ $# -eq 0 ]; then
    echo "❌ 请指定要标注的类别"
    echo ""
    echo "用法："
    echo "  ./annotate_priority.sh airplane"
    echo "  ./annotate_priority.sh airplane chair table"
    echo ""
    echo "可用类别："
    ls -1 "$DATA_DIR" | head -20
    echo "  ..."
    exit 1
fi

# 显示选择的类别
echo "📋 已选择的类别："
for cat in "$@"; do
    cat_dir="$DATA_DIR/$cat"
    if [ -d "$cat_dir" ]; then
        count=$(ls -1 "$cat_dir"/*.ply 2>/dev/null | wc -l)
        echo "  ✅ $cat: $count 个样本"
    else
        echo "  ❌ $cat: 类别不存在"
    fi
done

echo ""
echo "🚀 启动标注工具..."
echo "=========================================="
echo ""

# 依次标注每个类别
for cat in "$@"; do
    cat_dir="$DATA_DIR/$cat"

    if [ ! -d "$cat_dir" ]; then
        echo "⚠️  跳过不存在的类别: $cat"
        continue
    fi

    echo ""
    echo "📂 当前标注类别: $cat"
    echo "=========================================="

    # 启动Web工具（指定类别目录）
    python annotate_symmetry_web_v2.py \
        --data_dir "$cat_dir" \
        --output "$OUTPUT_FILE" \
        --port 8051

    echo ""
    echo "✅ $cat 标注完成（或已关闭）"
    echo ""

    # 询问是否继续
    if [ "$cat" != "${@: -1}" ]; then  # 如果不是最后一个
        read -p "继续标注下一个类别？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "🛑 已停止"
            exit 0
        fi
    fi
done

echo ""
echo "=========================================="
echo "🎉 所有选定的类别标注完成！"
echo "=========================================="

#!/usr/bin/env python3
"""测试断点续标功能"""
import json
from pathlib import Path

# 创建一些测试标注
test_annotations = {
    "airplane/airplane_0001.ply": {
        "file": "airplane/airplane_0001.ply",
        "K": 1,
        "symmetry_name": "1个正面",
        "front_direction": "-Z",
        "aligned": True,
        "index": 0
    },
    "airplane/airplane_0002.ply": {
        "file": "airplane/airplane_0002.ply",
        "K": 1,
        "symmetry_name": "1个正面",
        "front_direction": "-X",
        "aligned": False,
        "index": 1
    }
}

# 保存到文件（模拟第一天的标注）
output_file = Path("data/test_resume_annotations.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_annotations, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存 {len(test_annotations)} 个标注到 {output_file}")
print("\n模拟场景：")
print("  1. 第一天标注了2个样本")
print("  2. 保存到 test_resume_annotations.json")
print("  3. 关机回家")
print("  4. 第二天重新启动工具时...")
print("\n重新加载标注：")

# 模拟重新加载（第二天启动工具）
with open(output_file, 'r', encoding='utf-8') as f:
    loaded_annotations = json.load(f)

print(f"  ✅ 成功加载 {len(loaded_annotations)} 个已有标注")
print("  ✅ 可以从第 {0} 个样本继续标注".format(len(loaded_annotations) + 1))

# 显示已标注的内容
print("\n已标注的样本：")
for file_path, ann in loaded_annotations.items():
    status = "✅" if ann['aligned'] else "⚠️"
    print(f"  {status} {file_path}: {ann['symmetry_name']}, {ann['front_direction']}")


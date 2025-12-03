#!/usr/bin/env python3
"""
将 front, side, top 三张截图按从左到右的顺序拼接成一张图

使用方法：
    python concat_views.py <模型名前缀>

示例：
    python concat_views.py airplane_airplane_0008
    # 会拼接 airplane_airplane_0008_front.png, airplane_airplane_0008_side_right.png, airplane_airplane_0008_top.png
    # 输出: airplane_airplane_0008_concat.png

作者: Claude
创建日期: 2025-11-27
"""

import sys
from pathlib import Path
from PIL import Image


def concat_three_views(prefix: str, output_dir: Path = None):
    """
    拼接 front, side_right, top 三张图

    Args:
        prefix: 文件名前缀，如 "airplane_airplane_0008"
        output_dir: 输出目录，默认与输入相同
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    # 查找三张图
    front_path = output_dir / f"{prefix}_front.png"
    side_path = output_dir / f"{prefix}_side_right.png"
    top_path = output_dir / f"{prefix}_top.png"

    # 检查文件是否存在
    missing = []
    for p in [front_path, side_path, top_path]:
        if not p.exists():
            missing.append(p.name)

    if missing:
        print(f"[Error] Missing files: {missing}")
        return None

    # 加载图片
    img_front = Image.open(front_path)
    img_side = Image.open(side_path)
    img_top = Image.open(top_path)

    # 获取尺寸（假设三张图大小相同）
    w, h = img_front.size

    # 创建拼接画布
    concat_img = Image.new('RGB', (w * 3, h))

    # 粘贴三张图
    concat_img.paste(img_front, (0, 0))
    concat_img.paste(img_side, (w, 0))
    concat_img.paste(img_top, (w * 2, 0))

    # 保存
    output_path = output_dir / f"{prefix}_concat.png"
    concat_img.save(output_path)

    print(f"[Done] Saved: {output_path}")
    print(f"       Size: {w * 3} x {h}")

    return output_path


def main():
    if len(sys.argv) < 2:
        # 自动查找可用的前缀
        script_dir = Path(__file__).parent
        front_files = list(script_dir.glob("*_front.png"))

        if front_files:
            print("Available prefixes:")
            for f in front_files:
                prefix = f.stem.replace("_front", "")
                print(f"  {prefix}")
            print(f"\nUsage: python {Path(__file__).name} <prefix>")
        else:
            print("No *_front.png files found in screenshots/")
        return

    prefix = sys.argv[1]
    concat_three_views(prefix)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import numpy as np

def convert_txt_to_ply(txt_file, ply_file):
    """
    读取以逗号分隔的TXT文件，每行6个数字（前三个为坐标，后三个为法向量），
    将数据转换后写入到PLY文件中（ASCII 格式）。
    """
    try:
        data = np.loadtxt(txt_file, delimiter=",")
    except Exception as e:
        print(f"[ERROR] 读取文件 {txt_file} 时出错: {e}")
        return

    # 处理单行数据时的shape问题
    if data.ndim == 1:
        if data.shape[0] != 6:
            print(f"[WARN] 文件 {txt_file} 每行应包含6个数字，跳过。")
            return
        data = data.reshape(1, -1)
    elif data.shape[1] != 6:
        print(f"[WARN] 文件 {txt_file} 每行应包含6个数字，跳过。")
        return

    points = data[:, 0:3]
    normals = data[:, 3:6]

    try:
        with open(ply_file, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("end_header\n")
            for pt, norm in zip(points, normals):
                f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    pt[0], pt[1], pt[2], norm[0], norm[1], norm[2]
                ))
        print(f"[OK] {txt_file} → {ply_file}")
    except Exception as e:
        print(f"[ERROR] 写入文件 {ply_file} 时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert.py <input_dir> <output_dir> [category]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # 第三个参数可选，用来指定只转换哪个分类
    category = sys.argv[3] if len(sys.argv) >= 4 else None

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for root, dirs, files in os.walk(input_dir):
        shape_name = os.path.basename(root)
        # 如果指定了 category，则只转换这一类
        if category and shape_name != category:
            continue

        # 输出目录中为该类别创建文件夹
        out_shape_dir = os.path.join(output_dir, shape_name)
        os.makedirs(out_shape_dir, exist_ok=True)

        for file in files:
            if not file.endswith(".txt"):
                continue
            in_path = os.path.join(root, file)
            out_name = os.path.splitext(file)[0] + ".ply"
            out_path = os.path.join(out_shape_dir, out_name)
            convert_txt_to_ply(in_path, out_path)
            count += 1

    print(f"转换完成，总共转换 {count} 个文件。")

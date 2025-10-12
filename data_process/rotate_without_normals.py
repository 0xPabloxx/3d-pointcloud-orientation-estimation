#!/usr/bin/env python3
import os
import numpy as np

def random_y_rotation_matrix():
    """
    仅在水平(X-Z)平面里随机旋转 —— 即绕全球竖直(Y)轴旋转。
    """
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, 0,  sin_t],
        [0,     1,  0   ],
        [-sin_t,0,  cos_t]
    ])


def read_ply(file_path):
    """
    读取 ASCII PLY 文件，返回 Nx3 的 numpy 数组。
    如果出错或数据不足，返回 None。
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[read_ply] 无法打开文件 {file_path}: {e}")
        return None

    # 解析 header 获取顶点数
    vertex_count = 0
    header_ended = False
    header_line_count = 0
    for i, line in enumerate(lines):
        header_line_count += 1
        if line.startswith("element vertex"):
            try:
                vertex_count = int(line.split()[-1])
            except ValueError:
                print(f"[read_ply] 解析顶点数失败: {line}")
                return None
        if line.strip() == "end_header":
            header_ended = True
            break

    if not header_ended or vertex_count <= 0:
        print(f"[read_ply] 非法 PLY 文件或顶点数为零: {file_path}")
        return None

    # 读取点数据
    data_lines = lines[header_line_count: header_line_count + vertex_count]
    verts = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            verts.append([x, y, z])
        except ValueError:
            continue

    if len(verts) == 0:
        print(f"[read_ply] 未读到有效顶点: {file_path}")
        return None

    return np.array(verts, dtype=np.float32)


def write_ply(vertices, out_file):
    """
    将 Nx3 数组写为 ASCII PLY 文件。
    """
    num_pts = vertices.shape[0]
    with open(out_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"[write_ply] 已保存: {out_file}")


def process_ply_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for shape_name in os.listdir(input_dir):
        shape_path = os.path.join(input_dir, shape_name)
        if not os.path.isdir(shape_path):
            continue

        out_shape_dir = os.path.join(output_dir, shape_name)
        os.makedirs(out_shape_dir, exist_ok=True)

        for ply_file in os.listdir(shape_path):
            if not ply_file.lower().endswith(".ply"):
                continue

            src_path = os.path.join(shape_path, ply_file)
            verts = read_ply(src_path)
            if verts is None:
                # 读失败或返回 None，跳过
                continue

            # 应用绕 Y 轴的随机旋转
            R = random_y_rotation_matrix()
            rotated = verts.dot(R.T)

            # 写入旋转后的 PLY
            dst_ply = os.path.join(out_shape_dir, ply_file)
            write_ply(rotated, dst_ply)

            # 计算并保存单位方向向量
            original_axes = [
                np.array([-1, 0, 0], dtype=np.float32),
                np.array([0, 1, 0], dtype=np.float32),
                np.array([0, 0, -1], dtype=np.float32),
            ]
            vectors = []
            for axis in original_axes:
                v = R.dot(axis)
                norm = np.linalg.norm(v)
                if norm > 1e-6:
                    v = v / norm
                vectors.append(v)

            txt_path = dst_ply.replace('.ply', '.txt')
            with open(txt_path, 'w') as vf:
                for v in vectors:
                    vf.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            print(f"[process] 已保存方向向量: {txt_path}")


if __name__ == "__main__":
    input_dir = "/home/pablo/ForwardNet/data/processed/mn40_normal_resampled_ply"
    output_dir = "/home/pablo/ForwardNet/data/full_mn40_normal_resampled_2d_rotated_ply"
    process_ply_files(input_dir, output_dir)

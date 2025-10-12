#!/usr/bin/env python3
import os
import numpy as np


def random_rotation_matrix():
    """
    生成一个随机旋转矩阵。
    分别随机采样绕 X、Y、Z 轴的旋转角，然后按 R = Rz * Ry * Rx 的顺序组合旋转矩阵。
    """
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    R = Rz.dot(Ry).dot(Rx)
    return R


def read_ply(file_path):
    """
    读取 ASCII 格式的 PLY 文件（假定文件中每行点数据至少有3个数字，前3个为坐标）
    返回一个 Nx3 的 numpy 数组保存所有顶点坐标。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    vertex_count = 0
    header_ended = False
    header_line_count = 0
    for i, line in enumerate(lines):
        header_line_count += 1
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_ended = True
            break

    if not header_ended:
        raise ValueError("不是有效的 PLY 文件：缺少 end_header.")

    data_lines = lines[header_line_count:]
    vertices = []
    for i in range(vertex_count):
        parts = data_lines[i].strip().split()
        if len(parts) < 3:
            continue
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        vertices.append([x, y, z])
    return np.array(vertices)


def write_ply(vertices, out_file):
    """
    将 Nx3 的顶点坐标以 ASCII PLY 格式写入文件。
    """
    num_points = vertices.shape[0]
    with open(out_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for pt in vertices:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))
    print(f"保存旋转后的点云: {out_file}")


def process_ply_files(input_dir, output_dir):
    """
    遍历 input_dir 下按 shape names 存放的子文件夹，
    对每个 PLY 文件生成随机旋转矩阵，应用旋转后将点云写入文件，
    同时计算原始方向向量 [-1,0,0],[0,1,0],[0,0,-1] 经过旋转后的新方向，并保存至 .txt 文件中（三行输出）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shape_name in os.listdir(input_dir):
        shape_path = os.path.join(input_dir, shape_name)
        if not os.path.isdir(shape_path):
            continue

        output_shape_dir = os.path.join(output_dir, shape_name)
        if not os.path.exists(output_shape_dir):
            os.makedirs(output_shape_dir)

        ply_files = [f for f in os.listdir(shape_path) if f.endswith(".ply")]
        for ply_file in ply_files:
            input_ply_file = os.path.join(shape_path, ply_file)
            try:
                vertices = read_ply(input_ply_file)
            except Exception as e:
                print(f"读取文件 {input_ply_file} 出错: {e}")
                continue

            # 生成随机旋转矩阵，并对点云数据进行旋转
            R = random_rotation_matrix()
            rotated_vertices = vertices.dot(R.T)
            output_ply_file = os.path.join(output_shape_dir, ply_file)
            write_ply(rotated_vertices, output_ply_file)

            # 定义原始方向向量
            original_axes = [
                np.array([-1, 0, 0]),  # X 轴负方向
                np.array([0, 1, 0]),   # Y 轴正方向 (upright)
                np.array([0, 0, -1])   # Z 轴负方向（forward）
            ]
            # 计算旋转后方向向量
            rotated_axes = [R.dot(axis) for axis in original_axes]

            # 保存新方向向量到 .txt 文件，三行输出
            vector_file = output_ply_file.replace('.ply', '.txt')
            with open(vector_file, 'w') as vf:
                for vec in rotated_axes:
                    vf.write("{:.6f} {:.6f} {:.6f}\n".format(vec[0], vec[1], vec[2]))
            print(f"保存方向向量: {vector_file}")


if __name__ == "__main__":
    input_dir = "/home/pablo/ForwardNet/data/processed/mn40_normal_resampled_ply"
    output_dir = "/home/pablo/ForwardNet/data/full_mn40_normal_resampled_2d_rotated_ply"
    process_ply_files(input_dir, output_dir)

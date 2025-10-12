import numpy as np
import os


def load_ply(file_path):
    """
    读取 .ply 文件，跳过 header 部分，返回点云数据（假设每个顶点只有 x, y, z 三个属性）。
    如果需要处理更多属性，需根据文件格式做相应扩展。
    """
    with open(file_path, 'r') as f:
        # 读取 header 行，并统计需要跳过的行数
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        # 读取 header 之后的所有数值
        try:
            data = np.loadtxt(f)
        except Exception as e:
            raise RuntimeError(f"读取点云数据时出错: {e}")
    return data, header_lines


def save_ply(file_path, points):
    """
    将点云数据 points 保存为 .ply 文件，生成简单的 ASCII 格式 header。
    """
    n_points = points.shape[0]
    header = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {n_points}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "end_header\n"
    ]
    with open(file_path, 'w') as f:
        f.writelines(header)
        np.savetxt(f, points, fmt='%.6f')


def random_rotation_matrix():
    """
    生成一个随机旋转矩阵。
    分别随机采样绕 X、Y、Z 轴的旋转角，然后按 R = Rz * Ry * Rx 的顺序组合旋转矩阵。
    """
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    # 绕 X 轴
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    # 绕 Y 轴
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    # 绕 Z 轴
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵（注意旋转顺序）
    R = Rz.dot(Ry).dot(Rx)
    return R


def process_point_cloud(points):
    """
    对点云数据应用随机旋转，保持原点不变。
    """
    R = random_rotation_matrix()
    rotated = (R @ points.T).T
    return rotated


def main():
    # 输入路径，根据实际情况调整路径（例如：/home/pablo/ForwardNet/data/ModelNet40_output）
    input_dir = '/home/pablo/ForwardNet/data/modelnet40_normal_resampled_ply'
    # 输出路径，这里将处理后的模型保存到 /home/pablo/ForwardNet/data 下，按照 shape name 分类
    output_dir = '/home/pablo/ForwardNet/data/modelnet40_normal_resampled_3d_rotate'

    # 遍历输入目录下各个分类文件夹
    for shape in os.listdir(input_dir):
        shape_input_path = os.path.join(input_dir, shape)
        shape_output_path = os.path.join(output_dir, shape)

        if not os.path.isdir(shape_input_path):
            continue

        os.makedirs(shape_output_path, exist_ok=True)

        for filename in os.listdir(shape_input_path):
            file_path = os.path.join(shape_input_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == '.ply':
                    # 使用自定义的 load_ply 函数读取 .ply 文件
                    points, _ = load_ply(file_path)
                else:
                    points = np.loadtxt(file_path)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue

            # 对点云应用随机旋转
            rotated_points = process_point_cloud(points)

            output_file_path = os.path.join(shape_output_path, filename)
            try:
                if ext == '.ply':
                    # 保存为 .ply 文件
                    save_ply(output_file_path, rotated_points)
                else:
                    np.savetxt(output_file_path, rotated_points, fmt='%.6f')
                print(f"已处理 {file_path} -> {output_file_path}")
            except Exception as e:
                print(f"保存文件 {output_file_path} 时出错: {e}")


if __name__ == '__main__':
    main()

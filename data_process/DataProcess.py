import numpy as np

# 修改文件名为你自己的 TXT 文件路径
txt_file = 'model.txt'
ply_file = 'model.ply'

# 若数据用逗号分隔，请设置 delimiter=','；否则留空默认为空格分隔
data = np.loadtxt(txt_file, delimiter=',')

# 判断数据列数，如果有 6 列则认为包含法向量，否则只有坐标
has_normals = data.shape[1] >= 6
points = data[:, 0:3]
normals = data[:, 3:6] if has_normals else None

# 写入 PLY 文件
with open(ply_file, 'w') as f:
    # 写入 PLY 头部信息
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % points.shape[0])
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    if has_normals:
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
    f.write("end_header\n")

    # 将每个点的信息写入
    for i in range(points.shape[0]):
        if has_normals:
            f.write("%f %f %f %f %f %f\n" % (points[i, 0], points[i, 1], points[i, 2],
                                             normals[i, 0], normals[i, 1], normals[i, 2]))
        else:
            f.write("%f %f %f\n" % (points[i, 0], points[i, 1], points[i, 2]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本说明：
    1. 读取当前目录（或指定目录）下的 shape_names.txt，
       生成类别名称列表（例如 ["airplane", "bathtub", ... ,"chair", ...]）。
    2. 遍历所有 .h5 文件（例如 ply_data_train0.h5, ply_data_test0.h5 等）。
    3. 如果对应 h5 文件有 id2file 的 JSON 映射文件，则加载该映射，
       用作生成保存时的文件名；否则采用自动命名规则。
    4. 对每个样本：
         - 从 h5 文件中读取点云（假设数据键为 "data"，标签键为 "label"）
         - 根据标签（整数）对应到类别名称
         - 检查输出目录（output_base/类别名称）是否存在，不存在则自动创建
         - 将点云数据以 ASCII 的 .ply 格式写入文件

用法：
    修改 dataset_dir（数据所在路径）和 output_base（保存 .ply 文件的根目录），
    直接运行脚本即可。
"""

import os
import re
import json
import h5py
import numpy as np


def write_ply(points, out_filename):
    """
    将点云 (N, 3) 数组以 ASCII 格式写入 .ply 文件
    """
    num_points = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).format(num_points)

    with open(out_filename, 'w') as f:
        f.write(header)
        for pt in points:
            # 格式化到小数点后6位
            f.write("{:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))


def process_h5_file(h5_filename, json_filename, shape_names, output_base):
    """
    处理单个 h5 文件：
        - 读取 h5 中的 "data" 和 "label" 数据
        - 如果存在 json 映射文件，则加载 sample_id 与原始文件名的对应关系
        - 遍历每个样本，确定样本所属类别，并调用 write_ply 保存为 .ply 文件
    """
    print("正在处理文件:", h5_filename)
    with h5py.File(h5_filename, 'r') as f:
        data = f["data"][:]  # 假设 shape 为 (num_samples, 2048, 3)
        labels = f["label"][:]  # 假设 shape 为 (num_samples,) 或 (num_samples, 1)
        if labels.ndim > 1:
            labels = labels[:, 0]

    # 读取 id2file 映射（如果存在）
    id2file = {}
    if json_filename is not None and os.path.exists(json_filename):
        try:
            with open(json_filename, 'r') as jf:
                id2file = json.load(jf)
        except Exception as e:
            print("加载 JSON 映射文件出错:", json_filename, e)

    num_samples = data.shape[0]
    for i in range(num_samples):
        sample_points = data[i]  # (2048, 3)
        label = int(labels[i])
        # 根据 label 获取类别名称（确保 shape_names.txt 中每行一个类别）
        try:
            category = shape_names[label].strip()
        except IndexError:
            print("样本 {} 的标签 {} 超出范围".format(i, label))
            continue

        # 创建对应类别目录（例如 output_base/chair）
        out_dir = os.path.join(output_base, category)
        os.makedirs(out_dir, exist_ok=True)

        # 尝试使用 JSON 中记录的文件名，否则构造一个新文件名
        filename = None
        if id2file:
            if isinstance(id2file, dict):
                filename = id2file.get(str(i), None)
            elif isinstance(id2file, list):
                if i < len(id2file):
                    filename = id2file[i]
                else:
                    filename = None

            # 去掉文件名中可能存在的子目录信息
            if filename is not None:
                filename = os.path.basename(filename)

        if filename is None:
            # 利用 h5 文件名和样本索引构造新的文件名
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            filename = f"{base}_{i}.ply"

        output_path = os.path.join(out_dir, filename)
        write_ply(sample_points, output_path)
    print("完成处理文件:", h5_filename)


def main():
    # 数据集所在目录；请根据实际情况修改
    dataset_dir = "/home/pablo/ForwardNet/data/modelnet40_ply_hdf5_2048"
    # 输出 .ply 文件存放根目录，可自行设置路径
    output_base = "/home/pablo/ForwardNet/data/ModelNet40_output"
    os.makedirs(output_base, exist_ok=True)

    # 读取类别名称列表（假设 shape_names.txt 每行一个类别名称）
    shape_names_file = os.path.join(dataset_dir, "shape_names.txt")
    if not os.path.exists(shape_names_file):
        print("找不到 shape_names.txt 文件，请确认数据集目录是否正确。")
        return
    with open(shape_names_file, 'r') as f:
        shape_names = [line.strip() for line in f if line.strip()]

    # 获取所有 .h5 文件（既包含训练文件也包含测试文件）
    h5_files = [f for f in os.listdir(dataset_dir) if f.endswith(".h5")]
    if not h5_files:
        print("在目录 {} 中没有找到 .h5 文件".format(dataset_dir))
        return

    for h5_file in h5_files:
        h5_path = os.path.join(dataset_dir, h5_file)
        # 根据文件名查找对应的 JSON 文件（例如 ply_data_train0.h5 对应 ply_data_train_0_id2file.json）
        base_name = os.path.splitext(h5_file)[0]
        json_path = None
        m = re.match(r"(.*?)(\d+)$", base_name)
        if m:
            prefix = m.group(1)
            number = m.group(2)
            candidate = prefix + "_" + number + "_id2file.json"
            candidate_path = os.path.join(dataset_dir, candidate)
            if os.path.exists(candidate_path):
                json_path = candidate_path
        if json_path is None:
            candidate = base_name + "_id2file.json"
            candidate_path = os.path.join(dataset_dir, candidate)
            if os.path.exists(candidate_path):
                json_path = candidate_path

        print("处理文件:", h5_file, "对应映射文件:", json_path)
        process_h5_file(h5_path, json_path, shape_names, output_base)


if __name__ == '__main__':
    main()

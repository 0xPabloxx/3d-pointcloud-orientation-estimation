#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-peak von Mises mixture ground truth
支持 K_DICT 中指定 K = 0 表示完全对称的类别，转为单峰 kappa=0 的输出
"""

import math
import pathlib

# —— 配置区 ——  
# K_DICT: 指定哪些类别要生成，以及它们的 K 值（0 表示完全对称类别）
K_DICT = {
    "cone": 0,
    "bowl": 0,
    "chair": 1,
    "bottle": 0,   
    "plant": 0,  
    "car":1,
    "sofa":1,
    "toilet":1,
    "door"  :2,
    "curtain":2,
    "bathtub":4,
    "glass_box":4
}
GLOBAL_KAPPA = 8.0  # 默认集中度参数，用于非对称类别（K >= 1）

# 源和目标路径
SRC_ROOT = pathlib.Path("/home/pablo/ForwardNet/data/full_mn40_normal_resampled_2d_rotated_ply")
DST_ROOT = pathlib.Path("/home/pablo/ForwardNet/data/MN40_multi_peak_vM_gt")
DST_ROOT.mkdir(parents=True, exist_ok=True)


def parse_vectors(txt_path):
    xs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            xs.append([float(t) for t in parts])
    if len(xs) != 3 or len(xs[0]) != 3:
        raise ValueError(f"格式错误: {txt_path}")
    side, up, front = xs
    return side, up, front


def vec_to_angle(v):
    fx, _, fz = v
    r = math.hypot(fx, fz)
    if r < 1e-8:
        fx, fz = 0.0, -1.0
        r = 1.0
    fx /= r
    fz /= r
    theta = math.atan2(fx, -fz)
    return theta


def opposite_vec(v):
    return [-v[0], -v[1], -v[2]]


def gen_mixture_peaks(side, up, front, K):
    """
    根据 K 生成峰向量列表。
    如果 K >= 1：按前 / 反前 / 侧 / 反侧顺序选前 K 个峰
    """
    candidates = [front, opposite_vec(front), side, opposite_vec(side)]
    return candidates[:K]


def gen_gt_for_model(model_name, K_specified, global_kappa):
    """
    为某一类别生成 GT。
    K_specified: 从 K_DICT 中取出的值，可能是 0 或 >=1。
    如果 K_specified == 0：生成单峰，kappa = 0
    否则生成 K = K_specified 个峰，kappa = global_kappa
    """
    src_dir = SRC_ROOT / model_name
    if not src_dir.is_dir():
        print(f"[WARN] 源目录不存在：{src_dir}，跳过该类别")
        return

    dst_dir = DST_ROOT / model_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0
    for txt in sorted(src_dir.glob("*.txt")):
        # 跳过已经生成过的 ground truth 文件（假设名称包含 "_multi_peakK"）
        if "_multi_peakK" in txt.stem:
            continue

        try:
            side, up, front = parse_vectors(txt)
        except Exception as e:
            print(f"[WARN] 无法解析 {txt}：{e}")
            continue

        # 根据指定 K 生成峰与参数
        if K_specified == 0:
            # 对称类别，生成单峰，kappa = 0
            peaks = [front]
            kappa = 0.0
        else:
            peaks = gen_mixture_peaks(side, up, front, K_specified)
            kappa = global_kappa

        # 实际峰的数量（用于在输出写 K 行）
        K_actual = len(peaks)
        # 防止分母除零
        if K_actual <= 0:
            print(f"[ERROR] {model_name} 在 {txt} 得到 0 个峰，跳过")
            continue

        # 权重平均
        weight = 1.0 / K_actual

        # 输出文件名
        out = dst_dir / f"{txt.stem}_multi_peak_vM_gt.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("# von Mises mixture ground truth\n")
            # 第二行写 “K x”
            f.write(f"K {K_actual}\n")
            # 表头行
            f.write("mu(rad)\tkappa\tweight\n")
            for p in peaks:
                mu = vec_to_angle(p)
                f.write(f"{mu:.8f}\t{kappa:.6f}\t{weight:.6f}\n")

        cnt += 1

    print(f"[{model_name}] 完成，写入 {cnt} 个 GT 文件 (指定 K={K_specified})")


def main():
    models = list(K_DICT.keys())
    print("准备处理以下类别（基于 K_DICT）：", models)
    for model_name in models:
        K_spec = K_DICT[model_name]
        gen_gt_for_model(model_name, K_spec, GLOBAL_KAPPA)
    print("所有指定类别处理完毕。")


if __name__ == "__main__":
    main()

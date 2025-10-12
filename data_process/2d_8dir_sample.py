#!/usr/bin/env python3
from pathlib import Path                   # ⬅️ 面向对象文件路径:contentReference[oaicite:5]{index=5}
import numpy as np

BASE = Path('/home/pablo/ForwardNet/data/chair_toilet_sofa_plant_bowl_bottle')
UNIFORM = {'bottle', 'bowl', 'plant'}

DIRS_8 = np.array([
    [ 0.0, 0.0,-1.0],
    [ 0.70710678,0.0,-0.70710678],
    [ 1.0, 0.0, 0.0],
    [ 0.70710678,0.0, 0.70710678],
    [ 0.0, 0.0, 1.0],
    [-0.70710678,0.0, 0.70710678],
    [-1.0, 0.0, 0.0],
    [-0.70710678,0.0,-0.70710678],
])

for cls_dir in BASE.iterdir():          # 遍历 bottle/…/chair 文件夹:contentReference[oaicite:6]{index=6}
    if not cls_dir.is_dir(): continue
    label = cls_dir.name
    pattern = f'{label}_*.txt'          # e.g. chair_0001.txt
    for txt in cls_dir.glob(pattern):
        if txt.name.endswith('_8dir.txt'):
            continue                    # 已处理跳过
        # 读取三行向量
        with txt.open() as f:
            vecs = [line.strip() for line in f]
        if label in UNIFORM:            # 均匀分布类别
            probs = np.full(8, 0.125, dtype=np.float32)
        else:
            v = np.fromstring(vecs[2], sep=' ')  # 取第三行
            v = v / (np.linalg.norm(v) + 1e-8)   # 单位化
            sims = DIRS_8 @ v                    # dot product
            sims = np.clip(sims, 0.0, None)      # ReLU
            if sims.sum() == 0:
                probs = np.full(8, 0.125, dtype=np.float32)
            else:
                probs = sims / sims.sum()
        out = txt.with_name(txt.stem + '_8dir.txt')
        np.savetxt(out, probs[None], fmt='%.6f') # 一行 8 列
        print('✓', out)

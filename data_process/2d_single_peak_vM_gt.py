# gen_single_peak_vm_gt.py
import math, pathlib

ROOT = pathlib.Path("/home/pablo/ForwardNet/data/chair_toilet_sofa_plant_bowl_bottle")

CLEAR = {"chair","sofa","toilet"}     # 有明确前向
SYMM  = {"bottle","plant","bowl"}     # 对称/无前向
KAPPA_DEFAULT = 8.0                   # 可按类单独映射

def parse_forward_vec(txt_path):
    """
    文件三行向量：
    第1行：侧向(≈右/左)
    第2行：竖直 +y
    第3行：前向（旋转后的 -z）
    返回 fx, fz
    """
    xs = []
    file_name = ""
    with open(txt_path, "r", encoding="utf-8") as f:
        file_name = f.name
        for line in f:
            line=line.strip()
            if not line: continue
            xs.append([float(t) for t in line.split()])
    if len(xs) < 3 or len(xs[2]) < 3:
        raise ValueError(f"格式错误: {txt_path}")
    fx, fy, fz = xs[2]        # 取第3行为前向
    print(f"Processing {file_name}: forward vector ({fx}, {fy}, {fz})")
    # 投影到水平面（x–z），防止数值误差
    r = math.hypot(fx, fz)
    if r < 1e-8:
        # 极端异常：水平分量几乎为零，退化处理（把前向设为 -z）
        fx, fz = 0.0, -1.0
    else:
        fx, fz = fx/r, fz/r
    return fx, fz

def vec_to_mu(fx, fz):
    theta_std = math.atan2(fx, -fz)      
    return mu

def decide_kappa(cls_name):
    if cls_name in SYMM:
        return 0.0
    return KAPPA_DEFAULT

def main():
    cnt=0
    for cls_dir in sorted([d for d in ROOT.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        for txt in sorted(cls_dir.glob("*.txt")):
            # 跳过我们要生成的 *_single_peak_vM_gt.txt，避免重复处理
            if txt.name.endswith("_single_peak_vM_gt.txt"):
                continue
            try:
                fx, fz = parse_forward_vec(txt)
                mu = vec_to_mu(fx, fz)
                kappa = decide_kappa(cls)
                out = txt.with_name(txt.stem + "_single_peak_vM_gt.txt")
                with open(out, "w", encoding="utf-8") as f:
                    f.write("# mu(rad)\tkappa\n")
                    f.write(f"{mu:.8f}\t{kappa:.6f}\n")
                cnt += 1
                # 也可打印：print(f"Wrote {out}")
            except Exception as e:
                print(f"[WARN] 跳过 {txt}: {e}")
    print(f"生成完成，共写出 {cnt} 个 GT 文件。")

if __name__ == "__main__":
    main()

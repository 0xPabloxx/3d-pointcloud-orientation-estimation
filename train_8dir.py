import os, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from dataloader import PointCloudDataset
from models.pointnet_pp_8dir import PointNetPP8Dir, DIRS_8      # 不变

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- 配置 -----------
DATA_ROOT    = "/home/pablo/ForwardNet/data/full_mn40_normal_resampled_2d_rotated_ply"
RESULTS_ROOT = "/home/pablo/ForwardNet/results"
OUTPUT_ROOT  = "/home/pablo/ForwardNet/results/output"

LABEL_RANGE  = (9, 9)            # 只训练 chair
NUM_POINTS   = 10000
BATCH_SIZE   = 16
NUM_EPOCHS   = 200
LR           = 1e-3
SEED         = 42

# ---- 8-方向常量 ----
DIRS_8_T = DIRS_8.to(device)      # (8,3)

def target_probs(fwd):            # (B,3) → (B,8)
    fwd = F.normalize(fwd, dim=1)
    sims = torch.matmul(fwd, DIRS_8_T.t()).clamp(min=0)
    return sims / sims.sum(dim=1, keepdim=True).clamp(min=1e-8)

# ---- 画 loss ----
def plot_losses(tr, va, path):
    plt.figure(); plt.plot(tr, label='Train'); plt.plot(va, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(True); plt.legend()
    plt.savefig(path); plt.close()

# ---- 单标签训练 ----
def train_single(label, idx, total):
    print(f"\n=== [{idx}/{total}] {label} ===")
    ddir  = os.path.join(DATA_ROOT, label)
    files = sorted(f for f in os.listdir(ddir) if f.endswith('.ply')); random.shuffle(files)
    n_tr  = int(0.7 * len(files)); n_va = int(0.15 * len(files))

    tr_ds = PointCloudDataset(ddir, files[:n_tr],          num_points=NUM_POINTS)
    va_ds = PointCloudDataset(ddir, files[n_tr:n_tr+n_va], num_points=NUM_POINTS)
    te_ds = PointCloudDataset(ddir, files[n_tr+n_va:],     num_points=NUM_POINTS)

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=4)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=4)

    model = PointNetPP8Dir().to(device)
    opt   = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    tr_losses, va_losses = [], []; best = float('inf'); best_state = None
    for ep in range(1, NUM_EPOCHS+1):

        # ---- train ----
        model.train(); s = 0.0
        for pts, tgt in tr_ld:
            pts = pts.to(device)
            fwd = tgt[:, 2, :].to(device)            # 第 3 行是 forward
            y   = target_probs(fwd)

            opt.zero_grad()
            pred = F.softmax(model(pts), dim=1)
            loss = criterion(pred, y)
            loss.backward(); opt.step()
            s += loss.item() * pts.size(0)
        tr_losses.append(s / len(tr_ld.dataset))

        # ---- val ----
        model.eval(); s = 0.0
        with torch.no_grad():
            for pts, tgt in va_ld:
                pts = pts.to(device)
                fwd = tgt[:, 2, :].to(device)
                y   = target_probs(fwd)
                pred = F.softmax(model(pts), dim=1)
                s += criterion(pred, y).item() * pts.size(0)
        va_losses.append(s / len(va_ld.dataset))

        print(f"Ep{ep:3d}  Train {tr_losses[-1]:.4f}  Val {va_losses[-1]:.4f}")
        if va_losses[-1] < best:
            best = va_losses[-1]; best_state = model.state_dict()

    # ---- 保存曲线与模型 ----
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    torch.save(best_state, os.path.join(RESULTS_ROOT, f"{label}_best.pth"))
    plot_losses(tr_losses, va_losses, os.path.join(RESULTS_ROOT, f"{label}_loss.png"))

    # ---- 计算平均原始 / 预测概率 (测试集) ----
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=4)
    model.load_state_dict(best_state); model.eval()
    orig_sum = torch.zeros(8, device=device)
    pred_sum = torch.zeros(8, device=device)
    total    = 0
    with torch.no_grad():
        for pts, tgt in te_ld:
            pts  = pts.to(device)
            fwd  = tgt[:, 2, :].to(device)
            orig = target_probs(fwd)                 # (B,8)
            pred = F.softmax(model(pts), dim=1)      # (B,8)
            orig_sum += orig.sum(0)
            pred_sum += pred.sum(0)
            total    += pts.size(0)
    mean_orig = orig_sum / total                     # (8,)
    mean_pred = pred_sum / total

    # ---- 写 summary.txt ----
    with open(os.path.join(RESULTS_ROOT, "summary.txt"), "a") as f:
        f.write(f"{label}\t{best:.6f}\n")
        f.write(" ".join(f"{p:.4f}" for p in mean_orig.cpu().tolist()) + "\n")
        f.write(" ".join(f"{p:.4f}" for p in mean_pred.cpu().tolist()) + "\n")

    # ---- 可视化 10 样本 ----
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    te_files = files[n_tr+n_va:]
    for i, fname in enumerate(random.sample(te_files, min(10, len(te_files))), 1):
        verts = np.loadtxt(open(os.path.join(ddir, fname))).astype(np.float32)
        pts   = torch.tensor(verts).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(pts), dim=1)[0]
        vz = F.normalize((probs.unsqueeze(1) * DIRS_8_T).sum(0), dim=0)
        vy = torch.tensor([0, 1, 0.], device=device)
        vx = F.normalize(torch.cross(vy, vz), dim=0)
        axes = [vx.cpu().numpy(), vy.cpu().numpy(), vz.cpu().numpy()]

        out_ply = os.path.join(OUTPUT_ROOT, f"{label}_{i}.ply")
        num_pts, tot = verts.shape[0], verts.shape[0] + 4
        with open(out_ply, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {tot}\nproperty float x\nproperty float y\nproperty float z\n")
            f.write("element edge 3\nproperty int vertex1\nproperty int vertex2\nend_header\n")
            f.write("0 0 0\n")
            for v in axes: f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for p in verts: f.write(f"{p[0]} {p[1]} {p[2]}\n")
            f.write("0 1\n0 2\n0 3\n")
        print("Saved", out_ply)

# ---- 主程序 ----
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    labels = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith('.'))
    lo, hi = LABEL_RANGE
    labels = labels[lo-1:hi]
    open(os.path.join(RESULTS_ROOT, "summary.txt"), "w").close()
    for idx, lbl in enumerate(labels, 1):
        train_single(lbl, idx, len(labels))

if __name__ == '__main__':
    main()

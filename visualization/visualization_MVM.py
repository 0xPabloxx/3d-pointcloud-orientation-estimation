import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0
import math

def read_mvm_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    K = int(lines[0].split()[1])
    data = np.loadtxt(lines[2:], dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mu, kappa, w = data[:, 0], data[:, 1], data[:, 2]
    return K, mu, kappa, w

def von_mises(theta, mu, kappa):
    if np.isclose(kappa, 0.0):
        return np.ones_like(theta) / (2 * np.pi)
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))

def mixture_von_mises(theta, mu_list, kappa_list, weight_list):
    p = np.zeros_like(theta)
    for mu, kappa, w in zip(mu_list, kappa_list, weight_list):
        p += w * von_mises(theta, mu, kappa)
    p /= (np.trapz(p, theta) + 1e-8)
    return p

def plot_and_save(mu, kappa, w, save_path, theta_counts=720):
    theta = np.linspace(-math.pi, math.pi, theta_counts)
    p = mixture_von_mises(theta, mu, kappa, w)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta, p, lw=1.5, color='tab:blue')
    ax.fill_between(theta, 0, p, alpha=0.3, color='tab:blue')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title("", va='bottom')
    # 关闭图例等开销项（如果不必要）
    # ax.legend().remove()

    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def batch_plot(label_name, 
               gt_root="/home/pablo/ForwardNet/data/MN40_multi_peak_vM_gt", 
               out_root="/home/pablo/ForwardNet/visualization"):
    """
    label_name: 比如 'bottle'、'chair' 等子文件夹名
    """
    label_dir = os.path.join(gt_root, label_name)
    if not os.path.isdir(label_dir):
        print("Error: label dir not exists:", label_dir)
        return

    out_label_dir = os.path.join(out_root, label_name)
    os.makedirs(out_label_dir, exist_ok=True)

    # 假设所有 GT 文件名形如 xxx_multi_peak_vM_gt.txt
    pattern = os.path.join(label_dir, "*_multi_peak_vM_gt.txt")
    file_list = glob.glob(pattern)
    print(f"Found {len(file_list)} gt files under label {label_name}.")

    for idx, gt_path in enumerate(file_list):
        try:
            K, mu, kappa, w = read_mvm_gt(gt_path)
            fname = os.path.basename(gt_path)
            save_path = os.path.join(out_label_dir, fname.replace(".txt", ".png"))
            plot_and_save(mu, kappa, w, save_path)
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{len(file_list)}")
        except Exception as e:
            print("Error processing", gt_path, ":", e)

if __name__ == "__main__":
    label = "glass_box"  # 例如
    batch_plot(label)

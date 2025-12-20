#!/usr/bin/env python3
"""
DR Series: 基础方向投影 + Softmax
=================================

DR vs D 的根本区别：

  D系列 (槽分类):
    网络输出: logits (任意值，无物理意义)
    处理: softmax(logits) → 概率分布
    含义: "方向落在哪个bin的概率"

  DR系列 (基础方向投影):
    网络输出: 投影值回归 (tanh约束到[-1,1]，有物理意义)
    处理: softmax(投影值) → 概率分布
    含义: "与各基础方向的相似度分布"

GT对比 (方向270°):
  DR (cos投影 → softmax): [0.10, 0.05, 0.04, 0.05, 0.10, 0.20, 0.27, 0.20]
  D  (τ=5):               [0.01, 0.00, 0.00, 0.00, 0.01, 0.16, 0.68, 0.16]
"""

import subprocess
import sys
import os
from datetime import datetime

# Change to project directory
os.chdir("/home/pablo/ForwardNet-claude")

# Common parameters
EPOCHS = 50
BATCH_SIZE = 32
CATEGORIES = "1_front,4_fronts,no_front"
WANDB_PROJECT = "ForwardNet-LossAblation"

# Experiments configuration
experiments = [
    {
        "name": "DR_8a",
        "desc": "8 bins, KL loss",
        "num_bins": 8,
        "d_loss_type": "kl",
    },
    {
        "name": "DR_8b",
        "desc": "8 bins, CE loss",
        "num_bins": 8,
        "d_loss_type": "ce",
    },
    {
        "name": "DR_16a",
        "desc": "16 bins, KL loss",
        "num_bins": 16,
        "d_loss_type": "kl",
    },
]


def run_experiment(exp):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"[{exp['name']}] {exp['desc']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    cmd = [
        "python", "train_direction.py",
        "--mode", "discrete",
        "--exp_name", exp["name"],
        "--categories", CATEGORIES,
        "--num_bins", str(exp["num_bins"]),
        "--gt_mode", "dr",
        "--d_loss_type", exp["d_loss_type"],
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--wandb",
        "--wandb_project", WANDB_PROJECT,
    ]

    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] {exp['name']} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n[{exp['name']}] Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    print("=" * 60)
    print("  DR Series: Projection + Softmax")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nExperiments to run: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['desc']}")

    # Run all experiments
    for exp in experiments:
        run_experiment(exp)

    # Summary
    print("\n" + "=" * 60)
    print("  DR Series Completed!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nResults:")

    # Show checkpoints
    os.system("ls -lt checkpoints/ | grep 'DR_' | head -5")


if __name__ == "__main__":
    main()

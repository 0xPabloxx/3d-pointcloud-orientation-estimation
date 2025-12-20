#!/usr/bin/env python3
"""
D Series Experiment Runner - Discrete Direction Prediction
==========================================================

This script runs the D-series experiments for discrete direction prediction.
Each experiment tests different configurations of bins, temperature, and loss types.

Usage:
    python scripts/run_d_series.py              # Run all experiments
    python scripts/run_d_series.py --exp D_8a   # Run specific experiment
    python scripts/run_d_series.py --dry-run    # Show configs without running
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Change to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# =============================================================================
# Experiment Configurations
# =============================================================================

COMMON_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
    "categories": "1_front,4_fronts,no_front",
    "wandb_project": "ForwardNet-LossAblation",
}

EXPERIMENTS = {
    # 8 bins experiments
    "D_8a": {
        "desc": "8 bins, CE, τ=5 (baseline)",
        "num_bins": 8,
        "temperature": 5.0,
        "d_loss_type": "ce",
    },
    "D_8b": {
        "desc": "8 bins, CE, τ=3 (softer labels)",
        "num_bins": 8,
        "temperature": 3.0,
        "d_loss_type": "ce",
    },
    "D_8c": {
        "desc": "8 bins, CE, τ=10 (sharper labels)",
        "num_bins": 8,
        "temperature": 10.0,
        "d_loss_type": "ce",
    },
    "D_8d": {
        "desc": "8 bins, KL Divergence",
        "num_bins": 8,
        "temperature": 5.0,
        "d_loss_type": "kl",
    },
    # 16 bins experiments
    "D_16a": {
        "desc": "16 bins, CE, τ=5",
        "num_bins": 16,
        "temperature": 5.0,
        "d_loss_type": "ce",
    },
    "D_16b": {
        "desc": "16 bins, CE, τ=8",
        "num_bins": 16,
        "temperature": 8.0,
        "d_loss_type": "ce",
    },
}


def build_command(exp_name: str, config: dict) -> list:
    """Build the training command for an experiment."""
    cmd = [
        sys.executable,
        "train_direction.py",
        "--mode", "discrete",
        "--exp_name", exp_name,
        "--categories", COMMON_CONFIG["categories"],
        "--num_bins", str(config["num_bins"]),
        "--gt_mode", "projection",
        "--temperature", str(config["temperature"]),
        "--epochs", str(COMMON_CONFIG["epochs"]),
        "--batch_size", str(COMMON_CONFIG["batch_size"]),
        "--lr", str(COMMON_CONFIG["lr"]),
        "--d_loss_type", config["d_loss_type"],
        "--wandb",
        "--wandb_project", COMMON_CONFIG["wandb_project"],
    ]
    return cmd


def run_experiment(exp_name: str, config: dict, dry_run: bool = False) -> bool:
    """Run a single experiment."""
    print("\n" + "=" * 60)
    print(f"[{exp_name}] {config['desc']}")
    print(f"  bins={config['num_bins']}, τ={config['temperature']}, loss={config['d_loss_type']}")
    print("=" * 60)

    cmd = build_command(exp_name, config)

    if dry_run:
        print(f"Command: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=PROJECT_ROOT,
        )
        print(f"\n[{exp_name}] Completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[{exp_name}] FAILED with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[{exp_name}] Interrupted by user")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run D-series experiments")
    parser.add_argument("--exp", type=str, help="Run specific experiment (e.g., D_8a)")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")
    parser.add_argument("--start-from", type=str, help="Start from specific experiment")
    args = parser.parse_args()

    print("=" * 60)
    print("  D Series: Discrete Direction Experiments")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Select experiments to run
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{args.exp}'")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_list = [(args.exp, EXPERIMENTS[args.exp])]
    else:
        exp_list = list(EXPERIMENTS.items())
        if args.start_from:
            start_idx = None
            for i, (name, _) in enumerate(exp_list):
                if name == args.start_from:
                    start_idx = i
                    break
            if start_idx is None:
                print(f"Error: Unknown experiment '{args.start_from}'")
                sys.exit(1)
            exp_list = exp_list[start_idx:]

    print(f"\nExperiments to run: {len(exp_list)}")
    for name, config in exp_list:
        print(f"  - {name}: {config['desc']}")

    if args.dry_run:
        print("\n[DRY RUN MODE - Commands only]")

    # Run experiments
    results = {}
    for exp_name, config in exp_list:
        try:
            success = run_experiment(exp_name, config, dry_run=args.dry_run)
            results[exp_name] = "OK" if success else "FAILED"
        except KeyboardInterrupt:
            results[exp_name] = "INTERRUPTED"
            print("\n\nExperiments interrupted by user.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("  D Series Summary")
    print("=" * 60)
    for name, status in results.items():
        status_icon = "✓" if status == "OK" else "✗" if status == "FAILED" else "⊘"
        print(f"  {status_icon} {name}: {status}")

    failed = sum(1 for s in results.values() if s == "FAILED")
    if failed > 0:
        print(f"\n{failed} experiment(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

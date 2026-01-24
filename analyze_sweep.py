#!/usr/bin/env python3
"""
Analyze W&B sweep results and extract best configuration.

Usage:
    python analyze_sweep.py <sweep_id>

Example:
    python analyze_sweep.py ncridlig-ml4cv/runs-sweep/abc123xyz
"""

import sys
import os
from dotenv import load_dotenv
import wandb

load_dotenv()

# Set wandb API key
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key


def analyze_sweep(sweep_path):
    """Analyze sweep and print best configuration."""

    api = wandb.Api()

    # Get sweep
    sweep = api.sweep(sweep_path)

    print("=" * 70)
    print("SWEEP ANALYSIS")
    print("=" * 70)
    print(f"Sweep: {sweep.name}")
    print(f"ID: {sweep.id}")
    print(f"State: {sweep.state}")
    print(f"Method: {sweep.config.get('method', 'unknown')}")
    print()

    # Get all runs in sweep
    runs = list(sweep.runs)

    if not runs:
        print("No runs found in this sweep.")
        return

    print(f"Total runs: {len(runs)}")
    print()

    # Find best run by mAP50
    best_run = None
    best_map50 = 0

    for run in runs:
        if run.state == "finished":
            map50 = run.summary.get('metrics/mAP50(B)', 0)
            if map50 > best_map50:
                best_map50 = map50
                best_run = run

    if not best_run:
        print("No completed runs found.")
        return

    print("=" * 70)
    print("BEST RUN")
    print("=" * 70)
    print(f"Run name: {best_run.name}")
    print(f"Run ID: {best_run.id}")
    print(f"URL: {best_run.url}")
    print()

    print("METRICS:")
    print(f"  mAP50:     {best_run.summary.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95:  {best_run.summary.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {best_run.summary.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:    {best_run.summary.get('metrics/recall(B)', 0):.4f}")
    print()

    print("BASELINE COMPARISON:")
    baseline_map50 = 0.714
    improvement = ((best_map50 - baseline_map50) / baseline_map50) * 100
    print(f"  Baseline mAP50: {baseline_map50:.4f}")
    print(f"  Best mAP50:     {best_map50:.4f}")
    print(f"  Improvement:    {improvement:+.1f}%")
    print()

    print("=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)

    config = best_run.config

    # Group parameters
    lr_params = ['lr0', 'lrf', 'warmup_epochs']
    aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'close_mosaic',
                  'mixup', 'copy_paste', 'degrees']
    reg_params = ['weight_decay', 'dropout']

    print("\nLearning Rate:")
    for param in lr_params:
        if param in config:
            print(f"  {param:20s}: {config[param]}")

    print("\nAugmentation:")
    for param in aug_params:
        if param in config:
            print(f"  {param:20s}: {config[param]}")

    print("\nRegularization:")
    for param in reg_params:
        if param in config:
            print(f"  {param:20s}: {config[param]}")

    print("\nFixed:")
    for param in ['epochs', 'batch', 'imgsz', 'data']:
        if param in config:
            print(f"  {param:20s}: {config[param]}")

    print()
    print("=" * 70)
    print("PYTHON DICT (for train_best_config.py)")
    print("=" * 70)
    print("\nbest_config = {")
    for key, value in sorted(config.items()):
        if not key.startswith('_'):
            if isinstance(value, str):
                print(f"    '{key}': '{value}',")
            else:
                print(f"    '{key}': {value},")
    print("}")
    print()

    # Top 5 runs comparison
    print("=" * 70)
    print("TOP 5 RUNS")
    print("=" * 70)

    finished_runs = [r for r in runs if r.state == "finished"]
    sorted_runs = sorted(finished_runs,
                        key=lambda r: r.summary.get('metrics/mAP50(B)', 0),
                        reverse=True)

    print(f"{'Rank':<6} {'Run Name':<30} {'mAP50':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)

    for i, run in enumerate(sorted_runs[:5], 1):
        name = run.name[:28]
        map50 = run.summary.get('metrics/mAP50(B)', 0)
        prec = run.summary.get('metrics/precision(B)', 0)
        recall = run.summary.get('metrics/recall(B)', 0)
        print(f"{i:<6} {name:<30} {map50:<10.4f} {prec:<10.4f} {recall:<10.4f}")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Copy the best config dict above")
    print("2. Paste it into train_best_config.py")
    print("3. Run full 300-epoch training:")
    print("   python train_best_config.py")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep.py <sweep_path>")
        print()
        print("Example:")
        print("  python analyze_sweep.py ncridlig-ml4cv/runs-sweep/abc123xyz")
        print()
        print("To find your sweep ID:")
        print("  1. Go to https://wandb.ai/ncridlig-ml4cv/runs-sweep")
        print("  2. Click on your sweep")
        print("  3. Copy the ID from the URL or page")
        sys.exit(1)

    sweep_path = sys.argv[1]

    try:
        analyze_sweep(sweep_path)
    except Exception as e:
        print(f"Error analyzing sweep: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

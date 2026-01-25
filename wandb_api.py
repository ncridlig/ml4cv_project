#!/usr/bin/env python3
"""
W&B API interface for monitoring training runs.

Usage:
    python wandb_api.py <run_path> [--info] [--metrics] [--history] [--best]

Examples:
    python wandb_api.py ncridlig-ml4cv/runs-baseline/yolov11n_300ep_baseline8_20260122_153625 --info
    python wandb_api.py ncridlig-ml4cv/runs-baseline/yolov11n_300ep_baseline8_20260122_153625 --metrics
    python wandb_api.py ncridlig-ml4cv/runs-baseline/yolov11n_300ep_baseline8_20260122_153625 --all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Fix import shadowing: Remove current directory from sys.path temporarily
# to avoid importing local wandb/ directory instead of installed package
_original_path = sys.path.copy()
_script_dir = str(Path(__file__).parent.absolute())
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

try:
    import wandb
    from wandb.apis.public import Run
except ImportError:
    sys.exit("Error: wandb not installed. Run: pip install wandb")
finally:
    # Restore original sys.path
    sys.path = _original_path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_api_key() -> Optional[str]:
    """Load W&B API key from .env file or environment."""
    # Check environment first
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]

    # Try loading from .env in script directory
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("WAND_DB_API_KEY=") or line.startswith("WANDB_API_KEY="):
                    return line.split("=", 1)[1].strip()

    return None


def get_run(run_path: str) -> Run:
    """Get a W&B run by path."""
    api_key = load_api_key()
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    api = wandb.Api()
    return api.run(run_path)


def print_run_info(run: Run) -> None:
    """Print basic run information."""
    print("=" * 50)
    print("RUN INFO")
    print("=" * 50)
    print(f"Name:      {run.name}")
    print(f"State:     {run.state}")
    print(f"Created:   {run.created_at}")
    print(f"URL:       {run.url}")

    if hasattr(run, 'config') and run.config:
        print("\nConfig:")
        for key, value in sorted(run.config.items()):
            if not key.startswith('_'):
                print(f"  {key}: {value}")


def print_current_metrics(run: Run) -> None:
    """Print current/latest metrics from run summary."""
    print("=" * 50)
    print("CURRENT METRICS")
    print("=" * 50)

    summary = run.summary

    # Key metrics we care about
    key_metrics = [
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ]

    for key, label in key_metrics:
        if key in summary:
            print(f"{label:15}: {summary[key]:.4f}")

    print()

    # Loss values
    loss_metrics = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ]

    print("Loss Values:")
    for key, label in loss_metrics:
        if key in summary:
            print(f"  {label:15}: {summary[key]:.5f}")

    print()

    # Model info
    model_metrics = [
        ("model/GFLOPs", "GFLOPs"),
        ("model/parameters", "Parameters"),
        ("model/speed_PyTorch(ms)", "Speed (ms)"),
    ]

    print("Model Info:")
    for key, label in model_metrics:
        if key in summary:
            value = summary[key]
            if isinstance(value, float):
                print(f"  {label:15}: {value:.3f}")
            else:
                print(f"  {label:15}: {value}")


def print_history(run: Run, last_n: int = 10) -> None:
    """Print training history."""
    print("=" * 50)
    print(f"TRAINING HISTORY (last {last_n} epochs)")
    print("=" * 50)

    if not HAS_PANDAS:
        print("Warning: pandas not installed, limited history output")
        history = run.history(samples=5000, pandas=False)
        print(f"Total logged steps: {len(history)}")
        return

    history = run.history(samples=5000)

    if history.empty:
        print("No history data available.")
        return

    print(f"Total logged steps: {len(history)}")

    # Find the epoch-like column
    epoch_cols = [c for c in history.columns if 'epoch' in c.lower()]
    step_col = epoch_cols[0] if epoch_cols else '_step'

    # Key metric columns
    metric_cols = [c for c in history.columns if c.startswith('metrics/')]

    if metric_cols:
        # Get rows with metric data (validation epochs)
        metric_data = history[metric_cols].dropna(how='all')

        if not metric_data.empty:
            print(f"\nMetrics logged at {len(metric_data)} epochs")
            print()

            # Show last N epochs
            recent = metric_data.tail(last_n)

            # Rename columns for display
            display_df = recent.copy()
            display_df.columns = [c.replace('metrics/', '').replace('(B)', '') for c in display_df.columns]

            # Round values
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].round(4)

            print(display_df.to_string())


def print_best_metrics(run: Run) -> None:
    """Print best metrics achieved during training."""
    print("=" * 50)
    print("BEST METRICS")
    print("=" * 50)

    if not HAS_PANDAS:
        print("Warning: pandas not installed, cannot compute best metrics")
        return

    history = run.history(samples=5000)

    if history.empty:
        print("No history data available.")
        return

    metrics = {
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
    }

    for col, label in metrics.items():
        if col in history.columns:
            best = history[col].max()
            if pd.notna(best):
                print(f"Best {label:12}: {best:.4f}")


def print_comparison(run: Run, baseline: dict) -> None:
    """Print comparison against baseline metrics."""
    print("=" * 50)
    print("COMPARISON VS BASELINE")
    print("=" * 50)

    summary = run.summary

    metrics = [
        ("metrics/mAP50(B)", "mAP50", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95", "mAP50_95"),
        ("metrics/precision(B)", "Precision", "precision"),
        ("metrics/recall(B)", "Recall", "recall"),
    ]

    for key, label, baseline_key in metrics:
        if key in summary and baseline_key in baseline:
            current = summary[key]
            target = baseline[baseline_key]
            diff = current - target
            status = "+" if diff >= 0 else ""
            print(f"{label:12}: {current:.4f} (baseline: {target:.4f}, {status}{diff:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="W&B API interface for monitoring training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python wandb_api.py ncridlig-ml4cv/runs-baseline/yolov11n_300ep_baseline8_20260122_153625 --all
    python wandb_api.py ncridlig-ml4cv/runs-baseline/yolov11n_300ep_baseline8_20260122_153625 --metrics --compare
        """
    )

    parser.add_argument("run_path", help="W&B run path (entity/project/run_id)")
    parser.add_argument("--info", action="store_true", help="Show run info")
    parser.add_argument("--metrics", action="store_true", help="Show current metrics")
    parser.add_argument("--history", action="store_true", help="Show training history")
    parser.add_argument("--history-n", type=int, default=10, help="Number of history entries to show")
    parser.add_argument("--best", action="store_true", help="Show best metrics achieved")
    parser.add_argument("--compare", action="store_true", help="Compare against thesis baseline")
    parser.add_argument("--all", action="store_true", help="Show all information")

    args = parser.parse_args()

    # If no specific flags, show metrics by default
    if not any([args.info, args.metrics, args.history, args.best, args.compare, args.all]):
        args.metrics = True

    if args.all:
        args.info = args.metrics = args.history = args.best = args.compare = True

    try:
        run = get_run(args.run_path)
    except wandb.errors.CommError as e:
        sys.exit(f"Error connecting to W&B: {e}")
    except Exception as e:
        sys.exit(f"Error: {e}")

    # Baseline from Edo's thesis (YOLOv11n, 300 epochs, FSOCO)
    thesis_baseline = {
        "mAP50": 0.824,
        "mAP50_95": 0.570,
        "precision": 0.849,
        "recall": 0.765,
    }

    if args.info:
        print_run_info(run)
        print()

    if args.metrics:
        print_current_metrics(run)
        print()

    if args.history:
        print_history(run, args.history_n)
        print()

    if args.best:
        print_best_metrics(run)
        print()

    if args.compare:
        print_comparison(run, thesis_baseline)
        print()


if __name__ == "__main__":
    main()

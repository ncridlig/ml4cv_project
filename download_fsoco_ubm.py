#!/usr/bin/env python3
"""
Download the FSOCO-UBM test dataset (in-house UBM test set from car camera data).

This is a small test-only dataset (~96 images) extracted from the car's
camera recordings at Rioveggio test track (November 20, 2025).

Usage:
    python download_fsoco_ubm.py [--output-dir PATH]

The script loads the Roboflow API key from .env file or ROBOFLOW_API_KEY env var.
"""

import argparse
import os
import sys
from pathlib import Path


def load_api_key() -> str:
    """Load Roboflow API key from .env file or environment."""
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"]

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip()

    sys.exit("Error: ROBOFLOW_API_KEY not found in .env or environment")


def main():
    parser = argparse.ArgumentParser(description="Download FSOCO-UBM test dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Output directory (default: ./datasets)"
    )
    parser.add_argument(
        "--format",
        default="yolo26",
        help="Dataset format (default: yolo26)"
    )
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Error: roboflow not installed. Run: pip install roboflow")

    api_key = load_api_key()

    # FSOCO-UBM test dataset (in-house UBM test set)
    WORKSPACE = "fsae-okyoe"
    PROJECT = "ml4cv_project"
    VERSION = 1

    print(f"Downloading FSOCO-UBM test dataset...")
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Project:   {PROJECT}")
    print(f"  Version:   {VERSION}")
    print(f"  Format:    {args.format}")
    print(f"  Output:    {args.output_dir}")
    print()
    print("Dataset Info:")
    print("  - Source: UBM car camera (ZED 2i stereo)")
    print("  - Date: November 20, 2025 (Rioveggio test track)")
    print("  - Size: ~96 images (test-only dataset)")
    print("  - Use: Real-world validation for cone detection models")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(args.format)

    print()
    print(f"Download complete: {dataset.location}")
    print()
    print("To evaluate models on fsoco-ubm test set:")
    print(f"  python3 evaluate_yolo26_ubm_test.py")
    print(f"  python3 evaluate_yolo12_ubm_test.py")
    print(f"  python3 evaluate_baseline_ubm_test.py")
    print()
    print("Note: This is a TEST-ONLY dataset for real-world validation.")
    print("      Do NOT use for training or hyperparameter tuning.")


if __name__ == "__main__":
    main()

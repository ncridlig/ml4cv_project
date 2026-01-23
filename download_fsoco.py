#!/usr/bin/env python3
"""
Download the FSOCO dataset (same version used in Edo's thesis).

Usage:
    python download_fsoco.py [--output-dir PATH]

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
    parser = argparse.ArgumentParser(description="Download FSOCO dataset for YOLO training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Output directory (default: ./datasets)"
    )
    parser.add_argument(
        "--format",
        default="yolov11",
        help="Dataset format (default: yolov11)"
    )
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Error: roboflow not installed. Run: pip install roboflow")

    api_key = load_api_key()

    # Dataset used in Edo's thesis
    WORKSPACE = "fmdv"
    PROJECT = "fsoco-kxq3s"
    VERSION = 12

    print(f"Downloading FSOCO dataset...")
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Project:   {PROJECT}")
    print(f"  Version:   {VERSION}")
    print(f"  Format:    {args.format}")
    print(f"  Output:    {args.output_dir}")
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
    print("To train, run:")
    print(f"  yolo train model=yolo11n.pt data={dataset.location}/data.yaml epochs=300")


if __name__ == "__main__":
    main()

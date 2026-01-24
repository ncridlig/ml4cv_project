#!/usr/bin/env python3
"""
Baseline YOLOv11n training to reproduce Edo Fusa's thesis results.
Target metrics: mAP50 = 0.824, Precision = 0.849, Recall = 0.765
"""
import argparse
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train YOLOv11n baseline model')
parser.add_argument(
    '--data',
    type=str,
    default='datasets/FSOCO-12/data.yaml',
    help='Path to dataset YAML file (default: datasets/FSOCO-12/data.yaml - correct FSOCO dataset from thesis)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='Number of training epochs (default: 300)'
)
parser.add_argument(
    '--batch',
    type=int,
    default=64,
    help='Batch size (default: 64)'
)
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Set wandb environment variables for Ultralytics integration
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'yolov11n_baseline_300ep_FSOCO'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print("Wandb API key set - Ultralytics will handle wandb integration automatically")

# Print configuration
print(f"\n{'='*50}")
print("TRAINING CONFIGURATION")
print(f"{'='*50}")
print(f"Dataset: {args.data}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch}")
print(f"{'='*50}\n")

# Verify CUDA is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Initialize YOLOv11n model
model = YOLO('yolo11n.pt')

# Train with parameters matching Edo's thesis
# Note: Exact hyperparameters were lost per CLAUDE.md, using Ultralytics defaults
# Performance optimizations: larger batch, more workers for max GPU utilization
# Ultralytics automatically integrates with wandb when installed
results = model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,  # Edo used 640x640 resolution
    batch=args.batch,  # Stable batch size (tested working with 11GB VRAM usage)
    device=0,  # Use GPU 0
    project='runs/baseline',
    name='yolov11n_300ep_FSOCO_correct',
    exist_ok=False,
    verbose=True,
    # Performance optimizations
    workers=16,  # More CPU cores for data loading (reduce I/O bottleneck)
    # Save checkpoints
    save=True,
    save_period=50,  # Save every 50 epochs
    # Weights & Biases integration (automatic when wandb is installed)
    plots=True,  # Generate plots for wandb
)

# Print final metrics
print("\n" + "="*50)
print("BASELINE TRAINING COMPLETE")
print("="*50)
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
print("\nTarget metrics (from Edo's thesis):")
print("  mAP50: 0.824")
print("  Precision: 0.849")
print("  Recall: 0.765")
print("="*50)

# Validate on test set
print("\nRunning validation on test set...")
test_results = model.val(data=args.data, split='test')
print(f"Test mAP50: {test_results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

# Wandb finish is handled automatically by Ultralytics

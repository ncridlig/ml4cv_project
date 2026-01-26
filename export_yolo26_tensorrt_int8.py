#!/usr/bin/env python3
"""
Export YOLO26 to TensorRT INT8 engine.

This is Step 2 of the INT8 optimization pipeline.
Uses validation set for calibration (NEVER test set!).

Improvements over YOLO12 export:
- FP16 fallback enabled for better accuracy
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'yolo26-optimization'
os.environ['WANDB_NAME'] = 'YOLO26n_INT8_export'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("STEP 2: EXPORT YOLO26 TO TENSORRT INT8")
print("=" * 70)
print()

# Load trained model
model_path = 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt'
print(f"Loading model: {model_path}")
model = YOLO(model_path)

print("Model loaded successfully")
print()

# Export to TensorRT INT8
print("Exporting to TensorRT INT8 engine...")
print("  - Format: TensorRT engine")
print("  - Precision: INT8 (8-bit quantization)")
print("  - Image size: 640x640")
print("  - Batch size: 2 (stereo: left + right image)")
print("  - Calibration: Validation set (FSOCO-12)")
print("  - FP16 fallback: Enabled (better accuracy)")
print()
print("⚠️  CRITICAL: Using VALIDATION SET for calibration")
print("   - Test set remains unseen for final evaluation")
print("   - Calibration uses ~500 images from validation set")
print()
print("This may take 5-10 minutes...")
print()

model.export(
    format='engine',      # TensorRT format
    imgsz=640,
    batch=2,              # Batch size 2 for stereo (left + right image)
    half=True,            # Enable FP16 fallback for better accuracy
    int8=False,            # Enable INT8 quantization
    data='datasets/FSOCO-12/data.yaml',  # Provides validation data for calibration
    device=0,             # GPU to use
    workspace=16,          # Max workspace size in GB (increased from 4 GB)
)

print()
print("=" * 70)
print("✅ TENSORRT INT8 EXPORT COMPLETE")
print("=" * 70)
print()
print(f"Output: runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.engine")
print()
print("Next step: Run benchmark_yolo26_int8.py to measure speed and accuracy")
print("=" * 70)

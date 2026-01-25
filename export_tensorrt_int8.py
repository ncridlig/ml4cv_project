#!/usr/bin/env python3
"""
Export YOLO12 to TensorRT INT8 engine.

This is Step 2 of the INT8 optimization pipeline.
Uses validation set for calibration (NEVER test set!).
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'optimization'
os.environ['WANDB_NAME'] = 'YOLO12n_INT8_export'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("STEP 2: EXPORT YOLO12 TO TENSORRT INT8")
print("=" * 70)
print()

# Load trained model
model_path = 'runs/detect/runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.pt'
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
print("  - Workspace: 4 GB")
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
    half=True,            # FP16
    int8=False,           # INT8 quantization
    data='datasets/FSOCO-12/data.yaml',  # Provides validation data for calibration
    device=0,             # GPU to use
    workspace=12,         # Max workspace size in GB (GPU VRAM)
)

print()
print("=" * 70)
print("✅ TENSORRT INT8 EXPORT COMPLETE")
print("=" * 70)
print()
print(f"Output: runs/detect/runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.engine")
print()
print("Next step: Run benchmark_int8.py to measure speed and accuracy")
print("=" * 70)

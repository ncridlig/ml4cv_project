#!/usr/bin/env python3
"""
Export YOLO26 to ONNX format for TensorRT conversion.

This is Step 1 of the INT8 optimization pipeline.
ONNX is an intermediate format that TensorRT can convert to optimized engines.
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'yolo26-optimization'
os.environ['WANDB_NAME'] = 'YOLO26n_ONNX_export'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("STEP 1: EXPORT YOLO26 TO ONNX")
print("=" * 70)
print()

# Load trained YOLO26 model
model_path = 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt'
print(f"Loading model: {model_path}")
model = YOLO(model_path)

print("Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print()

# Export to ONNX
print("Exporting to ONNX format...")
print("  - Format: ONNX")
print("  - Image size: 640x640")
print("  - Batch size: 2 (stereo: left + right image)")
print("  - Simplify: True (optimize graph)")
print("  - Opset: 17 (stable version)")
print()

model.export(
    format='onnx',        # Export format
    imgsz=640,            # Input image size
    batch=2,              # Batch size (2 for stereo: left + right image)
    simplify=True,        # Simplify ONNX graph
    opset=17,             # ONNX opset version (17 is stable)
    dynamic=False,        # Static shapes (faster)
)

print()
print("=" * 70)
print("✅ ONNX EXPORT COMPLETE")
print("=" * 70)
print()
print(f"Output: runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.onnx")
print()
print("Next step: Run export_yolo26_tensorrt_int8.py to convert to INT8 TensorRT engine")
print("=" * 70)

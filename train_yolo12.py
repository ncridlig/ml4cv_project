#!/usr/bin/env python3
"""
Train YOLO12n on FSOCO-12 dataset.

YOLO12 is the 2025 state-of-the-art attention-centric YOLO architecture.
Expected performance: mAP50 ~0.83-0.84, inference ~1.5ms on RTX 4060

Branch A: Primary optimization strategy
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'YOLO12n_300ep_FSOCO_Branch_A'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("TRAINING YOLO12n - BRANCH A STRATEGY")
print("=" * 70)
print()
print("Model: YOLO12n (attention-centric architecture, 2025 release)")
print("Dataset: FSOCO-12 (7,120 train / 1,968 val / 689 test)")
print("Epochs: 300")
print("Batch: 48 (RTX 4080 Super)")
print("Expected: mAP50 ~0.83-0.84 (+16% over baseline)")
print()
print("Key innovations:")
print("  - Area Attention Mechanism (efficient self-attention)")
print("  - R-ELAN (Residual Efficient Layer Aggregation)")
print("  - FlashAttention integration")
print()
print("Hardware:")
print("  - Training: RTX 4080 Super")
print("  - Deployment: RTX 4060 (on car)")
print()
print("=" * 70)
print()

# Load YOLO12n pretrained model
print("Loading YOLO12n pretrained model...")
model = YOLO('yolo12n.pt')

print("Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print()

# Train on FSOCO-12
print("Starting training on FSOCO-12...")
print()

results = model.train(
    data='datasets/FSOCO-12/data.yaml',
    epochs=300,
    batch=64,
    imgsz=640,
    device=0,
    workers=12,
    project='runs/yolo12',
    name='yolo12n_300ep_FSOCO',

    # Use default hyperparameters (they work best!)
    # No custom augmentation - defaults are excellent

    # Patience for early stopping (optional)
    patience=50,  # Stop if no improvement for 50 epochs

    # Save best model
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs

    # Validation
    val=True,

    # Mixed precision training (faster)
    amp=True,
)

# Print final results
print("\n" + "=" * 70)
print("YOLO12n TRAINING COMPLETE")
print("=" * 70)
print(f"Best mAP50:      {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Best mAP50-95:   {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"Best Precision:  {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Best Recall:     {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()
print("Model saved to: runs/yolo12/yolo12n_300ep_FSOCO/weights/best.pt")
print()

# Compare to baseline
baseline_map50 = 0.714  # Your baseline on validation
yolo12_map50 = results.results_dict.get('metrics/mAP50(B)', 0)

print("COMPARISON TO BASELINE:")
print(f"  Baseline YOLOv11n: {baseline_map50:.4f} mAP50")
print(f"  YOLO12n:           {yolo12_map50:.4f} mAP50")
print(f"  Delta:             {yolo12_map50 - baseline_map50:+.4f} ({((yolo12_map50 - baseline_map50)/baseline_map50)*100:+.1f}%)")
print()

# Decision point for Day 3
if yolo12_map50 >= 0.7065:
    print("✅ SUCCESS: YOLO12n achieved target (≥ YOLO11n mAP50)")
    print("   → Continue with Branch A (INT8 quantization)")
    print()
    print("Next steps:")
    print("  1. Evaluate on test set: python evaluate_yolo12_test.py")
    print("  2. Export to ONNX: yolo export model=runs/yolo12/.../best.pt format=onnx batch=2")
    print("  3. INT8 quantization with TensorRT")
elif yolo12_map50 >= 0.6655:
    print("⚠️ MARGINAL: YOLO12n ≥ UBM Baseline")
    print("   → Consider: Try YOLO12s (larger model) or pivot to Branch B")
else:
    print("❌ BELOW TARGET: YOLO12n < 0.655 mAP50 (UBM Baseline)")
    print("   → PIVOT TO BRANCH B (Knowledge Distillation + RegNet)")
    print()
    print("Branch B steps:")
    print("  1. Train YOLOv11m teacher (large model)")
    print("  2. Knowledge distillation: YOLOv11m → YOLOv11n")
    print("  3. RegNet backbone integration")

print()
print("=" * 70)

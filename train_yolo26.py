#!/usr/bin/env python3
"""
Train YOLO26n on FSOCO-12 dataset.

YOLO26 is the latest YOLO architecture from Ultralytics.
Expected performance: Similar or better than YOLO12n (0.7081 mAP50 on test set)

This uses the same training procedure as YOLO12.
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'yolo26-training'
os.environ['WANDB_NAME'] = 'YOLO26n_300ep_FSOCO'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("TRAINING YOLO26n ON FSOCO-12")
print("=" * 70)
print()
print("Model: YOLO26n (latest Ultralytics architecture)")
print("Dataset: FSOCO-12 (7,120 train / 1,968 val / 689 test)")
print("Epochs: 300")
print("Batch: 64")
print("Expected: Similar or better than YOLO12n (0.7081 mAP50 on test)")
print()
print("Comparison targets:")
print("  - YOLO12n (test): 0.7081 mAP50")
print("  - YOLOv11n baseline (test): 0.7065 mAP50")
print("  - UBM production (test): 0.6655 mAP50")
print()
print("Hardware:")
print("  - Training: RTX 4080 Super")
print("  - Deployment: RTX 4060 (on car)")
print()
print("=" * 70)
print()

# Load YOLO26n pretrained model
print("Loading YOLO26n pretrained model...")
model = YOLO('yolo26n.pt')

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
    project='runs/yolo26',
    name='yolo26n_300ep_FSOCO',

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
print("YOLO26n TRAINING COMPLETE")
print("=" * 70)
print(f"Best mAP50:      {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Best mAP50-95:   {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"Best Precision:  {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Best Recall:     {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()
print("Model saved to: runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt")
print()

# Compare to baselines
yolo12_map50 = 0.7127  # YOLO12 validation
yolo26_map50 = results.results_dict.get('metrics/mAP50(B)', 0)

print("COMPARISON TO BASELINES (Validation Set):")
print(f"  YOLO12n:           {yolo12_map50:.4f} mAP50")
print(f"  YOLO26n:           {yolo26_map50:.4f} mAP50")
print(f"  Delta:             {yolo26_map50 - yolo12_map50:+.4f} ({((yolo26_map50 - yolo12_map50)/yolo12_map50)*100:+.1f}%)")
print()

# Decision point
if yolo26_map50 > yolo12_map50:
    improvement = ((yolo26_map50 - yolo12_map50) / yolo12_map50) * 100
    print(f"✅ SUCCESS: YOLO26n improved over YOLO12n by {improvement:.1f}%!")
    print("   Newer architecture provides better performance")
elif yolo26_map50 >= yolo12_map50 * 0.99:
    print("⚠️ SIMILAR: YOLO26n achieved similar performance to YOLO12n")
    print("   Both architectures perform comparably on this dataset")
else:
    gap = ((yolo12_map50 - yolo26_map50) / yolo12_map50) * 100
    print(f"⚠️ YOLO26n underperformed YOLO12n by {gap:.1f}%")
    print("   YOLO12 may be better suited for this task")

print()
print("Next steps:")
print("  1. Evaluate on test set: python3 evaluate_yolo26_test.py")
print("  2. Compare to YOLO12 test results (0.7081 mAP50)")
print("  3. If better, export to INT8 and deploy")
print()
print("=" * 70)

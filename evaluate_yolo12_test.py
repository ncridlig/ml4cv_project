#!/usr/bin/env python3
"""
Evaluate YOLO12n on FSOCO-12 TEST set.
Compare to UBM's baseline (mAP50 = 0.6655 on test set).
Compare to Yolo11n baseline (mAP50 = 0.7065 on test set).
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'YOLO12n_test_set_evaluation'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("EVALUATING YOLO12n ON FSOCO-12 TEST SET")
print("=" * 70)
print()
print("Model: runs/yolo12/yolo12n_300ep_FSOCO/weights/best.pt")
print("Dataset: FSOCO-12 test set (689 images)")
print()
print("Comparison target:")
print("UBM's baseline (YOLO11n, test set): mAP50 = 0.6655")
print()
print("=" * 70)
print()

# Load YOLO12n model
model_path = 'runs/detect/runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.pt'
model = YOLO(model_path)

print("Model loaded successfully")
print(f"Classes: {model.names}")
print()

# Evaluate on FSOCO-12 test set
print("Running evaluation on FSOCO-12 test set...")
print()

results = model.val(
    data='datasets/FSOCO-12/data.yaml',
    split='test',  # CRITICAL: Use test set, not validation!
    batch=32,
    device=0,
    plots=True,
    save_json=True,
    project='runs/evaluation',
    name='yolo12n_on_test_set',
)

# Print results
print("\n" + "=" * 70)
print("YOLO12n RESULTS ON FSOCO-12 TEST SET")
print("=" * 70)
print(f"mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"mAP50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()

print("COMPARISON TO BASELINES:")
print("-" * 70)
print(f"{'Model':<30} {'mAP50 (Test)':<15} {'Delta vs YOLO11n':<20}")
print("-" * 70)

yolo12_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
yolo12_precision = results.results_dict.get('metrics/precision(B)', 0)
yolo12_recall = results.results_dict.get('metrics/recall(B)', 0)

yolo11n_map50 = 0.7065

print(f"{'YOLO11n':<30} {yolo11n_map50:<15.4f} {'—':<20}")
print(f"{'YOLO12n':<30} {yolo12_map50:<15.4f} {(yolo12_map50 - yolo11n_map50):+.4f} ({((yolo12_map50 - yolo11n_map50)/yolo11n_map50)*100:+.1f}%)")
print()

# Success analysis
if yolo12_map50 >= yolo11n_map50:
    print("✅ SUCCESS: YOLO12n matched or exceeded YOLO11n's baseline!")
    improvement = ((yolo12_map50 - yolo11n_map50) / yolo11n_map50) * 100
    print(f"   Improvement: +{improvement:.1f}% over production baseline")
elif yolo12_map50 >= yolo11n_map50 * 0.97:  # Within 3%
    print("⚠️ CLOSE: YOLO12n is within 3% of YOLO11n's baseline")
    gap = ((yolo11n_map50 - yolo12_map50) / yolo11n_map50) * 100
    print(f"   Gap: -{gap:.1f}% (acceptable given architecture change)")
else:
    print("❌ GAP: YOLO12n fell short of YOLO11n's baseline")
    gap = ((yolo11n_map50 - yolo12_map50) / yolo11n_map50) * 100
    print(f"   Gap: -{gap:.1f}%")
    print("   Consider: Larger model (YOLO12s) or pivot to Branch B")

print()
print("=" * 70)
print()
print("Results saved to: runs/evaluation/yolo12n_on_test_set/")
print("=" * 70)

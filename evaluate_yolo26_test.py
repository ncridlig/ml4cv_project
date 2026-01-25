#!/usr/bin/env python3
"""
Evaluate YOLO26n on FSOCO-12 TEST set.
Compare to YOLO12n (0.7081 mAP50) and UBM production (0.6655 mAP50).
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'yolo26-training'
os.environ['WANDB_NAME'] = 'YOLO26n_test_set_evaluation'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("EVALUATING YOLO26n ON FSOCO-12 TEST SET")
print("=" * 70)
print()
print("Model: runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt")
print("Dataset: FSOCO-12 test set (689 images)")
print()
print("Comparison targets:")
print("  YOLO12n (test set): mAP50 = 0.7081")
print("  YOLOv11n baseline (test set): mAP50 = 0.7065")
print("  UBM production (test set): mAP50 = 0.6655")
print()
print("=" * 70)
print()

# Load YOLO26n model
model_path = 'runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt'
print(f"Loading model: {model_path}")
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
    name='yolo26n_on_test_set',
)

# Print results
print("\n" + "=" * 70)
print("YOLO26n RESULTS ON FSOCO-12 TEST SET")
print("=" * 70)
print(f"mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"mAP50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()

print("COMPARISON TO BASELINES:")
print("-" * 70)
print(f"{'Model':<30} {'mAP50 (Test)':<15} {'Delta vs YOLO12n':<20}")
print("-" * 70)

yolo26_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
yolo26_precision = results.results_dict.get('metrics/precision(B)', 0)
yolo26_recall = results.results_dict.get('metrics/recall(B)', 0)

ubm_map50 = 0.6655
yolo11n_map50 = 0.7065
yolo12_map50 = 0.7081

print(f"{'UBM Production':<30} {ubm_map50:<15.4f} {'—':<20}")
print(f"{'YOLOv11n Baseline':<30} {yolo11n_map50:<15.4f} {'—':<20}")
print(f"{'YOLO12n':<30} {yolo12_map50:<15.4f} {'—':<20}")
print(f"{'YOLO26n (newest)':<30} {yolo26_map50:<15.4f} {(yolo26_map50 - yolo12_map50):+.4f} ({((yolo26_map50 - yolo12_map50)/yolo12_map50)*100:+.1f}%)")
print()

# Success analysis
if yolo26_map50 > yolo12_map50:
    improvement = ((yolo26_map50 - yolo12_map50) / yolo12_map50) * 100
    print(f"✅ SUCCESS: YOLO26n outperformed YOLO12n!")
    print(f"   Improvement: +{improvement:.1f}% over YOLO12n")
    print(f"   Latest architecture provides better results")
elif yolo26_map50 >= yolo12_map50 * 0.99:
    gap = abs((yolo12_map50 - yolo26_map50) / yolo12_map50) * 100
    print(f"⚠️ SIMILAR: YOLO26n achieved comparable performance to YOLO12n")
    print(f"   Difference: {gap:.1f}% (within 1%)")
    print(f"   Both architectures work well for cone detection")
else:
    gap = ((yolo12_map50 - yolo26_map50) / yolo12_map50) * 100
    print(f"⚠️ YOLO26n underperformed YOLO12n")
    print(f"   Gap: -{gap:.1f}%")
    print(f"   YOLO12 may be better suited for this dataset")

print()

# vs UBM production
ubm_improvement = ((yolo26_map50 - ubm_map50) / ubm_map50) * 100
print(f"vs UBM Production: +{ubm_improvement:.1f}%")
print()

# Per-class breakdown (if needed)
print("Per-Class Performance:")
print("-" * 70)
# This will be printed by Ultralytics automatically during validation

print()
print("=" * 70)
print()
print("Results saved to: runs/evaluation/yolo26n_on_test_set/")
print("=" * 70)

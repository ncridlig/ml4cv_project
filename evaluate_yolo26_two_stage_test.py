#!/usr/bin/env python3
"""
Evaluate Two-Stage YOLO26n on FSOCO-12 Test Set.

Compares two-stage training (pre-train + fine-tune) against single-stage baseline.
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'two-stage-yolo26'
os.environ['WANDB_NAME'] = 'YOLO26n_TwoStage_TestEval'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 80)
print("EVALUATE TWO-STAGE YOLO26n ON TEST SET")
print("=" * 80)
print()

# Model path
model_path = 'runs/detect/runs/two-stage-yolo26/stage2b_full_finetune_250ep/weights/best.pt'

print(f"Model: {model_path}")
print(f"Dataset: FSOCO-12 test set (689 images, 12,054 instances)")
print()

# Load model
print("Loading two-stage YOLO26n model...")
model = YOLO(model_path)
print("✅ Model loaded successfully")
print()

# Run evaluation on test set
print("=" * 80)
print("RUNNING EVALUATION ON TEST SET")
print("=" * 80)
print()
print("⏳ This will take ~2-3 minutes...")
print()

results = model.val(
    data='datasets/FSOCO-12/data.yaml',
    split='test',  # CRITICAL: Use test set for final evaluation
    batch=32,
    device=0,
    plots=True,
    save_json=True,
    project='runs/evaluation',
    name='yolo26n_two_stage_on_test_set',
)

# Extract metrics
map50 = results.results_dict.get('metrics/mAP50(B)', 0)
map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
precision = results.results_dict.get('metrics/precision(B)', 0)
recall = results.results_dict.get('metrics/recall(B)', 0)

print()
print("=" * 80)
print("TWO-STAGE YOLO26n RESULTS ON FSOCO-12 TEST SET")
print("=" * 80)
print()
print(f"mAP50-95:  {map50_95:.4f}")
print(f"mAP50:     {map50:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print()

# Compare to baselines
print("=" * 80)
print("COMPARISON TO BASELINES")
print("=" * 80)
print()

# Baselines (test set results)
ubm_production = 0.6655
yolo11n_baseline = 0.7065
yolo12n = 0.7081
yolo26n_single = 0.7626  # Single-stage YOLO26n test result

print(f"{'Model':<30} {'mAP50 (Test)':<15} {'Delta vs Two-Stage':<20}")
print("-" * 80)
print(f"{'UBM Production':<30} {ubm_production:<15.4f} {'—':<20}")
print(f"{'YOLOv11n Baseline':<30} {yolo11n_baseline:<15.4f} {'—':<20}")
print(f"{'YOLO12n':<30} {yolo12n:<15.4f} {'—':<20}")
print(f"{'YOLO26n (single-stage)':<30} {yolo26n_single:<15.4f} {(map50 - yolo26n_single):+.4f} ({((map50 - yolo26n_single)/yolo26n_single)*100:+.2f}%)")
print(f"{'YOLO26n (two-stage)':<30} {map50:<15.4f} {'—':<20}")
print()

# Detailed comparison
print("=" * 80)
print("TWO-STAGE vs SINGLE-STAGE YOLO26n")
print("=" * 80)
print()
print(f"{'Metric':<15} {'Single-Stage':<15} {'Two-Stage':<15} {'Delta':<15}")
print("-" * 80)
print(f"{'mAP50':<15} {yolo26n_single:<15.4f} {map50:<15.4f} {(map50 - yolo26n_single):+.4f}")
print(f"{'mAP50-95':<15} {'0.5244':<15} {map50_95:<15.4f} {'—':<15}")
print(f"{'Precision':<15} {'0.8485':<15} {precision:<15.4f} {'—':<15}")
print(f"{'Recall':<15} {'0.6935':<15} {recall:<15.4f} {'—':<15}")
print()

# Analysis
improvement = map50 - yolo26n_single
improvement_pct = (improvement / yolo26n_single) * 100

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if map50 > yolo26n_single + 0.005:  # > 0.5% improvement
    print(f"✅ SUCCESS: Two-stage training improved over single-stage!")
    print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print()
    print(f"   🏆 New Best Test mAP50: {map50:.4f}")
    print()
    print("   Why it worked:")
    print("   - Pretraining on 22,725 images learned better features")
    print("   - Extended training (700 vs 300 epochs) allowed full convergence")
    print("   - Fine-tuning adapted features to FSOCO-12 distribution")
    print()
    print("   ✅ RECOMMENDATION: Deploy two-stage model")
    print(f"      Model: runs/two-stage-yolo26/stage2_fsoco12_300ep/weights/best.pt")

elif map50 > yolo26n_single * 0.995:  # Within 0.5%
    print(f"⚠️ SIMILAR: Two-stage and single-stage achieved comparable performance")
    print(f"   Difference: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print()
    print("   Interpretation:")
    print("   - Pretraining didn't harm performance")
    print("   - Extended training maintained quality")
    print("   - Cone-detector and FSOCO-12 have similar distributions")
    print()
    print("   ✅ RECOMMENDATION: Either model works")
    print(f"      Single-stage simpler, two-stage more robust")

else:
    print(f"❌ UNDERPERFORMED: Two-stage below single-stage")
    print(f"   Difference: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print()
    print("   Possible reasons:")
    print("   - Cone-detector distribution mismatch with FSOCO-12")
    print("   - Fine-tuning learning rate too low")
    print("   - Negative transfer from pretraining")
    print()
    print("   ✅ RECOMMENDATION: Stick with single-stage YOLO26n")
    print(f"      Model: runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt")

print()

# vs UBM production
vs_ubm = ((map50 - ubm_production) / ubm_production) * 100
print(f"vs UBM Production: +{vs_ubm:.1f}% ({map50:.4f} vs {ubm_production:.4f})")

print()
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print()
print(f"Evaluation results:")
print(f"  runs/evaluation/yolo26n_two_stage_on_test_set/")
print()
print(f"Plots:")
print(f"  - Confusion matrix")
print(f"  - F1 curve")
print(f"  - Precision-Recall curve")
print(f"  - Prediction examples")
print()
print("=" * 80)

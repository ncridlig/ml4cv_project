#!/usr/bin/env python3
"""
Evaluate our baseline model on FSOCO-12 TEST set.
This provides the fair comparison to Gabriele's baseline (mAP50 = 0.824 on test set).
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'baseline_test_set_evaluation'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("EVALUATING OUR BASELINE ON FSOCO-12 TEST SET")
print("=" * 70)
print()
print("Model: runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt")
print("Dataset: FSOCO-12 test set (689 images)")
print()
print("Comparison target:")
print("  Gabriele's baseline (from CV report): mAP50 = 0.824 on test set")
print()
print("=" * 70)
print()

# Load our baseline model
model_path = 'runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt'
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
    name='baseline_on_test_set',
)

# Print results
print("\n" + "=" * 70)
print("OUR BASELINE RESULTS ON FSOCO-12 TEST SET")
print("=" * 70)
print(f"mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"mAP50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()

print("COMPARISON TO GABRIELE'S BASELINE:")
print("-" * 70)
print(f"{'Metric':<15} {'Gabriele (Test)':<20} {'Ours (Test)':<20} {'Delta':<15}")
print("-" * 70)

our_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
our_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
our_precision = results.results_dict.get('metrics/precision(B)', 0)
our_recall = results.results_dict.get('metrics/recall(B)', 0)

gabriele_map50 = 0.824
gabriele_map50_95 = 0.570
gabriele_precision = 0.849
gabriele_recall = 0.765

print(f"{'mAP50':<15} {gabriele_map50:<20.4f} {our_map50:<20.4f} {(our_map50 - gabriele_map50):+.4f}")
print(f"{'mAP50-95':<15} {gabriele_map50_95:<20.4f} {our_map50_95:<20.4f} {(our_map50_95 - gabriele_map50_95):+.4f}")
print(f"{'Precision':<15} {gabriele_precision:<20.4f} {our_precision:<20.4f} {(our_precision - gabriele_precision):+.4f}")
print(f"{'Recall':<15} {gabriele_recall:<20.4f} {our_recall:<20.4f} {(our_recall - gabriele_recall):+.4f}")
print()

# Determine success
if our_map50 >= gabriele_map50:
    print("✅ SUCCESS: We matched or exceeded Gabriele's baseline!")
    print(f"   Our mAP50 ({our_map50:.4f}) >= Gabriele's ({gabriele_map50:.4f})")
elif our_map50 >= gabriele_map50 * 0.97:  # Within 3%
    print("⚠️ CLOSE: We're within 3% of Gabriele's baseline")
    print(f"   Gap: {(gabriele_map50 - our_map50):.4f} ({((gabriele_map50 - our_map50) / gabriele_map50 * 100):.1f}%)")
else:
    print("❌ GAP: We have work to do to match Gabriele's baseline")
    print(f"   Gap: {(gabriele_map50 - our_map50):.4f} ({((gabriele_map50 - our_map50) / gabriele_map50 * 100):.1f}%)")

print()
print("=" * 70)
print()
print("Results saved to: runs/evaluation/baseline_on_test_set/")
print("=" * 70)

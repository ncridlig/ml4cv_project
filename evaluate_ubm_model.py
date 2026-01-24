#!/usr/bin/env python3
"""
Evaluate UBM's official YOLOv11n model on FSOCO-12 dataset.
This will tell us the TRUE baseline that Gabriele/Patta achieved.
"""
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'UBM_official_yolov11n_evaluation'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

print("=" * 70)
print("EVALUATING UBM OFFICIAL YOLOv11n MODEL")
print("=" * 70)
print()
print("Model: /home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt")
print("Dataset: FSOCO-12 test set")
print("This is the model Gabriele/Patta trained and use on the actual car.")
print()
print("=" * 70)
print()

# Load UBM official model
model_path = '/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt'
model = YOLO(model_path)

print("Model loaded successfully")
print(f"Classes: {model.names}")
print()

# Evaluate on FSOCO-12 test set
print("Running evaluation on FSOCO-12 test set...")
print()

results = model.val(
    data='datasets/FSOCO-12/data.yaml',
    split='test',
    batch=32,  # Reduced from 64 to prevent OOM
    device=0,
    plots=True,
    save_json=True,
    project='runs/evaluation',
    name='ubm_official_on_fsoco12_test',
)

# Print results
print("\n" + "=" * 70)
print("UBM OFFICIAL MODEL RESULTS ON FSOCO-12 TEST SET")
print("=" * 70)
print(f"mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"mAP50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
print()

print("COMPARISON:")
print("-" * 70)
print(f"{'Metric':<15} {'UBM Official':<15} {'Our Baseline':<15} {'Delta':<15}")
print("-" * 70)

ubm_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
ubm_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
ubm_precision = results.results_dict.get('metrics/precision(B)', 0)
ubm_recall = results.results_dict.get('metrics/recall(B)', 0)

our_map50 = 0.714
our_map50_95 = 0.467
our_precision = 0.832
our_recall = 0.657

print(f"{'mAP50':<15} {ubm_map50:<15.4f} {our_map50:<15.4f} {(ubm_map50 - our_map50):+.4f}")
print(f"{'mAP50-95':<15} {ubm_map50_95:<15.4f} {our_map50_95:<15.4f} {(ubm_map50_95 - our_map50_95):+.4f}")
print(f"{'Precision':<15} {ubm_precision:<15.4f} {our_precision:<15.4f} {(ubm_precision - our_precision):+.4f}")
print(f"{'Recall':<15} {ubm_recall:<15.4f} {our_recall:<15.4f} {(ubm_recall - our_recall):+.4f}")
print()

print("THESIS BASELINE (from Edo's thesis, unknown dataset):")
print(f"  mAP50: 0.824 (UBM is {(ubm_map50 - 0.824):+.3f} from this)")
print()

print("=" * 70)
print()
print("Results saved to: runs/evaluation/ubm_official_on_fsoco12/")
print("=" * 70)

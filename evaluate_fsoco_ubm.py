#!/usr/bin/env python3
"""
Unified evaluation script for all models on fsoco-ubm test set.

Evaluates:
- YOLO26n
- YOLO26n (first-stage)
- YOLO26n (two-stage)
- YOLO12n
- YOLOv11n baseline
- YOLOv11n UBM production

fsoco-ubm is the in-house test set created from car camera data
(96 images from Rioveggio test track, November 2025).
"""
import os
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'fsoco-ubm-evaluation'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

# ============================================================================
# MODEL DICTIONARY - Define all models to evaluate
# ============================================================================
MODELS = {
    'YOLO26n': {
        'path': 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt',
        'description': '2025 architecture, 300 epochs on FSOCO-12',
        'results': {}  # Will be filled with evaluation results
    },
    'YOLO26n (first-stage)': {
        'path': 'runs/detect/runs/two-stage-yolo26/stage1_cone_detector_400ep2/weights/best.pt',
        'description': 'Pre-trained on cone-detector',
        'results': {}
    },
    'YOLO26n (two-stage)': {
        'path': 'runs/detect/runs/two-stage-yolo26/stage2b_full_finetune_250ep/weights/best.pt',
        'description': 'Pre-trained on cone-detector + fine-tuned on FSOCO-12',
        'results': {}
    },
    'YOLO12n': {
        'path': 'runs/detect/runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.pt',
        'description': '2025 architecture with attention mechanisms',
        'results': {}
    },
    'YOLOv11n (baseline)': {
        'path': 'runs/detect/runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt',
        'description': 'Our baseline model (300 epochs on FSOCO-12)',
        'results': {}
    },
    'UBM production': {
        'path': '/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt',
        'description': 'Current production model on the car',
        'results': {}
    }
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# First, download the dataset if not already present
DATASET_PATH = 'datasets/ml4cv_project-1'  # Default Roboflow download location

if not Path(DATASET_PATH).exists():
    print("=" * 80)
    print("fsoco-ubm DATASET NOT FOUND")
    print("=" * 80)
    print()
    print(f"Dataset not found at: {DATASET_PATH}")
    print()
    print("Please download the dataset first:")
    print("  python3 download_fsoco_ubm.py")
    print()
    print("=" * 80)
    exit(1)

DATA_YAML = f'{DATASET_PATH}/data.yaml'

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

print("=" * 80)
print("EVALUATING ALL MODELS ON fsoco-ubm TEST SET")
print("=" * 80)
print()
print(f"Dataset: {DATASET_PATH}")
print(f"Description: In-house UBM test set (96 images from car camera)")
print(f"Source: Rioveggio test track, November 20, 2025")
print(f"Camera: ZED 2i stereo (1280×720)")
print()
print(f"Models to evaluate: {len(MODELS)}")
for model_name, model_info in MODELS.items():
    status = "✅" if Path(model_info['path']).exists() else "❌ NOT FOUND"
    print(f"  - {model_name}: {status}")
print()
print("=" * 80)
print()

# Track which models are available
available_models = []
missing_models = []

for model_name, model_info in MODELS.items():
    model_path = model_info['path']

    # Check if model exists
    if not Path(model_path).exists():
        print(f"⚠️ SKIPPING {model_name}: Model not found at {model_path}")
        print()
        missing_models.append(model_name)
        continue

    available_models.append(model_name)

    # Evaluate model
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)
    print()
    print(f"Model path: {model_path}")
    print(f"Description: {model_info['description']}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    print(f"Model loaded successfully")
    print(f"Classes: {model.names}")
    print()

    # Set W&B name for this evaluation
    os.environ['WANDB_NAME'] = f'{model_name.replace(" ", "_")}_fsoco_ubm_eval'

    # Run evaluation
    print("Running evaluation on fsoco-ubm test set...")
    print()

    results = model.val(
        data=DATA_YAML,
        split='test',  # Use test split
        batch=64,
        conf=0.2,
        iou=0.45,
        device=0,
        plots=True,
        save_json=True,
        project='runs/evaluation',
        name=f'{model_name.replace(" ", "_")}_fsoco_ubm',
    )

    # Store results in dictionary
    model_info['results'] = {
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.results_dict.get('metrics/recall(B)', 0),
    }

    print()
    print(f"✅ {model_name} evaluation complete")
    print(f"   mAP50: {model_info['results']['mAP50']:.4f}")
    print(f"   Precision: {model_info['results']['precision']:.4f}")
    print(f"   Recall: {model_info['results']['recall']:.4f}")
    print()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print()
print("=" * 80)
print("FSOCO-UBM TEST SET RESULTS - ALL MODELS")
print("=" * 80)
print()

if missing_models:
    print("⚠️ Missing models (not evaluated):")
    for model_name in missing_models:
        print(f"  - {model_name}")
    print()

if not available_models:
    print("❌ ERROR: No models were available for evaluation!")
    print()
    print("Please ensure model weights exist at the specified paths.")
    exit(1)

# Print comparison table
print("Performance Comparison:")
print("-" * 80)
print(f"{'Model':<30} {'mAP50':<10} {'Precision':<12} {'Recall':<10} {'mAP50-95':<10}")
print("-" * 80)

# Sort by mAP50 (descending)
sorted_models = sorted(
    [(name, info) for name, info in MODELS.items() if name in available_models],
    key=lambda x: x[1]['results'].get('mAP50', 0),
    reverse=True
)

for model_name, model_info in sorted_models:
    results = model_info['results']
    print(f"{model_name:<30} "
          f"{results['mAP50']:<10.4f} "
          f"{results['precision']:<12.4f} "
          f"{results['recall']:<10.4f} "
          f"{results['mAP50-95']:<10.4f}")

print("-" * 80)
print()

# Analysis
if len(available_models) >= 2:
    best_model = sorted_models[0][0]
    best_map50 = sorted_models[0][1]['results']['mAP50']

    print("🏆 BEST MODEL (fsoco-ubm test set):")
    print(f"   {best_model}: {best_map50:.4f} mAP50")
    print()

    # Compare to others
    print("Performance vs Best:")
    for model_name, model_info in sorted_models[1:]:
        delta = model_info['results']['mAP50'] - best_map50
        delta_pct = (delta / best_map50) * 100
        print(f"   {model_name}: {delta:+.4f} ({delta_pct:+.1f}%)")
    print()

# Compare to FSOCO-12 test set performance (if available)
print("=" * 80)
print("FSOCO-12 vs fsoco-ubm COMPARISON")
print("=" * 80)
print()
print("Expected: fsoco-ubm may be more challenging (real car data)")
print()

# FSOCO-12 test set results (from previous evaluations)
fsoco12_results = {
    'YOLO26n (two-stage)': 0.7612,
    'YOLO26n': 0.7626,
    'YOLO26n (first-stage)': 0.7084,
    'YOLO12n': 0.7081,
    'YOLOv11n (baseline)': 0.7065,
    'UBM production': 0.6655,
}

print(f"{'Model':<30} {'FSOCO-12':<12} {'fsoco-ubm':<12} {'Delta':<15}")
print("-" * 80)

for model_name in available_models:
    if model_name in fsoco12_results:
        fsoco12_map50 = fsoco12_results[model_name]
        fsoco_ubm_map50 = MODELS[model_name]['results']['mAP50']
        delta = fsoco_ubm_map50 - fsoco12_map50
        delta_pct = (delta / fsoco12_map50) * 100

        print(f"{model_name:<30} "
              f"{fsoco12_map50:<12.4f} "
              f"{fsoco_ubm_map50:<12.4f} "
              f"{delta:+.4f} ({delta_pct:+.1f}%)")

print("-" * 80)
print()

# Key insights
print("KEY INSIGHTS:")
print()

# Check if rankings are preserved
fsoco12_ranking = ['YOLO26n', 'YOLO26n (first-stage)', 'YOLO12n', 'YOLOv11n (baseline)', 'UBM production']
fsoco_ubm_ranking = [name for name, _ in sorted_models if name in fsoco12_ranking]

if fsoco12_ranking == fsoco_ubm_ranking:
    print("✅ Model rankings PRESERVED between FSOCO-12 and fsoco-ubm")
    print("   This validates that performance translates to real-world data")
else:
    print("⚠️ Model rankings CHANGED between FSOCO-12 and fsoco-ubm")
    print("   Real-world conditions may favor different architectures")
    print()
    print(f"   FSOCO-12 ranking: {' > '.join(fsoco12_ranking)}")
    print(f"   fsoco-ubm ranking: {' > '.join(fsoco_ubm_ranking)}")

print()

# Average performance drop
if available_models:
    avg_drop = 0
    count = 0
    for model_name in available_models:
        if model_name in fsoco12_results:
            drop = MODELS[model_name]['results']['mAP50'] - fsoco12_results[model_name]
            avg_drop += drop
            count += 1

    if count > 0:
        avg_drop /= count
        avg_drop_pct = (avg_drop / 0.7) * 100  # Approximate baseline

        if avg_drop < -0.05:
            print(f"⚠️ Real-world data is MORE CHALLENGING: avg {avg_drop:.4f} ({avg_drop_pct:.1f}%)")
            print("   fsoco-ubm contains more difficult cases (motion blur, lighting, distance)")
        elif avg_drop > 0.05:
            print(f"✅ Real-world data is EASIER: avg {avg_drop:+.4f} ({avg_drop_pct:+.1f}%)")
            print("   Models generalize well to deployment conditions")
        else:
            print(f"✅ Real-world data SIMILAR to FSOCO-12: avg {avg_drop:+.4f} ({avg_drop_pct:+.1f}%)")
            print("   FSOCO-12 is a good proxy for real deployment")

print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print()
print("Results saved to: runs/evaluation/")
print()
print("Individual evaluation directories:")
for model_name in available_models:
    eval_dir = f'runs/evaluation/{model_name.replace(" ", "_")}_fsoco_ubm/'
    print(f"  - {model_name}: {eval_dir}")
print()
print("=" * 80)

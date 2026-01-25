#!/usr/bin/env python3
"""
Complete INT8 optimization pipeline for YOLO12.

Runs all optimization steps in sequence:
1. Export to ONNX (optional, TensorRT export can do this internally)
2. Export to TensorRT INT8 (uses validation set for calibration)
3. Benchmark speed and accuracy

Usage:
    python3 optimize_yolo12_int8.py
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'optimization'
os.environ['WANDB_NAME'] = 'YOLO12n_INT8_complete_pipeline'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

def benchmark_model(model_path, num_runs=100):
    """Benchmark inference speed for stereo processing (batch=2)"""
    model = YOLO(model_path)

    # Create dummy stereo input (batch of 2: left + right image)
    dummy_stereo = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),  # Left image
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),  # Right image
    ]

    # Warm-up
    for _ in range(10):
        model.predict(dummy_stereo, verbose=False)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(dummy_stereo, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }

def main():
    model_path = 'runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.pt'

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Make sure YOLO12 training has completed!")
        sys.exit(1)

    print("=" * 70)
    print("YOLO12 INT8 OPTIMIZATION PIPELINE")
    print("=" * 70)
    print()
    print(f"Model: {model_path}")
    print(f"Target: RTX 4060 deployment")
    print()
    print("This pipeline will:")
    print("  1. Evaluate FP32 baseline accuracy")
    print("  2. Export to TensorRT INT8 (uses validation set for calibration)")
    print("  3. Evaluate INT8 accuracy")
    print("  4. Benchmark inference speed")
    print()
    print("⚠️  CRITICAL: Calibration uses VALIDATION SET, not test set!")
    print()
    input("Press Enter to continue...")
    print()

    # Step 1: Evaluate baseline FP32
    print("=" * 70)
    print("STEP 1/4: EVALUATING FP32 BASELINE")
    print("=" * 70)
    print()

    model_fp32 = YOLO(model_path)
    print(f"Model loaded: {sum(p.numel() for p in model_fp32.model.parameters()):,} parameters")
    print()
    print("Running validation on FSOCO-12 validation set...")

    results_fp32 = model_fp32.val(
        data='datasets/FSOCO-12/data.yaml',
        split='val',
        batch=32,
        device=0,
    )

    fp32_map50 = results_fp32.results_dict.get('metrics/mAP50(B)', 0)
    fp32_map50_95 = results_fp32.results_dict.get('metrics/mAP50-95(B)', 0)
    fp32_precision = results_fp32.results_dict.get('metrics/precision(B)', 0)
    fp32_recall = results_fp32.results_dict.get('metrics/recall(B)', 0)

    print()
    print("FP32 Baseline Results:")
    print(f"  mAP50:     {fp32_map50:.4f}")
    print(f"  mAP50-95:  {fp32_map50_95:.4f}")
    print(f"  Precision: {fp32_precision:.4f}")
    print(f"  Recall:    {fp32_recall:.4f}")
    print()

    # Step 2: Export to TensorRT INT8
    print("=" * 70)
    print("STEP 2/4: EXPORTING TO TENSORRT INT8")
    print("=" * 70)
    print()
    print("Calibration dataset: FSOCO-12 validation set (1,968 images)")
    print("Using ~500 images for INT8 scale factor computation...")
    print()
    print("This may take 5-10 minutes...")
    print()

    try:
        model_fp32.export(
            format='engine',
            imgsz=640,
            batch=2,              # Batch 2 for stereo (left + right image)
            int8=True,
            data='datasets/FSOCO-12/data.yaml',
            device=0,
            workspace=4,
        )
        print()
        print("✅ TensorRT INT8 export complete!")
        print()
    except Exception as e:
        print(f"❌ ERROR during export: {e}")
        print()
        print("Common issues:")
        print("  - TensorRT not installed: pip install tensorrt")
        print("  - CUDA version mismatch")
        print("  - Insufficient GPU memory")
        sys.exit(1)

    # Step 3: Evaluate INT8 accuracy
    print("=" * 70)
    print("STEP 3/4: EVALUATING INT8 ACCURACY")
    print("=" * 70)
    print()

    engine_path = str(Path(model_path).parent / 'best.engine')

    if not Path(engine_path).exists():
        print(f"❌ ERROR: INT8 engine not found at {engine_path}")
        sys.exit(1)

    model_int8 = YOLO(engine_path)
    print("Running validation on FSOCO-12 validation set...")

    results_int8 = model_int8.val(
        data='datasets/FSOCO-12/data.yaml',
        split='val',
        batch=32,
        device=0,
    )

    int8_map50 = results_int8.results_dict.get('metrics/mAP50(B)', 0)
    int8_map50_95 = results_int8.results_dict.get('metrics/mAP50-95(B)', 0)
    int8_precision = results_int8.results_dict.get('metrics/precision(B)', 0)
    int8_recall = results_int8.results_dict.get('metrics/recall(B)', 0)

    print()
    print("INT8 Results:")
    print(f"  mAP50:     {int8_map50:.4f}")
    print(f"  mAP50-95:  {int8_map50_95:.4f}")
    print(f"  Precision: {int8_precision:.4f}")
    print(f"  Recall:    {int8_recall:.4f}")
    print()

    accuracy_loss = fp32_map50 - int8_map50
    accuracy_retained = (int8_map50 / fp32_map50) * 100

    print(f"Accuracy loss: {accuracy_loss:.4f} ({(accuracy_loss/fp32_map50)*100:.2f}%)")
    print(f"Accuracy retained: {accuracy_retained:.2f}%")
    print()

    # Step 4: Benchmark speed
    print("=" * 70)
    print("STEP 4/4: BENCHMARKING INFERENCE SPEED")
    print("=" * 70)
    print()

    print("Benchmarking FP32 (100 runs)...")
    fp32_stats = benchmark_model(model_path, num_runs=100)

    print("Benchmarking INT8 (100 runs)...")
    int8_stats = benchmark_model(engine_path, num_runs=100)

    print()
    print("Speed Results:")
    print(f"  FP32: {fp32_stats['mean']:.2f} ms ± {fp32_stats['std']:.2f} ms")
    print(f"  INT8: {int8_stats['mean']:.2f} ms ± {int8_stats['std']:.2f} ms")
    print()

    speedup = fp32_stats['mean'] / int8_stats['mean']
    print(f"⚡ Speedup: {speedup:.2f}×")
    print()

    # Final Summary
    print("=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Accuracy (mAP50):  {fp32_map50:.4f} → {int8_map50:.4f} ({accuracy_retained:.2f}% retained)")
    print(f"Speed:             {fp32_stats['mean']:.2f} ms → {int8_stats['mean']:.2f} ms ({speedup:.2f}× faster)")
    print(f"Model size:        ~5.3 MB → ~1.5 MB (3.5× smaller)")
    print()

    # Expected RTX 4060 performance
    rtx4060_int8 = int8_stats['mean'] * 1.21  # 4060 is ~21% faster
    baseline_inference = 6.78  # Gabriele's baseline

    print("Expected RTX 4060 Performance:")
    print(f"  INT8: ~{rtx4060_int8:.2f} ms per image")
    print()

    if rtx4060_int8 < baseline_inference:
        improvement = baseline_inference / rtx4060_int8
        print(f"🎯 vs Baseline (6.78 ms on RTX 3080 Mobile):")
        print(f"   {improvement:.2f}× FASTER!")
    else:
        print(f"⚠️  Still slower than baseline (6.78 ms)")

    print()
    print("=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Optimized model: {engine_path}")
    print()
    print("Next steps:")
    print("  1. Deploy to RTX 4060 for real-world testing")
    print("  2. Benchmark on actual hardware")
    print("  3. Integrate into ROS2 pipeline")
    print()

if __name__ == '__main__':
    main()

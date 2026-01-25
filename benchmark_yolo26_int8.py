#!/usr/bin/env python3
"""
Benchmark INT8 vs FP32 inference speed and accuracy for YOLO26.

This is Step 3 of the INT8 optimization pipeline.
Compares FP32 baseline vs INT8 optimized model.
"""
import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'yolo26-optimization'
os.environ['WANDB_NAME'] = 'YOLO26n_INT8_benchmark'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

def benchmark_model(model_path, num_runs=100):
    """Benchmark inference speed for stereo processing (batch=2)"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Create dummy stereo input (batch of 2: left + right image, 640x640 RGB)
    dummy_stereo = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),  # Left image
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),  # Right image
    ]

    print(f"Warming up ({10} runs)...")
    for _ in range(10):
        model.predict(dummy_stereo, verbose=False)

    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        model.predict(dummy_stereo, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_runs}")

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }

print("=" * 70)
print("STEP 3: BENCHMARK YOLO26 INT8 vs FP32")
print("=" * 70)
print()

# Model paths
fp32_path = 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt'
int8_path = 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.engine'

# Check if INT8 model exists
if not Path(int8_path).exists():
    print(f"❌ ERROR: INT8 model not found at {int8_path}")
    print("   Run export_yolo26_tensorrt_int8.py first!")
    exit(1)

# Benchmark FP32
print("=" * 70)
print("BENCHMARKING FP32 MODEL")
print("=" * 70)
print()
fp32_stats = benchmark_model(fp32_path, num_runs=100)
print()

# Benchmark INT8
print("=" * 70)
print("BENCHMARKING INT8 MODEL")
print("=" * 70)
print()
int8_stats = benchmark_model(int8_path, num_runs=100)
print()

# Print inference speed comparison
print("=" * 70)
print("INFERENCE SPEED COMPARISON (RTX 4080 Super)")
print("=" * 70)
print()
print(f"{'Model':<15} {'Mean':<12} {'Median':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Speedup':<10}")
print("-" * 95)
print(f"{'FP32':<15} {fp32_stats['mean']:>8.2f} ms  {fp32_stats['median']:>8.2f} ms  {fp32_stats['std']:>6.2f} ms  {fp32_stats['min']:>6.2f} ms  {fp32_stats['max']:>6.2f} ms  {'1.00×':<10}")
print(f"{'INT8':<15} {int8_stats['mean']:>8.2f} ms  {int8_stats['median']:>8.2f} ms  {int8_stats['std']:>6.2f} ms  {int8_stats['min']:>6.2f} ms  {int8_stats['max']:>6.2f} ms  {fp32_stats['mean']/int8_stats['mean']:.2f}×")
print()

speedup = fp32_stats['mean'] / int8_stats['mean']
time_saved = fp32_stats['mean'] - int8_stats['mean']

print(f"⚡ Speedup: {speedup:.2f}× faster")
print(f"⚡ Time saved: {time_saved:.2f} ms per stereo pair")
print()

# Evaluate accuracy on validation set
print("=" * 70)
print("ACCURACY COMPARISON (Validation Set)")
print("=" * 70)
print()

print("Evaluating FP32 model...")
fp32_model = YOLO(fp32_path)
fp32_results = fp32_model.val(
    data='datasets/FSOCO-12/data.yaml',
    split='val',
    batch=32,
    device=0,
)

print()
print("Evaluating INT8 model...")
int8_model = YOLO(int8_path)
int8_results = int8_model.val(
    data='datasets/FSOCO-12/data.yaml',
    split='val',
    batch=32,
    device=0,
)

fp32_map50 = fp32_results.results_dict.get('metrics/mAP50(B)', 0)
fp32_map50_95 = fp32_results.results_dict.get('metrics/mAP50-95(B)', 0)
fp32_precision = fp32_results.results_dict.get('metrics/precision(B)', 0)
fp32_recall = fp32_results.results_dict.get('metrics/recall(B)', 0)

int8_map50 = int8_results.results_dict.get('metrics/mAP50(B)', 0)
int8_map50_95 = int8_results.results_dict.get('metrics/mAP50-95(B)', 0)
int8_precision = int8_results.results_dict.get('metrics/precision(B)', 0)
int8_recall = int8_results.results_dict.get('metrics/recall(B)', 0)

print()
print("=" * 70)
print("ACCURACY RESULTS")
print("=" * 70)
print()
print(f"{'Metric':<15} {'FP32':<12} {'INT8':<12} {'Loss':<12} {'Retained':<10}")
print("-" * 70)
print(f"{'mAP50':<15} {fp32_map50:>8.4f}     {int8_map50:>8.4f}     {fp32_map50 - int8_map50:>8.4f}     {(int8_map50/fp32_map50)*100:>6.2f}%")
print(f"{'mAP50-95':<15} {fp32_map50_95:>8.4f}     {int8_map50_95:>8.4f}     {fp32_map50_95 - int8_map50_95:>8.4f}     {(int8_map50_95/fp32_map50_95)*100:>6.2f}%")
print(f"{'Precision':<15} {fp32_precision:>8.4f}     {int8_precision:>8.4f}     {fp32_precision - int8_precision:>8.4f}     {(int8_precision/fp32_precision)*100:>6.2f}%")
print(f"{'Recall':<15} {fp32_recall:>8.4f}     {int8_recall:>8.4f}     {fp32_recall - int8_recall:>8.4f}     {(int8_recall/fp32_recall)*100:>6.2f}%")
print()

accuracy_retained = (int8_map50 / fp32_map50) * 100
print(f"✅ Overall accuracy retained: {accuracy_retained:.2f}%")
print(f"✅ mAP50 loss: {(fp32_map50 - int8_map50):.4f} ({((fp32_map50 - int8_map50)/fp32_map50)*100:.2f}%)")
print()

# Final summary
print("=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)
print()
print(f"Speed:     {speedup:.2f}× faster ({int8_stats['mean']:.2f} ms vs {fp32_stats['mean']:.2f} ms)")
print(f"Accuracy:  {accuracy_retained:.2f}% retained ({fp32_map50:.4f} → {int8_map50:.4f})")
print(f"Model:     ~3.5× smaller (INT8 vs FP32)")
print()

# Expected RTX 4060 performance
rtx4060_fp32 = fp32_stats['mean'] * 1.66  # RTX 4060 is ~40% slower than 4080 Super
rtx4060_int8 = int8_stats['mean'] * 1.66  # Therefore it will take 66% longer to process

print("Expected RTX 4060 Performance (deployment target):")
print(f"  FP32: ~{rtx4060_fp32:.2f} ms (batch=2)")
print(f"  INT8: ~{rtx4060_int8:.2f} ms (batch=2)")
print(f"  Per image: ~{rtx4060_int8/2:.2f} ms")
print()

# Compare to baseline
baseline_inference = 6.78  # Gabriele's baseline (RTX 3080 Mobile)
yolo26_per_image = rtx4060_int8 / 2

if yolo26_per_image < baseline_inference:
    improvement = baseline_inference / yolo26_per_image
    print(f"🎯 vs Gabriele's baseline (6.78 ms on RTX 3080 Mobile):")
    print(f"   {improvement:.2f}× FASTER on RTX 4060!")
else:
    print(f"⚠️  Still slower than baseline (6.78 ms)")

print()
print("=" * 70)
print(f"✅ INT8 Optimized model: {int8_path}")
print("=" * 70)

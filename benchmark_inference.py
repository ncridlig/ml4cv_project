#!/usr/bin/env python3
"""
Benchmark inference speed for ONNX models on RTX 4080 Super.

This measures real-world deployment performance for the stereo camera pipeline.

Usage:
    python benchmark_inference.py --model path/to/model.onnx --runs 100
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

def benchmark_onnx_model(model_path: str, num_runs: int = 100, batch_size: int = 2):
    """
    Benchmark ONNX model inference speed.

    Args:
        model_path: Path to ONNX model
        num_runs: Number of inference runs for averaging
        batch_size: Batch size (2 for stereo, 1 for single image)
    """
    print("=" * 70)
    print("ONNX INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Runs: {num_runs}")
    print(f"Device: CUDA (RTX 4080 Super)")
    print()

    # Load ONNX model
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, session_options, providers=providers)

    # Get input shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print()

    # Create dummy input (batch_size, 3, 640, 640)
    if batch_size:
        input_shape[0] = batch_size
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    print("Warming up (10 runs)...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark runs
    print(f"Benchmarking ({num_runs} runs)...")
    times = []

    for i in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50_time = np.percentile(times, 50)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mean:   {mean_time:.2f} ms")
    print(f"Std:    {std_time:.2f} ms")
    print(f"Min:    {min_time:.2f} ms")
    print(f"Max:    {max_time:.2f} ms")
    print(f"P50:    {p50_time:.2f} ms")
    print(f"P95:    {p95_time:.2f} ms")
    print(f"P99:    {p99_time:.2f} ms")
    print()
    print(f"FPS (single image): {1000/mean_time:.1f} fps")
    if batch_size == 2:
        print(f"FPS (stereo pair):  {1000/(mean_time):.1f} fps (both images processed together)")
        print(f"Per-image time:     {mean_time/2:.2f} ms")
    print()

    # Real-time capability
    target_fps = 60
    target_time = 1000 / target_fps

    if batch_size == 2:
        # For stereo, we process both images in one forward pass
        capable = mean_time < target_time
    else:
        capable = mean_time < target_time

    print("REAL-TIME CAPABILITY:")
    print(f"  Target: {target_fps} fps ({target_time:.2f} ms per frame)")
    print(f"  Actual: {mean_time:.2f} ms")
    if capable:
        print(f"  ✅ CAPABLE (headroom: {target_time - mean_time:.2f} ms)")
    else:
        print(f"  ❌ NOT CAPABLE (too slow by: {mean_time - target_time:.2f} ms)")

    print()
    print("=" * 70)

    # Comparison to thesis baseline
    print("COMPARISON TO EDO'S THESIS BASELINE:")
    print("  Thesis (RTX 3080 Mobile, TensorRT): 6.78 ms")
    print(f"  Ours (RTX 4080 Super, ONNX):        {mean_time:.2f} ms")
    if mean_time < 6.78:
        print(f"  ✅ FASTER by {6.78 - mean_time:.2f} ms")
    else:
        print(f"  ⚠️ SLOWER by {mean_time - 6.78:.2f} ms (TensorRT would be faster)")
    print()
    print("Note: TensorRT conversion will further improve speed (~30-50% faster)")
    print("=" * 70)

    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'p50': p50_time,
        'p95': p95_time,
        'p99': p99_time,
        'fps': 1000 / mean_time,
        'realtime_capable': capable
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark ONNX model inference speed')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--batch', type=int, default=2, help='Batch size (2 for stereo)')

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    try:
        results = benchmark_onnx_model(str(model_path), args.runs, args.batch)
        return 0
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

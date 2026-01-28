#!/usr/bin/env python3
"""
Find optimal confidence threshold for best model on fsoco-ubm.
Tests multiple thresholds and reports best F1 score.

Usage:
    python3 optimize_confidence_threshold.py

    # Or specify custom model
    python3 optimize_confidence_threshold.py --model runs/detect/.../weights/best.pt
"""
import os
import argparse
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = 'fsoco-ubm-threshold-tuning'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

def main():
    parser = argparse.ArgumentParser(description='Optimize confidence threshold for best model')
    parser.add_argument('--model', type=str,
                        default='runs/detect/runs/two-stage-yolo26/stage2b_full_finetune_250ep/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--data', type=str,
                        default='datasets/ml4cv_project-1/data.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--min-conf', type=float, default=0.1,
                        help='Minimum confidence threshold to test')
    parser.add_argument('--max-conf', type=float, default=0.9,
                        help='Maximum confidence threshold to test')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Step size for threshold search')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size for validation')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device (0 for GPU, -1 for CPU)')

    args = parser.parse_args()

    # Load model
    print("=" * 80)
    print("CONFIDENCE THRESHOLD OPTIMIZATION ON fsoco-ubm")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data} (test split)")
    print()

    model = YOLO(args.model)

    # Generate threshold range
    thresholds = np.arange(args.min_conf, args.max_conf + args.step, args.step)

    print(f"Testing {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print()
    print("-" * 80)
    print(f"{'Conf':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP50':<10}")
    print("-" * 80)

    best_f1 = 0
    best_conf = 0.25
    results_table = []

    # Test each threshold
    for conf in thresholds:
        results = model.val(
            data=args.data,
            split='test',
            batch=args.batch,
            conf=conf,  # Override confidence threshold
            verbose=False,  # Suppress output
            plots=False,  # Skip plot generation
            device=args.device,
        )

        precision = results.results_dict.get('metrics/precision(B)', 0)
        recall = results.results_dict.get('metrics/recall(B)', 0)
        map50 = results.results_dict.get('metrics/mAP50(B)', 0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_table.append({
            'conf': conf,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map50': map50,
        })

        marker = "🏆" if f1 > best_f1 else "  "
        print(f"{marker} {conf:<6.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {map50:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf

    print("-" * 80)
    print()

    # Report optimal threshold
    print("=" * 80)
    print("OPTIMAL THRESHOLD")
    print("=" * 80)
    print(f"🏆 Best Confidence: {best_conf:.2f}")
    print(f"   F1 Score: {best_f1:.4f}")
    print()

    # Get detailed stats for optimal threshold
    optimal_result = next(r for r in results_table if r['conf'] == best_conf)
    print(f"   Precision: {optimal_result['precision']:.4f}")
    print(f"   Recall: {optimal_result['recall']:.4f}")
    print(f"   mAP50: {optimal_result['map50']:.4f}")
    print()

    # Compare to default threshold (0.25)
    default_result = next((r for r in results_table if abs(r['conf'] - 0.25) < 0.01), None)
    if default_result and best_conf != 0.25:
        f1_improvement = ((best_f1 - default_result['f1']) / default_result['f1']) * 100
        print("Improvement over default (conf=0.25):")
        print(f"   F1: {default_result['f1']:.4f} → {best_f1:.4f} ({f1_improvement:+.1f}%)")
        print()

    # Run final validation with optimal threshold to generate plots
    print("=" * 80)
    print("FINAL VALIDATION WITH OPTIMAL THRESHOLD")
    print("=" * 80)
    print()
    print(f"Running validation with conf={best_conf:.2f}...")
    print()

    # Set W&B name
    os.environ['WANDB_NAME'] = f'optimal_conf_{best_conf:.2f}_fsoco_ubm'

    final_results = model.val(
        data=args.data,
        split='test',
        batch=args.batch,
        conf=best_conf,
        plots=True,  # Generate plots
        save_json=True,
        project='runs/evaluation',
        name=f'best_model_optimized_conf_{best_conf:.2f}',
        device=args.device,
    )

    print()
    print(f"✅ Results saved to: runs/evaluation/best_model_optimized_conf_{best_conf:.2f}/")
    print()

    # Deployment recommendations
    print("=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)
    print()
    print("Update inference code with optimal threshold:")
    print()
    print("Python (Ultralytics):")
    print(f"    results = model.predict(source='...', conf={best_conf:.2f})")
    print()
    print("C++ (ROS2 node):")
    print(f"    float confidence_threshold = {best_conf:.2f}f;")
    print()
    print("ONNX/TensorRT:")
    print("    Apply threshold during postprocessing (filter detections)")
    print()
    print("=" * 80)

    # Save results to file
    output_file = f'runs/evaluation/optimal_conf_{best_conf:.2f}_results.txt'
    os.makedirs('runs/evaluation', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONFIDENCE THRESHOLD OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data}\n\n")
        f.write(f"Optimal Confidence: {best_conf:.2f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {optimal_result['precision']:.4f}\n")
        f.write(f"Recall: {optimal_result['recall']:.4f}\n")
        f.write(f"mAP50: {optimal_result['map50']:.4f}\n\n")

        f.write("All Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Conf':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP50':<10}\n")
        f.write("-" * 80 + "\n")
        for r in results_table:
            marker = "*" if r['conf'] == best_conf else " "
            f.write(f"{marker} {r['conf']:<6.2f} {r['precision']:<12.4f} {r['recall']:<12.4f} "
                   f"{r['f1']:<12.4f} {r['map50']:<10.4f}\n")
        f.write("-" * 80 + "\n")

    print(f"Results saved to: {output_file}")
    print()

if __name__ == '__main__':
    main()

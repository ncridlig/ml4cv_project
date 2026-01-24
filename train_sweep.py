#!/usr/bin/env python3
"""
Training script for W&B hyperparameter sweep.
Automatically picks up sweep parameters from W&B.
"""
import os
import gc
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
import wandb

# Load environment variables
load_dotenv()

# Set wandb API key
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

def train():
    # Initialize wandb run - sweep will inject config
    run = wandb.init()

    # Get hyperparameters from sweep
    config = wandb.config

    print(f"\n{'='*60}")
    print(f"Starting training with sweep config:")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Initialize model
    model = YOLO('yolo11n.pt')

    # Train with sweep parameters
    results = model.train(
        # Dataset
        data=config.data,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,

        # Device
        device=0,
        workers=12,  # Reduced from 16 (memory leak between runs)

        # Project organization
        project='runs/sweep',
        name=f'sweep_{run.id}',
        exist_ok=False,

        # Learning rate
        lr0=config.lr0,
        lrf=config.lrf,
        warmup_epochs=config.warmup_epochs,

        # Augmentation
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        mosaic=config.mosaic,
        close_mosaic=config.close_mosaic,
        mixup=config.mixup,
        copy_paste=config.copy_paste,
        degrees=config.degrees,

        # Regularization
        weight_decay=config.weight_decay,
        dropout=config.dropout,

        # Logging
        plots=True,
        verbose=True,

        # Save settings
        save=True,
        save_period=-1,  # Only save last and best
    )

    # Log final metrics to wandb
    final_metrics = {
        'final/mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'final/mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'final/precision': results.results_dict.get('metrics/precision(B)', 0),
        'final/recall': results.results_dict.get('metrics/recall(B)', 0),
    }
    wandb.log(final_metrics)

    print(f"\n{'='*60}")
    print(f"Training complete:")
    print(f"  mAP50: {final_metrics['final/mAP50']:.4f}")
    print(f"  mAP50-95: {final_metrics['final/mAP50-95']:.4f}")
    print(f"  Precision: {final_metrics['final/precision']:.4f}")
    print(f"  Recall: {final_metrics['final/recall']:.4f}")
    print(f"{'='*60}\n")

    # Finish run
    wandb.finish()

    # Explicit cleanup to prevent memory leaks between runs
    del model
    del results
    torch.cuda.empty_cache()
    gc.collect()

    print("Memory cleanup completed\n")

if __name__ == "__main__":
    train()

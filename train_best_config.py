#!/usr/bin/env python3
"""
Train final model with best hyperparameters from sweep.

After running analyze_sweep.py, paste the best config here and train for 300 epochs.
"""
import os
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Set wandb environment variables
os.environ['WANDB_ENTITY'] = 'ncridlig-ml4cv'
os.environ['WANDB_PROJECT'] = '3CFU'
os.environ['WANDB_NAME'] = 'yolov11n_best_config_300ep'
wandb_key = os.getenv('WAND_DB_API_KEY')
if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key

# TODO: Paste best config from analyze_sweep.py here
# Example:
best_config = {
    'batch': 64,
    'close_mosaic': 10,
    'copy_paste': 0.15,
    'data': 'datasets/FSOCO-12/data.yaml',
    'degrees': 5.0,
    'dropout': 0.1,
    'epochs': 300,  # Override to 300 for final training
    'hsv_h': 0.02,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'imgsz': 640,
    'lr0': 0.01,
    'lrf': 0.05,
    'mixup': 0.1,
    'mosaic': 0.9,
    'warmup_epochs': 3,
    'weight_decay': 0.0005,
}

# Override epochs to 300 for final training
best_config['epochs'] = 300

print("=" * 70)
print("FINAL TRAINING WITH BEST HYPERPARAMETERS")
print("=" * 70)
print("\nConfiguration:")
for key, value in sorted(best_config.items()):
    print(f"  {key:20s}: {value}")
print()
print("Starting training...")
print("=" * 70)
print()

# Initialize model
model = YOLO('yolo11n.pt')

# Train with best config
results = model.train(
    # Dataset
    data=best_config['data'],
    epochs=best_config['epochs'],
    imgsz=best_config['imgsz'],
    batch=best_config['batch'],

    # Device
    device=0,
    workers=16,

    # Project
    project='runs/best_config',
    name='yolov11n_best_300ep',
    exist_ok=False,

    # Learning rate
    lr0=best_config['lr0'],
    lrf=best_config['lrf'],
    warmup_epochs=best_config['warmup_epochs'],

    # Augmentation
    hsv_h=best_config['hsv_h'],
    hsv_s=best_config['hsv_s'],
    hsv_v=best_config['hsv_v'],
    mosaic=best_config['mosaic'],
    close_mosaic=best_config['close_mosaic'],
    mixup=best_config['mixup'],
    copy_paste=best_config['copy_paste'],
    degrees=best_config['degrees'],

    # Regularization
    weight_decay=best_config['weight_decay'],
    dropout=best_config['dropout'],

    # Logging
    plots=True,
    verbose=True,

    # Save
    save=True,
    save_period=50,
)

# Print results
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A')}")
print()
print("Baseline comparison:")
print(f"  Baseline mAP50:  0.714")
print(f"  Best config:     {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
improvement = ((results.results_dict.get('metrics/mAP50(B)', 0) - 0.714) / 0.714) * 100
print(f"  Improvement:     {improvement:+.1f}%")
print("=" * 70)

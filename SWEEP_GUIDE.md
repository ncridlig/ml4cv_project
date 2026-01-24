# Hyperparameter Sweep Guide

## Overview

This sweep will automatically search for better hyperparameters to improve on our baseline (mAP50 = 0.714).

**Method:** Bayesian optimization (smarter than grid search)
**Search space:** 13 hyperparameters
**Epochs per run:** 100 (faster evaluation)
**Early termination:** Hyperband (stops poor runs early)

---

## Quick Start

### 1. Launch the Sweep

```bash
chmod +x launch_sweep.sh
./launch_sweep.sh
```

This will:
1. Create a W&B sweep with the configuration in `sweep_config.yaml`
2. Start an agent that continuously runs training jobs
3. Each job gets different hyperparameters from W&B
4. Results are logged automatically

### 2. Monitor Progress

**W&B Dashboard:** https://wandb.ai/ncridlig-ml4cv/runs-sweep

The dashboard shows:
- Parallel coordinates plot (best hyperparameter combinations)
- Metric evolution over time
- Hyperparameter importance ranking
- Real-time training progress

### 3. Stop the Sweep

Press **Ctrl+C** in the terminal running the sweep agent.

The sweep will pause - you can resume it later by running:
```bash
wandb agent ncridlig-ml4cv/runs-sweep/<SWEEP_ID>
```

---

## Search Space

### Learning Rate Parameters
- `lr0`: 0.005 - 0.02 (initial learning rate)
- `lrf`: 0.01 - 0.1 (final LR multiplier)
- `warmup_epochs`: [1, 3, 5]

### Augmentation Parameters
- `hsv_h`: 0.0 - 0.03 (hue jitter)
- `hsv_s`: 0.5 - 0.9 (saturation jitter)
- `hsv_v`: 0.3 - 0.6 (value jitter)
- `mosaic`: 0.5 - 1.0 (mosaic augmentation probability)
- `close_mosaic`: [0, 5, 10, 15, 20] (epoch to stop mosaic)
- `mixup`: 0.0 - 0.3 (mixup augmentation)
- `copy_paste`: 0.0 - 0.3 (copy-paste augmentation)
- `degrees`: 0.0 - 10.0 (rotation degrees)

### Regularization
- `weight_decay`: 0.0001 - 0.001
- `dropout`: 0.0 - 0.2

---

## Timeline & Resource Planning

### Per Training Run
- Duration: ~45 minutes (100 epochs with batch=64)
- GPU: RTX 4080 Super fully utilized
- Disk: ~500MB per run (weights + logs)

### Sweep Estimates

| Runs | Duration | Expected Improvement |
|------|----------|---------------------|
| 10 runs | ~7.5 hours | Find decent config |
| 20 runs | ~15 hours | High confidence best config |
| 30 runs | ~22 hours | Diminishing returns |

**Recommendation:** Run 15-20 sweeps overnight (~15 hours)

---

## Strategy

### Phase 1: Sweep Search (15 hours, 20 runs)
1. Launch sweep overnight
2. Let it run for 20 iterations
3. W&B will explore hyperparameter space intelligently

### Phase 2: Best Config Full Training (4 hours)
1. Identify best config from sweep
2. Train for full 300 epochs with best hyperparameters
3. This becomes your improved baseline

### Phase 3: Architecture Comparison (15 hours)
1. Use best hyperparameters found
2. Train YOLOv11s and YOLOv11m with same config
3. Compare against baseline

---

## How Bayesian Optimization Works

Unlike grid search (tries every combination), Bayesian optimization:
1. **Starts with random configs** (first 5-10 runs)
2. **Builds a model** of hyperparameter → performance relationship
3. **Picks next config** that's most likely to improve
4. **Learns from results** and updates the model
5. **Converges quickly** to good regions of search space

This is 5-10x more efficient than grid search!

---

## Early Termination (Hyperband)

Poor runs are stopped early to save time:
- If a run is performing badly at epoch 30, it's terminated
- Resources are redirected to promising configs
- This allows testing more hyperparameter combinations

---

## Interpreting Results

### W&B Sweep Dashboard Features

1. **Parallel Coordinates Plot**
   - Shows relationship between hyperparameters and mAP50
   - Hover over lines to see configs
   - Best runs are highlighted in color

2. **Hyperparameter Importance**
   - Ranked list of which params matter most
   - Focus on tuning top 3-5 important params

3. **Best Run**
   - W&B automatically tracks the best run
   - Click to see full training curves and config

### Success Criteria

**Minimum Target:** mAP50 ≥ 0.75 (+5% over baseline)
**Good Target:** mAP50 ≥ 0.78 (+9% over baseline)
**Excellent Target:** mAP50 ≥ 0.80 (+12% over baseline)

---

## After the Sweep

### 1. Extract Best Configuration

```bash
# Get best run details from W&B
python analyze_sweep.py
```

This will print the best hyperparameters found.

### 2. Full Training with Best Config

```bash
# Train for 300 epochs with best config
python train_best_config.py
```

### 3. Compare Against Baseline

Use W&B to compare:
- Baseline (100 epochs): mAP50 = 0.714 (estimated from 300 epoch run)
- Best sweep run (100 epochs): mAP50 = ???
- Best config full (300 epochs): mAP50 = ???

---

## Troubleshooting

### Sweep Agent Stops
If the agent crashes:
```bash
# Resume the sweep
wandb agent ncridlig-ml4cv/runs-sweep/<SWEEP_ID>
```

### Out of Memory
If you get OOM errors, reduce batch size:
- Edit `sweep_config.yaml`, change `batch: value: 64` to `batch: value: 48`
- Recreate sweep

### Runs Too Slow
If 100 epochs is still too slow:
- Edit `sweep_config.yaml`, change `epochs: value: 100` to `epochs: value: 50`
- Quality: 50 epochs may not fully converge, but still useful for comparison

### All Runs Similar Performance
This suggests:
- Search space may not include the key parameter
- Need to widen ranges (e.g., try lr0: 0.001 - 0.05)
- Or the dataset/architecture is the limiting factor

---

## Advanced: Manual Sweep Control

### Create Sweep Without Starting Agent
```bash
wandb sweep sweep_config.yaml
# Note the sweep ID, but don't start agent yet
```

### Start Multiple Agents (Parallel Training)
If you have multiple GPUs:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID>
```

### Pause and Resume
- Pause: Ctrl+C
- Resume: `wandb agent <SWEEP_ID>`
- The sweep remembers all previous runs

---

## Alternative: Quick Manual Grid Search

If W&B sweeps seem complex, here's a simpler alternative:

```bash
# Test 3 learning rates manually
python train_baseline.py --epochs 100 --lr0 0.005 --name lr_0.005
python train_baseline.py --epochs 100 --lr0 0.01 --name lr_0.01
python train_baseline.py --epochs 100 --lr0 0.02 --name lr_0.02

# Pick best, then test augmentation
python train_baseline.py --epochs 100 --lr0 0.01 --mosaic 0.7 --name mosaic_0.7
python train_baseline.py --epochs 100 --lr0 0.01 --mosaic 1.0 --name mosaic_1.0
```

But W&B sweeps are more systematic and efficient!

---

## Files

- `sweep_config.yaml` - Hyperparameter search space definition
- `train_sweep.py` - Training script that reads W&B sweep config
- `launch_sweep.sh` - One-command sweep launcher
- `SWEEP_GUIDE.md` - This file

---

## Expected Outcome

After 15-20 hours of sweep:
1. ✅ Best hyperparameter configuration identified
2. ✅ Performance improvement quantified (hopefully mAP50 > 0.75)
3. ✅ Understanding of which parameters matter most
4. ✅ Ready to train final model with best config
5. ✅ Knowledge transfers to future model architecture experiments

Good luck! 🚀

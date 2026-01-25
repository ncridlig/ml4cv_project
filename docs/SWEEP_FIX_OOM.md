# Sweep OOM Fix - 2026-01-24

## Problem

Sweep crashed after 33 minutes due to Out Of Memory (OOM) killer.

**Evidence:**
```
[141620.867156] oom-kill:constraint=CONSTRAINT_NONE
[141620.867403] Out of memory: Killed process 1413497 (python)
                total-vm:50344988kB  (~50GB!)
```

## Root Cause

**Combination of high memory usage:**
- Batch size: 64 images
- Workers: 16 parallel data loaders
- Augmentation processing
- **Total memory usage: 50GB** (system has only 30GB RAM)

## Solution

### Changes Made

1. **Reduced batch size: 64 → 32**
   - File: `sweep_config.yaml` line 94
   - Impact: ~50% reduction in VRAM and batch memory
   - Trade-off: Training will be slightly slower (~10%)

2. **Reduced workers: 16 → 8**
   - File: `train_sweep.py` line 46
   - Impact: ~50% reduction in data loading memory
   - Trade-off: Slightly slower data loading, but not bottleneck

### Expected Memory Usage

**Before (crashed):**
- Batch 64 + Workers 16 = ~50GB peak
- Exceeded 30GB RAM + 8GB swap

**After (fixed):**
- Batch 32 + Workers 8 = ~15-20GB peak
- Well within 30GB RAM limit
- Safe margin for system overhead

## Performance Impact

### Training Speed

**Before:**
- ~3 min/epoch (when it didn't crash)

**After:**
- ~4-5 min/epoch (estimated)
- Still reasonable for 100-epoch runs
- 100 epochs = ~7-8 hours (vs 5 hours before)

### Quality Impact

**None expected:**
- Smaller batch size may actually improve generalization
- Same number of samples seen over 100 epochs
- Hyperparameter search still effective

## How to Restart Sweep

### Check Sweep Status

```bash
source venv/bin/activate
wandb sweep list
```

Look for your sweep ID (something like `nmzd9rqk`)

### Resume Sweep

```bash
./launch_sweep.sh
```

This will:
1. Pick up the existing sweep (sweep-nmzd9rqk)
2. Use the fixed batch=32, workers=8 settings
3. Continue from where it left off

**OR manually:**

```bash
source venv/bin/activate
wandb agent ncridlig-ml4cv/runs-sweep/nmzd9rqk
```

## Monitoring

### Check Memory Usage During Training

Terminal 1 (run sweep):
```bash
./launch_sweep.sh
```

Terminal 2 (monitor memory):
```bash
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

Look for:
- RAM usage staying below 25GB
- GPU VRAM usage staying below 14GB
- No swap usage increase

### If Still OOM

If it crashes again, further reduce batch size:

```bash
# Edit sweep_config.yaml
batch:
  value: 24  # Even more conservative
```

Then restart sweep.

## Alternative: Reduce Epochs

If time is critical and training is too slow:

```bash
# Edit sweep_config.yaml
epochs:
  value: 50  # Reduced from 100 for faster evaluation
```

50 epochs is enough to see hyperparameter impact, though less converged.

## Sweep Configuration Summary

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Batch | 64 | 32 | OOM: 50GB → 20GB |
| Workers | 16 | 8 | OOM: reduce data loading memory |
| Epochs | 100 | 100 | Unchanged |

## Expected Timeline

**With batch=32, workers=8:**
- Time per run: ~7-8 hours (100 epochs)
- Target runs: 15-20 runs
- Total time: ~5-7 days if run sequentially

**Recommendation:**
- Run 10 sweeps (2-3 days) for good coverage
- Analyze results and pick best config
- Use best config for final 300-epoch training

---

**Date:** 2026-01-24 05:30 AM
**Status:** Fixed, ready to restart sweep
**Files Modified:**
- `sweep_config.yaml` (batch: 64 → 32)
- `train_sweep.py` (workers: 16 → 8)

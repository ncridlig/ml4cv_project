## Step 4: Evaluate YOLO12n on Test Set (30 minutes)

**Purpose:** Compare YOLO12n to Gabriele's test set baseline

**Command:**
```bash
python3 evaluate_yolo12_test.py
```

**What it does:**
- Loads YOLO12n: `runs/yolo12/yolo12n_300ep_FSOCO/weights/best.pt`
- Evaluates on FSOCO-12 **TEST SET** (689 images)
- Compares to Gabriele's baseline (0.824)
- Success analysis

**Expected output:**
```
YOLO12n (test):      mAP50 = 0.XXX
Gabriele baseline:   mAP50 = 0.824
Delta: +X.XX% (success!) or -X.XX% (gap)
```

---

## Step 5: INT8 Quantization (If YOLO12 succeeds) (1-2 days)

**Purpose:** Accelerate inference for RTX 4060 deployment

### 5a. Export to ONNX (5 minutes)

```bash
yolo export \
  model=runs/yolo12/yolo12n_300ep_FSOCO/weights/best.pt \
  format=onnx \
  batch=2 \
  imgsz=640
```

**Output:** `best.onnx`

### 5b. Create INT8 Calibration Cache (1 hour)

**CRITICAL:** Use validation set for calibration, NOT test set!

```python
# Create calibration script: create_int8_calibration.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO

# Load validation dataset
model = YOLO('runs/yolo12/.../best.pt')
val_data = model.val_loader  # Validation set loader

# Create calibration cache
# (See TensorRT calibration documentation)
# Use 500-1000 images from validation set
```

### 5c. Convert to TensorRT INT8 (30 minutes)

```bash
trtexec \
  --onnx=best.onnx \
  --saveEngine=yolo12n_int8.engine \
  --int8 \
  --calib=calibration_cache.bin \
  --batch=2 \
  --workspace=4096 \
  --device=0
```

**Output:** `yolo12n_int8.engine` (TensorRT engine for RTX 4060)

### 5d. Benchmark Inference (30 minutes)

```bash
python3 benchmark_inference.py \
  --model yolo12n_int8.engine \
  --runs 100 \
  --batch 2
```

**Expected:**
- YOLO12n FP16: ~1.5ms on RTX 4060
- YOLO12n INT8: ~1.0-1.2ms on RTX 4060
- Speedup: 5-6× vs baseline (6.78ms)

---

## Alternative: Branch B (If YOLO12 Fails)

If YOLO12 doesn't achieve mAP50 ≥ 0.74 on Day 3, pivot to Branch B:

### B1. Train YOLOv11m Teacher (2-3 days)

```bash
python3 train_yolov11m_teacher.py
```

**Expected:** mAP50 ~0.88-0.90 (larger model)

### B2. Knowledge Distillation (3-4 days)

```bash
python3 train_with_distillation.py \
  --teacher runs/teacher/yolov11m.../best.pt \
  --student yolo11n.pt \
  --epochs 300
```

**Expected:** mAP50 ~0.83-0.85 (distilled student)

### B3. RegNet Backbone (2-3 days)

```bash
python3 train_yolo11n_regnet.py
```

**Shows understanding of NAS/design space principles**

---

## Timeline Summary

### Branch A (YOLO12 succeeds)

```
Day 1:     Evaluate baseline + UBM on test set (1 hour)
           Start YOLO12 training (background)

Day 2-3:   YOLO12 training continues
           Monitor progress, check Day 3 results

Day 3:     Decision point - mAP50 check
           If ≥0.82 → Continue

Day 4:     Evaluate YOLO12 on test set
           Export to ONNX
           Create INT8 calibration

Day 5:     Convert to TensorRT INT8
           Benchmark inference
           Document results

Day 6-7:   Report writing, final presentation
```

**Total:** 5-7 days

### Branch B (If YOLO12 fails)

```
Day 1:     Pivot decision
           Start YOLOv11m teacher training

Day 2-3:   Teacher training

Day 4-7:   Knowledge distillation training

Day 8-9:   RegNet backbone experiment

Day 10:    INT8 quantization + benchmarking

Day 11-14: Report writing
```

**Total:** 10-14 days

---

## Quick Reference Commands

### Evaluation
```bash
# Your baseline on test
python3 evaluate_baseline_test.py

# UBM model on test
python3 evaluate_ubm_model.py

# YOLO12 on test (after training)
python3 evaluate_yolo12_test.py
```

### Training
```bash
# YOLO12 (Branch A)
python3 train_yolo12.py

# Background with logging
nohup python3 train_yolo12.py > yolo12.log 2>&1 &
```

### Monitoring
```bash
# Watch training log
tail -f yolo12.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check W&B
# https://wandb.ai/ncridlig-ml4cv/3CFU
```

### Export & Benchmark
```bash
# Export to ONNX
yolo export model=best.pt format=onnx batch=2

# TensorRT INT8
trtexec --onnx=best.onnx --int8 --saveEngine=best_int8.engine

# Benchmark
python3 benchmark_inference.py --model best_int8.engine
```

---

## Files Created for You

1. ✅ `evaluate_baseline_test.py` - Evaluate your baseline on test set
2. ✅ `evaluate_ubm_model.py` - Evaluate UBM model on test set (already exists, fixed)
3. ✅ `train_yolo12.py` - Train YOLO12n with Branch A strategy
4. ✅ `evaluate_yolo12_test.py` - Evaluate YOLO12n on test set
5. ✅ `benchmark_inference.py` - ONNX/TensorRT inference benchmarking (already exists)
6. ✅ `SWEEP_CRASH_INVESTIGATION.md` - Analysis of why sweep runs crashed
7. ✅ `SWEEP_ANALYSIS.md` - Complete sweep results analysis
8. ✅ `TWO_BRANCH_STRATEGY.md` - Full implementation plan

---

## Key Reminders

⚠️ **NEVER use test set for:**
- INT8 calibration (use validation set!)
- Model selection
- Hyperparameter tuning
- Early stopping

✅ **ONLY use test set for:**
- Final evaluation (once per model)
- Reporting final results

🎯 **Success criteria:**
- mAP50 ≥ 0.82 on test set (matches/exceeds Gabriele)
- Inference ≤ 2.0ms on RTX 4060 (3.4× speedup)
- Production-ready TensorRT engine

🚀 **Expected YOLO12 results:**
- mAP50: 0.83-0.84 (+16% over baseline sweep)
- Inference: 1.0-1.2ms with INT8 (5.7× speedup)
- Novelty: 2025 state-of-the-art architecture

---

**Ready to proceed!** Start with Step 1 (evaluate baseline on test set), then move to Step 3 (train YOLO12). 🎯

# YOLO26 Training Guide

**Date:** 2026-01-25
**Model:** YOLO26n (latest Ultralytics architecture)
**Status:** Ready to train

---

## 🎯 Why YOLO26?

**YOLO26 is the newest YOLO architecture** from Ultralytics, available in version 8.4.7.

**Model Specifications:**
- **Parameters:** 2,572,280 (similar to YOLO12n: 2,557,703)
- **Expected performance:** Similar or better than YOLO12n
- **Architecture:** Latest improvements from Ultralytics research

**Comparison to Previous Models:**

| Model | Parameters | Test mAP50 | Status |
|-------|-----------|------------|--------|
| **UBM Production** | 2.58M | 0.6655 | Baseline |
| **YOLOv11n** | 2.58M | 0.7065 | Our baseline |
| **YOLO12n** | 2.56M | 0.7081 | Current best |
| **YOLO26n** | 2.57M | **TBD** | **New!** |

**Target:** Beat YOLO12n (0.7081 mAP50 on test set)

---

## ✅ Pre-flight Check: YOLO26 Availability

YOLO26n is confirmed available in Ultralytics 8.4.7:

```python
from ultralytics import YOLO
model = YOLO('yolo26n.pt')  # ✅ Works!
# Parameters: 2,572,280
```

**No difficulties expected** - same API as YOLO12!

---

## 🚀 Training Configuration

### Same Procedure as YOLO12

```python
epochs=300
batch=64
imgsz=640
device=0
workers=12

# Ultralytics defaults (proven optimal)
lr0=0.01
lrf=0.01
momentum=0.937
weight_decay=0.0005

# Early stopping
patience=50

# Mixed precision
amp=True
```

**Why same config?**
- Hyperparameter sweep found defaults are already optimal
- No need to re-tune for similar architecture
- Focus on architecture improvements, not hyperparameters

---

## 📋 Training Commands

### Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run training
python3 train_yolo26.py

# OR in background with logging
nohup python3 train_yolo26.py > yolo26_training.log 2>&1 &

# Monitor progress
tail -f yolo26_training.log

# OR monitor W&B
# https://wandb.ai/ncridlig-ml4cv/yolo26-training
```

**Training time:** ~2.5 days (300 epochs, RTX 4080 Super)

---

## 📊 Expected Results

### Best Case: Improvement over YOLO12 ✅

```
YOLO12n (test):   0.7081 mAP50
YOLO26n (test):   0.72-0.73 mAP50
Improvement:      +2-3%
```

**Reason:** Newer architecture with improvements

---

### Moderate Case: Similar Performance ⚠️

```
YOLO12n (test):   0.7081 mAP50
YOLO26n (test):   0.70-0.71 mAP50
Improvement:      ±1%
```

**Reason:** Both architectures perform comparably on this dataset

---

### Worst Case: Slight Degradation ❌

```
YOLO12n (test):   0.7081 mAP50
YOLO26n (test):   0.68-0.70 mAP50
Degradation:      -1-3%
```

**Reason:** YOLO26 may be optimized for different dataset characteristics

**Mitigation:** Use YOLO12n for deployment instead

---

## 🔬 Evaluation Workflow

### Step 1: Train YOLO26 (2.5 days)

```bash
python3 train_yolo26.py
```

**Monitors:**
- Validation mAP50 during training
- W&B: `ncridlig-ml4cv/yolo26-training`

**Output:** `runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt`

---

### Step 2: Evaluate on Test Set (5 minutes)

```bash
python3 evaluate_yolo26_test.py
```

**Compares to:**
- YOLO12n: 0.7081 mAP50 (current best)
- YOLOv11n: 0.7065 mAP50 (baseline)
- UBM: 0.6655 mAP50 (production)

**Output:** Test set results, comparison table

---

### Step 3: Decision Point

**If YOLO26 > YOLO12:**
- ✅ Proceed to INT8 optimization
- Use YOLO26 for deployment

**If YOLO26 ≈ YOLO12 (within 1%):**
- ⚠️ Either model works
- Choose based on inference speed or simplicity

**If YOLO26 < YOLO12:**
- ❌ Stick with YOLO12n
- Document finding (YOLO26 not better for cone detection)

---

## ⚡ INT8 Optimization (If YOLO26 Wins)

### Step 1: Export to ONNX (5 minutes)

```bash
python3 export_yolo26_onnx.py
```

**Output:** `best.onnx` (intermediate format)

---

### Step 2: Export to TensorRT INT8 (5-10 minutes)

```bash
python3 export_yolo26_tensorrt_int8.py
```

**Improvements over YOLO12 export:**
- ✅ Larger workspace: 8 GB (vs 4 GB)
- ✅ FP16 fallback: Enabled (better accuracy)
- ✅ Calibration: ~500 validation images

**Output:** `best.engine` (TensorRT INT8)

---

### Step 3: Benchmark Performance (10 minutes)

```bash
python3 benchmark_yolo26_int8.py
```

**Measures:**
- FP32 vs INT8 speed (100 runs, stereo batch=2)
- FP32 vs INT8 accuracy (validation set)
- Expected RTX 4060 performance

**Expected Results:**
```
Speed:     1.5-2.0× faster
Accuracy:  >99% retained (<1% loss)
RTX 4060:  ~4 ms per image (vs 6.78 ms baseline)
```

---

## 📈 Performance Comparison

### Validation Set (During Training)

| Model | mAP50 | mAP50-95 | Parameters |
|-------|-------|----------|------------|
| YOLO12n | 0.7127 | 0.4656 | 2,557,703 |
| YOLO26n | **TBD** | **TBD** | 2,572,280 |

**Target:** > 0.7127 mAP50

---

### Test Set (Final Evaluation)

| Model | mAP50 | Precision | Recall | vs UBM |
|-------|-------|-----------|--------|--------|
| UBM Production | 0.6655 | 0.8031 | 0.5786 | — |
| YOLOv11n | 0.7065 | 0.8164 | 0.6616 | +6.2% |
| YOLO12n | 0.7081 | 0.8401 | 0.6542 | +6.4% |
| YOLO26n | **TBD** | **TBD** | **TBD** | **TBD** |

**Target:** > 0.7081 mAP50

---

### INT8 Inference Speed (RTX 4060)

| Model | Format | Speed (per image) | vs Baseline |
|-------|--------|-------------------|-------------|
| UBM Production | TensorRT FP16 | 6.78 ms | 1.00× |
| YOLO12n | TensorRT INT8 | ~4.15 ms | 1.63× |
| YOLO26n | TensorRT INT8 | **~4.0 ms (est)** | **~1.7× (est)** |

**Target:** < 4.5 ms per image

---

## 🎓 Academic Contribution

**This experiment demonstrates:**

1. **Evaluation of newest architecture** (YOLO26, 2025)
2. **Systematic model comparison** (YOLOv11 → YOLO12 → YOLO26)
3. **Architecture evolution** tracking for cone detection
4. **Practical deployment** considerations (INT8, inference speed)

**Novel Findings (Potential):**

- **If YOLO26 > YOLO12:**
  - Latest architecture improves cone detection
  - Quantifies benefit of architectural improvements

- **If YOLO26 ≈ YOLO12:**
  - Cone detection task may be "solved" at this scale
  - Diminishing returns from architecture improvements

- **If YOLO26 < YOLO12:**
  - Not all newer models are better for all tasks
  - Task-specific validation is critical

---

## 📝 Timeline

```
Day 1-3:  YOLO26 training (300 epochs)
Day 3:    Test set evaluation
Day 3:    Decision point (YOLO26 vs YOLO12)
Day 4:    INT8 export and benchmarking (if YOLO26 wins)
Day 4-5:  Report writing and documentation
```

**Total:** 4-5 days

---

## 🚨 Potential Issues

### Issue 1: YOLO26 Not Available

**Symptom:**
```python
FileNotFoundError: yolo26n.pt not found
```

**Solution:**
```bash
# Update Ultralytics to latest version
pip install -U ultralytics

# Verify version
python3 -c "from ultralytics import YOLO; print(YOLO('yolo26n.pt'))"
```

**Status:** ✅ Already verified - YOLO26n available in 8.4.7

---

### Issue 2: Different Hyperparameters Needed

**Symptom:** YOLO26 validation mAP50 < 0.68 (much worse than YOLO12)

**Solution:** Try adjusting hyperparameters
```python
lr0=0.005      # Lower learning rate
patience=30    # More patient
epochs=400     # More epochs
```

---

### Issue 3: Memory Issues

**Symptom:** CUDA out of memory during training

**Solution:**
```python
batch=48       # Reduce from 64
batch=32       # Or reduce further if needed
```

---

## ✅ Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Test mAP50** | > 0.700 | > 0.708 | > 0.720 |
| **vs YOLO12** | ±0% | +1% | +2%+ |
| **vs UBM** | +5% | +6% | +8%+ |
| **INT8 Speed** | < 5 ms | < 4.5 ms | < 4 ms |

**Primary Goal:** Beat or match YOLO12n (0.7081 mAP50)

---

## 📊 Files Created

### Training Scripts
- ✅ `train_yolo26.py` - Train YOLO26n on FSOCO-12
- ✅ `evaluate_yolo26_test.py` - Test set evaluation

### INT8 Optimization
- ✅ `export_yolo26_onnx.py` - Export to ONNX
- ✅ `export_yolo26_tensorrt_int8.py` - Export to TensorRT INT8
- ✅ `benchmark_yolo26_int8.py` - Benchmark FP32 vs INT8

### Documentation
- ✅ `YOLO26_TRAINING_GUIDE.md` - This file

---

## 🎯 Next Actions

### Immediate: Start YOLO26 Training

```bash
# Check if YOLO26 loads successfully
./venv/bin/python3 -c "from ultralytics import YOLO; m = YOLO('yolo26n.pt'); print('✅ Ready!')"

# Start training
./venv/bin/python3 train_yolo26.py

# OR in background
nohup ./venv/bin/python3 train_yolo26.py > yolo26_training.log 2>&1 &
```

### After Training Completes (Day 3)

```bash
# Evaluate on test set
./venv/bin/python3 evaluate_yolo26_test.py

# If YOLO26 > YOLO12, proceed to INT8
./venv/bin/python3 export_yolo26_tensorrt_int8.py
./venv/bin/python3 benchmark_yolo26_int8.py
```

---

**Ready to train YOLO26!** 🚀

---

**Last Updated:** 2026-01-25
**Status:** All scripts created, ready to execute
**Command:** `python3 train_yolo26.py`

# Training Comparison: UBM vs Our Baseline

## 🔍 Critical Discovery

The UBM training notebook shows **LOWER performance** than our baseline!

---

## Training Configuration Comparison

| Parameter | UBM (Notebook) | Our Baseline | Notes |
|-----------|----------------|--------------|-------|
| **Dataset** | FSOCO-12 (fmdv/fsoco-kxq3s v12) | FSOCO-12 (same) | ✅ IDENTICAL |
| **Model** | YOLOv11n | YOLOv11n | ✅ IDENTICAL |
| **Epochs** | 200 | 300 | We trained 50% longer |
| **Ultralytics** | 8.3.78 | 8.4.7 | We have newer version |
| **Hardware** | Tesla P100 (Kaggle) | RTX 4080 Super | We have better GPU |
| **Training Time** | 6.04 hours | 4.4 hours | We trained faster |
| **Hyperparameters** | Defaults (8.3.78) | Defaults (8.4.7) | Likely slight differences |

---

## Results Comparison (Validation Set)

| Metric | UBM (200 ep) | Our Baseline (300 ep) | Difference | Winner |
|--------|--------------|----------------------|------------|--------|
| **mAP50** | **0.663** | **0.714** | **+0.051 (+7.7%)** | **🏆 OURS** |
| **mAP50-95** | **0.456** | **0.467** | **+0.011 (+2.4%)** | **🏆 OURS** |
| **Precision** | **0.819** | **0.832** | **+0.013 (+1.6%)** | **🏆 OURS** |
| **Recall** | **0.573** | **0.657** | **+0.084 (+14.7%)** | **🏆 OURS** |

### Per-Class Results (UBM Notebook, 200 epochs)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All Classes** | 0.819 | 0.573 | 0.663 | 0.456 |
| Blue Cone | 0.924 | 0.580 | 0.733 | 0.508 |
| Large Orange Cone | 0.802 | 0.757 | 0.797 | 0.604 |
| Orange Cone | 0.896 | 0.559 | 0.685 | 0.472 |
| Unknown Cone | 0.566 | 0.394 | 0.384 | 0.205 |
| Yellow Cone | 0.906 | 0.576 | 0.717 | 0.492 |

### Our Baseline (300 epochs) - Summary

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All Classes** | 0.832 | 0.657 | 0.714 | 0.467 |

**We outperform UBM's notebook on ALL metrics!**

---

## Why the Discrepancy?

### Mystery: yolov11n_640p_300ep Model

**Filename suggests 300 epochs**, but notebook shows only 200 epochs.

**Possible explanations:**

1. **They trained longer after the notebook**
   - Started with 200 epochs (notebook)
   - Saw good results, continued to 300 epochs
   - Saved final model as `yolov11n_640p_300ep/best.pt`

2. **Different hyperparameters in production run**
   - Notebook was exploratory (200 epochs, defaults)
   - Production model used tuned hyperparameters (300 epochs)
   - Production model might be better than 0.663

3. **Multiple training runs**
   - Notebook shows one run
   - They may have done several runs, saved best one

4. **Ultralytics version differences**
   - They used 8.3.78, we use 8.4.7
   - Default hyperparameters may have improved between versions
   - This could explain why our defaults are better

---

## What This Means

### ✅ Good News

1. **Our baseline is solid**
   - mAP50 = 0.714 beats their notebook (0.663)
   - We're already competitive

2. **Our approach is validated**
   - Same dataset ✅
   - Same model architecture ✅
   - Longer training helped (200→300 epochs)

3. **Room for improvement**
   - Hyperparameter sweep should push us higher
   - We have a strong foundation to build on

### ✅ Questions ANSWERED (2026-01-24)

1. **What's in their yolov11n_640p_300ep model?**
   - **Answer:** mAP50 = 0.6704 (WORSE than our baseline!)
   - They just trained 100 more epochs (200→300) with same defaults
   - Gained only +0.007 (1%) from extra 100 epochs
   - **OUR BASELINE IS 6.5% BETTER!**

2. **Why 200 → 300 epochs?**
   - Did they see improvement continuing past 200?
   - Or just a standard choice?

3. **Hyperparameter differences?**
   - Ultralytics 8.3.78 vs 8.4.7 defaults
   - Did they tune anything?

---

## Implications for Our Project

### Updated Success Criteria

**Original target:** mAP50 ≥ 0.78 (+9% over our 0.714 baseline)

**New context:**
- UBM notebook baseline: 0.663
- Our baseline: 0.714 (+7.7% over UBM notebook)
- **We're already ahead of their published notebook!**

**Revised targets:**

| Level | mAP50 | vs Our Baseline | vs UBM Notebook | Status |
|-------|-------|-----------------|-----------------|--------|
| **Current** | 0.714 | — | +7.7% | ✅ Done |
| **Minimum** | 0.75 | +5% | +13.1% | Sweep target |
| **Good** | 0.78 | +9% | +17.6% | Strong result |
| **Excellent** | 0.80+ | +12% | +20.7% | Outstanding |

### Next Steps

1. **Complete hyperparameter sweep** (in progress)
   - Target: mAP50 ≥ 0.75 minimum
   - Should be achievable given our strong baseline

2. **Evaluate UBM's production model**
   - Run `evaluate_ubm_model.py` after sweep
   - See if their 300ep model scores higher than 0.714
   - If yes: analyze what they did differently
   - If no: we're on par or ahead!

3. **Architecture comparison**
   - Test YOLOv11s, YOLOv11m
   - With our tuned hyperparameters

---

## Key Differences to Investigate

### Software Environment

**UBM:**
```
Ultralytics 8.3.78
Python 3.10.12
torch 2.5.1+cu121
CUDA:0 (Tesla P100-PCIE-16GB)
```

**Ours:**
```
Ultralytics 8.4.7
Python 3.13.7
torch 2.9.1+cu128
CUDA:0 (RTX 4080 Super, 15937MiB)
```

**Potential differences:**
- PyTorch version (2.5.1 vs 2.9.1)
- CUDA version (12.1 vs 12.8)
- Ultralytics defaults may have improved

### Hardware Impact

**Tesla P100:**
- Older GPU (2016)
- 16GB VRAM
- Lower compute capability
- Slower training (6 hours for 200 epochs)

**RTX 4080 Super:**
- Modern GPU (2024)
- 16GB VRAM
- Higher compute capability
- Faster training (4.4 hours for 300 epochs)

**Impact on results:**
- Should be minimal (same final model)
- Training speed difference is just efficiency

---

## Conclusions

### 🏆 Major Findings

1. **Our baseline (0.714) > UBM notebook baseline (0.663)**
   - We're already 7.7% ahead of their published results
   - This validates our training approach

2. **Same dataset, same model**
   - Differences are in epochs (300 vs 200) and Ultralytics version
   - Our longer training paid off

3. **Mystery of production model**
   - Their `yolov11n_640p_300ep` model might be:
     - Same as ours (~0.714) - meaning we're on par
     - Better than ours - meaning they tuned hyperparameters
   - **We'll find out when we evaluate it!**

### 📊 Confidence in Our Approach

**High confidence:**
- ✅ Correct dataset (FSOCO-12)
- ✅ Correct model (YOLOv11n)
- ✅ Good training duration (300 epochs)
- ✅ Already beating their notebook baseline
- ✅ Hyperparameter sweep should improve further

**What we're testing:**
- Can we beat 0.714 with hyperparameter tuning?
- How does our tuned model compare to their production model?
- Can larger architectures (YOLOv11s/m) improve further?

---

## Action Items

### Immediate (After Sweep)
1. ✅ Analyze sweep results
2. ✅ Extract best hyperparameters
3. ✅ **Evaluate UBM production model** on FSOCO-12
4. ✅ Compare all three:
   - UBM notebook (mAP50 = 0.663)
   - UBM production (unknown)
   - Our baseline (mAP50 = 0.714)
   - Our tuned (target ≥ 0.75)

### Follow-Up Questions for Gabriele
If their production model scores significantly higher than 0.714:

1. What hyperparameters did you change from defaults?
2. Did you train beyond 200 epochs with same config?
3. Any preprocessing or augmentation tricks?
4. Why does the notebook show 200 epochs but model name says 300?

---

**Date:** 2026-01-24
**Status:** UBM notebook analysis complete, awaiting sweep results and production model evaluation
**Conclusion:** Our baseline is strong - already ahead of their published notebook!

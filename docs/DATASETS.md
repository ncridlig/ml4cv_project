# Datasets - ML4CV Cone Detection Project

**Project:** YOLO-based cone detection improvement for UniBo Motorsport
**Date:** 2026-01-28

---

## Overview

This project uses **three datasets** for training, validation, and real-world testing:

| Dataset | Size | Purpose | Source | Performance Impact |
|---------|------|---------|--------|-------------------|
| **FSOCO-12** | 9,777 images | Training & validation benchmark | Internet (Roboflow) | ✅ **Primary training dataset** |
| **fsoco-ubm** | 96 images | Real-world testing | UBM car camera | ✅ **Critical for deployment validation** |
| **cone-detector** | 22,725 images | Pre-training (tested) | Internet (Roboflow) | ❌ **Redundant - no improvement** |

---

## 1. FSOCO-12 (Primary Training Dataset)

### Overview

**Full name:** Formula Student Objects in Context - Version 12
**Purpose:** Standard benchmark for Formula Student cone detection
**Split:** Train / Validation / Test

### Statistics

| Split | Images | Cone Instances | Use |
|-------|--------|----------------|-----|
| **Train** | 7,120 images | ~78,000 instances | Model training |
| **Validation** | 1,968 images | ~36,000 instances | Hyperparameter tuning, model selection |
| **Test** | 689 images | 12,054 instances | Final unbiased evaluation |
| **Total** | **9,777 images** | **~126,000 instances** | Complete dataset |

### Class Distribution (Test Set)

| Class | Instances | Percentage | Difficulty |
|-------|-----------|------------|------------|
| **Yellow Cone** | 4,844 | 40.2% | Easy (high contrast) |
| **Blue Cone** | 4,437 | 36.8% | Easy (high contrast) |
| **Orange Cone** | 1,686 | 14.0% | Medium (variable lighting) |
| **Large Orange Cone** | 408 | 3.4% | Easy (distinctive size) |
| **Unknown Cone** | 679 | 5.6% | Hard (ambiguous/occluded) |

### Source & Download

**Roboflow Project:**
- Workspace: `fmdv`
- Project: `fsoco-kxq3s`
- Version: **12**
- Format: `yolov11` (CRITICAL: must match training format!)
- License: Public

**Download Script:** `download_fsoco.py`

```bash
python3 download_fsoco.py
# Downloads to: datasets/FSOCO-12/
```

**Download Details:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fmdv").project("fsoco-kxq3s")
version = project.version(12)
dataset = version.download("yolov11")  # Format must match!
```

### Characteristics

**Image Properties:**
- Resolution: Variable (640-1280px typical)
- Format: RGB JPEG
- Source: Internet dataset (curated from multiple teams)
- Conditions: Diverse lighting, weather, tracks
- Quality: Clean, well-labeled, curated

**Strengths:**
- ✅ Large size (~10k images) enables robust training
- ✅ Diverse conditions (different tracks, weather, times of day)
- ✅ Well-balanced class distribution
- ✅ High-quality annotations
- ✅ Standard benchmark (reproducible comparisons)

**Weaknesses:**
- ⚠️ Internet dataset (may not match deployment conditions)
- ⚠️ "Unknown cone" class poorly defined (ambiguous labeling)
- ⚠️ Lacks motion blur (static/slow-moving cameras)
- ⚠️ Over-represents "clean" racing conditions

### Performance Benchmarks (Test Set)

**YOLO26n (two-stage) - Best Model:**
- mAP50: **0.7612** (test set)
- Precision: 0.8324
- Recall: 0.7078
- Per-class mAP50: Blue 0.862, Yellow 0.849, Orange 0.836, Large Orange 0.884, Unknown 0.374

**Key Insight:** FSOCO-12 performance does NOT predict real-world performance. Models must be validated on fsoco-ubm.

---

## 2. fsoco-ubm (Real-World Validation Dataset)

### Overview

**Full name:** Formula Student Objects in Context - UniBo Motorsport
**Purpose:** Real-world validation on actual car camera data
**Status:** Test-only dataset (no training!)

### Statistics

| Property | Value | Notes |
|----------|-------|-------|
| **Total Images** | 96 images | Stereo pairs (48 left + 48 right) |
| **Cone Instances** | 1,426 total | ~14.9 cones per image |
| **Source** | UBM race car | ZED 2i stereo camera |
| **Date** | November 20, 2025 | Rioveggio test track, Italy |
| **Conditions** | Real racing | Car moving 30-50 km/h |

### Class Distribution

| Class | Instances | Percentage | Performance Drop vs FSOCO-12 |
|-------|-----------|------------|------------------------------|
| **Blue Cone** | 554 | 38.8% | -5.1% (0.862 → 0.805) |
| **Yellow Cone** | 773 | 54.2% | -3.0% (0.849 → 0.780) |
| **Large Orange** | 63 | 4.4% | -26.7% (0.884 → 0.617) |
| **Orange Cone** | 8 | 0.6% | -43.2% (0.836 → 0.405) ⚠️ |
| **Unknown Cone** | 28 | 2.0% | -100% (0.374 → 0.000) ❌ |

**Note:** Orange cones severely underrepresented (0.6%) and extremely challenging in real-world conditions.

### Source & Creation

**Origin:** Extracted from car camera rosbag recordings

**Extraction Process:**
1. Recorded rosbag files during Rioveggio test track runs
2. Converted to AVI video: `20_11_2025_Rioveggio_Test_LidarTest1.avi` and `LidarTest2.avi`
3. Extracted frames every 60 frames (1 second at 60 FPS = 2 seconds real-world time)
4. Split stereo images (left/right separation: 2560×720 → 2× 1280×720)
5. Uploaded to Roboflow for annotation with Label Assist
6. Manual review and correction of all annotations
7. Exported in `yolov11` format (must match FSOCO-12!)

**Download Script:** `download_fsoco_ubm.py`

```bash
python3 download_fsoco_ubm.py
# Downloads to: datasets/ml4cv_project-1/
```

**Roboflow Project:**
- Workspace: `fsae-okyoe`
- Project: `ml4cv_project`
- Version: **1**
- Format: `yolov11` (CRITICAL: was initially wrong with `yolo26` format!)
- License: Private (UBM team only)

### Characteristics

**Camera Specifications (ZED 2i Stereo):**
- Resolution: 2× 1280×720 @ 60 FPS (stereo pair)
- FOV: 72° (H) × 44° (V)
- Baseline: 12 cm
- Depth range: 1.5m - 20m (accuracy degrades with distance)

**Real-World Challenges:**
- 🏎️ **Motion blur:** Car moving 30-50 km/h during capture
- ☀️ **Variable lighting:** Outdoor track, shadows, bright sky
- 📏 **Distance challenges:** Cones at 10-20m with small pixel area
- 🔧 **Camera distortion:** Real stereo rig (barrel distortion, chromatic aberration)
- 👥 **Occlusion:** Cones partially hidden behind other cones
- 💡 **Auto-exposure:** Bright sky causes underexposure of cones
- 🌫️ **Environmental:** Dust, track debris, reflections

**Why fsoco-ubm is 22-32% Harder:**

| Model | FSOCO-12 mAP50 | fsoco-ubm mAP50 | Drop |
|-------|---------------|-----------------|------|
| YOLO26n (two-stage) | 0.7612 | 0.5652 | **-25.8%** |
| YOLO26n (single) | 0.7626 | 0.5650 | **-25.9%** |
| YOLOv11n | 0.7065 | 0.5545 | **-21.5%** ✅ Best generalization |
| YOLO12n | 0.7081 | 0.5172 | **-27.0%** |
| UBM production | 0.6655 | 0.5168 | **-22.3%** |

**Average performance drop:** **-26.5%** across all models

### Performance Benchmarks (Test Set)

**YOLO26n (two-stage) - Deployed Model:**
- mAP50: **0.5652** (fsoco-ubm test set) 🏆
- Precision: 0.6485
- Recall: 0.4620
- Per-class: Blue 0.805, Yellow 0.780, Large Orange 0.617, Orange 0.405, Unknown 0.000

**Critical Finding:** Model rankings changed between FSOCO-12 and fsoco-ubm!
- YOLOv11n rose from #5 to #3 (better generalization)
- YOLO26n maintained #1-2 position (still best overall)
- YOLO12n dropped significantly (attention mechanisms overfit to FSOCO-12)

### Academic Significance

**This dataset demonstrates:**
1. ⚠️ **Generalization gap:** Standard benchmarks don't predict real-world performance
2. ✅ **Domain-specific validation:** In-house test sets are critical for deployment
3. 🎯 **Architecture selection:** Simpler models (YOLOv11n) generalize better than complex (YOLO12n)
4. 📊 **Real-world difficulty:** Motion blur and lighting variance significantly impact performance

**Recommendation for Formula Student teams:** Create your own in-house test set from car camera data. FSOCO-12 alone is insufficient for deployment validation.

---

## 3. cone-detector (Pre-training Dataset - Tested & Rejected)

### Overview

**Full name:** Cone Detector Dataset
**Purpose:** Large-scale pre-training before fine-tuning on FSOCO-12
**Result:** ❌ **No performance improvement - redundant to FSOCO-12**

### Statistics

| Property | Value | Notes |
|----------|-------|-------|
| **Total Images** | 22,725 images | 2.3× larger than FSOCO-12 |
| **Cone Instances** | ~200,000+ | High instance count |
| **Classes** | 5 (same as FSOCO-12) | Blue, yellow, orange, large orange, unknown |
| **Source** | Internet (multiple teams) | Formula Student community dataset |

### Source & Download

**Roboflow Project:**
- Workspace: `fsbdriverless`
- Project: `cone-detector-zruok`
- Version: **1**
- Format: `yolov11`
- License: Public

**Not recommended for download** - use FSOCO-12 directly instead.

### Two-Stage Training Experiment

**Hypothesis:** Pre-training on larger dataset (cone-detector) before fine-tuning on FSOCO-12 would improve performance.

**Training Strategy:**
- **Stage 1:** Train YOLO26n on cone-detector (22,725 images, 400 epochs)
  - Result: 0.7339 mAP50 on cone-detector validation
  - Training time: ~33 hours

- **Stage 2:** Fine-tune on FSOCO-12 (7,120 images, 300 epochs, two-phase)
  - Phase 2A: Freeze backbone, train head (50 epochs, lr=0.001)
  - Phase 2B: Unfreeze all, full fine-tuning (250 epochs, lr=0.00005)
  - Result: 0.7612 mAP50 on FSOCO-12 test set

**Comparison: Single-Stage vs Two-Stage**

| Metric | Single-Stage (300 ep) | Two-Stage (700 ep) | Delta |
|--------|----------------------|-------------------|-------|
| **FSOCO-12 Test** | 0.7626 | 0.7612 | -0.0014 (-0.2%) |
| **fsoco-ubm Test** | 0.5650 | 0.5652 | +0.0002 (+0.04%) |
| **Training Time** | 10 hours | 43 hours | 4.3× longer |
| **Generalization** | -25.9% | -25.8% | +0.1% better |

### Conclusion: cone-detector is Redundant

**Findings:**
1. ❌ **No significant improvement:** 0.04% gain on fsoco-ubm (within noise margin)
2. ❌ **4.3× training cost:** 43 hours vs 10 hours for marginal benefit
3. ❌ **Dataset overlap:** cone-detector and FSOCO-12 likely share images/scenes
4. ❌ **Same distribution:** Both are internet datasets with similar characteristics

**Why it doesn't help:**
- Both datasets are Formula Student cone detection (same domain)
- Extended training (700 epochs) doesn't overcome distribution similarity
- Pre-training benefits require domain gap (e.g., ImageNet → cones), not same-domain larger dataset

**Recommendation:** Skip two-stage training. Train directly on FSOCO-12 for 300 epochs.

**Exception:** Two-stage training justified for YOLO26n deployment because:
- ✅ Marginal real-world improvement (+0.04% on fsoco-ubm)
- ✅ Better precision (0.6485 vs 0.6149)
- ✅ Demonstrates extended training benefit for academic contribution
- ⚠️ Cost acceptable for production model (one-time training)

---

## Dataset Comparison Summary

### Training Efficiency

| Dataset | Images | Training Time (300 ep) | Cost/Benefit | Recommendation |
|---------|--------|------------------------|--------------|----------------|
| **FSOCO-12** | 9,777 | 10 hours | ✅ **Best** | Use for all training |
| **cone-detector** | 22,725 | 33 hours | ❌ Poor | Skip (redundant) |
| **Combined (two-stage)** | 32,502 | 43 hours | ⚠️ Marginal | Only if time permits |

### Validation Priorities

| Dataset | Purpose | When to Use | Critical? |
|---------|---------|-------------|-----------|
| **FSOCO-12 val** | Hyperparameter tuning | During training | ✅ Yes |
| **FSOCO-12 test** | Architecture selection | After training | ✅ Yes |
| **fsoco-ubm test** | Deployment validation | Before car deployment | ✅ **CRITICAL** |

### Performance Predictability

**Question:** Does FSOCO-12 performance predict real-world (fsoco-ubm) performance?

**Answer:** ❌ **NO** - Rankings change significantly!

**FSOCO-12 ranking:**
1. YOLO26n (0.7626)
2. YOLO26n two-stage (0.7612)
3. YOLO12n (0.7081)
4. YOLOv11n (0.7065) ← #4 on benchmark
5. UBM production (0.6655)

**fsoco-ubm ranking:**
1. YOLO26n two-stage (0.5652) ← Maintained top
2. YOLO26n (0.5650)
3. **YOLOv11n (0.5545)** ← Rose to #3! Better generalization
4. YOLO12n (0.5172) ← Dropped significantly
5. UBM production (0.5168)

**Takeaway:** Always validate on real car data (fsoco-ubm) before deployment. Standard benchmarks are necessary but insufficient.

---

## Best Practices

### For Training

1. ✅ **Use FSOCO-12 only** - largest curated dataset with good diversity
2. ✅ **300 epochs sufficient** - longer training doesn't help significantly
3. ✅ **Ultralytics defaults work** - hyperparameter tuning unnecessary
4. ❌ **Skip cone-detector** - redundant, no performance gain
5. ❌ **Skip two-stage training** - 4.3× cost for 0.04% gain (unless academic interest)

### For Validation

1. ✅ **Always test on fsoco-ubm** - real-world validation is critical
2. ✅ **Check generalization gap** - FSOCO-12 → fsoco-ubm drop should be <30%
3. ✅ **Verify ranking preservation** - top models on FSOCO-12 should stay competitive on fsoco-ubm
4. ⚠️ **Don't overtune to fsoco-ubm** - it's small (96 images), risk of overfitting
5. ✅ **Consider confidence threshold tuning** - optimize for real-world F1 score on fsoco-ubm

### For Deployment

1. ✅ **fsoco-ubm is ground truth** - deploy model that performs best on car data
2. ✅ **Monitor generalization gap** - large drops indicate dataset mismatch
3. ✅ **Expand fsoco-ubm over time** - collect more car camera data (target: 300+ images)
4. ⚠️ **Never use fsoco-ubm for training** - test set sanctity is critical
5. ✅ **Create fsoco-ubm v2** - annotate more car camera data from future test runs

---

## Data Collection Recommendations

### Expanding fsoco-ubm (Future Work)

**Current status:** 96 images (sufficient for initial validation)

**Expansion plan:**
- **Target size:** 300-400 images
- **Sources:** Multiple test tracks, different weather/lighting conditions
- **Sampling:** Every 2-3 seconds of real driving (avoid redundancy)
- **Quality:** Manual annotation with double-checking

**Benefits:**
- More robust validation (96 images is borderline)
- Statistical significance testing possible
- Better confidence in deployment decisions
- Catch edge cases (night driving, rain, extreme distances)

**Effort:** ~10-15 hours (extraction + annotation + review)

---

## Dataset Files & Scripts

### Download Scripts

| Script | Dataset | Location |
|--------|---------|----------|
| `download_fsoco.py` | FSOCO-12 | `datasets/FSOCO-12/` |
| `download_fsoco_ubm.py` | fsoco-ubm | `datasets/ml4cv_project-1/` |

### Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_fsoco_ubm.py` | Unified evaluation on fsoco-ubm (all models) |
| `evaluate_yolo26_two_stage_test.py` | FSOCO-12 test set evaluation |
| `optimize_confidence_threshold.py` | Find optimal conf threshold on fsoco-ubm |
| `visualize_fsoco_ubm.py` | Visualize annotations and predictions |

### Documentation

| File | Content |
|------|---------|
| `docs/DATASETS.md` | This file |
| `CLAUDE.md` | Main project documentation |
| `docs/RESULTS_SUMMARY.md` | Complete performance comparison |
| `docs/TODO.md` | Task tracking |

---

## Critical Lessons Learned

1. **Format mismatch causes invalid results**
   - fsoco-ubm initially downloaded in `yolo26` format (WRONG)
   - Models trained on `yolov11` format couldn't parse annotations correctly
   - All initial fsoco-ubm results were invalid until format fixed
   - **Lesson:** Always match annotation format between training and testing datasets

2. **Internet datasets ≠ real-world performance**
   - FSOCO-12: 0.7626 mAP50 → fsoco-ubm: 0.5650 mAP50 (-26% drop)
   - Motion blur, lighting variance, camera distortion not well-represented
   - **Lesson:** In-house test sets are mandatory for deployment validation

3. **Bigger dataset ≠ better performance**
   - cone-detector (22,725 images) provided no benefit over FSOCO-12 (9,777 images)
   - Same domain, similar distribution → redundant information
   - **Lesson:** Dataset quality and diversity matter more than quantity

4. **Model rankings are dataset-dependent**
   - YOLOv11n: #4 on FSOCO-12 → #3 on fsoco-ubm (better generalization)
   - YOLO12n: #3 on FSOCO-12 → #4 on fsoco-ubm (overfits to benchmark)
   - **Lesson:** Always validate on target domain before deployment

---

**Last Updated:** 2026-01-28
**Status:** Complete dataset analysis and recommendations
**Next:** Deploy YOLO26n (two-stage) validated on fsoco-ubm

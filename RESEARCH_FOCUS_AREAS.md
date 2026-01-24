# Research Focus Areas Based on Edoardo's Feedback

## Overview

Based on direct feedback from Edoardo Fusa (thesis author, 2026-01-24), this document outlines key technical challenges that should guide our improvement efforts.

---

## Challenge 1: Image Brightness Robustness

### The Problem
> "For what regards image quality, yes it is a big problem, expecially brightness of the image. If it's not good you can easily mistake cone type."
> — Edoardo Fusa

**Real-world scenario:**
- Sky conditions cause auto-exposure issues (bright sky → underexposed cones)
- Cone colors become ambiguous in poor lighting
- Classification errors: blue ↔ yellow, orange ↔ yellow misclassifications

### Impact on Competition
- **Track conditions vary:** Sunny, cloudy, shadows
- **Time of day matters:** Morning vs afternoon lighting
- **False classifications → Wrong trajectory planning → DNF**

### Current Baseline Status
Our YOLOv11n baseline (mAP50 = 0.714) trained on FSOCO-12 likely struggles with:
- Underexposed images
- Overexposed images
- High dynamic range scenes (bright sky + dark ground)

### Improvement Strategies

#### 1. **Augmentation-Based (Hyperparameter Sweep)**
Already included in our sweep:
- `hsv_h`: Hue jitter (0.0 - 0.03) - simulate color shifts
- `hsv_s`: Saturation jitter (0.5 - 0.9) - simulate brightness changes
- `hsv_v`: **Value jitter (0.3 - 0.6)** - **THIS IS CRITICAL FOR BRIGHTNESS**

**Action:** When analyzing sweep results, prioritize configs with:
- Higher `hsv_v` range (more brightness variation)
- Check if top runs use aggressive brightness augmentation

#### 2. **Dataset-Based (Future Work)**
- **Histogram equalization** preprocessing
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Brightness-augmented dataset:** Artificially darken/brighten FSOCO images
- **Multi-exposure training:** Train on images at different exposure levels

#### 3. **Architecture-Based (Model Improvements)**
- **Attention mechanisms:** Help focus on cone shape, not just color
- **Multi-scale features:** Better handle varying lighting across image
- **Color-invariant features:** Learn from cone shape/context, not just color

#### 4. **Post-Processing (Deployment)**
- **Adaptive preprocessing:** Auto-adjust image brightness before inference
- **Ensemble methods:** Combine predictions from multiple brightness versions
- **Temporal consistency:** Use previous frame detections to validate current frame

---

## Challenge 2: Orange Cone Size Classification

### The Problem
> "Even if you have a good image, you have to put care on the orange cones, since we have normal and big size: at long distance it is difficult to distinguish them."
> — Edoardo Fusa

**Classes:**
- `orange_cone`: Normal size
- `large_orange_cone`: Larger markers (start/finish line, track boundaries)

**Challenge:**
- At distance: pixel difference is minimal
- Scale ambiguity: Is it a large cone far away, or normal cone close up?
- **Critical for competition:** Misclassification affects track understanding

### Current Baseline Performance (from Thesis)
From Edo's thesis (likely better dataset/config than ours):

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Orange Cone | 0.925 | 0.787 | 0.877 |
| Large Orange Cone | **0.868** | **0.873** | **0.908** |

**Observations:**
- Large orange cone has **lower precision** (more false positives)
- Large orange cone actually has **higher recall & mAP50**
- This suggests model may be biased toward large orange predictions

**Our baseline likely has similar or worse confusion.**

### Improvement Strategies

#### 1. **Analyze Confusion Matrix (After Sweep)**
```python
# Check which classes are confused
# Hypothesis: orange_cone ↔ large_orange_cone confusion
```

**Action:** When analyzing sweep results:
- Check per-class mAP50 for orange variants
- Identify if confusion exists between normal/large orange

#### 2. **Context-Based Features**
- **Positional priors:** Large orange cones typically at start line
- **Multi-object context:** Look at neighboring cones for scale reference
- **Depth estimation:** Use stereo camera depth to help disambiguate

#### 3. **Aspect Ratio Features**
- Orange cones have different height/width ratios
- Large orange cones are taller AND wider
- **Action:** Check if YOLO's bounding box aspect ratio helps classification

#### 4. **Longer-Range Detection**
- Train with more small/distant cone examples
- Use higher resolution (640 → 800 or 1024) for distant cones
- Multi-scale training with emphasis on small objects

---

## Challenge 3: Real vs Internet Dataset Generalization

### The Problem
> "This last point would be critical to compare on a test set on our car, not from the internet!"
> — Edoardo Fusa

**Key insight:**
- FSOCO (internet dataset) ≠ UniBo Motorsport car data
- Domain shift: Different camera, exposure settings, track conditions
- **Final validation must use real car data**

### Implications for This Project

#### What We CAN Do (2-week timeline)
✅ Improve on FSOCO-12 baseline (mAP50 0.714 → 0.78+)
✅ Test robustness to brightness variations
✅ Compare model architectures (YOLOv11n/s/m)
✅ Deliver optimized ONNX/TensorRT models

#### What We CANNOT Guarantee
❌ Performance on real car data (no access to it)
❌ Perfect orange cone size classification at race distances
❌ Robustness to team's specific camera/lighting setup

#### Recommended Deliverable Structure
1. **FSOCO-12 Benchmark Results** (what we can measure)
2. **Robustness Analysis** (brightness augmentation, per-class performance)
3. **Deployment-Ready Models** (ONNX, TensorRT for RTX 4060)
4. **Recommendations for Real-World Validation:**
   - "Test on 100-200 images from team's car camera"
   - "Validate orange cone classification at 5m, 10m, 15m, 20m distances"
   - "Check performance in sunny vs cloudy conditions"

---

## Priority Action Items for Hyperparameter Sweep

### High Priority
1. **Monitor `hsv_v` (brightness jitter)** - Should be in top configs
2. **Check per-class mAP50** - Especially orange variants
3. **Analyze failed runs** - Are they failing on brightness-sensitive examples?

### Medium Priority
4. **Test higher resolution** - If time permits, try 800x800 or 1024x1024
5. **Aggressive augmentation** - Favor configs with strong augmentation

### Low Priority (Future Work)
6. **Custom dataset augmentation** - Create brightness-varied FSOCO
7. **Attention mechanisms** - Try YOLO variants with attention
8. **Ensemble methods** - Combine multiple models for robustness

---

## Updated Success Criteria

### Primary Goal (FSOCO-12 Performance)
- mAP50 ≥ 0.78 (+9% over baseline)
- Precision ≥ 0.85
- Recall ≥ 0.70

### Secondary Goal (Robustness)
- **Orange cone classification:** mAP50 > 0.75 for both classes
- **Brightness robustness:** Manual test with darkened/brightened images
- **No major class confusions:** Check confusion matrix

### Deliverables
1. Tuned YOLOv11n model (best hyperparameters)
2. Architecture comparison (YOLOv11s, YOLOv11m)
3. ONNX/TensorRT optimized models
4. **Robustness report** addressing brightness and orange cone challenges
5. **Real-world validation recommendations** for the team

---

## Next Steps

### Immediate (During Sweep)
1. ✅ Sweep running with brightness augmentation (`hsv_v`)
2. Monitor W&B for patterns in successful configs
3. Check if top runs have higher brightness jitter

### After Sweep (Tomorrow)
1. Analyze results with focus on:
   - Brightness augmentation levels
   - Per-class performance (orange variants)
2. Extract best config
3. Train full 300 epochs

### Week 2
1. Architecture comparison (YOLOv11s, YOLOv11m)
2. ONNX/TensorRT optimization
3. Robustness analysis
4. Final report with real-world validation recommendations

---

**Date:** 2026-01-24
**Based on:** Direct communication with Edoardo Fusa
**Document Purpose:** Guide research priorities based on real-world deployment challenges

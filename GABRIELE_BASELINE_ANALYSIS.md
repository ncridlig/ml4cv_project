# Gabriele's Baseline Analysis - The TRUE Baseline

**Source:** `report_ceccolini_esame_cv_yolo.pdf` (November 13, 2025)

**Date Analyzed:** 2026-01-24

---

## 🎯 The ACTUAL Production Baseline

Gabriele's report provides the **authoritative baseline** for the UBM YOLO pipeline.

---

## YOLO Training Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv11n |
| **Dataset** | FSOCO (Formula Student Objects in Context) |
| **Epochs** | 300 |
| **Training Set** | 7,120 images (no augmentation) |
| **Validation Set** | 1,969 images |
| **Test Set** | 689 images |
| **Ultralytics Version** | Not specified (likely 8.x) |
| **Hardware** | AMD Ryzen 9 6900HX + NVIDIA RTX 3080 Mobile |

---

## Performance on FSOCO Test Set

**Table 1 from Report (page 9):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All Classes** | **0.849** | **0.765** | **0.824** | **0.570** |
| Blue Cone | 0.922 | 0.806 | 0.896 | 0.615 |
| Yellow Cone | 0.926 | 0.794 | 0.892 | 0.608 |
| Orange Cone | 0.925 | 0.787 | 0.877 | 0.603 |
| Large Orange Cone | 0.868 | 0.873 | 0.908 | 0.710 |
| Unknown Cone | 0.603 | 0.566 | 0.547 | 0.315 |

**This is the same mAP50 = 0.824 cited in Edo's thesis!**

---

## Inference Performance (RTX 3080 Mobile + TensorRT)

**Table 3 from Report (page 22) - ORB Features Configuration:**

| Stage | Time (ms) | Std Dev (ms) |
|-------|-----------|--------------|
| Preprocessing | 0.34 | 0.07 |
| **Inference** | **6.78** | **2.05** |
| Postprocessing | 0.38 | 0.08 |
| BBox Matching | 0.40 | 0.28 |
| Feature Matching | 1.41 | 4.27 |
| Triangulation | 0.11 | 0.07 |
| Sending | 0.04 | 0.01 |
| **Total** | **9.46** | **6.35** |

**Real-time capable:** < 10 ms total, well under 16.7 ms budget for 60 fps

---

## 🔴 Critical Discovery: Dataset Split Mismatch

### The Problem

**Gabriele's evaluation:** Test set (689 images)
**Our evaluations:** Validation set (1,968 images)

| Model | Dataset Split | mAP50 | Source |
|-------|---------------|-------|--------|
| **Gabriele's baseline** | **Test** | **0.824** | Report Table 1 |
| Our baseline | Validation | 0.714 | Our training |
| UBM production model | Validation | 0.670 | Our evaluation |

**We cannot directly compare these results!**

### Why This Matters

1. **Test set is the gold standard** for final model evaluation
2. **Validation set was used during training** for early stopping, hyperparameter selection
3. **Comparing validation vs test results is invalid** - different distributions, sizes, difficulty

### What We Need to Do

✅ **Re-evaluate ALL models on the FSOCO test set:**
1. Our baseline (currently mAP50 = 0.714 on validation)
2. UBM production model (currently mAP50 = 0.670 on validation)
3. All sweep results
4. Final tuned model

**Only then can we compare to Gabriele's mAP50 = 0.824 baseline!**

---

## Pipeline Details from Report

### Multi-Stage Stereo Matching (Section 2.3)

The pipeline uses a sophisticated matching strategy:

1. **Geometric Candidate Filtering** - Fast, lightweight filters to eliminate invalid matches
2. **Template Matching** - Precise matching to refine correspondences
3. **Feature-Based Matching with ORB** (optional) - Multiple keypoints for robust triangulation

### ORB vs Center Point (Section 2.4)

**ORB advantages:**
- **Higher stability and reliability** especially at medium and long ranges
- **Lower standard deviation** (e.g., 0.68m at 15m vs 1.16m for center point)
- **More noise-resistant** due to multiple correspondence points

**Trade-off:**
- ORB adds ~1.4 ms overhead (9.46 ms vs 7.95 ms total)
- Both methods well within real-time budget

**Conclusion from report:** "ORB-based feature matching approach provides higher stability and reliability, especially at medium and long ranges. These characteristics make it the preferred configuration for the stereocamera pipeline used in the autonomous vehicle."

---

## Key Insights

### 1. Gabriele's Training Details

From page 9:
> "The model was trained for 300 epochs on a training set of 7,120 images (without augmentation), validated on 1,969 images, and tested on 689 images."

**No augmentation mentioned** - suggests default Ultralytics augmentation or minimal augmentation strategy.

### 2. Performance Analysis

From page 9:
> "The fine-tuned YOLOv11 model achieves an overall mAP50 of 0.824, with precision and recall of 0.849 and 0.765, respectively, confirming strong general performance. Results are particularly robust for the main cone classes (Blue, Yellow, Orange), all with precision above 0.92 and mAP50 close to 0.90."

### 3. Unknown Cone Challenge

From page 9:
> "Unknown class shows weaker performance, as expected due to its ambiguous nature."

Unknown cones: mAP50 = 0.547 (much lower than other classes)

### 4. Real-Time Capability

From page 21:
> "All configurations operate well within the time budget for the 60 fps camera (16.7 ms per frame). The TensorRT pipeline achieves full processing in under 10 ms, leaving limited but sufficient room for other tasks such as SLAM and motion planning algorithms."

---

## Updated Project Targets

### New Baseline Reference

**Target:** Match or exceed Gabriele's test set performance

| Metric | Gabriele's Baseline | Our Target | Stretch Goal |
|--------|---------------------|------------|--------------|
| **mAP50 (Test)** | **0.824** | **≥ 0.82** | **≥ 0.85** |
| **Precision (Test)** | 0.849 | ≥ 0.85 | ≥ 0.88 |
| **Recall (Test)** | 0.765 | ≥ 0.77 | ≥ 0.80 |
| **mAP50-95 (Test)** | 0.570 | ≥ 0.57 | ≥ 0.60 |
| **Inference (RTX 4080)** | 6.78 ms (RTX 3080) | ≤ 6.0 ms | ≤ 5.0 ms |

### Success Criteria

**Minimum Success:**
- Match Gabriele's test set performance (mAP50 ≥ 0.82)
- Inference speed ≤ 6.78 ms on RTX 4080 Super

**Good Success:**
- Exceed baseline by 3-5% (mAP50 ≥ 0.85)
- Faster inference (≤ 5.5 ms)

**Excellent Success:**
- Exceed baseline by 8-10% (mAP50 ≥ 0.90)
- Significantly faster inference (≤ 5.0 ms)
- Improved unknown cone detection (mAP50 > 0.60)

---

## Action Items

### Immediate (After Sweep Completes)

1. ✅ **Re-evaluate our baseline on test set**
   - Script: `evaluate_ubm_model.py` (already fixed to use test set)
   - Compare to Gabriele's 0.824 baseline

2. ✅ **Re-evaluate UBM production model on test set**
   - Use same script with test split
   - See if it matches Gabriele's 0.824 or differs

3. ✅ **Modify sweep validation to use test set**
   - Update `train_sweep.py` validation split
   - Ensure fair comparison

### After Identifying Best Config

4. Train final model with best hyperparameters (300 epochs)
5. Evaluate on test set and compare to Gabriele's 0.824 baseline
6. If we match or exceed: document hyperparameter improvements
7. If we fall short: analyze gap and potential causes

---

## Collaborators Mentioned

From page 34 (Acknowledgments):
> "Federico Fusa, Beatrice Bottari, and Gabriele Pattarozzi for their work on the perception and computer vision stack."

**So the YOLO work was done by:**
- Federico Fusa (Edoardo's brother?)
- Beatrice Bottari
- Gabriele Pattarozzi
- Gabriele Ceccolini (author)

This aligns with Edo's statement that "Gabriele and Patta" did the training!

---

## References

**Gabriele's Report:**
- Title: "Stereo vision pipeline and Graph-SLAM implementation for FSAE driverless vehicle competition"
- Date: November 13, 2025
- Course: Computer Vision and Image Processing M 2024/2025

**Edo's Thesis:**
- Edoardo Fusa, "Pushing Cars' Limits: Exploring Autonomous Technologies in the Formula SAE Driverless Competition"
- Link: https://amslaurea.unibo.it/id/eprint/35885/1/AI_Master_Thesis___Edoardo_Fusa.pdf

---

**Conclusion:**

Gabriele's report provides the **ground truth baseline** we need to target:
- **mAP50 = 0.824 on test set**
- **Inference: 6.78 ms on RTX 3080 Mobile**

We must re-evaluate all our models on the **test set** to make valid comparisons!

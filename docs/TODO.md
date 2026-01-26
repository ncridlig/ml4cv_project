# TODO List

## 📊 Current Status (2026-01-26)

**🎉 YOLO26n Training COMPLETE - NEW BEST MODEL!**

**Validation Set Performance:**
- **YOLO26n (NEW BEST): 0.7586 mAP50** 🏆 ✅
- YOLO12n: **0.7127 mAP50** ✅
- YOLOv11n baseline: **0.7140 mAP50** ✅
- **Improvement: +6.4%** over YOLO12n!

**Test Set Performance (689 images):**
- **YOLO12n: 0.7081 mAP50** (+6.4% vs UBM) ✅
- Our YOLOv11n baseline: **0.7065 mAP50** (+6.2% vs UBM) ✅
- UBM production model: **0.6655 mAP50**
- **YOLO26n: 🔄 PENDING EVALUATION**

**Training Status:**
- ✅ YOLOv11n baseline complete
- ✅ Hyperparameter sweep complete (stopped - no improvement)
- ✅ YOLO12n training complete (300/300 epochs)
- ✅ YOLO12n INT8 TensorRT export complete
- ✅ Test set evaluation complete (YOLO12, baseline, UBM)
- ✅ **YOLO26n training COMPLETE** (300/300 epochs) 🎉

**Next Immediate Actions:**
1. 🔄 **Evaluate YOLO26 on test set** - `python3 evaluate_yolo26_test.py`
2. 🔄 **Compare YOLO26 vs YOLO12** test results
3. 🔄 **If YOLO26 better:** Export to INT8 and deploy
4. 📅 Meeting with Alberto (workshop) - see goals below

---

## 🚀 YOLO26 Next Steps (PRIORITY)

### Step 1: Test Set Evaluation ⚡ (15 minutes)

**Command:**
```bash
source venv/bin/activate
python3 evaluate_yolo26_test.py
```

**What it does:**
- Evaluates YOLO26n on test set (689 images)
- Compares to YOLO12 test results (0.7081 mAP50)
- Generates confusion matrix, per-class metrics
- Saves results to: `runs/evaluation/yolo26n_on_test_set/`

**Expected Results:**
- **Best case:** 0.73-0.75 mAP50 (+3-6% over YOLO12)
- **Good case:** 0.71-0.72 mAP50 (+0.5-2% over YOLO12)
- **Similar:** 0.70-0.71 mAP50 (±0.5% vs YOLO12)

**Decision:**
- If mAP50 > 0.715 → **Use YOLO26** for deployment ✅
- If mAP50 < 0.715 → Stick with YOLO12 ⚠️

---

### Step 2: INT8 Optimization (30 minutes) - Conditional

**IF YOLO26 > YOLO12 on test set:**

```bash
# Export to TensorRT INT8 (batch=2 for stereo, workspace=8GB)
python3 export_yolo26_tensorrt_int8.py

# Benchmark speed and accuracy (FP32 vs INT8)
python3 benchmark_yolo26_int8.py
```

**Expected Performance (RTX 4080 Super):**
- FP32: ~3.4 ms (PyTorch baseline from W&B)
- INT8: ~2.0-2.5 ms (1.4-1.7× speedup)
- Accuracy loss: <1% (0.75 → ~0.74 mAP50)

**Expected Performance (RTX 4060 - Deployment):**
- INT8: ~3.3-4.2 ms (batch=2) = **1.65-2.1 ms per image**
- vs Baseline: 6.78 ms (3.2-4.1× FASTER than UBM!)
- Real-time: 60 fps capable (16.7 ms budget)

**Deliverables:**
- TensorRT INT8 engine: `runs/yolo26/yolo26n_300ep_FSOCO/weights/best.engine`
- Speed benchmark results
- Accuracy comparison (validation set)

---

### Step 3: Final Model Selection

**Comparison Matrix:**

| Metric | YOLO12n | YOLO26n | Winner |
|--------|---------|---------|--------|
| **Test mAP50** | 0.7081 | 🔄 TBD | 🔄 |
| **Validation mAP50** | 0.7127 | 0.7586 | YOLO26 ✅ |
| **Precision** | 0.8401 | 0.8325 | YOLO12 ✅ |
| **Recall** | 0.6542 | 0.7012 | YOLO26 ✅ |
| **PyTorch Speed** | 4.1 ms | 3.4 ms | YOLO26 ✅ |
| **INT8 Speed (est)** | ~2.5 ms | ~2.2 ms | YOLO26 ✅ |
| **vs UBM** | +6.4% | 🔄 TBD | 🔄 |

**Selection Criteria:**
1. **Primary:** Test set mAP50 (accuracy)
2. **Secondary:** INT8 inference speed (deployment)
3. **Tertiary:** Precision (safety-critical false positives)

**Recommendation (Provisional):**
- If YOLO26 test mAP50 > 0.715 → **YOLO26 wins** (better accuracy + faster)
- If YOLO26 test mAP50 < 0.715 → **YOLO12 wins** (proven reliability)

---

## 🤝 Meeting with Alberto - Workshop Goals

**Date:** TBD (tomorrow)
**Location:** Workshop

### Goal 1: Create Custom Test Set from .mcap Data 🎥

**Objective:** Create a small real-world test set from car's camera data

**Requirements:**
- Extract **100 frames** from .mcap recorded data
- Convert to YOLO format (images + labels)
- Use for real-world validation

**Why Important:**
- FSOCO-12 is internet dataset (may not match real conditions)
- Car data = ground truth for actual deployment
- Validates model on true edge cases (lighting, weather, track conditions)

**Deliverables:**
- [ ] 100 annotated frames from .mcap
- [ ] YOLO format dataset (`data.yaml`, images/, labels/)
- [ ] Test script: `evaluate_on_car_data.py`

**Tools Needed:**
- .mcap reader (ROS2 bag format)
- Frame extraction tool
- Annotation tool (LabelImg, Roboflow, or existing labels if available)

---

### Goal 2: Benchmark YOLO12 TensorRT Engine on Car's RTX 4060 ⚡

**Objective:** Measure real-world inference speed on deployment hardware

**Requirements:**
- Transfer `best.engine` to car's RTX 4060
- Run benchmark with actual stereo images (batch=2)
- Measure: mean, std, min, max latency (100 runs minimum)

**Why Important:**
- Current estimates based on RTX 4080 Super (training GPU)
- RTX 4060 is actual deployment target
- Need real numbers for 60 fps validation

**Expected Results:**
- Current estimate: ~8.3 ms (batch=2) = 4.15 ms per image
- Target: < 5 ms per image for 60 fps capability
- Baseline: 6.78 ms (UBM production on RTX 3080 Mobile)

**Deliverables:**
- [ ] YOLO12 INT8 engine transferred to car
- [ ] Inference benchmark results on RTX 4060
- [ ] Comparison to UBM baseline
- [ ] Real-time capability assessment (60 fps?)

**Files to Transfer:**
```
runs/detect/runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.engine
```

**Benchmark Script (run on car):**
```python
# benchmark_on_car.py (create similar to benchmark_int8.py)
# - Load best.engine on RTX 4060
# - Run 100 inference passes with stereo images
# - Report: mean, std, min, max latency
```

---

## 📋 Workshop Checklist

**✅ COMPLETED (2026-01-26):**
- [x] Transfer YOLO26 ONNX to ASU
- [x] Compile YOLO26 TensorRT FP16 engine on RTX 4060
- [x] Run inference benchmarks (YOLO26 + YOLOv11n)
- [x] Convert ROS bags to AVI videos (lidar1.avi, lidar2.avi)
- [x] Document performance (2.63 ms latency, 6.3× margin for 60 fps)

**Results:**
- ✅ **YOLO26n: 2.63 ms latency** on RTX 4060 (FP16)
- ✅ **2.58× faster** than UBM baseline (6.78 ms)
- ✅ **2.6% faster** than YOLOv11n production (2.70 ms)

**🔄 IN PROGRESS:**
- [ ] Create UBM test set from .avi videos (see workflow below)

---

## 🎯 UBM Test Set Creation Workflow (PRIORITY)

**Goal:** Create a real-world test set from car camera data for ongoing model evaluation

**Why Important:**
- FSOCO-12 is internet dataset (may not match real track conditions)
- Car data = ground truth for actual deployment
- Validates model on true edge cases (lighting, weather, motion blur)
- Becomes **permanent benchmark** for future model improvements

### Phase 1: Frame Extraction (30 minutes)

**Input:**
- `media/20_11_2025_Rioveggio_Test_LidarTest1.avi` - 2560×720, 60 FPS, 1454 frames (24.2s)
- `media/20_11_2025_Rioveggio_Test_LidarTest2.avi` - 2560×720, 60 FPS, 1374 frames (22.9s)

**Objective:** Extract ~46 stereo pairs (92 images total), split stereo images

**Sampling:** Every 60 frames = 1 second at 60 FPS = **2 seconds real-world time**

**Script:** `extract_frames_from_avi.py` ✅ READY

```bash
# Activate virtual environment
source venv/bin/activate

# Extract LidarTest1 frames (every 60 frames = 2 seconds real-world)
python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest1.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar1

# Extract LidarTest2 frames (every 60 frames = 2 seconds real-world)
python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest2.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar2
```

**Expected Output:**
- `ubm_test_set/images/lidar1_left_0000.jpg` through `lidar1_left_0023.jpg` (24 images)
- `ubm_test_set/images/lidar1_right_0000.jpg` through `lidar1_right_0023.jpg` (24 images)
- `ubm_test_set/images/lidar2_left_0000.jpg` through `lidar2_left_0021.jpg` (22 images)
- `ubm_test_set/images/lidar2_right_0000.jpg` through `lidar2_right_0021.jpg` (22 images)
- **Total: ~92 images (46 stereo pairs)**

**Calculation:**
- LidarTest1: 1454 frames / 60 = 24 stereo pairs
- LidarTest2: 1374 frames / 60 = 22 stereo pairs
- Each stereo pair = left + right image

### Phase 2: Roboflow Annotation (2-3 hours)

**Platform:** Roboflow (used by Alberto and Edoardo)

**Steps:**
1. **Create new project:** "UBM-Rioveggio-Test-2025"
2. **Upload images:** Drag-and-drop all ~92 extracted frames
3. **Configure classes:**
   - blue_cone
   - yellow_cone
   - orange_cone
   - large_orange_cone
   - unknown_cone
4. **Annotate cones:** Draw bounding boxes around all visible cones
5. **Quality check:** Review all annotations for accuracy
6. **Export dataset:** YOLOv11 PyTorch format

**Expected Annotations:**
- ~92 images × ~5-10 cones per image = **~500-900 total annotations**

**Tips:**
- Use keyboard shortcuts (faster annotation)
- Annotate in batches (left images first, then right)
- Mark occluded cones as "unknown_cone" if ambiguous
- Include challenging cases (distant cones, motion blur, shadows)
- Tight bounding boxes (minimize background)

### Phase 3: Dataset Integration (30 minutes)

**Download from Roboflow:**
```bash
# In ml4cv_project directory
mkdir -p datasets/UBM-Rioveggio-Test-2025
cd datasets/UBM-Rioveggio-Test-2025

# Download using Roboflow API
python3 -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('ubm').project('ubm-rioveggio-test-2025')
dataset = project.version(1).download('yolov11')
"
```

**Create `data.yaml`:**
```yaml
# UBM Rioveggio Test Set - Real Car Data (Nov 20, 2025)
path: datasets/UBM-Rioveggio-Test-2025
train: images  # Empty (test-only dataset)
val: images    # Empty
test: images   # All images here

nc: 5
names:
  0: blue_cone
  1: yellow_cone
  2: orange_cone
  3: large_orange_cone
  4: unknown_cone
```

### Phase 4: Model Evaluation (15 minutes)

**Evaluate all models on UBM test set:**

```bash
source venv/bin/activate

# Evaluate YOLO26n (best model)
python3 evaluate_yolo26_ubm_test.py

# Evaluate YOLO12n (comparison)
python3 evaluate_yolo12_ubm_test.py

# Evaluate YOLOv11n baseline (comparison)
python3 evaluate_baseline_ubm_test.py
```

**Expected Results:**
- Real-world mAP50 may be **lower** than FSOCO-12 (more challenging)
- Edge cases will be revealed (distant cones, motion blur, lighting)
- Becomes permanent benchmark for future improvements

### Phase 5: Documentation (30 minutes)

**Create:** `docs/UBM_TEST_SET_RESULTS.md`

**Include:**
- Per-model performance on UBM test set
- Comparison to FSOCO-12 test set
- Challenging cases analysis (false positives/negatives)
- Recommendations for model improvements

---

## 📋 UBM Test Set Checklist

**Phase 1: Extraction (~30 minutes)**
- [ ] Run `extract_frames_from_avi.py` on LidarTest1 (interval=60)
- [ ] Run `extract_frames_from_avi.py` on LidarTest2 (interval=60)
- [ ] Verify output: ~92 images (46 stereo pairs)
- [ ] Check image quality (no corruption, correct 1280×720 split)

**Phase 2: Annotation (~2-3 hours)**
- [ ] Create Roboflow project: "UBM-Rioveggio-Test-2025"
- [ ] Upload all ~92 images
- [ ] Configure 5 cone classes
- [ ] Annotate all cones (~500-900 annotations)
- [ ] Quality check all annotations
- [ ] Export YOLOv11 format

**Phase 3: Integration (~30 minutes)**
- [ ] Download dataset from Roboflow
- [ ] Create data.yaml with correct paths
- [ ] Verify dataset structure

**Phase 4: Evaluation (~15 minutes)**
- [ ] Evaluate YOLO26n on UBM test set
- [ ] Evaluate YOLO12n on UBM test set
- [ ] Evaluate YOLOv11n baseline on UBM test set

**Phase 5: Documentation (~30 minutes)**
- [ ] Document results: `UBM_TEST_SET_RESULTS.md`
- [ ] Compare FSOCO-12 vs UBM performance
- [ ] Analyze edge cases and failure modes
- [ ] Update project summary

**Estimated Total Time:** 4-5 hours

**See detailed guide:** `docs/UBM_TEST_SET_EXTRACTION.md`

---

## 🎓 Virtual Top-Down View (LOWEST PRIORITY)

**Status:** ❌ **Deprioritized**

**Reason:** Existing 3D visualization (see `media/Existing_3D_View.png`) already shows:
- 3D point cloud with cone positions
- Camera view with bounding boxes
- Sufficient for debugging and visualization

**Decision:** Focus on UBM test set creation instead (higher value)

---

## Current Tasks (Parallel to YOLO26 Training)

**While YOLO26 trains (~2.5 days):**

---

## ✅ COMPLETED

### 1. ✅ Download Correct Dataset
Downloaded `FSOCO-12` from `fmdv/fsoco-kxq3s` version 12 → `datasets/FSOCO-12/`

### 2. ✅ Modify `train_baseline.py` to Accept Dataset Argument
Script now accepts `--data`, `--epochs`, `--batch` arguments. Default: FSOCO-12.

### 3. ✅ Restart Baseline Training with Correct Dataset
**Currently running:** `python train_baseline.py` with FSOCO-12 dataset
- Log: `training_FSOCO_baseline.log`
- Expected completion: ~12 hours
- Target: mAP50 ≈ 0.824

---

### 4. ✅ Baseline Training Completed
**Validation Set Result:** mAP50 = 0.714 (Precision: 0.832, Recall: 0.657)
**Test Set Result:** mAP50 = 0.707 (Precision: 0.816, Recall: 0.662)
- 14.3% gap to Gabriele's test set baseline (0.824)
- Analysis: `BASELINE_RESULTS_ANALYSIS.md`
- Our baseline BEATS UBM production by 6.2% on test set

---

## Next Steps - Decision Required

### ✅ Decision Made: Option A - Hyperparameter Search

## 🔍 Critical Updates (2026-01-24)

### From Edoardo
1. **Thesis results (mAP50 = 0.824) came from Gabriele & Patta's training**, not Edo's own work
2. **They used a "particular dataset"** - exact details unknown to Edo
3. **Image brightness is a major problem** for cone type classification
4. **Orange cone size distinction** (normal vs large) is difficult at distance
5. **Real validation should be on car data**, not internet datasets like FSOCO

### From Gabriele's CV Report (Nov 2025) - 🔴 GAME CHANGER

**CRITICAL DISCOVERY: Dataset Split Mismatch!**

Gabriele's report reveals the **TRUE baseline** from their production system:

| Metric | Gabriele's Baseline | Dataset Split |
|--------|---------------------|---------------|
| **mAP50** | **0.824** | **Test set (689 images)** ✅ |
| Precision | 0.849 | Test set |
| Recall | 0.765 | Test set |
| mAP50-95 | 0.570 | Test set |

**✅ OUR TEST SET EVALUATIONS (COMPLETED 2026-01-25):**

| Model | mAP50 (Validation) | mAP50 (Test) | Gap from Gabriele |
|-------|-------------------|--------------|-------------------|
| Our baseline | 0.714 | **0.707** | -14.3% |
| UBM production | 0.670 | **0.666** | -19.2% |
| Gabriele's baseline | — | **0.824** | — |

**✅ Key Findings:**
- Our baseline on test: **0.7065 mAP50** (beats UBM by 6.2%)
- 14.3% gap to close to reach Gabriele's baseline
- Test set slightly harder than validation (1% drop)
- YOLO12 training is next step to close gap

**Inference performance (RTX 3080 Mobile + TensorRT):**
- Total pipeline: 9.46 ms (with ORB features)
- YOLO inference only: 6.78 ms
- Real-time capable at 60 fps

**See:** `GABRIELE_BASELINE_ANALYSIS.md` for full details

---

## Current Plan: YOLO12 Training (Branch A)

### ✅ Phase 1: Hyperparameter Sweep - STOPPED (2026-01-25)

**Results:** 10/21 runs completed, 61.9% crash rate
- **Best run:** 0.7088 mAP50 (WORSE than baseline 0.7140)
- **Mean of 10 runs:** 0.7030 mAP50
- **Conclusion:** Model performance agnostic to hyperparameters - Ultralytics defaults already near-optimal
- **Crash cause:** High mixup (>0.15) + high dropout (>0.12) = training instability

**See:** `SWEEP_CRASH_INVESTIGATION.md` and `SWEEP_ANALYSIS.md` for full analysis

**Decision:** Pivot to architectural improvements (YOLO12) instead of hyperparameter tuning

### ✅ Phase 2: Test Set Evaluation - COMPLETED (2026-01-25)

**🎯 CRITICAL RESULTS - Test Set Performance (689 images)**

| Model | mAP50 | mAP50-95 | Precision | Recall | Gap from Gabriele |
|-------|-------|----------|-----------|--------|-------------------|
| **Gabriele's baseline** (CV report) | **0.824** | 0.570 | 0.849 | 0.765 | — |
| **Our baseline** | **0.7065** | 0.4898 | 0.8164 | 0.6616 | **-0.1175 (-14.3%)** |
| **UBM production** | **0.6655** | 0.4613 | 0.8031 | 0.5786 | -0.1585 (-19.2%) |

**Key Findings:**
1. ✅ **Our baseline BEATS UBM production by 6.2%** (0.7065 vs 0.6655)
2. ⚠️ **14.3% gap to Gabriele's baseline** (0.1175 mAP50 to close)
3. ✅ Test set slightly harder than validation (our baseline: 0.714 val → 0.707 test, 1% drop)
4. ⚠️ NEITHER model achieves Gabriele's 0.824 on test set

**Evaluation Commands Run:**
```bash
python3 evaluate_baseline_test.py    # Our baseline on test
python3 evaluate_ubm_model.py         # UBM production on test
```

**Next Target:** Train YOLO12n to close 14.3% gap (target mAP50 ≥ 0.82)

### ✅ Phase 3: Train YOLO12n (Branch A) - COMPLETED (2026-01-25)

**Goal:** Close gap to proven baseline using 2025 state-of-the-art architecture

**Results:**
- Training time: 2.5 days (300 epochs, RTX 4080 Super)
- **Test mAP50: 0.7081** ✅
- **vs Our baseline: +0.2%** (0.7081 vs 0.7065)
- **vs UBM production: +6.4%** (0.7081 vs 0.6655)
- Inference: 4.1 ms on RTX 4080 Super

**YOLO12 Key Features:**
- Area Attention Mechanism (efficient self-attention)
- R-ELAN (Residual Efficient Layer Aggregation)
- FlashAttention integration

**Per-Class Performance (Test Set):**
```
Class              Precision   Recall    mAP50
blue_cone          0.912       0.738     0.804
large_orange_cone  0.912       0.821     0.871  ⭐ Best class
orange_cone        0.879       0.722     0.775
yellow_cone        0.890       0.727     0.796
unknown_cone       0.607       0.264     0.295  ⚠️ Challenging
```

**Decision:** ✅ **SUCCESS** - Proceed to INT8 optimization (Branch A)

### ✅ Phase 4: Evaluate YOLO12 on Test Set - COMPLETED

**Results:**
```bash
python3 evaluate_yolo12_test.py  # ✅ COMPLETED
```

**Test Set Performance (689 images):**
- **YOLO12n: 0.7081 mAP50** ✅
- Our YOLOv11n baseline: 0.7065 mAP50
- UBM production: 0.6655 mAP50

**Comparison to Proven Baselines:**
- vs UBM production: **+6.4%** (0.0426 mAP50 improvement)
- vs Our baseline: **+0.2%** (0.0016 mAP50 improvement)

**Success Criteria:** ✅ **MET** - Beats proven UBM baseline by 6.4%

**Log:** `yolo12_test_evaluation.log`

### 🚀 Phase 5: INT8 Optimization & Benchmarking (30 minutes) - NEXT STEP

**YOLO12 Training Complete:** ✅ Achieved 0.7081 mAP50 on test set (+6.4% vs UBM)

**Current Status:** Ready for INT8 optimization (Branch A)

**Deployment Target:**
- Hardware: RTX 4060 (on car)
- Baseline: 6.78ms inference (Gabriele, RTX 3080 Mobile, TensorRT FP16)
- Target: < 2.5ms on RTX 4060 (2.7× speedup minimum)

**INT8 Optimization Commands:**

**Option 1: Complete Pipeline (Recommended)**
```bash
python3 optimize_yolo12_int8.py
```

**Option 2: Step-by-Step**
```bash
# Step 1: Export to ONNX (optional)
python3 export_yolo12_onnx.py

# Step 2: Export to TensorRT INT8
# ⚠️ Uses VALIDATION SET for calibration, NOT test set!
python3 export_tensorrt_int8.py

# Step 3: Benchmark speed and accuracy
python3 benchmark_int8.py
```

**What INT8 Does:**
1. Evaluates FP32 baseline accuracy (validation set)
2. Exports to TensorRT INT8 (uses ~500 validation images for calibration)
3. Evaluates INT8 accuracy (validation set)
4. Benchmarks inference speed (100 runs)

**Expected Results:**
- Speed: 1.5-2.0× faster (~2.0-2.7 ms on RTX 4080 Super)
- Accuracy: <1% loss (0.7081 → ~0.702-0.706 mAP50)
- Size: 3.5× smaller (5.3 MB → 1.5 MB)
- **RTX 4060:** ~1.7-2.2 ms (3-4× faster than baseline!)

**Critical Notes:**
- ⚠️ Calibration uses VALIDATION SET only (NEVER test set!)
- ⚠️ Baseline (6.78ms) is ALREADY TensorRT FP16 optimized
- ✅ RTX 4060 has INT8 Tensor Cores (242 INT8 TOPS)

**See:** `INT8_OPTIMIZATION_GUIDE.md` for complete documentation

**Deliverables:**
- INT8 TensorRT engine (`best.engine`)
- Speed benchmark report (FP32 vs INT8)
- Accuracy comparison (validation set)
- Deployment recommendations for RTX 4060

### 🚀 Alternative: Train YOLO26 (Newest Architecture) - OPTIONAL

**Goal:** Test latest YOLO architecture (YOLO26, 2025) vs YOLO12

**Why YOLO26?**
- Latest Ultralytics architecture (available in 8.4.7)
- Similar parameters (2.57M) to YOLO12n (2.56M)
- May have architectural improvements over YOLO12

**Command:**
```bash
python3 train_yolo26.py  # 2.5 days training
```

**Expected Results:**
- **Best case:** 0.72-0.73 mAP50 (+2-3% vs YOLO12)
- **Moderate:** 0.70-0.71 mAP50 (similar to YOLO12)
- **Worst case:** 0.68-0.70 mAP50 (stick with YOLO12)

**Decision Point:**
- If YOLO26 > YOLO12 → Use YOLO26 for deployment
- If YOLO26 ≈ YOLO12 → Either works, choose simpler
- If YOLO26 < YOLO12 → Stick with YOLO12

**After Training:**
```bash
# Evaluate on test set
python3 evaluate_yolo26_test.py

# If better, export to INT8
python3 export_yolo26_tensorrt_int8.py
python3 benchmark_yolo26_int8.py
```

**See:** `YOLO26_TRAINING_GUIDE.md` for complete documentation

---

### Files Created

**YOLO26 Training (New):**
- ✅ `train_yolo26.py` - Train YOLO26n on FSOCO-12
- ✅ `evaluate_yolo26_test.py` - Test set evaluation
- ✅ `export_yolo26_onnx.py` - Export to ONNX
- ✅ `export_yolo26_tensorrt_int8.py` - Export to TensorRT INT8 (workspace=8GB, FP16 fallback)
- ✅ `benchmark_yolo26_int8.py` - Benchmark FP32 vs INT8
- ✅ `YOLO26_TRAINING_GUIDE.md` - Complete documentation

**Test Set Evaluation:**
- ✅ `evaluate_baseline_test.py` - Evaluate our baseline on test set
- ✅ `evaluate_ubm_model.py` - Evaluate UBM production on test set
- ✅ `evaluate_yolo12_test.py` - Evaluate YOLO12 on test set
- ✅ `yolo12_test_evaluation.log` - YOLO12 test results log

**YOLO12 Training (Branch A):**
- ✅ `train_yolo12.py` - Train YOLO12n (300 epochs, attention-centric architecture)
- ✅ `TWO_BRANCH_STRATEGY.md` - Complete implementation plan with decision tree
- ✅ `NEXT_STEPS_COMMANDS.md` - Command reference guide

**INT8 Optimization (Next Step):**
- ✅ `export_yolo12_onnx.py` - Export YOLO12 to ONNX format
- ✅ `export_tensorrt_int8.py` - Export to TensorRT INT8 engine
- ✅ `benchmark_int8.py` - Benchmark FP32 vs INT8 speed/accuracy
- ✅ `optimize_yolo12_int8.py` - Complete INT8 pipeline (all steps)
- ✅ `INT8_OPTIMIZATION_GUIDE.md` - Complete INT8 documentation

**Hyperparameter Sweep (Stopped):**
- ✅ `sweep_config.yaml` - Search space definition (13 hyperparameters)
- ✅ `train_sweep.py` - Training script for sweep
- ✅ `launch_sweep.sh` - One-command sweep launcher
- ✅ `analyze_sweep.py` - Extract best config from sweep results
- ✅ `SWEEP_CRASH_INVESTIGATION.md` - Why 61.9% of runs crashed (high mixup + dropout)
- ✅ `SWEEP_ANALYSIS.md` - Complete sweep results analysis

**Baseline Analysis:**
- ✅ `GABRIELE_BASELINE_ANALYSIS.md` - Ground truth from CV report (0.824 unverified)
- ✅ `BASELINE_RESULTS_ANALYSIS.md` - Our baseline analysis
- ✅ `UBM_MODEL_INFO.md` - UBM production model evaluation & comparison tables

**Utilities:**
- ✅ `wandb_api.py` - W&B API interface (fixed import shadowing)
- ✅ `benchmark_inference.py` - ONNX/TensorRT inference speed benchmarking
- ✅ `RESEARCH_FOCUS_AREAS.md` - Brightness & orange cone challenges

---

## Future Improvements

### 4. Model Experiments (Week 1-2)
After baseline is confirmed:
- Train YOLOv11s/m variants
- Test RT-DETR
- Experiment with image size (640 → 800/1024)
- Try different augmentations

### 5. Optimization (Week 2)
- ONNX export
- TensorRT conversion for RTX 4060
- Inference speed benchmarking

### 6. Report & Documentation
- Benchmark comparison table
- W&B run comparisons
- Final project report

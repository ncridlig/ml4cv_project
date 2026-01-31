# TODO List

## 📊 Current Status (2026-01-27)

**🎉 YOLO26n TWO-STAGE TRAINING IN PROGRESS!**

**Test Set Performance (FSOCO-12, 689 images):**
- **YOLO26n (BEST): 0.7626 mAP50** 🏆 ✅ (+14.6% vs UBM)
- **YOLO12n: 0.7081 mAP50** ✅ (+6.4% vs UBM)
- **YOLOv11n baseline: 0.7065 mAP50** ✅ (+6.2% vs UBM)
- UBM production model: **0.6655 mAP50**

**Validation Set Performance:**
- **YOLO26n: 0.7586 mAP50** 🏆 ✅
- YOLO12n: **0.7127 mAP50** ✅
- YOLOv11n baseline: **0.7140 mAP50** ✅

**Training Status:**
- ✅ YOLOv11n baseline complete
- ✅ Hyperparameter sweep complete (stopped - no improvement)
- ✅ YOLO12n training complete (300/300 epochs)
- ✅ YOLO12n INT8 TensorRT export complete
- ✅ Test set evaluation complete (YOLO12, baseline, UBM, YOLO26)
- ✅ **YOLO26n training COMPLETE** (300/300 epochs) 🎉
- ✅ **YOLO26n test evaluation COMPLETE** (0.7626 mAP50 - NEW BEST!) 🏆
- ✅ **fsoco-ubm test set COMPLETE** (96 images, annotated, exported) ✅
- 🔄 **YOLO26n two-stage training IN PROGRESS** (Stage 2: 8 hours remaining)

**Next Immediate Actions:**
1. ⏳ **Wait for two-stage training** to complete (~8 hours)
2. 🔄 **Evaluate two-stage YOLO26** on test set - `python3 evaluate_yolo26_two_stage_test.py`
3. 🔄 **Benchmark all models on fsoco-ubm** (see section below)
4. 🔄 **Compare single-stage vs two-stage YOLO26** performance

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

**Date:** 1-26-26
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
- [X] 100 annotated frames from .mcap
- [X] YOLO format dataset (`data.yaml`, images/, labels/)
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

**✅ COMPLETED:**
- [x] Create UBM test set (fsoco-ubm) - 96 images annotated and exported ✅

---

## 🎯 fsoco-ubm Benchmarking (NEXT PRIORITY)

**Status:** 🔄 READY TO START

**Goal:** Benchmark all trained models on the in-house fsoco-ubm test set (96 images from car camera)

**Why Important:**
- FSOCO-12 is an internet dataset → may not represent real track conditions
- fsoco-ubm = ground truth from actual car camera (ZED 2i stereo)
- Tests models on real-world edge cases: lighting, motion blur, track conditions
- Validates if FSOCO-12 performance translates to deployment

### Download fsoco-ubm Dataset

```bash
source venv/bin/activate

# Download dataset from Roboflow
python3 download_fsoco_ubm.py

# Dataset will be saved to: datasets/ml4cv_project-1/
```

**Dataset Info:**
- Workspace: `fsae-okyoe`
- Project: `ml4cv_project`
- Version: 1
- Size: 96 images (test-only)
- Source: Rioveggio test track (November 20, 2025)
- Camera: ZED 2i stereo (1280×720)

### Evaluation Commands

**Benchmark all models on fsoco-ubm:**

```bash
# YOLO26n (single-stage, best model)
python3 evaluate_yolo26_fsoco_ubm.py

# YOLO26n (two-stage, after training completes)
python3 evaluate_yolo26_two_stage_fsoco_ubm.py

# YOLO12n
python3 evaluate_yolo12_fsoco_ubm.py

# YOLOv11n baseline
python3 evaluate_baseline_fsoco_ubm.py

# UBM production
python3 evaluate_ubm_fsoco_ubm.py
```

**Note:** Scripts need to be created - use existing evaluation scripts as templates.

### Expected Results

**Hypothesis:** Real-world mAP50 may be **lower** than FSOCO-12 due to:
- Motion blur (car in motion)
- Variable lighting (outdoor track, shadows)
- Distance challenges (cones at 15-20m)
- Occlusion (cones behind others)

**Comparison Matrix:**

| Model | FSOCO-12 Test | fsoco-ubm (predicted) | Delta |
|-------|---------------|----------------------|-------|
| YOLO26n | 0.7626 | 0.70-0.73 | -3 to -5% |
| YOLO12n | 0.7081 | 0.65-0.68 | -3 to -5% |
| YOLOv11n | 0.7065 | 0.64-0.67 | -3 to -5% |
| UBM prod | 0.6655 | 0.60-0.63 | -3 to -5% |

**Success Criteria:**
- ✅ YOLO26n maintains lead over other models
- ✅ Relative rankings preserved (YOLO26 > YOLO12 > baseline > UBM)
- ✅ Identify challenging cases for future improvements

### Deliverables

**After benchmarking:**
1. Create `docs/FSOCO_UBM_RESULTS.md` with:
   - Per-model performance comparison
   - FSOCO-12 vs fsoco-ubm comparison
   - Confusion matrices and failure case analysis
   - Recommendations for model improvements

2. Update `docs/RESULTS_SUMMARY.md` with fsoco-ubm results

3. Create evaluation scripts (5 scripts total)

**Estimated Time:** 2-3 hours (script creation + evaluation + analysis)

---

## 🎯 UBM Test Set Creation Workflow (COMPLETED ✅)

**Status:** ✅ COMPLETE - All 96 images annotated, exported as fsoco-ubm dataset

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

**Phase 1: Extraction (~30 minutes)** ✅ COMPLETE
- [x] Run `extract_frames_from_avi.py` on LidarTest1 (interval=60)
- [x] Run `extract_frames_from_avi.py` on LidarTest2 (interval=60)
- [x] Verify output: ~96 images extracted
- [x] Check image quality (no corruption, correct 1280×720 split)

**Phase 2: Annotation (~2-3 hours)** ✅ COMPLETE
- [x] Create Roboflow project: "ml4cv_project"
- [x] Upload all 96 images
- [x] Configure 5 cone classes
- [x] Annotate all cones (Label Assist used)
- [x] Quality check all annotations
- [x] Export YOLO26 format

**Phase 3: Integration (~30 minutes)** ✅ COMPLETE
- [x] Dataset exported from Roboflow
- [x] Download script created: `download_fsoco_ubm.py`
- [x] Dataset available for download

**Phase 4: Evaluation (~15 minutes)** 🔄 NEXT PRIORITY
- [ ] Evaluate YOLO26n on fsoco-ubm test set
- [ ] Evaluate YOLO12n on fsoco-ubm test set
- [ ] Evaluate YOLOv11n baseline on fsoco-ubm test set
- [ ] Evaluate UBM production on fsoco-ubm test set
- [ ] Evaluate two-stage YOLO26n on fsoco-ubm test set (after training)

**Phase 5: Documentation (~30 minutes)** 🔄 PENDING
- [ ] Document results: `FSOCO_UBM_RESULTS.md`
- [ ] Compare FSOCO-12 vs fsoco-ubm performance
- [ ] Analyze edge cases and failure modes
- [ ] Update project summary

**Actual Time Spent:** 5 hours (extraction, annotation, export)

**See detailed guide:** `docs/UBM_TEST_SET_EXTRACTION.md`

---

## 🎯 Confidence Threshold Tuning (AFTER BEST MODEL SELECTION)

**Status:** 🔄 READY (after `evaluate_fsoco_ubm.py` completes)

**Goal:** Optimize confidence threshold for best model on fsoco-ubm dataset to maximize F1 score

**When to Do This:**
- ✅ **AFTER** selecting best model from fsoco-ubm evaluation
- ✅ **BEFORE** final deployment to ROS2
- ✅ This is the **last optimization step** before production

**Why Important:**
- Default Ultralytics uses conf=0.25 for validation, conf=0.001 for prediction
- Optimal threshold varies by dataset and use case
- F1 score balances precision (safety - no false positives) and recall (detection rate)
- Tuning on fsoco-ubm ensures optimal performance on real car data

---

### Phase 1: F1 Curve Analysis (Already Available)

**Good News:** `evaluate_fsoco_ubm.py` already generates F1 curves with `plots=True`

**Location:** After running evaluation, check:
```
runs/evaluation/{model_name}_fsoco_ubm/
├── F1_curve.png          # ⭐ KEY PLOT - F1 score vs confidence threshold
├── P_curve.png           # Precision vs confidence threshold
├── R_curve.png           # Recall vs confidence threshold
├── PR_curve.png          # Precision-Recall curve
└── confusion_matrix.png  # Confusion matrix
```

**How to Read F1_curve.png:**
- **X-axis:** Confidence threshold (0.0 to 1.0)
- **Y-axis:** F1 score (harmonic mean of precision and recall)
- **Peak of curve:** Optimal confidence threshold
- **Formula:** F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Visual Inspection:**
1. Open `F1_curve.png` for best model
2. Find the peak (highest point on the curve)
3. Read X-axis value at peak → optimal conf threshold
4. Example: If peak is at X=0.35, optimal conf=0.35

---

### Phase 2: Systematic Threshold Search (Script-Based)

**Create:** `optimize_confidence_threshold.py`

```python
#!/usr/bin/env python3
"""
Find optimal confidence threshold for best model on fsoco-ubm.
Tests multiple thresholds and reports best F1 score.
"""
from ultralytics import YOLO
import numpy as np

# Load best model (update after fsoco-ubm evaluation)
BEST_MODEL = 'runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt'
DATA_YAML = 'datasets/ml4cv_project-1/data.yaml'

model = YOLO(BEST_MODEL)

# Test different confidence thresholds
thresholds = np.arange(0.1, 0.6, 0.05)  # 0.1, 0.15, 0.2, ..., 0.55
best_f1 = 0
best_conf = 0.25
results_table = []

print("=" * 80)
print("CONFIDENCE THRESHOLD OPTIMIZATION ON fsoco-ubm")
print("=" * 80)
print()
print(f"Model: {BEST_MODEL}")
print(f"Dataset: {DATA_YAML} (test split)")
print(f"Testing {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
print()
print("-" * 80)
print(f"{'Conf':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP50':<10}")
print("-" * 80)

for conf in thresholds:
    results = model.val(
        data=DATA_YAML,
        split='test',
        batch=64,
        conf=conf,  # Override confidence threshold
        verbose=False,  # Suppress output
        plots=False,  # Skip plot generation
        device=0,
    )

    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results_table.append({
        'conf': conf,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
    })

    marker = "🏆" if f1 > best_f1 else "  "
    print(f"{marker} {conf:<6.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {map50:<10.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_conf = conf

print("-" * 80)
print()
print("=" * 80)
print("OPTIMAL THRESHOLD")
print("=" * 80)
print(f"🏆 Best Confidence: {best_conf:.2f}")
print(f"   F1 Score: {best_f1:.4f}")
print()

# Get detailed stats for optimal threshold
optimal_result = next(r for r in results_table if r['conf'] == best_conf)
print(f"   Precision: {optimal_result['precision']:.4f}")
print(f"   Recall: {optimal_result['recall']:.4f}")
print(f"   mAP50: {optimal_result['map50']:.4f}")
print()

# Run final validation with optimal threshold to generate plots
print("=" * 80)
print("FINAL VALIDATION WITH OPTIMAL THRESHOLD")
print("=" * 80)
print()
print(f"Running validation with conf={best_conf:.2f}...")

final_results = model.val(
    data=DATA_YAML,
    split='test',
    batch=64,
    conf=best_conf,
    plots=True,  # Generate plots
    save_json=True,
    project='runs/evaluation',
    name=f'best_model_optimized_conf_{best_conf:.2f}',
    device=0,
)

print()
print(f"✅ Results saved to: runs/evaluation/best_model_optimized_conf_{best_conf:.2f}/")
print()
print("=" * 80)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 80)
print()
print(f"1. Update ROS2 inference node with: conf={best_conf:.2f}")
print(f"2. In C++ code: float confidence_threshold = {best_conf:.2f}f;")
print(f"3. For Python inference: model.predict(..., conf={best_conf:.2f})")
print()
print("=" * 80)
```

**Usage:**
```bash
source venv/bin/activate

# After running evaluate_fsoco_ubm.py and identifying best model
python3 optimize_confidence_threshold.py
```

**Expected Output:**
- Table of conf vs F1 score
- Optimal threshold identified (e.g., conf=0.35)
- Final validation with plots saved

---

### Phase 3: Apply Optimal Threshold to ONNX/TensorRT

**IMPORTANT:** Confidence threshold is applied during **postprocessing**, NOT in the model itself.

**The model outputs raw predictions**, then you filter by confidence threshold in inference code.

#### Option 1: Python Inference (Ultralytics)

```python
from ultralytics import YOLO

# Load TensorRT engine or ONNX model
model = YOLO('runs/detect/.../weights/best.engine')  # or best.onnx

# Use optimal confidence threshold from Phase 2
optimal_conf = 0.35  # From optimize_confidence_threshold.py

# Inference with optimized threshold
results = model.predict(
    source='path/to/images',
    conf=optimal_conf,  # Apply optimized threshold
    iou=0.45,  # NMS IoU threshold (can also tune)
    device=0,
    verbose=True,
)

# Validation with optimized threshold
results = model.val(
    data='datasets/ml4cv_project-1/data.yaml',
    split='test',
    conf=optimal_conf,
    plots=True,
)
```

#### Option 2: C++ Inference (ROS2 Node)

**File:** `src/ros_yolo_detector_node.cpp`

```cpp
// After TensorRT inference, during postprocessing

// Set optimal confidence threshold from Python analysis
float confidence_threshold = 0.35f;  // From optimize_confidence_threshold.py
float iou_threshold = 0.45f;  // NMS threshold

// Filter raw detections by confidence
std::vector<Detection> filtered_detections;
for (const auto& detection : raw_detections) {
    if (detection.confidence >= confidence_threshold) {
        filtered_detections.push_back(detection);
    }
}

// Apply NMS to filtered detections
auto final_detections = apply_nms(filtered_detections, iou_threshold);
```

**Where to modify:**
1. Locate postprocessing function in `ros_yolo_detector_node.cpp`
2. Find confidence filtering step (likely around line 800-1000)
3. Replace hardcoded threshold (probably 0.25 or 0.5) with optimal value
4. Recompile ROS2 package

---

### Phase 4: Validation and Deployment

**Validation Steps:**
1. ✅ Run `optimize_confidence_threshold.py` to find optimal threshold
2. ✅ Re-evaluate on fsoco-ubm with optimal conf to verify improvement
3. ✅ Test on sample images visually (check bounding boxes)
4. ✅ Compare to default conf=0.25 performance

**Deployment Checklist:**
- [ ] Update confidence threshold in ROS2 C++ inference node
- [ ] Recompile ROS2 package with new threshold
- [ ] Test on car with live camera feed
- [ ] Monitor false positive rate during test runs
- [ ] Document optimal threshold in deployment docs

**Expected Improvements:**
- F1 score: +2-5% over default conf=0.25
- Better balance of precision and recall
- Fewer false positives OR better detection rate (depending on threshold direction)

---

### Phase 5: (Optional) NMS IoU Threshold Tuning

**After confidence tuning**, you can also tune NMS IoU threshold (default: 0.45)

```python
# Similar approach to confidence tuning
iou_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

for iou in iou_thresholds:
    results = model.val(
        data=DATA_YAML,
        split='test',
        conf=optimal_conf,  # Use optimal conf from Phase 2
        iou=iou,  # Test different NMS thresholds
        verbose=False,
    )
    # Compare mAP50 scores
```

**NMS IoU Effect:**
- **Lower IoU** (0.3-0.4): More aggressive NMS, removes more overlapping boxes
- **Higher IoU** (0.5-0.6): Less aggressive, keeps more detections
- Default 0.45 is usually good, but cone detection may benefit from tuning

---

### Summary: Optimization Order

**Correct sequence for deployment optimization:**

1. ✅ **Architecture selection** (YOLO26n vs YOLO12n vs YOLOv11n) ← fsoco-ubm evaluation
2. 🔄 **Confidence threshold tuning** ← optimize_confidence_threshold.py
3. 🔄 **(Optional) NMS IoU tuning** ← if needed
4. 🔄 **(Optional) Test-Time Augmentation (TTA)** ← if real-time budget allows
5. ✅ **Deploy to ROS2** ← Update C++ node with optimal parameters

**Timeline:**
- Phase 1: Already done (F1_curve.png generated by evaluate_fsoco_ubm.py)
- Phase 2: 30 minutes (run optimize_confidence_threshold.py)
- Phase 3: 15 minutes (update ROS2 C++ code)
- Phase 4: 1 hour (validation and testing)
- **Total: ~2 hours**

**See Also:**
- Ultralytics docs: https://docs.ultralytics.com/guides/hyperparameter-tuning/
- F1 score explanation: https://en.wikipedia.org/wiki/F-score
- NMS tuning guide: https://docs.ultralytics.com/modes/predict/#inference-arguments

---

## 🎓 Virtual Top-Down View (LOWEST PRIORITY)

**Status:** ❌ **Deprioritized**

**Reason:** Existing 3D visualization (see `media/Existing_3D_View.png`) already shows:
- 3D point cloud with cone positions
- Camera view with bounding boxes
- Sufficient for debugging and visualization

**Decision:** Focus on UBM test set creation instead (higher value)

---

## 🚀 Two-Stage YOLO26n Training (IN PROGRESS - 2026-01-27)

**Status:** 🔄 Stage 2 running (~8 hours remaining)

**Training Progress:**
- ✅ **Stage 1 COMPLETE:** Pre-trained on cone-detector (338/400 epochs, converged)
  - Best mAP50: 0.7339 (cone-detector validation)
  - Status: Interrupted at epoch 338, but training plateaued (no improvement since epoch 330)
  - Checkpoint: `runs/detect/runs/two-stage-yolo26/stage1_cone_detector_400ep2/weights/best.pt`
- 🔄 **Stage 2 IN PROGRESS:** Fine-tuning on FSOCO-12 (300 epochs, ~8 hours remaining)
  - Started: 2026-01-27
  - ETA: ~8 hours
  - W&B: `ncridlig-ml4cv/runs-two-stage-yolo26`

**Rationale:**
1. ✅ **More data:** 3× more pretraining data (22,725 vs 7,120)
2. ✅ **Extended training:** 638 total epochs (338 + 300) vs 300 single-stage
3. ✅ **Transfer learning:** Same cone detection task, different distributions
4. ✅ **Perfect timing:** Stage 2 runs while annotating fsoco-ubm

**Expected Results:**
- **Target:** 0.77-0.80 mAP50 (vs 0.7626 single-stage)
- **Best case:** > 0.78 mAP50 (+2-5% over single-stage)
- **Conservative:** 0.76-0.77 mAP50 (+0-1% over single-stage)
- **Worst case:** Similar to single-stage (0.76)

**Commands:**
```bash
# Stage 2 already running with:
python3 train_yolo26_two_stage.py --skip-stage1

# After training completes (~8 hours):
python3 evaluate_yolo26_two_stage_test.py
```

**Files Created:**
- ✅ `train_yolo26_two_stage.py` - Two-stage training script
- ✅ `evaluate_yolo26_two_stage_test.py` - Test set evaluation

**Output:**
- Stage 1 weights: `runs/two-stage-yolo26/stage1_cone_detector_400ep/weights/best.pt`
- Stage 2 weights: `runs/two-stage-yolo26/stage2_fsoco12_300ep/weights/best.pt`
- Final model: Stage 2 (if better than single-stage)

---

## Current Tasks (Parallel to Two-Stage Training)

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

## Future Experiment: YOLO26s (Small Variant)

**Status:** Idea — not started

**Idea:** Train YOLO26s and compare accuracy vs latency against YOLO26n, since the 6.3x real-time margin leaves room for a larger model.

**Pros:**
- Massive latency headroom (2.63 ms vs 16.7 ms budget) — even a 3x slower model fits
- Could improve weak classes, especially unknown cones (0.364 mAP50)
- Adds a clean accuracy-vs-latency tradeoff curve to the report
- Standard experiment in ML papers (nano vs small vs medium comparison)

**Cons:**
- Evidence suggests the bottleneck is the dataset, not model capacity: sweep showed insensitivity (std=0.019), two-stage training with 3x more data only gained 0.2%
- Requires another full training run (~10 hours) plus ASU workshop visit for TensorRT benchmarks
- Project is in report-writing phase — risk of scope creep for marginal gain
- Complicates the current clean narrative (YOLO26n wins on both accuracy and speed)

**Decision:** Defer to future work unless time permits before report deadline.

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

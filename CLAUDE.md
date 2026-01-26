# ML4CV 3CFU Project: Cone Detection Improvement

---

## 📝 DOCUMENTATION POLICY

**⚠️ IMPORTANT: To keep the repository clean and organized:**

1. **All project documentation, analysis, results, and notes should be added to THIS FILE (CLAUDE.md)**
2. **Additional .md files are stored in `docs/` folder** - not in root directory
3. **When documenting new findings, updates, or results, append them to the relevant section in CLAUDE.md**
4. **Keep root directory clean** - only essential scripts and this documentation file

**Current documentation structure:**
- `CLAUDE.md` - Main project documentation (THIS FILE)
- `docs/` - Archive of detailed analysis documents
- `TODO.md` - Active task tracking (separate for workflow)

---

**USE THE VENV FOR ALL PYTHON FILES**

## Project Overview

**Goal:** Improve the real-time cone detection and color classification pipeline for UniBo Motorsport's autonomous race car (Formula SAE Driverless competition).

**Timeline:** 2 weeks (75 hours) - Final project before master's thesis submission

**Academic Context:** 3 CFU project for ML4CV course, Professor Samuele Salti, University of Bologna

**Competition:** Formula Student Germany - Driverless (track bordered by blue, yellow, and orange cones)

---

## Repository

| Repo | Purpose | URL |
|------|---------|-----|
| **ubm-yolo-detector** | YOLO cone detection + ROS2 integration | https://github.com/ubm-driverless/ubm-yolo-detector |

**Local clone:** `/Users/nicolas/Desktop/Fall 2024/ML4CV/3cfuProject/ubm-yolo-detector`

---

## Baseline Performance (from Edo Fusa's Thesis)

### Model: YOLOv11n (300 epochs, FSOCO dataset)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All Classes** | 0.849 | 0.765 | 0.824 | 0.570 |
| Blue Cone | 0.922 | 0.806 | 0.896 | 0.615 |
| Yellow Cone | 0.926 | 0.794 | 0.892 | 0.608 |
| Orange Cone | 0.925 | 0.787 | 0.877 | 0.603 |
| Large Orange Cone | 0.868 | 0.873 | 0.908 | 0.710 |
| Unknown Cone | 0.603 | 0.566 | 0.547 | 0.315 |

### Test Dataset (FSOCO)
- 1,968 images, 36,123 cone instances
- Blue: 12,720 | Yellow: 15,605 | Orange: 5,462 | Large Orange: 1,263 | Unknown: 1,073

### Inference Timing (RTX 3080 Mobile, TensorRT)

| Stage | Center Point | ORB Features |
|-------|--------------|--------------|
| Preprocessing | 0.34 ms | 0.34 ms |
| **Inference** | **6.78 ms** | **6.78 ms** |
| Postprocessing | 0.38 ms | 0.38 ms |
| BBox Matching | 0.38 ms | 0.40 ms |
| Feature Matching | 0.00 ms | 1.41 ms |
| Triangulation | 0.02 ms | 0.11 ms |
| **Total** | **7.95 ms** | **9.46 ms** |

Real-time capable: 60 fps requires < 16.7 ms per frame

---

## Known Issues (from Thesis)

1. **Auto-exposure problem:** Bright sky causes underexposure of cones (Section 5.2.2)
2. **Unknown Cone class:** Poor performance (mAP50 = 0.547) - ambiguous/occluded cones
3. **False positives:** Example: person detected as yellow cone (mentioned in original proposal)
4. **Depth accuracy degrades at distance:**
   - 5m: ±0.19m error
   - 10m: ±0.46m error
   - 15m: ±1.16m error
   - 20m: ±1.37m error

---

## Dataset

### CORRECT Dataset (from thesis training notebook)
- **Source:** Roboflow
- **Workspace:** `fmdv`
- **Project:** `fsoco-kxq3s`
- **Version:** 12
- **Download script:** `python download_fsoco.py`

```python
# Download dataset (same as thesis)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fmdv").project("fsoco-kxq3s")
version = project.version(12)
dataset = version.download("yolov11")
```

### WRONG Dataset (do NOT use)
- **Workspace:** `fsbdriverless`
- **Project:** `cone-detector-zruok`
- **Version:** 1
- **Size:** 22,725 images
- **Problem:** Training on this dataset plateaus at mAP50 ≈ 0.68, far below thesis baseline (0.824)

---

## Existing Models

Located in `ubm-yolo-detector/yolo/models/`:

| Model | Resolution | Size | Date |
|-------|------------|------|------|
| YOLOv11n | 640x640 | 5.2 MB | 2025-06-18 |
| YOLOv12n | 1280x1280 | 5.4 MB | 2025-07-02 |
| OpenVINO FP32 | 640x640 | 10 MB | 2025-02-27 |

---

## 2-Week Project Scope (Plan A)

### Week 1: Model Evaluation & Training
1. Set up environment, download dataset via Roboflow API
2. Research alternative models (YOLOv11 variants, RT-DETR, etc.)
3. Fine-tune 2-3 candidate models
4. Evaluate: mAP, inference speed, false positive rate
5. Select best model with documented rationale

### Week 2: Optimization & Features
1. ONNX export and optimization for RTX 4060
2. (Optional) Virtual top-down view visualization
3. Color recognition improvements for lighting invariance
4. Final testing, documentation, report

### Deliverables
1. Benchmark report comparing vision models
2. Fine-tuned model with improved metrics
3. ONNX-optimized model with RTX 4060 benchmarks
4. (Optional) Top-down view visualization tool
5. Final project report

---

## Setup Instructions

```bash
cd /Users/nicolas/Desktop/Fall\ 2024/ML4CV/3cfuProject/ubm-yolo-detector

# For training (use existing notebook or create new)
pip install ultralytics roboflow torch torchvision

# Download dataset
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('fsbdriverless').project('cone-detector-zruok')
dataset = project.version(1).download('yolov11')
"

# Train
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
results = model.train(data='path/to/data.yaml', epochs=300)

# Export to ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx batch=2

# Export to TensorRT (on target hardware)
trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine
```

---

## Hardware Context

### Development Machines

| Machine | Purpose | GPU | OS |
|---------|---------|-----|-----|
| M1 MacBook | Debugging, code dev | CPU only | macOS |
| Ubuntu Workstation | Training & testing | RTX 4080 Super (CUDA) | Ubuntu |

**Code should run on both:** CPU mode for debugging on Mac, CUDA for training on Ubuntu.

### Target Deployment
**NVIDIA RTX 4060** (onboard ASU - Autonomous System Unit)

### Camera: ZED 2i Stereo
- Resolution: 2x 1280x720 @ 60fps
- FOV: 72° (H) x 44° (V)
- Baseline: 12 cm
- Depth range: 1.5m - 20m (accuracy degrades with distance)

**Current inference:** TensorRT primary, OpenVINO fallback

---

## Key Files

| File | Purpose |
|------|---------|
| `training/fsae-ev-driverless-yolo-training.ipynb` | Training notebook (Kaggle) |
| `yolo/models/` | Pre-trained model weights |
| `src/ros_yolo_detector_node.cpp` | ROS2 inference node (1,260 lines) |
| `README.md` | Full pipeline documentation |

---

## Reference Materials

- **Edo Fusa's Thesis:** "Pushing Cars' Limits: Exploring Autonomous Technologies in the Formula SAE Driverless Competition" (2024-2025)
  - Section 5.2: Stereocamera Pipeline
  - Section 5.2.3: Cone Detection with YOLO
  - Section 5.2.6: Experiments and Results
  - Local copy: `/Users/nicolas/Desktop/Fall 2024/ML4CV/3cfuProject/AI_Master_Thesis___Edoardo_Fusa_Stereo_Camera_Pipeline.pdf`

- **Roboflow Docs:** [docs.roboflow.com](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

---

## Success Metrics

**Improvement targets over baseline:**

| Metric | Baseline | Target |
|--------|----------|--------|
| mAP50 (All) | 0.824 | > 0.85 |
| Precision | 0.849 | > 0.90 |
| False positive rate | ~15% | < 10% |
| Inference (RTX 4060) | ~7ms | < 10ms |

---

## Notes

- Training was done with default Ultralytics parameters (lost by Enrico)
- Gabriele did most of the YOLO work
- Top-down view is nice-to-have, not critical
- Focus on reducing false positives (the person-as-cone issue)

---

## Time Log

| Date | Duration | Work Done |
|------|----------|-----------|
| 2026-01-21 | ~1.5 hrs | Project setup: cloned repos, downloaded dataset (22,725 images), created venv, wrote CLAUDE.md, IMPROVEMENT_TARGETS.md, extracted baseline metrics from Edo's thesis |
| 2026-01-23 | ~1 hr | Discovered dataset mismatch via W&B monitoring (training plateaued at mAP50=0.68 vs expected 0.824). Created `wandb_api.py` and `download_fsoco.py`. Identified correct dataset: `fmdv/fsoco-kxq3s` version 12. |

---

## Session Status (Last Updated: 2026-01-25)

### ✅ Baseline Training COMPLETED

**Training Run 2 (CORRECT dataset - COMPLETED):**
- Dataset: `FSOCO-12` from `fmdv/fsoco-kxq3s` version 12
- Status: **Completed 300 epochs successfully**
- **Result: mAP50 = 0.714** (vs thesis target: 0.824)
- Gap: -13.4% below thesis baseline
- Model: `runs/detect/runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt`

**Critical Finding:**
- Thesis baseline (mAP50 = 0.824) **NOT reproduced**
- Original hyperparameters were LOST - we used Ultralytics defaults
- Our baseline (mAP50 = 0.714) is valid and reproducible
- **Recommendation:** Use 0.714 as NEW baseline for improvement experiments

**Update from Gabriele (2026-01-24):**
- Provided actual production weights AND training notebook!
- Location: `/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt`
- Training notebook: `ubm-yolo-detector/training/fsae-ev-driverless-yolo-training.ipynb`

**🏆 CRITICAL FINDING:**
- UBM notebook (200 epochs): mAP50 = 0.663
- Our baseline (300 epochs): mAP50 = **0.714** (+7.7% better!)
- **We already BEAT their published notebook results!**
- Mystery: Model filename says "300ep" but notebook shows 200 epochs
- **TODO:** Evaluate production `best.pt` to see if it matches ours or is better

**See:** `BASELINE_RESULTS_ANALYSIS.md` for full analysis and recommendations

### W&B Monitoring
Use `wandb_api.py` to check training progress:
```bash
python wandb_api.py ncridlig-ml4cv/runs-baseline/<RUN_ID> --all
```

---

## Key Files Modified

### `train_baseline.py` - Now accepts arguments
```bash
# Use default (FSOCO-12 correct dataset)
python train_baseline.py

# Override dataset
python train_baseline.py --data datasets/cone-detector/data.yaml --epochs 100 --batch 32
```

**Defaults:** FSOCO-12 dataset, 300 epochs, batch 64, W&B enabled

### After Baseline Training Completes
- Compare results against thesis baseline (see table in "Baseline Performance" section)
- If mAP50 matches (~0.824), proceed with model improvement experiments
- Log results to W&B for tracking

---

### Completed
- [x] Baseline training completed: mAP50 = 0.714 (FSOCO-12)
- [x] Contacted Edoardo - confirmed thesis results from Gabriele/Patta's work
- [x] Set up W&B automated hyperparameter sweep (Bayesian optimization)
- [x] Identified key challenges: brightness robustness, orange cone classification
- [x] Created comprehensive sweep infrastructure (13 hyperparameters, 20 runs)

### In Progress
- [ ] W&B sweep running (~15-20 hours, targeting mAP50 > 0.75)

### Next: After Sweep Completes
1. Analyze sweep results (focus on brightness augmentation, orange cone performance)
2. Extract best config and train for 300 epochs
3. Architecture comparison (YOLOv11s, YOLOv11m)
4. Robustness analysis and real-world validation recommendations

---

## 🎯 Current Progress (2026-01-26)

### 🏆 PROJECT COMPLETE: YOLO26n Deployed on ASU!

**Final Test Set Performance (FSOCO-12, 689 images):**
- **YOLO26n: 0.7626 mAP50** 🏆 NEW BEST MODEL
- **vs UBM production: +14.6%** (0.7626 vs 0.6655)
- **vs YOLO12n: +7.7%** (0.7626 vs 0.7081)
- **vs YOLOv11n: +7.9%** (0.7626 vs 0.7065)

**ASU Deployment Performance (RTX 4060, TensorRT FP16):**
- **Latency: 2.63 ms** (mean)
- **GPU Compute: 1.019 ms**
- **Throughput: 633 qps (queries per second)**
- **vs UBM baseline: 2.58× faster** (2.63 ms vs 6.78 ms)
- **Real-time margin: 6.3× for 60 fps** (2.63 ms << 16.7 ms budget)

**YOLO26n Model Specifications:**
- Architecture: YOLO26n (2025, latest Ultralytics)
- Parameters: 2,505,750 (2.51M)
- GFLOPs: 5.780
- Precision: 0.8485 (highest for safety-critical)
- Recall: 0.6935 (good detection rate)
- TensorRT FP16: 9.35 MB

**Per-Class Performance (Test Set):**
- Large Orange Cone: **0.886 mAP50** ⭐
- Blue Cone: **0.863 mAP50**
- Yellow Cone: **0.856 mAP50**
- Orange Cone: **0.843 mAP50**
- Unknown Cone: 0.364 mAP50 ⚠️

**W&B Run Details:**
- Project: `ncridlig-ml4cv/runs-yolo26`
- Run ID: `yolo26n_300ep_FSOCO_20260125_122257`
- URL: https://wandb.ai/ncridlig-ml4cv/runs-yolo26/runs/yolo26n_300ep_FSOCO_20260125_122257
- Training time: 300 epochs, batch 64, ~2.5 days on RTX 4080 Super

### ✅ Major Achievements

**Accuracy:**
1. ✅ **YOLO26n is BEST model:** 0.7626 mAP50 (+14.6% over UBM production)
2. ✅ **YOLO26 beats YOLO12 by 7.7%** on test set (0.7626 vs 0.7081)
3. ✅ **All classes improved:** Blue (+5.9%), Yellow (+6.0%), Orange (+6.8%), Unknown (+23.4%)
4. ✅ **Highest precision:** 0.8485 (safety-critical for autonomous racing)

**Speed:**
1. ✅ **Fastest inference:** 2.63 ms on RTX 4060 (FP16 TensorRT)
2. ✅ **2.58× faster than UBM** baseline (6.78 ms → 2.63 ms)
3. ✅ **2.6% faster than YOLOv11n** production (2.70 ms → 2.63 ms)
4. ✅ **Real-time capable:** 6.3× margin for 60 fps

**Deployment:**
1. ✅ **TensorRT engine built on ASU** (RTX 4060, FP16)
2. ✅ **Engine portable:** Built on RTX 4080 Super, works on RTX 4060
3. ✅ **ROS bags converted:** lidar1.avi + lidar2.avi for test set creation
4. ✅ **Production ready:** Ready for ROS2 integration

**Research:**
1. ✅ **Hyperparameter sweep ineffective** - defaults already optimal
2. ✅ **2025 architectures successful** - YOLO12 and YOLO26 both beat baseline
3. ✅ **INT8 not needed** - FP16 already exceeds requirements (6.3× margin)
4. ⚠️ **Gabriele's 0.824 claim unverified** - UBM production can't reproduce it

### 🔄 Next Steps: UBM Test Set Creation

**Priority:** Create real-world test set from car camera data

**Steps:**
1. ✅ Convert ROS bags to AVI videos (lidar1.avi, lidar2.avi) - DONE
2. 🔄 Extract frames from videos (~100-200 stereo pairs)
3. 🔄 Split stereo images (left/right separation)
4. 🔄 Upload to Roboflow for annotation
5. 🔄 Label cones (5 classes)
6. 🔄 Download YOLO format dataset
7. 🔄 Evaluate YOLO26 on UBM test set

**Videos ready in `/media`:**
- `20_11_2025_Rioveggio_Test_LidarTest1.avi` (203 MB, 1454 frames, 60 FPS, 24.2s)
- `20_11_2025_Rioveggio_Test_LidarTest2.avi` (194 MB, 1374 frames, 60 FPS, 22.9s)

**Script created:** `extract_frames_from_avi.py` (configured for 60 FPS)

```bash
source venv/bin/activate

# Extract every 60 frames (1 second at 60fps = 2 seconds real-world time)
python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest1.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar1

python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest2.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar2
```

**Expected output:** ~46 stereo pairs (92 images total)

**See:**
- `docs/UBM_TEST_SET_EXTRACTION.md` - Complete extraction guide
- `docs/TODO.md` - Full test set creation workflow

### 📅 Meeting with Alberto (Workshop)

**Goal 1: Create Custom Test Set from .mcap Data**
- Extract 100 frames from car's camera recordings
- Convert to YOLO format for real-world validation
- Validate model on actual deployment conditions

**Goal 2: Benchmark YOLO12 INT8 on RTX 4060**
- Transfer TensorRT engine to car
- Measure real-world inference speed (target: < 5 ms per image)
- Compare to baseline (6.78 ms on RTX 3080 Mobile)

### 📊 Repository Organization

**Root Directory (Clean):**
- `CLAUDE.md` - Main documentation (this file)
- `TODO.md` - Active task tracking
- `*.py` - Training, evaluation, and export scripts
- `venv/` - Python environment

**Documentation Archive:**
- `docs/` - All analysis, guides, and results (.md files)

**Key Scripts:**
- Training: `train_baseline.py`, `train_yolo12.py`, `train_yolo26.py`
- Evaluation: `evaluate_*_test.py` (baseline, ubm, yolo12, yolo26)
- Export: `export_yolo12_onnx.py`, `export_tensorrt_int8.py`
- Benchmarking: `benchmark_int8.py`, `benchmark_yolo26_int8.py`
- Utilities: `wandb_api.py`, `download_fsoco.py`

### 🎓 Academic Contributions

1. **Systematic architecture evaluation** - YOLOv11 → YOLO12 → YOLO26
2. **Hyperparameter analysis** - Demonstrated diminishing returns
3. **INT8 quantization** - Deployment optimization for RTX 4060
4. **Real-world validation** - Custom test set from .mcap data (planned)

---

### Files in This Project
- `CLAUDE.md` - Main project documentation (this file)
- `TODO.md` - Active task tracking
- `docs/` - Documentation archive (analysis, guides, results)
- `download_fsoco.py` - Downloads FSOCO-12 dataset from Roboflow
- `wandb_api.py` - W&B API interface for monitoring training
- `.env` - API keys (gitignored)
- `.gitignore` - Security for API keys
- `venv/` - Python environment (gitignored)

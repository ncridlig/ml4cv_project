# ML4CV 3CFU Project: Cone Detection Improvement

---

## Documentation Policy

To keep the repository clean and organized:

1. **All project documentation, analysis, results, and notes should be added to THIS FILE (CLAUDE.md)**
2. **Additional .md files are stored in `docs/` folder** - not in root directory
3. **When documenting new findings, updates, or results, append them to the relevant section in CLAUDE.md**
4. **Keep root directory clean** - only essential scripts and this documentation file

**Current documentation structure:**
- `CLAUDE.md` - Main project documentation (THIS FILE)
- `docs/` - Archive of detailed analysis documents
- `TODO.md` - Active task tracking (separate for workflow)

---

**Use the virtual environment for all Python scripts.**

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

---

## Baseline Performance (from Edo Fusa's Thesis)

### Model: YOLO11n (300 epochs, FSOCO dataset)

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
   - 5m: +/-0.19m error
   - 10m: +/-0.46m error
   - 15m: +/-1.16m error
   - 20m: +/-1.37m error

---

## Datasets

### FSOCO-12 Dataset (Training and Testing)
- **Source:** Roboflow
- **Workspace:** `fmdv`
- **Project:** `fsoco-kxq3s`
- **Version:** 12
- **Download script:** `python download_fsoco.py`
- **Purpose:** Training and primary testing (FSOCO benchmark)

```python
# Download FSOCO-12 dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fmdv").project("fsoco-kxq3s")
version = project.version(12)
dataset = version.download("yolo11")
```

### fsoco-ubm Dataset (Real-World Testing)
- **Source:** Roboflow (in-house UBM test set)
- **Workspace:** `fsae-okyoe`
- **Project:** `ml4cv_project`
- **Version:** 1
- **Size:** 96 images (test-only)
- **Download script:** `python download_fsoco_ubm.py`
- **Purpose:** Real-world validation from car camera data

```python
# Download fsoco-ubm dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fsae-okyoe").project("ml4cv_project")
version = project.version(1)
dataset = version.download("yolo26")
```

**Dataset Details:**
- **Date:** November 20, 2025
- **Location:** Rioveggio test track
- **Camera:** ZED 2i stereo (1280x720)
- **Frames:** Extracted from ROS bag recordings (lidar1.avi, lidar2.avi)
- **Sampling:** Every 60 frames (2 seconds real-world time at 60 FPS)
- **Classes:** blue_cone, yellow_cone, orange_cone, large_orange_cone, unknown_cone
- **Annotations:** Created with Roboflow Label Assist
- **Use Case:** Validation on actual deployment conditions (motion blur, lighting, distance)

**Why Important:**
- FSOCO-12 is an internet dataset and may not match real track conditions
- fsoco-ubm is ground truth from the actual car camera
- Tests models on real-world edge cases
- Validates if FSOCO-12 performance translates to deployment

### cone-detector Dataset (Pre-training for Two-Stage)
- **Source:** Roboflow
- **Workspace:** `fsbdriverless`
- **Project:** `cone-detector-zruok`
- **Version:** 1
- **Size:** 22,725 images (3x larger than FSOCO-12)
- **Purpose:** Stage 1 pre-training for two-stage YOLO26n
- **Note:** Single-stage training on this dataset plateaus at mAP50 ~ 0.68 (not suitable as final dataset)
- **Use Case:** Pre-training provides better feature learning, then fine-tune on FSOCO-12

### Training Time (RTX 4080 Super)

**Two-Stage Training:**
- **Stage 1 (cone-detector):** ~5 minutes/epoch
  - 400 epochs = ~33 hours total
  - Dataset: 22,725 images
- **Stage 2 (FSOCO-12):** ~2 minutes/epoch
  - 300 epochs = ~10 hours total
  - Dataset: 7,120 images

**Single-Stage Training:**
- **FSOCO-12:** ~2 minutes/epoch
  - 300 epochs = ~10 hours

---

## Ultralytics File Path Convention

**Important:** Ultralytics adds a task prefix to all save paths.

When you specify in training:
```python
model.train(
    project='runs/two-stage-yolo26',
    name='stage2a_head_only_50ep',
    # task='detect' is auto-detected
)
```

**Actual save path structure:**
```
runs/{task}/{project}/{name}/
```

**Example - what you get:**
```
runs/detect/runs/two-stage-yolo26/stage2a_head_only_50ep/
         ^^^^^ task prefix added automatically!
```

**Why this matters:**
- When loading weights, you MUST include `runs/detect/` prefix
- Forgetting this causes `FileNotFoundError`
- The `detect` task is added for object detection models

**Correct loading:**
```python
model = YOLO('runs/detect/runs/two-stage-yolo26/stage2a_head_only_50ep/weights/best.pt')
```

**Common mistake:**
```python
model = YOLO('runs/two-stage-yolo26/stage2a_head_only_50ep/weights/best.pt')  # WRONG!
```

---

## Hardware Context

### Development Machines

| Machine | Purpose | GPU | OS |
|---------|---------|-----|-----|
| M1 MacBook | Debugging, code dev | CPU only | macOS |
| Ubuntu Workstation | Training and testing | RTX 4080 Super (CUDA) | Ubuntu |

**Code should run on both:** CPU mode for debugging on Mac, CUDA for training on Ubuntu.

### Target Deployment
**NVIDIA RTX 4060** (onboard ASU - Autonomous System Unit)

### Camera: ZED 2i Stereo
- Resolution: 2x 1280x720 @ 60fps
- FOV: 72 deg (H) x 44 deg (V)
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
| 2026-01-21 | ~1.5 hrs | Project setup: cloned repos, downloaded dataset (22,725 images), created venv, wrote CLAUDE.md, extracted baseline metrics from Edo's thesis |
| 2026-01-23 | ~1 hr | Discovered dataset mismatch via W&B monitoring (training plateaued at mAP50=0.68 vs expected 0.824). Created `wandb_api.py` and `download_fsoco.py`. Identified correct dataset: `fmdv/fsoco-kxq3s` version 12. |
| 2026-01-24 | 3 hrs | Hyperparameter sweep setup and execution (W&B Bayesian optimization, 13 hyperparameters, 20 runs). Added inference optimization research. Integrated Gabriele's YOLO pipeline documentation. |
| 2026-01-25 | 8 hrs | Trained and tested YOLO12n (mAP50 0.7081). Trained YOLO26n (mAP50 0.7626 - new best). Implemented INT8 quantization pipeline. Exported models to ONNX and TensorRT. Benchmarked inference speeds. Created documentation folder structure. |
| 2026-01-26 | 6 hrs | UBM workshop: tested models on ASU (RTX 4060) onboard racecar. Meeting with Alberto re: custom test set and deployment. Extracted test frames from .mcap rosbag files. Started UBM in-house test dataset creation. Uploaded models to Roboflow. |
| 2026-01-27 | 6 hrs | Created in-house UBM test dataset (fsoco-ubm): annotated 96 images in Roboflow with Label Assist, exported dataset. YOLO26n Stage 1 training ran all day (interrupted at epoch 338/400 but converged). Launched Stage 2 fine-tuning (failed due to optimizer='auto' ignoring lr0, causing catastrophic forgetting). Researched fine-tuning best practices, debugged optimizer issue, redesigned Stage 2 as two-phase fine-tuning with AdamW. |
| 2026-01-28 | 4 hrs | Completed YOLO26n two-stage training (Stage 2B: 250 epochs). Evaluated two-stage on FSOCO-12 test (0.7612 mAP50, -0.2% vs single-stage). Ran fsoco-ubm evaluation on all 6 models. Bug found: annotation format mismatch (yolo26 vs yolo11) invalidated all initial fsoco-ubm results. Fixed download_fsoco_ubm.py and re-evaluated. Corrected results: YOLO26n two-stage wins (0.5652 mAP50), YOLO11n shows best generalization (-21.5% drop). Updated docs/RESULTS_SUMMARY.md with complete deployment roadmap. Located confidence threshold in ROS node (line 79, change 0.5 to 0.20). Created comprehensive docs/DATASETS.md. Final decision: Deploy YOLO26n (two-stage) with conf=0.20. |
| 2026-01-29 | 5 hrs | Updated model from YOLO11 to YOLO26n with testing and documentation throughout. Created benchmark summary tables (docs/table.md) for academic and team audiences. Opened PR #30 on ubm-yolo-detector. Next: 2026-02-05 workshop session -- compile on ASU and run full stack for in-situ testing. |
| 2026-01-30 | 1.5 hrs | Wrote introduction.tex of report. |

---

## Results Summary

### YOLO26n Deployed on ASU

**Final Test Set Performance (FSOCO-12, 689 images):**
- **YOLO26n: 0.7626 mAP50** (new best model)
- **vs UBM production: +14.6%** (0.7626 vs 0.6655)
- **vs YOLO12n: +7.7%** (0.7626 vs 0.7081)
- **vs YOLO11n: +7.9%** (0.7626 vs 0.7065)

**ASU Deployment Performance (RTX 4060, TensorRT FP16):**
- **Latency: 2.63 ms** (mean)
- **GPU Compute: 1.019 ms**
- **Throughput: 633 qps (queries per second)**
- **vs UBM baseline: 2.58x faster** (2.63 ms vs 6.78 ms)
- **Real-time margin: 6.3x for 60 fps** (2.63 ms << 16.7 ms budget)

**YOLO26n Model Specifications:**
- Architecture: YOLO26n (2025, latest Ultralytics)
- Parameters: 2,505,750 (2.51M)
- GFLOPs: 5.780
- Precision: 0.8485
- Recall: 0.6935
- TensorRT FP16: 9.35 MB

**Per-Class Performance (Test Set):**
- Large Orange Cone: 0.886 mAP50
- Blue Cone: 0.863 mAP50
- Yellow Cone: 0.856 mAP50
- Orange Cone: 0.843 mAP50
- Unknown Cone: 0.364 mAP50

**W&B Run Details:**
- Project: `ncridlig-ml4cv/runs-yolo26`
- Run ID: `yolo26n_300ep_FSOCO_20260125_122257`
- URL: https://wandb.ai/ncridlig-ml4cv/runs-yolo26/runs/yolo26n_300ep_FSOCO_20260125_122257
- Training time: 300 epochs, batch 64, ~2.5 days on RTX 4080 Super

### Key Findings

**Accuracy:**
1. YOLO26n is best model: 0.7626 mAP50 (+14.6% over UBM production)
2. YOLO26 beats YOLO12 by 7.7% on test set (0.7626 vs 0.7081)
3. All classes improved: Blue (+5.9%), Yellow (+6.0%), Orange (+6.8%), Unknown (+23.4%)
4. Highest precision: 0.8485

**Speed:**
1. Fastest inference: 2.63 ms on RTX 4060 (FP16 TensorRT)
2. 2.58x faster than UBM baseline (6.78 ms to 2.63 ms)
3. Real-time capable: 6.3x margin for 60 fps

**Research:**
1. Hyperparameter sweep ineffective - defaults already optimal
2. 2025 architectures successful - YOLO12 and YOLO26 both beat baseline
3. INT8 not needed - FP16 already exceeds requirements (6.3x margin)
4. Gabriele's 0.824 claim unverified - UBM production can't reproduce it

---

## Repository Organization

**Root Directory:**
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

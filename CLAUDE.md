# ML4CV 3CFU Project: Cone Detection Improvement

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

### Primary Dataset (Gabriele's recommendation)
- **Source:** Roboflow
- **Workspace:** `fsbdriverless`
- **Project:** `cone-detector-zruok`
- **Version:** 1
- **Size:** 22,725 images (Train: 19,884 | Valid: 1,893 | Test: 948)
- **API:** Sign up free at [roboflow.com](https://roboflow.com), get API key from settings

```python
# Download dataset
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fsbdriverless").project("cone-detector-zruok")
version = project.version(1)
dataset = version.download("yolov11")
```

### Alternative (used in existing notebook)
- FSOCO-12 from workspace `fmdv` (same FSOCO data, different version)

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

---

## Session Status (Last Updated: 2026-01-21)

### Completed
- [x] Cloned ubm-yolo-detector repo
- [x] Downloaded dataset from Roboflow (22,725 images)
- [x] Created Python venv with ultralytics, roboflow
- [x] Created IMPROVEMENT_TARGETS.md with concrete metrics
- [x] Extracted baseline metrics from Edo's thesis

### Next Session: Start Here
1. **Transfer to Ubuntu workstation** - Copy `datasets/cone-detector/` folder
2. **Run baseline training:**
   ```bash
   source venv/bin/activate
   yolo train model=yolo11n.pt data=datasets/cone-detector/data.yaml epochs=300
   ```
3. **Verify baseline metrics** - Should match thesis (mAP50 ≈ 0.824)
4. **Day 4 checkpoint** - If mAP50 ≥ 0.84 continue, else pivot to Plan B

### Decision Point
- **Plan A (current):** Cone detection improvement
- **Plan B (backup):** Overtaking model debugging (ubm-ai repo also cloned)
- **Pivot trigger:** If Day 4 results don't reach Tier 1 targets (mAP50 ≥ 0.84)

### Files Created This Session
- `CLAUDE.md` - This file
- `IMPROVEMENT_TARGETS.md` - Detailed targets and experiment plan
- `competing_plans.md` - Plan A vs Plan B comparison
- `.env` - Roboflow API key (gitignored)
- `.gitignore` - Security for API keys
- `venv/` - Python environment
- `datasets/cone-detector/` - 22,725 training images

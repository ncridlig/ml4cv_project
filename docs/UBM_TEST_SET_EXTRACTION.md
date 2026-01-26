# UBM Test Set Frame Extraction

**Date:** 2026-01-26
**Location:** Rioveggio Test Track (November 20, 2025)
**Videos:** LidarTest1 + LidarTest2

---

## 📹 Video Properties

### LidarTest1
- **File:** `media/20_11_2025_Rioveggio_Test_LidarTest1.avi`
- **Size:** 203 MB
- **Resolution:** 2560×720 (stereo stitched)
- **FPS:** 60 (exported at 60 FPS, recorded at 30 FPS)
- **Total frames:** 1,454
- **Duration:** 24.2 seconds

### LidarTest2
- **File:** `media/20_11_2025_Rioveggio_Test_LidarTest2.avi`
- **Size:** 194 MB
- **Resolution:** 2560×720 (stereo stitched)
- **FPS:** 60 (exported at 60 FPS, recorded at 30 FPS)
- **Total frames:** 1,374
- **Duration:** 22.9 seconds

**Total Duration:** 47.1 seconds of car camera footage

---

## 🎯 Extraction Strategy

**Sampling Rate:** Every 60 frames

**Rationale:**
- Videos exported at 60 FPS (but recorded at 30 FPS, so frames are duplicated)
- Every 60 frames = 1 second at 60 FPS
- Since original was 30 FPS, this represents **2 seconds of real-world time**
- Provides good temporal diversity without extracting duplicate frames
- Avoids extracting consecutive frames that are nearly identical

**Expected Output:**
- LidarTest1: 1,454 / 60 = **24 stereo pairs** (48 images)
- LidarTest2: 1,374 / 60 = **22 stereo pairs** (44 images)
- **Total: ~46 stereo pairs = ~92 images**

**Image Format:**
- Split from 2560×720 stereo → 2× 1280×720 (left + right)
- Each stereo pair = 2 images (left_NNNN.jpg + right_NNNN.jpg)
- Format: JPEG (good compression for Roboflow upload)

---

## 🚀 Quick Start

**Extract frames from both videos:**

```bash
# Activate virtual environment
source venv/bin/activate

# Extract LidarTest1 frames (every 60 frames)
python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest1.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar1

# Extract LidarTest2 frames (every 60 frames)
python3 extract_frames_from_avi.py \
    media/20_11_2025_Rioveggio_Test_LidarTest2.avi \
    --output ubm_test_set/images \
    --interval 60 \
    --prefix lidar2
```

**Expected Output Structure:**
```
ubm_test_set/images/
├── lidar1_left_0000.jpg
├── lidar1_right_0000.jpg
├── lidar1_left_0001.jpg
├── lidar1_right_0001.jpg
├── ...
├── lidar1_left_0023.jpg
├── lidar1_right_0023.jpg
├── lidar2_left_0000.jpg
├── lidar2_right_0000.jpg
├── ...
├── lidar2_left_0021.jpg
└── lidar2_right_0021.jpg

Total: ~92 images (46 stereo pairs)
```

---

## 📊 Expected Frame Distribution

**Temporal Coverage:**
- Frames extracted every 2 seconds of real-world time
- LidarTest1: 24.2 seconds → 12 unique time points → 24 stereo pairs
- LidarTest2: 22.9 seconds → 11 unique time points → 22 stereo pairs

**Diversity:**
- Different lighting conditions (shadows, bright spots)
- Different cone distances (near field, far field)
- Different car orientations (turns, straights)
- Different cone types (blue, yellow, orange, large orange)

---

## 🎨 Roboflow Annotation Guidelines

**Upload to Roboflow:**
1. Create project: `UBM-Rioveggio-Test-2025`
2. Upload all ~92 images
3. Configure 5 classes:
   - `blue_cone`
   - `yellow_cone`
   - `orange_cone`
   - `large_orange_cone`
   - `unknown_cone`

**Annotation Tips:**
1. **Label all visible cones** - even partially visible or distant
2. **Tight bounding boxes** - minimize background, include full cone
3. **Unknown cones** - use for occluded or ambiguous cases
4. **Large orange cones** - distinctive size (start/finish cones)
5. **Quality check** - review all annotations before export

**Expected Annotation Time:**
- ~92 images × ~5-10 cones per image = ~500-900 total annotations
- Estimated time: 2-3 hours (with keyboard shortcuts)

---

## 📥 Dataset Export

**Export Format:** YOLOv11 PyTorch

**Download from Roboflow:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('ubm').project('ubm-rioveggio-test-2025')
dataset = project.version(1).download('yolov11')
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

---

## 🧪 Model Evaluation

**Evaluate all models on UBM test set:**

```bash
# Create evaluation scripts (similar to evaluate_*_test.py)
# Evaluate YOLO26n (best model)
python3 evaluate_yolo26_ubm_test.py

# Evaluate YOLO12n (comparison)
python3 evaluate_yolo12_ubm_test.py

# Evaluate YOLOv11n baseline (comparison)
python3 evaluate_baseline_ubm_test.py
```

**Expected Results:**
- Real-world mAP50 may be **lower** than FSOCO-12 (more challenging)
- Edge cases revealed:
  - Motion blur (car movement)
  - Distance degradation (far field cones)
  - Lighting variations (shadows, reflections)
  - Occlusions (cones behind other cones)

---

## 📈 Comparison: FSOCO-12 vs UBM Rioveggio

**FSOCO-12 Test Set (689 images):**
- Internet dataset (crowdsourced from multiple teams/tracks)
- High quality annotations
- Diverse conditions (multiple locations, weather, lighting)
- **YOLO26n: 0.7626 mAP50**

**UBM Rioveggio Test Set (~92 images):**
- Real car data from actual test run
- Same hardware (ZED 2i stereo camera)
- Same track conditions as deployment
- Expected mAP50: **0.70-0.75** (TBD)

**Why UBM test set is valuable:**
1. **Ground truth for deployment** - exact camera, lighting, track
2. **Edge case discovery** - reveals real-world failure modes
3. **Continuous improvement** - benchmark for future models
4. **Small but representative** - 46 stereo pairs capture key scenarios

---

## 🔄 Future Expansion

**Current test set:** 46 stereo pairs (24 + 22)

**Potential expansion sources:**
1. **More test runs** - extract from additional ROS bags
2. **Competition data** - frames from actual Formula Student events
3. **Different tracks** - test generalization to new environments
4. **Different conditions** - rain, night, different lighting

**Target size:** 100-200 stereo pairs (200-400 images)

---

## 📁 File Structure

```
ml4cv_project/
├── media/
│   ├── 20_11_2025_Rioveggio_Test_LidarTest1.avi  (203 MB, 1454 frames)
│   └── 20_11_2025_Rioveggio_Test_LidarTest2.avi  (194 MB, 1374 frames)
├── ubm_test_set/
│   └── images/
│       ├── lidar1_left_*.jpg   (24 images)
│       ├── lidar1_right_*.jpg  (24 images)
│       ├── lidar2_left_*.jpg   (22 images)
│       └── lidar2_right_*.jpg  (22 images)
├── datasets/
│   └── UBM-Rioveggio-Test-2025/  (after Roboflow download)
│       ├── data.yaml
│       ├── images/
│       └── labels/
└── extract_frames_from_avi.py
```

---

## ✅ Checklist

**Extraction:**
- [ ] Run extraction script for LidarTest1
- [ ] Run extraction script for LidarTest2
- [ ] Verify output: ~92 images in ubm_test_set/images/
- [ ] Check image quality (no corruption, correct split)

**Annotation:**
- [ ] Create Roboflow project: UBM-Rioveggio-Test-2025
- [ ] Upload all images (~92)
- [ ] Configure 5 cone classes
- [ ] Annotate all cones (~500-900 annotations)
- [ ] Quality check all annotations
- [ ] Export YOLOv11 format

**Evaluation:**
- [ ] Download dataset from Roboflow
- [ ] Create data.yaml
- [ ] Evaluate YOLO26n on UBM test set
- [ ] Evaluate YOLO12n on UBM test set
- [ ] Evaluate baseline on UBM test set
- [ ] Document results (UBM_TEST_SET_RESULTS.md)

**Estimated Total Time:** 4-5 hours

---

**Last Updated:** 2026-01-26
**Status:** Ready for extraction 🚀

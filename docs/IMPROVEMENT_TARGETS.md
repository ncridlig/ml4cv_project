# Cone Detection Improvement Targets

## Baseline Metrics (from Edo's Thesis - YOLOv11n, 300 epochs)

| Metric | Baseline Value | Source |
|--------|----------------|--------|
| mAP50 (All Classes) | 0.824 | Thesis Table 5.2 |
| mAP50-95 (All Classes) | 0.570 | Thesis Table 5.2 |
| Precision | 0.849 | Thesis Table 5.2 |
| Recall | 0.765 | Thesis Table 5.2 |
| Inference Time (TensorRT) | 6.78 ms | Thesis Table 5.3 |
| False Positive Rate | ~15% | (1 - Precision) |

---

## Target Improvements

### Tier 1: Must Achieve (for passing grade)
| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| mAP50 | 0.824 | **≥ 0.84** | +2% |
| Precision | 0.849 | **≥ 0.87** | +2% |
| Documentation | None | Complete training report | N/A |

### Tier 2: Good Achievement (solid project)
| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| mAP50 | 0.824 | **≥ 0.86** | +4% |
| Precision | 0.849 | **≥ 0.90** | +5% |
| False Positive Rate | ~15% | **< 10%** | -5% |
| Inference (RTX 4060) | ~7ms | **< 8ms** | Maintained |

### Tier 3: Excellent Achievement (impressive project)
| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| mAP50 | 0.824 | **≥ 0.88** | +6% |
| mAP50-95 | 0.570 | **≥ 0.60** | +5% |
| Precision | 0.849 | **≥ 0.92** | +8% |
| Unknown Cone mAP50 | 0.547 | **≥ 0.65** | +19% |

---

## Improvement Strategies

### 1. Model Architecture (Low Effort, Medium Impact)
- [ ] Try YOLOv11s (small) instead of YOLOv11n (nano) - more capacity
- [ ] Try YOLOv12n (already in repo, newer architecture)
- [ ] Compare with RT-DETR (transformer-based, potentially better accuracy)

### 2. Training Hyperparameters (Medium Effort, Medium Impact)
- [ ] Increase epochs: 300 → 500 (if overfitting allows)
- [ ] Learning rate scheduling: cosine annealing
- [ ] Image size: 640 → 1280 (trade speed for accuracy)
- [ ] Batch size optimization for RTX 4080 Super

### 3. Data Augmentation (Medium Effort, High Impact)
- [ ] Mosaic augmentation (default in YOLO)
- [ ] Color jitter for lighting robustness
- [ ] Random brightness/contrast
- [ ] Cutout/GridMask for occlusion robustness
- [ ] Weather augmentation (fog, rain simulation)

### 4. Dataset Improvements (High Effort, High Impact)
- [ ] Add hard negative examples (non-cone objects similar to cones)
- [ ] Balance classes (Unknown Cone is underrepresented)
- [ ] Add UBM-specific images if available

### 5. Post-Processing (Low Effort, Low-Medium Impact)
- [ ] Tune confidence threshold (currently 0.5)
- [ ] Tune NMS IoU threshold (currently 0.5)
- [ ] Class-specific thresholds

---

## Experiment Plan

### Day 1-2: Baseline Reproduction
1. Train YOLOv11n with default settings on new dataset
2. Verify metrics match or exceed thesis baseline
3. Establish local training pipeline

### Day 3-4: Quick Wins
1. Try YOLOv11s (small variant)
2. Try increased image size (1280)
3. Tune confidence/NMS thresholds
4. Compare metrics

### Day 4 Checkpoint
**Decision point:** If mAP50 ≥ 0.84 and Precision ≥ 0.87 → Continue Plan A
**Otherwise:** Pivot to Plan B (Overtaking model debugging)

### Day 5-7: Optimization & Polish
1. ONNX export and benchmark on RTX 4060
2. TensorRT optimization
3. Final model selection
4. Documentation

---

## Evaluation Protocol

### Test Set
Use the designated test split from Roboflow dataset (948 images)

### Metrics to Track
```python
# After training
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val(data='datasets/cone-detector/data.yaml')

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")  # mean precision
print(f"Recall: {metrics.box.mr}")     # mean recall
```

### Inference Timing
```python
import time
import torch

model = YOLO('best.pt')
model.to('cuda')  # or 'cpu' for Mac

# Warmup
for _ in range(10):
    model.predict('test_image.jpg', verbose=False)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    model.predict('test_image.jpg', verbose=False)
    times.append(time.time() - start)

print(f"Mean inference: {sum(times)/len(times)*1000:.2f} ms")
```

---

## Success Criteria Summary

| Outcome | mAP50 | Precision | Status |
|---------|-------|-----------|--------|
| **Minimum Pass** | ≥ 0.84 | ≥ 0.87 | Required |
| **Good** | ≥ 0.86 | ≥ 0.90 | Target |
| **Excellent** | ≥ 0.88 | ≥ 0.92 | Stretch |

---

## Notes

- All training should be done on Ubuntu workstation (RTX 4080 Super)
- Debugging/development on M1 MacBook (CPU mode)
- Final deployment target: RTX 4060 on race car
- Keep training logs and TensorBoard for final report

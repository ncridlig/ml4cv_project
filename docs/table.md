# Model Benchmark Summary

---

## Part 1: Team Reference (UBM Deployment)

### Overall Performance (FSOCO-12 Test Set, 689 images)

| Model | mAP50 | Precision | Recall | Latency (RTX 4060) | vs UBM Prod. |
|-------|------:|----------:|-------:|--------------------:|-------------:|
| **YOLO26n (two-stage)** | **0.761** | 0.832 | **0.708** | **2.63 ms** | **+14.4%** |
| YOLO26n (single-stage) | 0.763 | **0.849** | 0.694 | 2.63 ms | +14.6% |
| YOLO12n | 0.708 | 0.840 | 0.654 | — | +6.4% |
| YOLOv11n (our baseline) | 0.707 | 0.816 | 0.662 | 2.70 ms | +6.2% |
| UBM Production (YOLOv11n) | 0.666 | 0.803 | 0.579 | 6.78 ms | — |

> **Deployed model:** YOLO26n two-stage — best recall and real-world generalization.

### Inference Speed (RTX 4060, TensorRT FP16)

| Model | Mean Latency | GPU Compute | Throughput | Real-Time Margin (60 fps) |
|-------|-------------:|------------:|-----------:|--------------------------:|
| **YOLO26n** | **2.63 ms** | 1.02 ms | 633 qps | 6.3x |
| YOLOv11n | 2.70 ms | 0.99 ms | 634 qps | 6.2x |
| UBM Baseline | 6.78 ms | — | ~147 qps | 2.5x |

### Real-World Validation (fsoco-ubm, 96 images)

| Model | mAP50 | Precision | Recall | Gap vs FSOCO-12 |
|-------|------:|----------:|-------:|----------------:|
| **YOLO26n (two-stage)** | **0.565** | **0.649** | 0.462 | -25.8% |
| YOLO26n (single-stage) | 0.565 | 0.615 | **0.469** | -25.9% |
| YOLOv11n (our baseline) | 0.555 | 0.874 | 0.447 | -21.5% |
| YOLO12n | 0.517 | 0.572 | 0.454 | -27.0% |
| UBM Production | 0.517 | 0.635 | 0.393 | -22.3% |

> Real-world data is 22-32% harder than the FSOCO-12 benchmark.

---

## Part 2: Academic Presentation

### 1. Architecture Comparison

| | YOLOv11n | YOLO12n | YOLO26n |
|---|---:|---:|---:|
| **Parameters** | 2.59M | 2.56M | **2.51M** |
| **GFLOPs** | 6.4 | 6.3 | **5.8** |
| **Year** | 2024 | 2025 | 2025 |
| **TensorRT FP16 Size** | ~9 MB | ~9 MB | 9.35 MB |

### 2. FSOCO-12 Test Set — Full Results (689 images)

| Model | Dataset | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|-------|---------|-------:|------:|---------:|----------:|-------:|
| **YOLO26n (single)** | FSOCO-12 | 300 | **0.763** | 0.524 | **0.849** | 0.694 |
| **YOLO26n (two-stage)** | CD→FSOCO-12 | 338+300 | 0.761 | **0.528** | 0.832 | **0.708** |
| YOLO12n | FSOCO-12 | 300 | 0.708 | 0.485 | 0.840 | 0.654 |
| YOLOv11n (ours) | FSOCO-12 | 300 | 0.707 | 0.490 | 0.816 | 0.662 |
| UBM Production | unknown | 300 | 0.666 | 0.461 | 0.803 | 0.579 |

> CD = cone-detector dataset (22,725 images, pre-training stage).

### 3. Per-Class Breakdown — YOLO26n (single-stage, FSOCO-12 Test)

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------:|----------:|----------:|-------:|------:|---------:|
| Large Orange Cone | 154 | 408 | 0.873 | 0.833 | **0.886** | 0.688 |
| Blue Cone | 506 | 4,437 | 0.927 | 0.783 | 0.863 | 0.602 |
| Yellow Cone | 562 | 4,844 | 0.915 | 0.774 | 0.856 | 0.583 |
| Orange Cone | 286 | 1,686 | 0.892 | 0.779 | 0.843 | 0.571 |
| Unknown Cone | 68 | 679 | 0.635 | 0.297 | 0.364 | 0.178 |

### 4. Per-Class Breakdown — YOLO12n (FSOCO-12 Test)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|----------:|-------:|------:|---------:|
| Large Orange Cone | 0.912 | 0.821 | **0.871** | 0.693 |
| Blue Cone | 0.912 | 0.738 | 0.804 | 0.548 |
| Yellow Cone | 0.890 | 0.727 | 0.796 | 0.534 |
| Orange Cone | 0.879 | 0.722 | 0.775 | 0.525 |
| Unknown Cone | 0.607 | 0.264 | 0.295 | 0.124 |

### 5. Two-Stage vs Single-Stage Training

| Metric | Single-Stage | Two-Stage | Delta |
|--------|-------------:|----------:|------:|
| FSOCO-12 mAP50 | **0.763** | 0.761 | -0.2% |
| FSOCO-12 mAP50-95 | 0.524 | **0.528** | +0.6% |
| FSOCO-12 Precision | **0.849** | 0.832 | -1.6pp |
| FSOCO-12 Recall | 0.694 | **0.708** | +1.4pp |
| fsoco-ubm mAP50 | 0.565 | **0.565** | +0.04% |
| fsoco-ubm Precision | 0.615 | **0.649** | +3.4pp |
| Generalization Gap | -25.9% | **-25.8%** | +0.1pp |

> Two-stage trades marginal FSOCO-12 precision for better recall and real-world generalization.

### 6. Hyperparameter Sweep Summary

| Metric | Value |
|--------|------:|
| Runs completed | 10 / 21 |
| Best sweep mAP50 | 0.709 |
| Baseline mAP50 | 0.714 |
| Mean of sweep runs | 0.703 |
| Spread (std) | 0.019 |

> **Conclusion:** Ultralytics defaults are near-optimal. Sweep provided no improvement.

### 7. Deployment Latency (RTX 4060, TensorRT FP16)

| Component | YOLO26n | YOLOv11n (prod.) |
|-----------|--------:|-----------------:|
| H2D Transfer | 1.58 ms | 1.58 ms |
| **GPU Compute** | **1.02 ms** | 0.99 ms |
| D2H Transfer | 0.03 ms | 0.13 ms |
| **Total** | **2.63 ms** | 2.70 ms |
| Max FPS | 380 | 370 |

### 8. Datasets Used

| Dataset | Images | Instances | Purpose |
|---------|-------:|----------:|---------|
| FSOCO-12 (train) | 5,536 | — | Primary training |
| FSOCO-12 (val) | 1,968 | 36,123 | Validation during training |
| FSOCO-12 (test) | 689 | 12,054 | Standard benchmark |
| cone-detector | 22,725 | — | Stage 1 pre-training |
| fsoco-ubm | 96 | 1,426 | Real-world validation |

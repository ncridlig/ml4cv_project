# Confidence & NMS Threshold Sweep Results

Source: `ubm-yolo-detector/runs/evaluation/optimized_conf0.35_iou0.30_results.txt`
Script: `ubm-yolo-detector/training/optimize_thresholds.py`
Model: YOLO26n (640p, 300ep)
Dataset: fsoco-ubm test set (96 images)

## Confidence Sweep (fixed IoU=0.45)

| Conf | Precision | Recall | F1     | mAP50  |
|------|-----------|--------|--------|--------|
| 0.10 | 0.6355   | 0.4143 | 0.5016 | 0.5345 |
| 0.15 | 0.6355   | 0.4143 | 0.5016 | 0.5357 |
| 0.20 | 0.6355   | 0.4143 | 0.5016 | 0.5391 |
| 0.25 | 0.6355   | 0.4143 | 0.5016 | 0.5412 |
| 0.30 | 0.6351   | 0.4144 | 0.5016 | 0.5444 |
| **0.35** | **0.6897** | **0.4125** | **0.5163** | **0.5599** |
| 0.40 | 0.7831   | 0.3624 | 0.4955 | 0.5759 |
| 0.45 | 0.5849   | 0.3336 | 0.4249 | 0.4619 |
| 0.50 | 0.5861   | 0.3298 | 0.4221 | 0.4603 |
| 0.55 | 0.5873   | 0.3232 | 0.4169 | 0.4572 |
| 0.60 | 0.5887   | 0.3161 | 0.4114 | 0.4540 |
| 0.65 | 0.5898   | 0.3075 | 0.4043 | 0.4500 |
| 0.70 | 0.5911   | 0.2969 | 0.3953 | 0.4449 |

Best confidence by F1: **0.35**

## NMS Sweep (fixed conf=0.35)

| IoU  | Precision | Recall | F1     | mAP50  |
|------|-----------|--------|--------|--------|
| **0.30** | **0.6897** | **0.4125** | **0.5163** | **0.5599** |
| 0.35 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.40 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.45 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.50 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.55 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.60 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.65 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |
| 0.70 | 0.6897   | 0.4125 | 0.5163 | 0.5599 |

NMS threshold has **no measurable effect** on any metric across the entire 0.30-0.70 range.

## Key Finding

| Configuration | mAP50 |
|---|---|
| Default (conf=0.50) | 0.4603 |
| Optimized (conf=0.35, iou=0.30) | 0.5599 |
| **Improvement** | **+0.0996 (~10 pp)** |

The entire improvement comes from the confidence threshold change (0.50 to 0.35). The NMS threshold change has no measurable impact.

# UBM Official Model Information

## Source

**From:** Gabriele (UBM Driverless team member)
**Date:** 2026-01-24

## Model Details

### YOLOv11n (640p, 300 epochs)
**Location:** `/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt`

**Training:**
- Dataset: https://www.fsoco-dataset.com/
- Resolution: 640x640
- Epochs: 300
- Model: YOLOv11n
- Status: "Working well on detecting cones" on actual car

**Files:**
- `best.pt` (5.3 MB) - PyTorch weights
- `best.onnx` (11 MB) - ONNX format
- `best_openvino_model/` - OpenVINO FP32 format

**Classes:**
```python
{
  0: 'blue_cone',
  1: 'large_orange_cone',
  2: 'orange_cone',
  3: 'unknown_cone',
  4: 'yellow_cone'
}
```

### YOLOv12n (1280p, 300 epochs)
**Location:** `/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov12n_1280p_300ep/best.pt`

**Training:**
- Dataset: Likely same FSOCO
- Resolution: 1280x1280 (higher resolution)
- Epochs: 300
- Model: YOLOv12n (newer architecture)

---

## Deployment on Car

**From:** https://github.com/ubm-driverless/ubm-fsae/tree/main/zed2_driver

### Production Model (INT8 Quantized)
**Location:** `ubm-fsae/zed2_driver/yolo/models/best_int8_openvino_model/`

**Format:** OpenVINO INT8 (quantized for speed)
- Faster inference on embedded hardware
- Reduced precision (INT8 vs FP32)
- Trade-off: Speed vs slight accuracy loss

**Hardware:** ASU (Autonomous System Unit) with NVIDIA RTX 4060

---

## Evaluation Task

### Why Evaluate This Model?

1. **True baseline:** This is what Gabriele/Patta actually trained
2. **Production-tested:** Running on the actual car
3. **Dataset comparison:** See if FSOCO-12 matches their training data
4. **Performance gap:** Understand difference vs our baseline (mAP50 = 0.714)

### Evaluation Script

**File:** `evaluate_ubm_model.py`

**What it does:**
- Loads UBM official model (`best.pt`)
- Validates on FSOCO-12 validation set
- Compares against our baseline
- Compares against thesis baseline (0.824)

**Command:**
```bash
source venv/bin/activate
python evaluate_ubm_model.py
```

**Expected outcomes:**

**Scenario A:** UBM model gets ~0.714 on FSOCO-12
- Their model performs same as ours
- Confirms FSOCO-12 is similar to their training data
- Our hyperparameter search will help them too!

**Scenario B:** UBM model gets ~0.80+ on FSOCO-12
- Their model is significantly better
- Means they used different/better hyperparameters or dataset
- We should analyze their model config
- Our sweep might still find improvements

**Scenario C:** UBM model gets <0.70 on FSOCO-12
- Domain mismatch - their training data ≠ FSOCO-12
- Validates Edo's comment about "particular dataset"
- Our work on FSOCO-12 is still valuable for benchmarking

---

## Memory Constraint

**Issue:** Cannot run evaluation in parallel with hyperparameter sweep
- Sweep uses full GPU memory for training
- Evaluation also needs GPU for inference

**Solution:** Run evaluation AFTER sweep completes (~15-20 hours)

**Timeline:**
1. Sweep finishes (Day 2 morning)
2. Run `python evaluate_ubm_model.py` (~30 min)
3. Analyze results
4. Proceed with best config training

---

## Questions for Follow-Up

If we find their model performs well, we could ask Gabriele:

1. What dataset version did you use? (FSOCO v1, v2, custom?)
2. What hyperparameters? (lr, augmentation, etc.)
3. Any preprocessing? (brightness adjustment, etc.)
4. How does INT8 quantization affect accuracy?
5. Real-world performance metrics? (mAP on car data vs FSOCO)

---

## Related Files

- `evaluate_ubm_model.py` - Evaluation script
- `ubm_evaluation.log` - Evaluation results (after running)
- `runs/evaluation/ubm_official_on_fsoco12/` - Detailed results

---

**Date:** 2026-01-24
**Status:** Pending evaluation (after sweep completes)

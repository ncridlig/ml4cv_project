# TODO List

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
**Result:** mAP50 = 0.714 (Precision: 0.832, Recall: 0.657)
- 13.4% below thesis baseline (0.824)
- Analysis: `BASELINE_RESULTS_ANALYSIS.md`

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

### From Gabriele - UBM Training Notebook Analysis
**MAJOR FINDING:** Our baseline BEATS their notebook!

| Metric | UBM Notebook (200 ep) | Our Baseline (300 ep) | Winner |
|--------|----------------------|----------------------|--------|
| mAP50 | 0.663 | **0.714** | **🏆 OURS (+7.7%)** |
| Recall | 0.573 | **0.657** | **🏆 OURS (+14.7%)** |

**Implications:**
- ✅ Our baseline is STRONG - already ahead of their published results
- ✅ Same dataset (FSOCO-12), same model (YOLOv11n)
- ✅ Longer training (300 vs 200 epochs) helped
- 🔍 Mystery: Their production model is named "300ep" but notebook shows 200
- 📌 Must evaluate their production `best.pt` to see if it's better than notebook

**See:** `TRAINING_COMPARISON.md` for detailed analysis

---

## Current Plan: W&B Sweep for Hyperparameter Optimization

### Phase 1: Automated Sweep (15-20 hours, 20 runs) - IN PROGRESS
**Launch command:**
```bash
./launch_sweep.sh
```

**What it does:**
- Bayesian optimization explores 13 hyperparameters
- Each run: 100 epochs (~45 min)
- Auto-launches next run when current finishes
- Early termination stops poor runs at epoch 30
- W&B dashboard: https://wandb.ai/ncridlig-ml4cv/runs-sweep

**Target:** Find config with mAP50 > 0.75 (5% improvement)

### Phase 2: Evaluate UBM Official Model on Test Set (1 hour) - AFTER SWEEP
**Critical discovery:** Gabriele provided their actual production weights!
- Model: `/home/nicolas/Github/ubm-yolo-detector/yolo/models/yolov11n_640p_300ep/best.pt`
- This is the model running on the actual car
- **Task:** Run `python evaluate_ubm_model.py` to evaluate on FSOCO-12 **test set**
- **Status:** Previously evaluated on validation set (mAP50 = 0.670), needs re-run on test set for proper comparison
- **Why:** Test set provides unbiased performance estimate; validation set was used during our baseline training
- **Memory constraint:** Cannot run in parallel with sweep - do AFTER sweep completes

### Phase 3: Full Training with Best Config (4 hours)
1. Run `python analyze_sweep.py <sweep_id>` to extract best config
2. Paste config into `train_best_config.py`
3. Train for 300 epochs with best hyperparameters
4. Target: mAP50 ≥ 0.78 (+9% over baseline)

### Phase 4: Architecture Comparison (15 hours)
- Use best hyperparameters from sweep
- Train YOLOv11s and YOLOv11m
- Compare: YOLOv11n (baseline) vs YOLOv11n (tuned) vs YOLOv11s vs YOLOv11m vs UBM official

### Phase 5: ONNX Export & Inference Benchmarking (2 hours)
**Critical for deployment:** Measure real-world inference performance

**Models to benchmark:**
- Our baseline (best.pt)
- Our tuned (from sweep)
- UBM official (for comparison)
- YOLOv11s/m (if trained)

**Export to ONNX (batch=2 for stereo):**
```bash
yolo export model=runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt format=onnx batch=2
```

**Benchmark on RTX 4080 Super:**
- Test with ONNX runtime
- Measure: preprocess, inference, postprocess times
- Target: < 8ms total (60fps requires < 16.7ms, but stereo needs 2 inferences)
- Compare against Edo's thesis baseline (6.78ms on RTX 3080 Mobile)

**Deliverable:**
- Inference speed table (YOLOv11n/s/m on ONNX + RTX 4080 Super)
- Comparison: Our models vs UBM official vs Thesis baseline
- Real-time capability confirmation (can it run at 60fps?)

### Files Created
- ✅ `sweep_config.yaml` - Search space definition (13 hyperparameters)
- ✅ `train_sweep.py` - Training script for sweep
- ✅ `launch_sweep.sh` - One-command sweep launcher
- ✅ `analyze_sweep.py` - Extract best config from sweep results
- ✅ `train_best_config.py` - Final 300-epoch training with best config
- ✅ `SWEEP_GUIDE.md` - Complete documentation
- ✅ `evaluate_ubm_model.py` - Evaluate UBM's official model
- ✅ `benchmark_inference.py` - ONNX inference speed benchmarking
- ✅ `RESEARCH_FOCUS_AREAS.md` - Brightness & orange cone challenges from Edo
- ✅ `UBM_EVALUATION_RESULTS.md` - UBM model evaluation (YOU WIN! +6.5%)

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

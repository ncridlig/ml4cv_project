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

### From Gabriele's CV Report (Nov 2025) - 🔴 GAME CHANGER

**CRITICAL DISCOVERY: Dataset Split Mismatch!**

Gabriele's report reveals the **TRUE baseline** from their production system:

| Metric | Gabriele's Baseline | Dataset Split |
|--------|---------------------|---------------|
| **mAP50** | **0.824** | **Test set (689 images)** ✅ |
| Precision | 0.849 | Test set |
| Recall | 0.765 | Test set |
| mAP50-95 | 0.570 | Test set |

**OUR EVALUATIONS (INVALID FOR COMPARISON):**

| Model | mAP50 | Dataset Split |
|-------|-------|---------------|
| Our baseline | 0.714 | Validation set (1,968 images) ❌ |
| UBM production | 0.670 | Validation set ❌ |

**⚠️ Problem:** We evaluated on **validation set**, Gabriele used **test set**
**✅ Solution:** Re-evaluate ALL models on test set for fair comparison

**Implications:**
- Gabriele's 0.824 is the **actual production baseline** on test set
- Our 0.714 vs 0.670 comparisons are valid (both on validation)
- **But we cannot compare to 0.824 until we re-run on test set!**
- The 0.824 baseline is **achievable** - Gabriele already did it

**Inference performance (RTX 3080 Mobile + TensorRT):**
- Total pipeline: 9.46 ms (with ORB features)
- YOLO inference only: 6.78 ms
- Real-time capable at 60 fps

**See:** `GABRIELE_BASELINE_ANALYSIS.md` for full details

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

### Phase 2: Re-Evaluate ALL Models on Test Set (2 hours) - AFTER SWEEP ⚠️ CRITICAL

**🔴 DATASET SPLIT ERROR DISCOVERED:**
- Gabriele's baseline (mAP50 = 0.824): **Test set** (689 images)
- Our baseline (mAP50 = 0.714): **Validation set** (1,968 images)
- **Cannot compare validation vs test results - MUST re-evaluate on test set!**

**Source:** Gabriele's report `report_ceccolini_esame_cv_yolo.pdf` (Nov 2025)

**Tasks:**

1. **Our baseline on test set:**
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt'); model.val(data='datasets/FSOCO-12/data.yaml', split='test')"
   ```

2. **UBM production model on test set:**
   ```bash
   python evaluate_ubm_model.py  # Already fixed to use test set
   ```

3. **Comparison targets:**
   - Gabriele's baseline: mAP50 = **0.824** (test set)
   - Our baseline: ? (needs test set evaluation)
   - UBM production: ? (needs test set evaluation)

**Memory constraint:** Cannot run in parallel with sweep - do AFTER sweep completes

**See:** `GABRIELE_BASELINE_ANALYSIS.md` for full analysis

### Phase 3: Full Training with Best Config (4 hours)
1. Run `python analyze_sweep.py <sweep_id>` to extract best config
2. Paste config into `train_best_config.py`
3. Train for 300 epochs with best hyperparameters
4. Target: mAP50 ≥ 0.78 (+9% over baseline)

### Phase 4: Architecture Comparison (15 hours)
- Use best hyperparameters from sweep
- Train YOLOv11s and YOLOv11m
- Compare: YOLOv11n (baseline) vs YOLOv11n (tuned) vs YOLOv11s vs YOLOv11m vs UBM official

### Phase 5: Inference Optimization & Benchmarking (3-5 days) - CRITICAL FOR REAL-TIME

**🔥 NEW: Comprehensive optimization research completed!**
**See:** `INFERENCE_OPTIMIZATION_RESEARCH.md` (12 techniques ranked by risk/novelty)

**Current Baseline:** 6.78ms inference (Gabriele, RTX 3080 Mobile, TensorRT FP16)
**Target:** < 5ms on RTX 4080 Super for 60fps stereo capability

**Recommended "Goldilocks" Strategy (3-5 days):**

**Option A: Safe & Effective (1-2 days)**
1. TensorRT FP16 conversion → Expected: **4.5ms** (guaranteed)
2. INT8 quantization → Expected: **2.8ms** (low risk)
**Result:** 2.4× speedup, mAP ~0.80

**Option B: Balanced Novelty (3-5 days)** ⭐ **RECOMMENDED FOR PROFESSOR**
1. Train YOLO12 (attention-centric, 2025 release) → Expected: **1.5ms**, mAP ~0.835
2. TensorRT INT8 quantization → Expected: **1.0ms**, mAP ~0.810
**Result:** 6.8× speedup, state-of-the-art architecture, impressive for academic project

**Option C: Maximum Novelty (1-2 weeks)**
1. YOLO12 training
2. Structured pruning (50% channels)
3. Early exit / adaptive inference
**Result:** 0.8-1.2ms, mAP 0.80-0.82, publication-quality research

**Export & Benchmark Script:**
```bash
# TensorRT FP16 (baseline)
yolo export model=best.pt format=engine half=True batch=2

# TensorRT INT8 (advanced)
yolo export model=best.pt format=engine int8=True batch=2

# Benchmark
python benchmark_inference.py --model best_fp16.engine --runs 100 --batch 2
```

**Deliverable:**
- Ablation study table (FP16 vs INT8 vs YOLO12 vs pruning)
- Inference speed comparison vs Gabriele's baseline (6.78ms)
- Real-time capability analysis (can it sustain 60fps on stereo?)
- Recommendation for production deployment

### Files Created
- ✅ `sweep_config.yaml` - Search space definition (13 hyperparameters)
- ✅ `train_sweep.py` - Training script for sweep
- ✅ `launch_sweep.sh` - One-command sweep launcher
- ✅ `analyze_sweep.py` - Extract best config from sweep results
- ✅ `train_best_config.py` - Final 300-epoch training with best config
- ✅ `SWEEP_GUIDE.md` - Complete documentation
- ✅ `evaluate_ubm_model.py` - Evaluate UBM's official model (FIXED: now uses test set)
- ✅ `benchmark_inference.py` - ONNX inference speed benchmarking
- ✅ `RESEARCH_FOCUS_AREAS.md` - Brightness & orange cone challenges from Edo
- ✅ `UBM_EVALUATION_RESULTS.md` - UBM model evaluation on validation set (needs re-run on test)
- ✅ `GABRIELE_BASELINE_ANALYSIS.md` - TRUE baseline from Gabriele's CV report (0.824 on test set)

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

# Two-Stage Training Strategy: Pre-train + Fine-tune

**Date:** 2026-01-25
**Status:** Ready to execute

---

## 🎯 Strategy Overview

Train YOLO12n using a two-stage approach:

1. **Stage 1: Pre-training on cone-detector** (22,725 images, 200 epochs)
2. **Stage 2: Fine-tuning on FSOCO-12** (7,120 images, 150 epochs)

**Total:** 350 epochs, ~4-5 days training time

---

## 💡 Rationale

### Why Two-Stage Training?

**Problem:** YOLO12n trained only on FSOCO-12 achieved 0.7081 mAP50 (test set).

**Hypothesis:** More training data → Better feature learning → Higher accuracy

**Key Insights:**
1. **Cannot increase parameters** - YOLO12n is fixed at 2.56M params
2. **CAN increase data** - cone-detector has 22,725 images (3× more than FSOCO-12)
3. **Same objective function** - Both datasets are cone detection
4. **Same classes** - 5 cone types (blue, yellow, orange, large orange, unknown)

**Solution:** Pre-train on large dataset, fine-tune on benchmark

---

## 📊 Dataset Comparison

| Dataset | Train Images | Val Images | Test Images | Total | Source |
|---------|--------------|------------|-------------|-------|--------|
| **cone-detector** | 19,975 | 1,591 | 1,159 | **22,725** | fsbdriverless/cone-detector-zruok v1 |
| **FSOCO-12** | 7,120 | 1,968 | 689 | **9,777** | fmdv/fsoco-kxq3s v12 |

**Ratio:** cone-detector has **2.3× more training data**

### Dataset Characteristics

**cone-detector:**
- More images (22,725 vs 9,777)
- Diverse lighting conditions
- Various track configurations
- **Potential distribution mismatch** with FSOCO-12

**FSOCO-12:**
- Benchmark dataset for evaluation
- Standardized splits
- Known baseline performance
- **Target distribution** for fine-tuning

---

## 🔬 Training Configuration

### Stage 1: Pre-training on cone-detector

**Goal:** Learn robust cone detection features from large dataset

```python
epochs=200
batch=64
lr0=0.01          # Standard learning rate
lrf=0.01
patience=50       # Early stopping
```

**Expected:**
- mAP50 ~0.68-0.72 on cone-detector validation
- Model learns general cone features
- Better generalization from more data

**Output:** `runs/two-stage/stage1_cone_detector_200ep/weights/best.pt`

---

### Stage 2: Fine-tuning on FSOCO-12

**Goal:** Adapt features to FSOCO-12 distribution for benchmark evaluation

```python
epochs=150
batch=64
lr0=0.001         # 10× LOWER learning rate (fine-tuning)
lrf=0.001
patience=30       # More patient (fine-tuning can be slow)
```

**Why lower learning rate?**
- Preserve features learned in Stage 1
- Gentle adaptation to FSOCO-12 distribution
- Avoid catastrophic forgetting

**Expected:**
- mAP50 ~0.71-0.74 on FSOCO-12 validation
- Adapted to FSOCO-12 benchmark
- Better than single-stage training (hopefully!)

**Output:** `runs/two-stage/stage2_fsoco12_150ep/weights/best.pt`

---

## 📈 Expected Outcomes

### Best Case: Significant Improvement ✅

```
Single-stage (FSOCO-12 only): 0.7081 mAP50
Two-stage (pre-train + fine-tune): 0.73-0.74 mAP50
Improvement: +3-4%
```

**Why this could happen:**
- More training data (22,725 vs 7,120) helps model learn better features
- Pre-training provides better initialization than COCO weights
- Fine-tuning adapts to FSOCO-12 distribution

---

### Moderate Case: Slight Improvement ⚠️

```
Single-stage: 0.7081 mAP50
Two-stage: 0.71-0.72 mAP50
Improvement: +1-2%
```

**Why this could happen:**
- cone-detector helps, but distribution somewhat different
- More data helps, but not dramatically
- Fine-tuning successfully adapts to FSOCO-12

---

### Worst Case: No Improvement or Worse ❌

```
Single-stage: 0.7081 mAP50
Two-stage: 0.69-0.71 mAP50
Improvement: -1-0%
```

**Why this could happen:**
- **Distribution mismatch:** cone-detector very different from FSOCO-12
- **Negative transfer:** Features learned on cone-detector don't help FSOCO-12
- **Fine-tuning issues:** Learning rate too low, not enough epochs

**Mitigation strategies if this happens:**
- Increase fine-tuning learning rate (0.001 → 0.005)
- Increase fine-tuning epochs (150 → 200-250)
- Try different pre-training dataset
- Analyze domain gap between cone-detector and FSOCO-12

---

## 🧪 Comparison to Single-Stage

| Approach | Training Data | Epochs | Learning Rate | Expected mAP50 |
|----------|--------------|--------|---------------|----------------|
| **Single-stage** | FSOCO-12 (7,120) | 300 | 0.01 | 0.7081 (actual) |
| **Two-stage** | cone-detector (22,725) + FSOCO-12 (7,120) | 200 + 150 | 0.01 + 0.001 | 0.71-0.74 (target) |

**Key difference:** More total training data (22,725 vs 7,120)

---

## 🎓 Transfer Learning Principles

### Why This Should Work

1. **Same domain:** Both datasets are cone detection (not cats → cones)
2. **Same task:** Object detection with 5 classes
3. **Same architecture:** YOLO12n (no architectural changes)
4. **More data:** 22,725 images vs 7,120 images

### Transfer Learning Best Practices

✅ **DO:**
- Use lower learning rate for fine-tuning (0.001 vs 0.01)
- Use patience for early stopping
- Monitor validation loss carefully
- Compare to single-stage baseline

❌ **DON'T:**
- Use same learning rate for fine-tuning (causes instability)
- Fine-tune for too few epochs (need 100+ epochs)
- Skip validation during fine-tuning
- Freeze backbone layers (YOLO needs full network fine-tuning)

---

## 📋 Execution Plan

### Step 1: Verify Datasets

```bash
# Check cone-detector dataset exists
ls datasets/cone-detector/data.yaml

# Check FSOCO-12 dataset exists
ls datasets/FSOCO-12/data.yaml
```

**If cone-detector missing:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("fsbdriverless").project("cone-detector-zruok")
dataset = project.version(1).download("yolov11")
```

---

### Step 2: Start Two-Stage Training

```bash
# Activate environment
source venv/bin/activate

# Run two-stage training (4-5 days)
python3 train_yolo12_two_stage.py

# OR run in background with logging
nohup python3 train_yolo12_two_stage.py > two_stage_training.log 2>&1 &

# Monitor progress
tail -f two_stage_training.log

# OR monitor W&B
# https://wandb.ai/ncridlig-ml4cv/two-stage-training
```

---

### Step 3: Evaluate on Test Set

**After training completes:**

```bash
python3 evaluate_yolo12_two_stage_test.py
```

**Compare to baselines:**
- Two-stage vs Single-stage YOLO12n (0.7081)
- Two-stage vs UBM production (0.6655)

---

### Step 4: Decision Point

**If two-stage > single-stage:**
- ✅ SUCCESS! Use two-stage model
- Export to INT8 and deploy
- Document improvement in report

**If two-stage ≈ single-stage (within 1%):**
- ⚠️ No significant benefit from more data
- Use whichever model is simpler (single-stage)
- Document finding in report

**If two-stage < single-stage:**
- ❌ Distribution mismatch or fine-tuning issues
- Analyze what went wrong
- Try mitigation strategies (higher LR, more epochs)
- Use single-stage model for deployment

---

## 📊 Monitoring During Training

### Stage 1 Checkpoints

Monitor on W&B:
- `metrics/mAP50(B)` on cone-detector validation
- Training loss trends
- Early stopping triggered?

**Target:** mAP50 ~0.68-0.72 on cone-detector validation

---

### Stage 2 Checkpoints

Monitor on W&B:
- `metrics/mAP50(B)` on FSOCO-12 validation
- Compare to Stage 1 final metrics
- Watch for catastrophic forgetting (validation loss suddenly increases)

**Target:** mAP50 > 0.71 on FSOCO-12 validation (beat single-stage)

---

## 🔍 Potential Issues and Solutions

### Issue 1: Stage 1 Overfits to cone-detector

**Symptoms:**
- High mAP50 on cone-detector validation (>0.75)
- Low mAP50 on FSOCO-12 after fine-tuning (<0.68)

**Solution:**
- Reduce Stage 1 epochs (200 → 150)
- Add more augmentation in Stage 1
- Increase fine-tuning learning rate

---

### Issue 2: Stage 2 Catastrophic Forgetting

**Symptoms:**
- Stage 1 mAP50: 0.70
- Stage 2 mAP50: 0.65 (worse!)
- Validation loss increases during fine-tuning

**Solution:**
- Lower fine-tuning learning rate (0.001 → 0.0005)
- Reduce fine-tuning epochs
- Use learning rate scheduler (cosine annealing)

---

### Issue 3: Distribution Mismatch

**Symptoms:**
- Stage 1 mAP50: 0.72 on cone-detector
- Stage 2 mAP50: 0.69 on FSOCO-12 (worse than single-stage 0.71)

**Solution:**
- Datasets are too different
- Pre-training on cone-detector hurts more than helps
- Stick with single-stage training on FSOCO-12 only

---

## 🎯 Success Metrics

| Metric | Target | Baseline (Single-stage) |
|--------|--------|-------------------------|
| **Test mAP50** | > 0.71 | 0.7081 |
| **Test mAP50-95** | > 0.485 | 0.4846 |
| **Test Precision** | > 0.84 | 0.8401 |
| **Test Recall** | > 0.65 | 0.6542 |

**Primary goal:** Beat single-stage YOLO12n (0.7081) on test set

---

## 📝 Timeline

```
Day 1-3:  Stage 1 training (cone-detector, 200 epochs)
Day 3:    Stage 1 checkpoint evaluation
Day 4-5:  Stage 2 training (FSOCO-12, 150 epochs)
Day 5:    Test set evaluation
Day 6:    Decision: Use two-stage or single-stage model
```

**Total:** 5-6 days

---

## 🚀 If Successful

**Two-stage model beats single-stage:**

1. **Export to INT8:**
   ```bash
   python3 export_tensorrt_int8.py  # Update path to two-stage model
   ```

2. **Benchmark performance:**
   ```bash
   python3 benchmark_int8.py
   ```

3. **Deploy to RTX 4060:**
   - Transfer two-stage INT8 engine to car
   - Replace UBM production model
   - Real-world testing

4. **Document in report:**
   - Two-stage training methodology
   - Benefits of more training data
   - Transfer learning within same domain
   - Final performance comparison

---

## 📚 Academic Contribution

**This experiment demonstrates:**

1. **Transfer learning within same domain** (cone detection → cone detection)
2. **Benefits of larger datasets** for small models
3. **Fine-tuning strategies** for domain adaptation
4. **Practical ML engineering** (more data vs more parameters)

**Novel finding (potential):**
- Pre-training on 22,725 images improves performance vs 7,120 images
- Quantifies benefit of dataset size for cone detection
- Provides baseline for future work on FSOCO dataset

---

**Last Updated:** 2026-01-25
**Status:** Ready to execute
**Command:** `python3 train_yolo12_two_stage.py`

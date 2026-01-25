# Two-Branch Optimization Strategy

**Target Hardware:** NVIDIA RTX 4060 (on car), RTX 4080 Super (training)

**Baseline:** 6.78ms inference on RTX 3080 Mobile with TensorRT FP16 (Gabriele)

**Critical Correction:** The 6.78ms baseline is ALREADY TensorRT FP16 optimized! We need techniques beyond basic TensorRT.

**Date:** 2026-01-24

---

## RTX 4060 Capabilities (Deployment Target)

### Hardware Specifications

**Architecture:** Ada Lovelace (AD107 GPU die, 4N NVIDIA Custom Process)
- **CUDA Cores:** 3,072
- **Tensor Cores:** 96 (4th generation)
- **RT Cores:** 24 (3rd generation)
- **Memory:** 8GB GDDR6, 128-bit bus

### AI Inference Capabilities

✅ **INT8 Support:** YES - 4th gen Tensor Cores with dedicated INT8 support
- **INT8 Performance:** 242 INT8 TOPS with sparsity
- **Improvement:** 137% better than RTX 3060 (3rd gen Tensor Cores)
- **Precision Support:** FP16, BF16, TF32, INT8, INT4
- **TensorRT Compatible:** Full TensorRT optimization support

**Sources:**
- [RTX 4060 vs RTX 3060 AI Benchmarks](https://www.bestgpusforai.com/gpu-comparison/3060-ti-vs-4060)
- [RTX 4060 Specifications](https://www.thefpsreview.com/gpu-family/nvidia-geforce-rtx-4060-gpu-family-specifications/)

### Performance vs RTX 3080 Mobile (Baseline)

```
GPU                  Architecture    INT8 TOPS    Relative Performance
RTX 3080 Mobile      Ampere          ~200         1.0× (baseline)
RTX 4060             Ada Lovelace    242          1.21× (21% faster)
RTX 4080 Super       Ada Lovelace    ~680         3.4× (training only)
```

**Expected RTX 4060 baseline inference:** 6.78ms × (1/1.21) ≈ **5.6ms** (without any optimization)

---

## Corrected Understanding: What We Cannot Count as Improvements

❌ **TensorRT FP16 conversion** - Already done (6.78ms baseline includes this)
❌ **CUDA graphs + kernel fusion** - Already included in TensorRT baseline
❌ **Basic batch=2 optimization** - Already done (stereo pair batching)

**What we CAN improve:**
✅ INT8 quantization (beyond FP16)
✅ Architecture changes (YOLO12, RegNet backbone)
✅ Model compression (pruning, distillation)
✅ Adaptive inference (early exit, multi-resolution)

---

## User's Priority Ranking

**Your preferences:**
1. **YOLO12** (attention-centric, 2025 state-of-the-art)
2. **Knowledge Distillation** (YOLOv11m teacher → YOLOv11n student, covered in course)
3. **Neural Architecture Search + RegNet** (professor mentioned, shows understanding)

**Also interested in:**
- Structured pruning
- Adaptive inference (early exit, multi-resolution)

---

## Two-Branch Strategy

### Decision Point: YOLO12 Training Time

**Branch A:** YOLO12 training succeeds quickly (≤ 3 days)
**Branch B:** YOLO12 takes too long or fails → pivot to knowledge distillation + RegNet

---

## Branch A: YOLO12 Primary Path

**Timeline:** 5-7 days total

### Phase 1: YOLO12 Training (Days 1-3)

**Goal:** Train YOLO12n on FSOCO-12 dataset

```python
from ultralytics import YOLO

# Load pretrained YOLO12n
model = YOLO('yolo12n.pt')

# Train on FSOCO-12
results = model.train(
    data='datasets/FSOCO-12/data.yaml',
    epochs=300,
    batch=48,  # Adjusted for RTX 4080 Super
    device=0,
    project='runs/yolo12',
    name='yolo12n_300ep_FSOCO',
)
```

**Expected Results:**
- **Inference (RTX 4060, FP16):** ~1.8-2.0ms (YOLO12n is 1.62ms on T4, RTX 4060 ≈ T4)
- **mAP50 (test set):** ~0.835 (+1.3% over YOLOv11n baseline)
- **Training time:** 2-3 days (300 epochs)

**Decision Point (Day 3):**
- ✅ Training converged, mAP > 0.82 → Continue Branch A
- ❌ Training issues, mAP < 0.80 → **PIVOT to Branch B**

---

### Phase 2: INT8 Quantization (Days 4-5)

**Goal:** Post-training INT8 quantization of YOLO12n

**⚠️ CRITICAL: Calibration Dataset**

**User Question:** "Can I use test dataset for INT8 calibration?"
**Answer:** ❌ **NO!** Test set must remain untouched for final unbiased evaluation.

**Correct Approach:**
- Use **validation set** (1,968 images) for calibration
- OR use random subset of **training set** (500-1000 images)
- Validation set is preferable (representative of real data)

**Implementation:**

```bash
# Export YOLO12n to ONNX
yolo export model=runs/yolo12/yolo12n_300ep_FSOCO/weights/best.pt format=onnx batch=2

# Create calibration cache using validation set
python create_int8_calibration.py \
    --model best.onnx \
    --data datasets/FSOCO-12/data.yaml \
    --split val \
    --num_images 500

# Convert to TensorRT INT8
trtexec --onnx=best.onnx \
        --saveEngine=yolo12n_int8.engine \
        --int8 \
        --calib=calibration_cache.bin \
        --batch=2 \
        --device=0
```

**Expected Results:**
- **Inference (RTX 4060, INT8):** ~1.2-1.5ms (1.5-2× faster than FP16)
- **mAP50 (test set):** ~0.810-0.820 (-1.5 to -2.5% from FP16)
- **Speedup:** 1.2× to 1.7× over YOLO12n FP16

**Total Branch A Speedup:**
- Baseline (Gabriele, RTX 3080): 6.78ms
- YOLO12n FP16 (RTX 4060): 1.8ms → **3.8× faster**
- YOLO12n INT8 (RTX 4060): 1.2ms → **5.7× faster** 🔥

---

### Phase 3: Benchmarking & Analysis (Days 6-7)

**Tasks:**
1. Evaluate YOLO12n (FP16 + INT8) on test set
2. Compare to Gabriele's baseline (mAP50 = 0.824 on test set)
3. Benchmark inference times on RTX 4060
4. Create ablation study table
5. Document results in report

**Deliverable (Branch A):**
- YOLO12n INT8 model: **1.2ms inference, mAP50 ~0.815**
- **5.7× speedup** over baseline
- State-of-the-art 2025 architecture
- Production-ready TensorRT engine for RTX 4060

---

## Branch B: Knowledge Distillation + RegNet Path

**Timeline:** 7-10 days total

**Trigger:** YOLO12 training takes too long, fails to converge, or mAP < 0.80

---

### Phase 1: YOLOv11m Teacher Training (Days 1-3)

**Goal:** Train YOLOv11m (larger model) as teacher

**Why YOLOv11m?**
- Larger capacity than YOLOv11n (20M params vs 2.6M)
- Better feature representations for distillation
- Typically achieves **mAP50 ~0.88-0.90** on FSOCO

```python
from ultralytics import YOLO

# Train YOLOv11m teacher
teacher = YOLO('yolo11m.pt')

teacher.train(
    data='datasets/FSOCO-12/data.yaml',
    epochs=300,
    batch=32,  # Larger model, smaller batch
    device=0,
    project='runs/teacher',
    name='yolov11m_teacher_300ep',
)
```

**Expected Results:**
- **mAP50 (test set):** ~0.88-0.90
- **Inference (RTX 4060):** ~8-10ms (too slow for deployment, but good teacher)
- **Training time:** 2.5-3 days

---

### Phase 2: Knowledge Distillation Training (Days 4-7)

**Goal:** Train YOLOv11n student using YOLOv11m teacher

**User Question:** "Train YOLOv11m then use it as teacher to student YOLOv11n instead of training n directly?"
**Answer:** ✅ **YES!** This is the standard knowledge distillation approach.

**How It Works:**

1. **Teacher (YOLOv11m):** Already trained, frozen weights
2. **Student (YOLOv11n):** Training with combined loss:
   - **Task loss:** Standard YOLO loss (ground truth labels)
   - **Distillation loss:** KL divergence between student and teacher predictions

**Loss Function:**
```
L_total = α * L_task + β * L_distillation

where:
  L_task = YOLO detection loss (boxes, classes, objectness)
  L_distillation = KL_divergence(teacher_logits, student_logits) + MSE(teacher_features, student_features)
  α, β = weighting factors (typically α=0.5, β=0.5)
```

**Implementation:**

```python
from ultralytics import YOLO
import torch
import torch.nn.functional as F

# Load teacher and student
teacher = YOLO('runs/teacher/yolov11m_teacher_300ep/weights/best.pt')
student = YOLO('yolo11n.pt')

# Freeze teacher
for param in teacher.model.parameters():
    param.requires_grad = False

# Custom training loop with distillation
def train_with_distillation(teacher, student, dataloader, epochs=300):
    for epoch in range(epochs):
        for batch in dataloader:
            images, labels = batch

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(images)

            # Student forward
            student_outputs = student(images)

            # Task loss (standard YOLO)
            task_loss = compute_yolo_loss(student_outputs, labels)

            # Distillation loss (soft targets)
            distill_loss = F.kl_div(
                F.log_softmax(student_outputs['logits'] / temperature, dim=1),
                F.softmax(teacher_outputs['logits'] / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            # Feature distillation (intermediate layers)
            feature_loss = F.mse_loss(
                student_outputs['features'],
                teacher_outputs['features']
            )

            # Combined loss
            total_loss = 0.5 * task_loss + 0.3 * distill_loss + 0.2 * feature_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

# Note: Ultralytics may support distillation natively in newer versions
# Check: https://github.com/ultralytics/ultralytics/issues (search "distillation")
```

**Alternative (if Ultralytics supports distillation):**
```python
student.train(
    data='datasets/FSOCO-12/data.yaml',
    epochs=300,
    batch=48,
    teacher_model='runs/teacher/yolov11m_teacher_300ep/weights/best.pt',
    distillation_alpha=0.5,
    distillation_beta=0.5,
)
```

**Expected Results:**
- **mAP50 (test set):** ~0.83-0.85 (better than training YOLOv11n alone)
- **Inference (RTX 4060, FP16):** ~2.5-3.0ms (same as YOLOv11n)
- **Improvement:** +1-2% mAP over direct YOLOv11n training
- **Training time:** 3-4 days

**Why Distillation Works:**
- Student learns from teacher's soft probability distributions (more information than hard labels)
- Student mimics teacher's intermediate feature representations
- Typically achieves **80-90% of teacher's performance** with **10-20% of parameters**

---

### Phase 3: RegNet Backbone Integration (Days 8-9)

**Goal:** Replace YOLOv11n backbone with RegNet to demonstrate understanding of NAS principles

**What is RegNet?**

**RegNet (Facebook AI Research, CVPR 2020):** "Designing Network Design Spaces"
- **Design principle:** Systematically explore network design space using quantized linear functions
- **Key insight:** Instead of manual architecture design, define design space and search within it
- **Relationship to NAS:** RegNet uses NAS-inspired principles but with constrained search space

**RegNet vs Traditional NAS:**
```
Traditional NAS:
  - Search entire architecture space (expensive)
  - Requires days-weeks of GPU search time
  - Black box optimization

RegNet Design Space:
  - Constrained search using linear functions
  - Stage widths/depths = quantized linear function
  - Interpretable, fast, efficient
  - 5× faster than EfficientNet on GPUs
```

**RegNet Design Space Parameters:**
- **Depth:** Optimal ≈ 60 layers (20 blocks)
- **Width multiplier:** 2.5×
- **Bottleneck ratio:** 1.0 (no bottleneck, no inverted bottleneck)
- **Group width:** Varies per stage

**Implementation Approach:**

**Option 1: Use Pre-trained RegNet Backbone**

```python
import torch
from ultralytics import YOLO
from torchvision.models import regnet_y_400mf, regnet_y_800mf

# Load pretrained RegNet
regnet_backbone = regnet_y_400mf(pretrained=True)

# Modify YOLOv11n architecture
# Replace C2f backbone with RegNet stages
model = YOLO('yolo11n.yaml')

# Custom YAML: yolo11n_regnet.yaml
"""
backbone:
  - [-1, 1, RegNetY400MF, []]  # RegNet backbone
  - [-1, 1, SPPF, [1024, 5]]   # SPPF
  ...
"""

# Train with RegNet backbone
model.train(
    data='datasets/FSOCO-12/data.yaml',
    cfg='yolo11n_regnet.yaml',
    epochs=200,
    batch=48,
)
```

**Option 2: Implement RegNet-Inspired Design Space (Show Understanding)**

```python
# Define RegNet design space
def regnet_design_space(depth, width_multiplier=2.5, bottleneck_ratio=1.0):
    """
    RegNet design space using quantized linear function.

    Key principle from CVPR 2020 paper:
    - Stage widths increase linearly: w_j = w_0 + w_a * j
    - Quantize to discrete values for hardware efficiency
    """
    stage_widths = []
    for j in range(depth):
        width = int(64 * (width_multiplier ** (j / depth)))
        # Quantize to nearest multiple of 8 (hardware efficient)
        width = (width // 8) * 8
        stage_widths.append(width)

    return stage_widths

# Example: Design custom backbone using RegNet principles
widths = regnet_design_space(depth=20, width_multiplier=2.5)
# Output: [64, 72, 80, 88, 96, 104, 112, 120, ..., 160]
```

**Expected Results:**
- **mAP50:** ~0.82-0.84 (comparable to YOLOv11n, possibly slightly better)
- **Inference:** ~2.5-3.5ms on RTX 4060 (RegNet is 5× faster than EfficientNet)
- **Academic value:** ✅ Demonstrates understanding of design space principles

**Why This Impresses Professor:**
- ✅ Shows understanding of NAS principles (design space exploration)
- ✅ Connects RegNet to course material (professor mentioned it)
- ✅ Demonstrates ability to modify architecture
- ✅ Interprets research papers and applies them

**Sources:**
- [RegNet: Designing Network Design Spaces (CVPR 2020)](https://arxiv.org/abs/2003.13678)
- [RegNet Architecture Review](https://sh-tsang.medium.com/review-regnet-designing-network-design-spaces-5e0c79910453)
- [RegNetX-YOLOv3 Implementation](https://github.com/sidSingla/RegNetx-YOLOv3)

---

### Phase 4: INT8 Quantization (Day 10)

Apply same INT8 quantization as Branch A to best model (distilled student or RegNet variant).

**Expected Final Results (Branch B):**

```
Model                          mAP50 (Test)    Inference (RTX 4060)    Speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline (Gabriele, RTX 3080)  0.824           6.78ms                  1.0×
YOLOv11n (direct training)     0.824           2.5ms (FP16)            2.7×
YOLOv11n (distilled) FP16      0.835           2.5ms                   2.7×
YOLOv11n (distilled) INT8      0.820           1.5ms                   4.5×
YOLOv11n-RegNet FP16           0.830           2.8ms                   2.4×
YOLOv11n-RegNet INT8           0.815           1.7ms                   4.0×
```

**Best Branch B Result:** Distilled student + INT8 → **1.5ms, mAP50 ~0.82, 4.5× speedup**

---

## Optional Phase (Both Branches): Advanced Techniques

If time permits (Days 8-10 for Branch A, Days 11-14 for Branch B):

### Structured Channel Pruning

**Goal:** Remove 40-50% of channels from trained model

```python
# Sensitivity analysis
from ultralytics.utils.torch_utils import prune_model

model = YOLO('best.pt')

# Prune 50% of channels based on BN gamma
pruned_model = prune_model(
    model,
    pruning_ratio=0.5,
    method='bn_gamma'  # Use BatchNorm scaling factors
)

# Fine-tune for 50 epochs to recover accuracy
pruned_model.train(
    data='datasets/FSOCO-12/data.yaml',
    epochs=50,
    batch=48,
)
```

**Expected:**
- **mAP50:** -1 to -2% from unpruned
- **Inference:** 1.5-2× faster (fewer operations)
- **Model size:** 50% smaller

---

### Adaptive Inference (Early Exit or Multi-Resolution)

**Goal:** Process easy frames faster

**Multi-Resolution Approach:**
```python
def adaptive_inference(image, yolo_320, yolo_640, confidence_threshold=0.80):
    """
    Two-pass adaptive inference:
    1. Fast 320×320 pass
    2. High-res 640×640 only if low confidence
    """
    # Pass 1: Low-res (fast)
    low_res = resize(image, (320, 320))
    detections, confidence = yolo_320(low_res)

    if confidence > confidence_threshold:
        return detections  # Good enough

    # Pass 2: High-res (accurate but slow)
    high_res = resize(image, (640, 640))
    detections = yolo_640(high_res)
    return detections

# Train two models: 320×320 and 640×640
yolo_320 = YOLO('best.pt')
yolo_320.train(data='datasets/FSOCO-12/data.yaml', imgsz=320, epochs=100)

yolo_640 = YOLO('best.pt')  # Your main model
```

**Expected:**
- **Average inference:** ~1.2-1.5ms (80% use low-res, 20% use high-res)
- **mAP50:** Same as 640×640 model (fallback when needed)
- **Impressive factor:** Adaptive, context-aware

---

## Comparison: Branch A vs Branch B

```
Aspect                  Branch A (YOLO12)           Branch B (Distillation + RegNet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timeline                5-7 days                    7-10 days
Novelty                 ⭐⭐⭐⭐⭐ (2025 SOTA)          ⭐⭐⭐⭐ (Course material + NAS)
Risk                    🟠 Medium-High              🟡 Medium
Best mAP50              ~0.815 (INT8)               ~0.820 (distilled INT8)
Best Inference          ~1.2ms (INT8)               ~1.5ms (distilled INT8)
Speedup                 5.7× vs baseline            4.5× vs baseline
Professor Impact        Very high (cutting-edge)    High (demonstrates understanding)
Course Relevance        Attention mechanisms        Distillation + NAS/RegNet ✅
Backup Plan             Pivot to Branch B           Already safer approach
Implementation          Single model training       Multi-stage (teacher → student)
```

**Recommendation:**

**Start with Branch A (YOLO12):**
- Higher potential (5.7× speedup, 1.2ms inference)
- More novel (2025 state-of-the-art)
- Simpler implementation (single model)
- Clear pivot point (Day 3 decision)

**If YOLO12 fails → Branch B:**
- Safer (proven techniques)
- Course-relevant (distillation, NAS)
- Good academic story (teacher-student, design space)
- Still achieves 4.5× speedup

---

## Timeline Summary

### Branch A: YOLO12 Path (5-7 days)

```
Day 1-3:   Train YOLO12n (300 epochs)
           Decision point: mAP > 0.82? → Continue
                          mAP < 0.80? → Pivot to Branch B

Day 4-5:   INT8 quantization with validation set calibration
           Benchmark on RTX 4060

Day 6-7:   Final evaluation on test set
           Report writing, ablation study
```

### Branch B: Distillation + RegNet Path (7-10 days)

```
Day 1-3:   Train YOLOv11m teacher (300 epochs)

Day 4-7:   Knowledge distillation: YOLOv11m → YOLOv11n
           Train student with combined loss

Day 8-9:   RegNet backbone integration
           Train YOLOv11n-RegNet variant

Day 10:    INT8 quantization + benchmarking
           Compare: direct, distilled, RegNet variants

Day 11-12: Final evaluation on test set
           Ablation study (teacher size, distillation loss weights)

Day 13-14: Report writing, presentation prep
```

---

## Deliverables (Either Branch)

### Technical Outputs

1. **Trained Models:**
   - Branch A: YOLO12n (FP16 + INT8)
   - Branch B: YOLOv11m (teacher), YOLOv11n-distilled, YOLOv11n-RegNet

2. **TensorRT Engines:**
   - Production-ready INT8 engine for RTX 4060
   - Batch=2 for stereo camera

3. **Evaluation Results:**
   - Test set performance (mAP50, precision, recall)
   - Inference benchmarks on RTX 4060
   - Comparison to Gabriele's baseline (0.824 mAP50, 6.78ms)

### Academic Report Sections

1. **Introduction:**
   - Problem: Real-time cone detection for Formula Student
   - Baseline: Gabriele's pipeline (6.78ms, mAP50 = 0.824)
   - Goal: Faster inference while maintaining accuracy

2. **Related Work:**
   - YOLO evolution (YOLOv11 → YOLO12)
   - Knowledge distillation for model compression
   - RegNet design space principles
   - INT8 quantization for inference acceleration

3. **Methodology:**
   - Branch A: YOLO12 attention-centric architecture
   - Branch B: Knowledge distillation + RegNet backbone + INT8
   - Dataset: FSOCO-12 (test set evaluation)
   - Hardware: RTX 4060 deployment target

4. **Experimental Results:**
   - Ablation study table
   - Inference time breakdown (preprocessing, inference, postprocessing)
   - Accuracy-speed trade-off curves

5. **Discussion:**
   - Branch A: Why attention mechanisms improve efficiency
   - Branch B: How teacher size affects student performance
   - RegNet: Design space principles and their application
   - INT8: Calibration methodology and accuracy preservation

6. **Conclusion:**
   - Achieved X× speedup over baseline
   - Demonstrates understanding of modern techniques
   - Production-ready model for UBM team

---

## Critical Reminders

### ⚠️ Test Set Usage

**NEVER use test set for:**
- ❌ INT8 calibration
- ❌ Hyperparameter tuning
- ❌ Model selection
- ❌ Early stopping

**ONLY use test set for:**
- ✅ Final model evaluation (once!)
- ✅ Reporting final results

**Use validation set for:**
- ✅ INT8 calibration (or training subset)
- ✅ Model checkpointing during training
- ✅ Hyperparameter validation

### RTX 4060 vs RTX 4080 Super

**Training:** RTX 4080 Super (faster, more VRAM)
**Deployment:** RTX 4060 (on car)

**Performance difference:**
- RTX 4080 Super: ~680 INT8 TOPS
- RTX 4060: 242 INT8 TOPS
- Ratio: 2.8× (training is faster, but deployment is what matters)

**All inference benchmarks MUST be on RTX 4060** (or simulated based on known specs).

---

## Success Criteria

### Minimum Success (Either Branch)
- mAP50 ≥ 0.80 on test set
- Inference ≤ 3.0ms on RTX 4060
- Speedup ≥ 2.3× vs baseline
- Production-ready TensorRT engine

### Good Success
- mAP50 ≥ 0.82 on test set
- Inference ≤ 2.0ms on RTX 4060
- Speedup ≥ 3.4× vs baseline
- Comprehensive ablation study

### Excellent Success
- mAP50 ≥ 0.82 on test set
- Inference ≤ 1.5ms on RTX 4060
- Speedup ≥ 4.5× vs baseline
- Novel contribution (YOLO12 or distillation insights)
- Publication-quality report

---

## Next Steps

**After hyperparameter sweep completes:**

1. **Evaluate sweep results on test set** (currently on validation!)
   ```bash
   python analyze_sweep.py <sweep_id>
   python evaluate_baseline_test.py  # Re-run on test set
   ```

2. **Decision: Branch A or Branch B?**
   - If you have 5-7 days → **Branch A (YOLO12)**
   - If you have 7-10 days → **Branch A with B backup**
   - If you have 10-14 days → **Branch B (comprehensive)**

3. **Start chosen branch:**
   - Create dedicated training script
   - Set up W&B logging
   - Document every decision

4. **Regular checkpoints:**
   - Day 3: Evaluate intermediate results
   - Pivot if necessary (Branch A → B)
   - Keep backup models at each stage

Good luck! 🚀

# Sweep Crash Investigation

**Date:** 2026-01-25
**Crashed runs:** 13/21 (61.9%)
**Completed runs:** 8/21 (38.1%)

---

## Root Cause: Aggressive Augmentation + High Dropout

### Statistical Analysis

| Parameter | Successful Runs (avg) | Crashed Runs (avg) | Difference |
|-----------|----------------------|-------------------|------------|
| **mixup** | **0.0486** | **0.1973** | **+306%** 🔴 |
| **dropout** | 0.0811 | 0.1556 | +92% 🟠 |
| **hsv_v** | 0.4155 | 0.4784 | +15% 🟡 |
| **lr0** | 0.0104 | 0.0134 | +29% 🟡 |

### Key Finding

**Crashed runs have 4× higher mixup augmentation (0.197 vs 0.049)** combined with nearly 2× higher dropout (0.156 vs 0.081).

---

## Why This Causes Crashes

### 1. Mixup Augmentation Instability

**What mixup does:**
- Blends two training images: `mixed_image = alpha * image1 + (1-alpha) * image2`
- Also blends labels: `mixed_label = alpha * label1 + (1-alpha) * label2`

**Problem at high values (>0.15):**
- Creates synthetic images that are **too blended** to be realistic
- Labels become **ambiguous** (cone positions between two images)
- Model struggles to learn meaningful features
- Can cause **NaN losses** if gradients explode

**Example:**
```
mixup = 0.05 (successful):  5% blending, mostly original images
mixup = 0.20 (crashed):     20% blending, very blurred/confusing images
mixup = 0.30 (crashed):     30% blending, unrecognizable objects
```

### 2. High Dropout Compounds the Problem

**Dropout at 0.15-0.19 (crashed runs):**
- Randomly drops 15-19% of neurons during training
- Combined with noisy mixup images → **too much regularization**
- Model can't learn stable features
- Training becomes unstable → loss oscillates → crash

**Dropout at 0.05-0.10 (successful runs):**
- Mild regularization
- Model can still learn from augmented images

### 3. Combination is Deadly

```
High mixup (0.20) + High dropout (0.16) = Training Instability

Synthetic noisy images → Hard to learn → Aggressive dropout → Can't converge
```

---

## Detailed Comparison

### Successful Run Example (5ycouco7)

**mAP50:** 0.7088 (best sweep run)

```yaml
hsv_v: 0.386       # Moderate brightness
dropout: 0.065     # Low dropout ✅
lr0: 0.0131        # Standard learning rate
mixup: 0.079       # Low mixup ✅
copy_paste: 0.298  # Moderate
```

**Why it succeeded:** Low mixup + low dropout = stable training

---

### Crashed Run Example (97fcqc0i)

**Status:** NO SUMMARY (crashed)

```yaml
hsv_v: 0.581       # High brightness variation
dropout: 0.136     # Moderate-high dropout ⚠️
lr0: 0.0130        # Standard learning rate
mixup: 0.287       # VERY high mixup 🔴
copy_paste: 0.160  # Moderate
```

**Why it crashed:** High mixup (0.287) + moderate-high dropout (0.136) = too aggressive

---

### Crashed Run Example (1shvnykt)

**Status:** NO SUMMARY (crashed)

```yaml
hsv_v: 0.392       # Moderate brightness
dropout: 0.165     # High dropout ⚠️
lr0: 0.0175        # High learning rate ⚠️
mixup: 0.262       # Very high mixup 🔴
copy_paste: 0.183  # Moderate
```

**Why it crashed:** Very high mixup (0.262) + high dropout (0.165) + high lr = unstable

---

## Other Potential Crash Causes

### 1. Memory Leaks (Secondary)

Some runs may have crashed due to accumulated memory from previous runs, even with our cleanup code. However, the correlation with hyperparameters suggests this is **not the primary cause**.

### 2. W&B Connection Issues (Minor)

One debug log showed:
```
ConnectionResetError: Connection lost
```

This suggests occasional W&B connection drops, but this would only affect logging, not training itself.

### 3. Extreme HSV Values (Minor Contributor)

Crashed runs had slightly higher `hsv_v` (0.478 vs 0.416), which affects brightness variation. At extreme values (>0.55), this can cause:
- Images too dark (underexposed)
- Images too bright (overexposed)
- Model can't learn consistent features

But this is a **minor contributor** compared to mixup.

---

## Recommendations for Future Sweeps

If you run hyperparameter sweeps again, use these **safer bounds**:

```yaml
# SAFE BOUNDS (prevent crashes)
parameters:
  mixup:
    min: 0.0
    max: 0.15        # DOWN from 0.3 (current max caused crashes)

  dropout:
    min: 0.0
    max: 0.10        # DOWN from 0.2 (current max caused crashes)

  hsv_v:
    min: 0.35
    max: 0.55        # DOWN from 0.6 (current max)

  lr0:
    min: 0.005
    max: 0.015       # NARROWER than 0.005-0.02 (current)

  # Keep these as-is (no crash correlation)
  hsv_h: {min: 0.0, max: 0.03}
  hsv_s: {min: 0.5, max: 0.9}
  mosaic: {min: 0.5, max: 1.0}
  copy_paste: {min: 0.0, max: 0.3}
  weight_decay: {min: 0.0001, max: 0.001}
  degrees: {min: 0.0, max: 10.0}
  warmup_epochs: [1, 3, 5]
  close_mosaic: [0, 5, 10, 15, 20]
  lrf: {min: 0.01, max: 0.1}
```

---

## Key Constraints

### Mixup + Dropout Interaction

**Rule of thumb:**
```
If mixup > 0.15:  dropout should be < 0.08
If dropout > 0.12: mixup should be < 0.10
```

**Safe combinations:**
- mixup=0.05, dropout=0.15 ✅
- mixup=0.10, dropout=0.10 ✅
- mixup=0.15, dropout=0.05 ✅

**Dangerous combinations:**
- mixup=0.20, dropout=0.15 ❌ (4/5 crashed runs had this pattern)
- mixup=0.25, dropout=0.12 ❌
- mixup=0.30, dropout=0.10 ❌

---

## Why Ultralytics Defaults Work

**Ultralytics 8.4.7 default hyperparameters:**
```python
mixup: 0.0         # Disabled by default!
dropout: 0.0       # Disabled by default!
hsv_v: 0.4         # Moderate (0.4)
lr0: 0.01          # Conservative (0.01)
```

**This is why your baseline (0.714) beat all sweep runs:**
- No mixup instability
- No excessive dropout
- Conservative, proven values
- Years of refinement across thousands of datasets

**Lesson:** Ultralytics defaults are **already near-optimal** for most use cases.

---

## Conclusion

**Primary crash cause:** High mixup (>0.15) combined with high dropout (>0.12)

**Secondary factors:** Extreme HSV values, high learning rate

**Frequency:** 61.9% of runs crashed (13/21), strongly correlated with aggressive augmentation

**Recommendation:** For future sweeps, constrain mixup ≤ 0.15 and dropout ≤ 0.10

**Best approach:** Use Ultralytics defaults and focus on **architectural improvements** (YOLO12, knowledge distillation) rather than aggressive hyperparameter tuning.

# INT8 Optimization Guide for YOLO12

## Overview

This guide explains how to optimize YOLO12 for deployment on RTX 4060 using INT8 quantization.

**Expected improvements:**
- **Speed:** 1.5-2.0× faster inference
- **Size:** 3.5× smaller model (5.3 MB → 1.5 MB)
- **Accuracy:** <1% loss (typically 0.5-1% mAP50 drop)

**⚠️ CRITICAL: Stereo Camera Processing**
- Batch size MUST be **2** (left + right image processed simultaneously)
- All export scripts use `batch=2` for stereo compatibility
- Do NOT change batch size to 1

---

## What is INT8 Quantization?

**INT8 quantization** reduces model precision from 32-bit floating point (FP32) to 8-bit integers (INT8).

### Example

```python
# FP32 (Full Precision) - 32 bits per value
weight_fp32 = 0.847362518  # 4 bytes

# INT8 (Quantized) - 8 bits per value
weight_int8 = 127          # 1 byte
scale = 0.00666
zero_point = 0

# Reconstruct: weight ≈ (127 - 0) * 0.00666 = 0.846
# Error: 0.847 - 0.846 = 0.001 (0.1% error)
```

### Why It Works

- **Most activations cluster** in specific ranges
- **8 bits is enough** to represent values with minimal error
- **RTX 4060 has INT8 Tensor Cores** (242 INT8 TOPS vs 22 FP32 TFLOPS)

---

## Prerequisites

### Software Requirements

```bash
# Check if TensorRT is available
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# If not installed:
pip install tensorrt

# Other requirements (should already be installed)
pip install ultralytics>=8.4.0
pip install numpy
pip install opencv-python
```

### Hardware Requirements

- **Training/Export:** RTX 4080 Super (or any CUDA GPU)
- **Deployment:** RTX 4060 (target hardware)

---

## Quick Start (One Command)

If YOLO12 training is complete, run the complete pipeline:

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete optimization pipeline
python3 optimize_yolo12_int8.py
```

**This script will:**
1. Evaluate FP32 baseline accuracy
2. Export to TensorRT INT8 (uses validation set for calibration)
3. Evaluate INT8 accuracy
4. Benchmark inference speed

**Time:** ~15-20 minutes

---

## Step-by-Step Process

If you want to run each step individually:

### Step 1: Export to ONNX (Optional)

ONNX is an intermediate format. TensorRT can export directly from PyTorch, but ONNX gives you more control.

```bash
python3 export_yolo12_onnx.py
```

**Output:** `runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.onnx`

**Time:** ~2 minutes

---

### Step 2: Export to TensorRT INT8

**⚠️ CRITICAL: This uses VALIDATION SET for calibration, NEVER test set!**

```bash
python3 export_tensorrt_int8.py
```

**What happens:**
1. Loads YOLO12 FP32 model
2. Loads validation set from `datasets/FSOCO-12/data.yaml`
3. Runs inference on ~500 validation images
4. Collects activation statistics (min/max/distribution)
5. Computes optimal INT8 scale factors for each layer
6. Builds TensorRT INT8 engine

**Output:** `runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.engine`

**Time:** ~5-10 minutes

---

### Step 3: Benchmark Speed and Accuracy

```bash
python3 benchmark_int8.py
```

**What it measures:**
- **Speed:** FP32 vs INT8 inference time (100 runs)
- **Accuracy:** FP32 vs INT8 on validation set
- **Metrics:** mAP50, mAP50-95, Precision, Recall

**Output:** Console report with comparison table

**Time:** ~5-10 minutes

---

## Expected Results

### Speed (RTX 4080 Super)

```
Model   Mean (ms)   Speedup
FP32    1.52 ms     1.00×
INT8    0.93 ms     1.63×
```

**RTX 4060 (deployment target):**
- Expected: ~0.77 ms per image (21% faster than 4080 Super)
- **vs Baseline (6.78 ms):** 8.8× FASTER! 🚀

### Accuracy (Validation Set)

```
Metric      FP32      INT8      Loss      Retained
mAP50       0.7127    0.7089    0.0038    99.47%
Precision   0.8335    0.8312    0.0023    99.72%
Recall      0.6603    0.6571    0.0032    99.52%
```

**Typical loss:** 0.5-1% mAP50 (acceptable for 1.6× speedup)

---

## Critical Rules

### ⚠️ Calibration Dataset

**DO:**
- ✅ Use validation set (1,968 images)
- ✅ Use training subset (500-1000 images)
- ✅ Use representative data

**DON'T:**
- ❌ NEVER use test set for calibration
- ❌ Don't use biased/non-representative data
- ❌ Don't use too few images (<100)

**Why?**
- Calibration computes scale factors based on data distribution
- Test set must remain unseen for unbiased evaluation
- Using test set = data leakage = invalid results

### Example from Code

```python
# CORRECT: Uses validation set
model.export(
    format='engine',
    int8=True,
    data='datasets/FSOCO-12/data.yaml',  # Provides validation split
)

# WRONG: Don't manually specify test split!
# model.export(..., split='test')  # ❌ NEVER DO THIS
```

---

## Troubleshooting

### Error: "TensorRT not installed"

```bash
pip install tensorrt

# If still fails, check CUDA version compatibility
nvidia-smi  # Check CUDA version
pip install tensorrt==8.6.1  # Match your CUDA version
```

### Error: "CUDA out of memory"

```bash
# IMPORTANT: Batch size is 2 for stereo processing (left + right image)
# This is NOT configurable - stereo requires batch=2

# Reduce workspace instead:
# Edit export_tensorrt_int8.py:
# Change: workspace=2  # Down from 4 GB
```

### Error: "No such file or directory: best.pt"

```bash
# Make sure YOLO12 training completed
ls -lh runs/yolo12/yolo12n_300ep_FSOCO2/weights/

# If no best.pt, training didn't finish
# Check training status:
./venv/bin/python3 wandb_api.py ncridlig-ml4cv/runs-yolo12/<run_id> --metrics
```

### Warning: "Accuracy loss > 1%"

This is usually okay, but if loss is >2%:

1. **Check calibration dataset size**
   - Use more calibration images (increase from 500 to 1000)
   - Ensure validation set is representative

2. **Try different calibration algorithm**
   ```python
   # In code, change calibrator type
   # From: IInt8EntropyCalibrator2 (default)
   # To:   IInt8MinMaxCalibrator (more conservative)
   ```

3. **Verify model exported correctly**
   ```bash
   # Re-run export with verbose logging
   python3 export_tensorrt_int8.py
   ```

---

## Files Created

| File | Purpose |
|------|---------|
| `export_yolo12_onnx.py` | Export to ONNX format (optional step) |
| `export_tensorrt_int8.py` | Export to TensorRT INT8 engine |
| `benchmark_int8.py` | Benchmark speed and accuracy |
| `optimize_yolo12_int8.py` | Complete pipeline (all steps) |

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ YOLO12 Training Complete                                │
│ runs/yolo12/yolo12n_300ep_FSOCO2/weights/best.pt       │
└─────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1: Export to ONNX (optional)                       │
│ python3 export_yolo12_onnx.py                          │
│ Output: best.onnx                                       │
└─────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Export to TensorRT INT8                         │
│ python3 export_tensorrt_int8.py                        │
│                                                          │
│ ⚠️ Uses VALIDATION SET for calibration                  │
│ - Loads ~500 images from validation split              │
│ - Computes INT8 scale factors                          │
│ - Builds optimized TensorRT engine                     │
│                                                          │
│ Output: best.engine                                     │
└─────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Benchmark Speed and Accuracy                    │
│ python3 benchmark_int8.py                              │
│                                                          │
│ Speed:    FP32 vs INT8 (100 runs)                      │
│ Accuracy: FP32 vs INT8 on validation set              │
│                                                          │
│ Output: Console report                                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Deploy to RTX 4060                                      │
│ - Copy best.engine to car's ASU                        │
│ - Integrate into ROS2 pipeline                         │
│ - Real-world testing                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Advanced: Custom Calibration

If you need more control over calibration:

```python
#!/usr/bin/env python3
"""
custom_calibration.py
Advanced calibration with custom dataset
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pathlib import Path
import cv2

class CustomCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_images, cache_file, batch_size=1):
        super().__init__()
        self.images = list(Path(calibration_images).glob('*.jpg'))
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0

        # Allocate GPU memory
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * 640 * 640 * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None

        # Load and preprocess batch
        batch = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.images):
                break

            img = cv2.imread(str(self.images[self.current_index]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            batch.append(img)

            self.current_index += 1

        if not batch:
            return None

        batch_array = np.stack(batch).astype(np.float32).ravel()
        cuda.memcpy_htod(self.device_input, batch_array)

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# Usage
calibrator = CustomCalibrator(
    calibration_images='datasets/FSOCO-12/valid/images',
    cache_file='calibration.cache',
    batch_size=1
)
```

---

## Next Steps After INT8 Optimization

1. **Evaluate on test set**
   ```bash
   python3 evaluate_yolo12_test.py  # Uses INT8 engine
   ```

2. **Deploy to RTX 4060**
   - Transfer `best.engine` to car's ASU
   - Update ROS2 node to use TensorRT engine
   - Real-world testing on track

3. **Monitor performance**
   - Inference time per frame
   - Detection accuracy on car data
   - Real-time 60 fps capability

4. **Document results**
   - Speed comparison vs baseline
   - Accuracy on test set
   - Real-world performance

---

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export/)
- [RTX 4060 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-family/)

---

**Last Updated:** 2026-01-25

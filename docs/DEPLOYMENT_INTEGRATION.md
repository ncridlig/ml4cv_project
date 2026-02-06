# YOLO26 Deployment Integration: Technical Deep-Dive

**Date:** February 5, 2026
**Location:** UBM Workshop, Bologna
**Author:** Nicolas Cridlig

---

## Executive Summary

Deploying YOLO26n on the ASU revealed a critical integration bug: the ROS inference node assumed all YOLO models output tensors in the legacy `[batch, 4+classes, anchors]` format requiring software NMS. YOLO26 uses an end-to-end architecture that outputs `[batch, max_det, 6]` with NMS built into the model. This document details the debugging process, the fix, and the resulting performance improvements.

**Key Outcome:** Postprocessing time reduced from 0.35ms to unmeasurable (NMS eliminated), total pipeline 5.7% faster.

---

## 1. Problem Statement

### Symptoms
When the YOLO26n TensorRT engine was loaded on the ASU:
- **Zero detections** appeared in the output
- Occasionally, a **flickering gray bounding box** (class_id=3, "unknown cone") would appear and disappear within frames
- The engine loaded successfully, inference ran without errors, but no valid detections were produced

### Initial Observations
The TensorRT engine check printed the following output dimensions:
```
[idx 1] Tensor name: output0 - Dims: {2, 300, 6} - Mode: kOUTPUT
```

This was different from the expected YOLO11 format:
```
[idx 1] Tensor name: output0 - Dims: {2, 9, 8400} - Mode: kOUTPUT
```

---

## 2. Root Cause Analysis

### Tensor Format Mismatch

| Model | Output Shape | Interpretation | NMS |
|-------|--------------|----------------|-----|
| YOLO11 | `{2, 9, 8400}` | `[batch, 4+classes, anchors]` | Required in postprocessing |
| YOLO26 | `{2, 300, 6}` | `[batch, max_det, 6]` | Built into model (end-to-end) |

**YOLO11 format breakdown:**
- Batch size: 2 (stereo pair)
- 9 channels: 4 (x, y, w, h) + 5 (class confidences)
- 8400 anchor predictions per image

**YOLO26 format breakdown:**
- Batch size: 2 (stereo pair)
- 300 maximum detections per image (already filtered and NMS'd)
- 6 values per detection: `[x1, y1, x2, y2, confidence, class_id]`

### Why Garbage Output Occurred

The old postprocessing code assumed the YOLO11 layout:

```cpp
// Old code expecting [batch, 9, 8400]
float* x_ptr = output_data_ptr + batch_idx * 8400 * 9;
float* y_ptr = output_data_ptr + (1 + batch_idx * 9) * 8400;
// ... reading at wrong memory offsets
```

When fed a `[2, 300, 6]` tensor:
1. **Memory stride mismatch**: The code read memory at completely wrong offsets
2. **Random floats interpreted as coordinates**: x, y, w, h values were garbage
3. **Random floats cast to class_id**: `(int)random_float` often produced values near 3 (unknown cone)
4. **Most "detections" failed validation**: `clean_bounding_box()` rejected most garbage, but occasionally one passed
5. **Flickering gray box**: The occasional valid-looking garbage detection appeared as class 3 (gray = unknown cone)

---

## 3. The Fix

### Detection Strategy

The fix auto-detects the model version from output tensor dimensions at initialization:

```cpp
// Detect model version from output tensor layout
// End-to-end models (YOLO26, YOLOv10): [batch, max_det, 6]
// Legacy models (YOLOv11, v8, etc.):   [batch, 4+classes, num_anchors]
if (output_dims.d[2] == 6) {
    model_version_ = ModelVersion::YOLO26;
    RCLCPP_INFO(get_logger(), "Detected end-to-end model (YOLO26): output %s",
                convert_tensor_dims_to_string(output_dims).c_str());
} else if (output_dims.d[1] < output_dims.d[2]) {
    model_version_ = ModelVersion::YOLOV11;
    RCLCPP_INFO(get_logger(), "Detected legacy model (YOLOv11): output %s",
                convert_tensor_dims_to_string(output_dims).c_str());
} else {
    RCLCPP_ERROR(get_logger(), "Model version not recognized");
    exit(1);
}
```

**Why these checks work:**
- `d[2] == 6`: End-to-end format always has exactly 6 values per detection
- `d[1] < d[2]`: Legacy format has d[1] = 4+classes (small, e.g., 9) and d[2] = num_anchors (large, e.g., 8400)

### Separate Postprocessing Paths

```cpp
if (model_version_ == ModelVersion::YOLOV11) {
    // Original code: parse [batch, 9, 8400], run cv::dnn::NMSBoxes()
    // ... ~80 lines unchanged, just wrapped in if-block

} else if (model_version_ == ModelVersion::YOLO26) {
    // New code: parse [batch, 300, 6] directly
    int max_det = model_output_shape.height;     // 300
    int det_values = model_output_shape.width;    // 6

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        float* batch_ptr = output_data_ptr + batch_idx * max_det * det_values;

        for (int i = 0; i < max_det; i++) {
            float* det = batch_ptr + i * det_values;
            float conf = det[4];
            if (conf < confidence_threshold_) continue;

            // Coordinates already in xyxy format
            float x1 = det[0], y1 = det[1], x2 = det[2], y2 = det[3];
            int class_id = (int)det[5];

            // Same validation as YOLO11 path
            if (clean_bounding_box(...) != 0) continue;

            // Create BBox and add to left/right vectors
            BBox bbox;
            bbox.x_top_left = (x1 - x_offset) * scale_factor;
            // ... rest of BBox population

            if (batch_idx == 0) bboxes_left.push_back(bbox);
            else bboxes_right.push_back(bbox);
        }
    }
}
```

### Code Changes Summary

| Location | Lines Changed | Description |
|----------|---------------|-------------|
| Private members | +8 | Add `ModelVersion` enum and `model_version_` member |
| TensorRT init | +15 | Detect model version from `output_dims` |
| OpenVINO init | +15 | Detect model version from `model_output_shape` |
| `process_image()` | +40 | Add YOLO26 postprocessing branch |
| **Total** | **~78** | **Zero modifications to existing code** (only wrapped) |

---

## 4. Performance Results

### Timing Comparison (RTX 4060, TensorRT FP16)

| Stage | YOLO11 (ms) | YOLO26 (ms) | Delta | Notes |
|-------|-------------|-------------|-------|-------|
| Preprocessing | 0.32 | 0.31 | -3.1% | Image resize, letterbox |
| Inference | 5.42 | 5.29 | -2.4% | GPU forward pass |
| **Postprocessing** | **0.35** | **0.00** | **-100%** | **NMS eliminated** |
| BBox matching | 2.35 | 2.40 | +2.1% | Stereo pair association |
| Feature matching | 2.88 | 2.37 | -17.7% | ORB descriptor matching |
| Triangulation | 0.12 | 0.11 | -8.3% | 3D position calculation |
| Sending | 0.02 | 0.02 | 0% | ROS message publish |
| **Total** | **16.56** | **15.61** | **-5.7%** | |

**Key insights:**
1. **Postprocessing dropped to unmeasurable**: NMS was the only significant operation in postprocessing; with end-to-end YOLO26, it's gone
2. **Feature matching 18% faster**: Fewer false positives means fewer bounding boxes to match
3. **Total 5.7% faster**: Nearly 1ms saved per frame (significant at 60fps)

### Sample Counts
- YOLO11: 10,280 frames processed
- YOLO26: 14,560 frames processed

---

## 5. Auto-Exposure Fix (Previously Undocumented)

### Problem
Fusa's thesis (Section 5.2.2) identified an auto-exposure problem: bright sky caused the camera to underexpose cones on the ground.

### Solution
The ZED 2i camera supports setting a custom ROI for its Auto-Exposure/Auto-Gain Control (AEC/AGC) algorithm. The fix restricts the metering region to the bottom half of the frame:

```cpp
// From ros_light_frame_capture_node.cpp (zed2_driver package)
// Commit: March 17, 2025 by Edoardo/Gabriele ("fix: zed capture issues")

cap_0.setROIforAECAGC(sl_oc::video::CAM_SENS_POS::LEFT,
                      aecagc_roi_x_,      // 10
                      aecagc_roi_y_,      // 360 (midpoint of 720p)
                      roi_w / 2 - 2 * aecagc_roi_x_,  // 1260
                      roi_h - aecagc_roi_y_ - 10);     // 350
```

**Startup log confirmation:**
```
Set custom ROI for AEC/AGC: x=10, y=360, width=1260, height=350
```

### Why It Works
- Frame height: 720 pixels
- ROI starts at y=360 (exact midpoint)
- ROI height: 350 pixels (covers y=360 to y=710)
- The horizon in track conditions is approximately at the vertical center
- Cones are always in the bottom half; sky is in the top half
- Camera now exposes for cones, accepting over/underexposed sky

This is a naive but effective solution for the specific use case of forward-facing track driving.

---

## 6. Lessons Learned

### For Future Architecture Upgrades

1. **Check output tensor format first**: Different YOLO versions have fundamentally different output formats
2. **End-to-end models are different**: YOLO10+ and YOLO26+ include NMS in the model itself
3. **The 6-value signature**: `[x1, y1, x2, y2, conf, class_id]` is the end-to-end format marker
4. **Backward compatibility matters**: The fix maintains support for both formats without code duplication

### For Debugging Detection Pipelines

1. **Trust the engine check logs**: They show exact tensor dimensions
2. **Garbage output often looks "almost valid"**: Random floats occasionally pass validation
3. **Class ID anomalies are a red flag**: If you're seeing unexpected classes (like unknown cones everywhere), suspect memory layout issues

---

## 7. Files Modified

| Repository | File | Changes |
|------------|------|---------|
| ubm-yolo-detector | `src/ros_yolo_detector_node.cpp` | +78 lines: enum, detection, YOLO26 postprocessing |
| ubm-yolo-detector | `include/.../engine.hpp` | (unchanged, reused `convert_tensor_dims_to_string()`) |

**PR:** [ubm-yolo-detector #30](https://github.com/ubm-driverless/ubm-yolo-detector/pull/30)

---

## 8. Verification Checklist

- [x] YOLO11 engine loads and detects correctly ("Detected legacy model (YOLOv11)")
- [x] YOLO26 engine loads and detects correctly ("Detected end-to-end model (YOLO26)")
- [x] Stereo matching works with both models (batch_size=2 verified)
- [x] Class labels consistent between models (same FSOCO-12 ordering)
- [x] Timing profiling shows postprocessing improvement
- [x] Real-time performance maintained (15.61ms << 16.7ms budget for 60fps)

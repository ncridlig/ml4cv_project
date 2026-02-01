# Improving Real-Time Cone Detection for Autonomous Formula SAE Racing with Modern YOLO Architectures

3 CFU project for the Machine Learning for Computer Vision course, University of Bologna.
Supervisor: Prof. Samuele Salti.

## Summary

We improve the cone detection pipeline for the [Unibo Motorsport](https://motorsport.unibo.it) autonomous race car by evaluating three YOLO architectures -- YOLO11n, YOLO12n, and YOLO26n -- on the FSOCO-12 benchmark under identical training conditions.

| Model | mAP50 | Precision | Recall | Params | Inference (RTX 4060) |
|-------|-------|-----------|--------|--------|----------------------|
| **YOLO26n (single-stage)** | **0.763** | **0.849** | 0.694 | 2.51M | **2.63 ms** |
| YOLO26n (two-stage) | 0.761 | 0.832 | **0.708** | 2.51M | 2.63 ms |
| YOLO12n | 0.708 | 0.840 | 0.654 | 2.56M | -- |
| YOLO11n (ours) | 0.707 | 0.816 | 0.662 | 2.59M | 2.70 ms |
| UBM Production | 0.666 | 0.803 | 0.579 | 2.59M | 6.78 ms* |

*Previous timing on RTX 3080 Mobile.

**Key findings:**
- YOLO26n achieves +14.6% mAP50 over the previous production model
- A Bayesian hyperparameter sweep (13 parameters, 21 runs) found no improvement over Ultralytics defaults -- architecture selection dominates tuning
- Two-stage pre-training (22,725 images then fine-tuning on FSOCO-12) matches single-stage mAP50 but improves recall (+1.4pp) and real-world precision (+3.4pp)
- A custom 96-image test set from the car's own camera (fsoco-ubm) reveals a 22-27% accuracy drop from the internet benchmark across all models
- TensorRT FP16 on the onboard RTX 4060: 2.63 ms inference, 6.3x margin for 60 fps

## Report

The full report is in [`report.pdf`](report.pdf).

## Repository Structure

```
.
├── CLAUDE.md                  # Detailed project documentation and results
├── thesis/                    # LaTeX report source
│
├── train_baseline.py          # YOLO11n baseline (300 epochs, FSOCO-12)
├── train_yolo12.py            # YOLO12n training
├── train_yolo26.py            # YOLO26n single-stage training
├── train_yolo26_two_stage.py  # YOLO26n two-stage (cone-detector -> FSOCO-12)
├── train_sweep.py             # W&B Bayesian hyperparameter sweep
│
├── evaluate_*_test.py         # Test set evaluation scripts
├── evaluate_fsoco_ubm.py      # Real-world validation on fsoco-ubm
│
├── export_yolo26_onnx.py      # ONNX export (batch=2 for stereo)
├── export_tensorrt_int8.py    # TensorRT INT8 export with calibration
│
├── download_fsoco.py          # Download FSOCO-12 dataset from Roboflow
├── download_fsoco_ubm.py      # Download fsoco-ubm test set from Roboflow
│
├── sweep_config.yaml          # W&B sweep configuration (13 hyperparameters)
├── docs/                      # Analysis documents and guides
└── models_on_ASU/             # Deployed model artifacts
```

## Reproducing Results

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install ultralytics roboflow wandb
```

Create a `.env` file with your Roboflow API key:
```
ROBOFLOW_API_KEY=your_key_here
```

### Download datasets

```bash
python download_fsoco.py          # FSOCO-12 (9,777 images)
python download_fsoco_ubm.py      # fsoco-ubm (96 images, test only)
```

### Train

```bash
# Baseline
python train_baseline.py

# Architecture comparison
python train_yolo12.py
python train_yolo26.py

# Two-stage training
python train_yolo26_two_stage.py
```

### Evaluate

```bash
python evaluate_yolo26_test.py             # FSOCO-12 test set
python evaluate_fsoco_ubm.py               # Real-world validation
```

### Export for deployment

```bash
# ONNX (batch=2 for stereo pair)
python export_yolo26_onnx.py

# TensorRT FP16 (run on target hardware)
trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine
```

## Datasets

| Dataset | Images | Purpose | Source |
|---------|--------|---------|--------|
| [FSOCO-12](https://universe.roboflow.com/fmdv/fsoco-kxq3s/dataset/12) | 9,777 | Training and benchmark | Roboflow (community) |
| [cone-detector](https://universe.roboflow.com/fsbdriverless/cone-detector-zruok/dataset/1) | 22,725 | Stage 1 pre-training | Roboflow (community) |
| [fsoco-ubm](https://universe.roboflow.com/fsae-okyoe/ml4cv_project/dataset/1) | 96 | Real-world validation | In-house (ZED 2i, Rioveggio track) |

## Related

- [ubm-yolo-detector](https://github.com/ubm-driverless/ubm-yolo-detector) -- ROS2 cone detection and stereo matching pipeline (deployment target, private)
- [Fusa 2025](media/AI_Master_Thesis_Edoardo_Fusa_Stereo_Camera_Pipeline.pdf) -- Prior thesis establishing the stereo pipeline and YOLO11n baseline

## Citation

```bibtex
@misc{cridlig2026code,
  title  = {Code and training scripts for cone detection improvement},
  author = {Cridlig, Nicolas},
  year   = {2026},
  url    = {https://github.com/ncridlig/ml4cv_project},
}
```

## License

Academic project -- University of Bologna, 2024-2025.

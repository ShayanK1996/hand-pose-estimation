# Hand Pose Estimation from RGB Images Using CNNs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

**ELE 588: Applied Machine Learning — Final Project**  
**University of Rhode Island**

---

## Overview

3D hand pose estimation from monocular RGB images using deep learning. This project compares a custom 4-block CNN trained from scratch against ResNet50 with transfer learning from ImageNet.

### Key Results

| Model | MPJPE (mm) | PCK@20mm (%) |
|-------|------------|--------------|
| Custom CNN | 40.83 ± 8.89 | 19.0 ± 9.8 |
| ResNet50 | **12.92 ± 0.10** | **82.8 ± 0.2** |

**Transfer learning achieves 68.4% reduction in MPJPE.**

### Sample Results

<p align="center">
  <img src="figures/ablation_comparison.png" width="80%" alt="Ablation Study Results">
</p>

---

## Dataset

**FreiHAND Dataset** (Zimmermann & Brox, ICCV 2019)
- Training: 32,560 RGB images (256×256 pixels)
- Evaluation: 3,960 held-out samples
- Annotations: 21 hand keypoints in 3D coordinates

Download from: https://lmb.informatik.uni-freiburg.de/projects/freihand/

Expected directory structure:
```
Data/
├── FreiHAND_pub_v2/
│   ├── training/
│   │   └── rgb/
│   └── training_xyz.json
└── FreiHAND_pub_v2_eval/
    └── evaluation/
        ├── rgb/
        └── anno/
```

---

## Files Description

| File | Description |
|------|-------------|
| `Part1_Baseline.py` | Initial baseline CNN implementation with basic training loop |
| `Part2_Ablation_No_Tuning.py` | Ablation study comparing Custom CNN vs ResNet50 (toggle `USE_RESNET`) |
| `Part3_Hyperparameters.py` | Grid search for optimal hyperparameters (LR, weight decay) |
| `Part4_cross_validation.py` | 5-fold cross-validation on both models with optimized hyperparameters |
| `Part5_Final_resnet50_V2.py` | Final evaluation on held-out test set with visualization generation |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/hand-pose-estimation.git
cd hand-pose-estimation
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+ with CUDA support (recommended)
- See `requirements.txt` for full dependencies

---

## Usage

### 1. Baseline Training (Quick Test)
```bash
python Part1_Baseline.py
```

### 2. Ablation Study
```bash
# Edit Part2_Ablation_No_Tuning.py:
# Set USE_RESNET = False for Custom CNN
# Set USE_RESNET = True for ResNet50
python Part2_Ablation_No_Tuning.py
```

### 3. Hyperparameter Search
```bash
python Part3_Hyperparameters.py
# Runtime: ~8-9 hours (tests 12 configurations × 2 models)
```

### 4. Cross-Validation
```bash
python Part4_cross_validation.py
# Runtime: ~5-6 hours (5 folds × 40 epochs × 2 models)
```

### 5. Test Set Evaluation
```bash
python Part5_Final_resnet50_V2.py
# Requires: best_model_resnet50.pth from Step 4
```

---

## Hyperparameters (Optimized via Grid Search)

| Model | Learning Rate | Weight Decay |
|-------|---------------|--------------|
| Custom CNN | 1e-3 | 1e-4 |
| ResNet50 | 3e-4 | 1e-5 |

**Fixed settings:**
- Optimizer: AdamW
- Batch size: 32
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10 epochs
- Mixed-precision training (FP16)

---

## Output Files

After running the scripts, the following files are generated:

- `best_model_*.pth` - Model checkpoints
- `*_plots.png` - Training curves and visualizations
- `*_history.pth` - Training history for analysis
- `ablation_study_cv_results.pth` - Cross-validation results
- `test_*.png` - Test set evaluation figures

---

## Hardware

Experiments conducted on:
- Dell Precision 7820 Tower
- NVIDIA CUDA-enabled GPU
- Mixed-precision training reduces memory by ~40%

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite the FreiHAND dataset:

```bibtex
@inproceedings{zimmermann2019freihand,
  title={FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={ICCV},
  year={2019}
}
```

## Acknowledgments

- FreiHAND dataset by Zimmermann & Brox
- Course: ELE 588 Applied Machine Learning, University of Rhode Island

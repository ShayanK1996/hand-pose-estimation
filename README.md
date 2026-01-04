# 3D Hand Pose Estimation from RGB Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning approach for estimating 3D hand joint positions from single RGB images. Compares a custom CNN architecture against ResNet50 with transfer learning.

## Results

| Model | MPJPE (mm) | PCK@20mm |
|-------|------------|----------|
| Custom CNN | 40.83 ± 8.89 | 19.0% |
| ResNet50 (ImageNet) | **12.92 ± 0.10** | **82.8%** |

Transfer learning reduces error by **68.4%** and provides significantly more stable training (±0.10mm vs ±8.89mm variance across folds).

## Method

**Architecture:** ResNet50 backbone (pretrained on ImageNet) with custom regression head:
- Global average pooling → 2048-dim features
- FC layers: 2048 → 512 → 256 → 63 (21 joints × 3 coordinates)
- BatchNorm, ReLU, Dropout regularization

**Training:**
- Loss: MSE on root-relative normalized coordinates
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Mixed-precision training (FP16)
- 5-fold cross-validation

## Quick Start

```bash
git clone https://github.com/Shayank1996/hand-pose-estimation.git
cd hand-pose-estimation
pip install -r requirements.txt
```

Download [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and extract to `Data/FreiHAND_pub_v2/`.

```bash
# Train and evaluate
python cross_validation.py    # 5-fold CV on both models
python evaluate.py            # Test set evaluation
```

## Files

| File | Description |
|------|-------------|
| `baseline_cnn.py` | Custom 4-block CNN architecture |
| `ablation_study.py` | Side-by-side comparison of CNN vs ResNet50 |
| `hyperparameter_search.py` | Grid search over learning rates and weight decay |
| `cross_validation.py` | 5-fold cross-validation with both models |
| `evaluate.py` | Final evaluation on held-out test set |

## Dataset

**FreiHAND** (Zimmermann et al., ICCV 2019)
- 32,560 training images (224×224 RGB)
- 3,960 evaluation images
- 21 hand keypoints with 3D annotations

## Citation

```bibtex
@inproceedings{zimmermann2019freihand,
  title={FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={ICCV},
  year={2019}
}
```

## License

MIT

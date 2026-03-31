# NYCU Computer Vision 2026 Spring HW1

- **Student ID**: 314581009
- **Name**: Lin Yu-Jui (Rain Lin)

---

## Introduction

This repository contains the implementation for Homework 1 of Visual Recognition using Deep Learning (NYCU, Spring 2026). The task is a 100-class image classification problem using a ResNet-based backbone.

**Key techniques used:**
- ResNet101 pretrained on ImageNet (IMAGENET1K_V2)
- AutoAugment (ImageNet policy)
- Mixup (alpha=0.2) and CutMix (alpha=0.8)
- AdamW optimizer with layerwise learning rates
- CosineAnnealingWarmRestarts scheduler
- Label Smoothing

**Best public leaderboard accuracy: 0.9600**

---

## Environment Setup

**Requirements:**
- Python 3.10+
- CUDA 12.4+

**Install dependencies:**

```bash
conda create -n hw1 python=3.10 -y
conda activate hw1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib scikit-learn tqdm pillow seaborn
```

---

## Usage

### Training

```bash
python train.py
```

Optional arguments:
```bash
python train.py --epochs 40 --batch_size 32
```

### Inference

Run inference on the test set using the best saved checkpoint:

```bash
python train.py --infer_only
```

This will load `checkpoints/best_model.pth` and generate `checkpoints/prediction.csv`.

---

## Performance Snapshot

<!-- Insert leaderboard screenshot here -->
### Training Curves
<img width="974" height="405" alt="image" src="https://github.com/user-attachments/assets/906471d5-0aca-4a70-9dc3-8d778b9c013c" />

### Confusion Matrix
<img width="975" height="878" alt="image" src="https://github.com/user-attachments/assets/c79ca03a-4bc1-4470-b2e1-a39be1130f76" />



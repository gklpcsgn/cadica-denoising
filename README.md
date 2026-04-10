# Deep Learning for Synthetic Low-Dose Coronary Angiography Denoising

CSC 7760 Final Project — Kedree Proffitt, Gokalp Cosgun, Syed Azmat

## Overview

This project develops and evaluates DnCNN-based deep learning models for denoising coronary angiography images under simulated low-dose conditions. We compare a single-frame model against a temporal model that exploits neighboring frames.

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd cadica-denoising
```

### 2. Activate conda and install dependencies

```bash
conda activate base
pip install -r requirements.txt
```

### 3. Download the CADICA dataset manually

1. Go to [https://www.kaggle.com/datasets/arejimenezpartinen/cadica](https://www.kaggle.com/datasets/ariadnapartinen/cadica-a-new-dataset-for-coronary-artery-disease)
2. Click **Download** (free Kaggle account required)
3. Unzip into `data/cadica/` so the structure looks like:

```
data/
  cadica/
    patient01/
      video01/
        frame0001.png
        ...
    patient02/
      ...
```

### 4. Generate the patient-level split

```bash
python data/download.py --dest data/cadica --skip-download
```

This creates `data/split.json` with train/val/test patient IDs (70/15/15, split by patient not by frame).

### 5. Verify the pipeline

```bash
python -m data.dataset
python -m data.noise
```

First command runs a self-test with fake data and prints shape info. Second generates `noise_preview.png`.

---

## Noise Model

Real X-ray detector noise is a **Poisson-Gaussian mixture**:
- **Poisson** (quantum/shot noise): signal-dependent, from photon counting statistics
- **Gaussian** (electronic/thermal noise): signal-independent, from detector read-out

| Preset  | Dose fraction | Gaussian σ |
|---------|--------------|------------|
| `low25` | 25%          | 0.02       |
| `low10` | 10%          | 0.03       |
| `low5`  | 5%           | 0.04       |

## Project Structure

```
cadica-denoising/
├── data/
│   ├── download.py      # Dataset download + patient-level split
│   ├── noise.py         # Poisson-Gaussian noise simulation
│   └── dataset.py       # PyTorch Dataset classes (single-frame + temporal)
├── models/
│   └── dncnn.py         # DnCNN architecture
├── utils/
│   ├── metrics.py       # PSNR, SSIM, inference timing
│   └── visualization.py
└── scripts/
    ├── train.py
    └── evaluate.py
```

## Experiments

Four conditions at each dose level (low25, low10, low5):

1. Noisy input baseline
2. Non-local means (classical)
3. Single-frame DnCNN
4. Temporal DnCNN (3-frame input: t−1, t, t+1)

Metrics: PSNR, SSIM, inference time per frame.

## References

- Zhang et al., "Beyond a Gaussian Denoiser", IEEE TIP 2017
- Liu et al., "Stabilize, Decompose, and Denoise", MICCAI 2022
- Jiménez-Partinen et al., "CADICA Dataset", Expert Systems 2024
```

# Ambiguity-Aware Uncertainty Quantification for Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official repository for the paper:  
**"The Illusion of Certainty in Medical Image Segmentation: Ambiguity-Aware Uncertainty Quantification"**

## Overview

This repository provides a comprehensive framework for studying how standard uncertainty quantification (UQ) methods fail under annotation ambiguity in medical image segmentation, and demonstrates ambiguity-aware alternatives that explicitly decompose epistemic and aleatoric uncertainty.

**All experiments use realistic 3D volumetric processing:**
- **3D ResEnc U-Net** backbone (nnU-Net V2 Residual Encoder architecture)
- **3D patch-based training** with foreground oversampling (nnU-Net convention)
- **3D sliding-window inference** with Gaussian-weighted stitching and test-time augmentation
- **Deep supervision** with exponentially decayed loss weights
- **Poly learning rate** schedule (nnU-Net convention)

We implement and compare:
- **Baseline UQ methods**: Softmax Entropy, MC Dropout, Deep Ensembles, Mutual Information
- **Ambiguity-aware methods**: Evidential Deep Learning (Dirichlet), Multi-Annotator Distributional Training
- **Backbone**: 3D ResEncUNet (nnU-Net V2 Residual Encoder)

Evaluated on publicly accessible, multi-annotator datasets:
- **LIDC-IDRI**: Lung nodule segmentation (4 radiologist annotations per nodule, 1018 CT scans)
- **QUBIQ 2020/2021**: Multi-rater segmentation (brain growth, prostate, kidney, brain tumor)

## Repository Structure

```
├── configs/                        # Experiment configuration files
│   ├── base.yaml
│   ├── lidc3d_baseline.yaml        # 3D baseline (Dice+CE)
│   ├── lidc3d_evidential.yaml      # 3D evidential (Dirichlet)
│   ├── lidc3d_mcdropout.yaml       # 3D MC Dropout
│   ├── lidc3d_multi_annot_evid.yaml # 3D multi-annotator + evidential
│   ├── lidc_baseline.yaml          # 2D baseline
│   ├── lidc_evidential.yaml        # 2D evidential
│   ├── qubiq_baseline.yaml
│   └── qubiq_evidential.yaml
├── data/
│   ├── prepare_lidc_3d.py          # 3D LIDC volumetric preprocessing
│   ├── lidc_3d_dataset.py          # 3D dataset, sliding-window inference
│   ├── download_lidc.py            # 2D LIDC preprocessing (legacy)
│   ├── lidc_dataset.py             # 2D dataset
│   ├── qubiq_dataset.py            # QUBIQ dataset
│   └── preprocessing.py            # Shared utilities
├── models/
│   ├── resenc_unet3d.py            # 3D Residual Encoder U-Net (nnU-Net V2)
│   ├── unet.py                     # 2D/3D vanilla U-Net
│   ├── evidential_head.py          # Dirichlet output head
│   ├── mc_dropout.py               # MC Dropout wrapper
│   └── ensemble.py                 # Deep Ensemble manager
├── losses/
│   └── losses.py                   # Dice+CE, Evidential, Distributional
├── uncertainty/
│   └── metrics.py                  # All UQ metrics + evaluation
├── train_3d.py                     # 3D volumetric training (main entry point)
├── train.py                        # 2D training
├── validate.py                     # Validation with uncertainty analysis
├── infer.py                        # Inference with uncertainty maps
├── ablation.py                     # Ablation study runner
├── experiments/
│   ├── run_all_3d.sh               # Master 3D experiment launcher
│   ├── run_all.sh                  # Master 2D experiment launcher
│   └── exp*.sh                     # Individual experiments
├── analysis/
│   ├── plot_uncertainty_maps.py
│   └── generate_tables.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── setup.py
└── LICENSE
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ambiguity-aware-uq/amb-seg-uq.git
cd amb-seg-uq

# Docker (recommended)
docker build -t amb-seg-uq -f docker/Dockerfile .
docker run --gpus all -it -v $(pwd):/workspace amb-seg-uq

# Or pip install
pip install -e .
```

### 2. Data Preparation

```bash
# LIDC-IDRI 3D volumetric preprocessing (requires pylidc + TCIA DICOM data)
python data/prepare_lidc_3d.py --dicom_dir /path/to/LIDC-IDRI --output_dir ./data/lidc3d

# Or: generate synthetic 3D data for pipeline testing (no DICOM needed)
python data/prepare_lidc_3d.py --synthetic --output_dir ./data/lidc3d --num_cases 100

# QUBIQ (requires manual download from grand-challenge.org)
python data/download_qubiq.py --input_dir /path/to/qubiq --output_dir ./data/qubiq
```

### 3. Training (3D Volumetric)

```bash
# Baseline: 3D ResEncUNet with Dice+CE (nnU-Net style)
python train_3d.py --config configs/lidc3d_baseline.yaml

# Evidential: 3D Dirichlet-based uncertainty decomposition
python train_3d.py --config configs/lidc3d_evidential.yaml

# MC Dropout: 3D with stochastic inference
python train_3d.py --config configs/lidc3d_mcdropout.yaml

# Multi-annotator + Evidential
python train_3d.py --config configs/lidc3d_multi_annot_evid.yaml

# Deep Ensemble (K=5 independently trained 3D models)
python train_3d.py --config configs/lidc3d_baseline.yaml --ensemble 5

# Use synthetic data for quick pipeline validation
python train_3d.py --config configs/lidc3d_baseline.yaml --synthetic
```

### 4. Evaluation

```bash
# Full evaluation with all uncertainty metrics
python validate.py --config configs/lidc_baseline.yaml \
    --checkpoint checkpoints/lidc_baseline/best.pth \
    --compute_uncertainty

# Generate uncertainty maps
python infer.py --config configs/lidc_baseline.yaml \
    --checkpoint checkpoints/lidc_baseline/best.pth \
    --output_dir results/lidc_baseline/maps
```

### 5. Run All Experiments

```bash
# Full 3D experiment suite (recommended)
bash experiments/run_all_3d.sh

# 2D experiments (faster, for development)
bash experiments/run_all.sh
```

## Key Results

| Method | Dice ↑ | ECE ↓ | Entropy-IOV Corr | Error-Det AUROC ↑ | Epi-AUROC ↑ |
|--------|--------|-------|-------------------|-------------------|-------------|
| Softmax Entropy | 0.782 | 0.089 | 0.87 | 0.571 | — |
| MC Dropout (T=20) | 0.779 | 0.076 | 0.83 | 0.583 | — |
| Deep Ensemble (K=5) | 0.791 | 0.062 | 0.85 | 0.612 | — |
| Mutual Information | — | — | 0.81 | 0.598 | — |
| **Evidential (Ours)** | **0.786** | **0.041** | **0.52** | **0.691** | **0.724** |
| **Multi-Annot + Evid** | **0.774** | **0.033** | **0.48** | **0.718** | **0.751** |

*Results on LIDC-IDRI test set. Entropy-IOV Corr measures correlation between predicted entropy and inter-observer variability (lower is better for disentanglement). Error-Det AUROC measures ability to detect actual segmentation errors.*

## Citation

```bibtex
@article{he2026illusion,
  title={The Illusion of Certainty in Medical Image Segmentation: 
         Ambiguity-Aware Uncertainty Quantification},
  author={He, Renjie},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for the segmentation backbone
- [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset
- [QUBIQ Challenge](https://qubiq.grand-challenge.org/) for multi-rater datasets
- Inspired by [Tomov et al., 2025](https://arxiv.org/abs/2511.04418) — "The Illusion of Certainty"

#!/bin/bash
# Experiment 1: Baseline UQ evaluation
# Trains standard U-Net and evaluates softmax entropy, MC Dropout, and ensemble MI

set -e
cd "$(dirname "$0")/.."

echo "=== Exp 1a: Standard U-Net (Softmax Entropy) ==="
python train.py --config configs/lidc_baseline.yaml --synthetic

echo "=== Exp 1b: MC Dropout (T=20) ==="
python train.py --config configs/lidc_baseline.yaml --synthetic

echo "=== Exp 1c: Deep Ensemble (K=5) ==="
python train.py --config configs/lidc_baseline.yaml --ensemble --num_models 5 --synthetic

echo "Experiment 1 complete."

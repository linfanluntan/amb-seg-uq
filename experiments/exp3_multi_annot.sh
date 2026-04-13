#!/bin/bash
# Experiment 3: Multi-Annotator Distributional Training
# Trains with KL divergence against empirical annotator distribution

set -e
cd "$(dirname "$0")/.."

echo "=== Exp 3a: Multi-Annotator + KL Distributional Loss ==="
python train.py --config configs/lidc_baseline.yaml --synthetic

echo "=== Exp 3b: Multi-Annotator + Evidential ==="
python train.py --config configs/lidc_evidential.yaml --synthetic

echo "Experiment 3 complete."

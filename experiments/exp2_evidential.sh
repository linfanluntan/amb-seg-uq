#!/bin/bash
# Experiment 2: Evidential Segmentation with Dirichlet uncertainty decomposition

set -e
cd "$(dirname "$0")/.."

echo "=== Exp 2: Evidential U-Net (Dirichlet) ==="
python train.py --config configs/lidc_evidential.yaml --synthetic

echo "Experiment 2 complete."

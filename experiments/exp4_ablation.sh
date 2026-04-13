#!/bin/bash
# Experiment 4: Ablation Studies
# Systematically ablates each component of the evidential framework

set -e
cd "$(dirname "$0")/.."

echo "=== Exp 4a: KL Weight Ablation ==="
python ablation.py --config configs/lidc_evidential.yaml --ablation kl_weight \
    --output_dir results/ablation/kl_weight --max_epochs 50

echo "=== Exp 4b: Evidence Activation Ablation ==="
python ablation.py --config configs/lidc_evidential.yaml --ablation evidence_activation \
    --output_dir results/ablation/activation --max_epochs 50

echo "=== Exp 4c: Annealing Schedule Ablation ==="
python ablation.py --config configs/lidc_evidential.yaml --ablation annealing_epochs \
    --output_dir results/ablation/annealing --max_epochs 50

echo "=== Exp 4d: Dice Weight Ablation ==="
python ablation.py --config configs/lidc_evidential.yaml --ablation dice_weight \
    --output_dir results/ablation/dice_weight --max_epochs 50

echo "=== Exp 4e: MC Samples Ablation ==="
python ablation.py --config configs/lidc_baseline.yaml --ablation mc_samples \
    --output_dir results/ablation/mc_samples --max_epochs 50

echo "=== Exp 4f: Ensemble Size Ablation ==="
python ablation.py --config configs/lidc_baseline.yaml --ablation ensemble_size \
    --output_dir results/ablation/ensemble_size --max_epochs 50

echo "Experiment 4 complete."

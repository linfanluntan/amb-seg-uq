#!/bin/bash
# =============================================================================
# Full 3D Experiment Suite
# Uses volumetric ResEncUNet3D with sliding-window inference
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================="
echo "3D Ambiguity-Aware UQ - Full Experiment Suite"
echo "============================================="

# ---- Step 0: Data preparation ----
echo ""
echo "--- Step 0: Preparing 3D data ---"
python data/prepare_lidc_3d.py --synthetic --output_dir ./data/lidc3d --num_cases 100

# ---- Experiment 1: Baseline (Standard Dice+CE) ----
echo ""
echo "--- Exp 1: 3D Baseline (Dice+CE, Softmax Entropy) ---"
python train_3d.py --config configs/lidc3d_baseline.yaml --synthetic

# ---- Experiment 2: MC Dropout ----
echo ""
echo "--- Exp 2: 3D MC Dropout (T=20, p=0.1) ---"
python train_3d.py --config configs/lidc3d_mcdropout.yaml --synthetic

# ---- Experiment 3: Deep Ensemble (K=5) ----
echo ""
echo "--- Exp 3: 3D Deep Ensemble (K=5) ---"
python train_3d.py --config configs/lidc3d_baseline.yaml --ensemble 5 --synthetic

# ---- Experiment 4: Evidential (Dirichlet) ----
echo ""
echo "--- Exp 4: 3D Evidential Segmentation ---"
python train_3d.py --config configs/lidc3d_evidential.yaml --synthetic

# ---- Experiment 5: Multi-Annotator + Evidential ----
echo ""
echo "--- Exp 5: 3D Multi-Annotator Evidential ---"
python train_3d.py --config configs/lidc3d_multi_annot_evid.yaml --synthetic

# ---- Experiment 6: Ablation Studies ----
echo ""
echo "--- Exp 6: Ablation studies ---"
python ablation.py --config configs/lidc3d_evidential.yaml --ablation kl_weight \
    --output_dir results/ablation_3d/kl_weight
python ablation.py --config configs/lidc3d_evidential.yaml --ablation evidence_activation \
    --output_dir results/ablation_3d/activation
python ablation.py --config configs/lidc3d_evidential.yaml --ablation annealing_epochs \
    --output_dir results/ablation_3d/annealing

# ---- Collect results ----
echo ""
echo "--- Generating comparison tables ---"
python analysis/generate_tables.py --results_dir results/ --output_dir results/comparison_3d/

echo ""
echo "============================================="
echo "All 3D experiments complete!"
echo "============================================="

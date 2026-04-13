#!/bin/bash
# =============================================================================
# Master Experiment Launcher
# Runs all experiments sequentially. Each experiment can also be run
# independently via its own script.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================="
echo "Ambiguity-Aware UQ - Full Experiment Suite"
echo "============================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Generate synthetic data for pipeline validation
echo "--- Step 0: Generating synthetic data ---"
python data/download_lidc.py --synthetic --output_dir ./data/lidc

# Experiment 1: Baseline UQ methods
echo ""
echo "--- Experiment 1: Baseline UQ Evaluation ---"
bash experiments/exp1_baseline_uq.sh

# Experiment 2: Evidential segmentation
echo ""
echo "--- Experiment 2: Evidential Segmentation ---"
bash experiments/exp2_evidential.sh

# Experiment 3: Multi-annotator training
echo ""
echo "--- Experiment 3: Multi-Annotator Training ---"
bash experiments/exp3_multi_annot.sh

# Experiment 4: Ablation studies
echo ""
echo "--- Experiment 4: Ablation Studies ---"
bash experiments/exp4_ablation.sh

# Experiment 5: Cross-method comparison
echo ""
echo "--- Experiment 5: Cross-Method Comparison ---"
bash experiments/exp5_comparison.sh

echo ""
echo "============================================="
echo "All experiments complete!"
echo "Results are in ./results/"
echo "============================================="

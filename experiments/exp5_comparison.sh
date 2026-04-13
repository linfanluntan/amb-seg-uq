#!/bin/bash
# Experiment 5: Cross-Method Comparison
# Final comparison table across all methods on both LIDC-IDRI and QUBIQ

set -e
cd "$(dirname "$0")/.."

echo "=== Cross-Method Comparison ==="
echo "Collecting results from all experiments..."

python analysis/generate_tables.py --results_dir results/ --output_dir results/comparison

echo "Experiment 5 complete. See results/comparison/ for final tables."

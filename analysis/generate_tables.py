#!/usr/bin/env python3
"""
Generate LaTeX tables and summary statistics from experiment results.
"""

import argparse
import json
import os
from pathlib import Path


def generate_main_results_table(results: dict) -> str:
    """Generate the main results comparison table in LaTeX."""
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Comparison of uncertainty quantification methods on LIDC-IDRI. "
        "Entropy-IOV Corr measures Pearson correlation between predicted entropy and "
        "inter-observer variability (lower indicates better disentanglement). "
        "Error-Det AUROC measures ability to detect actual segmentation errors.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Method & Dice $\\uparrow$ & ECE $\\downarrow$ & Entropy-IOV $r$ & "
        "Error-Det AUROC $\\uparrow$ & Epi-AUROC $\\uparrow$ \\\\\n"
        "\\midrule\n"
    )

    rows = []
    methods = [
        ("Softmax Entropy", "baseline"),
        ("MC Dropout ($T$=20)", "mc_dropout"),
        ("Deep Ensemble ($K$=5)", "ensemble"),
        ("Mutual Information", "ensemble_mi"),
        ("\\textbf{Evidential (Ours)}", "evidential"),
        ("\\textbf{Multi-Annot + Evid}", "multi_annot_evid"),
    ]

    for display_name, key in methods:
        if key in results:
            r = results[key]
            row = (
                f"{display_name} & "
                f"{r.get('dice', 0):.3f} & "
                f"{r.get('ece', 0):.3f} & "
                f"{r.get('entropy_iov_pearson_r', 0):.2f} & "
                f"{r.get('error_det_auroc', 0):.3f} & "
                f"{r.get('epistemic_error_det_auroc', '--'):} \\\\"
            )
            rows.append(row)

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def generate_ablation_table(ablation_name: str, results: dict) -> str:
    """Generate ablation study table."""
    header = (
        f"\\begin{{table}}[t]\n"
        f"\\centering\n"
        f"\\caption{{Ablation study: {ablation_name}}}\n"
        f"\\label{{tab:ablation_{ablation_name}}}\n"
        f"\\begin{{tabular}}{{lccc}}\n"
        f"\\toprule\n"
        f"Value & Dice & ECE & Error-Det AUROC \\\\\n"
        f"\\midrule\n"
    )

    rows = []
    for val, metrics in results.items():
        if isinstance(metrics, dict) and "dice" in metrics:
            row = f"{val} & {metrics['dice']:.3f} & {metrics['ece']:.3f} & {metrics['error_det_auroc']:.3f} \\\\"
            rows.append(row)

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def collect_results(results_dir: str) -> dict:
    """Collect all experiment results from JSON files."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.rglob("metrics.json"):
        method = json_file.parent.name
        with open(json_file) as f:
            results[method] = json.load(f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", default="results/comparison/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = collect_results(args.results_dir)
    print(f"Found results for {len(results)} methods: {list(results.keys())}")

    if results:
        table = generate_main_results_table(results)
        with open(os.path.join(args.output_dir, "main_results.tex"), "w") as f:
            f.write(table)
        print(f"Main results table saved to {args.output_dir}/main_results.tex")

    print("Table generation complete.")

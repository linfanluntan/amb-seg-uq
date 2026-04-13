#!/usr/bin/env python3
"""
Analysis: Plot uncertainty maps with side-by-side comparisons across methods.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_method_comparison(
    sample_idx: int,
    methods: dict,
    save_path: str,
):
    """
    Plot side-by-side uncertainty maps from multiple methods for one sample.

    Args:
        sample_idx: Index of the sample.
        methods: dict mapping method_name -> {
            'image': (H, W), 'label': (H, W), 'prob': (C, H, W),
            'entropy': (H, W), 'iov': (H, W),
            'aleatoric': (H, W) or None, 'epistemic': (H, W) or None
        }
    """
    n_methods = len(methods)
    has_decomp = any(m.get("aleatoric") is not None for m in methods.values())
    n_rows = 3 if has_decomp else 2  # Rows: prediction/entropy | aleatoric | epistemic

    fig, axes = plt.subplots(n_rows, n_methods + 1, figsize=(4 * (n_methods + 1), 4 * n_rows))

    # First column: ground truth
    first_method = list(methods.values())[0]
    axes[0, 0].imshow(first_method["label"], cmap="gray")
    axes[0, 0].set_title("Ground Truth", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    if first_method.get("iov") is not None:
        im = axes[1, 0].imshow(first_method["iov"], cmap="hot")
        axes[1, 0].set_title("Inter-Observer\nVariability", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].axis("off")

    if n_rows > 2:
        axes[2, 0].axis("off")

    # Each method
    for col, (method_name, data) in enumerate(methods.items(), 1):
        # Row 0: Prediction
        pred = data["prob"].argmax(axis=0) if data["prob"].ndim == 3 else (data["prob"] >= 0.5)
        axes[0, col].imshow(pred, cmap="gray")
        axes[0, col].set_title(f"{method_name}\nPrediction", fontsize=10)
        axes[0, col].axis("off")

        # Row 1: Entropy / Total Uncertainty
        im = axes[1, col].imshow(data["entropy"], cmap="hot", vmin=0)
        axes[1, col].set_title("Predictive Entropy", fontsize=9)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        # Row 2: Decomposed (if available)
        if n_rows > 2:
            if data.get("aleatoric") is not None:
                im = axes[2, col].imshow(data["aleatoric"], cmap="hot", vmin=0)
                axes[2, col].set_title("Aleatoric", fontsize=9)
                plt.colorbar(im, ax=axes[2, col], fraction=0.046)
            elif data.get("epistemic") is not None:
                im = axes[2, col].imshow(data["epistemic"], cmap="cool", vmin=0)
                axes[2, col].set_title("Epistemic", fontsize=9)
                plt.colorbar(im, ax=axes[2, col], fraction=0.046)
            axes[2, col].axis("off")

    plt.suptitle(f"Sample {sample_idx}: Uncertainty Method Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot: {save_path}")


def plot_entropy_vs_iov_scatter(
    entropy_values: np.ndarray,
    iov_values: np.ndarray,
    method_name: str,
    save_path: str,
    pearson_r: float = None,
):
    """Scatter plot of predictive entropy vs inter-observer variability."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Subsample for visualization
    n = len(entropy_values)
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        entropy_values = entropy_values[idx]
        iov_values = iov_values[idx]

    ax.scatter(iov_values, entropy_values, alpha=0.1, s=1, c="steelblue")
    ax.set_xlabel("Inter-Observer Variability", fontsize=12)
    ax.set_ylabel("Predictive Entropy", fontsize=12)

    title = f"{method_name}: Entropy vs IOV"
    if pearson_r is not None:
        title += f" (r={pearson_r:.3f})"
    ax.set_title(title, fontsize=13)

    # Trend line
    if len(entropy_values) > 2:
        z = np.polyfit(iov_values, entropy_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(iov_values.min(), iov_values.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", default="results/figures/")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Run this after experiments complete to generate comparison figures.")

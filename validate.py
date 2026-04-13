#!/usr/bin/env python3
"""
Validation and inference with uncertainty estimation.

Evaluates trained models and generates:
- Uncertainty maps (entropy, MC variance, MI, evidential decomposition)
- Calibration curves and reliability diagrams
- Error-detection AUROC
- Entropy vs IOV correlation analysis
- Visualization outputs

Usage:
    python validate.py --config configs/lidc_baseline.yaml \
        --checkpoint checkpoints/lidc_baseline/best.pth \
        --compute_uncertainty --save_maps
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from train import load_config, set_seed, get_dataloader
from models import build_model, MCDropoutWrapper, DeepEnsemble, UNet
from losses import build_loss
from uncertainty.metrics import (
    compute_all_metrics,
    expected_calibration_error,
    entropy_iov_correlation,
    error_detection_auroc,
    predictive_entropy,
)


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_standard(model, dataloader, device, config, output_dir):
    """Evaluate standard model with softmax entropy."""
    all_probs, all_labels, all_iov, all_entropy = [], [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        output = model(images)

        if "prob" in output:
            probs = output["prob"].cpu().numpy()
        else:
            probs = torch.softmax(output["logits"], dim=1).cpu().numpy()

        labels = batch["label"].squeeze(1).cpu().numpy()
        if labels.max() <= 1.0 and labels.dtype == np.float32:
            labels = (labels >= 0.5).astype(np.int64)

        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

        all_probs.append(probs)
        all_labels.append(labels)
        all_entropy.append(entropy)

        if "iov_map" in batch:
            all_iov.append(batch["iov_map"].squeeze(1).cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_entropy = np.concatenate(all_entropy)
    iov = np.concatenate(all_iov) if all_iov else None

    metrics = compute_all_metrics(all_probs, all_labels, iov_map=iov)
    return metrics, all_probs, all_labels, all_entropy, iov


@torch.no_grad()
def evaluate_evidential(model, dataloader, device, config, output_dir):
    """Evaluate evidential model with uncertainty decomposition."""
    all_probs, all_labels, all_iov = [], [], []
    all_alpha, all_aleatoric, all_epistemic = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating (Evidential)"):
        images = batch["image"].to(device)
        output = model(images)

        alpha = output["alpha"].cpu().numpy()
        probs = output["prob"].cpu().numpy()

        labels = batch["label"].squeeze(1).cpu().numpy()
        if labels.max() <= 1.0:
            labels = (labels >= 0.5).astype(np.int64)

        # Decompose uncertainty
        C = alpha.shape[1]
        S = alpha.sum(axis=1, keepdims=True)
        aleatoric = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        epistemic = C / (S.squeeze(1) + 1)

        all_probs.append(probs)
        all_labels.append(labels)
        all_alpha.append(alpha)
        all_aleatoric.append(aleatoric)
        all_epistemic.append(epistemic)

        if "iov_map" in batch:
            all_iov.append(batch["iov_map"].squeeze(1).cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_alpha = np.concatenate(all_alpha)
    all_aleatoric = np.concatenate(all_aleatoric)
    all_epistemic = np.concatenate(all_epistemic)
    iov = np.concatenate(all_iov) if all_iov else None

    metrics = compute_all_metrics(all_probs, all_labels, iov_map=iov, alpha=all_alpha)
    return metrics, all_probs, all_labels, all_aleatoric, all_epistemic, iov


def plot_uncertainty_maps(
    image, label, prob, entropy, iov, aleatoric=None, epistemic=None,
    save_path=None, idx=0
):
    """Generate visualization of uncertainty maps for a single sample."""
    n_cols = 4 if aleatoric is None else 6
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred = (prob[1] >= 0.5).astype(np.float32) if prob.shape[0] == 2 else prob.argmax(axis=0)
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    im3 = axes[3].imshow(entropy, cmap="hot", vmin=0)
    axes[3].set_title("Predictive Entropy")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    if aleatoric is not None:
        im4 = axes[4].imshow(aleatoric, cmap="hot", vmin=0)
        axes[4].set_title("Aleatoric")
        axes[4].axis("off")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        im5 = axes[5].imshow(epistemic, cmap="hot", vmin=0)
        axes[5].set_title("Epistemic")
        axes[5].axis("off")
        plt.colorbar(im5, ax=axes[5], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(probs, labels, save_path, n_bins=15):
    """Plot reliability diagram."""
    confidence = probs.max(axis=1).flatten()
    preds = probs.argmax(axis=1).flatten()
    correct = (preds == labels.flatten()).astype(np.float32)

    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        confidence, correct, n_bins
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)

    ax.bar(bin_centers, bin_accs, width=1 / n_bins, alpha=0.7,
           edgecolor="black", label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/evaluation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--num_vis", type=int, default=20, help="Number of samples to visualize")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(config, args.checkpoint, device)
    test_loader = get_dataloader(config, "test")

    method = config["training"]["method"]
    print(f"Evaluating {method} model on test set...")

    if method == "evidential":
        metrics, probs, labels, aleatoric, epistemic, iov = evaluate_evidential(
            model, test_loader, device, config, args.output_dir
        )
    else:
        metrics, probs, labels, entropy, iov = evaluate_standard(
            model, test_loader, device, config, args.output_dir
        )
        aleatoric = None
        epistemic = None

    # Print metrics
    print("\n" + "=" * 60)
    print("Test Set Metrics:")
    print("=" * 60)
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
    print("=" * 60)

    # Save metrics
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()}, f, indent=2)

    # Plot calibration
    plot_calibration(probs, labels, output_dir / "calibration.png")
    print(f"Calibration plot saved to {output_dir / 'calibration.png'}")

    # Visualize uncertainty maps
    if args.save_maps:
        vis_dir = output_dir / "uncertainty_maps"
        vis_dir.mkdir(exist_ok=True)
        n_vis = min(args.num_vis, len(probs))

        for i in range(n_vis):
            if method == "evidential":
                ent = -(probs[i] * np.log(probs[i] + 1e-10)).sum(axis=0)
                plot_uncertainty_maps(
                    image=np.zeros_like(labels[i]),  # Placeholder
                    label=labels[i],
                    prob=probs[i],
                    entropy=ent,
                    iov=iov[i] if iov is not None else None,
                    aleatoric=aleatoric[i],
                    epistemic=epistemic[i],
                    save_path=vis_dir / f"sample_{i:03d}.png",
                )
            else:
                plot_uncertainty_maps(
                    image=np.zeros_like(labels[i]),
                    label=labels[i],
                    prob=probs[i],
                    entropy=entropy[i],
                    iov=iov[i] if iov is not None else None,
                    save_path=vis_dir / f"sample_{i:03d}.png",
                )
        print(f"Uncertainty maps saved to {vis_dir}")


if __name__ == "__main__":
    main()

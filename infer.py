#!/usr/bin/env python3
"""
Inference script: generate uncertainty maps for new images.

Usage:
    python infer.py --config configs/lidc_evidential.yaml \
        --checkpoint checkpoints/lidc_evidential/best.pth \
        --input_dir /path/to/nifti_images \
        --output_dir results/inference
"""

import argparse
import os
import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm

from train import load_config, set_seed
from models import build_model
from data.preprocessing import normalize_intensity, center_crop_or_pad


@torch.no_grad()
def infer_single(model, image_np, config, device):
    """Run inference on a single 2D image array."""
    method = config["training"]["method"]
    patch_size = tuple(config["preprocessing"]["spatial_size"])

    # Preprocess
    image = normalize_intensity(
        image_np,
        method=config["preprocessing"]["intensity_norm"],
        ct_window=tuple(config["preprocessing"].get("ct_window", [-1000, 400])),
    )
    image = center_crop_or_pad(image, patch_size, pad_value=image.min())
    tensor = torch.from_numpy(image[np.newaxis, np.newaxis]).float().to(device)

    # Forward
    output = model(tensor)

    result = {}
    if method == "evidential":
        alpha = output["alpha"].cpu().numpy()[0]  # (C, H, W)
        prob = output["prob"].cpu().numpy()[0]
        C = alpha.shape[0]
        S = alpha.sum(axis=0, keepdims=True)
        result["prediction"] = prob.argmax(axis=0)
        result["prob"] = prob
        result["entropy"] = -(prob * np.log(prob + 1e-10)).sum(axis=0)
        result["aleatoric"] = -(prob * np.log(prob + 1e-10)).sum(axis=0)
        result["epistemic"] = C / (S.squeeze(0) + 1)
        result["total_evidence"] = S.squeeze(0)
    else:
        logits = output["logits"].cpu().numpy()[0]
        prob = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
        result["prediction"] = prob.argmax(axis=0)
        result["prob"] = prob
        result["entropy"] = -(prob * np.log(prob + 1e-10)).sum(axis=0)

    return result


def save_results(result, save_dir, case_name):
    """Save inference results as NIfTI and visualization."""
    os.makedirs(save_dir, exist_ok=True)

    # Save prediction as NIfTI
    pred_nii = nib.Nifti1Image(result["prediction"].astype(np.float32), np.eye(4))
    nib.save(pred_nii, os.path.join(save_dir, f"{case_name}_prediction.nii.gz"))

    # Save entropy map
    ent_nii = nib.Nifti1Image(result["entropy"].astype(np.float32), np.eye(4))
    nib.save(ent_nii, os.path.join(save_dir, f"{case_name}_entropy.nii.gz"))

    # Save visualization
    has_decomp = "aleatoric" in result
    n_cols = 5 if has_decomp else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(result["prob"][1] if result["prob"].shape[0] == 2 else result["prob"][0], cmap="gray")
    axes[0].set_title("Foreground Prob")
    axes[0].axis("off")

    axes[1].imshow(result["prediction"], cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    im2 = axes[2].imshow(result["entropy"], cmap="hot")
    axes[2].set_title("Entropy")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if has_decomp:
        im3 = axes[3].imshow(result["aleatoric"], cmap="hot")
        axes[3].set_title("Aleatoric")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(result["epistemic"], cmap="cool")
        axes[4].set_title("Epistemic")
        axes[4].axis("off")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

    plt.suptitle(case_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{case_name}_vis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if has_decomp:
        ale_nii = nib.Nifti1Image(result["aleatoric"].astype(np.float32), np.eye(4))
        nib.save(ale_nii, os.path.join(save_dir, f"{case_name}_aleatoric.nii.gz"))
        epi_nii = nib.Nifti1Image(result["epistemic"].astype(np.float32), np.eye(4))
        nib.save(epi_nii, os.path.join(save_dir, f"{case_name}_epistemic.nii.gz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True, help="Directory with NIfTI images")
    parser.add_argument("--output_dir", default="results/inference")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Find input images
    nii_files = sorted(glob(os.path.join(args.input_dir, "*.nii.gz")))
    if not nii_files:
        nii_files = sorted(glob(os.path.join(args.input_dir, "*.nii")))
    print(f"Found {len(nii_files)} input images")

    for nii_path in tqdm(nii_files, desc="Inference"):
        case_name = Path(nii_path).stem.replace(".nii", "")
        img_nii = nib.load(nii_path)
        image = img_nii.get_fdata().astype(np.float32)

        if image.ndim == 3:
            # Process each slice
            for z in range(image.shape[2]):
                result = infer_single(model, image[:, :, z], config, device)
                save_results(result, args.output_dir, f"{case_name}_z{z:03d}")
        else:
            result = infer_single(model, image, config, device)
            save_results(result, args.output_dir, case_name)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
3D volumetric training pipeline for ambiguity-aware segmentation.

Features:
- ResEncUNet3D backbone (nnU-Net V2 Residual Encoder architecture)
- 3D patch-based training with foreground oversampling
- Deep supervision with exponentially decayed weights
- Poly learning rate schedule (nnU-Net convention)
- 3D sliding-window inference with Gaussian stitching and TTA
- Supports standard, evidential, and multi-annotator training
- Ensemble and MC Dropout inference modes

Usage:
    # Standard baseline
    python train_3d.py --config configs/lidc3d_baseline.yaml

    # Evidential
    python train_3d.py --config configs/lidc3d_evidential.yaml

    # With synthetic data for testing
    python train_3d.py --config configs/lidc3d_baseline.yaml --synthetic

    # Deep ensemble
    python train_3d.py --config configs/lidc3d_baseline.yaml --ensemble 5
"""

import argparse
import os
import json
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

from models.resenc_unet3d import ResEncUNet3D
from models.ensemble import DeepEnsemble
from losses.losses import DiceCELoss, EvidentialLoss, DistributionalLoss
from data.lidc_3d_dataset import (
    LIDC3DDataset, LIDC3DInferenceDataset,
    sliding_window_inference,
)
from uncertainty.metrics import compute_all_metrics


# ============================================================================
# Config loading
# ============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg["_base_"])
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
        cfg.pop("_base_", None)
    return cfg


def _deep_merge(base, over):
    result = deepcopy(base)
    for k, v in over.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


# ============================================================================
# Model factory
# ============================================================================

def build_model_3d(cfg: dict) -> nn.Module:
    """Build 3D ResEnc U-Net from config."""
    mcfg = cfg["model"]
    patch_size = tuple(cfg["preprocessing"]["patch_size"])
    evidential = mcfg.get("evidential", {}).get("enabled", False)
    dropout = mcfg.get("dropout", 0.0)
    if cfg.get("uncertainty", {}).get("mc_dropout", {}).get("enabled", False):
        dropout = cfg["uncertainty"]["mc_dropout"].get("dropout_rate", 0.1)

    model = ResEncUNet3D(
        in_channels=mcfg.get("in_channels", 1),
        num_classes=mcfg.get("num_classes", 2),
        patch_size=patch_size,
        base_features=mcfg.get("base_features", 32),
        max_features=mcfg.get("max_features", 320),
        n_stages=mcfg.get("n_stages", 5),
        blocks_per_stage=mcfg.get("blocks_per_stage", 2),
        dropout=dropout,
        deep_supervision=mcfg.get("deep_supervision", True),
        evidential=evidential,
        evidential_activation=mcfg.get("evidential", {}).get("activation", "softplus"),
    )
    return model


def build_loss_3d(cfg: dict):
    lcfg = cfg["training"]["loss"]
    name = lcfg["name"]
    if name == "dice_ce":
        return DiceCELoss(
            dice_weight=lcfg.get("dice_weight", 1.0),
            ce_weight=lcfg.get("ce_weight", 1.0),
        )
    elif name == "evidential":
        return EvidentialLoss(
            kl_weight=lcfg.get("kl_weight", 0.05),
            annealing_epochs=cfg["model"].get("evidential", {}).get("annealing_epochs", 20),
            dice_weight=lcfg.get("dice_weight", 0.5),
        )
    elif name == "distributional":
        return DistributionalLoss(
            divergence=lcfg.get("divergence", "kl"),
            dice_weight=lcfg.get("dice_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss: {name}")


# ============================================================================
# nnU-Net-style poly learning rate
# ============================================================================

class PolyLRScheduler:
    """Polynomial LR decay: lr = initial_lr * (1 - epoch/max_epoch)^0.9"""
    def __init__(self, optimizer, initial_lr, max_epochs, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent

    def step(self, epoch):
        new_lr = self.initial_lr * (1 - epoch / self.max_epochs) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr


# ============================================================================
# Deep supervision loss wrapper
# ============================================================================

def compute_ds_loss(
    output: dict,
    target: torch.Tensor,
    criterion,
    ds_weight_decay: float = 0.5,
    method: str = "standard",
    prob_map: torch.Tensor = None,
    annotator_masks: torch.Tensor = None,
):
    """
    Compute loss with deep supervision.

    Deep supervision weights decay exponentially from the finest resolution:
    w_0 = 1.0, w_1 = 0.5, w_2 = 0.25, ...
    """
    # Main output loss
    if method == "evidential":
        loss_dict = criterion(output["alpha"], target, output.get("prob"))
    elif method == "distributional" or method == "multi_annotator":
        loss_dict = criterion(output["logits"], prob_map, annotator_masks)
    else:
        loss_dict = criterion(output["logits"], target)

    total_loss = loss_dict["loss"]

    # Deep supervision losses
    if "deep_supervision" in output:
        ds_outputs = output["deep_supervision"]
        weight = ds_weight_decay
        for ds_logits in ds_outputs:
            # Downsample target to match
            target_ds = nn.functional.interpolate(
                target.float(), size=ds_logits.shape[2:],
                mode="nearest",
            )
            if method == "evidential":
                # For DS with evidential, just use CE on logits
                ds_loss = nn.functional.cross_entropy(
                    ds_logits, target_ds.squeeze(1).long(),
                    reduction="mean",
                )
            else:
                ds_loss = nn.functional.cross_entropy(
                    ds_logits, target_ds.squeeze(1).long(),
                    reduction="mean",
                )
            total_loss = total_loss + weight * ds_loss
            weight *= ds_weight_decay

    loss_dict["loss"] = total_loss
    return loss_dict


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    method = cfg["training"]["method"]
    if isinstance(criterion, EvidentialLoss):
        criterion.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["training"].get("mixed_precision", True)):
            output = model(images)
            loss_dict = compute_ds_loss(
                output, labels, criterion,
                ds_weight_decay=0.5,
                method=method,
                prob_map=batch.get("prob_map", labels).to(device) if method in ("distributional", "multi_annotator") else None,
                annotator_masks=batch.get("annotator_masks", None),
            )

        scaler.scale(loss_dict["loss"]).backward()

        # Gradient clipping
        clip_val = cfg["training"].get("gradient_clip", 12.0)
        if clip_val > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict["loss"].item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

    return total_loss / max(n_batches, 1)


# ============================================================================
# 3D Validation with sliding-window inference
# ============================================================================

@torch.no_grad()
def validate_3d(model, inference_dataset, device, cfg):
    """
    Full volumetric validation using sliding-window inference.

    For each test volume:
    1. Run sliding-window inference to get full-resolution predictions
    2. Compute Dice, HD95, entropy, calibration, error-detection AUROC
    3. Compute entropy-IOV correlation
    """
    model.eval()
    patch_size = tuple(cfg["preprocessing"]["patch_size"])
    num_classes = cfg["model"]["num_classes"]
    evidential = cfg["model"].get("evidential", {}).get("enabled", False)
    step_size = cfg.get("inference", {}).get("step_size", 0.5)

    all_metrics = []

    for idx in tqdm(range(len(inference_dataset)), desc="Validating (3D)"):
        sample = inference_dataset[idx]
        volume = sample["volume"]       # (D, H, W)
        label = sample["label"]         # (D, H, W)
        iov_map = sample["iov_map"]     # (D, H, W)

        # Sliding-window inference
        result = sliding_window_inference(
            model=model,
            volume=volume,
            patch_size=patch_size,
            num_classes=num_classes,
            step_size=step_size,
            device=device,
            batch_size=cfg.get("inference", {}).get("batch_size", 2),
            mirror_axes=(0, 1, 2) if cfg.get("inference", {}).get("tta", True) else None,
            evidential=evidential,
        )

        # Compute per-case metrics
        probs = result["prob"]                  # (C, D, H, W)
        prediction = result["prediction"]       # (D, H, W)
        entropy = result["entropy"]

        # Dice
        pred_fg = (prediction == 1).astype(np.float32)
        true_fg = (label >= 0.5).astype(np.float32)
        intersection = (pred_fg * true_fg).sum()
        dice = 2 * intersection / (pred_fg.sum() + true_fg.sum() + 1e-8)

        # Entropy-IOV correlation
        from uncertainty.metrics import entropy_iov_correlation, error_detection_auroc
        corr = entropy_iov_correlation(entropy, iov_map)

        # Error detection
        label_int = (label >= 0.5).astype(np.int64)
        auroc, auprc = error_detection_auroc(
            entropy.flatten(), prediction.flatten(), label_int.flatten(),
        )

        case_metrics = {
            "case_id": sample["case_id"],
            "dice": float(dice),
            "entropy_iov_pearson_r": corr["pearson_r"],
            "error_det_auroc": float(auroc),
            "mean_entropy": float(entropy.mean()),
        }

        # Evidential decomposition
        if evidential and "epistemic" in result:
            epi_auroc, _ = error_detection_auroc(
                result["epistemic"].flatten(),
                prediction.flatten(),
                label_int.flatten(),
            )
            case_metrics["epistemic_error_det_auroc"] = float(epi_auroc)
            case_metrics["mean_aleatoric"] = float(result["aleatoric"].mean())
            case_metrics["mean_epistemic"] = float(result["epistemic"].mean())

        all_metrics.append(case_metrics)

    # Aggregate
    agg = {}
    numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], float)]
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    return agg, all_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Volumetric Training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate & use synthetic 3D data")
    parser.add_argument("--ensemble", type=int, default=0,
                        help="Train ensemble with N members (0 = single model)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Synthetic data generation
    if args.synthetic:
        from data.prepare_lidc_3d import create_synthetic_lidc_3d
        create_synthetic_lidc_3d(cfg["data"]["root"], num_cases=80)

    # Datasets
    patch_size = tuple(cfg["preprocessing"]["patch_size"])
    multi_annot = cfg["model"].get("multi_annotator", {}).get("enabled", False)

    train_ds = LIDC3DDataset(
        root=cfg["data"]["root"],
        split="train",
        patch_size=patch_size,
        augment=True,
        multi_annotator=multi_annot,
    )
    val_ds = LIDC3DInferenceDataset(
        root=cfg["data"]["root"],
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Model
    if args.ensemble > 0:
        def model_fn():
            return build_model_3d(cfg)
        model = DeepEnsemble(model_fn, num_models=args.ensemble)
    else:
        model = build_model_3d(cfg)

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    criterion = build_loss_3d(cfg)

    # Optimizer (SGD with momentum, nnU-Net convention)
    opt_cfg = cfg["training"]["optimizer"]
    lr = opt_cfg.get("lr", 0.01)
    if opt_cfg.get("name", "sgd") == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.99, weight_decay=3e-5, nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

    # Scheduler (poly LR, nnU-Net convention)
    max_epochs = cfg["training"]["epochs"]
    scheduler = PolyLRScheduler(optimizer, lr, max_epochs, exponent=0.9)

    # Mixed precision
    scaler = GradScaler(enabled=cfg["training"].get("mixed_precision", True))

    # Save directory
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 1
    best_dice = -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", -1.0)
        print(f"Resumed from epoch {start_epoch}, best Dice {best_dice:.4f}")

    # Training
    print(f"\n{'='*60}")
    print(f"3D Training: {cfg['training']['method']} | {max_epochs} epochs")
    print(f"Patch size: {patch_size} | Batch: {cfg['training']['batch_size']}")
    print(f"Deep supervision: {cfg['model'].get('deep_supervision', True)}")
    print(f"Evidential: {cfg['model'].get('evidential', {}).get('enabled', False)}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    val_interval = cfg.get("logging", {}).get("val_every", 10)
    patience = cfg["training"].get("early_stopping", {}).get("patience", 50)
    patience_counter = 0

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch,
        )

        # LR step
        current_lr = scheduler.step(epoch)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:4d}/{max_epochs} | "
              f"Loss: {train_loss:.4f} | LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Validation
        if epoch % val_interval == 0 or epoch == max_epochs:
            print(f"  Running 3D sliding-window validation...")
            agg_metrics, per_case = validate_3d(model, val_ds, device, cfg)

            metric_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in agg_metrics.items()
                if not k.endswith("_std") and isinstance(v, float)
            )
            print(f"  Val: {metric_str}")

            # Save best
            current_dice = agg_metrics.get("dice", 0)
            if current_dice > best_dice:
                best_dice = current_dice
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "metrics": agg_metrics,
                    "config": cfg,
                }, save_dir / "best.pth")
                print(f"  -> Saved best model (Dice={best_dice:.4f})")

                # Save per-case metrics
                with open(save_dir / "best_per_case.json", "w") as f:
                    json.dump(per_case, f, indent=2)
            else:
                patience_counter += val_interval

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save final
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_dice": best_dice,
        "config": cfg,
    }, save_dir / "final.pth")

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()

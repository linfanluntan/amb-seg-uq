#!/usr/bin/env python3
"""
Main training script for ambiguity-aware segmentation.

Supports:
- Standard training (Dice+CE with softmax)
- Evidential training (Dirichlet output with evidential loss)
- Multi-annotator distributional training (KL to label distribution)
- Deep ensemble training (trains K independent models)
- MC Dropout (trained with dropout, evaluated stochastically)

Usage:
    python train.py --config configs/lidc_baseline.yaml
    python train.py --config configs/lidc_evidential.yaml
    python train.py --config configs/lidc_baseline.yaml --ensemble --num_models 5
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from models import build_model, UNet, DeepEnsemble
from losses import build_loss, EvidentialLoss, DistributionalLoss
from data.lidc_dataset import LIDCDataset, LIDCDataModule
from data.qubiq_dataset import QUBIQDataset
from uncertainty.metrics import compute_all_metrics


def load_config(config_path: str) -> dict:
    """Load YAML config with base config inheritance."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load base config if specified
    if "_base_" in config:
        base_path = os.path.join(os.path.dirname(config_path), config["_base_"])
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        # Deep merge
        merged = deep_merge(base_config, config)
        merged.pop("_base_", None)
        return merged
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataloader(config: dict, split: str) -> DataLoader:
    """Create dataloader based on config."""
    dataset_name = config["data"]["dataset"]
    batch_size = config["training"]["batch_size"] if split == "train" else config["training"]["batch_size"]
    multi_annot = config["model"].get("multi_annotator", {}).get("enabled", False)
    aug_config = config["preprocessing"].get("augmentation", {}) if split == "train" else None

    if dataset_name == "lidc":
        dataset = LIDCDataset(
            root=config["data"]["root"],
            split=split,
            patch_size=tuple(config["preprocessing"]["spatial_size"]),
            intensity_norm=config["preprocessing"]["intensity_norm"],
            ct_window=tuple(config["preprocessing"].get("ct_window", [-1000, 400])),
            augmentation_config=aug_config,
            multi_annotator=multi_annot,
        )
    elif dataset_name == "qubiq":
        dataset = QUBIQDataset(
            root=config["data"]["root"],
            task=config["data"].get("task", "prostate"),
            split=split,
            patch_size=tuple(config["preprocessing"]["spatial_size"]),
            intensity_norm=config["preprocessing"]["intensity_norm"],
            augmentation_config=aug_config,
            multi_annotator=multi_annot,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=(split == "train"),
    )


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, config, epoch
):
    """Train for one epoch."""
    model.train()
    method = config["training"]["method"]
    total_loss = 0.0
    num_batches = 0

    if isinstance(criterion, EvidentialLoss):
        criterion.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config["training"].get("mixed_precision", True)):
            output = model(images)

            if method == "evidential":
                loss_dict = criterion(
                    alpha=output["alpha"],
                    target=labels,
                    prob=output.get("prob"),
                )
            elif method == "multi_annotator":
                prob_map = batch["prob_map"].to(device)
                annotator_masks = batch.get("annotator_masks")
                if annotator_masks is not None:
                    annotator_masks = annotator_masks.to(device)
                loss_dict = criterion(
                    logits=output["logits"],
                    prob_map=prob_map,
                    annotator_masks=annotator_masks,
                )
            else:
                loss_dict = criterion(output["logits"], labels)

        loss = loss_dict["loss"]
        scaler.scale(loss).backward()

        # Gradient clipping
        if config["training"].get("gradient_clip", 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate and compute metrics."""
    model.eval()
    method = config["training"]["method"]
    total_loss = 0.0
    num_batches = 0

    all_probs = []
    all_labels = []
    all_iov = []
    all_alpha = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)

        if method == "evidential":
            loss_dict = criterion(output["alpha"], labels, output.get("prob"))
            probs = output["prob"].cpu().numpy()
            all_alpha.append(output["alpha"].cpu().numpy())
        elif method == "multi_annotator":
            prob_map = batch["prob_map"].to(device)
            loss_dict = criterion(output["logits"], prob_map)
            probs = output["prob"].cpu().numpy() if "prob" in output else \
                    torch.softmax(output["logits"], dim=1).cpu().numpy()
        else:
            loss_dict = criterion(output["logits"], labels)
            probs = torch.softmax(output["logits"], dim=1).cpu().numpy()

        total_loss += loss_dict["loss"].item()
        num_batches += 1

        all_probs.append(probs)
        lbl = labels.squeeze(1).cpu().numpy()
        if lbl.max() <= 1.0 and lbl.dtype == np.float32:
            lbl = (lbl >= 0.5).astype(np.int64)
        all_labels.append(lbl)

        if "iov_map" in batch:
            all_iov.append(batch["iov_map"].squeeze(1).cpu().numpy())

    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    iov = np.concatenate(all_iov, axis=0) if all_iov else None
    alpha = np.concatenate(all_alpha, axis=0) if all_alpha else None

    # Compute metrics
    metrics = compute_all_metrics(all_probs, all_labels, iov_map=iov, alpha=alpha)
    metrics["val_loss"] = total_loss / max(num_batches, 1)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--ensemble", action="store_true", help="Train deep ensemble")
    parser.add_argument("--num_models", type=int, default=5, help="Ensemble size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data if needed
    if args.synthetic:
        from data.download_lidc import create_synthetic_lidc
        create_synthetic_lidc(config["data"]["root"], num_samples=200)

    # Data
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")

    # Model
    if args.ensemble:
        def model_fn(**kwargs):
            return build_model(config)
        model = DeepEnsemble(model_fn, num_models=args.num_models)
    else:
        model = build_model(config)
    model = model.to(device)

    # Loss
    criterion = build_loss(config)

    # Optimizer
    opt_config = config["training"]["optimizer"]
    if opt_config["name"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 1e-4),
        )
    elif opt_config["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config["lr"],
            momentum=0.99,
            weight_decay=opt_config.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_config["lr"])

    # Scheduler
    sched_config = config["training"].get("scheduler", {})
    if sched_config.get("name") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get("T_max", config["training"]["epochs"]),
            eta_min=sched_config.get("eta_min", 1e-6),
        )
    else:
        scheduler = None

    # Mixed precision
    scaler = GradScaler(enabled=config["training"].get("mixed_precision", True))

    # Directories
    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_metric = -float("inf")
    patience_counter = 0
    patience = config["training"].get("early_stopping", {}).get("patience", 30)

    print(f"\nStarting training for {config['training']['epochs']} epochs")
    print(f"Method: {config['training']['method']}")
    print(f"Save directory: {save_dir}\n")

    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        if args.ensemble:
            # Train each ensemble member separately
            for k in range(args.num_models):
                member_model = model.models[k]
                member_optimizer = torch.optim.AdamW(
                    member_model.parameters(), lr=opt_config["lr"]
                )
                member_scaler = GradScaler()
                # Temporarily replace model for training function
                train_loss = train_one_epoch(
                    member_model, train_loader, criterion,
                    member_optimizer, member_scaler, device, config, epoch
                )
            train_loss = train_loss  # Last member's loss as representative
        else:
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config, epoch
            )

        # Validate
        metrics = validate(model, val_loader, criterion, device, config)

        if scheduler is not None:
            scheduler.step()

        # Print metrics
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()
                                  if isinstance(v, (int, float))])
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | {metric_str}")

        # Save best
        monitor = config["training"].get("early_stopping", {}).get("metric", "dice")
        current_metric = metrics.get(monitor, metrics.get("val_loss", 0))
        if config["training"].get("early_stopping", {}).get("mode", "max") == "min":
            current_metric = -current_metric

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            if args.ensemble:
                model.save_ensemble(str(save_dir / "best"))
            else:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }, save_dir / "best.pth")
            print(f"  -> Saved best model (metric={abs(best_metric):.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\nTraining complete!")
    print(f"Best {monitor}: {abs(best_metric):.4f}")


if __name__ == "__main__":
    main()

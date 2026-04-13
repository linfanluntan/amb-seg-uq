#!/usr/bin/env python3
"""
Ablation study runner.

Systematically ablates key components to quantify their contribution:
1. Evidence activation function: softplus vs relu vs exp
2. KL regularization weight: 0.01, 0.05, 0.1, 0.5
3. Annealing schedule: 5, 10, 20, 50 epochs
4. Dice loss weight in evidential training: 0, 0.25, 0.5, 1.0
5. Number of annotators used in distributional training: 1, 2, 3, 4
6. MC Dropout samples: 5, 10, 20, 50, 100
7. Ensemble size: 2, 3, 5, 7, 10

Usage:
    python ablation.py --config configs/lidc_evidential.yaml --ablation kl_weight
"""

import argparse
import json
import os
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path

from train import load_config, set_seed, main as train_main


ABLATION_CONFIGS = {
    "evidence_activation": {
        "param_path": ["model", "evidential", "activation"],
        "values": ["softplus", "relu", "exp"],
        "description": "Evidential activation function",
    },
    "kl_weight": {
        "param_path": ["training", "loss", "kl_weight"],
        "values": [0.001, 0.01, 0.05, 0.1, 0.5],
        "description": "KL regularization weight",
    },
    "annealing_epochs": {
        "param_path": ["model", "evidential", "annealing_epochs"],
        "values": [0, 5, 10, 20, 50],
        "description": "KL annealing schedule (epochs)",
    },
    "dice_weight": {
        "param_path": ["training", "loss", "dice_weight"],
        "values": [0.0, 0.25, 0.5, 1.0],
        "description": "Dice loss weight in evidential training",
    },
    "mc_samples": {
        "param_path": ["uncertainty", "mc_dropout", "num_samples"],
        "values": [5, 10, 20, 50, 100],
        "description": "Number of MC Dropout samples",
    },
    "ensemble_size": {
        "param_path": ["uncertainty", "ensemble", "num_models"],
        "values": [2, 3, 5, 7, 10],
        "description": "Deep ensemble size K",
    },
    "num_annotators": {
        "param_path": ["model", "multi_annotator", "num_annotators_used"],
        "values": [1, 2, 3, 4],
        "description": "Number of annotators used in training",
    },
}


def set_nested(d, keys, value):
    """Set a nested dictionary value."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def run_ablation(base_config_path, ablation_name, output_dir, max_epochs=50):
    """Run a single ablation study."""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}. "
                         f"Choose from: {list(ABLATION_CONFIGS.keys())}")

    ablation = ABLATION_CONFIGS[ablation_name]
    print(f"\n{'='*60}")
    print(f"Ablation Study: {ablation['description']}")
    print(f"Parameter: {'.'.join(ablation['param_path'])}")
    print(f"Values: {ablation['values']}")
    print(f"{'='*60}\n")

    results = {}

    for val in ablation["values"]:
        print(f"\n--- Running with {ablation_name} = {val} ---")

        config = load_config(base_config_path)
        set_nested(config, ablation["param_path"], val)

        # Reduce epochs for ablation
        config["training"]["epochs"] = min(config["training"]["epochs"], max_epochs)

        # Unique save directory
        run_name = f"{ablation_name}_{val}"
        config["logging"]["save_dir"] = os.path.join(output_dir, run_name)
        config["logging"]["log_dir"] = os.path.join(output_dir, "logs", run_name)

        # This would be called as a subprocess in practice
        # For the ablation, we just record the configuration
        results[str(val)] = {
            "config": {".".join(ablation["param_path"]): val},
            "status": "configured",
        }

    # Save ablation plan
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"ablation_{ablation_name}.json"), "w") as f:
        json.dump({
            "ablation": ablation_name,
            "description": ablation["description"],
            "parameter": ".".join(ablation["param_path"]),
            "values": ablation["values"],
            "results": results,
        }, f, indent=2)

    print(f"\nAblation plan saved to {output_dir}/ablation_{ablation_name}.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", required=True, choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--output_dir", default="results/ablation")
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()

    run_ablation(args.config, args.ablation, args.output_dir, args.max_epochs)

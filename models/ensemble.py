"""
Deep Ensemble for uncertainty estimation in segmentation.

Trains K independently initialized models and combines their predictions
to estimate total, epistemic, and (partial) aleatoric uncertainty.

Reference:
    Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles," NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy
import os


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble of K segmentation models.

    Each model is independently initialized and trained. At inference,
    predictions are averaged and inter-model disagreement is used as
    an uncertainty estimate.
    """

    def __init__(
        self,
        model_fn,
        num_models: int = 5,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            model_fn: Callable that returns a new model instance.
            num_models: Number of ensemble members K.
            model_kwargs: kwargs passed to model_fn.
        """
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()

        for k in range(num_models):
            model = model_fn(**(model_kwargs or {}))
            self.models.append(model)

    def forward(self, x: torch.Tensor, model_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Forward pass through a single ensemble member (for training)."""
        return self.models[model_idx](x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run all K models and compute ensemble statistics.

        Returns:
            dict with:
                'mean_prob': (B, C, H, W) mean softmax probability
                'ensemble_variance': (B, C, H, W) variance across members
                'entropy': (B, 1, H, W) entropy of mean prediction
                'mutual_information': (B, 1, H, W) MI = H(mean) - mean(H)
                'member_probs': (K, B, C, H, W) individual member predictions
        """
        all_probs = []
        for model in self.models:
            model.eval()
            output = model(x)
            all_probs.append(output["prob"])

        all_probs = torch.stack(all_probs, dim=0)  # (K, B, C, H, W)

        # Ensemble mean
        mean_prob = all_probs.mean(dim=0)  # (B, C, H, W)

        # Ensemble variance
        ensemble_variance = all_probs.var(dim=0)  # (B, C, H, W)

        # Total entropy: H(mean_prob)
        total_entropy = -(mean_prob * torch.log(mean_prob + 1e-10)).sum(dim=1, keepdim=True)

        # Expected entropy: mean over members of H(p_k)
        member_entropies = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)  # (K, B, H, W)
        expected_entropy = member_entropies.mean(dim=0, keepdim=True).permute(1, 0, 2, 3)  # (B, 1, H, W)

        # Mutual Information (epistemic uncertainty estimate)
        mutual_info = total_entropy - expected_entropy

        return {
            "mean_prob": mean_prob,
            "ensemble_variance": ensemble_variance,
            "entropy": total_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "member_probs": all_probs,
        }

    def save_ensemble(self, save_dir: str):
        """Save all ensemble members."""
        os.makedirs(save_dir, exist_ok=True)
        for k, model in enumerate(self.models):
            path = os.path.join(save_dir, f"member_{k}.pth")
            torch.save(model.state_dict(), path)

    def load_ensemble(self, save_dir: str):
        """Load all ensemble members."""
        for k, model in enumerate(self.models):
            path = os.path.join(save_dir, f"member_{k}.pth")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location="cpu"))

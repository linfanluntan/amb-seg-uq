"""
Monte Carlo Dropout for approximate Bayesian inference in segmentation.

Enables dropout at test time and runs T stochastic forward passes to
estimate predictive uncertainty via variance of softmax outputs.

Reference:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing
    Model Uncertainty in Deep Learning," ICML 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class MCDropoutWrapper(nn.Module):
    """
    Wraps a segmentation model for Monte Carlo Dropout inference.

    At test time, keeps dropout active and runs T forward passes to compute:
    - Mean prediction
    - Predictive entropy
    - MC variance (epistemic uncertainty proxy)
    """

    def __init__(self, model: nn.Module, num_samples: int = 20, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

        # Ensure model has dropout layers; add if not present
        self._ensure_dropout()

    def _ensure_dropout(self):
        """Add dropout to the model if it doesn't have any."""
        has_dropout = any(
            isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
            for m in self.model.modules()
        )
        if not has_dropout and self.dropout_rate > 0:
            # Add dropout after each ConvBlock in encoder/decoder
            for name, module in self.model.named_modules():
                if hasattr(module, 'block') and isinstance(module.block, nn.Sequential):
                    # Insert dropout before last activation
                    module.block.add_module(
                        f"mc_dropout",
                        nn.Dropout2d(p=self.dropout_rate)
                    )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standard forward (single pass)."""
        return self.model(x)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run T stochastic forward passes with dropout enabled.

        Returns:
            dict with:
                'mean_prob': (B, C, H, W) mean softmax probabilities
                'mc_variance': (B, C, H, W) variance across samples
                'entropy': (B, 1, H, W) predictive entropy of mean
                'samples': (T, B, C, H, W) all softmax samples
        """
        T = num_samples or self.num_samples
        self.model.enable_dropout()
        self.model.eval()  # Keep batchnorm in eval, only dropout in train

        all_probs = []
        for _ in range(T):
            output = self.model(x)
            probs = output["prob"]  # (B, C, H, W)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)  # (T, B, C, H, W)

        # Mean prediction
        mean_prob = all_probs.mean(dim=0)  # (B, C, H, W)

        # MC Variance
        mc_variance = all_probs.var(dim=0)  # (B, C, H, W)

        # Predictive entropy of mean
        entropy = -(mean_prob * torch.log(mean_prob + 1e-10)).sum(dim=1, keepdim=True)

        # Per-sample entropy (for mutual information)
        sample_entropies = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)  # (T, B, H, W)
        mean_sample_entropy = sample_entropies.mean(dim=0, keepdim=True).permute(1, 0, 2, 3)  # (B, 1, H, W)

        # Mutual Information = Total Entropy - Mean Sample Entropy
        mutual_info = entropy - mean_sample_entropy

        return {
            "mean_prob": mean_prob,
            "mc_variance": mc_variance,
            "entropy": entropy,
            "mutual_information": mutual_info,
            "samples": all_probs,
        }

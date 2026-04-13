"""
Evidential Deep Learning head for segmentation with Dirichlet uncertainty.

Replaces the standard softmax layer with a Dirichlet distribution over
class probabilities, enabling explicit decomposition of epistemic and
aleatoric uncertainty.

References:
    Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty," NeurIPS 2018.

    Zou et al., "Towards Reliable Medical Image Segmentation by Modeling
    Evidential Calibrated Uncertainty," arXiv:2301.00349, 2023.

    Hemmer et al., "Region-based Evidential Deep Learning to Quantify
    Uncertainty and Improve Robustness of Brain Tumor Segmentation,"
    Neural Networks, 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialHead(nn.Module):
    """
    Evidential segmentation head using Dirichlet distribution.

    For each voxel x, outputs Dirichlet concentration parameters:
        alpha(x) = [alpha_1, ..., alpha_C], alpha_c > 0

    This defines:
        p(p(y|x)) = Dir(alpha(x))

    From which:
        Mean prediction: p_hat_c = alpha_c / S, S = sum(alpha)
        Aleatoric: H(p_hat) — entropy of the mean prediction
        Epistemic: C / (S + 1) — inversely proportional to total evidence
    """

    def __init__(self, num_classes: int = 2, activation: str = "softplus"):
        super().__init__()
        self.num_classes = num_classes
        if activation == "softplus":
            self.act = nn.Softplus()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "exp":
            self.act = lambda x: torch.exp(torch.clamp(x, max=10))
        else:
            self.act = nn.Softplus()

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert raw logits to Dirichlet parameters and uncertainty components.

        Args:
            logits: (B, C, H, W) raw network output

        Returns:
            dict with:
                'alpha': (B, C, H, W) Dirichlet concentration parameters
                'evidence': (B, C, H, W) evidence = alpha - 1
                'prob': (B, C, H, W) expected class probabilities
                'total_evidence': (B, 1, H, W) sum of alpha
                'aleatoric': (B, 1, H, W) entropy of mean prediction
                'epistemic': (B, 1, H, W) C / (S + 1)
                'total_uncertainty': (B, 1, H, W) aleatoric + epistemic proxy
        """
        # Evidence: non-negative via activation
        evidence = self.act(logits)  # (B, C, H, W)

        # Dirichlet concentration: alpha = evidence + 1
        alpha = evidence + 1.0

        # Total evidence (Dirichlet strength)
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Expected class probabilities (mean of Dirichlet)
        prob = alpha / S  # (B, C, H, W)

        # Aleatoric uncertainty: entropy of the mean prediction
        aleatoric = -(prob * torch.log(prob + 1e-10)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Epistemic uncertainty: inversely proportional to evidence strength
        epistemic = self.num_classes / (S + 1.0)  # (B, 1, H, W)

        # Dirichlet variance for each class
        dirichlet_var = (alpha * (S - alpha)) / (S.pow(2) * (S + 1))  # (B, C, H, W)

        return {
            "alpha": alpha,
            "evidence": evidence,
            "prob": prob,
            "total_evidence": S,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": aleatoric + epistemic,
            "dirichlet_variance": dirichlet_var,
        }


def evidential_uncertainty_decomposition(
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose uncertainty from Dirichlet parameters.

    Args:
        alpha: (B, C, H, W) Dirichlet concentrations

    Returns:
        total_uncertainty: (B, 1, H, W)
        aleatoric: (B, 1, H, W)
        epistemic: (B, 1, H, W)
    """
    C = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    prob = alpha / S

    # Aleatoric: entropy of expected distribution
    aleatoric = -(prob * torch.log(prob + 1e-10)).sum(dim=1, keepdim=True)

    # Epistemic: expected entropy of the Dirichlet
    # E[H(p)] = psi(S+1) - sum(alpha_c/S * psi(alpha_c + 1))
    # Simplified proxy: C / (S + 1)
    epistemic = C / (S + 1.0)

    total = aleatoric + epistemic

    return total, aleatoric, epistemic

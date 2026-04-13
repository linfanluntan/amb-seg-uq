"""
Loss functions for standard, evidential, and distributional segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss for segmentation.
    Standard baseline loss assuming single ground-truth labels.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        softmax: bool = True,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.softmax = softmax

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, C, H, W) raw model output
            target: (B, 1, H, W) or (B, H, W) integer labels or soft labels
        """
        B, C, H, W = logits.shape

        # Prepare target
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # (B, H, W)

        # One-hot encode if integer labels
        if target.dtype in (torch.long, torch.int):
            target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
        elif target.max() <= 1.0 and C == 2:
            # Soft label for binary: (B, H, W) in [0,1] -> (B, 2, H, W)
            target_onehot = torch.stack([1 - target, target], dim=1)
        else:
            target_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()

        # Softmax probabilities
        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        # Dice loss (per-class, then averaged)
        intersection = (probs * target_onehot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        # Cross-entropy
        ce_loss = F.cross_entropy(logits, target.long(), reduction="mean")

        total = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return {
            "loss": total,
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
        }


class EvidentialLoss(nn.Module):
    """
    Evidential loss for Dirichlet-based segmentation.

    Combines:
    1. Bayes risk of cross-entropy under the Dirichlet
    2. KL divergence regularizer toward non-informative prior
    3. Optional Dice component for shape-awareness

    Reference:
        Sensoy et al., NeurIPS 2018.
        Hemmer et al., Neural Networks 2023 (region-based EDL).
    """

    def __init__(
        self,
        kl_weight: float = 0.05,
        annealing_epochs: int = 10,
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            alpha: (B, C, H, W) Dirichlet concentration parameters
            target: (B, 1, H, W) or (B, H, W) ground-truth labels
            prob: (B, C, H, W) mean prediction (alpha/S), optional
        """
        B, C, H, W = alpha.shape

        if target.dim() == 4:
            target = target.squeeze(1)

        # One-hot target
        if target.dtype in (torch.long, torch.int):
            target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
        elif C == 2 and target.max() <= 1.0:
            target_onehot = torch.stack([1 - target, target], dim=1)
        else:
            target_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()

        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 1. Bayes risk of cross-entropy: E_{Dir(alpha)}[-log p(y)]
        # = psi(S) - psi(alpha_y) = digamma(S) - digamma(alpha_y)
        bayes_risk = (
            target_onehot * (torch.digamma(S) - torch.digamma(alpha))
        ).sum(dim=1).mean()

        # 2. KL divergence: KL(Dir(alpha_tilde) || Dir(1))
        # where alpha_tilde removes evidence for the correct class
        alpha_tilde = target_onehot + (1 - target_onehot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

        kl_div = (
            torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(torch.tensor(C, dtype=torch.float32, device=alpha.device))
            - (torch.lgamma(alpha_tilde)).sum(dim=1)
            + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1)
        ).mean()

        # Annealing coefficient
        if self.annealing_epochs > 0:
            anneal = min(1.0, self.current_epoch / self.annealing_epochs)
        else:
            anneal = 1.0

        # 3. Optional Dice loss on mean prediction
        dice_loss = torch.tensor(0.0, device=alpha.device)
        if self.dice_weight > 0 and prob is not None:
            intersection = (prob * target_onehot).sum(dim=(2, 3))
            cardinality = prob.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
            dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss = 1.0 - dice_per_class.mean()

        total = bayes_risk + anneal * self.kl_weight * kl_div + self.dice_weight * dice_loss

        return {
            "loss": total,
            "bayes_risk": bayes_risk,
            "kl_div": kl_div,
            "dice_loss": dice_loss,
            "anneal_coeff": torch.tensor(anneal),
        }


class DistributionalLoss(nn.Module):
    """
    Multi-annotator distributional loss.

    Instead of training against a single ground truth, minimizes the
    divergence between predicted probabilities and the empirical label
    distribution from multiple annotators.

    L = KL(p* || p) or JS(p*, p)

    This teaches the model to reproduce human disagreement rather than
    collapsing it into a single label.
    """

    def __init__(
        self,
        divergence: str = "kl",  # kl | js | ce_avg
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.divergence = divergence
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        prob_map: torch.Tensor,
        annotator_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, C, H, W) model output
            prob_map: (B, 1, H, W) empirical probability from annotators [0,1]
            annotator_masks: (B, M, H, W) individual annotator masks (optional)
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Target distribution: p*(y|x)
        p_star = prob_map.squeeze(1)  # (B, H, W) — probability of foreground
        if C == 2:
            target_dist = torch.stack([1 - p_star, p_star], dim=1)  # (B, 2, H, W)
        else:
            raise NotImplementedError("Multi-class distributional loss not yet implemented")

        if self.divergence == "kl":
            # KL(p* || p) = sum p* log(p* / p)
            div_loss = F.kl_div(
                torch.log(probs + 1e-10),
                target_dist,
                reduction="batchmean",
                log_target=False,
            )
        elif self.divergence == "js":
            # JS divergence = 0.5 * KL(p*||m) + 0.5 * KL(p||m), m = (p*+p)/2
            m = 0.5 * (target_dist + probs)
            kl1 = F.kl_div(torch.log(m + 1e-10), target_dist, reduction="batchmean", log_target=False)
            kl2 = F.kl_div(torch.log(m + 1e-10), probs, reduction="batchmean", log_target=False)
            div_loss = 0.5 * (kl1 + kl2)
        elif self.divergence == "ce_avg":
            # Average cross-entropy over all annotators
            if annotator_masks is None:
                raise ValueError("ce_avg requires annotator_masks")
            M = annotator_masks.shape[1]
            ce_sum = torch.tensor(0.0, device=logits.device)
            for m in range(M):
                mask_m = annotator_masks[:, m]  # (B, H, W)
                ce_sum += F.cross_entropy(logits, mask_m.long(), reduction="mean")
            div_loss = ce_sum / M
        else:
            raise ValueError(f"Unknown divergence: {self.divergence}")

        # Dice on mean prediction vs majority vote
        dice_loss = torch.tensor(0.0, device=logits.device)
        if self.dice_weight > 0:
            majority = (p_star >= 0.5).float()
            target_onehot = torch.stack([1 - majority, majority], dim=1)
            intersection = (probs * target_onehot).sum(dim=(2, 3))
            cardinality = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
            dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss = 1.0 - dice_per_class.mean()

        total = div_loss + self.dice_weight * dice_loss

        return {
            "loss": total,
            "divergence_loss": div_loss,
            "dice_loss": dice_loss,
        }

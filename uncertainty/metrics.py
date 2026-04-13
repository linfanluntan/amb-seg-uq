"""
Uncertainty estimation and evaluation metrics.

Implements:
- Predictive entropy
- Mutual information (ensemble/MC Dropout)
- Evidential decomposition (Dirichlet)
- Calibration metrics (ECE, Brier, reliability diagrams)
- Error-detection AUROC
- Entropy vs Inter-Observer Variability correlation
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


# ============================================================================
# Uncertainty Maps
# ============================================================================

def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute per-voxel predictive entropy.
    H(p) = -sum_c p_c * log(p_c)

    Args:
        probs: (B, C, H, W) softmax probabilities
    Returns:
        entropy: (B, 1, H, W)
    """
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1, keepdim=True)


def mutual_information(member_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mutual information from ensemble/MC samples.
    MI = H(mean_p) - E[H(p_k)]

    Args:
        member_probs: (K, B, C, H, W) predictions from K models/samples
    Returns:
        total_entropy, expected_entropy, mutual_info: each (B, 1, H, W)
    """
    mean_prob = member_probs.mean(dim=0)  # (B, C, H, W)
    total_ent = predictive_entropy(mean_prob)

    # Expected entropy
    member_ent = -(member_probs * torch.log(member_probs + 1e-10)).sum(dim=2)  # (K, B, H, W)
    exp_ent = member_ent.mean(dim=0, keepdim=True).permute(1, 0, 2, 3)  # (B, 1, H, W)

    mi = total_ent - exp_ent
    return total_ent, exp_ent, mi


def mc_variance(member_probs: torch.Tensor) -> torch.Tensor:
    """
    Per-voxel variance across MC/ensemble samples.

    Args:
        member_probs: (K, B, C, H, W)
    Returns:
        variance: (B, C, H, W)
    """
    return member_probs.var(dim=0)


# ============================================================================
# Calibration Metrics
# ============================================================================

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: (N,) predicted confidence (max softmax probability)
        labels: (N,) binary correctness (1 if predicted class matches true)
        n_bins: Number of confidence bins

    Returns:
        ece: scalar ECE
        bin_accs: per-bin accuracy
        bin_confs: per-bin mean confidence
        bin_counts: per-bin sample count
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs[i] = labels[mask].mean()
            bin_confs[i] = probs[mask].mean()
            bin_counts[i] = mask.sum()

    total = bin_counts.sum()
    if total == 0:
        return 0.0, bin_accs, bin_confs, bin_counts

    ece = (bin_counts / total * np.abs(bin_accs - bin_confs)).sum()
    return float(ece), bin_accs, bin_confs, bin_counts


def brier_score(probs: np.ndarray, labels_onehot: np.ndarray) -> float:
    """
    Compute Brier score: mean squared difference between predicted probs and one-hot labels.
    """
    return float(np.mean((probs - labels_onehot) ** 2))


# ============================================================================
# Error Detection
# ============================================================================

def error_detection_auroc(
    uncertainty: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute AUROC for using uncertainty to detect segmentation errors.

    "Error" = voxels where predicted class != ground truth class.
    A good uncertainty estimator assigns high uncertainty to errors.

    Args:
        uncertainty: (N,) per-voxel uncertainty values
        predictions: (N,) predicted class labels
        ground_truth: (N,) true class labels
        mask: (N,) optional mask (e.g., exclude background)

    Returns:
        auroc: Area under ROC curve
        auprc: Area under Precision-Recall curve
    """
    errors = (predictions != ground_truth).astype(np.float32)

    if mask is not None:
        uncertainty = uncertainty[mask > 0]
        errors = errors[mask > 0]

    if len(np.unique(errors)) < 2:
        return 0.5, 0.0  # All correct or all wrong

    auroc = roc_auc_score(errors, uncertainty)
    auprc = average_precision_score(errors, uncertainty)
    return float(auroc), float(auprc)


# ============================================================================
# Correlation Analysis
# ============================================================================

def entropy_iov_correlation(
    entropy_map: np.ndarray,
    iov_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute correlation between predicted entropy and inter-observer variability.

    High correlation indicates uncertainty mirrors human disagreement (aleatoric),
    not model ignorance (epistemic).

    Args:
        entropy_map: (H, W) predictive entropy
        iov_map: (H, W) inter-observer variability (variance across annotators)
        mask: (H, W) optional foreground mask

    Returns:
        dict with pearson_r, spearman_rho, and p-values
    """
    if mask is not None:
        ent_flat = entropy_map[mask > 0].flatten()
        iov_flat = iov_map[mask > 0].flatten()
    else:
        ent_flat = entropy_map.flatten()
        iov_flat = iov_map.flatten()

    # Remove zeros for meaningful correlation
    nonzero = (iov_flat > 0) | (ent_flat > 0)
    ent_flat = ent_flat[nonzero]
    iov_flat = iov_flat[nonzero]

    if len(ent_flat) < 10:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}

    pr, pp = pearsonr(ent_flat, iov_flat)
    sr, sp = spearmanr(ent_flat, iov_flat)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    iov_map: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty evaluation metrics.

    Args:
        probs: (B, C, H, W) predicted probabilities
        labels: (B, H, W) ground truth integer labels
        iov_map: (B, H, W) inter-observer variability (optional)
        alpha: (B, C, H, W) Dirichlet concentrations (optional, for evidential)

    Returns:
        Dictionary of all computed metrics
    """
    B, C, H, W = probs.shape
    metrics = {}

    # Segmentation accuracy
    preds = probs.argmax(axis=1)  # (B, H, W)
    correct = (preds == labels).astype(np.float32)
    metrics["accuracy"] = float(correct.mean())

    # Dice score (foreground class)
    if C == 2:
        pred_fg = (preds == 1).astype(np.float32)
        true_fg = (labels == 1).astype(np.float32)
        intersection = (pred_fg * true_fg).sum()
        dice = 2 * intersection / (pred_fg.sum() + true_fg.sum() + 1e-8)
        metrics["dice"] = float(dice)

    # Predictive entropy
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)  # (B, H, W)
    metrics["mean_entropy"] = float(entropy.mean())

    # Calibration
    confidence = probs.max(axis=1).flatten()
    correct_flat = correct.flatten()
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        confidence, correct_flat, n_bins
    )
    metrics["ece"] = ece

    # Brier score
    labels_onehot = np.eye(C)[labels.astype(int)]  # (B, H, W, C)
    labels_onehot = labels_onehot.transpose(0, 3, 1, 2)  # (B, C, H, W)
    metrics["brier"] = brier_score(probs, labels_onehot)

    # Error detection
    entropy_flat = entropy.flatten()
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    auroc, auprc = error_detection_auroc(entropy_flat, preds_flat, labels_flat)
    metrics["error_det_auroc"] = auroc
    metrics["error_det_auprc"] = auprc

    # Entropy vs IOV correlation
    if iov_map is not None:
        for b in range(B):
            corr = entropy_iov_correlation(entropy[b], iov_map[b])
            for key, val in corr.items():
                if f"entropy_iov_{key}" not in metrics:
                    metrics[f"entropy_iov_{key}"] = []
                metrics[f"entropy_iov_{key}"].append(val)
        # Average over batch
        for key in list(metrics.keys()):
            if key.startswith("entropy_iov_") and isinstance(metrics[key], list):
                metrics[key] = float(np.mean(metrics[key]))

    # Evidential decomposition
    if alpha is not None:
        S = alpha.sum(axis=1, keepdims=True)
        prob_evid = alpha / S
        aleatoric = -(prob_evid * np.log(prob_evid + 1e-10)).sum(axis=1)
        epistemic = C / (S.squeeze(1) + 1)

        metrics["mean_aleatoric"] = float(aleatoric.mean())
        metrics["mean_epistemic"] = float(epistemic.mean())

        # Epistemic-based error detection
        epi_auroc, epi_auprc = error_detection_auroc(
            epistemic.flatten(), preds_flat, labels_flat
        )
        metrics["epistemic_error_det_auroc"] = epi_auroc

        # Aleatoric vs IOV correlation
        if iov_map is not None:
            for b in range(B):
                corr = entropy_iov_correlation(aleatoric[b], iov_map[b])
                metrics[f"aleatoric_iov_pearson_r"] = float(corr["pearson_r"])
            corr_epi = entropy_iov_correlation(epistemic[0], iov_map[0])
            metrics["epistemic_iov_pearson_r"] = float(corr_epi["pearson_r"])

    return metrics

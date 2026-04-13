"""
3D volumetric LIDC-IDRI dataset for multi-annotator lung nodule segmentation.

Handles:
- Full 3D CT volumes with per-nodule ROI crops
- All 4 radiologist volumetric annotations
- Proper 3D resampling to isotropic spacing
- 3D patch extraction with foreground oversampling
- 3D sliding-window inference with Gaussian weighting for stitching

Reference:
    Armato III et al., "The Lung Image Database Consortium (LIDC) and Image
    Database Resource Initiative (IDRI): A completed reference database of lung
    nodules on CT scans," Medical Physics, 38(2):915-931, 2011.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy.ndimage import zoom as scipy_zoom, gaussian_filter
import warnings


# ---------------------------------------------------------------------------
# 3D preprocessing helpers
# ---------------------------------------------------------------------------

def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 3D volume to target isotropic spacing."""
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resampled = scipy_zoom(volume, zoom_factors, order=order)
    return resampled, zoom_factors


def resample_mask(
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Resample a binary mask to target spacing (nearest-neighbor)."""
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    return scipy_zoom(mask.astype(np.float32), zoom_factors, order=0).astype(np.uint8)


def normalize_ct(volume: np.ndarray, window: Tuple[float, float] = (-1000, 400)) -> np.ndarray:
    """CT windowing followed by [0,1] normalization."""
    v = np.clip(volume.astype(np.float32), window[0], window[1])
    v = (v - window[0]) / (window[1] - window[0])
    return v


def extract_3d_patch(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Extract a 3D patch centered at `center`, with zero-padding if needed."""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    d0 = center[0] - pd // 2
    h0 = center[1] - ph // 2
    w0 = center[2] - pw // 2

    # Compute source and destination slices
    src_slices = []
    dst_slices = []
    for dim_size, start, psize in [(D, d0, pd), (H, h0, ph), (W, w0, pw)]:
        s_start = max(0, start)
        s_end = min(dim_size, start + psize)
        d_start = s_start - start
        d_end = d_start + (s_end - s_start)
        src_slices.append(slice(s_start, s_end))
        dst_slices.append(slice(d_start, d_end))

    patch = np.full(patch_size, pad_value, dtype=volume.dtype)
    patch[tuple(dst_slices)] = volume[tuple(src_slices)]
    return patch


def random_3d_crop(
    volume: np.ndarray,
    masks: np.ndarray,
    patch_size: Tuple[int, int, int],
    foreground_prob: float = 0.67,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random 3D crop with foreground oversampling.

    Following nnU-Net convention: ~2/3 of patches are centered on foreground
    voxels, ~1/3 are fully random.

    Args:
        volume: (D, H, W) CT volume
        masks: (N_ann, D, H, W) annotator masks
        patch_size: (pD, pH, pW)
        foreground_prob: probability of centering on foreground

    Returns:
        patch_vol, patch_masks: cropped volume and masks
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size

    # Union of all annotator masks for foreground detection
    fg_mask = masks.max(axis=0) > 0

    if np.random.random() < foreground_prob and fg_mask.any():
        fg_coords = np.argwhere(fg_mask)
        idx = np.random.randint(len(fg_coords))
        center = fg_coords[idx]
    else:
        center = np.array([
            np.random.randint(0, max(1, D)),
            np.random.randint(0, max(1, H)),
            np.random.randint(0, max(1, W)),
        ])

    patch_vol = extract_3d_patch(volume, center, patch_size, pad_value=volume.min())
    patch_masks = np.stack([
        extract_3d_patch(m.astype(np.float32), center, patch_size, pad_value=0.0)
        for m in masks
    ])

    return patch_vol, patch_masks


def augment_3d(
    volume: np.ndarray,
    masks: np.ndarray,
    p_flip: float = 0.5,
    p_noise: float = 0.15,
    noise_std: float = 0.01,
    p_brightness: float = 0.15,
    brightness_range: Tuple[float, float] = (0.7, 1.3),
    p_gamma: float = 0.15,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D data augmentation following nnU-Net conventions.

    Includes random flips along all three axes, additive Gaussian noise,
    multiplicative brightness, and gamma correction.
    """
    # Random flips along each axis
    for axis in range(3):
        if np.random.random() < p_flip:
            volume = np.flip(volume, axis=axis).copy()
            masks = np.flip(masks, axis=axis + 1).copy()  # masks have extra leading dim

    # Additive Gaussian noise
    if np.random.random() < p_noise:
        volume = volume + np.random.normal(0, noise_std, volume.shape).astype(np.float32)

    # Multiplicative brightness
    if np.random.random() < p_brightness:
        factor = np.random.uniform(*brightness_range)
        volume = np.clip(volume * factor, 0, 1)

    # Gamma correction
    if np.random.random() < p_gamma:
        gamma = np.random.uniform(*gamma_range)
        eps = 1e-7
        volume = np.clip(volume, eps, None)
        volume = np.power(volume, gamma)

    return volume, masks


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def get_gaussian_importance_map(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 1.0 / 8,
) -> np.ndarray:
    """
    Compute a Gaussian importance map for weighting overlapping patches
    during sliding-window inference (nnU-Net convention).
    """
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = [s // 2 for s in patch_size]
    tmp[tuple(center)] = 1.0
    sigmas = [s * sigma_scale for s in patch_size]
    importance = gaussian_filter(tmp, sigmas, mode="constant", cval=0)
    importance /= importance.max()
    importance = np.clip(importance, 1e-5, None)
    return importance


def sliding_window_positions(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    step_size: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    Compute all starting positions for sliding-window inference.

    Args:
        volume_shape: (D, H, W) of the full volume
        patch_size: (pD, pH, pW)
        step_size: fraction of patch_size to step (0.5 = 50% overlap)

    Returns:
        List of (d, h, w) starting coordinates
    """
    positions = []
    steps = [max(1, int(p * step_size)) for p in patch_size]

    for d in range(0, max(1, volume_shape[0] - patch_size[0] + 1), steps[0]):
        for h in range(0, max(1, volume_shape[1] - patch_size[1] + 1), steps[1]):
            for w in range(0, max(1, volume_shape[2] - patch_size[2] + 1), steps[2]):
                positions.append((d, h, w))

    # Ensure the volume corners are covered
    for d in [0, max(0, volume_shape[0] - patch_size[0])]:
        for h in [0, max(0, volume_shape[1] - patch_size[1])]:
            for w in [0, max(0, volume_shape[2] - patch_size[2])]:
                pos = (d, h, w)
                if pos not in positions:
                    positions.append(pos)

    return positions


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    step_size: float = 0.5,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 2,
    mirror_axes: Optional[Tuple[int, ...]] = (0, 1, 2),
    evidential: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run sliding-window inference on a full 3D volume.

    Produces a full-resolution prediction by:
    1. Extracting overlapping 3D patches
    2. Running each through the model
    3. Aggregating with Gaussian-weighted averaging

    Also supports test-time augmentation via mirroring along specified axes.

    Args:
        model: Trained 3D segmentation model
        volume: (D, H, W) normalized CT volume
        patch_size: (pD, pH, pW)
        num_classes: number of output classes
        step_size: overlap fraction (0.5 = 50%)
        mirror_axes: axes to mirror for TTA; None to disable

    Returns:
        dict with 'prob', 'prediction', and uncertainty maps
    """
    model.eval()
    vol_shape = volume.shape  # (D, H, W)

    # Pad volume so it's at least patch_size in each dim
    pad_d = max(0, patch_size[0] - vol_shape[0])
    pad_h = max(0, patch_size[1] - vol_shape[1])
    pad_w = max(0, patch_size[2] - vol_shape[2])
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(
            volume,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=volume.min(),
        )

    padded_shape = volume.shape
    positions = sliding_window_positions(padded_shape, patch_size, step_size)

    # Gaussian importance map
    importance = get_gaussian_importance_map(patch_size)

    # Accumulation buffers
    aggregated_prob = np.zeros((num_classes,) + padded_shape, dtype=np.float64)
    weight_map = np.zeros(padded_shape, dtype=np.float64)

    # If evidential, also accumulate alpha
    if evidential:
        aggregated_alpha = np.zeros((num_classes,) + padded_shape, dtype=np.float64)

    # Collect augmentation configs
    aug_configs = [None]  # No augmentation
    if mirror_axes is not None:
        for ax in mirror_axes:
            aug_configs.append((ax,))

    # Process patches in batches
    for aug in aug_configs:
        batch_patches = []
        batch_positions = []

        for pos in positions:
            d, h, w = pos
            patch = volume[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]

            # Ensure correct size
            if patch.shape != patch_size:
                pad_spec = tuple(
                    (0, ps - cs) for ps, cs in zip(patch_size, patch.shape)
                )
                patch = np.pad(patch, pad_spec, mode="constant",
                               constant_values=volume.min())

            # Apply augmentation
            if aug is not None:
                for ax in aug:
                    patch = np.flip(patch, axis=ax).copy()

            batch_patches.append(patch)
            batch_positions.append(pos)

            if len(batch_patches) >= batch_size:
                _process_batch(
                    model, batch_patches, batch_positions, patch_size,
                    importance, aggregated_prob, weight_map,
                    aggregated_alpha if evidential else None,
                    aug, device, evidential,
                )
                batch_patches = []
                batch_positions = []

        if batch_patches:
            _process_batch(
                model, batch_patches, batch_positions, patch_size,
                importance, aggregated_prob, weight_map,
                aggregated_alpha if evidential else None,
                aug, device, evidential,
            )

    # Normalize
    weight_map = np.clip(weight_map, 1e-8, None)
    for c in range(num_classes):
        aggregated_prob[c] /= weight_map

    if evidential:
        for c in range(num_classes):
            aggregated_alpha[c] /= weight_map

    # Crop back to original size
    aggregated_prob = aggregated_prob[:, :vol_shape[0], :vol_shape[1], :vol_shape[2]]
    if evidential:
        aggregated_alpha = aggregated_alpha[:, :vol_shape[0], :vol_shape[1], :vol_shape[2]]

    # Normalize probabilities
    prob_sum = aggregated_prob.sum(axis=0, keepdims=True)
    prob_sum = np.clip(prob_sum, 1e-8, None)
    aggregated_prob /= prob_sum

    result = {
        "prob": aggregated_prob.astype(np.float32),
        "prediction": aggregated_prob.argmax(axis=0).astype(np.uint8),
    }

    # Uncertainty maps
    entropy = -(aggregated_prob * np.log(aggregated_prob + 1e-10)).sum(axis=0)
    result["entropy"] = entropy.astype(np.float32)

    if evidential:
        alpha = aggregated_alpha.astype(np.float32)
        S = alpha.sum(axis=0, keepdims=True)
        result["alpha"] = alpha
        result["aleatoric"] = entropy  # H(p_hat)
        result["epistemic"] = (num_classes / (S.squeeze(0) + 1)).astype(np.float32)

    return result


def _process_batch(
    model, patches, positions, patch_size, importance,
    aggregated_prob, weight_map, aggregated_alpha,
    aug, device, evidential,
):
    """Process a batch of patches through the model and aggregate."""
    # Stack into batch tensor: (B, 1, D, H, W)
    batch = np.stack(patches)[:, np.newaxis]
    batch_tensor = torch.from_numpy(batch).float().to(device)

    output = model(batch_tensor)
    probs = output["prob"].cpu().numpy()  # (B, C, D, H, W)

    if evidential and "alpha" in output:
        alphas = output["alpha"].cpu().numpy()

    for i, pos in enumerate(positions):
        d, h, w = pos
        pred = probs[i]  # (C, pD, pH, pW)

        # Reverse augmentation
        if aug is not None:
            for ax in reversed(aug):
                pred = np.flip(pred, axis=ax + 1).copy()  # +1 for class dim

        # Weighted accumulation
        for c in range(pred.shape[0]):
            aggregated_prob[c, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += \
                pred[c] * importance

        weight_map[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += importance

        if aggregated_alpha is not None and evidential:
            alpha_i = alphas[i]
            if aug is not None:
                for ax in reversed(aug):
                    alpha_i = np.flip(alpha_i, axis=ax + 1).copy()
            for c in range(alpha_i.shape[0]):
                aggregated_alpha[c, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += \
                    alpha_i[c] * importance


# ---------------------------------------------------------------------------
# 3D Dataset
# ---------------------------------------------------------------------------

class LIDC3DDataset(Dataset):
    """
    3D LIDC-IDRI dataset yielding volumetric patches with multi-annotator labels.

    Expects preprocessed HDF5 with structure:
        /case_{idx}/volume    -> (D, H, W) float32, resampled CT
        /case_{idx}/masks     -> (N_ann, D, H, W) uint8, all annotator masks
        /case_{idx}/spacing   -> (3,) float32, original spacing
        /case_{idx}/nodule_ids -> list of nodule identifiers in this volume
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        augment: bool = True,
        multi_annotator: bool = True,
        foreground_oversample: float = 0.67,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.augment = augment and (split == "train")
        self.multi_annotator = multi_annotator
        self.foreground_oversample = foreground_oversample

        self.h5_path = self.root / "lidc_3d.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"3D LIDC HDF5 not found at {self.h5_path}. "
                "Run 'python data/prepare_lidc_3d.py' first."
            )

        # Load case list
        split_file = self.root / f"{split}_cases.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.case_ids = [l.strip() for l in f if l.strip()]
        else:
            self.case_ids = self._auto_split()

        # For training, oversample by repeating each case proportional
        # to number of foreground voxels (nnU-Net convention)
        self.samples_per_epoch = max(250, len(self.case_ids) * 10) if split == "train" else len(self.case_ids)

        print(f"LIDC-3D {split}: {len(self.case_ids)} volumes, "
              f"{self.samples_per_epoch} samples/epoch, "
              f"patch {patch_size}")

    def _auto_split(self) -> List[str]:
        """Deterministic patient-level split."""
        with h5py.File(self.h5_path, "r") as f:
            all_ids = sorted(f.keys())

        rng = np.random.RandomState(42)
        rng.shuffle(all_ids)
        n = len(all_ids)
        t_end = int(0.7 * n)
        v_end = int(0.85 * n)

        if self.split == "train":
            return all_ids[:t_end]
        elif self.split == "val":
            return all_ids[t_end:v_end]
        else:
            return all_ids[v_end:]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Randomly sample a case (training) or sequential (val/test)
        if self.split == "train":
            case_id = self.case_ids[np.random.randint(len(self.case_ids))]
        else:
            case_id = self.case_ids[idx % len(self.case_ids)]

        with h5py.File(self.h5_path, "r") as f:
            volume = f[case_id]["volume"][:]       # (D, H, W) float32
            masks = f[case_id]["masks"][:]          # (N, D, H, W) uint8

        # Normalize
        volume = normalize_ct(volume, window=(-1000, 400))

        # Random 3D crop
        vol_patch, mask_patches = random_3d_crop(
            volume, masks, self.patch_size,
            foreground_prob=self.foreground_oversample if self.split == "train" else 0.5,
        )

        # Augment
        if self.augment:
            vol_patch, mask_patches = augment_3d(vol_patch, mask_patches)

        # Compute consensus and IOV
        prob_map = mask_patches.mean(axis=0)   # (D, H, W) in [0,1]
        iov_map = mask_patches.astype(np.float32).var(axis=0)
        label = (prob_map >= 0.5).astype(np.float32)

        # To tensors: add channel dim
        vol_tensor = torch.from_numpy(vol_patch[np.newaxis].copy()).float()   # (1, D, H, W)
        label_tensor = torch.from_numpy(label[np.newaxis].copy()).float()      # (1, D, H, W)
        prob_tensor = torch.from_numpy(prob_map[np.newaxis].copy()).float()
        iov_tensor = torch.from_numpy(iov_map[np.newaxis].copy()).float()

        out = {
            "image": vol_tensor,
            "label": label_tensor,
            "prob_map": prob_tensor,
            "iov_map": iov_tensor,
            "case_id": case_id,
        }

        if self.multi_annotator:
            out["annotator_masks"] = torch.from_numpy(
                mask_patches.astype(np.float32).copy()
            ).float()  # (N, D, H, W)

        return out


class LIDC3DInferenceDataset(Dataset):
    """
    Dataset that yields full 3D volumes for sliding-window inference.
    No cropping or augmentation; returns the entire resampled volume.
    """

    def __init__(self, root: str, split: str = "test"):
        self.root = Path(root)
        self.h5_path = self.root / "lidc_3d.h5"

        split_file = self.root / f"{split}_cases.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.case_ids = [l.strip() for l in f if l.strip()]
        else:
            with h5py.File(self.h5_path, "r") as f:
                self.case_ids = sorted(f.keys())

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict:
        case_id = self.case_ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            volume = f[case_id]["volume"][:]
            masks = f[case_id]["masks"][:]

        volume = normalize_ct(volume, window=(-1000, 400))
        prob_map = masks.astype(np.float32).mean(axis=0)
        iov_map = masks.astype(np.float32).var(axis=0)
        label = (prob_map >= 0.5).astype(np.float32)

        return {
            "volume": volume,
            "label": label,
            "prob_map": prob_map,
            "iov_map": iov_map,
            "masks": masks,
            "case_id": case_id,
        }

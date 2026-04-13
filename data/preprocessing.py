"""
Shared preprocessing utilities for medical image segmentation.
Handles intensity normalization, resampling, patching, and augmentation.
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates
from typing import Tuple, Optional, Dict, Any


def resample_image(
    image: np.ndarray,
    original_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    order: int = 3,
) -> np.ndarray:
    """Resample image to target spacing using scipy zoom."""
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    return zoom(image, zoom_factors, order=order)


def resample_label(
    label: np.ndarray,
    original_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
) -> np.ndarray:
    """Resample label map using nearest-neighbor interpolation."""
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    return zoom(label, zoom_factors, order=0)


def normalize_intensity(
    image: np.ndarray,
    method: str = "zscore",
    ct_window: Optional[Tuple[float, float]] = None,
    clip_percentile: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """
    Normalize image intensity.

    Args:
        image: Input image array.
        method: 'zscore', 'minmax', or 'ct_window'.
        ct_window: (min_hu, max_hu) for CT windowing.
        clip_percentile: Percentile clipping before normalization.
    """
    image = image.astype(np.float32)

    if method == "ct_window" and ct_window is not None:
        image = np.clip(image, ct_window[0], ct_window[1])
        image = (image - ct_window[0]) / (ct_window[1] - ct_window[0])
    elif method == "zscore":
        # Clip outliers
        low = np.percentile(image, clip_percentile[0])
        high = np.percentile(image, clip_percentile[1])
        image = np.clip(image, low, high)
        mean = image.mean()
        std = image.std() + 1e-8
        image = (image - mean) / std
    elif method == "minmax":
        low = np.percentile(image, clip_percentile[0])
        high = np.percentile(image, clip_percentile[1])
        image = np.clip(image, low, high)
        image = (image - low) / (high - low + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return image


def center_crop_or_pad(
    image: np.ndarray,
    target_size: Tuple[int, ...],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop or pad image to target size."""
    result = np.full(target_size, pad_value, dtype=image.dtype)
    # Compute crop/pad offsets for each dimension
    slices_src = []
    slices_dst = []
    for i in range(len(target_size)):
        src_size = image.shape[i]
        tgt_size = target_size[i]
        if src_size > tgt_size:
            # Crop
            start = (src_size - tgt_size) // 2
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(0, tgt_size))
        else:
            # Pad
            start = (tgt_size - src_size) // 2
            slices_src.append(slice(0, src_size))
            slices_dst.append(slice(start, start + src_size))

    result[tuple(slices_dst)] = image[tuple(slices_src)]
    return result


def random_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: Tuple[int, ...],
    foreground_prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random crop with optional foreground centering.

    Args:
        foreground_prob: Probability of centering crop on a foreground voxel.
    """
    ndim = image.ndim
    # Decide whether to center on foreground
    if np.random.random() < foreground_prob:
        fg_coords = np.argwhere(label > 0)
        if len(fg_coords) > 0:
            center = fg_coords[np.random.randint(len(fg_coords))]
        else:
            center = np.array([s // 2 for s in image.shape])
    else:
        center = np.array([
            np.random.randint(crop_size[i] // 2, max(image.shape[i] - crop_size[i] // 2, crop_size[i] // 2 + 1))
            for i in range(ndim)
        ])

    slices = []
    for i in range(ndim):
        start = max(0, center[i] - crop_size[i] // 2)
        start = min(start, max(0, image.shape[i] - crop_size[i]))
        end = start + crop_size[i]
        slices.append(slice(start, end))

    cropped_img = image[tuple(slices)]
    cropped_lbl = label[tuple(slices)]

    # Pad if needed
    cropped_img = center_crop_or_pad(cropped_img, crop_size, pad_value=cropped_img.min())
    cropped_lbl = center_crop_or_pad(cropped_lbl, crop_size, pad_value=0)

    return cropped_img, cropped_lbl


class DataAugmentation:
    """On-the-fly data augmentation for 2D medical image patches."""

    def __init__(self, config: Dict[str, Any]):
        self.flip = config.get("random_flip", True)
        self.rotate_deg = config.get("random_rotate", 0)
        self.scale_range = config.get("random_scale", None)
        self.elastic = config.get("elastic_deform", False)
        self.noise_std = config.get("gaussian_noise_std", 0.0)

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Random horizontal flip
        if self.flip and np.random.random() > 0.5:
            image = np.flip(image, axis=-1).copy()
            label = np.flip(label, axis=-1).copy()

        # Random vertical flip
        if self.flip and np.random.random() > 0.5:
            image = np.flip(image, axis=-2).copy()
            label = np.flip(label, axis=-2).copy()

        # Random rotation
        if self.rotate_deg > 0:
            angle = np.random.uniform(-self.rotate_deg, self.rotate_deg)
            image = rotate(image, angle, axes=(-2, -1), reshape=False, order=3, mode="nearest")
            label = rotate(label, angle, axes=(-2, -1), reshape=False, order=0, mode="nearest")

        # Random scaling
        if self.scale_range is not None:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            orig_shape = image.shape
            zoom_factors = [1.0] * (image.ndim - 2) + [scale, scale]
            image = zoom(image, zoom_factors, order=3)
            label = zoom(label, zoom_factors, order=0)
            image = center_crop_or_pad(image, orig_shape, pad_value=image.min())
            label = center_crop_or_pad(label, orig_shape, pad_value=0)

        # Elastic deformation (2D only)
        if self.elastic and image.ndim >= 2:
            image, label = self._elastic_deform(image, label)

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            image = image + noise

        return image, label

    @staticmethod
    def _elastic_deform(
        image: np.ndarray,
        label: np.ndarray,
        alpha: float = 100.0,
        sigma: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to 2D image and label."""
        shape = image.shape[-2:]
        dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        coords = [y + dy, x + dx]

        # Apply to image (cubic interpolation)
        if image.ndim == 2:
            image = map_coordinates(image, coords, order=3, mode="nearest")
        else:
            for c in range(image.shape[0]):
                image[c] = map_coordinates(image[c], coords, order=3, mode="nearest")

        # Apply to label (nearest neighbor)
        if label.ndim == 2:
            label = map_coordinates(label, coords, order=0, mode="nearest")
        else:
            for c in range(label.shape[0]):
                label[c] = map_coordinates(label[c], coords, order=0, mode="nearest")

        return image, label


def compute_inter_observer_variability(
    annotations: np.ndarray,
) -> np.ndarray:
    """
    Compute per-voxel inter-observer variability from multiple annotations.

    Args:
        annotations: Array of shape (num_annotators, H, W) or (num_annotators, D, H, W),
                     binary masks from each annotator.

    Returns:
        iov_map: Per-voxel variance across annotators, shape (H, W) or (D, H, W).
    """
    # Compute mean annotation probability
    mean_annotation = annotations.astype(np.float32).mean(axis=0)
    # Variance across annotators
    iov = annotations.astype(np.float32).var(axis=0)
    return iov


def compute_annotation_probability_map(
    annotations: np.ndarray,
) -> np.ndarray:
    """
    Compute per-voxel probability map from multiple binary annotations.

    p*(y=1 | x_i) = (1/M) * sum_{m=1}^{M} 1[y_i^(m) = 1]

    Args:
        annotations: (num_annotators, H, W) binary masks.

    Returns:
        prob_map: (H, W) probability of foreground.
    """
    return annotations.astype(np.float32).mean(axis=0)

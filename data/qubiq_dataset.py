"""
QUBIQ Dataset for multi-rater biomedical image segmentation.

Supports tasks: prostate (MRI), brain_growth (MRI), kidney (CT), brain_tumor (MRI).
Each case has 2D slices with multiple annotator segmentations.

Reference:
    Li et al., "QUBIQ: Uncertainty Quantification for Biomedical Image
    Segmentation Challenge," arXiv:2405.18435, 2024.
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from glob import glob

from .preprocessing import (
    normalize_intensity,
    center_crop_or_pad,
    DataAugmentation,
    compute_annotation_probability_map,
    compute_inter_observer_variability,
)


QUBIQ_TASKS = {
    "prostate": {"modality": "MRI", "num_annotators": 6, "label_tasks": 2},
    "brain_growth": {"modality": "MRI", "num_annotators": 7, "label_tasks": 1},
    "kidney": {"modality": "CT", "num_annotators": 3, "label_tasks": 1},
    "brain_tumor": {"modality": "MRI", "num_annotators": 3, "label_tasks": 3},
}


class QUBIQDataset(Dataset):
    """
    QUBIQ multi-rater segmentation dataset.

    Expects data organized as:
        root/
        ├── prostate/
        │   ├── case_001/
        │   │   ├── image.nii.gz
        │   │   ├── task01_annotator01.nii.gz
        │   │   ├── task01_annotator02.nii.gz
        │   │   └── ...
        │   └── ...
        ├── brain_growth/
        └── ...
    """

    def __init__(
        self,
        root: str,
        task: str = "prostate",
        label_task: int = 0,
        split: str = "train",
        patch_size: Tuple[int, int] = (256, 256),
        intensity_norm: str = "zscore",
        augmentation_config: Optional[Dict] = None,
        multi_annotator: bool = True,
    ):
        super().__init__()
        self.root = Path(root) / task
        self.task = task
        self.label_task = label_task
        self.split = split
        self.patch_size = patch_size
        self.intensity_norm = intensity_norm
        self.multi_annotator = multi_annotator

        if task not in QUBIQ_TASKS:
            raise ValueError(f"Unknown task: {task}. Choose from {list(QUBIQ_TASKS.keys())}")

        self.task_info = QUBIQ_TASKS[task]

        # Discover cases
        self.cases = self._load_cases()

        # Train/val/test split
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(self.cases))
        n = len(self.cases)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if split == "train":
            self.cases = [self.cases[i] for i in indices[:train_end]]
        elif split == "val":
            self.cases = [self.cases[i] for i in indices[train_end:val_end]]
        else:
            self.cases = [self.cases[i] for i in indices[val_end:]]

        # Augmentation
        self.augmentation = None
        if split == "train" and augmentation_config is not None:
            self.augmentation = DataAugmentation(augmentation_config)

        print(f"QUBIQ {task} {split}: {len(self.cases)} cases")

    def _load_cases(self) -> List[Dict]:
        """Discover and validate QUBIQ cases."""
        cases = []
        if not self.root.exists():
            raise FileNotFoundError(
                f"QUBIQ data not found at {self.root}. "
                "Download from https://qubiq.grand-challenge.org/participation/"
            )

        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue

            # Find image
            image_path = case_dir / "image.nii.gz"
            if not image_path.exists():
                # Try alternate naming
                nii_files = list(case_dir.glob("*image*.nii.gz"))
                if nii_files:
                    image_path = nii_files[0]
                else:
                    continue

            # Find annotator masks
            task_str = f"task{self.label_task + 1:02d}"
            mask_paths = sorted(case_dir.glob(f"{task_str}_annotator*.nii.gz"))
            if not mask_paths:
                # Try alternate naming
                mask_paths = sorted(case_dir.glob(f"*label*annotator*.nii.gz"))

            if len(mask_paths) >= 2:  # Need at least 2 annotators
                cases.append({
                    "case_id": case_dir.name,
                    "image_path": str(image_path),
                    "mask_paths": [str(p) for p in mask_paths],
                })

        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]

        # Load image
        img_nii = nib.load(case["image_path"])
        image = img_nii.get_fdata().astype(np.float32)

        # If 3D, take the stored 2D slice (QUBIQ provides 2D slices as 3D with D=1)
        if image.ndim == 3:
            if image.shape[-1] == 1:
                image = image[:, :, 0]
            else:
                # Take middle slice
                image = image[:, :, image.shape[-1] // 2]

        # Load all annotator masks
        masks = []
        for mp in case["mask_paths"]:
            mask_nii = nib.load(mp)
            mask = mask_nii.get_fdata().astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[:, :, 0] if mask.shape[-1] == 1 else mask[:, :, mask.shape[-1] // 2]
            masks.append(mask)
        masks = np.stack(masks, axis=0)  # (num_annotators, H, W)

        # Normalize intensity
        image = normalize_intensity(image, method=self.intensity_norm)

        # Compute consensus and IOV
        prob_map = compute_annotation_probability_map(masks)
        iov_map = compute_inter_observer_variability(masks)
        label = (prob_map >= 0.5).astype(np.float32)

        # Crop/pad
        image = center_crop_or_pad(image, self.patch_size, pad_value=image.min())
        label = center_crop_or_pad(label, self.patch_size, pad_value=0.0)
        prob_map = center_crop_or_pad(prob_map, self.patch_size, pad_value=0.0)
        iov_map = center_crop_or_pad(iov_map, self.patch_size, pad_value=0.0)

        # Augmentation
        if self.augmentation is not None:
            image, label = self.augmentation(image, label)

        # Channel dimension
        image = image[np.newaxis, ...]

        output = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).float().unsqueeze(0),
            "prob_map": torch.from_numpy(prob_map).float().unsqueeze(0),
            "iov_map": torch.from_numpy(iov_map).float().unsqueeze(0),
            "case_id": case["case_id"],
        }

        if self.multi_annotator:
            masks_padded = np.stack([
                center_crop_or_pad(m.astype(np.float32), self.patch_size, pad_value=0.0)
                for m in masks
            ])
            output["annotator_masks"] = torch.from_numpy(masks_padded).float()

        return output

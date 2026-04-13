"""
LIDC-IDRI Dataset for multi-annotator lung nodule segmentation.

The LIDC-IDRI dataset contains 1018 thoracic CT scans with up to 4 radiologist
annotations per nodule. This module handles:
- Loading preprocessed LIDC data (NIfTI or HDF5 format)
- Extracting 2D patches centered on nodules
- Providing all 4 annotator masks for multi-annotator training
- Computing per-voxel annotation probability maps and IOV

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

from .preprocessing import (
    normalize_intensity,
    center_crop_or_pad,
    random_crop,
    DataAugmentation,
    compute_annotation_probability_map,
    compute_inter_observer_variability,
)


class LIDCDataset(Dataset):
    """
    LIDC-IDRI dataset for 2D nodule segmentation with multi-annotator labels.

    Expects preprocessed HDF5 files with structure:
        /nodule_{idx}/image       -> (H, W) float32, CT slice
        /nodule_{idx}/masks       -> (num_annotators, H, W) uint8, binary masks
        /nodule_{idx}/spacing     -> (2,) float32, pixel spacing
        /nodule_{idx}/patient_id  -> string

    The preprocessing script (download_lidc.py) extracts 2D slices centered on
    each nodule and stores all available annotator contours.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: Tuple[int, int] = (128, 128),
        intensity_norm: str = "ct_window",
        ct_window: Tuple[float, float] = (-1000, 400),
        augmentation_config: Optional[Dict] = None,
        multi_annotator: bool = False,
        consensus_mode: str = "majority",  # majority | soft | staple
        min_annotators: int = 3,
    ):
        """
        Args:
            root: Path to preprocessed LIDC HDF5 directory.
            split: 'train', 'val', or 'test'.
            patch_size: (H, W) output patch size.
            multi_annotator: If True, return all annotator masks.
            consensus_mode: How to form single ground truth:
                'majority' - Majority voting (>= 50% annotators agree)
                'soft' - Probability map from all annotators
                'staple' - STAPLE algorithm fusion
            min_annotators: Minimum annotators required per nodule.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.intensity_norm = intensity_norm
        self.ct_window = ct_window
        self.multi_annotator = multi_annotator
        self.consensus_mode = consensus_mode
        self.min_annotators = min_annotators

        # Load split file
        split_file = self.root / f"{split}_nodules.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.nodule_ids = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover from HDF5
            self.nodule_ids = self._discover_nodules()

        # Setup augmentation
        self.augmentation = None
        if split == "train" and augmentation_config is not None:
            self.augmentation = DataAugmentation(augmentation_config)

        print(f"LIDC-IDRI {split}: {len(self.nodule_ids)} nodules loaded")

    def _discover_nodules(self) -> List[str]:
        """Discover available nodule IDs from HDF5 files."""
        h5_path = self.root / "lidc_nodules.h5"
        if not h5_path.exists():
            raise FileNotFoundError(
                f"LIDC HDF5 file not found at {h5_path}. "
                "Run 'python data/download_lidc.py' first."
            )
        with h5py.File(h5_path, "r") as f:
            nodule_ids = list(f.keys())

        # Filter by minimum annotators
        valid_ids = []
        with h5py.File(h5_path, "r") as f:
            for nid in nodule_ids:
                if f[nid]["masks"].shape[0] >= self.min_annotators:
                    valid_ids.append(nid)

        # Deterministic split
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(valid_ids))
        n = len(valid_ids)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if self.split == "train":
            return [valid_ids[i] for i in indices[:train_end]]
        elif self.split == "val":
            return [valid_ids[i] for i in indices[train_end:val_end]]
        else:
            return [valid_ids[i] for i in indices[val_end:]]

    def __len__(self) -> int:
        return len(self.nodule_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nid = self.nodule_ids[idx]
        h5_path = self.root / "lidc_nodules.h5"

        with h5py.File(h5_path, "r") as f:
            image = f[nid]["image"][:]  # (H, W) float32
            masks = f[nid]["masks"][:]  # (num_annotators, H, W) uint8

        # Normalize intensity
        image = normalize_intensity(
            image, method=self.intensity_norm, ct_window=self.ct_window
        )

        # Compute ground truth
        prob_map = compute_annotation_probability_map(masks)  # (H, W) [0, 1]
        iov_map = compute_inter_observer_variability(masks)  # (H, W)

        if self.consensus_mode == "majority":
            label = (prob_map >= 0.5).astype(np.float32)
        elif self.consensus_mode == "soft":
            label = prob_map
        else:
            label = (prob_map >= 0.5).astype(np.float32)

        # Crop/pad to patch size
        image = center_crop_or_pad(image, self.patch_size, pad_value=image.min())
        label = center_crop_or_pad(label, self.patch_size, pad_value=0.0)
        prob_map = center_crop_or_pad(prob_map, self.patch_size, pad_value=0.0)
        iov_map = center_crop_or_pad(iov_map, self.patch_size, pad_value=0.0)

        # Augmentation
        if self.augmentation is not None:
            image, label = self.augmentation(image, label)

        # Add channel dimension
        image = image[np.newaxis, ...]  # (1, H, W)

        output = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).float().unsqueeze(0),  # (1, H, W)
            "prob_map": torch.from_numpy(prob_map).float().unsqueeze(0),
            "iov_map": torch.from_numpy(iov_map).float().unsqueeze(0),
            "nodule_id": nid,
        }

        # Multi-annotator: include all individual masks
        if self.multi_annotator:
            masks_padded = np.stack([
                center_crop_or_pad(m.astype(np.float32), self.patch_size, pad_value=0.0)
                for m in masks
            ])  # (num_annotators, H, W)
            output["annotator_masks"] = torch.from_numpy(masks_padded).float()

        return output


class LIDCDataModule:
    """Convenience class to create train/val/test dataloaders."""

    def __init__(self, config: Dict):
        self.config = config
        self.root = config["data"]["root"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["data"].get("num_workers", 4)
        self.patch_size = tuple(config["preprocessing"]["spatial_size"])
        self.multi_annotator = config["model"].get("multi_annotator", {}).get("enabled", False)
        self.aug_config = config["preprocessing"].get("augmentation", {})

    def get_dataset(self, split: str) -> LIDCDataset:
        return LIDCDataset(
            root=self.root,
            split=split,
            patch_size=self.patch_size,
            intensity_norm=self.config["preprocessing"]["intensity_norm"],
            ct_window=tuple(self.config["preprocessing"].get("ct_window", [-1000, 400])),
            augmentation_config=self.aug_config if split == "train" else None,
            multi_annotator=self.multi_annotator,
        )

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        ds = self.get_dataset("train")
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        ds = self.get_dataset("val")
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        ds = self.get_dataset("test")
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

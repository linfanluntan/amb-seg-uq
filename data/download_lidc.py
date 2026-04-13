#!/usr/bin/env python3
"""
Download and preprocess LIDC-IDRI dataset for multi-annotator segmentation.

This script:
1. Uses pylidc to query the LIDC-IDRI database
2. Extracts 2D slices containing annotated nodules (>= 3mm, >= 3 annotators)
3. Stores all annotator contours as separate binary masks
4. Saves everything in HDF5 format for fast loading

Prerequisites:
    - Install pylidc: pip install pylidc
    - Download LIDC-IDRI DICOM data from TCIA:
      https://www.cancerimagingarchive.net/collection/lidc-idri/
    - Configure pylidc: create ~/.pylidcrc with path to DICOM data

Usage:
    python data/download_lidc.py --dicom_dir /path/to/LIDC-IDRI --output_dir ./data/lidc
"""

import argparse
import os
import sys
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

try:
    import pylidc as pl
    from pylidc.utils import consensus
    HAS_PYLIDC = True
except ImportError:
    HAS_PYLIDC = False


def extract_nodule_slices(
    dicom_dir: str,
    output_dir: str,
    min_annotators: int = 3,
    min_diameter_mm: float = 3.0,
    patch_margin: int = 32,
    verbose: bool = True,
):
    """
    Extract 2D slices and multi-annotator masks from LIDC-IDRI.

    For each nodule with >= min_annotators annotations:
    - Extract the central axial slice
    - Crop a patch around the nodule with margin
    - Store the CT image and all annotator binary masks

    Args:
        dicom_dir: Root directory of LIDC-IDRI DICOM data.
        output_dir: Where to save processed HDF5 file.
        min_annotators: Minimum number of annotators per nodule.
        min_diameter_mm: Minimum nodule diameter to include.
        patch_margin: Extra margin (pixels) around nodule bounding box.
    """
    if not HAS_PYLIDC:
        print("ERROR: pylidc not installed. Install with: pip install pylidc")
        print("Also ensure LIDC-IDRI DICOM data is downloaded and ~/.pylidcrc is configured.")
        print("\nExample ~/.pylidcrc:")
        print("[dicom]")
        print(f"path = {dicom_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "lidc_nodules.h5")

    # Query all scans
    scans = pl.query(pl.Scan).all()
    if verbose:
        print(f"Found {len(scans)} LIDC-IDRI scans")

    nodule_count = 0
    patient_ids = set()

    with h5py.File(h5_path, "w") as h5f:
        for scan in tqdm(scans, desc="Processing scans"):
            try:
                patient_id = scan.patient_id
                nods = scan.cluster_annotations()

                for nod_idx, anns in enumerate(nods):
                    if len(anns) < min_annotators:
                        continue

                    # Check diameter
                    diameters = [a.diameter for a in anns if a.diameter is not None]
                    if not diameters or np.mean(diameters) < min_diameter_mm:
                        continue

                    # Get consensus and individual masks
                    try:
                        cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=patch_margin)
                    except Exception:
                        continue

                    if cmask.sum() == 0:
                        continue

                    # Get the CT volume
                    vol = scan.to_volume()
                    spacing = scan.pixel_spacing  # in-plane spacing

                    # Extract central slice
                    z_center = cmask.shape[2] // 2
                    image_slice = vol[
                        cbbox[0].start:cbbox[0].stop,
                        cbbox[1].start:cbbox[1].stop,
                        cbbox[2].start + z_center
                    ].astype(np.float32)

                    # Extract each annotator's mask at central slice
                    annotator_masks = []
                    for ann in anns:
                        try:
                            mask_bool = ann.boolean_mask(pad=patch_margin)
                            # Get corresponding slice
                            ann_bbox = ann.bbox(pad=patch_margin)
                            ann_slice = mask_bool[:, :, mask_bool.shape[2] // 2].astype(np.uint8)

                            # Ensure same shape as image
                            if ann_slice.shape == image_slice.shape:
                                annotator_masks.append(ann_slice)
                        except Exception:
                            continue

                    if len(annotator_masks) < min_annotators:
                        continue

                    annotator_masks = np.stack(annotator_masks, axis=0)  # (N_ann, H, W)

                    # Save to HDF5
                    key = f"nodule_{nodule_count:05d}"
                    grp = h5f.create_group(key)
                    grp.create_dataset("image", data=image_slice, compression="gzip")
                    grp.create_dataset("masks", data=annotator_masks, compression="gzip")
                    grp.create_dataset("spacing", data=np.array([spacing, spacing], dtype=np.float32))
                    grp.attrs["patient_id"] = patient_id
                    grp.attrs["num_annotators"] = len(annotator_masks)
                    grp.attrs["mean_diameter_mm"] = float(np.mean(diameters))

                    nodule_count += 1
                    patient_ids.add(patient_id)

            except Exception as e:
                if verbose:
                    print(f"Warning: Error processing {scan.patient_id}: {e}")
                continue

    if verbose:
        print(f"\nExtracted {nodule_count} nodule patches from {len(patient_ids)} patients")
        print(f"Saved to {h5_path}")

    # Create train/val/test splits (patient-level, no leakage)
    create_splits(h5_path, output_dir)


def create_splits(h5_path: str, output_dir: str, seed: int = 42):
    """Create patient-level train/val/test splits to avoid data leakage."""
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, "r") as f:
        nodule_ids = list(f.keys())
        patient_map = {}
        for nid in nodule_ids:
            pid = f[nid].attrs["patient_id"]
            if pid not in patient_map:
                patient_map[pid] = []
            patient_map[pid].append(nid)

    patients = sorted(patient_map.keys())
    rng.shuffle(patients)
    n = len(patients)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "train": patients[:train_end],
        "val": patients[train_end:val_end],
        "test": patients[val_end:],
    }

    for split_name, split_patients in splits.items():
        nodules = []
        for pid in split_patients:
            nodules.extend(patient_map[pid])
        with open(os.path.join(output_dir, f"{split_name}_nodules.txt"), "w") as f:
            for nid in sorted(nodules):
                f.write(nid + "\n")
        print(f"  {split_name}: {len(split_patients)} patients, {len(nodules)} nodules")


def create_synthetic_lidc(output_dir: str, num_samples: int = 500):
    """
    Create synthetic LIDC-like data for testing the pipeline.
    Generates random nodule-like patches with multiple 'annotator' masks
    that have controlled inter-observer variability.
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "lidc_nodules.h5")

    rng = np.random.RandomState(42)
    H, W = 80, 80

    with h5py.File(h5_path, "w") as h5f:
        for i in tqdm(range(num_samples), desc="Generating synthetic data"):
            # Create a nodule-like ellipse
            cy, cx = H // 2 + rng.randint(-10, 10), W // 2 + rng.randint(-10, 10)
            ry, rx = rng.randint(8, 25), rng.randint(8, 25)

            yy, xx = np.mgrid[:H, :W]
            base_mask = ((yy - cy)**2 / ry**2 + (xx - cx)**2 / rx**2) < 1.0

            # Simulate CT values
            image = rng.normal(-800, 100, (H, W)).astype(np.float32)
            image[base_mask] = rng.normal(-200, 80, base_mask.sum()).astype(np.float32)
            image = np.clip(image, -1024, 400)

            # Create 4 annotator masks with variability
            masks = []
            for a in range(4):
                # Perturb ellipse parameters
                perturb_cy = cy + rng.randint(-3, 4)
                perturb_cx = cx + rng.randint(-3, 4)
                perturb_ry = ry + rng.randint(-3, 4)
                perturb_rx = rx + rng.randint(-3, 4)
                ann_mask = (
                    (yy - perturb_cy)**2 / max(perturb_ry, 3)**2 +
                    (xx - perturb_cx)**2 / max(perturb_rx, 3)**2
                ) < 1.0
                masks.append(ann_mask.astype(np.uint8))

            masks = np.stack(masks, axis=0)

            key = f"nodule_{i:05d}"
            grp = h5f.create_group(key)
            grp.create_dataset("image", data=image, compression="gzip")
            grp.create_dataset("masks", data=masks, compression="gzip")
            grp.create_dataset("spacing", data=np.array([0.7, 0.7], dtype=np.float32))
            grp.attrs["patient_id"] = f"SYNTH-{i // 5:04d}"
            grp.attrs["num_annotators"] = 4
            grp.attrs["mean_diameter_mm"] = float(ry + rx)

    # Create splits
    create_splits(h5_path, output_dir)
    print(f"Synthetic LIDC data created at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/preprocess LIDC-IDRI dataset")
    parser.add_argument("--dicom_dir", type=str, default=None,
                        help="Path to LIDC-IDRI DICOM data")
    parser.add_argument("--output_dir", type=str, default="./data/lidc",
                        help="Output directory for preprocessed data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for pipeline testing")
    parser.add_argument("--num_synthetic", type=int, default=500,
                        help="Number of synthetic samples")
    parser.add_argument("--min_annotators", type=int, default=3)
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_lidc(args.output_dir, args.num_synthetic)
    elif args.dicom_dir:
        extract_nodule_slices(
            args.dicom_dir, args.output_dir,
            min_annotators=args.min_annotators,
        )
    else:
        print("Specify --dicom_dir for real data or --synthetic for test data")

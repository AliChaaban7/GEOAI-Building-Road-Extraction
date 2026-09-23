"""
instance_dataset.py

Dataset utilities for Mask R-CNN / instance segmentation.

This file reads image-mask pairs and converts each mask into:
- boxes
- labels
- instance masks
- image_id
- area
- iscrowd

Expected mask behavior:
- background = 0
- building pixels > 0

If the mask has unique instance IDs, each ID becomes one building.
If the mask is binary, connected components are used to separate buildings.

Important:
- Uses PIL first.
- If PIL cannot read an ArcGIS TIFF, it falls back to ArcPy.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
MASK_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


# ============================================================
# Raster reading helpers
# ============================================================

def _read_with_arcpy(path):
    """
    Read raster using ArcPy fallback.

    This is useful for ArcGIS-exported TIFF files that PIL cannot identify.
    """

    import arcpy

    arr = arcpy.RasterToNumPyArray(
        str(path),
        nodata_to_value=0
    )

    return arr


def _bands_first_to_hwc(arr):
    """
    Convert ArcPy multiband output [bands, rows, cols] to [rows, cols, bands].
    """

    if arr.ndim == 3:
        if arr.shape[0] <= 10 and arr.shape[1] > 10 and arr.shape[2] > 10:
            arr = np.moveaxis(arr, 0, -1)

    return arr


def _safe_normalize_image(arr):
    arr = np.asarray(arr).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    arr = _bands_first_to_hwc(arr)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    max_value = float(arr.max()) if arr.size > 0 else 0.0

    if max_value > 1.0:
        arr = np.clip(arr, 0, 255) / 255.0

    arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr).float()


def read_image_as_tensor(image_path):
    """
    Read image as torch tensor [3, H, W].

    Uses:
    1. PIL
    2. ArcPy fallback
    """

    image_path = Path(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image)

    except Exception:
        arr = _read_with_arcpy(image_path)

    return _safe_normalize_image(arr)


def read_mask_array(mask_path):
    """
    Read mask as numpy array [H, W].

    Uses:
    1. PIL
    2. ArcPy fallback

    This fixes:
    PIL.UnidentifiedImageError: cannot identify image file ...
    """

    mask_path = Path(mask_path)

    try:
        mask = Image.open(mask_path)
        arr = np.array(mask)

    except (UnidentifiedImageError, OSError, ValueError):
        arr = _read_with_arcpy(mask_path)

    except Exception:
        arr = _read_with_arcpy(mask_path)

    arr = np.asarray(arr)

    if arr.ndim == 3:
        # ArcPy may return [bands, rows, cols]
        if arr.shape[0] <= 10 and arr.shape[1] > 10 and arr.shape[2] > 10:
            arr = arr[0, :, :]
        else:
            arr = arr[:, :, 0]

    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    arr = arr.astype(np.int32)

    return arr


# ============================================================
# Instance mask conversion
# ============================================================

def connected_components(binary_mask):
    """
    Connected components for binary masks.

    Uses scipy if available.
    Falls back to one whole foreground object if scipy is unavailable.
    """

    binary_mask = binary_mask.astype(np.uint8)

    try:
        from scipy import ndimage

        labeled, num_features = ndimage.label(binary_mask)

        return labeled.astype(np.int32), int(num_features)

    except Exception:
        labeled = np.zeros_like(binary_mask, dtype=np.int32)

        if binary_mask.sum() > 0:
            labeled[binary_mask > 0] = 1
            return labeled, 1

        return labeled, 0


def mask_to_instances(mask_array, min_object_pixels=5):
    """
    Convert a mask array into instance masks and boxes.

    Returns:
        masks: numpy array [N, H, W]
        boxes: numpy array [N, 4]
    """

    mask_array = np.asarray(mask_array)

    foreground = mask_array > 0

    if foreground.sum() == 0:
        h, w = mask_array.shape
        return (
            np.zeros((0, h, w), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32)
        )

    unique_values = np.unique(mask_array)
    unique_values = unique_values[unique_values > 0]

    # If the mask is binary, split using connected components.
    if len(unique_values) <= 1:
        instance_map, num_instances = connected_components(foreground)
        instance_ids = list(range(1, num_instances + 1))
    else:
        instance_map = mask_array
        instance_ids = unique_values.tolist()

    instance_masks = []
    boxes = []

    for instance_id in instance_ids:
        instance_mask = (instance_map == instance_id).astype(np.uint8)

        if int(instance_mask.sum()) < int(min_object_pixels):
            continue

        ys, xs = np.where(instance_mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            continue

        xmin = float(xs.min())
        xmax = float(xs.max())
        ymin = float(ys.min())
        ymax = float(ys.max())

        if xmax <= xmin or ymax <= ymin:
            continue

        instance_masks.append(instance_mask)
        boxes.append([xmin, ymin, xmax, ymax])

    h, w = mask_array.shape

    if len(instance_masks) == 0:
        return (
            np.zeros((0, h, w), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32)
        )

    return (
        np.stack(instance_masks).astype(np.uint8),
        np.array(boxes, dtype=np.float32)
    )


def create_target_from_mask(mask_path, image_id, min_object_pixels=5):
    mask_array = read_mask_array(mask_path)

    masks, boxes = mask_to_instances(
        mask_array=mask_array,
        min_object_pixels=min_object_pixels
    )

    num_objects = masks.shape[0]

    labels = np.ones((num_objects,), dtype=np.int64)
    iscrowd = np.zeros((num_objects,), dtype=np.int64)

    if num_objects > 0:
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
        area = np.zeros((0,), dtype=np.float32)

    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(masks, dtype=torch.uint8),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": torch.as_tensor(area, dtype=torch.float32),
        "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
    }

    return target


# ============================================================
# Dataset class
# ============================================================

class BuildingMaskRCNNDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        min_object_pixels=5
    ):
        self.manifest_csv = Path(manifest_csv)
        self.min_object_pixels = int(min_object_pixels)

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        self.df = pd.read_csv(self.manifest_csv)

        required_columns = ["image_path", "mask_path"]

        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(
                    f"Manifest must contain column '{col}'. "
                    f"Columns found: {list(self.df.columns)}"
                )

        self.df = self.df.dropna(
            subset=["image_path", "mask_path"]
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        try:
            image = read_image_as_tensor(image_path)

            target = create_target_from_mask(
                mask_path=mask_path,
                image_id=idx,
                min_object_pixels=self.min_object_pixels
            )

        except Exception as e:
            raise RuntimeError(
                "Failed to read Mask R-CNN training sample.\n"
                f"Index: {idx}\n"
                f"Image: {image_path}\n"
                f"Mask: {mask_path}\n"
                f"Error: {e}"
            )

        return image, target


def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


# ============================================================
# Manifest creation
# ============================================================

def list_files_recursively(folder, extensions):
    folder = Path(folder)

    files = []

    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))

    return sorted(files)


def looks_like_mask_path(path):
    text = str(path).lower()

    mask_keywords = [
        "mask",
        "masks",
        "label",
        "labels",
        "annotation",
        "annotations",
        "groundtruth",
        "ground_truth"
    ]

    return any(keyword in text for keyword in mask_keywords)


def looks_like_image_path(path):
    text = str(path).lower()

    image_keywords = [
        "image",
        "images",
        "img",
        "chips",
        "raster"
    ]

    return any(keyword in text for keyword in image_keywords)


def normalize_stem(path):
    stem = Path(path).stem.lower()

    remove_tokens = [
        "_mask",
        "_label",
        "_labels",
        "_annotation",
        "_annotations",
        "_groundtruth",
        "_ground_truth"
    ]

    for token in remove_tokens:
        stem = stem.replace(token, "")

    return stem


def discover_image_mask_pairs(dataset_path):
    """
    Discover image-mask pairs from an ArcGIS exported dataset.

    This is intentionally flexible because ArcGIS export folder structures
    can differ depending on export format.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    all_files = list_files_recursively(dataset_path, IMAGE_EXTENSIONS)

    image_candidates = []
    mask_candidates = []

    for file_path in all_files:
        if looks_like_mask_path(file_path):
            mask_candidates.append(file_path)
        elif looks_like_image_path(file_path):
            image_candidates.append(file_path)

    # Fallback: split by folder name if keyword detection was weak.
    if len(image_candidates) == 0 or len(mask_candidates) == 0:
        image_candidates = []
        mask_candidates = []

        for file_path in all_files:
            parent_text = str(file_path.parent).lower()

            if any(k in parent_text for k in ["label", "labels", "mask", "masks"]):
                mask_candidates.append(file_path)
            else:
                image_candidates.append(file_path)

    mask_by_stem = {}

    for mask_path in mask_candidates:
        mask_by_stem[normalize_stem(mask_path)] = mask_path

    pairs = []

    for image_path in image_candidates:
        key = normalize_stem(image_path)

        if key in mask_by_stem:
            pairs.append(
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_by_stem[key]),
                    "image_name": image_path.name,
                    "mask_name": mask_by_stem[key].name
                }
            )

    pairs = sorted(pairs, key=lambda x: x["image_path"])

    return pairs


def validate_pair_readable(pair):
    """
    Check that both image and mask can be read.

    Uses the same readers as training.
    """

    try:
        _ = read_image_as_tensor(pair["image_path"])
        _ = read_mask_array(pair["mask_path"])
        return True, ""

    except Exception as e:
        return False, str(e)


def create_train_val_manifests(
    dataset_path,
    output_folder,
    validation_ratio=0.2,
    seed=42,
    validate_readability=True
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pairs = discover_image_mask_pairs(dataset_path)

    if len(pairs) == 0:
        raise ValueError(
            "No image-mask pairs found.\n"
            f"Dataset path: {dataset_path}\n"
            "Check the RCNN dataset folder structure."
        )

    if validate_readability:
        valid_pairs = []
        bad_rows = []

        print("Checking readability of RCNN image-mask pairs...")

        for pair in pairs:
            ok, error = validate_pair_readable(pair)

            if ok:
                valid_pairs.append(pair)
            else:
                bad_pair = dict(pair)
                bad_pair["error"] = error
                bad_rows.append(bad_pair)

        if len(bad_rows) > 0:
            bad_csv = output_folder / "unreadable_pairs.csv"
            pd.DataFrame(bad_rows).to_csv(bad_csv, index=False)

            print(f"Warning: skipped unreadable pairs: {len(bad_rows)}")
            print(f"Unreadable pairs saved to: {bad_csv}")

        pairs = valid_pairs

    if len(pairs) == 0:
        raise ValueError(
            "All RCNN image-mask pairs are unreadable.\n"
            f"Dataset path: {dataset_path}"
        )

    random.seed(seed)
    random.shuffle(pairs)

    val_count = max(1, int(len(pairs) * float(validation_ratio)))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    train_csv = output_folder / "train_manifest.csv"
    val_csv = output_folder / "val_manifest.csv"
    all_csv = output_folder / "all_manifest.csv"

    pd.DataFrame(train_pairs).to_csv(train_csv, index=False)
    pd.DataFrame(val_pairs).to_csv(val_csv, index=False)
    pd.DataFrame(pairs).to_csv(all_csv, index=False)

    return {
        "pair_count": len(pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "all_csv": str(all_csv)
    }
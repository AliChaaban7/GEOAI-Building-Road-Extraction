"""
Mask R-CNN data utilities.

Contains:
- manifest reading
- robust RGB/mask raster loading
- binary-mask -> instance-target conversion
- Dataset / DataLoader construction

This code is moved from the historical shared Mask R-CNN engine without
changing its data behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from scipy import ndimage
except Exception as exc:
    raise ImportError(
        "scipy is required for Mask R-CNN connected-component handling."
    ) from exc


def _infer_manifest_columns(
    df: pd.DataFrame,
) -> Tuple[str, str]:
    image_candidates = (
        "image_path",
        "image",
        "image_file",
        "image_filepath",
        "chip_path",
        "raster_path",
    )

    label_candidates = (
        "label_path",
        "mask_path",
        "mask",
        "label",
        "label_file",
        "mask_filepath",
    )

    image_col = next(
        (
            c
            for c in image_candidates
            if c in df.columns
        ),
        None,
    )

    label_col = next(
        (
            c
            for c in label_candidates
            if c in df.columns
        ),
        None,
    )

    if (
        image_col is None
        or label_col is None
    ):
        raise ValueError(
            "Could not detect image/label columns in manifest. "
            f"Available columns: {list(df.columns)}"
        )

    return image_col, label_col


def read_manifest_pairs(
    csv_path: Path,
) -> List[Tuple[Path, Path]]:
    csv_path = Path(
        csv_path
    )

    df = pd.read_csv(
        csv_path
    )

    image_col, label_col = (
        _infer_manifest_columns(
            df
        )
    )

    pairs: List[
        Tuple[
            Path,
            Path,
        ]
    ] = []

    for _, row in df.iterrows():
        img = Path(
            str(
                row[
                    image_col
                ]
            )
        )

        lbl = Path(
            str(
                row[
                    label_col
                ]
            )
        )

        if not img.exists():
            raise FileNotFoundError(
                f"Manifest image not found: {img}"
            )

        if not lbl.exists():
            raise FileNotFoundError(
                f"Manifest label not found: {lbl}"
            )

        pairs.append(
            (
                img,
                lbl,
            )
        )

    if not pairs:
        raise RuntimeError(
            f"No image-label pairs found in {csv_path}"
        )

    return pairs


def _read_raster_with_fallback(
    path: Path,
    is_mask: bool = False,
) -> np.ndarray:
    path = Path(
        path
    )

    errors = []

    try:
        with Image.open(
            path
        ) as img:
            if is_mask:
                return np.asarray(
                    img
                )

            return np.asarray(
                img.convert(
                    "RGB"
                )
            )

    except Exception as exc:
        errors.append(
            f"Pillow: {exc}"
        )

    try:
        import rasterio

        with rasterio.open(
            path
        ) as src:
            arr = src.read()

        if is_mask:
            return arr[0]

        if arr.shape[0] >= 3:
            return np.moveaxis(
                arr[:3],
                0,
                -1,
            )

    except Exception as exc:
        errors.append(
            f"rasterio: {exc}"
        )

    try:
        import arcpy

        arr = (
            arcpy.RasterToNumPyArray(
                str(
                    path
                )
            )
        )

        arr = np.asarray(
            arr
        )

        if is_mask:
            if arr.ndim == 3:
                return arr[0]

            return arr

        if arr.ndim == 3:
            if arr.shape[0] >= 3:
                return np.moveaxis(
                    arr[:3],
                    0,
                    -1,
                )

        elif arr.ndim == 2:
            return np.repeat(
                arr[
                    :,
                    :,
                    None,
                ],
                3,
                axis=2,
            )

    except Exception as exc:
        errors.append(
            f"ArcPy: {exc}"
        )

    raise RuntimeError(
        f"Unable to read raster: {path}\n"
        + "\n".join(
            errors
        )
    )


def read_rgb_image(
    path: Path,
) -> torch.Tensor:
    arr = (
        _read_raster_with_fallback(
            path,
            is_mask=False,
        )
    )

    arr = np.asarray(
        arr
    ).astype(
        np.float32
    )

    if arr.ndim == 2:
        arr = np.repeat(
            arr[
                :,
                :,
                None,
            ],
            3,
            axis=2,
        )

    if arr.shape[-1] > 3:
        arr = arr[
            :,
            :,
            :3,
        ]

    if arr.max() > 1.0:
        if arr.max() <= 255.0:
            arr /= 255.0

        else:
            out = np.zeros_like(
                arr,
                dtype=np.float32,
            )

            for c in range(
                arr.shape[2]
            ):
                band = arr[
                    :,
                    :,
                    c,
                ]

                lo, hi = np.percentile(
                    band,
                    [
                        2,
                        98,
                    ],
                )

                if hi > lo:
                    out[
                        :,
                        :,
                        c,
                    ] = np.clip(
                        (
                            band
                            - lo
                        )
                        / (
                            hi
                            - lo
                        ),
                        0,
                        1,
                    )

            arr = out

    return (
        torch.from_numpy(
            arr
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        .contiguous()
    )


def read_binary_mask(
    path: Path,
) -> np.ndarray:
    arr = (
        _read_raster_with_fallback(
            path,
            is_mask=True,
        )
    )

    arr = np.asarray(
        arr
    )

    if arr.ndim == 3:
        arr = (
            arr[0]
            if arr.shape[0] <= 10
            else arr[
                ...,
                0,
            ]
        )

    arr = np.nan_to_num(
        arr,
        nan=0,
    )

    return (
        arr > 0
    ).astype(
        np.uint8
    )


def binary_mask_to_target(
    mask_array: np.ndarray,
    image_id: int,
    min_component_pixels: int = 5,
) -> Dict[str, torch.Tensor]:
    labeled, num = (
        ndimage.label(
            mask_array > 0
        )
    )

    boxes: List[
        List[
            float
        ]
    ] = []

    instance_masks: List[
        np.ndarray
    ] = []

    for component_id in range(
        1,
        num + 1,
    ):
        comp = (
            labeled
            == component_id
        )

        if int(
            comp.sum()
        ) < int(
            min_component_pixels
        ):
            continue

        ys, xs = np.where(
            comp
        )

        if len(
            xs
        ) == 0:
            continue

        xmin = int(
            xs.min()
        )

        xmax = int(
            xs.max()
        )

        ymin = int(
            ys.min()
        )

        ymax = int(
            ys.max()
        )

        if (
            xmax <= xmin
            or ymax <= ymin
        ):
            continue

        boxes.append(
            [
                xmin,
                ymin,
                xmax,
                ymax,
            ]
        )

        instance_masks.append(
            comp.astype(
                np.uint8
            )
        )

    h, w = (
        mask_array.shape
    )

    if boxes:
        boxes_t = torch.tensor(
            boxes,
            dtype=torch.float32,
        )

        masks_t = (
            torch.from_numpy(
                np.stack(
                    instance_masks
                )
            )
            .to(
                torch.uint8
            )
        )

        labels_t = torch.ones(
            (
                len(
                    boxes
                ),
            ),
            dtype=torch.int64,
        )

        area_t = (
            (
                boxes_t[
                    :,
                    2,
                ]
                - boxes_t[
                    :,
                    0,
                ]
            )
            * (
                boxes_t[
                    :,
                    3,
                ]
                - boxes_t[
                    :,
                    1,
                ]
            )
        )

        iscrowd_t = torch.zeros(
            (
                len(
                    boxes
                ),
            ),
            dtype=torch.int64,
        )

    else:
        boxes_t = torch.zeros(
            (
                0,
                4,
            ),
            dtype=torch.float32,
        )

        masks_t = torch.zeros(
            (
                0,
                h,
                w,
            ),
            dtype=torch.uint8,
        )

        labels_t = torch.zeros(
            (
                0,
            ),
            dtype=torch.int64,
        )

        area_t = torch.zeros(
            (
                0,
            ),
            dtype=torch.float32,
        )

        iscrowd_t = torch.zeros(
            (
                0,
            ),
            dtype=torch.int64,
        )

    return {
        "boxes":
            boxes_t,

        "labels":
            labels_t,

        "masks":
            masks_t,

        "image_id":
            torch.tensor(
                [
                    image_id
                ],
                dtype=torch.int64,
            ),

        "area":
            area_t,

        "iscrowd":
            iscrowd_t,
    }


class MaskRCNNManifestDataset(
    Dataset
):
    def __init__(
        self,
        pairs: Sequence[
            Tuple[
                Path,
                Path,
            ]
        ],
        augmentation=None,
        min_component_pixels: int = 5,
    ):
        self.pairs = list(
            pairs
        )

        self.augmentation = (
            augmentation
        )

        self.min_component_pixels = int(
            min_component_pixels
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.pairs
        )

    def __getitem__(
        self,
        idx: int,
    ):
        (
            image_path,
            label_path,
        ) = self.pairs[
            idx
        ]

        image = read_rgb_image(
            image_path
        )

        mask_np = read_binary_mask(
            label_path
        )

        mask_t = torch.from_numpy(
            mask_np
        ).float()

        # Augmentation intentionally occurs before connected-component
        # conversion so image/mask geometry stays synchronized.
        if self.augmentation is not None:
            (
                image,
                mask_t,
            ) = self.augmentation(
                image,
                mask_t,
            )

        mask_np = (
            mask_t
            .detach()
            .cpu()
            .numpy()
            > 0.5
        ).astype(
            np.uint8
        )

        target = (
            binary_mask_to_target(
                mask_np,
                image_id=idx,
                min_component_pixels=self.min_component_pixels,
            )
        )

        return (
            image,
            target,
        )


def collate_fn(
    batch,
):
    return tuple(
        zip(
            *batch
        )
    )


def make_loaders(
    train_pairs: Sequence[
        Tuple[
            Path,
            Path,
        ]
    ],
    val_pairs: Sequence[
        Tuple[
            Path,
            Path,
        ]
    ],
    batch_size: int,
    train_augmentation=None,
    num_workers: int = 0,
    pin_memory: Optional[
        bool
    ] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
]:
    if pin_memory is None:
        pin_memory = (
            torch.cuda.is_available()
        )

    train_ds = (
        MaskRCNNManifestDataset(
            train_pairs,
            augmentation=train_augmentation,
        )
    )

    val_ds = (
        MaskRCNNManifestDataset(
            val_pairs,
            augmentation=None,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
        num_workers=int(
            num_workers
        ),
        pin_memory=bool(
            pin_memory
        ),
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(
            num_workers
        ),
        pin_memory=bool(
            pin_memory
        ),
        collate_fn=collate_fn,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
    )


def make_validation_loader(
    val_pairs: Sequence[
        Tuple[
            Path,
            Path,
        ]
    ],
    num_workers: int = 0,
    pin_memory: Optional[
        bool
    ] = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = (
            torch.cuda.is_available()
        )

    val_ds = (
        MaskRCNNManifestDataset(
            val_pairs,
            augmentation=None,
        )
    )

    return DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(
            num_workers
        ),
        pin_memory=bool(
            pin_memory
        ),
        collate_fn=collate_fn,
        drop_last=False,
    )


__all__ = [
    "MaskRCNNManifestDataset",
    "binary_mask_to_target",
    "collate_fn",
    "make_loaders",
    "make_validation_loader",
    "read_binary_mask",
    "read_manifest_pairs",
    "read_rgb_image",
]

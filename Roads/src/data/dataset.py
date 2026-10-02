"""
PyTorch Dataset implementation for Approach 2 - Road Extraction.

This module reads Train/Validation manifests created by:

    Roads/src/data/manifests.py

and converts ArcGIS Classified Tiles into PyTorch tensors.

Output convention
-----------------
Image:
    shape = [3, H, W]
    dtype = float32
    range = [0, 1]

Mask:
    shape = [1, H, W]
    dtype = float32
    values = {0, 1}

The dataset supports:
- Train and Validation manifests.
- Optional image/mask augmentation.
- Optional path/sample metadata return.
- Binary Road masks.
- Tile-size validation.
- Image/mask alignment checks.

This module does not access final-test imagery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifests import read_manifest
from .raster_io import (
    read_binary_mask,
    read_rgb_image,
)


class RoadSegmentationDataset(Dataset):
    """
    PyTorch dataset for binary road-surface segmentation.

    Parameters
    ----------
    manifest_csv:
        Path to train_manifest.csv or val_manifest.csv.

    augmentation:
        Optional augmentation callable.

        Supported forms:

        Albumentations-like:
            result = augmentation(
                image=image,
                mask=mask
            )

            returning:
                {
                    "image": ...,
                    "mask": ...
                }

        Custom callable:
            image, mask = augmentation(
                image,
                mask
            )

    expected_tile_size:
        Optional expected chip size.

        Example:
            256

        When supplied, every image/mask sample must have:

            256 x 256

    foreground_values:
        Optional explicit class values representing Road.

        Example:
            [1]

        When None:
            every non-zero mask value is considered Road.

    normalize:
        If True, image values are returned in [0, 1].

        Currently this should remain True for the Road pipeline.

    return_metadata:
        When True, __getitem__ also returns metadata such as
        sample ID and file paths.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        augmentation: Optional[Any] = None,
        expected_tile_size: Optional[int] = None,
        foreground_values: Optional[list[int | float]] = None,
        normalize: bool = True,
        return_metadata: bool = False,
    ):
        super().__init__()

        self.manifest_csv = Path(
            manifest_csv
        )

        self.augmentation = augmentation

        self.expected_tile_size = (
            int(expected_tile_size)
            if expected_tile_size is not None
            else None
        )

        self.foreground_values = (
            list(foreground_values)
            if foreground_values is not None
            else None
        )

        self.normalize = bool(
            normalize
        )

        self.return_metadata = bool(
            return_metadata
        )

        self.records = read_manifest(
            self.manifest_csv
        )

        if not self.records:
            raise RuntimeError(
                "Road dataset manifest contains no samples:\n"
                f"{self.manifest_csv}"
            )

        self._validate_manifest_records()


    # -----------------------------------------------------------------
    # Basic Dataset interface
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return number of samples.
        """
        return len(
            self.records
        )


    def __getitem__(
        self,
        index: int,
    ):
        """
        Load and return one Road image/mask sample.
        """
        record = self.records[
            index
        ]

        image_path = Path(
            record["image_path"]
        )

        mask_path = Path(
            record["mask_path"]
        )

        # -------------------------------------------------------------
        # Read image and mask
        # -------------------------------------------------------------

        image = read_rgb_image(
            image_path
        )

        mask = read_binary_mask(
            mask_path,
            foreground_values=self.foreground_values,
        )

        # -------------------------------------------------------------
        # Spatial validation
        # -------------------------------------------------------------

        self._validate_sample_shape(
            image=image,
            mask=mask,
            image_path=image_path,
            mask_path=mask_path,
        )

        # -------------------------------------------------------------
        # Augmentation
        # -------------------------------------------------------------

        if self.augmentation is not None:

            image, mask = (
                self._apply_augmentation(
                    image=image,
                    mask=mask,
                )
            )

            # Re-check alignment after augmentation.
            self._validate_sample_shape(
                image=image,
                mask=mask,
                image_path=image_path,
                mask_path=mask_path,
                check_expected_tile_size=False,
            )

        # -------------------------------------------------------------
        # Convert image to tensor
        # -------------------------------------------------------------

        image_tensor = (
            self._image_to_tensor(
                image
            )
        )

        # -------------------------------------------------------------
        # Convert mask to tensor
        # -------------------------------------------------------------

        mask_tensor = (
            self._mask_to_tensor(
                mask
            )
        )

        # -------------------------------------------------------------
        # Return
        # -------------------------------------------------------------

        if self.return_metadata:

            metadata = {
                "sample_id": record[
                    "sample_id"
                ],

                "split": record[
                    "split"
                ],

                "image_path": str(
                    image_path
                ),

                "mask_path": str(
                    mask_path
                ),

                "height": int(
                    mask_tensor.shape[
                        -2
                    ]
                ),

                "width": int(
                    mask_tensor.shape[
                        -1
                    ]
                ),
            }

            return (
                image_tensor,
                mask_tensor,
                metadata,
            )

        return (
            image_tensor,
            mask_tensor,
        )


    # -----------------------------------------------------------------
    # Manifest validation
    # -----------------------------------------------------------------

    def _validate_manifest_records(
        self,
    ) -> None:
        """
        Validate paths and required manifest fields.
        """
        required_fields = {
            "split",
            "sample_id",
            "image_path",
            "mask_path",
        }

        for index, record in enumerate(
            self.records
        ):

            missing = (
                required_fields
                - set(
                    record.keys()
                )
            )

            if missing:
                raise ValueError(
                    f"Manifest sample {index} is missing fields:\n"
                    + "\n".join(
                        sorted(
                            missing
                        )
                    )
                )

            image_path = Path(
                record["image_path"]
            )

            mask_path = Path(
                record["mask_path"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    "Image referenced by manifest does not exist:\n"
                    f"{image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    "Mask referenced by manifest does not exist:\n"
                    f"{mask_path}"
                )


    # -----------------------------------------------------------------
    # Sample validation
    # -----------------------------------------------------------------

    def _validate_sample_shape(
        self,
        image: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        image_path: Path,
        mask_path: Path,
        check_expected_tile_size: bool = True,
    ) -> None:
        """
        Verify image/mask spatial compatibility.
        """
        image_height, image_width = (
            self._get_spatial_shape(
                image,
                is_image=True,
            )
        )

        mask_height, mask_width = (
            self._get_spatial_shape(
                mask,
                is_image=False,
            )
        )

        if (
            image_height != mask_height
            or image_width != mask_width
        ):
            raise ValueError(
                "Road image/mask dimension mismatch.\n\n"
                f"Image:\n{image_path}\n"
                f"Shape: {image_height} x {image_width}\n\n"
                f"Mask:\n{mask_path}\n"
                f"Shape: {mask_height} x {mask_width}"
            )

        if (
            check_expected_tile_size
            and self.expected_tile_size is not None
        ):

            expected = (
                self.expected_tile_size
            )

            if (
                image_height != expected
                or image_width != expected
            ):
                raise ValueError(
                    "Road chip does not match the configured tile size.\n"
                    f"Expected: {expected} x {expected}\n"
                    f"Received: {image_height} x {image_width}\n"
                    f"Image: {image_path}"
                )


    @staticmethod
    def _get_spatial_shape(
        array: np.ndarray | torch.Tensor,
        is_image: bool,
    ) -> tuple[int, int]:
        """
        Get H/W dimensions from NumPy arrays or tensors.
        """
        shape = tuple(
            array.shape
        )

        if len(shape) == 2:
            return (
                int(shape[0]),
                int(shape[1]),
            )

        if len(shape) != 3:
            raise ValueError(
                "Expected 2D or 3D sample. "
                f"Received shape: {shape}"
            )

        if isinstance(
            array,
            torch.Tensor,
        ):

            # CHW tensor
            if shape[0] in (
                1,
                3,
                4,
            ):
                return (
                    int(shape[1]),
                    int(shape[2]),
                )

        # NumPy / HWC form.
        return (
            int(shape[0]),
            int(shape[1]),
        )


    # -----------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Apply an augmentation while preserving image/mask alignment.
        """
        try:
            result = self.augmentation(
                image=image,
                mask=mask,
            )

        except TypeError:

            result = self.augmentation(
                image,
                mask,
            )

        # Albumentations-style output.
        if isinstance(
            result,
            dict,
        ):

            if "image" not in result:
                raise ValueError(
                    "Augmentation dictionary does not contain 'image'."
                )

            if "mask" not in result:
                raise ValueError(
                    "Augmentation dictionary does not contain 'mask'."
                )

            image = result[
                "image"
            ]

            mask = result[
                "mask"
            ]

        # Custom tuple/list output.
        elif isinstance(
            result,
            (
                tuple,
                list,
            ),
        ):

            if len(result) != 2:
                raise ValueError(
                    "Custom augmentation must return exactly "
                    "(image, mask)."
                )

            image, mask = result

        else:
            raise TypeError(
                "Unsupported augmentation output. "
                "Expected dict or (image, mask)."
            )

        return (
            image,
            mask,
        )


    # -----------------------------------------------------------------
    # Image tensor conversion
    # -----------------------------------------------------------------

    def _image_to_tensor(
        self,
        image: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert image into float32 [3, H, W].
        """
        if isinstance(
            image,
            torch.Tensor,
        ):

            tensor = image.detach().clone().float()

            if tensor.ndim != 3:
                raise ValueError(
                    "Image tensor must have 3 dimensions. "
                    f"Received: {tuple(tensor.shape)}"
                )

            # HWC -> CHW
            if (
                tensor.shape[0]
                not in (
                    1,
                    3,
                )
                and tensor.shape[-1]
                in (
                    1,
                    3,
                )
            ):
                tensor = tensor.permute(
                    2,
                    0,
                    1,
                )

        else:

            image = np.asarray(
                image
            )

            if image.ndim != 3:
                raise ValueError(
                    "Image array must have shape H x W x C. "
                    f"Received: {image.shape}"
                )

            image = np.ascontiguousarray(
                image
            )

            tensor = torch.from_numpy(
                image
            ).permute(
                2,
                0,
                1,
            ).float()

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(
                3,
                1,
                1,
            )

        if tensor.shape[0] != 3:
            raise ValueError(
                "Road models currently expect 3-channel imagery. "
                f"Received tensor shape: {tuple(tensor.shape)}"
            )

        if self.normalize:

            # uint8 image -> [0, 1]
            if (
                float(
                    tensor.max()
                )
                > 1.0
            ):
                tensor = (
                    tensor
                    / 255.0
                )

            tensor = torch.clamp(
                tensor,
                0.0,
                1.0,
            )

        return tensor.contiguous()


    # -----------------------------------------------------------------
    # Mask tensor conversion
    # -----------------------------------------------------------------

    @staticmethod
    def _mask_to_tensor(
        mask: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert mask into float32 [1, H, W] with values {0, 1}.
        """
        if isinstance(
            mask,
            torch.Tensor,
        ):

            tensor = mask.detach().clone()

            if tensor.ndim == 3:

                if tensor.shape[0] == 1:
                    tensor = tensor[
                        0
                    ]

                elif tensor.shape[-1] == 1:
                    tensor = tensor[
                        :,
                        :,
                        0
                    ]

                else:
                    # Defensive fallback if an RGB-style mask reaches here.
                    tensor = torch.max(
                        tensor,
                        dim=0,
                    ).values

            if tensor.ndim != 2:
                raise ValueError(
                    "Mask tensor must resolve to H x W. "
                    f"Received: {tuple(tensor.shape)}"
                )

            tensor = (
                tensor > 0
            ).float()

        else:

            mask = np.asarray(
                mask
            )

            if mask.ndim == 3:

                if mask.shape[-1] == 1:
                    mask = mask[
                        :,
                        :,
                        0
                    ]

                else:
                    mask = np.max(
                        mask,
                        axis=-1,
                    )

            if mask.ndim != 2:
                raise ValueError(
                    "Mask array must resolve to H x W. "
                    f"Received: {mask.shape}"
                )

            mask = (
                mask > 0
            ).astype(
                np.float32
            )

            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    mask
                )
            )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.contiguous()


    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def get_sample_record(
        self,
        index: int,
    ) -> Dict[str, str]:
        """
        Return manifest metadata without loading image data.
        """
        return dict(
            self.records[index]
        )


    def get_split_name(
        self,
    ) -> str:
        """
        Return the split name when the manifest contains one split.
        """
        split_names = {
            str(
                record["split"]
            ).lower()
            for record in self.records
        }

        if len(split_names) == 1:
            return next(
                iter(
                    split_names
                )
            )

        return "mixed"


    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return basic dataset metadata.
        """
        return {
            "manifest": str(
                self.manifest_csv
            ),

            "split": self.get_split_name(),

            "sample_count": len(
                self
            ),

            "expected_tile_size": self.expected_tile_size,

            "normalize": self.normalize,

            "augmentation_enabled": (
                self.augmentation
                is not None
            ),

            "return_metadata": self.return_metadata,
        }
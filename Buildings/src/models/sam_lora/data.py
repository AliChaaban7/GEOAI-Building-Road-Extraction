"""
Buildings/src/models/sam_lora/data.py

Dataset/DataLoader bridge for SAM-LoRA.

IMPORTANT
---------
SAM-LoRA does NOT introduce a new dataset implementation.

The Buildings module already has a shared semantic-segmentation dataset
pipeline in:

    Buildings/src/train.py

That pipeline already provides:

    image:
        float32
        [3, H, W]
        range [0, 1]

    mask:
        float32
        [1, H, W]
        binary values 0 / 1

It also provides:

    - train_manifest.csv
    - val_manifest.csv
    - controlled train/validation split
    - shared building augmentation
    - validation augmentation disabled

SAM-LoRA reuses that pipeline so comparisons against U-Net and
DeepLabV3 remain dataset-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Reuse the EXISTING Buildings dataset architecture.
# ---------------------------------------------------------------------

from ...train import create_dataloaders


# ---------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------


@dataclass
class SAMLoRADataBundle:
    """
    Complete SAM-LoRA train/validation data bundle.
    """

    train_loader: DataLoader
    val_loader: DataLoader

    train_dataset: Dataset
    val_dataset: Dataset

    train_manifest: str
    val_manifest: str

    batch_size: int
    train_samples: int
    validation_samples: int

    augmentation_enabled: bool

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "train_manifest": self.train_manifest,
            "val_manifest": self.val_manifest,
            "batch_size": int(
                self.batch_size
            ),
            "train_samples": int(
                self.train_samples
            ),
            "validation_samples": int(
                self.validation_samples
            ),
            "augmentation_enabled": bool(
                self.augmentation_enabled
            ),
        }


# ---------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------


def _validate_manifest(
    manifest_path: Union[
        str,
        Path,
    ],
    name: str,
) -> Path:
    """
    Verify that a train/validation manifest exists.
    """

    path = Path(
        manifest_path
    ).expanduser().resolve()

    if not path.exists():

        raise FileNotFoundError(
            f"SAM-LoRA {name} manifest was not found:\n"
            f"    {path}"
        )

    if not path.is_file():

        raise FileNotFoundError(
            f"SAM-LoRA {name} manifest path is not a file:\n"
            f"    {path}"
        )

    return path


# ---------------------------------------------------------------------
# Sample validation
# ---------------------------------------------------------------------


def validate_sam_lora_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    sample_name: str = "sample",
) -> None:
    """
    Verify one dataset sample matches SAM-LoRA requirements.

    Expected image:

        [3, H, W]
        float
        values approximately [0, 1]

    Expected mask:

        [1, H, W]
        float
        values 0 / 1
    """

    # ==============================================================
    # IMAGE
    # ==============================================================

    if not torch.is_tensor(
        image
    ):

        raise TypeError(
            f"{sample_name}: image is not a torch.Tensor."
        )

    if image.ndim != 3:

        raise ValueError(
            f"{sample_name}: image must have shape [C,H,W]. "
            f"Received {tuple(image.shape)}."
        )

    if int(
        image.shape[0]
    ) != 3:

        raise ValueError(
            f"{sample_name}: SAM requires exactly 3 RGB channels. "
            f"Received C={image.shape[0]}."
        )

    if not torch.is_floating_point(
        image
    ):

        raise TypeError(
            f"{sample_name}: image must be floating point. "
            f"Received {image.dtype}."
        )

    if not torch.isfinite(
        image
    ).all():

        raise ValueError(
            f"{sample_name}: image contains NaN or Inf."
        )

    image_min = float(
        image.min().item()
    )

    image_max = float(
        image.max().item()
    )

    # Shared Buildings dataset should provide [0,1].
    if (
        image_min < -1e-6
        or image_max > 1.0 + 1e-6
    ):

        raise ValueError(
            f"{sample_name}: expected image range [0,1].\n"
            f"Observed range: "
            f"[{image_min:.6f}, {image_max:.6f}]"
        )

    # ==============================================================
    # MASK
    # ==============================================================

    if not torch.is_tensor(
        mask
    ):

        raise TypeError(
            f"{sample_name}: mask is not a torch.Tensor."
        )

    if mask.ndim != 3:

        raise ValueError(
            f"{sample_name}: mask must have shape [1,H,W]. "
            f"Received {tuple(mask.shape)}."
        )

    if int(
        mask.shape[0]
    ) != 1:

        raise ValueError(
            f"{sample_name}: building mask must contain "
            f"one channel. Received C={mask.shape[0]}."
        )

    if not torch.is_floating_point(
        mask
    ):

        raise TypeError(
            f"{sample_name}: mask must be floating point. "
            f"Received {mask.dtype}."
        )

    if not torch.isfinite(
        mask
    ).all():

        raise ValueError(
            f"{sample_name}: mask contains NaN or Inf."
        )

    # ==============================================================
    # IMAGE / MASK SIZE
    # ==============================================================

    if (
        image.shape[-2:]
        != mask.shape[-2:]
    ):

        raise ValueError(
            f"{sample_name}: image and mask sizes differ.\n"
            f"Image: {tuple(image.shape)}\n"
            f"Mask : {tuple(mask.shape)}"
        )

    # ==============================================================
    # BINARY MASK
    # ==============================================================

    mask_min = float(
        mask.min().item()
    )

    mask_max = float(
        mask.max().item()
    )

    if (
        mask_min < 0.0
        or mask_max > 1.0
    ):

        raise ValueError(
            f"{sample_name}: mask must use values between 0 and 1.\n"
            f"Observed range: [{mask_min}, {mask_max}]"
        )

    unique_values = torch.unique(
        mask
    )

    allowed = torch.logical_or(
        torch.isclose(
            unique_values,
            torch.zeros_like(
                unique_values
            ),
        ),
        torch.isclose(
            unique_values,
            torch.ones_like(
                unique_values
            ),
        ),
    )

    if not bool(
        allowed.all().item()
    ):

        raise ValueError(
            f"{sample_name}: mask is not binary.\n"
            f"Unique values: "
            f"{unique_values.detach().cpu().tolist()}"
        )


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------


def validate_sam_lora_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Dict[str, Any]:
    """
    Perform lightweight compatibility validation.

    We intentionally validate representative samples rather than
    reading the entire dataset again because dataset quality checking
    already happens earlier in the Buildings workflow.
    """

    if len(
        train_dataset
    ) <= 0:

        raise RuntimeError(
            "SAM-LoRA training dataset is empty."
        )

    if len(
        val_dataset
    ) <= 0:

        raise RuntimeError(
            "SAM-LoRA validation dataset is empty."
        )

    # --------------------------------------------------------------
    # Validate first train sample.
    # --------------------------------------------------------------

    train_sample = (
        train_dataset[0]
    )

    if (
        not isinstance(
            train_sample,
            (tuple, list),
        )
        or len(
            train_sample
        ) < 2
    ):

        raise TypeError(
            "Shared Buildings training dataset must return "
            "(image, mask)."
        )

    train_image = (
        train_sample[0]
    )

    train_mask = (
        train_sample[1]
    )

    validate_sam_lora_sample(
        image=train_image,
        mask=train_mask,
        sample_name="training sample",
    )

    # --------------------------------------------------------------
    # Validate first validation sample.
    # --------------------------------------------------------------

    val_sample = (
        val_dataset[0]
    )

    if (
        not isinstance(
            val_sample,
            (tuple, list),
        )
        or len(
            val_sample
        ) < 2
    ):

        raise TypeError(
            "Shared Buildings validation dataset must return "
            "(image, mask)."
        )

    val_image = (
        val_sample[0]
    )

    val_mask = (
        val_sample[1]
    )

    validate_sam_lora_sample(
        image=val_image,
        mask=val_mask,
        sample_name="validation sample",
    )

    return {
        "train_samples": int(
            len(
                train_dataset
            )
        ),

        "validation_samples": int(
            len(
                val_dataset
            )
        ),

        "train_image_shape": list(
            train_image.shape
        ),

        "train_mask_shape": list(
            train_mask.shape
        ),

        "validation_image_shape": list(
            val_image.shape
        ),

        "validation_mask_shape": list(
            val_mask.shape
        ),

        "image_range": [
            0.0,
            1.0,
        ],

        "mask_type": (
            "binary"
        ),

        "channels": 3,
    }


# ---------------------------------------------------------------------
# Main DataLoader builder
# ---------------------------------------------------------------------


def create_sam_lora_dataloaders(
    train_manifest_csv: Union[
        str,
        Path,
    ],
    val_manifest_csv: Union[
        str,
        Path,
    ],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: Optional[
        bool
    ] = None,
    train_augmentation: Optional[
        Any
    ] = None,
    validate_data: bool = True,
    verbose: bool = True,
) -> SAMLoRADataBundle:
    """
    Create SAM-LoRA DataLoaders using the EXISTING Buildings dataset.

    This function does NOT create a new train/validation split.

    It receives the exact manifests already created by the Buildings
    pipeline.

    Therefore:

        U-Net
        DeepLabV3
        SAM-LoRA

    can use the same train/validation samples.

    Parameters
    ----------
    train_manifest_csv:
        Existing Buildings training manifest.

    val_manifest_csv:
        Existing Buildings validation manifest.

    batch_size:
        SAM-LoRA batch size.

    num_workers:
        PyTorch DataLoader workers.

    pin_memory:
        If None:

            CUDA available -> True
            otherwise      -> False

    train_augmentation:
        Existing shared Buildings augmentation object.

        Applied ONLY to training data.

    validate_data:
        Check representative train/validation samples.

    verbose:
        Print data summary.
    """

    # ==============================================================
    # Validate basic configuration
    # ==============================================================

    train_manifest = (
        _validate_manifest(
            train_manifest_csv,
            "training",
        )
    )

    val_manifest = (
        _validate_manifest(
            val_manifest_csv,
            "validation",
        )
    )

    batch_size = int(
        batch_size
    )

    if batch_size <= 0:

        raise ValueError(
            "SAM-LoRA batch_size must be > 0."
        )

    num_workers = int(
        num_workers
    )

    if num_workers < 0:

        raise ValueError(
            "num_workers must be >= 0."
        )

    if pin_memory is None:

        pin_memory = bool(
            torch.cuda.is_available()
        )

    # ==============================================================
    # REUSE SHARED BUILDINGS DATALOADER
    # ==============================================================

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    ) = create_dataloaders(

        train_manifest_csv=(
            train_manifest
        ),

        val_manifest_csv=(
            val_manifest
        ),

        batch_size=(
            batch_size
        ),

        num_workers=(
            num_workers
        ),

        pin_memory=bool(
            pin_memory
        ),

        # Important:
        # prevents DeepLab-specific BatchNorm logic from being used.
        model_type=(
            "sam_lora"
        ),

        # Existing shared augmentation.
        # Validation remains augmentation=None inside src/train.py.
        train_augmentation=(
            train_augmentation
        ),
    )

    # ==============================================================
    # Compatibility validation
    # ==============================================================

    validation_summary = None

    if validate_data:

        validation_summary = (
            validate_sam_lora_datasets(
                train_dataset=(
                    train_dataset
                ),
                val_dataset=(
                    val_dataset
                ),
            )
        )

    # ==============================================================
    # Summary
    # ==============================================================

    bundle = SAMLoRADataBundle(

        train_loader=(
            train_loader
        ),

        val_loader=(
            val_loader
        ),

        train_dataset=(
            train_dataset
        ),

        val_dataset=(
            val_dataset
        ),

        train_manifest=str(
            train_manifest
        ),

        val_manifest=str(
            val_manifest
        ),

        batch_size=int(
            batch_size
        ),

        train_samples=int(
            len(
                train_dataset
            )
        ),

        validation_samples=int(
            len(
                val_dataset
            )
        ),

        augmentation_enabled=bool(
            train_augmentation
            is not None
        ),
    )

    if verbose:

        print()
        print("=" * 72)
        print(
            "SAM-LoRA Dataset"
        )
        print("=" * 72)

        print(
            f"Training manifest           : "
            f"{train_manifest}"
        )

        print(
            f"Validation manifest         : "
            f"{val_manifest}"
        )

        print(
            f"Training samples            : "
            f"{bundle.train_samples}"
        )

        print(
            f"Validation samples          : "
            f"{bundle.validation_samples}"
        )

        print(
            f"Batch size                  : "
            f"{bundle.batch_size}"
        )

        print(
            f"Workers                     : "
            f"{num_workers}"
        )

        print(
            f"Pin memory                  : "
            f"{bool(pin_memory)}"
        )

        print(
            f"Training augmentation       : "
            f"{bundle.augmentation_enabled}"
        )

        print(
            "Validation augmentation     : "
            "False"
        )

        if validation_summary is not None:

            print(
                f"Image shape                 : "
                f"{validation_summary['train_image_shape']}"
            )

            print(
                f"Mask shape                  : "
                f"{validation_summary['train_mask_shape']}"
            )

            print(
                "Image range                 : "
                "[0, 1]"
            )

            print(
                "Mask                         : "
                "binary 0/1"
            )

        print("=" * 72)
        print()

    return bundle
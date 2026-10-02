"""
Native PyTorch inference for Approach 2 - Road Extraction.

Supported
---------
- U-Net
- DeepLabV3+
- SAM-LoRA

Checkpoint formats
------------------
U-Net / DeepLabV3+
    complete model state

SAM-LoRA
    official Meta SAM ViT-B
        +
    exact LoRA architecture from checkpoint
        +
    trained Road LoRA / mask-decoder parameters

The caller does not need special SAM-LoRA handling.

Scientific rule
---------------
This module performs prediction only.

It does NOT:
- select thresholds on final test data
- select post-processing on final test data
- access Ground Truth
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from src.checkpointing import (
    build_checkpoint_context,
    load_model_state,
    load_torch_checkpoint,
)

from src.data import (
    RoadSegmentationDataset,
)

from src.models import (
    build_pytorch_model,
    extract_model_logits,
)


# =====================================================================
# DEVICE
# =====================================================================

def resolve_inference_device(
    device: Optional[
        str | torch.device
    ] = None,
) -> torch.device:

    if isinstance(
        device,
        torch.device,
    ):

        resolved = device

    elif device is not None:

        resolved = torch.device(
            str(
                device
            )
        )

    else:

        resolved = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA was requested for inference but CUDA is unavailable."
        )

    return resolved


# =====================================================================
# AMP
# =====================================================================

def _autocast_context(
    device: torch.device,
    enabled: bool,
):

    enabled = bool(
        enabled
        and device.type == "cuda"
    )

    try:

        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):

        return torch.cuda.amp.autocast(
            enabled=enabled
        )


# =====================================================================
# CHECKPOINT METADATA
# =====================================================================

def checkpoint_epoch(
    checkpoint,
):

    if isinstance(
        checkpoint,
        dict,
    ):

        value = checkpoint.get(
            "epoch"
        )

        if value is not None:

            return int(
                value
            )

    return None


def checkpoint_format(
    checkpoint,
) -> str:

    if isinstance(
        checkpoint,
        dict,
    ):

        value = checkpoint.get(
            "checkpoint_format"
        )

        if value:

            return str(
                value
            )

    return "legacy_or_raw_state_dict"


def checkpoint_validation_metrics(
    checkpoint,
) -> Dict[str, Any]:

    if not isinstance(
        checkpoint,
        dict,
    ):

        return {}

    value = checkpoint.get(
        "validation_metrics",
        {},
    )

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


# =====================================================================
# CHECKPOINT RESTORATION
# =====================================================================

def restore_model_from_checkpoint(
    checkpoint_path: str | Path,
    context: Dict[str, Any],
    device: Optional[
        str | torch.device
    ] = None,
    strict: bool = True,
):
    """
    Reconstruct and restore one trained native Road model.

    Returns
    -------
    model
    checkpoint
    resolved_device

    Notes
    -----
    For SAM-LoRA this automatically performs:

        official SAM
            +
        exact saved LoRA architecture
            +
        Road trained adapters / mask decoder
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )

    (
        reconstruction_context,
        reconstruction_info,
    ) = build_checkpoint_context(
        checkpoint=checkpoint,
        context=context,
        checkpoint_path=checkpoint_path,
    )

    backend = reconstruction_context.get(
        "model_backend",
        "pytorch",
    )

    if backend != "pytorch":

        raise RuntimeError(
            "restore_model_from_checkpoint() supports the native "
            "PyTorch Road backend only.\n"
            f"Resolved backend: {backend}"
        )

    # --------------------------------------------------------------
    # Build exact architecture first.
    # --------------------------------------------------------------

    model = build_pytorch_model(
        context=reconstruction_context
    )

    # --------------------------------------------------------------
    # Then restore:
    #
    # U-Net/DeepLab:
    #     full model state
    #
    # SAM-LoRA:
    #     trainable-only state over official SAM
    # --------------------------------------------------------------

    state_info = load_model_state(
        model=model,
        checkpoint=checkpoint,
        strict=bool(
            strict
        ),
    )

    resolved_device = resolve_inference_device(
        device
    )

    model = model.to(
        resolved_device
    )

    model.eval()

    # --------------------------------------------------------------
    # Preserve reconstruction provenance for all downstream modules.
    # --------------------------------------------------------------

    model._road_checkpoint_context = (
        reconstruction_context
    )

    model._road_checkpoint_reconstruction = {
        **reconstruction_info,
        **state_info,

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "epoch":
            checkpoint_epoch(
                checkpoint
            ),

        "checkpoint_format":
            checkpoint_format(
                checkpoint
            ),

        "validation_metrics":
            checkpoint_validation_metrics(
                checkpoint
            ),
    }

    return (
        model,
        checkpoint,
        resolved_device,
    )


# =====================================================================
# CHECKPOINT-RESOLVED EXPERIMENT SETTINGS
# =====================================================================

def get_restored_experiment_config(
    model,
    fallback_context: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Prefer architecture information reconstructed from checkpoint.
    """

    reconstructed = getattr(
        model,
        "_road_checkpoint_context",
        {},
    )

    if isinstance(
        reconstructed,
        dict,
    ):

        experiment = reconstructed.get(
            "experiment_config"
        )

        if isinstance(
            experiment,
            dict,
        ):

            return dict(
                experiment
            )

    if isinstance(
        fallback_context,
        dict,
    ):

        experiment = fallback_context.get(
            "experiment_config"
        )

        if isinstance(
            experiment,
            dict,
        ):

            return dict(
                experiment
            )

    return {}


def get_restored_tile_size(
    model,
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> int:
    """
    Resolve tile size from the trained checkpoint before falling back to
    the current experiment configuration.

    Important for Master Optuna winners.
    """

    experiment = get_restored_experiment_config(
        model=model,
        fallback_context=context,
    )

    value = experiment.get(
        "tile_size"
    )

    if value is None:

        raise RuntimeError(
            "Could not resolve the trained model tile size."
        )

    value = int(
        value
    )

    if value <= 0:

        raise ValueError(
            "Resolved tile_size must be > 0."
        )

    return value


# =====================================================================
# TENSOR PREPARATION
# =====================================================================

def prepare_image_tensor(
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize tensor shape/value range expected by all native Road models.

    Input:
        [C,H,W]
        or
        [B,C,H,W]

    Output:
        float32 [B,3,H,W] in [0,1]
    """

    if not torch.is_tensor(
        images
    ):

        raise TypeError(
            "prepare_image_tensor() expects torch.Tensor."
        )

    if images.ndim == 3:

        images = images.unsqueeze(
            0
        )

    if images.ndim != 4:

        raise ValueError(
            "Expected [C,H,W] or [B,C,H,W]."
        )

    if images.shape[
        1
    ] != 3:

        raise ValueError(
            "Native Road models require exactly 3 RGB channels."
        )

    images = images.float()

    if images.numel() > 0:

        minimum = float(
            images.min()
            .detach()
            .cpu()
        )

        maximum = float(
            images.max()
            .detach()
            .cpu()
        )

        if minimum < 0.0:

            raise ValueError(
                "Road inference image contains negative values."
            )

        if maximum > 255.0:

            raise ValueError(
                "Road inference image contains values >255.\n"
                "No automatic radiometric stretching is performed."
            )

        if maximum > 1.5:

            images = (
                images
                / 255.0
            )

    return images


# =====================================================================
# TENSOR PREDICTION
# =====================================================================

@torch.no_grad()
def predict_tensor(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: Optional[
        str | torch.device
    ] = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Run semantic Road prediction.

    Works identically for:
    - U-Net
    - DeepLabV3+
    - SAM-LoRA
    """

    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "threshold must be between 0 and 1."
        )

    resolved_device = resolve_inference_device(
        device
    )

    images = prepare_image_tensor(
        images
    )

    images = images.to(
        resolved_device,
        non_blocking=True,
    )

    model = model.to(
        resolved_device
    )

    model.eval()

    amp_enabled = bool(
        use_amp
        and resolved_device.type == "cuda"
    )

    with _autocast_context(
        device=resolved_device,
        enabled=amp_enabled,
    ):

        output = model(
            images
        )

        logits = extract_model_logits(
            output
        )

    probabilities = torch.sigmoid(
        logits
    )

    binary_masks = (
        probabilities
        >= threshold
    )

    return {
        "logits":
            logits,

        "probabilities":
            probabilities,

        "binary_masks":
            binary_masks,

        # Compatibility alias.
        "predictions":
            binary_masks,

        "threshold":
            threshold,

        "device":
            str(
                resolved_device
            ),
    }


# =====================================================================
# SINGLE IMAGE
# =====================================================================

def image_to_tensor(
    image,
) -> torch.Tensor:
    """
    Convert path / PIL / NumPy image into [3,H,W] float tensor.
    """

    if isinstance(
        image,
        (
            str,
            Path,
        ),
    ):

        with Image.open(
            image
        ) as source:

            array = np.asarray(
                source.convert(
                    "RGB"
                )
            )

    elif isinstance(
        image,
        Image.Image,
    ):

        array = np.asarray(
            image.convert(
                "RGB"
            )
        )

    else:

        array = np.asarray(
            image
        )

    if array.ndim == 2:

        array = np.stack(
            [
                array,
                array,
                array,
            ],
            axis=-1,
        )

    if (
        array.ndim != 3
        or array.shape[
            2
        ] < 3
    ):

        raise ValueError(
            "Expected H x W x 3 image."
        )

    array = np.ascontiguousarray(
        array[
            :,
            :,
            :3,
        ]
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1,
    )

    return prepare_image_tensor(
        tensor
    )[
        0
    ]


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image,
    device: Optional[
        str | torch.device
    ] = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, Any]:

    tensor = image_to_tensor(
        image
    )

    return predict_tensor(
        model=model,
        images=tensor,
        device=device,
        threshold=threshold,
        use_amp=use_amp,
    )


# =====================================================================
# MANIFEST DATASET
# =====================================================================

def _build_manifest_dataset(
    manifest_csv: str | Path,
    tile_size: int,
):

    try:

        return RoadSegmentationDataset(
            manifest_csv=manifest_csv,
            augmentation=None,
            expected_tile_size=int(
                tile_size
            ),
            normalize=True,
            return_metadata=False,
        )

    except TypeError:

        pass

    try:

        return RoadSegmentationDataset(
            manifest_path=manifest_csv,
            augmentation=None,
            expected_tile_size=int(
                tile_size
            ),
            normalize=True,
            return_metadata=False,
        )

    except TypeError:

        pass

    return RoadSegmentationDataset(
        manifest_csv
    )


def _unpack_dataset_batch(
    batch,
):

    if isinstance(
        batch,
        dict,
    ):

        images = batch.get(
            "image",
            batch.get(
                "images"
            ),
        )

        masks = batch.get(
            "mask",
            batch.get(
                "masks",
                batch.get(
                    "label",
                    batch.get(
                        "labels"
                    ),
                ),
            ),
        )

        return (
            images,
            masks,
        )

    if isinstance(
        batch,
        (
            tuple,
            list,
        ),
    ):

        return (
            batch[
                0
            ],
            (
                batch[
                    1
                ]
                if len(
                    batch
                ) > 1
                else None
            ),
        )

    return (
        batch,
        None,
    )


# =====================================================================
# SAVE MASK
# =====================================================================

def _save_binary_png(
    mask: np.ndarray,
    path: str | Path,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        np.asarray(
            mask
        )
        > 0
    ).astype(
        np.uint8
    ) * 255

    Image.fromarray(
        output,
        mode="L",
    ).save(
        path
    )


# =====================================================================
# MANIFEST INFERENCE
# =====================================================================

@torch.no_grad()
def predict_manifest(
    model: torch.nn.Module,
    manifest_csv: str | Path,
    tile_size: int,
    output_folder: str | Path,
    device: Optional[
        str | torch.device
    ] = None,
    threshold: float = 0.5,
    batch_size: int = 4,
    num_workers: int = 0,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Run prediction over Train/Validation-style manifest.

    This is primarily used by validation-side modules.
    """

    output_folder = Path(
        output_folder
    )

    mask_folder = (
        output_folder
        / "binary_masks"
    )

    mask_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = _build_manifest_dataset(
        manifest_csv=manifest_csv,
        tile_size=tile_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=False,
        num_workers=int(
            num_workers
        ),
        pin_memory=True,
        drop_last=False,
    )

    records = []

    sample_index = 0

    for batch in loader:

        (
            images,
            _,
        ) = _unpack_dataset_batch(
            batch
        )

        prediction = predict_tensor(
            model=model,
            images=images,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
        )

        masks = (
            prediction[
                "binary_masks"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        probabilities = (
            prediction[
                "probabilities"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        for index in range(
            masks.shape[
                0
            ]
        ):

            sample_index += 1

            filename = (
                f"sample_{sample_index:06d}.png"
            )

            output_path = (
                mask_folder
                / filename
            )

            _save_binary_png(
                masks[
                    index,
                    0,
                ],
                output_path,
            )

            records.append(
                {
                    "sample_index":
                        sample_index,

                    "mask_path":
                        str(
                            output_path
                        ),

                    "mean_probability":
                        float(
                            probabilities[
                                index,
                                0,
                            ].mean()
                        ),
                }
            )

    return {
        "manifest":
            str(
                manifest_csv
            ),

        "sample_count":
            len(
                records
            ),

        "tile_size":
            int(
                tile_size
            ),

        "threshold":
            float(
                threshold
            ),

        "output_folder":
            str(
                output_folder
            ),

        "records":
            records,
    }


# =====================================================================
# EXPERIMENT INFERENCE
# =====================================================================

def run_experiment_inference(
    context: Dict[str, Any],
    manifest_csv: str | Path,
    checkpoint_path: Optional[
        str | Path
    ] = None,
    threshold: float = 0.5,
    output_folder: Optional[
        str | Path
    ] = None,
    device: Optional[
        str | torch.device
    ] = None,
    batch_size: Optional[
        int
    ] = None,
) -> Dict[str, Any]:
    """
    Restore trained model and run manifest inference.
    """

    if checkpoint_path is None:

        checkpoint_path = (
            Path(
                context[
                    "checkpoint_folder"
                ]
            )
            / "best_model.pth"
        )

    if output_folder is None:

        output_folder = (
            Path(
                context[
                    "output_folder"
                ]
            )
            / "inference"
        )

    (
        model,
        checkpoint,
        resolved_device,
    ) = restore_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        context=context,
        device=device,
        strict=True,
    )

    tile_size = get_restored_tile_size(
        model=model,
        context=context,
    )

    if batch_size is None:

        model_type = str(
            getattr(
                model,
                "_road_checkpoint_reconstruction",
                {},
            ).get(
                "model_type",
                context.get(
                    "model_type",
                    "",
                ),
            )
        ).lower()

        # SAM ViT-B on the thesis 8 GB GPU must stay conservative.
        batch_size = (
            1
            if model_type == "sam_lora"
            else 4
        )

    result = predict_manifest(
        model=model,
        manifest_csv=manifest_csv,
        tile_size=tile_size,
        output_folder=output_folder,
        device=resolved_device,
        threshold=threshold,
        batch_size=int(
            batch_size
        ),
    )

    result[
        "checkpoint"
    ] = str(
        checkpoint_path
    )

    result[
        "checkpoint_epoch"
    ] = checkpoint_epoch(
        checkpoint
    )

    result[
        "checkpoint_format"
    ] = checkpoint_format(
        checkpoint
    )

    result[
        "checkpoint_reconstruction"
    ] = getattr(
        model,
        "_road_checkpoint_reconstruction",
        {},
    )

    return result
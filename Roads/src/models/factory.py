"""
Unified model factory for Approach 2 - Road Extraction.

Native PyTorch
--------------
- U-Net
- DeepLabV3+
- SAM-LoRA

ArcGIS Learn
------------
- ConnectNet
- MultiTaskRoadExtractor

Design rule
-----------
ArcGIS Learn imports remain lazy.

Therefore the native PyTorch pipeline can run without initializing an
ArcGIS Pro product license.

SAM-LoRA also remains lazy so U-Net/DeepLabV3+ do not require the
segment-anything package.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------

_MODEL_ALIASES = {
    "u-net":
        "unet",

    "u_net":
        "unet",

    "unet":
        "unet",

    "deeplab":
        "deeplabv3",

    "deeplabv3+":
        "deeplabv3",

    "deeplabv3plus":
        "deeplabv3",

    "deeplab_v3":
        "deeplabv3",

    "deeplabv3":
        "deeplabv3",

    "sam-lora":
        "sam_lora",

    "sam lora":
        "sam_lora",

    "samlora":
        "sam_lora",

    "sam_lora":
        "sam_lora",

    "connect-net":
        "connectnet",

    "connect_net":
        "connectnet",

    "connectnet":
        "connectnet",

    "multitask":
        "multitask_road_extractor",

    "multi_task_road_extractor":
        "multitask_road_extractor",

    "multitaskroadextractor":
        "multitask_road_extractor",

    "multitask_road_extractor":
        "multitask_road_extractor",
}


def normalize_model_type(
    value,
) -> str:

    key = (
        str(
            value
        )
        .strip()
        .lower()
    )

    return _MODEL_ALIASES.get(
        key,
        key,
    )


# ---------------------------------------------------------------------
# Generic builder invocation
# ---------------------------------------------------------------------

def _call_builder_flexibly(
    builder,
    context: Dict[str, Any],
):
    """
    Support the minor signature differences between native model
    builders while keeping one shared factory.
    """

    model_config = context[
        "model_config"
    ]

    experiment_config = context.get(
        "experiment_config",
        {},
    )

    attempts = [
        lambda:
            builder(
                model_config=model_config,
            ),

        lambda:
            builder(
                config=model_config,
            ),

        lambda:
            builder(
                context=context,
            ),

        lambda:
            builder(
                experiment_config=experiment_config,
                model_config=model_config,
            ),

        lambda:
            builder(
                model_config,
            ),

        lambda:
            builder(),
    ]

    errors = []

    for attempt in attempts:

        try:

            return attempt()

        except TypeError as exc:

            errors.append(
                str(
                    exc
                )
            )

    raise TypeError(
        "Could not call model builder with supported signatures.\n"
        "Errors:\n- "
        + "\n- ".join(
            errors
        )
    )


def _build_from_module(
    module_name: str,
    candidate_names,
    context: Dict[str, Any],
):
    """
    Import one model lazily and call the first available builder.
    """

    module = importlib.import_module(
        module_name
    )

    for name in candidate_names:

        if not hasattr(
            module,
            name,
        ):

            continue

        builder = getattr(
            module,
            name,
        )

        return _call_builder_flexibly(
            builder=builder,

            context=context,
        )

    raise AttributeError(
        f"No supported builder found in {module_name}.\n"
        f"Tried: {candidate_names}"
    )


# ---------------------------------------------------------------------
# Native PyTorch model factory
# ---------------------------------------------------------------------

def build_pytorch_model(
    context: Dict[str, Any],
) -> nn.Module:
    """
    Build the selected native PyTorch Road model.

    Returns logits model only. It does not move the model to CUDA.
    """

    if not isinstance(
        context,
        dict,
    ):

        raise TypeError(
            "Model factory context must be a dictionary."
        )

    if "model_config" not in context:

        raise KeyError(
            "context['model_config'] is required."
        )

    model_type = normalize_model_type(
        context.get(
            "model_type",
            context.get(
                "experiment_config",
                {},
            ).get(
                "model_type"
            ),
        )
    )

    # --------------------------------------------------------------
    # U-Net
    # --------------------------------------------------------------

    if model_type == "unet":

        model = _build_from_module(
            module_name=(
                "src.models.unet"
            ),

            candidate_names=(
                "build_unet",
                "create_unet",
                "build_model",
            ),

            context=context,
        )

    # --------------------------------------------------------------
    # DeepLabV3+
    # --------------------------------------------------------------

    elif model_type == "deeplabv3":

        model = _build_from_module(
            module_name=(
                "src.models.deeplabv3"
            ),

            candidate_names=(
                "build_deeplabv3",
                "build_deeplabv3plus",
                "create_deeplabv3",
                "build_model",
            ),

            context=context,
        )

    # --------------------------------------------------------------
    # Road SAM-LoRA
    # --------------------------------------------------------------

    elif model_type == "sam_lora":

        from src.models.sam_lora.factory import (
            build_sam_lora_from_context,
        )

        model = build_sam_lora_from_context(
            context=context,

            return_info=False,
        )

    # --------------------------------------------------------------
    # ArcGIS models cannot be built as native PyTorch models.
    # --------------------------------------------------------------

    elif model_type in {
        "connectnet",
        "multitask_road_extractor",
    }:

        raise RuntimeError(
            f"{model_type} uses backend='arcgis_learn', not the "
            "native PyTorch model factory."
        )

    else:

        raise ValueError(
            "Unsupported Road model_type:\n"
            f"{model_type}\n\n"
            "Supported native PyTorch models:\n"
            "- unet\n"
            "- deeplabv3\n"
            "- sam_lora"
        )

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "Native Road model builder did not return torch.nn.Module.\n"
            f"Received: {type(model).__name__}"
        )

    return model


# ---------------------------------------------------------------------
# ArcGIS model factory
# ---------------------------------------------------------------------

def build_arcgis_model(
    context: Dict[str, Any],
    data,
):
    """
    Build ConnectNet or MultiTaskRoadExtractor from the real Esri API.
    """

    model_type = normalize_model_type(
        context.get(
            "model_type",
            context.get(
                "experiment_config",
                {},
            ).get(
                "model_type"
            ),
        )
    )

    if model_type == "connectnet":

        from src.models.connectnet import (
            build_connectnet,
        )

        return build_connectnet(
            data=data,

            model_config=(
                context[
                    "model_config"
                ]
            ),
        )

    if model_type == "multitask_road_extractor":

        from src.models.multitask_road_extractor import (
            build_multitask_road_extractor,
        )

        return build_multitask_road_extractor(
            data=data,

            model_config=(
                context[
                    "model_config"
                ]
            ),
        )

    raise ValueError(
        "build_arcgis_model() supports only:\n"
        "- connectnet\n"
        "- multitask_road_extractor\n\n"
        f"Received: {model_type}"
    )


# ---------------------------------------------------------------------
# Unified backend-aware factory
# ---------------------------------------------------------------------

def build_model(
    context: Dict[str, Any],
    data=None,
):
    """
    Backend-aware Road model builder.
    """

    model_type = normalize_model_type(
        context.get(
            "model_type",
            context.get(
                "experiment_config",
                {},
            ).get(
                "model_type"
            ),
        )
    )

    if model_type in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:

        return build_pytorch_model(
            context
        )

    if model_type in {
        "connectnet",
        "multitask_road_extractor",
    }:

        if data is None:

            raise ValueError(
                f"{model_type} requires ArcGIS Learn prepared data."
            )

        return build_arcgis_model(
            context=context,

            data=data,
        )

    raise ValueError(
        f"Unsupported Road model: {model_type}"
    )


# ---------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------

def get_model_parameter_summary(
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Return common PyTorch model parameter counts.
    """

    total = int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )
    )

    trainable = int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
    )

    summary = {
        "total_parameters":
            total,

        "trainable_parameters":
            trainable,

        "frozen_parameters":
            int(
                total
                - trainable
            ),

        "trainable_percentage":
            (
                float(
                    trainable
                    / total
                    * 100.0
                )
                if total
                else 0.0
            ),
    }

    # --------------------------------------------------------------
    # Add SAM-specific provenance when available.
    # --------------------------------------------------------------

    sam_info = getattr(
        model,
        "road_sam_lora_build_info",
        None,
    )

    if isinstance(
        sam_info,
        dict,
    ):

        summary[
            "sam_lora"
        ] = dict(
            sam_info
        )

    return summary


# ---------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------

def extract_model_logits(
    output,
) -> torch.Tensor:
    """
    Normalize native model output to one segmentation logits tensor.
    """

    if torch.is_tensor(
        output
    ):

        return output

    if isinstance(
        output,
        dict,
    ):

        for key in (
            "out",
            "logits",
            "prediction",
            "predictions",
            "mask",
            "masks",
        ):

            value = output.get(
                key
            )

            if torch.is_tensor(
                value
            ):

                return value

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):

        for value in output:

            if torch.is_tensor(
                value
            ):

                return value

    raise TypeError(
        "Could not extract logits from model output.\n"
        f"Output type: {type(output).__name__}"
    )


# ---------------------------------------------------------------------
# Architecture verification
# ---------------------------------------------------------------------

def verify_pytorch_model_output(
    model: nn.Module,
    tile_size: int,
    input_channels: int = 3,
    batch_size: int = 1,
    device: Optional[
        str | torch.device
    ] = None,
) -> Dict[str, Any]:
    """
    Lightweight native-model shape verification.

    Expected:
        input  -> [B, 3, H, W]
        output -> [B, 1, H, W]
    """

    tile_size = int(
        tile_size
    )

    batch_size = int(
        batch_size
    )

    input_channels = int(
        input_channels
    )

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0."
        )

    if device is None:

        resolved_device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif isinstance(
        device,
        torch.device,
    ):

        resolved_device = device

    else:

        resolved_device = torch.device(
            device
        )

    model = model.to(
        resolved_device
    )

    model.eval()

    sample = torch.zeros(
        batch_size,
        input_channels,
        tile_size,
        tile_size,

        dtype=torch.float32,

        device=resolved_device,
    )

    with torch.no_grad():

        output = model(
            sample
        )

        logits = extract_model_logits(
            output
        )

    expected_shape = (
        batch_size,
        1,
        tile_size,
        tile_size,
    )

    actual_shape = tuple(
        int(
            value
        )
        for value
        in logits.shape
    )

    if actual_shape != expected_shape:

        raise ValueError(
            "Unexpected Road model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {actual_shape}"
        )

    parameters = get_model_parameter_summary(
        model
    )

    return {
        "input_shape":
            tuple(
                sample.shape
            ),

        "output_shape":
            actual_shape,

        "device":
            str(
                resolved_device
            ),

        **parameters,
    }
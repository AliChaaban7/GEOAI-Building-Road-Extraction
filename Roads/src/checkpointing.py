"""
Checkpoint utilities for Approach 2 - Road Extraction.

Supported checkpoint styles
---------------------------
U-Net / DeepLabV3+
    Full model state_dict.

SAM-LoRA
    Trainable-only checkpoint:
        - Q/V LoRA adapters
        - SAM mask decoder
        - exact model configuration
        - exact experiment architecture information
        - official pretrained SAM checkpoint reference

Why SAM-LoRA is different
-------------------------
The frozen official SAM ViT-B parameters should not be duplicated inside
every Road experiment checkpoint.

Instead:

    official pretrained SAM
              +
    exact LoRA architecture
              +
    trained LoRA + mask decoder
              =
    reconstructed trained Road SAM-LoRA

This also keeps Master Optuna safe when rank/alpha differ between trials.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Safe-ish loading for trusted thesis checkpoints
# ---------------------------------------------------------------------

def load_torch_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
):
    """
    Load a trusted thesis checkpoint.

    Explicit weights_only=False supports dictionaries containing
    optimizer/config metadata on modern PyTorch while retaining
    compatibility with older ArcGIS Pro PyTorch builds.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

    try:

        return torch.load(
            checkpoint_path,

            map_location=map_location,

            weights_only=False,
        )

    except TypeError:

        return torch.load(
            checkpoint_path,

            map_location=map_location,
        )


# ---------------------------------------------------------------------
# Model type
# ---------------------------------------------------------------------

def _normalize_model_type(
    value,
) -> str:

    value = (
        str(
            value
        )
        .strip()
        .lower()
    )

    aliases = {
        "u-net":
            "unet",

        "u_net":
            "unet",

        "deeplab":
            "deeplabv3",

        "deeplabv3+":
            "deeplabv3",

        "deeplabv3plus":
            "deeplabv3",

        "sam-lora":
            "sam_lora",

        "sam lora":
            "sam_lora",

        "samlora":
            "sam_lora",
    }

    return aliases.get(
        value,
        value,
    )


def _model_type_from_context(
    context: Dict[str, Any],
) -> str:

    value = context.get(
        "model_type"
    )

    if value is None:

        value = context.get(
            "experiment_config",
            {},
        ).get(
            "model_type"
        )

    if value is None:

        raise RuntimeError(
            "Could not determine model_type from context."
        )

    return _normalize_model_type(
        value
    )


# ---------------------------------------------------------------------
# State dictionaries
# ---------------------------------------------------------------------

def _is_tensor_mapping(
    value,
) -> bool:

    if not isinstance(
        value,
        dict,
    ):

        return False

    if not value:

        return False

    return all(
        torch.is_tensor(
            item
        )
        for item
        in value.values()
    )


def clean_state_dict(
    state_dict: Dict[
        str,
        torch.Tensor,
    ],
) -> Dict[
    str,
    torch.Tensor,
]:
    """
    Remove DataParallel / DDP module prefix.
    """

    cleaned = {}

    for (
        key,
        value,
    ) in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len(
                    "module."
                ):
            ]

        cleaned[
            key
        ] = value

    return cleaned


def extract_model_state_dict(
    checkpoint,
) -> Dict[
    str,
    torch.Tensor,
]:
    """
    Extract a complete model state_dict from common checkpoint layouts.
    """

    if _is_tensor_mapping(
        checkpoint
    ):

        return clean_state_dict(
            checkpoint
        )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Unsupported checkpoint type."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "weights",
    ):

        candidate = checkpoint.get(
            key
        )

        if _is_tensor_mapping(
            candidate
        ):

            return clean_state_dict(
                candidate
            )

    raise KeyError(
        "Could not locate full model state_dict.\n"
        f"Available keys: {list(checkpoint.keys())}"
    )


def extract_trainable_state_dict(
    model: nn.Module,
) -> Dict[
    str,
    torch.Tensor,
]:
    """
    Save only parameters that are trainable.

    Used primarily by Road SAM-LoRA.
    """

    trainable_names = {
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    }

    complete_state = (
        model.state_dict()
    )

    trainable_state = {}

    for (
        name,
        tensor,
    ) in complete_state.items():

        if name not in trainable_names:

            continue

        trainable_state[
            name
        ] = (
            tensor.detach()
            .cpu()
            .clone()
        )

    if not trainable_state:

        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    return trainable_state


# ---------------------------------------------------------------------
# Exact configuration saved in every checkpoint
# ---------------------------------------------------------------------

def _copy_config(
    context: Dict[str, Any],
    key: str,
) -> Dict[str, Any]:

    value = context.get(
        key,
        {},
    )

    if not isinstance(
        value,
        dict,
    ):

        return {}

    return deepcopy(
        value
    )


# ---------------------------------------------------------------------
# Build checkpoint payload
# ---------------------------------------------------------------------

def build_training_checkpoint_payload(
    model: nn.Module,
    context: Dict[str, Any],
    epoch: int,
    validation_metrics: Optional[
        Dict[str, Any]
    ] = None,
    train_metrics: Optional[
        Dict[str, Any]
    ] = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build one reproducible training checkpoint.

    SAM-LoRA:
        saves trainable parameters only.

    U-Net / DeepLabV3+:
        saves complete model state.
    """

    model_type = (
        _model_type_from_context(
            context
        )
    )

    validation_metrics = deepcopy(
        validation_metrics
        or {}
    )

    train_metrics = deepcopy(
        train_metrics
        or {}
    )

    payload = {
        "checkpoint_version":
            2,

        "epoch":
            int(
                epoch
            ),

        "model_type":
            model_type,

        "model_config":
            _copy_config(
                context,
                "model_config",
            ),

        "experiment_config":
            _copy_config(
                context,
                "experiment_config",
            ),

        "validation_metrics":
            validation_metrics,

        "train_metrics":
            train_metrics,
    }

    # --------------------------------------------------------------
    # SAM-LoRA
    # --------------------------------------------------------------

    if model_type == "sam_lora":

        trainable_state = (
            extract_trainable_state_dict(
                model
            )
        )

        payload[
            "checkpoint_format"
        ] = (
            "road_sam_lora_trainable_v1"
        )

        payload[
            "trainable_state_dict"
        ] = trainable_state

        payload[
            "trainable_parameter_names"
        ] = sorted(
            trainable_state.keys()
        )

        build_info = getattr(
            model,
            "road_sam_lora_build_info",
            None,
        )

        if isinstance(
            build_info,
            dict,
        ):

            payload[
                "sam_lora_build_info"
            ] = deepcopy(
                build_info
            )

            official_path = (
                build_info.get(
                    "official_sam_checkpoint"
                )
            )

            if official_path:

                payload[
                    "official_sam_checkpoint"
                ] = str(
                    official_path
                )

    # --------------------------------------------------------------
    # Standard semantic model
    # --------------------------------------------------------------

    else:

        payload[
            "checkpoint_format"
        ] = (
            "road_full_model_v2"
        )

        payload[
            "model_state_dict"
        ] = {
            key:
                tensor.detach()
                .cpu()
                .clone()

            for (
                key,
                tensor,
            ) in model.state_dict().items()
        }

    # --------------------------------------------------------------
    # Optimizer / scheduler / AMP state
    # --------------------------------------------------------------

    if optimizer is not None:

        payload[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:

        try:

            payload[
                "scheduler_state_dict"
            ] = scheduler.state_dict()

        except Exception:

            pass

    if scaler is not None:

        try:

            payload[
                "scaler_state_dict"
            ] = scaler.state_dict()

        except Exception:

            pass

    if extra:

        payload[
            "extra"
        ] = deepcopy(
            extra
        )

    return payload


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    context: Dict[str, Any],
    epoch: int,
    validation_metrics: Optional[
        Dict[str, Any]
    ] = None,
    train_metrics: Optional[
        Dict[str, Any]
    ] = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> Path:
    """
    Build and save one checkpoint.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        build_training_checkpoint_payload(
            model=model,

            context=context,

            epoch=epoch,

            validation_metrics=(
                validation_metrics
            ),

            train_metrics=(
                train_metrics
            ),

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            extra=extra,
        )
    )

    torch.save(
        payload,
        checkpoint_path,
    )

    return checkpoint_path


# ---------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------

def _load_json_if_exists(
    path: str | Path,
) -> Optional[
    Dict[str, Any]
]:

    path = Path(
        path
    )

    if not path.exists():

        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise TypeError(
            "Expected JSON object:\n"
            f"{path}"
        )

    return data


# ---------------------------------------------------------------------
# Embedded model config
# ---------------------------------------------------------------------

def _embedded_model_config(
    checkpoint,
) -> Optional[
    Dict[str, Any]
]:

    if not isinstance(
        checkpoint,
        dict,
    ):

        return None

    for key in (
        "resolved_model_config",
        "model_config",
    ):

        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):

            return deepcopy(
                value
            )

    for container_key in (
        "context",
        "config_snapshot",
        "experiment_snapshot",
    ):

        container = checkpoint.get(
            container_key
        )

        if not isinstance(
            container,
            dict,
        ):

            continue

        value = container.get(
            "model_config"
        )

        if isinstance(
            value,
            dict,
        ):

            return deepcopy(
                value
            )

    return None


def _embedded_experiment_config(
    checkpoint,
) -> Optional[
    Dict[str, Any]
]:

    if not isinstance(
        checkpoint,
        dict,
    ):

        return None

    value = checkpoint.get(
        "experiment_config"
    )

    if isinstance(
        value,
        dict,
    ):

        return deepcopy(
            value
        )

    for container_key in (
        "context",
        "config_snapshot",
        "experiment_snapshot",
    ):

        container = checkpoint.get(
            container_key
        )

        if not isinstance(
            container,
            dict,
        ):

            continue

        value = container.get(
            "experiment_config"
        )

        if isinstance(
            value,
            dict,
        ):

            return deepcopy(
                value
            )

    return None


# ---------------------------------------------------------------------
# Model-config resolution
# ---------------------------------------------------------------------

def resolve_checkpoint_model_config(
    checkpoint,
    context: Dict[str, Any],
    checkpoint_path: str | Path,
) -> Tuple[
    Dict[str, Any],
    str,
]:
    """
    Resolve exact architecture configuration.
    """

    embedded = (
        _embedded_model_config(
            checkpoint
        )
    )

    if embedded is not None:

        return (
            embedded,
            "checkpoint",
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    candidate_files = []

    output_folder = context.get(
        "output_folder"
    )

    if output_folder:

        candidate_files.append(
            Path(
                output_folder
            )
            / "resolved_master_winner_model_config.json"
        )

    candidate_files.extend(
        [
            checkpoint_path.parent
            / "resolved_master_winner_model_config.json",

            checkpoint_path.parent.parent
            / "resolved_master_winner_model_config.json",
        ]
    )

    seen = set()

    for path in candidate_files:

        try:

            identity = str(
                path.resolve()
            )

        except Exception:

            identity = str(
                path
            )

        if identity in seen:

            continue

        seen.add(
            identity
        )

        config = _load_json_if_exists(
            path
        )

        if config is not None:

            return (
                config,
                str(
                    path
                ),
            )

    fallback = context.get(
        "model_config"
    )

    if not isinstance(
        fallback,
        dict,
    ):

        raise RuntimeError(
            "Could not resolve checkpoint model configuration."
        )

    return (
        deepcopy(
            fallback
        ),
        "current_experiment_context",
    )


# ---------------------------------------------------------------------
# Model type from checkpoint
# ---------------------------------------------------------------------

def resolve_checkpoint_model_type(
    checkpoint,
    context: Dict[str, Any],
) -> str:

    if isinstance(
        checkpoint,
        dict,
    ):

        value = checkpoint.get(
            "model_type"
        )

        if value:

            return _normalize_model_type(
                value
            )

        experiment = (
            _embedded_experiment_config(
                checkpoint
            )
        )

        if experiment:

            value = experiment.get(
                "model_type"
            )

            if value:

                return _normalize_model_type(
                    value
                )

    return _model_type_from_context(
        context
    )


# ---------------------------------------------------------------------
# Reconstruction context
# ---------------------------------------------------------------------

def build_checkpoint_context(
    checkpoint,
    context: Dict[str, Any],
    checkpoint_path: str | Path,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Reconstruct exact architecture context without mutating the caller.
    """

    resolved_context = deepcopy(
        context
    )

    (
        model_config,
        model_config_source,
    ) = resolve_checkpoint_model_config(
        checkpoint=checkpoint,

        context=context,

        checkpoint_path=checkpoint_path,
    )

    model_type = (
        resolve_checkpoint_model_type(
            checkpoint,
            context,
        )
    )

    # --------------------------------------------------------------
    # SAM-LoRA official checkpoint provenance
    # --------------------------------------------------------------

    if (
        model_type
        == "sam_lora"
        and isinstance(
            checkpoint,
            dict,
        )
    ):

        official_path = checkpoint.get(
            "official_sam_checkpoint"
        )

        if official_path is None:

            build_info = checkpoint.get(
                "sam_lora_build_info",
                {},
            )

            if isinstance(
                build_info,
                dict,
            ):

                official_path = (
                    build_info.get(
                        "official_sam_checkpoint"
                    )
                )

        if official_path:

            # Put it directly into model_config so the Road SAM
            # resolver can rebuild the exact base model.
            model_config[
                "official_sam_checkpoint"
            ] = str(
                official_path
            )

    resolved_context[
        "model_config"
    ] = model_config

    resolved_context[
        "model_type"
    ] = model_type

    # --------------------------------------------------------------
    # Recover architecture-sensitive experiment values.
    # --------------------------------------------------------------

    embedded_experiment = (
        _embedded_experiment_config(
            checkpoint
        )
    )

    if embedded_experiment:

        current_experiment = deepcopy(
            resolved_context.get(
                "experiment_config",
                {},
            )
        )

        for key in (
            "model_type",
            "backbone",
            "tile_size",
            "input_channels",
            "output_channels",
        ):

            if key in embedded_experiment:

                current_experiment[
                    key
                ] = deepcopy(
                    embedded_experiment[
                        key
                    ]
                )

        resolved_context[
            "experiment_config"
        ] = current_experiment

    # --------------------------------------------------------------
    # Refresh registry metadata when available.
    # --------------------------------------------------------------

    try:

        from src.pipeline.model_registry import (
            get_model_backend,
            get_model_spec,
        )

        resolved_context[
            "model_spec"
        ] = get_model_spec(
            model_type
        )

        resolved_context[
            "model_backend"
        ] = get_model_backend(
            model_type
        )

    except Exception:

        pass

    metadata = {
        "model_type":
            model_type,

        "model_config_source":
            model_config_source,

        "checkpoint_format":
            (
                checkpoint.get(
                    "checkpoint_format"
                )
                if isinstance(
                    checkpoint,
                    dict,
                )
                else "raw_state_dict"
            ),

        "checkpoint_has_embedded_model_config":
            (
                _embedded_model_config(
                    checkpoint
                )
                is not None
            ),
    }

    return (
        resolved_context,
        metadata,
    )


# ---------------------------------------------------------------------
# Load trainable-only state
# ---------------------------------------------------------------------

def _load_trainable_state(
    model: nn.Module,
    trainable_state: Dict[
        str,
        torch.Tensor,
    ],
    strict: bool,
) -> Dict[str, Any]:
    """
    Load SAM-LoRA trainable weights over freshly reconstructed official
    SAM base parameters.
    """

    trainable_state = clean_state_dict(
        trainable_state
    )

    full_state = (
        model.state_dict()
    )

    expected_trainable = {
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    }

    provided = set(
        trainable_state.keys()
    )

    missing_trainable = sorted(
        expected_trainable
        - provided
    )

    unexpected = sorted(
        provided
        - set(
            full_state.keys()
        )
    )

    if strict and (
        missing_trainable
        or unexpected
    ):

        raise RuntimeError(
            "SAM-LoRA trainable checkpoint mismatch.\n"
            f"Missing trainable parameters: {missing_trainable}\n"
            f"Unexpected parameters: {unexpected}"
        )

    shape_errors = []

    for (
        name,
        value,
    ) in trainable_state.items():

        if name not in full_state:

            continue

        if tuple(
            value.shape
        ) != tuple(
            full_state[
                name
            ].shape
        ):

            shape_errors.append(
                (
                    name,
                    tuple(
                        value.shape
                    ),
                    tuple(
                        full_state[
                            name
                        ].shape
                    ),
                )
            )

    if shape_errors:

        raise RuntimeError(
            "SAM-LoRA checkpoint tensor-shape mismatch:\n"
            + "\n".join(
                f"{name}: checkpoint={saved}, model={expected}"
                for (
                    name,
                    saved,
                    expected,
                ) in shape_errors
            )
        )

    # --------------------------------------------------------------
    # Merge trained adapters/decoder over official SAM weights.
    # --------------------------------------------------------------

    for (
        name,
        value,
    ) in trainable_state.items():

        if name in full_state:

            full_state[
                name
            ] = value

    model.load_state_dict(
        full_state,
        strict=True,
    )

    return {
        "tensor_count":
            int(
                len(
                    trainable_state
                )
            ),

        "missing_trainable_keys":
            missing_trainable,

        "unexpected_keys":
            unexpected,

        "strict":
            bool(
                strict
            ),

        "trainable_only":
            True,
    }


# ---------------------------------------------------------------------
# Public state loader
# ---------------------------------------------------------------------

def load_model_state(
    model: nn.Module,
    checkpoint,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load full or SAM-LoRA trainable-only checkpoint.
    """

    if isinstance(
        checkpoint,
        dict,
    ):

        trainable_state = checkpoint.get(
            "trainable_state_dict"
        )

        if _is_tensor_mapping(
            trainable_state
        ):

            return _load_trainable_state(
                model=model,

                trainable_state=(
                    trainable_state
                ),

                strict=strict,
            )

    state_dict = (
        extract_model_state_dict(
            checkpoint
        )
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=bool(
            strict
        ),
    )

    missing_keys = list(
        getattr(
            incompatible,
            "missing_keys",
            [],
        )
    )

    unexpected_keys = list(
        getattr(
            incompatible,
            "unexpected_keys",
            [],
        )
    )

    return {
        "tensor_count":
            int(
                len(
                    state_dict
                )
            ),

        "missing_keys":
            missing_keys,

        "unexpected_keys":
            unexpected_keys,

        "strict":
            bool(
                strict
            ),

        "trainable_only":
            False,
    }
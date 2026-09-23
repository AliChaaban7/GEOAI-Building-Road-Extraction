"""
Buildings/src/models/sam_lora/inference.py

Inference/checkpoint utilities for SAM-LoRA building segmentation.

Purpose
-------
This module reconstructs the trained SAM-LoRA model from:

    1. the official pretrained SAM checkpoint
    2. the SAM-LoRA JSON configuration
    3. the thesis trainable checkpoint (LoRA + mask decoder)

It is used by:
- validation threshold search
- validation post-processing
- full-scene inference
- final independent testing

Important compatibility behavior
--------------------------------
Older SAM-LoRA builder versions returned:

    model, build_info

The current factory may return:

    model

directly.

This file supports BOTH return styles so the rest of the Buildings module
does not break when loading the trained SAM-LoRA checkpoint.

The external test area is not accessed here. This module only loads a model
and performs tensor inference on data supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import (
    build_sam_lora_from_config,
    load_sam_lora_config,
    resolve_sam_checkpoint_path,
)

from .trainer import (
    load_sam_lora_trainable_checkpoint,
)


# ============================================================
# LOAD INFO
# ============================================================

@dataclass
class SAMLoRALoadInfo:
    """
    Metadata describing a reconstructed trained SAM-LoRA model.
    """

    config_path: Path
    checkpoint_path: Path
    sam_checkpoint_path: Path

    backbone: str
    device: str

    checkpoint_format: str

    epoch: Optional[int]
    best_validation_iou: Optional[float]

    resolved_learning_rate: Optional[float]
    learning_rate_mode: Optional[str]

    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float

    loaded_tensor_count: int
    strict_trainable_check: bool

    # --------------------------------------------------------
    # Compatibility aliases used by older callers.
    # --------------------------------------------------------

    @property
    def best_val_iou(self) -> Optional[float]:
        return self.best_validation_iou

    @property
    def checkpoint_epoch(self) -> Optional[int]:
        return self.epoch

    @property
    def trained_epoch(self) -> Optional[int]:
        """
        Backward-compatible alias expected by run_threshold_search.py.
        """
        return self.epoch

    @property
    def validation_metrics(self) -> Dict[str, float]:
        """
        Backward-compatible validation-metrics view expected by the
        shared threshold-search pipeline.

        The best checkpoint stores the selected validation IoU.  Expose
        that value using the common metric keys used by the shared scripts.
        """
        if self.best_validation_iou is None:
            return {}

        value = float(self.best_validation_iou)

        return {
            "iou": value,
            "val_iou": value,
            "validation_iou": value,
        }

    @property
    def pretrained_sam_checkpoint_path(self) -> Path:
        """
        Backward-compatible alias for the official pretrained SAM
        checkpoint path expected by run_threshold_search.py.
        """
        return self.sam_checkpoint_path

    @property
    def total_parameter_count(self) -> int:
        return self.total_parameters

    @property
    def trainable_parameter_count(self) -> int:
        return self.trainable_parameters

    @property
    def official_sam_checkpoint(self) -> Path:
        return self.sam_checkpoint_path

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)

        for key in (
            "config_path",
            "checkpoint_path",
            "sam_checkpoint_path",
        ):
            payload[key] = str(
                payload[key]
            )

        return payload


# ============================================================
# GENERIC HELPERS
# ============================================================

def _normalize_backbone_name(
    value: Any,
) -> str:
    backbone = str(
        value
    ).strip().lower().replace(
        "-",
        "_",
    )

    aliases = {
        "vitb": "vit_b",
        "vit_b": "vit_b",
        "b": "vit_b",

        "vitl": "vit_l",
        "vit_l": "vit_l",
        "l": "vit_l",

        "vith": "vit_h",
        "vit_h": "vit_h",
        "h": "vit_h",
    }

    return aliases.get(
        backbone,
        backbone,
    )


def _resolve_device(
    device: Optional[
        str | torch.device
    ],
) -> torch.device:
    if device is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    resolved = torch.device(
        device
    )

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested for SAM-LoRA inference, "
            "but torch.cuda.is_available() is False."
        )

    return resolved


def _resolve_built_model(
    build_result: Any,
) -> nn.Module:
    """
    Resolve SAM-LoRA builder return styles.

    Supported:
        model

        (model, build_info)

        [model, build_info]

        object with .model / .segmenter / .network

    This directly fixes the compatibility error:

        TypeError:
        cannot unpack non-iterable SAMLoRABuildingSegmenter object
    """

    candidate = build_result

    # --------------------------------------------------------
    # Tuple/list compatibility.
    # --------------------------------------------------------
    if isinstance(
        candidate,
        (tuple, list),
    ):
        if len(candidate) == 0:
            raise RuntimeError(
                "build_sam_lora_from_config() returned "
                "an empty tuple/list."
            )

        module_candidates = [
            item
            for item in candidate
            if isinstance(
                item,
                nn.Module,
            )
        ]

        if module_candidates:
            candidate = (
                module_candidates[0]
            )
        else:
            candidate = (
                candidate[0]
            )

    # --------------------------------------------------------
    # Object-wrapper compatibility.
    # --------------------------------------------------------
    if not isinstance(
        candidate,
        nn.Module,
    ):
        for attribute in (
            "model",
            "segmenter",
            "network",
        ):
            value = getattr(
                candidate,
                attribute,
                None,
            )

            if isinstance(
                value,
                nn.Module,
            ):
                candidate = value
                break

    if not isinstance(
        candidate,
        nn.Module,
    ):
        raise TypeError(
            "build_sam_lora_from_config() did not "
            "produce a torch.nn.Module. "
            f"Received: "
            f"{type(build_result).__name__}"
        )

    return candidate


def _extract_logits(
    model_output: Any,
) -> torch.Tensor:
    """
    Normalize common semantic-segmentation output styles to:

        [B, 1, H, W] logits
    """

    if torch.is_tensor(
        model_output
    ):
        logits = model_output

    elif isinstance(
        model_output,
        dict,
    ):
        logits = None

        for key in (
            "logits",
            "out",
            "mask_logits",
            "masks",
        ):
            candidate = model_output.get(
                key,
                None,
            )

            if torch.is_tensor(
                candidate
            ):
                logits = candidate
                break

    else:
        logits = None

        for attribute in (
            "logits",
            "out",
            "mask_logits",
            "masks",
        ):
            candidate = getattr(
                model_output,
                attribute,
                None,
            )

            if torch.is_tensor(
                candidate
            ):
                logits = candidate
                break

    if not torch.is_tensor(
        logits
    ):
        raise TypeError(
            "Could not extract a tensor from "
            "SAM-LoRA model output."
        )

    if logits.ndim == 3:
        logits = logits.unsqueeze(
            1
        )

    if logits.ndim != 4:
        raise ValueError(
            "SAM-LoRA output must be 4D "
            "[B,C,H,W]. "
            f"Received shape: "
            f"{tuple(logits.shape)}"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "SAM-LoRA building segmentation "
            "expects exactly one output channel. "
            f"Received: "
            f"{tuple(logits.shape)}"
        )

    return logits


def _validate_input_tensor(
    images: torch.Tensor,
) -> None:
    if not torch.is_tensor(
        images
    ):
        raise TypeError(
            "SAM-LoRA inference expects "
            "a torch.Tensor."
        )

    if images.ndim != 4:
        raise ValueError(
            "SAM-LoRA inference images must have "
            "shape [B,3,H,W]. "
            f"Received: "
            f"{tuple(images.shape)}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            "SAM-LoRA inference is RGB-only. "
            "Expected 3 channels but received "
            f"{images.shape[1]}."
        )

    if not torch.is_floating_point(
        images
    ):
        raise TypeError(
            "SAM-LoRA input must be floating point "
            "with values in [0,1]."
        )


def _parameter_counts(
    model: nn.Module,
) -> tuple[int, int, float]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    percentage = (
        100.0
        * float(trainable)
        / max(
            int(total),
            1,
        )
    )

    return (
        int(total),
        int(trainable),
        float(percentage),
    )


def _number_or_none(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(
            value
        )
    except Exception:
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def _int_or_none(
    value: Any,
) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(
            value
        )
    except Exception:
        return None


def _resolve_effective_config_path(
    requested_config_path: str | Path,
    checkpoint_path: str | Path,
    prefer_experiment_config: bool = True,
) -> Path:
    """
    Resolve the SAM-LoRA configuration that must be used to reconstruct
    a trained checkpoint.

    Why this is necessary
    ---------------------
    Baseline SAM-LoRA uses the canonical:

        Buildings/config/models/sam_lora.json

    but Focused Optuna and Master Optuna may produce a checkpoint whose
    training configuration differs from that canonical file.

    In particular, Master Optuna may change:
        - LoRA rank
        - LoRA alpha

    and Focused/Master Optuna may change:
        - learning rate
        - weight decay
        - BCE/Dice balance

    Loading a rank-16 checkpoint into a rank-8 model is invalid. Therefore,
    when an experiment-specific SAM-LoRA config exists beside the promoted
    checkpoint, inference must prefer that exact configuration.

    Priority
    --------
    1. sam_lora_selected_config.json
       Master Optuna promoted winner.

    2. sam_lora_optuna_final_config.json
       Focused Optuna final retraining.

    3. sam_lora_master_config.json
       Direct Master Optuna trial checkpoint.

    4. sam_lora_trial_config.json
       Direct Focused Optuna trial checkpoint.

    5. requested_config_path
       Normal baseline / augmentation path.

    Only explicitly SAM-LoRA-named files are auto-selected. Generic files
    such as model_config_used.json are deliberately not auto-selected here,
    preventing accidental use of an unrelated model/config artifact.
    """

    requested = Path(
        requested_config_path
    ).resolve()

    checkpoint = Path(
        checkpoint_path
    ).resolve()

    if not bool(
        prefer_experiment_config
    ):
        return requested

    candidate_names = (
        "sam_lora_selected_config.json",
        "sam_lora_optuna_final_config.json",
        "sam_lora_master_config.json",
        "sam_lora_trial_config.json",
    )

    # Usually best_model.pth and the selected config are in the same
    # experiment folder. Search a small number of parents too so this remains
    # safe if a caller points at a checkpoint inside a nested stage folder.
    search_folders = []

    current = checkpoint.parent

    for _ in range(3):
        if current not in search_folders:
            search_folders.append(
                current
            )

        if current.parent == current:
            break

        current = current.parent

    for folder in search_folders:
        for name in candidate_names:
            candidate = (
                folder
                / name
            )

            if candidate.exists():
                return candidate.resolve()

    return requested


# ============================================================
# TRAINED MODEL LOADER
# ============================================================

def load_trained_sam_lora(
    config_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    backbone: Optional[str] = None,
    device: Optional[
        str | torch.device
    ] = None,
    map_location: Optional[
        str | torch.device
    ] = None,
    strict: bool = True,
    verbose: bool = True,
    verbose_model_build: bool = False,
    return_info: bool = True,
    checkpoint: Optional[str | Path] = None,
    trained_checkpoint: Optional[str | Path] = None,
    trained_checkpoint_path: Optional[str | Path] = None,
    best_model_path: Optional[str | Path] = None,
    model_checkpoint: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    prefer_experiment_config: bool = True,
    **extra_kwargs: Any,
):
    """
    Rebuild and load a trained SAM-LoRA model.

    Parameters
    ----------
    config_path:
        Fallback SAM-LoRA configuration path, normally:

            Buildings/config/models/sam_lora.json

        If the trained checkpoint belongs to Focused Optuna or Master Optuna
        and an experiment-specific SAM-LoRA config exists beside that
        checkpoint, the experiment-specific config is preferred automatically.

    checkpoint_path:
        Path to the trained SAM-LoRA checkpoint, normally:

            outputs/experiments/<experiment_id>/best_model.pth

        For backward compatibility, the same path may also be supplied
        using checkpoint, trained_checkpoint, trained_checkpoint_path,
        best_model_path, model_checkpoint, or model_path.

    backbone:
        Optional backbone override:
            vit_b
            vit_l
            vit_h

        If omitted, use config["default_backbone"].

    device:
        Final inference device.

    map_location:
        Device used while reading the trainable checkpoint.
        Defaults to CPU for safer/lighter checkpoint loading.

    strict:
        Verify that every trainable parameter required by the
        reconstructed model exists in the trainable checkpoint.

    return_info:
        True:
            return (model, SAMLoRALoadInfo)

        False:
            return model

    prefer_experiment_config:
        True by default. If an exact experiment-specific SAM-LoRA config is
        found beside the trained checkpoint, use it instead of the canonical
        fallback config. Set False only for an explicit diagnostic/rollback
        reconstruction.

    Returns
    -------
    model
        or
    (model, load_info)

    Notes
    -----
    The official SAM weights and the thesis trainable checkpoint are
    intentionally separate.

    Reconstruction is therefore:

        Official SAM
            +
        same LoRA structure
            +
        trained LoRA / mask-decoder weights
            =
        trained SAM-LoRA model
    """

    # ========================================================
    # CHECKPOINT ARGUMENT COMPATIBILITY
    #
    # Different integration scripts in the Buildings module have used
    # slightly different names for the same trained SAM-LoRA checkpoint.
    # Accept all known aliases so threshold search, post-processing,
    # inference, and final testing can share this loader safely.
    # ========================================================

    checkpoint_candidates = [
        checkpoint_path,
        checkpoint,
        trained_checkpoint,
        trained_checkpoint_path,
        best_model_path,
        model_checkpoint,
        model_path,
        extra_kwargs.get("best_checkpoint"),
        extra_kwargs.get("best_checkpoint_path"),
        extra_kwargs.get("checkpoint_file"),
        extra_kwargs.get("trained_model_path"),
    ]

    resolved_checkpoint_value = next(
        (
            value
            for value in checkpoint_candidates
            if value is not None
            and str(value).strip() != ""
        ),
        None,
    )

    if resolved_checkpoint_value is None:
        raise ValueError(
            "A trained SAM-LoRA checkpoint path is required. "
            "Accepted argument names include: checkpoint_path, checkpoint, "
            "trained_checkpoint, trained_checkpoint_path, best_model_path, "
            "model_checkpoint, and model_path."
        )

    resolved_checkpoint_path = Path(
        resolved_checkpoint_value
    ).resolve()

    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(
            "SAM-LoRA trained checkpoint was not found:\n"
            f"{resolved_checkpoint_path}"
        )

    requested_config_path = Path(
        config_path
    ).resolve()

    resolved_config_path = (
        _resolve_effective_config_path(
            requested_config_path=requested_config_path,
            checkpoint_path=resolved_checkpoint_path,
            prefer_experiment_config=bool(
                prefer_experiment_config
            ),
        )
    )

    if not resolved_config_path.exists():
        raise FileNotFoundError(
            "SAM-LoRA configuration was not found:\n"
            f"{resolved_config_path}"
        )

    resolved_device = _resolve_device(
        device
    )

    if map_location is None:
        resolved_map_location = (
            torch.device("cpu")
        )
    else:
        resolved_map_location = (
            torch.device(
                map_location
            )
        )

    config = load_sam_lora_config(
        resolved_config_path
    )

    resolved_backbone = _normalize_backbone_name(
        backbone
        if backbone is not None
        else config.get(
            "default_backbone",
            "vit_b",
        )
    )

    official_sam_checkpoint = (
        resolve_sam_checkpoint_path(
            config=config,
            backbone=resolved_backbone,
            config_path=resolved_config_path,
        )
    )

    if verbose:
        print()
        print("=" * 72)
        print("LOAD TRAINED SAM-LoRA")
        print("=" * 72)
        print(
            "Backbone                   :",
            resolved_backbone,
        )
        print(
            "SAM Configuration          :",
            resolved_config_path,
        )

        if (
            resolved_config_path
            != requested_config_path
        ):
            print(
                "Config Selection          :",
                "EXPERIMENT-SPECIFIC",
            )
            print(
                "Requested Fallback Config :",
                requested_config_path,
            )
        else:
            print(
                "Config Selection          :",
                "REQUESTED / CANONICAL",
            )
        print(
            "Official SAM Checkpoint    :",
            official_sam_checkpoint,
        )
        print(
            "Trainable Checkpoint       :",
            resolved_checkpoint_path,
        )
        print(
            "Inference Device           :",
            resolved_device,
        )
        print("=" * 72)

    # ========================================================
    # IMPORTANT COMPATIBILITY FIX
    #
    # Previous code did:
    #
    #     model, _ = build_sam_lora_from_config(...)
    #
    # which fails now because the current factory returns the
    # SAMLoRABuildingSegmenter directly.
    #
    # We resolve both model-only and tuple/list return styles.
    # ========================================================

    build_result = (
        build_sam_lora_from_config(
            config=config,
            backbone=resolved_backbone,
            device=resolved_device,
            config_path=resolved_config_path,
            verbose=bool(
                verbose_model_build
            ),
        )
    )

    model = _resolve_built_model(
        build_result
    )

    # --------------------------------------------------------
    # Load ONLY the trainable thesis checkpoint.
    # Frozen pretrained SAM parameters remain those loaded from
    # the official Meta checkpoint above.
    # --------------------------------------------------------

    checkpoint_info = (
        load_sam_lora_trainable_checkpoint(
            model=model,
            checkpoint_path=resolved_checkpoint_path,
            map_location=resolved_map_location,
            strict=bool(
                strict
            ),
            device=resolved_device,
            verbose=bool(
                verbose
            ),
        )
    )

    model = model.to(
        resolved_device
    )

    model.eval()

    (
        total_parameters,
        trainable_parameters,
        trainable_percentage,
    ) = _parameter_counts(
        model
    )

    epoch = _int_or_none(
        checkpoint_info.get(
            "epoch",
            None,
        )
    )

    best_validation_iou = (
        _number_or_none(
            checkpoint_info.get(
                "best_val_iou",
                None,
            )
        )
    )

    resolved_learning_rate = (
        _number_or_none(
            checkpoint_info.get(
                "resolved_learning_rate",
                None,
            )
        )
    )

    learning_rate_mode = (
        checkpoint_info.get(
            "learning_rate_mode",
            None,
        )
    )

    if learning_rate_mode is not None:
        learning_rate_mode = str(
            learning_rate_mode
        ).strip().lower()

    load_info = SAMLoRALoadInfo(
        config_path=(
            resolved_config_path
        ),
        checkpoint_path=(
            resolved_checkpoint_path
        ),
        sam_checkpoint_path=(
            official_sam_checkpoint
        ),

        backbone=(
            resolved_backbone
        ),
        device=str(
            resolved_device
        ),

        checkpoint_format=str(
            checkpoint_info.get(
                "checkpoint_format",
                "unknown",
            )
        ),

        epoch=epoch,
        best_validation_iou=(
            best_validation_iou
        ),

        resolved_learning_rate=(
            resolved_learning_rate
        ),
        learning_rate_mode=(
            learning_rate_mode
        ),

        total_parameters=(
            total_parameters
        ),
        trainable_parameters=(
            trainable_parameters
        ),
        trainable_percentage=(
            trainable_percentage
        ),

        loaded_tensor_count=int(
            checkpoint_info.get(
                "loaded_tensor_count",
                0,
            )
        ),
        strict_trainable_check=bool(
            checkpoint_info.get(
                "strict_trainable_check",
                strict,
            )
        ),
    )

    if verbose:
        print()
        print("=" * 72)
        print("TRAINED SAM-LoRA READY")
        print("=" * 72)
        print(
            "Checkpoint Epoch            :",
            (
                load_info.epoch
                if load_info.epoch
                is not None
                else "unknown"
            ),
        )

        if (
            load_info.best_validation_iou
            is not None
        ):
            print(
                "Best Validation IoU        :",
                f"{load_info.best_validation_iou * 100.0:.2f}%",
            )

        print(
            "Total Parameters            :",
            f"{total_parameters:,}",
        )
        print(
            "Trainable Parameters        :",
            f"{trainable_parameters:,}",
        )
        print(
            "Trainable Percentage        :",
            f"{trainable_percentage:.4f}%",
        )
        print(
            "Model Mode                  :",
            "EVAL",
        )
        print("=" * 72)
        print()

    if return_info:
        return (
            model,
            load_info,
        )

    return model


# Backward-compatible alias.
load_sam_lora_for_inference = (
    load_trained_sam_lora
)


# ============================================================
# TENSOR INFERENCE
# ============================================================

@torch.no_grad()
def predict_sam_lora_logits(
    model: nn.Module,
    images: torch.Tensor,
    device: Optional[
        str | torch.device
    ] = None,
    mixed_precision: bool = True,
    output_size: Optional[
        Tuple[int, int]
    ] = None,
) -> torch.Tensor:
    """
    Run SAM-LoRA tensor inference and return logits.

    Input
    -----
    images:
        [B,3,H,W]
        float RGB
        expected [0,1]

    Output
    ------
    logits:
        [B,1,H,W]
        or output_size if explicitly requested.
    """

    _validate_input_tensor(
        images
    )

    if device is None:
        try:
            resolved_device = next(
                model.parameters()
            ).device
        except StopIteration:
            resolved_device = (
                _resolve_device(
                    None
                )
            )
    else:
        resolved_device = (
            _resolve_device(
                device
            )
        )

    model = model.to(
        resolved_device
    )

    model.eval()

    input_height = int(
        images.shape[-2]
    )

    input_width = int(
        images.shape[-1]
    )

    images = images.to(
        resolved_device,
        non_blocking=True,
    )

    use_amp = bool(
        mixed_precision
        and resolved_device.type
        == "cuda"
    )

    if use_amp:
        try:
            amp_context = (
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
            )
        except Exception:
            amp_context = (
                torch.cuda.amp.autocast(
                    enabled=True
                )
            )
    else:
        from contextlib import (
            nullcontext
        )

        amp_context = (
            nullcontext()
        )

    with amp_context:
        model_output = model(
            images
        )

        logits = _extract_logits(
            model_output
        )

    if output_size is None:
        target_size = (
            input_height,
            input_width,
        )
    else:
        target_size = (
            int(
                output_size[0]
            ),
            int(
                output_size[1]
            ),
        )

    if (
        tuple(
            logits.shape[-2:]
        )
        != tuple(
            target_size
        )
    ):
        logits = F.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return logits


@torch.no_grad()
def predict_sam_lora_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    device: Optional[
        str | torch.device
    ] = None,
    mixed_precision: bool = True,
    output_size: Optional[
        Tuple[int, int]
    ] = None,
    move_to_cpu: bool = False,
) -> torch.Tensor:
    """
    Return building probabilities in [0,1].
    """

    logits = predict_sam_lora_logits(
        model=model,
        images=images,
        device=device,
        mixed_precision=mixed_precision,
        output_size=output_size,
    )

    probabilities = torch.sigmoid(
        logits
    )

    if move_to_cpu:
        probabilities = (
            probabilities.cpu()
        )

    return probabilities


@torch.no_grad()
def predict_sam_lora_binary_mask(
    model: nn.Module,
    images: torch.Tensor,
    threshold: float = 0.5,
    device: Optional[
        str | torch.device
    ] = None,
    mixed_precision: bool = True,
    output_size: Optional[
        Tuple[int, int]
    ] = None,
    move_to_cpu: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert SAM-LoRA probabilities to a binary building mask.
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
            "threshold must satisfy "
            "0 <= threshold <= 1. "
            f"Received: {threshold}"
        )

    probabilities = (
        predict_sam_lora_probabilities(
            model=model,
            images=images,
            device=device,
            mixed_precision=mixed_precision,
            output_size=output_size,
            move_to_cpu=False,
        )
    )

    mask = (
        probabilities
        >= threshold
    ).to(
        dtype=dtype
    )

    if move_to_cpu:
        mask = mask.cpu()

    return mask


# Common short alias.
predict_sam_lora_mask = (
    predict_sam_lora_binary_mask
)


# ============================================================
# SINGLE-TENSOR CONVENIENCE
# ============================================================

@torch.no_grad()
def predict_sam_lora(
    model: nn.Module,
    images: torch.Tensor,
    threshold: Optional[
        float
    ] = None,
    device: Optional[
        str | torch.device
    ] = None,
    mixed_precision: bool = True,
    move_to_cpu: bool = False,
) -> torch.Tensor:
    """
    Convenience function.

    threshold=None:
        return probabilities

    threshold=<float>:
        return binary mask
    """

    if threshold is None:
        return (
            predict_sam_lora_probabilities(
                model=model,
                images=images,
                device=device,
                mixed_precision=mixed_precision,
                move_to_cpu=move_to_cpu,
            )
        )

    return (
        predict_sam_lora_binary_mask(
            model=model,
            images=images,
            threshold=threshold,
            device=device,
            mixed_precision=mixed_precision,
            move_to_cpu=move_to_cpu,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "SAMLoRALoadInfo",
    "load_trained_sam_lora",
    "load_sam_lora_for_inference",
    "predict_sam_lora_logits",
    "predict_sam_lora_probabilities",
    "predict_sam_lora_binary_mask",
    "predict_sam_lora_mask",
    "predict_sam_lora",
    "_resolve_effective_config_path",
]

"""
Buildings/src/models/sam_lora/factory.py

Configuration-driven factory for the SAM-LoRA building-segmentation model.

This version supports BOTH learning-rate policies:

    1. fixed
       - uses training.learning_rate.value directly
       - skips the Auto-LR range test completely

    2. auto
       - keeps the existing Auto-LR workflow available as a rollback/
         future option

The rest of the SAM-LoRA architecture is unchanged:
- official Meta SAM checkpoint
- ViT-B / ViT-L / ViT-H
- frozen base SAM image encoder
- LoRA on Q + V
- frozen prompt encoder
- trainable mask decoder
- automatic whole-tile building segmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import inspect
import json
import math

import torch
import torch.nn as nn


THIS_FILE = Path(__file__).resolve()
BUILDINGS_ROOT = THIS_FILE.parents[3]
PROJECT_ROOT = BUILDINGS_ROOT.parent

DEFAULT_CONFIG_PATH = (
    BUILDINGS_ROOT
    / "config"
    / "models"
    / "sam_lora.json"
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class SAMLoRAFactoryBundle:
    """
    Resolved SAM-LoRA components used by run_train.py.

    The public fields intentionally match the integration already used
    by the Buildings module.
    """

    config: Dict[str, Any]
    config_path: Path
    backbone: str
    checkpoint_path: Path
    model_factory: Callable[[], nn.Module]
    criterion: nn.Module
    trainer_kwargs: Dict[str, Any]
    batch_size: int

    @property
    def loss_fn(self) -> nn.Module:
        return self.criterion

    @property
    def sam_checkpoint_path(self) -> Path:
        return self.checkpoint_path


# ============================================================
# GENERIC HELPERS
# ============================================================

def _call_with_supported_kwargs(
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """
    Call a function while passing only keyword arguments accepted by
    its current signature.

    This keeps the factory compatible with the SAM-LoRA builder/loss
    modules already present in the project.
    """

    signature = inspect.signature(function)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return function(**kwargs)

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return function(**filtered)


def _normalize_learning_rate_mode(value: Any) -> str:
    mode = str(value).strip().lower()

    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "auto_lr": "auto",
        "lr_finder": "auto",
        "fixed": "fixed",
        "manual": "fixed",
        "constant": "fixed",
    }

    if mode not in aliases:
        raise ValueError(
            "SAM-LoRA learning_rate.mode must be either "
            "'fixed' or 'auto'. "
            f"Received: {value!r}"
        )

    return aliases[mode]


def _require_positive_finite(
    value: Any,
    name: str,
) -> float:
    try:
        resolved = float(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be a positive number. Received: {value!r}"
        ) from exc

    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            f"{name} must be finite and > 0. Received: {resolved}"
        )

    return resolved


def _normalize_backbone_name(value: Any) -> str:
    backbone = str(value).strip().lower().replace("-", "_")

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

    return aliases.get(backbone, backbone)


# ============================================================
# CONFIGURATION
# ============================================================

def load_sam_lora_config(
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    path = Path(
        config_path
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            "SAM-LoRA configuration was not found:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise TypeError(
            "SAM-LoRA configuration must be a JSON object."
        )

    return config


def _validate_training_policy(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate SAM-LoRA training policy.

    IMPORTANT CHANGE:
    The old implementation required Auto-LR only.
    This implementation deliberately accepts:

        learning_rate.mode = "fixed"
        learning_rate.mode = "auto"

    Fixed mode requires:
        learning_rate.value > 0

    Auto mode preserves the existing finder configuration.
    """

    training = config.get("training", {})

    if not isinstance(training, dict):
        raise TypeError(
            "sam_lora.json -> training must be an object."
        )

    optimizer_name = str(
        training.get("optimizer", "adamw")
    ).strip().lower()

    if optimizer_name != "adamw":
        raise ValueError(
            "Current SAM-LoRA training supports optimizer='adamw'. "
            f"Received: {optimizer_name!r}"
        )

    weight_decay = _require_positive_finite(
        training.get("weight_decay", 1e-5),
        "SAM-LoRA weight_decay",
    )

    learning_rate_config = training.get(
        "learning_rate",
        {},
    )

    if not isinstance(learning_rate_config, dict):
        raise TypeError(
            "sam_lora.json -> training.learning_rate must be an object."
        )

    mode = _normalize_learning_rate_mode(
        learning_rate_config.get(
            "mode",
            "auto",
        )
    )

    resolved = {
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "learning_rate_mode": mode,
    }

    if mode == "fixed":
        fixed_lr = _require_positive_finite(
            learning_rate_config.get(
                "value",
                None,
            ),
            "SAM-LoRA fixed learning rate",
        )

        resolved["fixed_learning_rate"] = fixed_lr
        resolved["learning_rate"] = fixed_lr

    else:
        finder = learning_rate_config.get(
            "finder",
            {},
        )

        if not isinstance(finder, dict):
            raise TypeError(
                "sam_lora.json -> "
                "training.learning_rate.finder must be an object "
                "when mode='auto'."
            )

        min_lr = _require_positive_finite(
            finder.get("min_lr", 1e-7),
            "SAM-LoRA Auto-LR min_lr",
        )

        max_lr = _require_positive_finite(
            finder.get("max_lr", 3e-3),
            "SAM-LoRA Auto-LR max_lr",
        )

        if max_lr <= min_lr:
            raise ValueError(
                "SAM-LoRA Auto-LR max_lr must be greater than min_lr."
            )

        num_steps = int(
            finder.get(
                "num_steps",
                80,
            )
        )

        if num_steps < 2:
            raise ValueError(
                "SAM-LoRA Auto-LR num_steps must be >= 2."
            )

        resolved.update(
            {
                "auto_lr_min_lr": min_lr,
                "auto_lr_max_lr": max_lr,
                "auto_lr_num_steps": num_steps,
                "auto_lr_smoothing_beta": float(
                    finder.get(
                        "smoothing_beta",
                        0.98,
                    )
                ),
                "auto_lr_divergence_threshold": float(
                    finder.get(
                        "divergence_threshold",
                        4.0,
                    )
                ),
                "auto_lr_selection_method": str(
                    finder.get(
                        "selection_method",
                        "steepest_descent",
                    )
                ),
            }
        )

    return resolved


# ============================================================
# CHECKPOINT
# ============================================================

def resolve_sam_checkpoint_path(
    config: Dict[str, Any],
    backbone: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> Path:
    supported = config.get(
        "supported_backbones",
        {},
    )

    if not isinstance(supported, dict) or not supported:
        raise ValueError(
            "SAM-LoRA configuration has no supported_backbones."
        )

    resolved_backbone = _normalize_backbone_name(
        backbone
        if backbone is not None
        else config.get(
            "default_backbone",
            "vit_b",
        )
    )

    if resolved_backbone not in supported:
        raise ValueError(
            f"Unsupported SAM-LoRA backbone: {resolved_backbone}. "
            f"Supported: {sorted(supported.keys())}"
        )

    info = supported[resolved_backbone]

    if not isinstance(info, dict):
        raise TypeError(
            f"Backbone configuration for {resolved_backbone} "
            "must be an object."
        )

    if not bool(info.get("enabled", True)):
        raise ValueError(
            f"SAM-LoRA backbone {resolved_backbone} is disabled."
        )

    checkpoint_value = info.get(
        "checkpoint",
        None,
    )

    if not checkpoint_value:
        raise ValueError(
            f"No checkpoint configured for {resolved_backbone}."
        )

    raw = Path(
        str(checkpoint_value)
    )

    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    else:
        # Configuration currently stores:
        # Buildings/checkpoints/sam/...
        candidates.extend(
            [
                PROJECT_ROOT / raw,
                BUILDINGS_ROOT / raw,
                BUILDINGS_ROOT / "checkpoints" / "sam" / raw.name,
            ]
        )

        if config_path is not None:
            candidates.append(
                Path(config_path).resolve().parent / raw
            )

        candidates.append(
            Path.cwd() / raw
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Official SAM checkpoint was not found.\n"
        f"Backbone: {resolved_backbone}\n"
        f"Configured value: {checkpoint_value}\n"
        "Checked:\n- "
        + "\n- ".join(
            str(path)
            for path in candidates
        )
    )


# ============================================================
# MODEL BUILDING
# ============================================================

def _resolve_builder_output(
    build_result: Any,
) -> nn.Module:
    """
    Resolve different builder return styles without changing the
    existing builder.py public contract.
    """

    candidate = build_result

    if isinstance(
        candidate,
        (tuple, list),
    ):
        if not candidate:
            raise RuntimeError(
                "SAM-LoRA builder returned an empty tuple/list."
            )

        module_candidates = [
            item
            for item in candidate
            if isinstance(item, nn.Module)
        ]

        if module_candidates:
            candidate = module_candidates[0]
        else:
            candidate = candidate[0]

    if not isinstance(candidate, nn.Module):
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

            if isinstance(value, nn.Module):
                candidate = value
                break

    if not isinstance(candidate, nn.Module):
        raise TypeError(
            "Could not resolve a torch.nn.Module from "
            "build_sam_lora_model(). "
            f"Received: {type(build_result).__name__}"
        )

    # If builder.py returns the raw SAM object, wrap it in our
    # automatic whole-tile building segmenter.
    if (
        hasattr(candidate, "image_encoder")
        and hasattr(candidate, "prompt_encoder")
        and hasattr(candidate, "mask_decoder")
    ):
        try:
            from .model import SAMLoRABuildingSegmenter

            if not isinstance(
                candidate,
                SAMLoRABuildingSegmenter,
            ):
                candidate = SAMLoRABuildingSegmenter(
                    candidate
                )
        except Exception:
            # If it is already a custom wrapper exposing the same
            # attributes, let the trainer/model validation catch it.
            pass

    return candidate


def build_sam_lora_from_config(
    config: Dict[str, Any],
    backbone: Optional[str] = None,
    device: Optional[str | torch.device] = None,
    config_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> nn.Module:
    from .builder import build_sam_lora_model

    resolved_backbone = _normalize_backbone_name(
        backbone
        if backbone is not None
        else config.get(
            "default_backbone",
            "vit_b",
        )
    )

    checkpoint_path = resolve_sam_checkpoint_path(
        config=config,
        backbone=resolved_backbone,
        config_path=config_path,
    )

    lora = config.get(
        "lora",
        {},
    )

    build_result = _call_with_supported_kwargs(
        build_sam_lora_model,
        checkpoint=checkpoint_path,
        checkpoint_path=checkpoint_path,
        sam_checkpoint_path=checkpoint_path,
        backbone=resolved_backbone,
        rank=int(
            lora.get(
                "rank",
                8,
            )
        ),
        alpha=float(
            lora.get(
                "alpha",
                16.0,
            )
        ),
        dropout=float(
            lora.get(
                "dropout",
                0.0,
            )
        ),
        target_projections=lora.get(
            "target_projections",
            ["q", "v"],
        ),
        transformer_blocks=lora.get(
            "apply_to_transformer_blocks",
            "all",
        ),
        device=device,
        verbose=bool(verbose),
    )

    model = _resolve_builder_output(
        build_result
    )

    if device is not None:
        model = model.to(
            torch.device(device)
        )

    return model


def create_sam_lora_model_factory(
    config: Dict[str, Any],
    backbone: Optional[str] = None,
    device: Optional[str | torch.device] = None,
    config_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> Callable[[], nn.Module]:
    """
    Return a zero-argument factory.

    A new model is built on every call. This is required by Auto-LR
    because the LR-range-test model must be discarded before final
    training. In fixed-LR mode the trainer calls the factory only once.
    """

    def model_factory() -> nn.Module:
        return build_sam_lora_from_config(
            config=config,
            backbone=backbone,
            device=device,
            config_path=config_path,
            verbose=verbose,
        )

    return model_factory


# ============================================================
# LOSS
# ============================================================

def build_sam_lora_loss_from_config(
    config: Dict[str, Any],
) -> nn.Module:
    from .loss import build_sam_lora_loss

    training = config.get(
        "training",
        {},
    )

    loss_config = training.get(
        "loss",
        {},
    )

    return _call_with_supported_kwargs(
        build_sam_lora_loss,
        loss_type=loss_config.get(
            "type",
            "bce_dice",
        ),
        bce_weight=float(
            loss_config.get(
                "bce_weight",
                0.5,
            )
        ),
        dice_weight=float(
            loss_config.get(
                "dice_weight",
                0.5,
            )
        ),
        dice_smooth=float(
            loss_config.get(
                "dice_smooth",
                1.0,
            )
        ),
        smooth=float(
            loss_config.get(
                "dice_smooth",
                1.0,
            )
        ),
    )


# ============================================================
# TRAINER KWARGS
# ============================================================

def build_sam_lora_trainer_kwargs(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    policy = _validate_training_policy(
        config
    )

    training = config.get(
        "training",
        {},
    )

    validation = config.get(
        "validation",
        {},
    )

    learning_rate = training.get(
        "learning_rate",
        {},
    )

    finder = (
        learning_rate.get(
            "finder",
            {},
        )
        if isinstance(
            learning_rate,
            dict,
        )
        else {}
    )

    kwargs: Dict[str, Any] = {
        "epochs": int(
            training.get(
                "epochs",
                40,
            )
        ),
        "optimizer_name": str(
            training.get(
                "optimizer",
                "adamw",
            )
        ).strip().lower(),
        "weight_decay": float(
            training.get(
                "weight_decay",
                1e-5,
            )
        ),
        "scheduler_name": str(
            training.get(
                "scheduler",
                "none",
            )
        ).strip().lower(),
        "mixed_precision": bool(
            training.get(
                "mixed_precision",
                True,
            )
        ),
        "validation_threshold": float(
            validation.get(
                "default_threshold",
                0.5,
            )
        ),

        # NEW shared policy.
        "learning_rate_mode":
            policy["learning_rate_mode"],
    }

    if (
        policy["learning_rate_mode"]
        == "fixed"
    ):
        fixed_lr = float(
            policy[
                "fixed_learning_rate"
            ]
        )

        # Both names are included for compatibility. trainer.py accepts
        # either and resolves to the same value.
        kwargs[
            "fixed_learning_rate"
        ] = fixed_lr

        kwargs[
            "learning_rate"
        ] = fixed_lr

    else:
        kwargs.update(
            {
                "auto_lr_min_lr": float(
                    finder.get(
                        "min_lr",
                        1e-7,
                    )
                ),
                "auto_lr_max_lr": float(
                    finder.get(
                        "max_lr",
                        3e-3,
                    )
                ),
                "auto_lr_num_steps": int(
                    finder.get(
                        "num_steps",
                        80,
                    )
                ),
                "auto_lr_smoothing_beta": float(
                    finder.get(
                        "smoothing_beta",
                        0.98,
                    )
                ),
                "auto_lr_divergence_threshold": float(
                    finder.get(
                        "divergence_threshold",
                        4.0,
                    )
                ),
                "auto_lr_selection_method": str(
                    finder.get(
                        "selection_method",
                        "steepest_descent",
                    )
                ),
            }
        )

    return kwargs


# ============================================================
# BATCH SIZE
# ============================================================

def resolve_sam_lora_batch_size(
    config: Dict[str, Any],
    tile_size: Optional[int] = None,
    backbone: Optional[str] = None,
    default: Optional[int] = None,
) -> int:
    """
    Resolve a safe configured batch size.

    tile_size is accepted because run_train.py already passes it.
    Current SAM-LoRA batch recommendations are backbone-based.
    """

    del tile_size

    training = config.get(
        "training",
        {},
    )

    resolved_backbone = _normalize_backbone_name(
        backbone
        if backbone is not None
        else config.get(
            "default_backbone",
            "vit_b",
        )
    )

    recommendations = training.get(
        "recommended_batch_size_by_backbone",
        {},
    )

    value = None

    if isinstance(
        recommendations,
        dict,
    ):
        value = recommendations.get(
            resolved_backbone,
            None,
        )

    if value is None:
        value = training.get(
            "default_batch_size",
            default if default is not None else 1,
        )

    batch_size = int(value)

    if batch_size <= 0:
        raise ValueError(
            "SAM-LoRA batch size must be > 0. "
            f"Received: {batch_size}"
        )

    return batch_size


# ============================================================
# BUNDLE
# ============================================================

def build_sam_lora_factory_bundle(
    config_path: Optional[str | Path] = None,
    backbone: Optional[str] = None,
    device: Optional[str | torch.device] = None,
    verbose_model_build: bool = False,
) -> SAMLoRAFactoryBundle:
    resolved_config_path = Path(
        config_path
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    config = load_sam_lora_config(
        resolved_config_path
    )

    _validate_training_policy(
        config
    )

    resolved_backbone = _normalize_backbone_name(
        backbone
        if backbone is not None
        else config.get(
            "default_backbone",
            "vit_b",
        )
    )

    checkpoint_path = resolve_sam_checkpoint_path(
        config=config,
        backbone=resolved_backbone,
        config_path=resolved_config_path,
    )

    criterion = build_sam_lora_loss_from_config(
        config
    )

    trainer_kwargs = build_sam_lora_trainer_kwargs(
        config
    )

    model_factory = create_sam_lora_model_factory(
        config=config,
        backbone=resolved_backbone,
        device=device,
        config_path=resolved_config_path,
        verbose=verbose_model_build,
    )

    batch_size = resolve_sam_lora_batch_size(
        config=config,
        backbone=resolved_backbone,
    )

    return SAMLoRAFactoryBundle(
        config=config,
        config_path=resolved_config_path.resolve(),
        backbone=resolved_backbone,
        checkpoint_path=checkpoint_path,
        model_factory=model_factory,
        criterion=criterion,
        trainer_kwargs=trainer_kwargs,
        batch_size=batch_size,
    )


def print_sam_lora_factory_summary(
    bundle: SAMLoRAFactoryBundle,
) -> None:
    training = bundle.config.get(
        "training",
        {},
    )

    learning_rate = training.get(
        "learning_rate",
        {},
    )

    mode = _normalize_learning_rate_mode(
        learning_rate.get(
            "mode",
            "auto",
        )
    )

    print()
    print("=" * 72)
    print("SAM-LoRA FACTORY")
    print("=" * 72)
    print("Backbone:", bundle.backbone)
    print(
        "Official SAM Checkpoint:",
        bundle.checkpoint_path,
    )
    print("Batch Size:", bundle.batch_size)
    print("Learning Rate Mode:", mode.upper())

    if mode == "fixed":
        print(
            "Fixed Learning Rate:",
            f"{float(learning_rate['value']):.3e}",
        )
        print("Auto-LR Range Test:", "SKIPPED")
    else:
        print("Auto-LR Range Test:", "ENABLED")

    print("=" * 72)
    print()

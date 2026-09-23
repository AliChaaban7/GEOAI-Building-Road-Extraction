"""
Model registry for the Buildings module.

The UI reads model JSON files dynamically for detailed backbone/configuration
information. This registry provides canonical model keys, display names,
family routing, fallback backbones, and high-level feature capability flags.

Focused Optuna is enabled for SAM-LoRA. Focused SAM-LoRA Optuna searches
training hyperparameters only; LoRA architecture parameters are reserved for
Master Optuna.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    default_backbone: str
    supported_backbones: Tuple[str, ...]
    supports_augmentation: bool = True
    supports_optuna: bool = True
    supports_threshold: bool = True
    supports_postprocessing: bool = True


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "unet": ModelSpec(
        key="unet",
        display_name="U-Net",
        family="semantic",
        default_backbone="vanilla_unet",
        supported_backbones=("vanilla_unet",),
    ),
    "deeplabv3": ModelSpec(
        key="deeplabv3",
        display_name="DeepLabV3",
        family="semantic",
        default_backbone="resnet50",
        supported_backbones=(
            "resnet50",
            "resnet101",
            "mobilenet_v3_large",
        ),
    ),
    "maskrcnn": ModelSpec(
        key="maskrcnn",
        display_name="Mask R-CNN",
        family="instance",
        default_backbone="resnet50_fpn",
        supported_backbones=("resnet50_fpn",),
    ),
    "sam_lora": ModelSpec(
        key="sam_lora",
        display_name="SAM-LoRA",
        family="semantic",
        default_backbone="vit_b",
        supported_backbones=(
            "vit_b",
            "vit_l",
            "vit_h",
        ),
        supports_augmentation=True,
        supports_optuna=True,
        supports_threshold=True,
        supports_postprocessing=True,
    ),
}


ALIASES = {
    "u-net": "unet",
    "u_net": "unet",
    "deeplab": "deeplabv3",
    "deeplab_v3": "deeplabv3",
    "deep_lab_v3": "deeplabv3",
    "deeplabv3+": "deeplabv3",
    "deeplabv3plus": "deeplabv3",
    "mask_rcnn": "maskrcnn",
    "mask-r-cnn": "maskrcnn",
    "mask r-cnn": "maskrcnn",
    "rcnn": "maskrcnn",
    "sam-lora": "sam_lora",
    "sam lora": "sam_lora",
    "samlora": "sam_lora",
    "sam_lora": "sam_lora",
}


def normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    key = ALIASES.get(key, key)

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {', '.join(MODEL_REGISTRY)}"
        )

    return key


def get_model_spec(model_type: str) -> ModelSpec:
    return MODEL_REGISTRY[
        normalize_model_type(model_type)
    ]


def is_semantic(model_type: str) -> bool:
    return get_model_spec(model_type).family == "semantic"


def is_instance(model_type: str) -> bool:
    return get_model_spec(model_type).family == "instance"


def ui_capabilities() -> dict:
    return {
        key: {
            "display_name": spec.display_name,
            "family": spec.family,
            "default_backbone": spec.default_backbone,
            "supported_backbones": list(spec.supported_backbones),
            "optional_options": {
                "augmentation": spec.supports_augmentation,
                "optuna": spec.supports_optuna,
                "threshold": spec.supports_threshold,
                "postprocessing": spec.supports_postprocessing,
            },
        }
        for key, spec in MODEL_REGISTRY.items()
    }

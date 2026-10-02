"""
Road SAM-LoRA package.

This implementation is independent from Buildings.

Architecture
------------
Official Meta SAM ViT-B
        +
Q/V LoRA adapters
        +
trainable mask decoder
        ↓
binary Road-surface logits

Isolation
---------
No automatic Buildings checkpoint discovery is permitted.
"""

from .lora_qkv import (
    LoRAQKV,
    count_lora_parameters,
    inject_lora_into_sam_image_encoder,
    iter_lora_parameters,
)

from .sam_lora import (
    RoadSAMLoRA,
    build_sam_lora,
    resolve_sam_lora_settings,
    sam_lora_parameter_summary,
)

from .factory import (
    RoadSAMLoRABuildInfo,
    build_sam_lora_from_context,
    build_sam_lora_model,
    resolve_official_sam_checkpoint,
)


# Generic-factory compatibility aliases.
build_model = build_sam_lora_model
create_sam_lora = build_sam_lora_model


__all__ = [
    "LoRAQKV",
    "inject_lora_into_sam_image_encoder",
    "iter_lora_parameters",
    "count_lora_parameters",

    "RoadSAMLoRA",
    "resolve_sam_lora_settings",
    "build_sam_lora",
    "sam_lora_parameter_summary",

    "RoadSAMLoRABuildInfo",
    "resolve_official_sam_checkpoint",
    "build_sam_lora_from_context",
    "build_sam_lora_model",

    "build_model",
    "create_sam_lora",
]
"""
Buildings/src/models/sam_lora/builder.py

Builder for the thesis SAM-LoRA building-segmentation model.

Architecture
------------
Official pretrained Segment Anything Model (SAM)

    Image Encoder
        - pretrained SAM weights frozen
        - LoRA injected into Q and V projections
        - only LoRA adapter weights trainable

    Prompt Encoder
        - frozen

    Mask Decoder
        - trainable

The builder intentionally does not modify Meta's original SAM package.

Supported SAM backbones
-----------------------
    vit_b
    vit_l
    vit_h

The default backbone for the current thesis implementation is:

    vit_b

because this is the architecture selected for the first SAM-LoRA
experiments and for comparison with the ArcGIS Pro SAM-LoRA workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .lora_qkv import (
    count_lora_parameters,
    inject_lora_into_sam_image_encoder,
    iter_lora_parameters,
)


# ---------------------------------------------------------------------
# Supported official SAM backbones.
# ---------------------------------------------------------------------

SUPPORTED_SAM_BACKBONES = (
    "vit_b",
    "vit_l",
    "vit_h",
)


@dataclass(frozen=True)
class SAMLoRABuildInfo:
    """
    Metadata describing one constructed SAM-LoRA model.

    This is useful for:

        training logs
        training_summary.json
        experiment reports
        reproducibility
        debugging
    """

    backbone: str
    checkpoint: str

    lora_rank: int
    lora_alpha: float
    lora_dropout: float

    target_projections: List[str]
    injected_transformer_blocks: List[int]

    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int

    lora_parameters: int
    mask_decoder_parameters: int
    trainable_mask_decoder_parameters: int
    trainable_prompt_encoder_parameters: int
    trainable_image_encoder_parameters: int

    def to_dict(self) -> Dict[str, object]:
        """
        Convert build information into a JSON-serializable dictionary.
        """

        return asdict(self)


def _count_parameters(
    module: nn.Module,
) -> int:
    """
    Count all parameters inside a module.
    """

    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def _count_trainable_parameters(
    module: nn.Module,
) -> int:
    """
    Count parameters whose requires_grad flag is True.
    """

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _freeze_module(
    module: nn.Module,
) -> None:
    """
    Freeze every parameter in a PyTorch module.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_module(
    module: nn.Module,
) -> None:
    """
    Enable gradient computation for every parameter in a module.
    """

    for parameter in module.parameters():
        parameter.requires_grad = True


def _normalize_backbone(
    backbone: str,
) -> str:
    """
    Validate and normalize the requested SAM backbone.
    """

    normalized = str(backbone).strip().lower()

    if normalized not in SUPPORTED_SAM_BACKBONES:
        raise ValueError(
            f"Unsupported SAM backbone '{backbone}'. "
            f"Supported backbones are: "
            f"{list(SUPPORTED_SAM_BACKBONES)}."
        )

    return normalized


def _resolve_checkpoint(
    checkpoint: Union[str, Path],
) -> Path:
    """
    Validate the official SAM checkpoint path.

    The SAM-LoRA thesis implementation intentionally requires a
    pretrained checkpoint. Accidentally creating SAM with random
    initialization would invalidate the transfer-learning experiment.
    """

    checkpoint_path = Path(checkpoint).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "SAM checkpoint was not found:\n"
            f"    {checkpoint_path}\n\n"
            "Provide the path to an official pretrained SAM checkpoint."
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "SAM checkpoint path is not a file:\n"
            f"    {checkpoint_path}"
        )

    return checkpoint_path.resolve()


def _load_official_sam(
    backbone: str,
    checkpoint: Path,
) -> nn.Module:
    """
    Load the official Meta Segment Anything implementation.
    """

    try:
        from segment_anything import sam_model_registry

    except ImportError as exc:
        raise ImportError(
            "The 'segment_anything' package could not be imported.\n"
            "SAM-LoRA requires Meta's official Segment Anything "
            "implementation."
        ) from exc

    if backbone not in sam_model_registry:
        available = sorted(
            str(key)
            for key in sam_model_registry.keys()
        )

        raise KeyError(
            f"SAM backbone '{backbone}' is not available in the "
            "'segment_anything' installation.\n"
            f"Available model registry keys: {available}"
        )

    sam = sam_model_registry[backbone](
        checkpoint=str(checkpoint)
    )

    if not isinstance(sam, nn.Module):
        raise TypeError(
            "sam_model_registry returned an unexpected object. "
            f"Expected nn.Module, received {type(sam).__name__}."
        )

    return sam


def _validate_sam_structure(
    sam: nn.Module,
) -> None:
    """
    Validate the SAM components required by the thesis architecture.
    """

    required_components = (
        "image_encoder",
        "prompt_encoder",
        "mask_decoder",
    )

    for component_name in required_components:
        if not hasattr(sam, component_name):
            raise AttributeError(
                "Loaded SAM model is incompatible with the current "
                "SAM-LoRA implementation.\n"
                f"Missing required component: '{component_name}'."
            )


def _validate_image_encoder_trainability(
    image_encoder: nn.Module,
) -> None:
    """
    Verify that the only trainable image-encoder parameters are LoRA
    adapter parameters.

    This protects the experiment from accidentally fine-tuning the
    original pretrained SAM image encoder.
    """

    lora_parameter_ids = {
        id(parameter)
        for parameter in iter_lora_parameters(image_encoder)
    }

    unexpected_trainable: List[str] = []

    for name, parameter in image_encoder.named_parameters():

        if not parameter.requires_grad:
            continue

        if id(parameter) not in lora_parameter_ids:
            unexpected_trainable.append(name)

    if unexpected_trainable:
        formatted = "\n".join(
            f"    - {name}"
            for name in unexpected_trainable
        )

        raise RuntimeError(
            "Unexpected trainable parameters were found inside the "
            "SAM image encoder.\n\n"
            "Only LoRA adapters are allowed to be trainable.\n\n"
            f"Unexpected parameters:\n{formatted}"
        )


def _validate_prompt_encoder_frozen(
    prompt_encoder: nn.Module,
) -> None:
    """
    Ensure that the complete SAM prompt encoder is frozen.
    """

    trainable = [
        name
        for name, parameter
        in prompt_encoder.named_parameters()
        if parameter.requires_grad
    ]

    if trainable:
        formatted = "\n".join(
            f"    - {name}"
            for name in trainable
        )

        raise RuntimeError(
            "SAM prompt encoder must remain frozen, but trainable "
            "parameters were detected:\n"
            f"{formatted}"
        )


def _validate_mask_decoder_trainable(
    mask_decoder: nn.Module,
) -> None:
    """
    Ensure that the SAM mask decoder contains trainable parameters.
    """

    trainable_count = _count_trainable_parameters(
        mask_decoder
    )

    if trainable_count <= 0:
        raise RuntimeError(
            "SAM mask decoder contains zero trainable parameters.\n"
            "The current thesis SAM-LoRA architecture requires the "
            "mask decoder to be trainable."
        )


def build_sam_lora_model(
    checkpoint: Union[str, Path],
    backbone: str = "vit_b",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_projections: Sequence[str] = ("q", "v"),
    transformer_blocks: Union[
        str,
        Sequence[int],
    ] = "all",
    device: Optional[
        Union[str, torch.device]
    ] = None,
    verbose: bool = True,
) -> tuple[nn.Module, SAMLoRABuildInfo]:
    """
    Build the thesis SAM-LoRA model.

    Training policy
    ---------------

    Image Encoder:
        Original SAM parameters:
            FROZEN

        LoRA Q/V adapters:
            TRAINABLE

    Prompt Encoder:
        FROZEN

    Mask Decoder:
        TRAINABLE

    Parameters
    ----------
    checkpoint:
        Path to official pretrained SAM checkpoint.

    backbone:
        One of:
            vit_b
            vit_l
            vit_h

    rank:
        LoRA rank.

    alpha:
        LoRA scaling alpha.

    dropout:
        Dropout applied only inside the LoRA branch.

    target_projections:
        LoRA targets.

        Current implementation supports:
            ("q", "v")

    transformer_blocks:
        Either:

            "all"

        or explicit image-encoder Transformer block indices.

    device:
        Optional device.

        Examples:

            "cuda"
            "cuda:0"
            "cpu"

        If None, the caller/training engine may move the model later.

    verbose:
        Print the model-build summary.

    Returns
    -------
    sam:
        Fully configured SAM-LoRA model.

    build_info:
        Parameter counts and architecture metadata.
    """

    # --------------------------------------------------------------
    # Validate configuration.
    # --------------------------------------------------------------

    backbone = _normalize_backbone(
        backbone
    )

    checkpoint_path = _resolve_checkpoint(
        checkpoint
    )

    normalized_targets = [
        str(target).strip().lower()
        for target in target_projections
    ]

    # --------------------------------------------------------------
    # Load official pretrained SAM.
    # --------------------------------------------------------------

    sam = _load_official_sam(
        backbone=backbone,
        checkpoint=checkpoint_path,
    )

    _validate_sam_structure(
        sam
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Freeze ALL existing SAM parameters before LoRA injection.
    #
    # LoRA layers are created after this step, therefore their newly
    # created parameters remain trainable.
    # --------------------------------------------------------------

    _freeze_module(
        sam
    )

    # --------------------------------------------------------------
    # Inject LoRA into the SAM image encoder.
    #
    # Only Q and V are adapted according to the selected configuration.
    # --------------------------------------------------------------

    injected_blocks = inject_lora_into_sam_image_encoder(
        image_encoder=sam.image_encoder,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_projections=normalized_targets,
        transformer_blocks=transformer_blocks,
    )

    # --------------------------------------------------------------
    # Explicitly guarantee that all LoRA parameters are trainable.
    # --------------------------------------------------------------

    for parameter in iter_lora_parameters(
        sam.image_encoder
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------------
    # Prompt encoder stays frozen.
    # --------------------------------------------------------------

    _freeze_module(
        sam.prompt_encoder
    )

    # --------------------------------------------------------------
    # Mask decoder is trainable.
    # --------------------------------------------------------------

    _unfreeze_module(
        sam.mask_decoder
    )

    # --------------------------------------------------------------
    # Architecture safety checks.
    # --------------------------------------------------------------

    _validate_image_encoder_trainability(
        sam.image_encoder
    )

    _validate_prompt_encoder_frozen(
        sam.prompt_encoder
    )

    _validate_mask_decoder_trainable(
        sam.mask_decoder
    )

    # --------------------------------------------------------------
    # Move model to selected device if requested.
    # --------------------------------------------------------------

    if device is not None:
        sam = sam.to(
            torch.device(device)
        )

    # --------------------------------------------------------------
    # Parameter statistics.
    # --------------------------------------------------------------

    total_parameters = _count_parameters(
        sam
    )

    trainable_parameters = _count_trainable_parameters(
        sam
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    lora_parameters = count_lora_parameters(
        sam.image_encoder
    )

    mask_decoder_parameters = _count_parameters(
        sam.mask_decoder
    )

    trainable_mask_decoder_parameters = (
        _count_trainable_parameters(
            sam.mask_decoder
        )
    )

    trainable_prompt_encoder_parameters = (
        _count_trainable_parameters(
            sam.prompt_encoder
        )
    )

    trainable_image_encoder_parameters = (
        _count_trainable_parameters(
            sam.image_encoder
        )
    )

    # --------------------------------------------------------------
    # Additional safety check.
    # --------------------------------------------------------------

    if lora_parameters <= 0:
        raise RuntimeError(
            "SAM-LoRA was built with zero LoRA parameters.\n"
            "LoRA injection failed or no Transformer blocks were "
            "selected."
        )

    if trainable_image_encoder_parameters != lora_parameters:
        raise RuntimeError(
            "SAM image-encoder trainable parameter count does not "
            "match LoRA parameter count.\n"
            f"Trainable image encoder parameters: "
            f"{trainable_image_encoder_parameters:,}\n"
            f"LoRA parameters: {lora_parameters:,}"
        )

    # --------------------------------------------------------------
    # Build reproducibility metadata.
    # --------------------------------------------------------------

    build_info = SAMLoRABuildInfo(
        backbone=backbone,
        checkpoint=str(checkpoint_path),
        lora_rank=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_projections=normalized_targets,
        injected_transformer_blocks=list(
            injected_blocks
        ),
        total_parameters=int(
            total_parameters
        ),
        trainable_parameters=int(
            trainable_parameters
        ),
        frozen_parameters=int(
            frozen_parameters
        ),
        lora_parameters=int(
            lora_parameters
        ),
        mask_decoder_parameters=int(
            mask_decoder_parameters
        ),
        trainable_mask_decoder_parameters=int(
            trainable_mask_decoder_parameters
        ),
        trainable_prompt_encoder_parameters=int(
            trainable_prompt_encoder_parameters
        ),
        trainable_image_encoder_parameters=int(
            trainable_image_encoder_parameters
        ),
    )

    if verbose:
        print_sam_lora_build_summary(
            build_info
        )

    return sam, build_info


def print_sam_lora_build_summary(
    info: SAMLoRABuildInfo,
) -> None:
    """
    Print a compact SAM-LoRA architecture summary.
    """

    if info.total_parameters > 0:
        trainable_percentage = (
            info.trainable_parameters
            / info.total_parameters
            * 100.0
        )
    else:
        trainable_percentage = 0.0

    blocks_text = ", ".join(
        str(index)
        for index
        in info.injected_transformer_blocks
    )

    projections_text = ", ".join(
        projection.upper()
        for projection
        in info.target_projections
    )

    print()
    print("=" * 72)
    print("SAM-LoRA Model")
    print("=" * 72)

    print(
        f"Backbone                    : "
        f"{info.backbone}"
    )

    print(
        f"Checkpoint                  : "
        f"{info.checkpoint}"
    )

    print(
        f"LoRA rank                   : "
        f"{info.lora_rank}"
    )

    print(
        f"LoRA alpha                  : "
        f"{info.lora_alpha}"
    )

    print(
        f"LoRA dropout                : "
        f"{info.lora_dropout}"
    )

    print(
        f"LoRA projections            : "
        f"{projections_text}"
    )

    print(
        f"LoRA Transformer blocks     : "
        f"{blocks_text}"
    )

    print("-" * 72)

    print(
        f"Total parameters            : "
        f"{info.total_parameters:,}"
    )

    print(
        f"Frozen parameters           : "
        f"{info.frozen_parameters:,}"
    )

    print(
        f"Trainable parameters        : "
        f"{info.trainable_parameters:,}"
    )

    print(
        f"Trainable percentage        : "
        f"{trainable_percentage:.4f}%"
    )

    print("-" * 72)

    print(
        f"LoRA parameters             : "
        f"{info.lora_parameters:,}"
    )

    print(
        f"Trainable image encoder     : "
        f"{info.trainable_image_encoder_parameters:,}"
    )

    print(
        f"Trainable prompt encoder    : "
        f"{info.trainable_prompt_encoder_parameters:,}"
    )

    print(
        f"Trainable mask decoder      : "
        f"{info.trainable_mask_decoder_parameters:,}"
    )

    print("=" * 72)
    print()
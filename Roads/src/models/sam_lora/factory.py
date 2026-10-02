"""
Road SAM-LoRA factory and official-checkpoint resolver.

Approach 2 - Road Extraction

Important isolation rule
------------------------
The Road SAM-LoRA implementation must NEVER automatically discover or
reuse a checkpoint from:

    Buildings/

The official pretrained Meta SAM checkpoint may physically be copied or
explicitly referenced anywhere by the user, but the Road pipeline only
uses an explicitly configured path.

Resolution order
----------------
1. Explicit checkpoint argument.
2. Roads model_config.
3. Roads paths_config.

No Buildings fallback exists.

This module builds:

    official Meta SAM ViT-B
            +
    Road-specific Q/V LoRA structure
            +
    trainable SAM mask decoder
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .sam_lora import (
    RoadSAMLoRA,
    build_sam_lora,
    sam_lora_parameter_summary,
)


# ---------------------------------------------------------------------
# Roads root
# ---------------------------------------------------------------------

ROAD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


# ---------------------------------------------------------------------
# Build information
# ---------------------------------------------------------------------

@dataclass
class RoadSAMLoRABuildInfo:
    """
    Provenance for one reconstructed Road SAM-LoRA model.
    """

    backbone: str

    official_sam_checkpoint: str

    checkpoint_source: str

    lora_rank: int

    lora_alpha: float

    lora_dropout: float

    lora_enable_q: bool

    lora_enable_v: bool

    lora_blocks: list

    total_parameters: int

    trainable_parameters: int

    frozen_parameters: int

    trainable_percentage: float

    lora_parameters: int

    mask_decoder_trainable_parameters: int

    road_module_only: bool = True

    buildings_checkpoint_auto_discovery: bool = False

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _as_dict(
    value,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


def _clean_path_value(
    value,
):

    if value is None:

        return None

    text = str(
        value
    ).strip()

    if not text:

        return None

    return text


def _resolve_path(
    value: str | Path,
) -> Path:
    """
    Resolve configured checkpoint path.

    Relative paths are interpreted relative to Roads/.
    """

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            ROAD_ROOT
            / path
        )

    return path.resolve()


# ---------------------------------------------------------------------
# Search one configuration dictionary
# ---------------------------------------------------------------------

def _checkpoint_from_model_config(
    model_config: Dict[str, Any],
) -> Tuple[
    Optional[str],
    Optional[str],
]:
    """
    Search only explicitly supported SAM-pretrained keys.

    We intentionally do NOT accept generic "checkpoint" because that
    could refer to a trained Road model rather than official SAM.
    """

    model_config = _as_dict(
        model_config
    )

    direct_keys = (
        "official_sam_checkpoint",
        "official_sam_checkpoint_path",
        "sam_pretrained_checkpoint",
        "sam_pretrained_checkpoint_path",
        "pretrained_sam_checkpoint",
        "pretrained_sam_checkpoint_path",
    )

    for key in direct_keys:

        value = _clean_path_value(
            model_config.get(
                key
            )
        )

        if value is not None:

            return (
                value,
                f"model_config.{key}",
            )

    # --------------------------------------------------------------
    # Nested "pretrained" block
    # --------------------------------------------------------------

    pretrained = _as_dict(
        model_config.get(
            "pretrained"
        )
    )

    for key in (
        "sam_checkpoint",
        "sam_checkpoint_path",
        "checkpoint",
        "checkpoint_path",
        "path",
    ):

        value = _clean_path_value(
            pretrained.get(
                key
            )
        )

        if value is not None:

            return (
                value,
                f"model_config.pretrained.{key}",
            )

    # --------------------------------------------------------------
    # Nested "sam" block
    # --------------------------------------------------------------

    sam = _as_dict(
        model_config.get(
            "sam"
        )
    )

    for key in (
        "official_checkpoint",
        "official_checkpoint_path",
        "pretrained_checkpoint",
        "pretrained_checkpoint_path",
    ):

        value = _clean_path_value(
            sam.get(
                key
            )
        )

        if value is not None:

            return (
                value,
                f"model_config.sam.{key}",
            )

    return (
        None,
        None,
    )


def _checkpoint_from_paths_config(
    paths_config: Dict[str, Any],
) -> Tuple[
    Optional[str],
    Optional[str],
]:
    """
    Resolve explicit Road SAM checkpoint settings from paths.json.
    """

    paths_config = _as_dict(
        paths_config
    )

    direct_keys = (
        "sam_pretrained_checkpoint",
        "sam_pretrained_checkpoint_path",
        "official_sam_checkpoint",
        "official_sam_checkpoint_path",
        "pretrained_sam_checkpoint",
        "pretrained_sam_checkpoint_path",
    )

    for key in direct_keys:

        value = _clean_path_value(
            paths_config.get(
                key
            )
        )

        if value is not None:

            return (
                value,
                f"paths_config.{key}",
            )

    # --------------------------------------------------------------
    # Nested SAM blocks supported for forward compatibility.
    # --------------------------------------------------------------

    for block_name in (
        "sam",
        "sam_lora",
        "pretrained",
        "checkpoints",
    ):

        block = _as_dict(
            paths_config.get(
                block_name
            )
        )

        for key in (
            "official_sam_checkpoint",
            "official_sam_checkpoint_path",
            "sam_pretrained_checkpoint",
            "sam_pretrained_checkpoint_path",
            "pretrained_sam_checkpoint",
            "pretrained_sam_checkpoint_path",
        ):

            value = _clean_path_value(
                block.get(
                    key
                )
            )

            if value is not None:

                return (
                    value,
                    f"paths_config.{block_name}.{key}",
                )

    return (
        None,
        None,
    )


# ---------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------

def resolve_official_sam_checkpoint(
    context: Optional[
        Dict[str, Any]
    ] = None,
    model_config: Optional[
        Dict[str, Any]
    ] = None,
    checkpoint_path: Optional[
        str | Path
    ] = None,
) -> Tuple[
    Path,
    str,
]:
    """
    Resolve the OFFICIAL pretrained SAM checkpoint.

    No Buildings-path fallback exists.

    Returns
    -------
    checkpoint_path
    source_description
    """

    context = dict(
        context
        or {}
    )

    if model_config is None:

        model_config = context.get(
            "model_config",
            {},
        )

    # --------------------------------------------------------------
    # 1. Explicit caller path
    # --------------------------------------------------------------

    explicit = _clean_path_value(
        checkpoint_path
    )

    if explicit is not None:

        resolved = _resolve_path(
            explicit
        )

        source = (
            "explicit_argument"
        )

    else:

        # ----------------------------------------------------------
        # 2. Road model configuration
        # ----------------------------------------------------------

        (
            configured,
            source,
        ) = _checkpoint_from_model_config(
            model_config
        )

        # ----------------------------------------------------------
        # 3. Road paths configuration
        # ----------------------------------------------------------

        if configured is None:

            (
                configured,
                source,
            ) = _checkpoint_from_paths_config(
                context.get(
                    "paths_config",
                    {},
                )
            )

        if configured is None:

            raise RuntimeError(
                "\nOfficial SAM ViT-B checkpoint is not configured "
                "for the Roads module.\n\n"
                "Configure one of these Road-specific fields, for example:\n\n"
                'Roads/config/paths.json:\n'
                '    "sam_pretrained_checkpoint": '
                '"D:/.../sam_vit_b_01ec64.pth"\n\n'
                "No automatic Buildings checkpoint fallback is used."
            )

        resolved = _resolve_path(
            configured
        )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if not resolved.exists():

        raise FileNotFoundError(
            "Configured official SAM checkpoint does not exist:\n"
            f"{resolved}\n\n"
            f"Resolved from: {source}"
        )

    if not resolved.is_file():

        raise FileNotFoundError(
            "Configured official SAM checkpoint is not a file:\n"
            f"{resolved}"
        )

    if resolved.suffix.lower() not in {
        ".pth",
        ".pt",
    }:

        raise ValueError(
            "Official SAM checkpoint should be a .pth or .pt file.\n"
            f"Received: {resolved}"
        )

    return (
        resolved,
        str(
            source
        ),
    )


# ---------------------------------------------------------------------
# Build from Road context
# ---------------------------------------------------------------------

def build_sam_lora_from_context(
    context: Dict[str, Any],
    checkpoint_path: Optional[
        str | Path
    ] = None,
    return_info: bool = False,
):
    """
    Build Road SAM-LoRA using the current experiment context.

    Master Optuna-safe
    ------------------
    The model is constructed from context["model_config"], so a winning
    LoRA rank/alpha inserted by Master Optuna is honored automatically.
    """

    if not isinstance(
        context,
        dict,
    ):

        raise TypeError(
            "SAM-LoRA context must be a dictionary."
        )

    model_config = deepcopy(
        context.get(
            "model_config",
            {}
        )
    )

    if not model_config:

        raise ValueError(
            "SAM-LoRA model_config is missing from experiment context."
        )

    (
        official_checkpoint,
        checkpoint_source,
    ) = resolve_official_sam_checkpoint(
        context=context,

        model_config=model_config,

        checkpoint_path=checkpoint_path,
    )

    model = build_sam_lora(
        model_config=model_config,

        sam_checkpoint_path=(
            official_checkpoint
        ),
    )

    summary = sam_lora_parameter_summary(
        model
    )

    settings = model.settings

    build_info = RoadSAMLoRABuildInfo(
        backbone=str(
            settings[
                "backbone"
            ]
        ),

        official_sam_checkpoint=str(
            official_checkpoint
        ),

        checkpoint_source=str(
            checkpoint_source
        ),

        lora_rank=int(
            settings[
                "rank"
            ]
        ),

        lora_alpha=float(
            settings[
                "alpha"
            ]
        ),

        lora_dropout=float(
            settings[
                "dropout"
            ]
        ),

        lora_enable_q=bool(
            settings[
                "enable_q"
            ]
        ),

        lora_enable_v=bool(
            settings[
                "enable_v"
            ]
        ),

        lora_blocks=list(
            model.selected_lora_blocks
        ),

        total_parameters=int(
            summary[
                "total_parameters"
            ]
        ),

        trainable_parameters=int(
            summary[
                "trainable_parameters"
            ]
        ),

        frozen_parameters=int(
            summary[
                "frozen_parameters"
            ]
        ),

        trainable_percentage=float(
            summary[
                "trainable_percentage"
            ]
        ),

        lora_parameters=int(
            summary[
                "lora_parameters"
            ]
        ),

        mask_decoder_trainable_parameters=int(
            summary[
                "mask_decoder_trainable_parameters"
            ]
        ),
    )

    # Useful for training-summary/checkpoint provenance.
    model.road_sam_lora_build_info = (
        build_info.to_dict()
    )

    if return_info:

        return (
            model,
            build_info,
        )

    return model


# ---------------------------------------------------------------------
# Flexible compatibility builder
# ---------------------------------------------------------------------

def build_sam_lora_model(
    context: Optional[
        Dict[str, Any]
    ] = None,
    model_config: Optional[
        Dict[str, Any]
    ] = None,
    paths_config: Optional[
        Dict[str, Any]
    ] = None,
    checkpoint_path: Optional[
        str | Path
    ] = None,
):
    """
    Compatibility entry point for generic model factories.
    """

    if context is None:

        context = {}

    context = deepcopy(
        context
    )

    if model_config is not None:

        context[
            "model_config"
        ] = deepcopy(
            model_config
        )

    if paths_config is not None:

        context[
            "paths_config"
        ] = deepcopy(
            paths_config
        )

    return build_sam_lora_from_context(
        context=context,

        checkpoint_path=checkpoint_path,

        return_info=False,
    )


# Compatibility aliases.
build_model = build_sam_lora_model
create_sam_lora = build_sam_lora_model
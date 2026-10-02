"""
Roads model registry.

This module is the authoritative definition of which model families and
architectures are actually available in the Road Extraction workflow.

Important design rule
---------------------
Model JSON files may contain parameters and historical configuration,
but they must NOT be allowed to invent architectures that the Python
implementation does not support.

Current explicit Roads architectures:

    U-Net
        -> vanilla_unet

    DeepLabV3+
        -> resnet50

    ConnectNet
        -> ArcGIS Learn built-in / fixed implementation

    MultiTaskRoadExtractor
        -> ArcGIS Learn built-in / fixed implementation

    SAM-LoRA
        -> vit_b

Dataset policy
--------------
Road extraction is treated as semantic road-surface segmentation.
The current controlled mixed-source Roads dataset uses CT-style
image/mask pairs.

Scientific policy
-----------------
Training and validation are development data.

External aerial / satellite / drone final-test areas must never be used
to select model architecture, threshold, Optuna parameters, Master
Optuna parameters or post-processing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Optional,
    Tuple,
)

import copy


# ============================================================
# MODEL SPECIFICATION
# ============================================================

@dataclass(
    frozen=True
)
class ModelSpec:

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    key: str

    display_name: str

    # semantic / instance
    family: str

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    default_backbone: Optional[str]

    supported_backbones: Tuple[
        str,
        ...,
    ]

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_type: str

    # --------------------------------------------------------
    # Optional capabilities
    # --------------------------------------------------------

    supports_augmentation: bool = True

    supports_optuna: bool = True

    supports_master_optuna: bool = True

    supports_threshold: bool = True

    supports_postprocessing: bool = True


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY: Dict[
    str,
    ModelSpec,
] = {

    # ========================================================
    # U-NET
    # ========================================================

    "unet": ModelSpec(

        key=
            "unet",

        display_name=
            "U-Net",

        family=
            "semantic",

        default_backbone=
            "vanilla_unet",

        supported_backbones=(
            "vanilla_unet",
        ),

        dataset_type=
            "ct",

        supports_augmentation=
            True,

        supports_optuna=
            True,

        supports_master_optuna=
            True,

        supports_threshold=
            True,

        supports_postprocessing=
            True,
    ),


    # ========================================================
    # DEEPLABV3+
    # ========================================================
    #
    # IMPORTANT:
    #
    # The current Roads DeepLab implementation is explicitly
    # wired for ResNet50.
    #
    # DO NOT put:
    #
    #     resnet101
    #     resnet152
    #     mobilenet
    #
    # here unless deeplabv3.py is actually updated and tested
    # for those architectures.
    #
    # This is the key protection against the crash-producing
    # resnet152 experiment.
    # ========================================================

    "deeplabv3": ModelSpec(

        key=
            "deeplabv3",

        display_name=
            "DeepLabV3+",

        family=
            "semantic",

        default_backbone=
            "resnet50",

        supported_backbones=(
            "resnet50",
        ),

        dataset_type=
            "ct",

        supports_augmentation=
            True,

        supports_optuna=
            True,

        supports_master_optuna=
            True,

        supports_threshold=
            True,

        supports_postprocessing=
            True,
    ),


    # ========================================================
    # CONNECTNET
    # ========================================================
    #
    # ArcGIS Learn owns the internal architecture.
    #
    # Therefore we deliberately expose NO fake selectable
    # PyTorch backbone through the generic UI.
    # ========================================================

    "connectnet": ModelSpec(

        key=
            "connectnet",

        display_name=
            "ConnectNet",

        family=
            "semantic",

        default_backbone=
            None,

        supported_backbones=(),

        dataset_type=
            "ct",

        supports_augmentation=
            False,

        supports_optuna=
            False,

        supports_master_optuna=
            False,

        supports_threshold=
            True,

        supports_postprocessing=
            True,
    ),


    # ========================================================
    # MULTI-TASK ROAD EXTRACTOR
    # ========================================================

    "multitask": ModelSpec(

        key=
            "multitask",

        display_name=
            "MultiTask Road Extractor",

        family=
            "semantic",

        default_backbone=
            None,

        supported_backbones=(),

        dataset_type=
            "ct",

        supports_augmentation=
            False,

        supports_optuna=
            False,

        supports_master_optuna=
            False,

        supports_threshold=
            True,

        supports_postprocessing=
            True,
    ),


    # ========================================================
    # SAM-LORA
    # ========================================================

    "sam_lora": ModelSpec(

        key=
            "sam_lora",

        display_name=
            "SAM-LoRA",

        family=
            "semantic",

        default_backbone=
            "vit_b",

        supported_backbones=(
            "vit_b",
        ),

        dataset_type=
            "ct",

        supports_augmentation=
            True,

        # Focused Optuna / Master Optuna remain disabled here
        # until the dedicated SAM-LoRA optimization workflow
        # explicitly supports them.
        supports_optuna=
            False,

        supports_master_optuna=
            False,

        supports_threshold=
            True,

        supports_postprocessing=
            True,
    ),
}


# ============================================================
# ALIASES
# ============================================================

ALIASES = {

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    "u-net":
        "unet",

    "u_net":
        "unet",

    "unet":
        "unet",


    # --------------------------------------------------------
    # DeepLabV3+
    # --------------------------------------------------------

    "deeplab":
        "deeplabv3",

    "deeplab_v3":
        "deeplabv3",

    "deep_lab_v3":
        "deeplabv3",

    "deeplabv3":
        "deeplabv3",

    "deeplabv3+":
        "deeplabv3",

    "deeplabv3plus":
        "deeplabv3",

    "deeplab_v3_plus":
        "deeplabv3",


    # --------------------------------------------------------
    # ConnectNet
    # --------------------------------------------------------

    "connectnet":
        "connectnet",

    "connect_net":
        "connectnet",

    "connect-net":
        "connectnet",


    # --------------------------------------------------------
    # MultiTaskRoadExtractor
    # --------------------------------------------------------

    "multitask":
        "multitask",

    "multi_task":
        "multitask",

    "multi-task":
        "multitask",

    "multitaskroadextractor":
        "multitask",

    "multitask_road_extractor":
        "multitask",

    "multi_task_road_extractor":
        "multitask",


    # --------------------------------------------------------
    # SAM-LoRA
    # --------------------------------------------------------

    "sam_lora":
        "sam_lora",

    "sam-lora":
        "sam_lora",

    "sam lora":
        "sam_lora",

    "samlora":
        "sam_lora",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_model_type(
    model_type,
):
    """
    Normalize historical / friendly model names into the canonical
    registry key.
    """

    key = (
        str(
            model_type
        )
        .strip()
        .lower()
    )

    key = ALIASES.get(
        key,
        key,
    )

    if (
        key
        not in MODEL_REGISTRY
    ):

        raise ValueError(
            "\nUnsupported Roads model_type.\n"
            f"Requested : {model_type}\n"
            "Supported : "
            + ", ".join(
                MODEL_REGISTRY.keys()
            )
        )

    return key


# ============================================================
# MODEL LOOKUP
# ============================================================

def get_model_spec(
    model_type,
):
    """
    Return the authoritative ModelSpec.
    """

    key = normalize_model_type(
        model_type
    )

    return MODEL_REGISTRY[
        key
    ]


# ============================================================
# FAMILY HELPERS
# ============================================================

def is_semantic(
    model_type,
):
    return (
        get_model_spec(
            model_type
        ).family
        == "semantic"
    )


def is_instance(
    model_type,
):
    return (
        get_model_spec(
            model_type
        ).family
        == "instance"
    )


# ============================================================
# DATASET TYPE
# ============================================================

def expected_dataset_type(
    model_type,
):
    """
    Resolve the required training export type from the selected model.
    """

    return (
        get_model_spec(
            model_type
        ).dataset_type
    )


# ============================================================
# BACKBONE VALIDATION
# ============================================================

def validate_backbone(
    model_type,
    backbone,
):
    """
    Validate an explicitly supplied architecture.

    Models with no generic selectable backbone return None.

    This is especially important for preventing configurations such as:

        DeepLabV3+ + resnet152

    when resnet152 is not implemented.
    """

    spec = get_model_spec(
        model_type
    )

    supported = (
        spec.supported_backbones
    )

    # --------------------------------------------------------
    # Fixed / internally managed architecture
    # --------------------------------------------------------

    if not supported:
        return None

    # --------------------------------------------------------
    # Missing architecture -> default
    # --------------------------------------------------------

    if (
        backbone is None
        or str(backbone).strip() == ""
    ):

        return (
            spec.default_backbone
        )

    normalized = (
        str(
            backbone
        )
        .strip()
        .lower()
    )

    if (
        normalized
        not in supported
    ):

        raise ValueError(
            "\nUnsupported backbone for Roads model.\n"
            f"Model     : {spec.display_name}\n"
            f"Requested : {normalized}\n"
            "Supported : "
            + ", ".join(
                supported
            )
        )

    return normalized


# ============================================================
# APPLY MODEL DEFAULTS
# ============================================================

def apply_model_defaults(
    experiment_config,
):
    """
    Normalize and validate an experiment configuration.

    The registry is authoritative.

    Rules
    -----
    1. Normalize model aliases.
    2. Force the correct dataset type.
    3. Resolve / validate architecture.
    4. Never silently convert one architecture into another.
    """

    config = copy.deepcopy(
        experiment_config
        or {}
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_type = (
        normalize_model_type(
            config.get(
                "model_type",
                "",
            )
        )
    )

    spec = get_model_spec(
        model_type
    )

    config[
        "model_type"
    ] = model_type


    # --------------------------------------------------------
    # Dataset type
    # --------------------------------------------------------

    config[
        "dataset_type"
    ] = spec.dataset_type


    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    backbone = validate_backbone(
        model_type=
            model_type,

        backbone=
            config.get(
                "backbone"
            ),
    )

    if backbone is None:

        # ArcGIS/fixed implementation.
        #
        # Removing a stale generic backbone is safer than leaving
        # an invalid value such as "resnet152" attached.
        config.pop(
            "backbone",
            None,
        )

    else:

        config[
            "backbone"
        ] = backbone


    return config


# ============================================================
# CAPABILITIES
# ============================================================

def ui_capabilities():
    """
    Return a JSON-safe description used by Streamlit and orchestration.
    """

    result = {}

    for (
        key,
        spec,
    ) in MODEL_REGISTRY.items():

        result[
            key
        ] = {

            "display_name":
                spec.display_name,

            "family":
                spec.family,

            "dataset_type":
                spec.dataset_type,

            "default_backbone":
                spec.default_backbone,

            "supported_backbones":
                list(
                    spec.supported_backbones
                ),

            "optional_options": {

                "augmentation":
                    spec.supports_augmentation,

                "optuna":
                    spec.supports_optuna,

                "master_optuna":
                    spec.supports_master_optuna,

                "threshold":
                    spec.supports_threshold,

                "postprocessing":
                    spec.supports_postprocessing,
            },
        }

    return result


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [

    "ModelSpec",

    "MODEL_REGISTRY",

    "ALIASES",

    "normalize_model_type",

    "get_model_spec",

    "is_semantic",

    "is_instance",

    "expected_dataset_type",

    "validate_backbone",

    "apply_model_defaults",

    "ui_capabilities",
]
"""
factory.py

Flexible shared model factory for the Buildings module.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- Mask R-CNN
- SAM-LoRA

Goal
----
Changing experiment.json should select the correct model without
changing model-construction code.

Compatibility policy
--------------------
The existing U-Net / DeepLabV3 / Mask R-CNN builder behavior is kept
unchanged.

SAM-LoRA is integrated through its own model-specific factory:

    src.models.sam_lora.factory

This avoids duplicating SAM checkpoint handling, LoRA injection,
freezing rules, and configuration parsing inside the shared factory.

Important
---------
The shared factory can BUILD a SAM-LoRA model, but the SAM-LoRA
training workflow should still use the SAM-LoRA training engine so
Auto-LR can:

    temporary model
        ->
    LR range test
        ->
    discard temporary model
        ->
    fresh model
        ->
    real training
"""

from pathlib import Path
import importlib

import torch


# ============================================================
# FLEXIBLE LEGACY BUILDER CALLING
# ============================================================

def _call_builder_flexibly(
    builder,
    experiment_config,
    model_config,
    backbone_info,
):
    """
    Call a standard model builder with multiple supported signatures.

    This protects the module if older model files use slightly different
    function signatures.

    Used by:
        U-Net
        DeepLabV3
        Mask R-CNN

    SAM-LoRA does not use this helper because it has a dedicated
    configuration/factory implementation.
    """

    errors = []

    call_attempts = [
        lambda: builder(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
        ),

        lambda: builder(
            experiment_config,
            model_config,
            backbone_info,
        ),

        lambda: builder(
            experiment_config=experiment_config,
            model_config=model_config,
        ),

        lambda: builder(
            model_config=model_config,
            experiment_config=experiment_config,
        ),

        lambda: builder(
            model_config,
        ),

        lambda: builder(),
    ]

    for attempt in call_attempts:
        try:
            return attempt()

        except TypeError as exc:
            errors.append(
                str(exc)
            )

    raise TypeError(
        "Could not call model builder with any supported signature.\n"
        "Errors:\n- "
        + "\n- ".join(errors)
    )


def _build_from_module(
    module_name,
    candidate_builder_names,
    experiment_config,
    model_config,
    backbone_info,
):
    """
    Import a standard model module and call the first supported builder.
    """

    module = importlib.import_module(
        module_name
    )

    for builder_name in candidate_builder_names:

        if hasattr(
            module,
            builder_name,
        ):
            builder = getattr(
                module,
                builder_name,
            )

            return _call_builder_flexibly(
                builder=builder,
                experiment_config=experiment_config,
                model_config=model_config,
                backbone_info=backbone_info,
            )

    raise AttributeError(
        f"No supported builder found in {module_name}. "
        f"Tried: {candidate_builder_names}"
    )


# ============================================================
# MODEL TYPE NORMALIZATION
# ============================================================

def _normalize_model_type(
    model_type,
):
    """
    Normalize model aliases used across historical experiment configs.
    """

    model_type = (
        str(
            model_type
        )
        .lower()
        .strip()
    )

    aliases = {
        # U-Net
        "u-net": "unet",
        "u_net": "unet",

        # DeepLab
        "deeplab": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",
        "deeplab_v3": "deeplabv3",
        "deep_lab_v3": "deeplabv3",

        # Mask R-CNN
        "mask_rcnn": "maskrcnn",
        "mask-r-cnn": "maskrcnn",
        "mask r-cnn": "maskrcnn",
        "rcnn": "maskrcnn",

        # SAM-LoRA
        "sam-lora": "sam_lora",
        "samlora": "sam_lora",
        "sam lora": "sam_lora",
        "sam_lora": "sam_lora",
    }

    return aliases.get(
        model_type,
        model_type,
    )


# ============================================================
# SAM-LORA HELPERS
# ============================================================

def _build_sam_lora(
    experiment_config,
    model_config,
    backbone_info,
):
    """
    Build SAM-LoRA through its dedicated factory.

    Why this path is separate
    -------------------------
    SAM-LoRA requires configuration-aware handling of:

        official SAM checkpoint
        selected ViT backbone
        LoRA Q/V injection
        frozen SAM encoder
        frozen prompt encoder
        trainable mask decoder

    Those rules belong in:

        src.models.sam_lora.factory

    and are intentionally not duplicated here.
    """

    # Lazy import:
    # existing models must remain usable even when SAM dependencies
    # are not installed in a particular environment.
    from src.models.sam_lora.factory import (
        build_sam_lora_factory_bundle,
    )

    buildings_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[2]
    )

    config_path = (
        buildings_root
        / "config"
        / "models"
        / "sam_lora.json"
    )

    if not config_path.exists():

        raise FileNotFoundError(
            "SAM-LoRA configuration was not found:\n"
            f"    {config_path}"
        )

    bundle = (
        build_sam_lora_factory_bundle(
            config_path=config_path,
            verbose_model_build=False,
        )
    )

    # --------------------------------------------------------
    # Protect against silent backbone disagreement.
    # --------------------------------------------------------

    experiment_backbone = (
        experiment_config.get(
            "backbone",
            None,
        )
    )

    if experiment_backbone is not None:

        experiment_backbone = (
            str(
                experiment_backbone
            )
            .strip()
            .lower()
        )

        resolved_backbone = (
            str(
                bundle.backbone
            )
            .strip()
            .lower()
        )

        if (
            experiment_backbone
            != resolved_backbone
        ):

            raise ValueError(
                "SAM-LoRA backbone mismatch.\n"
                f"experiment.json : {experiment_backbone}\n"
                f"sam_lora.json   : {resolved_backbone}\n"
                "Keep both values equal."
            )

    # --------------------------------------------------------
    # Optional consistency check against supplied model_config.
    #
    # model_config may be a dict loaded by shared infrastructure.
    # We do not use it to build SAM-LoRA because the dedicated
    # SAM-LoRA factory is the single source of truth.
    # --------------------------------------------------------

    if isinstance(
        model_config,
        dict,
    ):

        config_backbone = (
            model_config.get(
                "default_backbone",
                None,
            )
        )

        if config_backbone is not None:

            if (
                str(
                    config_backbone
                )
                .strip()
                .lower()
                != str(
                    bundle.backbone
                )
                .strip()
                .lower()
            ):

                raise ValueError(
                    "The supplied SAM-LoRA model_config does not "
                    "match the resolved sam_lora.json backbone."
                )

    # --------------------------------------------------------
    # Build exactly one clean model.
    #
    # IMPORTANT:
    # For real SAM-LoRA TRAINING, run_train.py does not call this
    # shared build path before Auto-LR. It passes model_factory to
    # train_sam_lora(), which controls temporary/fresh model creation.
    #
    # This shared builder remains useful for:
    #   - inference infrastructure
    #   - generic model inspection
    #   - future UI integration
    #   - generic shared callers
    # --------------------------------------------------------

    model = (
        bundle.model_factory()
    )

    return model


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    experiment_config,
    model_config,
    backbone_info=None,
    move_to_device=False,
):
    """
    Build the model specified in experiment.json.

    Parameters
    ----------
    experiment_config:
        Experiment configuration dictionary.

    model_config:
        Model configuration dictionary.

    backbone_info:
        Optional shared backbone metadata.

    move_to_device:
        When True, move the built model to CUDA when available,
        otherwise CPU.

    Returns
    -------
    move_to_device=False:
        model

    move_to_device=True:
        (model, device)

    Notes
    -----
    SAM-LoRA training should normally be orchestrated by the
    SAM-LoRA training engine rather than by directly calling this
    shared builder before Auto-LR.
    """

    if backbone_info is None:
        backbone_info = {}

    model_type = _normalize_model_type(
        experiment_config.get(
            "model_type",
            model_config.get(
                "model_type",
                "",
            )
            if isinstance(
                model_config,
                dict,
            )
            else "",
        )
    )

    # ========================================================
    # U-NET
    # ========================================================

    if model_type == "unet":

        model = _build_from_module(
            module_name="src.models.unet",
            candidate_builder_names=[
                "build_unet",
                "build_model",
                "get_model",
            ],
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
        )

    # ========================================================
    # DEEPLABV3 / DEEPLABV3+
    # ========================================================

    elif model_type == "deeplabv3":

        model = _build_from_module(
            module_name="src.models.deeplabv3",
            candidate_builder_names=[
                "build_deeplabv3",
                "build_deeplab",
                "build_model",
                "get_model",
            ],
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
        )

    # ========================================================
    # MASK R-CNN
    # ========================================================

    elif model_type == "maskrcnn":

        model = _build_from_module(
            module_name="src.models.maskrcnn",
            candidate_builder_names=[
                "build_maskrcnn",
                "build_model",
                "get_model",
            ],
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
        )

    # ========================================================
    # SAM-LORA
    # ========================================================

    elif model_type == "sam_lora":

        model = _build_sam_lora(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
        )

    # ========================================================
    # UNSUPPORTED
    # ========================================================

    else:

        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Supported model types: "
            "unet, deeplabv3, maskrcnn, sam_lora"
        )

    # ========================================================
    # LEGACY BUILDER RETURN NORMALIZATION
    # ========================================================
    #
    # Some old builders may return:
    #
    #     (model, device)
    #
    # Keep this behavior exactly compatible.
    # ========================================================

    detected_device = None

    if isinstance(
        model,
        (tuple, list),
    ):

        if len(model) == 0:

            raise RuntimeError(
                "Model builder returned an empty tuple/list."
            )

        if len(model) >= 1:

            detected_device = (
                model[1]
                if len(model) > 1
                else None
            )

            model = model[0]

    if not isinstance(
        model,
        torch.nn.Module,
    ):

        raise TypeError(
            "Model factory did not return a torch.nn.Module. "
            f"Received: {type(model)}"
        )

    # ========================================================
    # OPTIONAL DEVICE MOVE
    # ========================================================

    if move_to_device:

        if detected_device is None:

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            device = detected_device

            if not isinstance(
                device,
                torch.device,
            ):

                device = torch.device(
                    str(
                        device
                    )
                )

        model = model.to(
            device
        )

        return (
            model,
            device,
        )

    return model


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def create_model(
    experiment_config,
    model_config,
    backbone_info=None,
    move_to_device=False,
):
    """
    Compatibility alias for build_model().
    """

    return build_model(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        move_to_device=move_to_device,
    )


def get_model(
    experiment_config,
    model_config,
    backbone_info=None,
    move_to_device=False,
):
    """
    Compatibility alias for build_model().
    """

    return build_model(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        move_to_device=move_to_device,
    )

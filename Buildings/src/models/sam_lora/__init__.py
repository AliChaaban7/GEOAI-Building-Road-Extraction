"""
Buildings/src/models/sam_lora

SAM-LoRA implementation for automatic building extraction.

Architecture
------------
Official pretrained Segment Anything Model (SAM)

    Image Encoder
        - pretrained SAM weights frozen
        - LoRA injected into Query and Value projections
        - LoRA parameters trainable

    Prompt Encoder
        - frozen
        - automatic / no manual prompt

    Mask Decoder
        - trainable

Training
--------
    BCE + Dice loss
    AdamW
    Automatic Learning Rate finder
    deterministic Train / Validation split
    shared Buildings augmentation pipeline

Inference
---------
    official pretrained SAM checkpoint
        +
    trained LoRA + mask-decoder checkpoint
        ↓
    reconstructed trained SAM-LoRA

Shared Buildings functionality such as:

    threshold search
    post-processing
    evaluation
    reporting
    GIS export
    Master Optuna orchestration

remains outside this package and is shared with the other models.
"""

# =====================================================================
# LoRA
# =====================================================================

from .lora_qkv import (
    LoRAQKV,
    count_lora_parameters,
    inject_lora_into_sam_image_encoder,
    iter_lora_modules,
    iter_lora_parameters,
)


# =====================================================================
# SAM BUILDER
# =====================================================================

from .builder import (
    SAMLoRABuildInfo,
    SUPPORTED_SAM_BACKBONES,
    build_sam_lora_model,
    print_sam_lora_build_summary,
)


# =====================================================================
# MODEL
# =====================================================================

from .model import (
    SAMLoRABuildingSegmenter,
)


# =====================================================================
# LOSS
# =====================================================================

from .loss import (
    SAMLoRALoss,
    SAMLoRALossResult,
    SoftDiceLoss,
    build_sam_lora_loss,
)


# =====================================================================
# AUTO LEARNING RATE
# =====================================================================

from .auto_lr import (
    AutoLRResult,
    default_batch_extractor,
    resolve_sam_lora_learning_rate,
    run_sam_lora_lr_range_test,
)


# =====================================================================
# TRAINING
# =====================================================================

from .trainer import (
    SAMLoRATrainingResult,
    load_sam_lora_trainable_checkpoint,
    train_sam_lora,
)


# =====================================================================
# CONFIG / FACTORY
# =====================================================================

from .factory import (
    SAMLoRAFactoryBundle,
    build_sam_lora_factory_bundle,
    build_sam_lora_from_config,
    build_sam_lora_loss_from_config,
    build_sam_lora_trainer_kwargs,
    create_sam_lora_model_factory,
    load_sam_lora_config,
    print_sam_lora_factory_summary,
    resolve_sam_checkpoint_path,
    resolve_sam_lora_batch_size,
)


# =====================================================================
# DATA BRIDGE
# =====================================================================

from .data import (
    SAMLoRADataBundle,
    create_sam_lora_dataloaders,
    validate_sam_lora_datasets,
    validate_sam_lora_sample,
)


# =====================================================================
# TRAINED MODEL RECONSTRUCTION / INFERENCE
# =====================================================================

from .inference import (
    SAMLoRALoadInfo,
    load_trained_sam_lora,
    predict_sam_lora_logits,
    predict_sam_lora_mask,
    predict_sam_lora_probabilities,
)


# =====================================================================
# PUBLIC API
# =====================================================================

__all__ = [

    # -----------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------

    "LoRAQKV",

    "inject_lora_into_sam_image_encoder",

    "iter_lora_modules",

    "iter_lora_parameters",

    "count_lora_parameters",


    # -----------------------------------------------------------------
    # SAM builder
    # -----------------------------------------------------------------

    "SUPPORTED_SAM_BACKBONES",

    "SAMLoRABuildInfo",

    "build_sam_lora_model",

    "print_sam_lora_build_summary",


    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    "SAMLoRABuildingSegmenter",


    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    "SoftDiceLoss",

    "SAMLoRALoss",

    "SAMLoRALossResult",

    "build_sam_lora_loss",


    # -----------------------------------------------------------------
    # Automatic learning rate
    # -----------------------------------------------------------------

    "AutoLRResult",

    "default_batch_extractor",

    "run_sam_lora_lr_range_test",

    "resolve_sam_lora_learning_rate",


    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    "SAMLoRATrainingResult",

    "train_sam_lora",

    "load_sam_lora_trainable_checkpoint",


    # -----------------------------------------------------------------
    # Factory / configuration
    # -----------------------------------------------------------------

    "SAMLoRAFactoryBundle",

    "load_sam_lora_config",

    "resolve_sam_checkpoint_path",

    "build_sam_lora_loss_from_config",

    "build_sam_lora_trainer_kwargs",

    "build_sam_lora_from_config",

    "create_sam_lora_model_factory",

    "resolve_sam_lora_batch_size",

    "build_sam_lora_factory_bundle",

    "print_sam_lora_factory_summary",


    # -----------------------------------------------------------------
    # Dataset bridge
    # -----------------------------------------------------------------

    "SAMLoRADataBundle",

    "create_sam_lora_dataloaders",

    "validate_sam_lora_datasets",

    "validate_sam_lora_sample",


    # -----------------------------------------------------------------
    # Reconstruction / inference
    # -----------------------------------------------------------------

    "SAMLoRALoadInfo",

    "load_trained_sam_lora",

    "predict_sam_lora_logits",

    "predict_sam_lora_probabilities",

    "predict_sam_lora_mask",
]
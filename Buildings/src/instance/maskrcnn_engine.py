"""
Compatibility wrapper for the historical Mask R-CNN engine.

The real Mask R-CNN implementation now lives under:

    src.models.maskrcnn

This wrapper preserves the old public import path:

    from src.instance.maskrcnn_engine import ...

so existing training, Optuna, threshold-search, post-processing and testing
scripts continue to work without immediate rewrites.

No training mathematics or checkpoint format is changed by this file.
"""

from __future__ import annotations

from src.models.maskrcnn.factory import (
    create_maskrcnn_model,
)

from src.models.maskrcnn.data import (
    MaskRCNNManifestDataset,
    binary_mask_to_target,
    collate_fn,
    make_loaders,
    make_validation_loader,
    read_binary_mask,
    read_manifest_pairs,
    read_rgb_image,
)

from src.models.maskrcnn.trainer import (
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    train_model,
    train_one_epoch,
)

from src.models.maskrcnn.evaluation import (
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    evaluate_validation,
    metrics_from_counts,
    output_union_mask,
    remove_small_components,
    search_postprocessing,
    search_thresholds,
    target_union_mask,
    validation_pixel_iou,
)

from src.models.maskrcnn.inference import (
    load_checkpoint_model,
)


__all__ = [
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "MaskRCNNManifestDataset",
    "binary_mask_to_target",
    "collate_fn",
    "create_maskrcnn_model",
    "evaluate_validation",
    "load_checkpoint_model",
    "make_loaders",
    "make_optimizer",
    "make_scheduler",
    "make_validation_loader",
    "metrics_from_counts",
    "output_union_mask",
    "read_binary_mask",
    "read_manifest_pairs",
    "read_rgb_image",
    "remove_small_components",
    "save_checkpoint",
    "search_postprocessing",
    "search_thresholds",
    "seed_everything",
    "target_union_mask",
    "train_model",
    "train_one_epoch",
    "validation_pixel_iou",
]

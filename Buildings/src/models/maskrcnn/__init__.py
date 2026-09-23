"""
Mask R-CNN model package for the Buildings module.

Package organization
--------------------
maskrcnn.py
    Shared model builder used by src.models.factory.

factory.py
    Dedicated Mask R-CNN factory used by the instance workflow.

data.py
    Manifest, raster, Dataset and DataLoader utilities.

trainer.py
    Training, optimizer, scheduler and checkpoint utilities.

evaluation.py
    Validation metrics, threshold search and post-processing search.

inference.py
    Trained checkpoint reconstruction/loading.

Backward compatibility
----------------------
The old src.instance.maskrcnn_engine API remains available through a thin
compatibility wrapper, so existing scripts do not need to change immediately.
"""

from .maskrcnn import (
    build_maskrcnn,
    build_model,
)

from .factory import (
    create_maskrcnn_model,
)

from .data import (
    MaskRCNNManifestDataset,
    binary_mask_to_target,
    collate_fn,
    make_loaders,
    make_validation_loader,
    read_binary_mask,
    read_manifest_pairs,
    read_rgb_image,
)

from .trainer import (
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    train_model,
    train_one_epoch,
)

from .evaluation import (
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

from .inference import (
    load_checkpoint_model,
)


__all__ = [
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "MaskRCNNManifestDataset",
    "binary_mask_to_target",
    "build_maskrcnn",
    "build_model",
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

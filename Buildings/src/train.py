"""
train.py

Standard training utilities for the Building Extraction module.

Stable standard version:
- Image normalization: image / 255.0 only
- No ImageNet normalization
- BCE + Dice loss
- Optional aux loss support, but standard config uses aux_weight = 0.0
- Compatible with Optuna old style: train_model(model=model, train_loader=...)
- Compatible with standard style: train_model(experiment_config=..., manifest=...)
- Includes threshold_search_on_loader
- Includes save_threshold_results
- Optional shared training augmentation for U-Net / DeepLabV3 / SAM-LoRA
- Shared model-output adapter supports tensor, torchvision {'out': ...},
  and SAM-style {'logits': ...} outputs
- Shared threshold search performs one model forward pass per validation
  batch for all thresholds and records IoU / Precision / Recall / F1
- Validation augmentation is always disabled
"""

from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

from src.raster_io import read_rgb_array, read_mask_array

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class BuildingSegmentationDataset(Dataset):
    """
    Dataset for binary building segmentation.

    Expected manifest columns can be any of:
        image_path / image / image_file
        mask_path / mask / label_path / label

    Output:
        image tensor: float32, shape [3, H, W], range [0, 1]
        mask tensor:  float32, shape [1, H, W], values 0 or 1

    Optional:
        augmentation:
            Shared BuildingAugmentation instance.

            Augmentation must be used for TRAINING ONLY.
            Validation/test datasets should use augmentation=None.
    """

    def __init__(
        self,
        manifest_csv,
        augmentation=None,
        **kwargs
    ):
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)

        self.augmentation = augmentation

        self.image_col = self._find_column(
            [
                "image_path",
                "image",
                "image_file",
                "image_filepath",
                "chip_path",
                "raster_path"
            ]
        )

        self.mask_col = self._find_column(
            [
                "mask_path",
                "mask",
                "label_path",
                "label",
                "label_file",
                "mask_filepath"
            ]
        )


    def _find_column(
        self,
        possible_names
    ):
        for col in possible_names:
            if col in self.df.columns:
                return col

        raise ValueError(
            f"Could not find required column in {self.manifest_csv}. "
            f"Expected one of {possible_names}. "
            f"Available columns: {list(self.df.columns)}"
        )


    def __len__(self):
        return len(self.df)


    def _read_image(
        self,
        path
    ):
        """
        Read an RGB image with shared Pillow/rasterio/ArcPy fallbacks.

        This keeps semantic training compatible with GeoTIFF chips that
        Pillow alone cannot decode.

        Standard experiment normalization remains unchanged:
            image / 255.0
        """
        image = read_rgb_array(
            path
        ).astype(
            np.float32
        )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        # ----------------------------------------------------
        # Standard normalization
        # ----------------------------------------------------

        image = (
            image
            / 255.0
        )

        image = torch.from_numpy(
            image
        )

        image = image.permute(
            2,
            0,
            1
        )

        return image.float().contiguous()


    def _read_mask(
        self,
        path
    ):
        """
        Read a binary mask with shared Pillow/rasterio/ArcPy fallbacks.
        """
        mask = read_mask_array(
            path
        ).astype(
            np.float32
        )

        mask = np.nan_to_num(
            mask,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        if mask.max() > 1.0:
            mask = (
                mask
                / 255.0
            )

        mask = (
            mask > 0.5
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        )

        mask = mask.unsqueeze(
            0
        )

        return mask.float().contiguous()


    def __getitem__(
        self,
        index
    ):
        row = self.df.iloc[
            index
        ]

        image = self._read_image(
            row[
                self.image_col
            ]
        )

        mask = self._read_mask(
            row[
                self.mask_col
            ]
        )

        # ====================================================
        # TRAINING AUGMENTATION
        # ====================================================
        #
        # The same geometric transformation is applied to:
        #
        #   image + mask
        #
        # Photometric augmentation such as brightness and
        # contrast is handled by BuildingAugmentation and is
        # applied to the image only.
        #
        # Validation receives augmentation=None.
        # ====================================================

        if self.augmentation is not None:

            image, mask = (
                self.augmentation(
                    image,
                    mask
                )
            )

        return image, mask


# ============================================================
# DataLoaders
# ============================================================

def create_dataloaders(
    train_manifest_csv,
    val_manifest_csv,
    batch_size,
    num_workers=0,
    pin_memory=True,
    **kwargs
):
    """
    Create train and validation dataloaders.

    Augmentation:
        train_augmentation:
            Optional BuildingAugmentation object.

        Validation augmentation is ALWAYS disabled.

    Extra kwargs remain accepted for compatibility with
    older scripts.
    """

    # --------------------------------------------------------
    # Optional shared augmentation
    # --------------------------------------------------------

    train_augmentation = kwargs.get(
        "train_augmentation",
        kwargs.get(
            "augmentation",
            None
        )
    )


    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = BuildingSegmentationDataset(
        manifest_csv=train_manifest_csv,
        augmentation=train_augmentation
    )


    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Validation data must stay unchanged.
    # --------------------------------------------------------

    val_dataset = BuildingSegmentationDataset(
        manifest_csv=val_manifest_csv,
        augmentation=None
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
        num_workers=int(
            num_workers
        ),
        pin_memory=bool(
            pin_memory
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=False,
        num_workers=int(
            num_workers
        ),
        pin_memory=bool(
            pin_memory
        ),
        drop_last=False
    )

    return (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset
    )


# ============================================================
# Batch helper
# ============================================================

def unpack_batch(
    batch
):
    """
    Support tuple/list batches and dictionary batches.
    """

    if isinstance(
        batch,
        dict
    ):
        images = batch.get(
            "image",
            batch.get(
                "images"
            )
        )

        masks = batch.get(
            "mask",
            batch.get(
                "masks",
                batch.get(
                    "label",
                    batch.get(
                        "labels"
                    )
                )
            )
        )

        if images is None:
            raise KeyError(
                "Batch dictionary missing image/images key."
            )

        if masks is None:
            raise KeyError(
                "Batch dictionary missing "
                "mask/masks/label/labels key."
            )

        return images, masks


    if isinstance(
        batch,
        (
            list,
            tuple
        )
    ):
        if len(batch) < 2:
            raise ValueError(
                "Batch tuple/list must "
                "contain image and mask."
            )

        return (
            batch[0],
            batch[1]
        )


    raise TypeError(
        f"Unsupported batch type: {type(batch)}"
    )


# ============================================================
# Losses
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(
        self,
        smooth=1.0
    ):
        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):
        probs = torch.sigmoid(
            logits
        )

        probs = probs.view(
            probs.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            probs
            * targets
        ).sum(
            dim=1
        )

        denominator = (
            probs.sum(
                dim=1
            )
            +
            targets.sum(
                dim=1
            )
        )

        dice = (
            2.0
            * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0
            - dice.mean()
        )


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits + Dice loss.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0
    ):
        super().__init__()

        self.bce_weight = float(
            bce_weight
        )

        self.dice_weight = float(
            dice_weight
        )

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss(
            smooth=smooth
        )


    def forward(
        self,
        logits,
        targets
    ):
        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return (
            self.bce_weight
            * bce_loss
            +
            self.dice_weight
            * dice_loss
        )


def build_loss_function(
    model_config
):
    """
    Build a binary segmentation loss from model configuration.

    Supported configuration layouts:

        Standard semantic models:
            model_config["loss"]

        SAM-LoRA:
            model_config["training"]["loss"]

    Existing U-Net / DeepLab behavior remains unchanged.
    """

    if not isinstance(
        model_config,
        dict
    ):
        raise TypeError(
            "model_config must be a dictionary."
        )


    # --------------------------------------------------------
    # Standard model layout
    # --------------------------------------------------------

    loss_config = model_config.get(
        "loss",
        None
    )


    # --------------------------------------------------------
    # SAM-LoRA / nested-training layout
    # --------------------------------------------------------

    if not isinstance(
        loss_config,
        dict
    ):

        training_config = (
            model_config.get(
                "training",
                {}
            )
        )

        if isinstance(
            training_config,
            dict
        ):

            loss_config = (
                training_config.get(
                    "loss",
                    {}
                )
            )


    if not isinstance(
        loss_config,
        dict
    ):

        loss_config = {}


    loss_type = str(
        loss_config.get(
            "type",
            "bce_dice"
        )
    ).lower()


    bce_weight = float(
        loss_config.get(
            "bce_weight",
            0.5
        )
    )


    dice_weight = float(
        loss_config.get(
            "dice_weight",
            0.5
        )
    )


    aux_weight = float(
        loss_config.get(
            "aux_weight",
            0.0
        )
    )


    dice_smooth = float(
        loss_config.get(
            "dice_smooth",
            loss_config.get(
                "smooth",
                1.0
            )
        )
    )


    if loss_type in [
        "bce_dice",
        "dice_bce",
        "bce+dice",
        "bce_dice_loss"
    ]:

        loss_fn = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=dice_smooth
        )


    elif loss_type in [
        "bce",
        "bce_logits",
        "binary_cross_entropy"
    ]:

        loss_fn = (
            nn.BCEWithLogitsLoss()
        )


    else:

        raise ValueError(
            f"Unsupported loss type: {loss_type}"
        )


    # --------------------------------------------------------
    # Preserve existing auxiliary-loss compatibility.
    # --------------------------------------------------------

    loss_fn.aux_weight = (
        aux_weight
    )

    return loss_fn


def get_model_output(
    outputs
):
    """
    Extract the main segmentation logits from a model output.

    Supported outputs:

        Tensor
            U-Net and SAM-LoRA default forward output.

        {
            "out": Tensor,
            "aux": ...
        }
            torchvision semantic segmentation models.

        {
            "logits": Tensor,
            ...
        }
            SAM-LoRA auxiliary-output style.

    This function is intentionally shared and model-agnostic.
    """

    if torch.is_tensor(
        outputs
    ):

        return outputs


    if isinstance(
        outputs,
        dict
    ):

        if (
            "out" in outputs
            and
            outputs["out"] is not None
        ):

            return outputs[
                "out"
            ]


        if (
            "logits" in outputs
            and
            outputs["logits"] is not None
        ):

            return outputs[
                "logits"
            ]


        raise KeyError(
            "Model output dictionary does not contain "
            "'out' or 'logits'. "
            f"Available keys: {list(outputs.keys())}"
        )


    raise TypeError(
        "Unsupported model output type: "
        f"{type(outputs)}"
    )


def compute_segmentation_loss(
    outputs,
    masks,
    loss_fn,
    aux_weight=0.0
):
    """
    Compute shared binary segmentation loss.

    Standard tensor outputs and SAM-LoRA logits use the same main-loss
    path.

    torchvision models may additionally expose an auxiliary output
    under:

        outputs["aux"]

    Existing U-Net / DeepLab behavior is preserved.
    """

    main_logits = (
        get_model_output(
            outputs
        )
    )


    loss = loss_fn(
        main_logits,
        masks
    )


    # --------------------------------------------------------
    # Auxiliary loss is relevant only when the model actually
    # provides an "aux" output.
    # --------------------------------------------------------

    if (
        isinstance(
            outputs,
            dict
        )
        and
        aux_weight > 0
        and
        "aux" in outputs
        and
        outputs["aux"] is not None
    ):

        aux_logits = outputs[
            "aux"
        ]


        aux_loss = loss_fn(
            aux_logits,
            masks
        )


        loss = (
            loss
            +
            float(
                aux_weight
            )
            * aux_loss
        )


    return loss


# ============================================================
# Metrics
# ============================================================

def _prepare_binary_masks(
    masks
):
    """
    Convert masks to boolean building/background masks.
    """

    if masks.ndim == 3:

        masks = masks.unsqueeze(
            1
        )


    return (
        masks
        >= 0.5
    )


def segmentation_counts_from_probabilities(
    probabilities,
    masks,
    threshold=0.5
):
    """
    Compute global binary-segmentation pixel counts.

    Returns:
        true_positive,
        false_positive,
        false_negative,
        true_negative

    This helper is shared by threshold search and future model
    evaluation adapters.
    """

    if probabilities.ndim == 3:

        probabilities = (
            probabilities.unsqueeze(
                1
            )
        )


    targets = (
        _prepare_binary_masks(
            masks
        )
    )


    predictions = (
        probabilities
        >= float(
            threshold
        )
    )


    if (
        predictions.shape
        != targets.shape
    ):

        raise ValueError(
            "Prediction and target shapes do not match. "
            f"Predictions={tuple(predictions.shape)}, "
            f"Targets={tuple(targets.shape)}"
        )


    true_positive = torch.logical_and(
        predictions,
        targets
    ).sum()


    false_positive = torch.logical_and(
        predictions,
        torch.logical_not(
            targets
        )
    ).sum()


    false_negative = torch.logical_and(
        torch.logical_not(
            predictions
        ),
        targets
    ).sum()


    true_negative = torch.logical_and(
        torch.logical_not(
            predictions
        ),
        torch.logical_not(
            targets
        )
    ).sum()


    return (
        int(
            true_positive.item()
        ),
        int(
            false_positive.item()
        ),
        int(
            false_negative.item()
        ),
        int(
            true_negative.item()
        )
    )


def metrics_from_segmentation_counts(
    true_positive,
    false_positive,
    false_negative,
    true_negative=0,
):
    """
    Convert accumulated binary pixel counts into shared metrics.

    Empty-mask convention:
        - if prediction and GT are both empty, IoU = 1
        - precision = 1 when there are no predicted positives and
          there are also no missed positives
        - recall = 1 when GT contains no positive pixels
    """

    tp = int(
        true_positive
    )

    fp = int(
        false_positive
    )

    fn = int(
        false_negative
    )

    tn = int(
        true_negative
    )


    union = (
        tp
        + fp
        + fn
    )


    if union == 0:

        iou = 1.0

    else:

        iou = (
            tp
            /
            float(
                union
            )
        )


    precision_denominator = (
        tp
        + fp
    )


    if precision_denominator == 0:

        precision = (
            1.0
            if fn == 0
            else 0.0
        )

    else:

        precision = (
            tp
            /
            float(
                precision_denominator
            )
        )


    recall_denominator = (
        tp
        + fn
    )


    if recall_denominator == 0:

        recall = 1.0

    else:

        recall = (
            tp
            /
            float(
                recall_denominator
            )
        )


    f1_denominator = (
        precision
        + recall
    )


    if f1_denominator == 0.0:

        f1 = 0.0

    else:

        f1 = (
            2.0
            * precision
            * recall
            /
            f1_denominator
        )


    return {
        "iou":
            float(
                iou
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "true_positive":
            int(
                tp
            ),

        "false_positive":
            int(
                fp
            ),

        "false_negative":
            int(
                fn
            ),

        "true_negative":
            int(
                tn
            )
    }


def batch_iou_from_logits(
    logits,
    masks,
    threshold=0.5,
    eps=1e-7
):
    """
    Compute mean per-image IoU for one batch from raw logits.

    IMPORTANT:
        This preserves the existing training/validation IoU behavior
        used by the standard U-Net and DeepLab runs.
    """

    probs = torch.sigmoid(
        logits
    )


    preds = (
        probs
        >= float(
            threshold
        )
    ).float()


    masks = (
        masks
        >= 0.5
    ).float()


    intersection = (
        preds
        * masks
    ).sum(
        dim=(
            1,
            2,
            3
        )
    )


    union = (
        (
            preds
            + masks
        )
        > 0
    ).float().sum(
        dim=(
            1,
            2,
            3
        )
    )


    iou = (
        intersection
        + eps
    ) / (
        union
        + eps
    )


    return iou.mean().item()


def segmentation_metrics_from_logits(
    logits,
    masks,
    threshold=0.5,
):
    """
    Compute shared global pixel IoU / Precision / Recall / F1 for one
    tensor batch.
    """

    probabilities = torch.sigmoid(
        logits
    )


    (
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    ) = (
        segmentation_counts_from_probabilities(
            probabilities=probabilities,
            masks=masks,
            threshold=threshold
        )
    )


    return metrics_from_segmentation_counts(
        true_positive=(
            true_positive
        ),
        false_positive=(
            false_positive
        ),
        false_negative=(
            false_negative
        ),
        true_negative=(
            true_negative
        )
    )


# ============================================================
# Threshold Search
# ============================================================

def threshold_search_on_loader(
    model,
    data_loader=None,
    device=None,
    thresholds=None,
    output_folder=None,
    output_csv=None,
    summary_json=None,
    **kwargs
):
    """
    Shared validation-threshold search for binary semantic models.

    Compatible models include:
        - U-Net
        - DeepLabV3+
        - SAM-LoRA

    Model outputs may be:
        - Tensor
        - {"out": Tensor}
        - {"logits": Tensor}

    Performance
    -----------
    The previous implementation reran the complete model once for every
    threshold.

    This implementation performs:

        ONE model forward pass per validation batch

    and evaluates every threshold from the resulting probabilities.

    This is especially important for SAM-LoRA because its image encoder
    is much more expensive than thresholding a probability tensor.

    Backward compatibility
    ----------------------
    The legacy "iou" / "val_iou" fields remain the mean batch IoU used
    by the previous threshold-search implementation.

    Additional shared metrics are now recorded:
        global_iou
        precision
        recall
        f1

    Optional keyword:
        selection_metric

    Supported:
        "iou"         -> legacy mean-batch IoU (default)
        "val_iou"     -> same as iou
        "global_iou"  -> accumulated pixel IoU
        "f1"

    Returns:
        rows, best_threshold_row
    """

    if data_loader is None:

        data_loader = kwargs.get(
            "val_loader",
            None
        )


    if data_loader is None:

        data_loader = kwargs.get(
            "validation_loader",
            None
        )


    if data_loader is None:

        data_loader = kwargs.get(
            "loader",
            None
        )


    if data_loader is None:

        raise ValueError(
            "data_loader is required. "
            "Pass data_loader, val_loader, "
            "or validation_loader."
        )


    if device is None:

        device = kwargs.get(
            "device",
            None
        )


    if device is None:

        device = next(
            model.parameters()
        ).device


    device = torch.device(
        device
    )


    if thresholds is None:

        thresholds = kwargs.get(
            "threshold_list",
            None
        )


    if thresholds is None:

        thresholds = kwargs.get(
            "threshold_values",
            None
        )


    if thresholds is None:

        thresholds = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.70,
            0.80,
            0.90
        ]


    thresholds = [
        float(
            threshold
        )
        for threshold
        in thresholds
    ]


    if len(
        thresholds
    ) == 0:

        raise ValueError(
            "thresholds cannot be empty."
        )


    for threshold in thresholds:

        if (
            threshold < 0.0
            or threshold > 1.0
        ):

            raise ValueError(
                "All thresholds must satisfy "
                "0 <= threshold <= 1. "
                f"Received {threshold}."
            )


    selection_metric = str(
        kwargs.get(
            "selection_metric",
            "iou"
        )
    ).strip().lower()


    supported_selection_metrics = {
        "iou",
        "val_iou",
        "global_iou",
        "f1"
    }


    if (
        selection_metric
        not in supported_selection_metrics
    ):

        raise ValueError(
            "Unsupported threshold selection_metric: "
            f"{selection_metric}. "
            f"Supported: "
            f"{sorted(supported_selection_metrics)}"
        )


    model.eval()


    # --------------------------------------------------------
    # One statistics accumulator per threshold.
    #
    # batch_iou_sum preserves the legacy threshold-search IoU.
    #
    # TP / FP / FN / TN provide shared global metrics.
    # --------------------------------------------------------

    statistics = {
        threshold: {
            "batch_iou_sum": 0.0,
            "batch_count": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0
        }

        for threshold
        in thresholds
    }


    with torch.no_grad():

        for batch in data_loader:

            images, masks = (
                unpack_batch(
                    batch
                )
            )


            images = images.to(
                device,
                non_blocking=True
            )


            masks = masks.to(
                device,
                non_blocking=True
            )


            outputs = model(
                images
            )


            logits = (
                get_model_output(
                    outputs
                )
            )


            probabilities = torch.sigmoid(
                logits
            )


            # ------------------------------------------------
            # Evaluate every threshold WITHOUT rerunning model.
            # ------------------------------------------------

            for threshold in thresholds:

                # Legacy batch-mean IoU.
                predictions_float = (
                    probabilities
                    >= threshold
                ).float()


                binary_masks_float = (
                    masks
                    >= 0.5
                ).float()


                intersection = (
                    predictions_float
                    * binary_masks_float
                ).sum(
                    dim=(
                        1,
                        2,
                        3
                    )
                )


                union = (
                    (
                        predictions_float
                        + binary_masks_float
                    )
                    > 0
                ).float().sum(
                    dim=(
                        1,
                        2,
                        3
                    )
                )


                batch_iou = (
                    (
                        intersection
                        + 1e-7
                    )
                    /
                    (
                        union
                        + 1e-7
                    )
                ).mean().item()


                statistics[
                    threshold
                ][
                    "batch_iou_sum"
                ] += float(
                    batch_iou
                )


                statistics[
                    threshold
                ][
                    "batch_count"
                ] += 1


                # --------------------------------------------
                # Shared accumulated pixel metrics.
                # --------------------------------------------

                (
                    true_positive,
                    false_positive,
                    false_negative,
                    true_negative,
                ) = (
                    segmentation_counts_from_probabilities(
                        probabilities=(
                            probabilities
                        ),
                        masks=masks,
                        threshold=threshold
                    )
                )


                statistics[
                    threshold
                ][
                    "true_positive"
                ] += int(
                    true_positive
                )


                statistics[
                    threshold
                ][
                    "false_positive"
                ] += int(
                    false_positive
                )


                statistics[
                    threshold
                ][
                    "false_negative"
                ] += int(
                    false_negative
                )


                statistics[
                    threshold
                ][
                    "true_negative"
                ] += int(
                    true_negative
                )


    rows = []


    for threshold in thresholds:

        stats = statistics[
            threshold
        ]


        legacy_iou = (
            stats[
                "batch_iou_sum"
            ]
            /
            max(
                stats[
                    "batch_count"
                ],
                1
            )
        )


        global_metrics = (
            metrics_from_segmentation_counts(
                true_positive=(
                    stats[
                        "true_positive"
                    ]
                ),
                false_positive=(
                    stats[
                        "false_positive"
                    ]
                ),
                false_negative=(
                    stats[
                        "false_negative"
                    ]
                ),
                true_negative=(
                    stats[
                        "true_negative"
                    ]
                )
            )
        )


        rows.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                # --------------------------------------------
                # Backward-compatible fields.
                # --------------------------------------------

                "iou":
                    float(
                        legacy_iou
                    ),

                "val_iou":
                    float(
                        legacy_iou
                    ),

                # --------------------------------------------
                # New shared metrics.
                # --------------------------------------------

                "global_iou":
                    float(
                        global_metrics[
                            "iou"
                        ]
                    ),

                "precision":
                    float(
                        global_metrics[
                            "precision"
                        ]
                    ),

                "recall":
                    float(
                        global_metrics[
                            "recall"
                        ]
                    ),

                "f1":
                    float(
                        global_metrics[
                            "f1"
                        ]
                    ),

                "true_positive":
                    int(
                        global_metrics[
                            "true_positive"
                        ]
                    ),

                "false_positive":
                    int(
                        global_metrics[
                            "false_positive"
                        ]
                    ),

                "false_negative":
                    int(
                        global_metrics[
                            "false_negative"
                        ]
                    ),

                "true_negative":
                    int(
                        global_metrics[
                            "true_negative"
                        ]
                    )
            }
        )


    best_threshold_row = max(
        rows,
        key=lambda row:
            row[
                selection_metric
            ]
    )


    save_threshold_results(
        threshold_results=rows,
        output_folder=output_folder,
        output_csv=output_csv,
        summary_json=summary_json,
        selection_metric=(
            selection_metric
        )
    )


    return (
        rows,
        best_threshold_row
    )

def save_threshold_results(
    threshold_results,
    output_folder=None,
    output_csv=None,
    summary_json=None,
    **kwargs
):
    """
    Save threshold search results.

    Accepts:
        - dict
        - list of threshold rows
    """

    if output_csv is None:

        output_csv = kwargs.get(
            "output_csv_path",
            None
        )


    if summary_json is None:

        summary_json = kwargs.get(
            "summary_json_path",
            None
        )


    if output_folder is None:

        output_folder = kwargs.get(
            "save_folder",
            None
        )


    selection_metric = str(
        kwargs.get(
            "selection_metric",
            "iou"
        )
    ).strip().lower()


    if isinstance(
        threshold_results,
        dict
    ):

        result = dict(
            threshold_results
        )

        rows = result.get(
            "threshold_results",
            []
        )


        selection_metric = str(
            result.get(
                "selection_metric",
                selection_metric
            )
        ).strip().lower()


    elif isinstance(
        threshold_results,
        list
    ):

        rows = (
            threshold_results
        )


        if len(rows) == 0:

            raise ValueError(
                "threshold_results list is empty."
            )


        if (
            selection_metric
            not in rows[0]
        ):

            raise KeyError(
                "Threshold result rows do not contain "
                f"selection metric '{selection_metric}'. "
                f"Available keys: {list(rows[0].keys())}"
            )


        best_row = max(
            rows,
            key=lambda row:
                row.get(
                    selection_metric,
                    float(
                        "-inf"
                    )
                )
        )


        result = {
            "selection_metric":
                selection_metric,

            "best_threshold":
                float(
                    best_row[
                        "threshold"
                    ]
                ),

            "best_iou":
                float(
                    best_row.get(
                        "iou",
                        best_row.get(
                            "val_iou",
                            0.0
                        )
                    )
                ),

            "best_val_iou":
                float(
                    best_row.get(
                        "val_iou",
                        best_row.get(
                            "iou",
                            0.0
                        )
                    )
                ),

            "best_global_iou":
                (
                    float(
                        best_row[
                            "global_iou"
                        ]
                    )
                    if "global_iou"
                    in best_row
                    else None
                ),

            "best_precision":
                (
                    float(
                        best_row[
                            "precision"
                        ]
                    )
                    if "precision"
                    in best_row
                    else None
                ),

            "best_recall":
                (
                    float(
                        best_row[
                            "recall"
                        ]
                    )
                    if "recall"
                    in best_row
                    else None
                ),

            "best_f1":
                (
                    float(
                        best_row[
                            "f1"
                        ]
                    )
                    if "f1"
                    in best_row
                    else None
                ),

            "threshold_results":
                rows
        }


    else:

        raise TypeError(
            "Unsupported threshold_results "
            f"type: {type(threshold_results)}"
        )


    if output_folder is not None:

        output_folder = Path(
            output_folder
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        if output_csv is None:

            output_csv = (
                output_folder
                / "threshold_results.csv"
            )


        if summary_json is None:

            summary_json = (
                output_folder
                / "threshold_summary.json"
            )


    if output_csv is not None:

        output_csv = Path(
            output_csv
        )

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(
            rows
        ).to_csv(
            output_csv,
            index=False
        )

        result[
            "threshold_results_csv"
        ] = str(
            output_csv
        )


    if summary_json is not None:

        summary_json = Path(
            summary_json
        )

        summary_json.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            summary_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                _json_safe(
                    result
                ),
                f,
                indent=2
            )


        result[
            "threshold_summary_json"
        ] = str(
            summary_json
        )


    return result


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
    general_params,
    model_config=None
):
    """
    Build optimizer from general_params.
    """

    training_config = (
        general_params.get(
            "training",
            {}
        )
    )

    optimizer_name = str(
        training_config.get(
            "optimizer",
            "adamw"
        )
    ).lower()

    learning_rate = float(
        training_config.get(
            "learning_rate",
            training_config.get(
                "lr",
                1e-4
            )
        )
    )

    weight_decay = float(
        training_config.get(
            "weight_decay",
            1e-5
        )
    )


    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    if optimizer_name == "sgd":

        momentum = float(
            training_config.get(
                "momentum",
                0.9
            )
        )

        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )


    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


# ============================================================
# Training loops
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    threshold=0.5
):
    """
    Train one epoch.
    """

    model.train()

    total_loss = 0.0
    total_iou = 0.0
    batch_count = 0

    aux_weight = float(
        getattr(
            loss_fn,
            "aux_weight",
            0.0
        )
    )


    for batch in train_loader:

        images, masks = (
            unpack_batch(
                batch
            )
        )

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = (
            compute_segmentation_loss(
                outputs=outputs,
                masks=masks,
                loss_fn=loss_fn,
                aux_weight=aux_weight
            )
        )

        logits = (
            get_model_output(
                outputs
            )
        )

        loss.backward()

        optimizer.step()


        batch_iou = (
            batch_iou_from_logits(
                logits=logits.detach(),
                masks=masks.detach(),
                threshold=threshold
            )
        )

        total_loss += (
            loss.item()
        )

        total_iou += (
            batch_iou
        )

        batch_count += 1


    avg_loss = (
        total_loss
        /
        max(
            batch_count,
            1
        )
    )

    avg_iou = (
        total_iou
        /
        max(
            batch_count,
            1
        )
    )


    return (
        avg_loss,
        avg_iou
    )


def validate_one_epoch(
    model,
    val_loader,
    loss_fn,
    device,
    threshold=0.5
):
    """
    Validate one epoch.

    Validation augmentation is never applied by create_dataloaders().
    """

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    batch_count = 0

    aux_weight = float(
        getattr(
            loss_fn,
            "aux_weight",
            0.0
        )
    )


    with torch.no_grad():

        for batch in val_loader:

            images, masks = (
                unpack_batch(
                    batch
                )
            )

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            outputs = model(
                images
            )

            loss = (
                compute_segmentation_loss(
                    outputs=outputs,
                    masks=masks,
                    loss_fn=loss_fn,
                    aux_weight=aux_weight
                )
            )

            logits = (
                get_model_output(
                    outputs
                )
            )

            batch_iou = (
                batch_iou_from_logits(
                    logits=logits,
                    masks=masks,
                    threshold=threshold
                )
            )

            total_loss += (
                loss.item()
            )

            total_iou += (
                batch_iou
            )

            batch_count += 1


    avg_loss = (
        total_loss
        /
        max(
            batch_count,
            1
        )
    )

    avg_iou = (
        total_iou
        /
        max(
            batch_count,
            1
        )
    )


    return (
        avg_loss,
        avg_iou
    )


# ============================================================
# JSON helpers
# ============================================================

def _json_safe(
    obj
):
    """
    Convert objects to JSON-safe values.
    """

    if isinstance(
        obj,
        Path
    ):

        return str(
            obj
        )


    if isinstance(
        obj,
        dict
    ):

        return {
            key:
                _json_safe(
                    value
                )

            for key, value
            in obj.items()
        }


    if isinstance(
        obj,
        list
    ):

        return [
            _json_safe(
                value
            )

            for value
            in obj
        ]


    if isinstance(
        obj,
        tuple
    ):

        return [
            _json_safe(
                value
            )

            for value
            in obj
        ]


    if isinstance(
        obj,
        np.integer
    ):

        return int(
            obj
        )


    if isinstance(
        obj,
        np.floating
    ):

        return float(
            obj
        )


    return obj


def _save_json(
    data,
    path
):
    """
    Save JSON file.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            _json_safe(
                data
            ),
            f,
            indent=2
        )


# ============================================================
# Train from ready objects
# ============================================================

def _train_model_from_ready_objects(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    num_epochs,
    output_folder,
    threshold=0.5,
    experiment_config=None,
    model_config=None,
    backbone_info=None,
    trainable_params=None,
    augmentation_enabled=False,
    augmentation_config_path=None
):
    """
    Training path used when model/dataloaders already exist.

    This keeps compatibility with older Optuna scripts.
    """

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    if trainable_params is None:

        trainable_params = sum(
            p.numel()

            for p
            in model.parameters()

            if p.requires_grad
        )


    best_val_iou = -1.0
    best_epoch = -1


    best_model_path = (
        output_folder
        / "best_model.pth"
    )

    training_log_csv = (
        output_folder
        / "training_log.csv"
    )

    training_summary_json = (
        output_folder
        / "training_summary.json"
    )


    logs = []


    print(
        "\n========== TRAINING START =========="
    )

    print(
        f"Epochs: {num_epochs}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Train Batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation Batches: "
        f"{len(val_loader)}"
    )

    print(
        "Image Normalization: "
        "/255.0 only"
    )

    print(
        f"Aux Weight: "
        f"{getattr(loss_fn, 'aux_weight', 0.0)}"
    )

    print(
        "Training Augmentation: "
        f"{'ON' if augmentation_enabled else 'OFF'}"
    )


    if (
        augmentation_enabled
        and
        augmentation_config_path
        is not None
    ):

        print(
            "Augmentation Config: "
            f"{augmentation_config_path}"
        )


    print(
        "Validation Augmentation: OFF"
    )

    print(
        "====================================\n"
    )


    for epoch in range(
        1,
        int(
            num_epochs
        ) + 1
    ):

        start_time = (
            time.time()
        )


        train_loss, train_iou = (
            train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                threshold=threshold
            )
        )


        val_loss, val_iou = (
            validate_one_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                threshold=threshold
            )
        )


        elapsed = (
            time.time()
            - start_time
        )


        row = {
            "epoch":
                int(
                    epoch
                ),

            "train_loss":
                float(
                    train_loss
                ),

            "train_iou":
                float(
                    train_iou
                ),

            "val_loss":
                float(
                    val_loss
                ),

            "val_iou":
                float(
                    val_iou
                ),

            "time_seconds":
                float(
                    elapsed
                )
        }


        logs.append(
            row
        )


        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Time: {elapsed:.2f}s"
        )


        if val_iou > best_val_iou:

            best_val_iou = float(
                val_iou
            )

            best_epoch = int(
                epoch
            )


            checkpoint = {
                "epoch":
                    best_epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_iou":
                    best_val_iou,

                "best_validation_iou":
                    best_val_iou,

                "best_training_val_iou":
                    best_val_iou,

                "trainable_params":
                    int(
                        trainable_params
                    ),

                "experiment_config":
                    experiment_config,

                "model_config":
                    model_config,

                "backbone_info":
                    backbone_info,

                "augmentation_enabled":
                    bool(
                        augmentation_enabled
                    ),

                "augmentation_config_path":
                    (
                        str(
                            augmentation_config_path
                        )

                        if augmentation_config_path
                        is not None

                        else None
                    )
            }


            torch.save(
                checkpoint,
                best_model_path
            )


            print(
                f"Best model saved: "
                f"{best_model_path}"
            )


        pd.DataFrame(
            logs
        ).to_csv(
            training_log_csv,
            index=False
        )


    summary = {
        "epochs":
            int(
                num_epochs
            ),

        "train_batches":
            int(
                len(
                    train_loader
                )
            ),

        "validation_batches":
            int(
                len(
                    val_loader
                )
            ),

        "trainable_params":
            int(
                trainable_params
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_iou":
            float(
                best_val_iou
            ),

        "best_validation_iou":
            float(
                best_val_iou
            ),

        "best_training_val_iou":
            float(
                best_val_iou
            ),

        "best_model_path":
            str(
                best_model_path
            ),

        "training_log_csv":
            str(
                training_log_csv
            ),

        "training_summary_json":
            str(
                training_summary_json
            ),

        "normalization":
            "divide_by_255_only",

        "aux_weight":
            float(
                getattr(
                    loss_fn,
                    "aux_weight",
                    0.0
                )
            ),

        "augmentation_enabled":
            bool(
                augmentation_enabled
            ),

        "augmentation_config_path":
            (
                str(
                    augmentation_config_path
                )

                if augmentation_config_path
                is not None

                else None
            ),

        "validation_augmentation":
            False
    }


    if isinstance(
        experiment_config,
        dict
    ):

        summary.update(
            {
                "model_type":
                    experiment_config.get(
                        "model_type"
                    ),

                "backbone":
                    experiment_config.get(
                        "backbone"
                    ),

                "source_type":
                    experiment_config.get(
                        "source_type"
                    ),

                "tile_size":
                    experiment_config.get(
                        "tile_size"
                    ),

                "dataset_type":
                    experiment_config.get(
                        "dataset_type"
                    )
            }
        )


    _save_json(
        summary,
        training_summary_json
    )


    print(
        "\n========== TRAINING FINISHED =========="
    )

    print(
        f"Best Validation IoU: "
        f"{best_val_iou:.4f}"
    )

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Model Path: "
        f"{best_model_path}"
    )

    print(
        f"Training Log CSV: "
        f"{training_log_csv}"
    )

    print(
        f"Training Summary: "
        f"{training_summary_json}"
    )

    print(
        "=======================================\n"
    )


    return summary


# ============================================================
# Main compatible train_model
# ============================================================

def train_model(
    *args,
    **kwargs
):
    """
    Compatible train_model.

    Supports two styles:

    1. Ready-object / Optuna style:

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=...,
            output_folder=...
        )

    2. Standard config style:

        train_model(
            experiment_config=...,
            general_params=...,
            model_config=...,
            backbone_info=...,
            train_manifest_csv=...,
            val_manifest_csv=...,
            output_folder=...
        )
    """

    # ========================================================
    # STYLE 1
    # Ready objects / older Optuna compatibility
    # ========================================================

    if "model" in kwargs:

        model = kwargs[
            "model"
        ]


        train_loader = kwargs.get(
            "train_loader",
            kwargs.get(
                "training_loader",
                None
            )
        )


        val_loader = kwargs.get(
            "val_loader",
            kwargs.get(
                "validation_loader",
                None
            )
        )


        loss_fn = kwargs.get(
            "loss_fn",
            kwargs.get(
                "loss_function",
                kwargs.get(
                    "criterion",
                    None
                )
            )
        )


        optimizer = kwargs.get(
            "optimizer",
            None
        )


        device = kwargs.get(
            "device",
            None
        )


        num_epochs = kwargs.get(
            "num_epochs",
            kwargs.get(
                "epochs",
                kwargs.get(
                    "n_epochs",
                    40
                )
            )
        )


        output_folder = kwargs.get(
            "output_folder",
            kwargs.get(
                "save_folder",
                None
            )
        )


        threshold = float(
            kwargs.get(
                "threshold",
                kwargs.get(
                    "training_threshold",
                    0.5
                )
            )
        )


        if train_loader is None:

            raise ValueError(
                "train_loader is required "
                "when using model= style."
            )


        if val_loader is None:

            raise ValueError(
                "val_loader is required "
                "when using model= style."
            )


        if loss_fn is None:

            raise ValueError(
                "loss_fn is required "
                "when using model= style."
            )


        if optimizer is None:

            raise ValueError(
                "optimizer is required "
                "when using model= style."
            )


        if device is None:

            device = next(
                model.parameters()
            ).device


        if output_folder is None:

            raise ValueError(
                "output_folder is required "
                "when using model= style."
            )


        return (
            _train_model_from_ready_objects(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                output_folder=output_folder,
                threshold=threshold,

                experiment_config=kwargs.get(
                    "experiment_config",
                    None
                ),

                model_config=kwargs.get(
                    "model_config",
                    None
                ),

                backbone_info=kwargs.get(
                    "backbone_info",
                    None
                ),

                trainable_params=kwargs.get(
                    "trainable_params",
                    None
                ),

                augmentation_enabled=bool(
                    kwargs.get(
                        "augmentation_enabled",
                        False
                    )
                ),

                augmentation_config_path=kwargs.get(
                    "augmentation_config_path",
                    None
                )
            )
        )


    # ========================================================
    # STYLE 2
    # Standard config-based training
    # ========================================================

    experiment_config = kwargs.get(
        "experiment_config",
        None
    )

    general_params = kwargs.get(
        "general_params",
        None
    )

    model_config = kwargs.get(
        "model_config",
        None
    )

    backbone_info = kwargs.get(
        "backbone_info",
        None
    )


    train_manifest_csv = kwargs.get(
        "train_manifest_csv",
        kwargs.get(
            "train_csv",
            None
        )
    )


    val_manifest_csv = kwargs.get(
        "val_manifest_csv",
        kwargs.get(
            "validation_manifest_csv",
            kwargs.get(
                "val_csv",
                None
            )
        )
    )


    output_folder = kwargs.get(
        "output_folder",
        kwargs.get(
            "save_folder",
            None
        )
    )


    batch_size = kwargs.get(
        "batch_size",
        None
    )


    num_epochs = kwargs.get(
        "num_epochs",
        kwargs.get(
            "epochs",
            None
        )
    )


    # --------------------------------------------------------
    # Positional compatibility
    # --------------------------------------------------------

    if (
        experiment_config is None
        and
        len(args) >= 1
    ):
        experiment_config = (
            args[0]
        )


    if (
        general_params is None
        and
        len(args) >= 2
    ):
        general_params = (
            args[1]
        )


    if (
        model_config is None
        and
        len(args) >= 3
    ):
        model_config = (
            args[2]
        )


    if (
        backbone_info is None
        and
        len(args) >= 4
    ):
        backbone_info = (
            args[3]
        )


    if (
        train_manifest_csv is None
        and
        len(args) >= 5
    ):
        train_manifest_csv = (
            args[4]
        )


    if (
        val_manifest_csv is None
        and
        len(args) >= 6
    ):
        val_manifest_csv = (
            args[5]
        )


    if (
        output_folder is None
        and
        len(args) >= 7
    ):
        output_folder = (
            args[6]
        )


    # --------------------------------------------------------
    # Required arguments
    # --------------------------------------------------------

    if experiment_config is None:

        raise ValueError(
            "experiment_config is required."
        )


    if general_params is None:

        raise ValueError(
            "general_params is required."
        )


    if model_config is None:

        raise ValueError(
            "model_config is required."
        )


    if backbone_info is None:

        raise ValueError(
            "backbone_info is required."
        )


    if train_manifest_csv is None:

        raise ValueError(
            "train_manifest_csv is required."
        )


    if val_manifest_csv is None:

        raise ValueError(
            "val_manifest_csv is required."
        )


    if output_folder is None:

        raise ValueError(
            "output_folder is required."
        )


    # ========================================================
    # Module imports
    # ========================================================

    from src.models.factory import (
        build_model
    )

    from src.utils import (
        get_automatic_batch_size,
        set_random_seed
    )


    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # General training configuration
    # ========================================================

    training_config = (
        general_params.get(
            "training",
            {}
        )
    )


    seed = int(
        training_config.get(
            "random_seed",
            training_config.get(
                "seed",
                42
            )
        )
    )


    set_random_seed(
        seed
    )


    # Deterministic CUDA behavior for reproducible standard runs.
    if torch.cuda.is_available():

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    if batch_size is None:

        batch_size = (
            get_automatic_batch_size(
                experiment_config=experiment_config,
                backbone_info=backbone_info
            )
        )


    if num_epochs is None:

        num_epochs = int(
            training_config.get(
                "epochs",
                40
            )
        )


    num_workers = int(
        training_config.get(
            "num_workers",
            0
        )
    )


    pin_memory = bool(
        training_config.get(
            "pin_memory",
            True
        )
    )


    threshold = float(
        model_config.get(
            "output",
            {}
        ).get(
            "threshold",
            0.5
        )
    )


    # ========================================================
    # OPTIONAL SHARED AUGMENTATION
    # ========================================================

    optional_config = (
        experiment_config.get(
            "optional",
            {}
        )
    )


    use_augmentation = bool(
        experiment_config.get(
            "use_augmentation",
            optional_config.get(
                "use_augmentation",
                False
            )
        )
    )


    augmentation_config_path = (
        kwargs.get(
            "augmentation_config_path",
            experiment_config.get(
                "augmentation_config_path",
                None
            )
        )
    )


    train_augmentation = None


    if use_augmentation:

        from src.optional.augmentation import (
            build_augmentation,
            describe_augmentation
        )


        # ----------------------------------------------------
        # Default augmentation configuration
        # ----------------------------------------------------

        if augmentation_config_path is None:

            buildings_root = (
                Path(
                    __file__
                )
                .resolve()
                .parents[1]
            )


            augmentation_config_path = (
                buildings_root
                / "config"
                / "optional"
                / "augmentation.json"
            )


        else:

            augmentation_config_path = Path(
                augmentation_config_path
            )


        # ----------------------------------------------------
        # Build shared augmentation pipeline
        # ----------------------------------------------------
        #
        # experiment_config["use_augmentation"] is treated
        # as the experiment-level master switch.
        # ----------------------------------------------------

        train_augmentation = (
            build_augmentation(
                config_path=augmentation_config_path,
                enabled_override=True
            )
        )


        if train_augmentation is None:

            raise RuntimeError(
                "use_augmentation=True but "
                "augmentation could not be built."
            )


        print(
            "\nTraining augmentation enabled."
        )


        describe_augmentation(
            augmentation_config_path
        )


    else:

        augmentation_config_path = None

        print(
            "\nTraining augmentation disabled."
        )


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader, val_loader, train_dataset, val_dataset = (
        create_dataloaders(
            train_manifest_csv=train_manifest_csv,
            val_manifest_csv=val_manifest_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,

            # Only training gets augmentation
            train_augmentation=train_augmentation
        )
    )


    # ========================================================
    # Model
    # ========================================================

    model, device, trainable_params = (
        build_model(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
            move_to_device=True
        )
    )


    # ========================================================
    # Loss
    # ========================================================

    loss_fn = (
        build_loss_function(
            model_config
        )
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = (
        build_optimizer(
            model=model,
            general_params=general_params,
            model_config=model_config
        )
    )


    # ========================================================
    # Training
    # ========================================================

    summary = (
        _train_model_from_ready_objects(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            output_folder=output_folder,
            threshold=threshold,
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
            trainable_params=trainable_params,

            augmentation_enabled=use_augmentation,

            augmentation_config_path=(
                augmentation_config_path
            )
        )
    )


    # ========================================================
    # Final experiment metadata
    # ========================================================

    summary[
        "train_samples"
    ] = int(
        len(
            train_dataset
        )
    )


    summary[
        "validation_samples"
    ] = int(
        len(
            val_dataset
        )
    )


    summary[
        "augmentation_enabled"
    ] = bool(
        use_augmentation
    )


    summary[
        "augmentation_config_path"
    ] = (
        str(
            augmentation_config_path
        )

        if augmentation_config_path
        is not None

        else None
    )


    summary[
        "validation_augmentation"
    ] = False


    summary[
        "training_seed"
    ] = int(
        seed
    )


    summary[
        "deterministic_cudnn"
    ] = bool(
        torch.cuda.is_available()
    )


    # ========================================================
    # Save final summary
    # ========================================================

    _save_json(
        summary,
        output_folder
        / "training_summary.json"
    )


    return summary
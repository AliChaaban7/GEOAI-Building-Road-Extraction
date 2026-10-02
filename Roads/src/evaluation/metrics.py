"""
Core evaluation metrics for Approach 2 - Road Extraction.

Primary metrics
---------------
- IoU
- Precision
- Recall
- F1-score

These metrics operate on binary road masks:

    0 = background
    1 = road

The functions support:
- raw model logits
- sigmoid probabilities
- binary predictions
- individual batches
- accumulated full-dataset evaluation

Important
---------
clDice / connectivity-aware metrics are implemented separately in:

    src/evaluation/connectivity_metrics.py

This file contains only the common binary segmentation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _validate_threshold(
    threshold: float,
) -> float:
    """
    Validate and normalize a probability threshold.
    """
    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return threshold


def _prepare_target(
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a target mask to boolean foreground representation.
    """
    if not isinstance(
        target,
        torch.Tensor,
    ):
        target = torch.as_tensor(
            target
        )

    return (
        target > 0.5
    )


def _prepare_prediction(
    prediction: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Convert logits/probabilities/binary values into a boolean mask.
    """
    if not isinstance(
        prediction,
        torch.Tensor,
    ):
        prediction = torch.as_tensor(
            prediction
        )

    threshold = _validate_threshold(
        threshold
    )

    prediction = prediction.float()

    if from_logits:
        prediction = torch.sigmoid(
            prediction
        )

    return (
        prediction >= threshold
    )


# ---------------------------------------------------------------------
# Confusion counts
# ---------------------------------------------------------------------

@dataclass
class BinaryConfusionCounts:
    """
    Pixel-level confusion counts for binary Road segmentation.
    """

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        other: "BinaryConfusionCounts",
    ) -> None:
        """
        Add counts from another result.
        """
        self.true_positive += int(
            other.true_positive
        )

        self.false_positive += int(
            other.false_positive
        )

        self.false_negative += int(
            other.false_negative
        )

        self.true_negative += int(
            other.true_negative
        )

    def total_pixels(
        self,
    ) -> int:
        """
        Return total evaluated pixels.
        """
        return int(
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    def to_dict(
        self,
    ) -> Dict[str, int]:
        """
        Convert counts to a serializable dictionary.
        """
        return {
            "true_positive":
                int(
                    self.true_positive
                ),

            "false_positive":
                int(
                    self.false_positive
                ),

            "false_negative":
                int(
                    self.false_negative
                ),

            "true_negative":
                int(
                    self.true_negative
                ),

            "total_pixels":
                self.total_pixels(),
        }


@torch.no_grad()
def binary_confusion_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> BinaryConfusionCounts:
    """
    Calculate pixel-level binary confusion counts.

    Parameters
    ----------
    prediction:
        Model output.

        Typical shape:
            [B, 1, H, W]

    target:
        Ground-truth binary mask.

        Typical shape:
            [B, 1, H, W]

    threshold:
        Probability threshold used to convert predictions into
        binary Road masks.

    from_logits:
        True:
            prediction contains raw logits.

        False:
            prediction already contains probabilities or binary values.
    """

    pred_binary = _prepare_prediction(
        prediction=prediction,
        threshold=threshold,
        from_logits=from_logits,
    )

    target_binary = _prepare_target(
        target
    )

    if (
        pred_binary.shape
        != target_binary.shape
    ):
        raise ValueError(
            "Prediction/target shape mismatch.\n"
            f"Prediction: {tuple(pred_binary.shape)}\n"
            f"Target: {tuple(target_binary.shape)}"
        )

    pred_binary = pred_binary.reshape(
        -1
    )

    target_binary = target_binary.reshape(
        -1
    )

    true_positive = torch.logical_and(
        pred_binary,
        target_binary,
    ).sum()

    false_positive = torch.logical_and(
        pred_binary,
        torch.logical_not(
            target_binary
        ),
    ).sum()

    false_negative = torch.logical_and(
        torch.logical_not(
            pred_binary
        ),
        target_binary,
    ).sum()

    true_negative = torch.logical_and(
        torch.logical_not(
            pred_binary
        ),
        torch.logical_not(
            target_binary
        ),
    ).sum()

    return BinaryConfusionCounts(
        true_positive=int(
            true_positive.item()
        ),

        false_positive=int(
            false_positive.item()
        ),

        false_negative=int(
            false_negative.item()
        ),

        true_negative=int(
            true_negative.item()
        ),
    )


# ---------------------------------------------------------------------
# Metrics from counts
# ---------------------------------------------------------------------

def metrics_from_counts(
    counts: BinaryConfusionCounts,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculate Road segmentation metrics from accumulated confusion counts.

    Returns
    -------
    dict
        iou
        precision
        recall
        f1
        pixel_accuracy
    """

    tp = float(
        counts.true_positive
    )

    fp = float(
        counts.false_positive
    )

    fn = float(
        counts.false_negative
    )

    tn = float(
        counts.true_negative
    )

    epsilon = float(
        epsilon
    )

    # --------------------------------------------------------------
    # IoU
    #
    # TP / (TP + FP + FN)
    # --------------------------------------------------------------

    iou_denominator = (
        tp + fp + fn
    )

    iou = (
        tp
        / (
            iou_denominator
            + epsilon
        )
        if iou_denominator > 0
        else 1.0
    )

    # --------------------------------------------------------------
    # Precision
    #
    # TP / (TP + FP)
    # --------------------------------------------------------------

    precision_denominator = (
        tp + fp
    )

    precision = (
        tp
        / (
            precision_denominator
            + epsilon
        )
        if precision_denominator > 0
        else (
            1.0
            if tp + fn == 0
            else 0.0
        )
    )

    # --------------------------------------------------------------
    # Recall
    #
    # TP / (TP + FN)
    # --------------------------------------------------------------

    recall_denominator = (
        tp + fn
    )

    recall = (
        tp
        / (
            recall_denominator
            + epsilon
        )
        if recall_denominator > 0
        else 1.0
    )

    # --------------------------------------------------------------
    # F1
    #
    # 2TP / (2TP + FP + FN)
    # --------------------------------------------------------------

    f1_denominator = (
        2.0 * tp
        + fp
        + fn
    )

    f1 = (
        2.0
        * tp
        / (
            f1_denominator
            + epsilon
        )
        if f1_denominator > 0
        else 1.0
    )

    # --------------------------------------------------------------
    # Pixel accuracy
    # --------------------------------------------------------------

    total = (
        tp
        + fp
        + fn
        + tn
    )

    pixel_accuracy = (
        (
            tp + tn
        )
        / (
            total + epsilon
        )
        if total > 0
        else 1.0
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

        "pixel_accuracy":
            float(
                pixel_accuracy
            ),
    }


# ---------------------------------------------------------------------
# Single-call metric calculation
# ---------------------------------------------------------------------

@torch.no_grad()
def binary_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Dict[str, float]:
    """
    Calculate all common binary Road segmentation metrics.

    This is convenient for evaluating one batch or one full tensor.
    """

    counts = binary_confusion_counts(
        prediction=prediction,
        target=target,
        threshold=threshold,
        from_logits=from_logits,
    )

    metrics = metrics_from_counts(
        counts
    )

    metrics.update(
        counts.to_dict()
    )

    return metrics


# ---------------------------------------------------------------------
# Running dataset-level metrics
# ---------------------------------------------------------------------

class RunningBinarySegmentationMetrics:
    """
    Accumulate confusion counts over a complete dataset.

    Why accumulate counts?
    ----------------------
    Validation IoU should preferably be calculated from all pixels in
    the complete validation dataset rather than averaging independent
    IoUs from batches of different foreground content.

    Example
    -------
        meter = RunningBinarySegmentationMetrics(
            threshold=0.5,
            from_logits=True
        )

        for images, masks in loader:
            logits = model(images)

            meter.update(
                logits,
                masks
            )

        results = meter.compute()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        from_logits: bool = True,
    ):
        self.threshold = _validate_threshold(
            threshold
        )

        self.from_logits = bool(
            from_logits
        )

        self.reset()


    def reset(
        self,
    ) -> None:
        """
        Reset all accumulated statistics.
        """
        self.counts = BinaryConfusionCounts()

        self.batch_count = 0


    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Add one validation/training batch.
        """

        batch_counts = binary_confusion_counts(
            prediction=prediction,
            target=target,
            threshold=self.threshold,
            from_logits=self.from_logits,
        )

        self.counts.update(
            batch_counts
        )

        self.batch_count += 1


    def compute(
        self,
    ) -> Dict[str, float]:
        """
        Calculate metrics from all accumulated pixels.
        """

        metrics = metrics_from_counts(
            self.counts
        )

        metrics.update(
            self.counts.to_dict()
        )

        metrics[
            "batch_count"
        ] = int(
            self.batch_count
        )

        metrics[
            "threshold"
        ] = float(
            self.threshold
        )

        return metrics


# ---------------------------------------------------------------------
# Percentage formatting
# ---------------------------------------------------------------------

def metrics_to_percent(
    metrics: Dict[str, float],
    decimals: int = 4,
) -> Dict[str, float]:
    """
    Convert normalized metrics into percentages.

    Example
    -------
    0.514068
        ->
    51.4068
    """

    metric_names = {
        "iou",
        "precision",
        "recall",
        "f1",
        "pixel_accuracy",
    }

    result = dict(
        metrics
    )

    for metric_name in metric_names:

        if metric_name not in result:
            continue

        result[
            f"{metric_name}_percent"
        ] = round(
            float(
                result[
                    metric_name
                ]
            )
            * 100.0,
            int(
                decimals
            ),
        )

    return result
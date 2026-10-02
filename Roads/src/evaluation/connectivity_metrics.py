"""
Road-connectivity evaluation metrics.

Approach 2 - Road Extraction
----------------------------

Primary segmentation metrics such as IoU, Precision, Recall, and F1 are
implemented in:

    src/evaluation/metrics.py

This module adds topology/connectivity-aware evaluation through clDice.

Why clDice?
-----------
Two Road predictions may have similar IoU values while having very different
network continuity.

Example:

    Prediction A:
        good road area
        several broken road connections

    Prediction B:
        similar road area
        much better continuity

IoU may score them similarly.

clDice gives additional information about how well the predicted Road network
preserves the topology / centerline structure of the Ground Truth.

Important
---------
clDice is a SECONDARY Road-specific metric.

The primary model-selection metric remains:

    validation_iou

unless explicitly changed in configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_threshold(
    threshold: float,
) -> float:
    """
    Validate probability threshold.
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


def _ensure_4d(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a segmentation tensor to:

        [B, 1, H, W]

    Accepted forms
    --------------
    H x W
    B x H x W
    B x 1 x H x W
    """

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        tensor = torch.as_tensor(
            tensor
        )

    if tensor.ndim == 2:

        tensor = tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

    elif tensor.ndim == 3:

        tensor = tensor.unsqueeze(
            1
        )

    elif tensor.ndim == 4:

        if tensor.shape[1] != 1:
            raise ValueError(
                "Connectivity metrics currently expect one "
                "binary Road channel.\n"
                f"Received shape: {tuple(tensor.shape)}"
            )

    else:

        raise ValueError(
            "Expected Road mask with 2, 3, or 4 dimensions.\n"
            f"Received shape: {tuple(tensor.shape)}"
        )

    return tensor.float()


# ---------------------------------------------------------------------
# Prediction / target preparation
# ---------------------------------------------------------------------

def prepare_binary_prediction(
    prediction: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Convert model output to a binary Road mask.

    Returns
    -------
    torch.Tensor
        Shape:

            [B, 1, H, W]

        Values:

            0.0 or 1.0
    """

    threshold = _validate_threshold(
        threshold
    )

    prediction = _ensure_4d(
        prediction
    )

    if from_logits:

        prediction = torch.sigmoid(
            prediction
        )

    return (
        prediction
        >= threshold
    ).float()


def prepare_binary_target(
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Convert Ground Truth to binary [B, 1, H, W].
    """

    target = _ensure_4d(
        target
    )

    return (
        target > 0.5
    ).float()


# ---------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------

def soft_erode(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable approximation of binary erosion.

    For 2D Road masks, erosion is approximated using max pooling
    over inverted / negative values.
    """

    image = _ensure_4d(
        image
    )

    vertical = -F.max_pool2d(
        -image,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )

    horizontal = -F.max_pool2d(
        -image,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )

    return torch.minimum(
        vertical,
        horizontal,
    )


def soft_dilate(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable approximation of binary dilation.
    """

    image = _ensure_4d(
        image
    )

    return F.max_pool2d(
        image,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def soft_open(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Morphological opening:

        erosion
        ↓
        dilation
    """

    return soft_dilate(
        soft_erode(
            image
        )
    )


# ---------------------------------------------------------------------
# Soft skeletonization
# ---------------------------------------------------------------------

def soft_skeletonize(
    image: torch.Tensor,
    iterations: int = 50,
) -> torch.Tensor:
    """
    Approximate the centerline / skeleton of a binary Road mask.

    Parameters
    ----------
    image:
        Binary or continuous mask:

            [B, 1, H, W]

    iterations:
        Maximum number of iterative thinning steps.

    Returns
    -------
    torch.Tensor
        Approximate Road skeleton.

    Notes
    -----
    This avoids requiring OpenCV or scikit-image and can run directly
    on CPU or GPU through PyTorch.
    """

    image = _ensure_4d(
        image
    )

    iterations = int(
        iterations
    )

    if iterations < 1:
        raise ValueError(
            "Skeletonization iterations must be >= 1."
        )

    image = torch.clamp(
        image,
        0.0,
        1.0,
    )

    opened = soft_open(
        image
    )

    skeleton = F.relu(
        image - opened
    )

    working = image

    for _ in range(
        iterations
    ):

        working = soft_erode(
            working
        )

        opened = soft_open(
            working
        )

        delta = F.relu(
            working - opened
        )

        skeleton = (
            skeleton
            + F.relu(
                delta
                - skeleton
                * delta
            )
        )

    return torch.clamp(
        skeleton,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------
# clDice statistics
# ---------------------------------------------------------------------

@dataclass
class ClDiceStatistics:
    """
    Sufficient statistics for global clDice calculation.

    pred_skeleton_on_target:
        Predicted skeleton pixels supported by Ground Truth.

    pred_skeleton_total:
        Total predicted skeleton content.

    target_skeleton_on_prediction:
        Ground-Truth skeleton supported by the prediction.

    target_skeleton_total:
        Total Ground-Truth skeleton content.
    """

    pred_skeleton_on_target: float = 0.0

    pred_skeleton_total: float = 0.0

    target_skeleton_on_prediction: float = 0.0

    target_skeleton_total: float = 0.0


    def update(
        self,
        other: "ClDiceStatistics",
    ) -> None:

        self.pred_skeleton_on_target += float(
            other.pred_skeleton_on_target
        )

        self.pred_skeleton_total += float(
            other.pred_skeleton_total
        )

        self.target_skeleton_on_prediction += float(
            other.target_skeleton_on_prediction
        )

        self.target_skeleton_total += float(
            other.target_skeleton_total
        )


    def to_dict(
        self,
    ) -> Dict[str, float]:

        return {
            "pred_skeleton_on_target":
                float(
                    self.pred_skeleton_on_target
                ),

            "pred_skeleton_total":
                float(
                    self.pred_skeleton_total
                ),

            "target_skeleton_on_prediction":
                float(
                    self.target_skeleton_on_prediction
                ),

            "target_skeleton_total":
                float(
                    self.target_skeleton_total
                ),
        }


# ---------------------------------------------------------------------
# Calculate clDice statistics
# ---------------------------------------------------------------------

@torch.no_grad()
def cldice_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    skeleton_iterations: int = 50,
) -> ClDiceStatistics:
    """
    Calculate the topology statistics required for clDice.
    """

    prediction_binary = (
        prepare_binary_prediction(
            prediction=prediction,

            threshold=threshold,

            from_logits=from_logits,
        )
    )

    target_binary = (
        prepare_binary_target(
            target
        )
    )

    if (
        prediction_binary.shape
        != target_binary.shape
    ):

        raise ValueError(
            "Prediction/target shape mismatch for clDice.\n"
            f"Prediction: "
            f"{tuple(prediction_binary.shape)}\n"
            f"Target: "
            f"{tuple(target_binary.shape)}"
        )

    prediction_skeleton = (
        soft_skeletonize(
            prediction_binary,

            iterations=skeleton_iterations,
        )
    )

    target_skeleton = (
        soft_skeletonize(
            target_binary,

            iterations=skeleton_iterations,
        )
    )

    pred_skeleton_on_target = torch.sum(
        prediction_skeleton
        * target_binary
    )

    pred_skeleton_total = torch.sum(
        prediction_skeleton
    )

    target_skeleton_on_prediction = torch.sum(
        target_skeleton
        * prediction_binary
    )

    target_skeleton_total = torch.sum(
        target_skeleton
    )

    return ClDiceStatistics(
        pred_skeleton_on_target=float(
            pred_skeleton_on_target.item()
        ),

        pred_skeleton_total=float(
            pred_skeleton_total.item()
        ),

        target_skeleton_on_prediction=float(
            target_skeleton_on_prediction.item()
        ),

        target_skeleton_total=float(
            target_skeleton_total.item()
        ),
    )


# ---------------------------------------------------------------------
# Metrics from statistics
# ---------------------------------------------------------------------

def cldice_from_statistics(
    statistics: ClDiceStatistics,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculate topology precision, topology sensitivity, and clDice.

    Definitions
    -----------
    Topology Precision:

        predicted skeleton overlapping Ground Truth
        -------------------------------------------
             total predicted skeleton

    Topology Sensitivity:

        Ground-Truth skeleton covered by prediction
        -------------------------------------------
             total Ground-Truth skeleton

    clDice:

        2 * topology_precision * topology_sensitivity
        ------------------------------------------------
        topology_precision + topology_sensitivity
    """

    epsilon = float(
        epsilon
    )

    prediction_total = float(
        statistics.pred_skeleton_total
    )

    target_total = float(
        statistics.target_skeleton_total
    )

    # --------------------------------------------------------------
    # Topology precision
    # --------------------------------------------------------------

    if prediction_total > 0:

        topology_precision = (
            statistics.pred_skeleton_on_target
            / (
                prediction_total
                + epsilon
            )
        )

    else:

        # Empty prediction and empty target represent perfect agreement.
        topology_precision = (
            1.0
            if target_total <= 0
            else 0.0
        )

    # --------------------------------------------------------------
    # Topology sensitivity
    # --------------------------------------------------------------

    if target_total > 0:

        topology_sensitivity = (
            statistics.target_skeleton_on_prediction
            / (
                target_total
                + epsilon
            )
        )

    else:

        topology_sensitivity = (
            1.0
            if prediction_total <= 0
            else 0.0
        )

    # --------------------------------------------------------------
    # clDice
    # --------------------------------------------------------------

    denominator = (
        topology_precision
        + topology_sensitivity
    )

    if denominator > 0:

        cldice = (
            2.0
            * topology_precision
            * topology_sensitivity
            / (
                denominator
                + epsilon
            )
        )

    else:

        cldice = 0.0

    return {
        "cldice":
            float(
                cldice
            ),

        "topology_precision":
            float(
                topology_precision
            ),

        "topology_sensitivity":
            float(
                topology_sensitivity
            ),
    }


# ---------------------------------------------------------------------
# Single-call calculation
# ---------------------------------------------------------------------

@torch.no_grad()
def cldice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    skeleton_iterations: int = 50,
) -> Dict[str, float]:
    """
    Calculate clDice and its two topology components.
    """

    statistics = cldice_statistics(
        prediction=prediction,

        target=target,

        threshold=threshold,

        from_logits=from_logits,

        skeleton_iterations=skeleton_iterations,
    )

    results = cldice_from_statistics(
        statistics
    )

    results.update(
        statistics.to_dict()
    )

    results[
        "threshold"
    ] = float(
        threshold
    )

    results[
        "skeleton_iterations"
    ] = int(
        skeleton_iterations
    )

    return results


# ---------------------------------------------------------------------
# Running dataset-level clDice
# ---------------------------------------------------------------------

class RunningClDice:
    """
    Accumulate Road-connectivity statistics across an entire dataset.

    This is preferable to simply averaging clDice from individual
    batches because batches may contain different amounts of Road
    network structure.

    Example
    -------
        meter = RunningClDice(
            threshold=0.5
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
        skeleton_iterations: int = 50,
    ):

        self.threshold = _validate_threshold(
            threshold
        )

        self.from_logits = bool(
            from_logits
        )

        self.skeleton_iterations = int(
            skeleton_iterations
        )

        if self.skeleton_iterations < 1:

            raise ValueError(
                "skeleton_iterations must be >= 1."
            )

        self.reset()


    def reset(
        self,
    ) -> None:

        self.statistics = (
            ClDiceStatistics()
        )

        self.batch_count = 0


    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:

        batch_statistics = (
            cldice_statistics(
                prediction=prediction,

                target=target,

                threshold=self.threshold,

                from_logits=self.from_logits,

                skeleton_iterations=(
                    self.skeleton_iterations
                ),
            )
        )

        self.statistics.update(
            batch_statistics
        )

        self.batch_count += 1


    def compute(
        self,
    ) -> Dict[str, float]:

        results = (
            cldice_from_statistics(
                self.statistics
            )
        )

        results.update(
            self.statistics.to_dict()
        )

        results[
            "threshold"
        ] = float(
            self.threshold
        )

        results[
            "skeleton_iterations"
        ] = int(
            self.skeleton_iterations
        )

        results[
            "batch_count"
        ] = int(
            self.batch_count
        )

        return results
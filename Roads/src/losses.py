"""
Loss functions for Approach 2 - Road Extraction.

Default baseline
----------------
BCE + Dice

Optional Road-specific losses
-----------------------------
- Focal Loss
- Tversky Loss
- Soft clDice Loss

Important
---------
The baseline Road experiments use:

    BCE weight  = 0.5
    Dice weight = 0.5

The optional losses are implemented here so they can be activated later
without changing the training engine.

All losses expect RAW MODEL LOGITS:

    [B, 1, H, W]

Ground Truth:

    [B, 1, H, W]
    values {0, 1}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.connectivity_metrics import (
    soft_skeletonize,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _validate_shapes(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Validate prediction and Ground-Truth dimensions.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "Loss input shape mismatch.\n"
            f"Logits : {tuple(logits.shape)}\n"
            f"Targets: {tuple(targets.shape)}"
        )

    if logits.ndim != 4:
        raise ValueError(
            "Road segmentation losses expect tensors shaped:\n"
            "[B, 1, H, W]\n"
            f"Received: {tuple(logits.shape)}"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "Binary Road segmentation expects one output channel.\n"
            f"Received channels: {logits.shape[1]}"
        )


def _prepare_targets(
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Convert Ground Truth to float binary representation.
    """

    return (
        targets > 0.5
    ).float()


# ---------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary Road segmentation.

    Dice coefficient:

        2 * intersection
        ----------------------------
        prediction + target

    Loss:

        1 - Dice
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.smooth = float(
            smooth
        )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        _validate_shapes(
            logits,
            targets,
        )

        targets = _prepare_targets(
            targets
        )

        probabilities = torch.sigmoid(
            logits
        )

        # Calculate per-sample Dice rather than mixing the entire
        # batch into one mask.
        dimensions = tuple(
            range(
                1,
                probabilities.ndim,
            )
        )

        intersection = torch.sum(
            probabilities * targets,
            dim=dimensions,
        )

        prediction_sum = torch.sum(
            probabilities,
            dim=dimensions,
        )

        target_sum = torch.sum(
            targets,
            dim=dimensions,
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            prediction_sum
            + target_sum
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ---------------------------------------------------------------------
# BCE + Dice
# ---------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """
    Weighted combination of:

        BCEWithLogitsLoss
        +
        DiceLoss

    Default:

        0.5 BCE
        0.5 Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()

        self.bce_weight = float(
            bce_weight
        )

        self.dice_weight = float(
            dice_weight
        )

        if self.bce_weight < 0:
            raise ValueError(
                "bce_weight cannot be negative."
            )

        if self.dice_weight < 0:
            raise ValueError(
                "dice_weight cannot be negative."
            )

        if (
            self.bce_weight
            + self.dice_weight
            <= 0
        ):
            raise ValueError(
                "At least one BCE/Dice weight must be greater than zero."
            )

        self.dice_loss = DiceLoss(
            smooth=smooth
        )

        if pos_weight is None:

            self.register_buffer(
                "_pos_weight",
                None,
            )

        else:

            self.register_buffer(
                "_pos_weight",
                torch.tensor(
                    [float(pos_weight)],
                    dtype=torch.float32,
                ),
            )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        _validate_shapes(
            logits,
            targets,
        )

        targets = _prepare_targets(
            targets
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self._pos_weight,
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        total = (
            self.bce_weight * bce
            +
            self.dice_weight * dice
        )

        return total


# ---------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """
    Optional binary Focal Loss.

    Useful when easy background pixels dominate training.

    This is NOT enabled in the baseline.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.alpha = float(
            alpha
        )

        self.gamma = float(
            gamma
        )

        if not (
            0.0
            <= self.alpha
            <= 1.0
        ):
            raise ValueError(
                "Focal alpha must be between 0 and 1."
            )

        if self.gamma < 0:
            raise ValueError(
                "Focal gamma cannot be negative."
            )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        _validate_shapes(
            logits,
            targets,
        )

        targets = _prepare_targets(
            targets
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probabilities = torch.sigmoid(
            logits
        )

        p_t = (
            probabilities * targets
            +
            (1.0 - probabilities)
            * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            +
            (1.0 - self.alpha)
            * (1.0 - targets)
        )

        focal_weight = (
            alpha_t
            * torch.pow(
                1.0 - p_t,
                self.gamma,
            )
        )

        loss = (
            focal_weight * bce
        )

        return loss.mean()


# ---------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Optional Tversky Loss.

    Tversky can control the relative penalty for:

        False Positives
        False Negatives

    This may later be useful when missing Road segments is considered
    more costly than small over-predictions.

    Baseline:
        disabled
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.alpha = float(
            alpha
        )

        self.beta = float(
            beta
        )

        self.smooth = float(
            smooth
        )

        if self.alpha < 0:
            raise ValueError(
                "Tversky alpha cannot be negative."
            )

        if self.beta < 0:
            raise ValueError(
                "Tversky beta cannot be negative."
            )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        _validate_shapes(
            logits,
            targets,
        )

        targets = _prepare_targets(
            targets
        )

        probabilities = torch.sigmoid(
            logits
        )

        dimensions = tuple(
            range(
                1,
                probabilities.ndim,
            )
        )

        true_positive = torch.sum(
            probabilities * targets,
            dim=dimensions,
        )

        false_positive = torch.sum(
            probabilities
            * (1.0 - targets),
            dim=dimensions,
        )

        false_negative = torch.sum(
            (1.0 - probabilities)
            * targets,
            dim=dimensions,
        )

        tversky = (
            true_positive
            + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return (
            1.0 - tversky
        ).mean()


# ---------------------------------------------------------------------
# Soft clDice Loss
# ---------------------------------------------------------------------

class SoftClDiceLoss(nn.Module):
    """
    Optional differentiable clDice-inspired connectivity loss.

    Unlike the evaluation clDice metric, this implementation does NOT
    threshold probabilities. It operates on continuous sigmoid outputs
    so gradients can propagate during training.

    Baseline:
        disabled
    """

    def __init__(
        self,
        iterations: int = 20,
        smooth: float = 1e-6,
    ):
        super().__init__()

        self.iterations = int(
            iterations
        )

        self.smooth = float(
            smooth
        )

        if self.iterations < 1:
            raise ValueError(
                "Soft clDice iterations must be >= 1."
            )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        _validate_shapes(
            logits,
            targets,
        )

        targets = _prepare_targets(
            targets
        )

        probabilities = torch.sigmoid(
            logits
        )

        prediction_skeleton = (
            soft_skeletonize(
                probabilities,
                iterations=self.iterations,
            )
        )

        target_skeleton = (
            soft_skeletonize(
                targets,
                iterations=self.iterations,
            )
        )

        dimensions = tuple(
            range(
                1,
                logits.ndim,
            )
        )

        # Topology precision:
        # predicted centerline supported by GT Road area.
        topology_precision = (
            torch.sum(
                prediction_skeleton
                * targets,
                dim=dimensions,
            )
            + self.smooth
        ) / (
            torch.sum(
                prediction_skeleton,
                dim=dimensions,
            )
            + self.smooth
        )

        # Topology sensitivity:
        # GT centerline covered by predicted Road area.
        topology_sensitivity = (
            torch.sum(
                target_skeleton
                * probabilities,
                dim=dimensions,
            )
            + self.smooth
        ) / (
            torch.sum(
                target_skeleton,
                dim=dimensions,
            )
            + self.smooth
        )

        cldice = (
            2.0
            * topology_precision
            * topology_sensitivity
            + self.smooth
        ) / (
            topology_precision
            + topology_sensitivity
            + self.smooth
        )

        return (
            1.0 - cldice
        ).mean()


# ---------------------------------------------------------------------
# Complete Road Loss
# ---------------------------------------------------------------------

class RoadLoss(nn.Module):
    """
    Configurable Road segmentation loss.

    Core baseline:
        BCE + Dice

    Optional additions:
        Focal
        Tversky
        Soft clDice

    Disabled optional components contribute exactly zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,

        focal_enabled: bool = False,
        focal_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,

        tversky_enabled: bool = False,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,

        cldice_enabled: bool = False,
        cldice_weight: float = 0.0,
        cldice_iterations: int = 20,

        pos_weight: Optional[float] = None,
    ):
        super().__init__()

        self.bce_dice = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
            pos_weight=pos_weight,
        )

        self.focal_enabled = bool(
            focal_enabled
        )

        self.focal_weight = float(
            focal_weight
        )

        self.tversky_enabled = bool(
            tversky_enabled
        )

        self.tversky_weight = float(
            tversky_weight
        )

        self.cldice_enabled = bool(
            cldice_enabled
        )

        self.cldice_weight = float(
            cldice_weight
        )

        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
        )

        self.cldice_loss = SoftClDiceLoss(
            iterations=cldice_iterations,
        )


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        total_loss = self.bce_dice(
            logits,
            targets,
        )

        if (
            self.focal_enabled
            and self.focal_weight > 0
        ):

            total_loss = (
                total_loss
                + self.focal_weight
                * self.focal_loss(
                    logits,
                    targets,
                )
            )

        if (
            self.tversky_enabled
            and self.tversky_weight > 0
        ):

            total_loss = (
                total_loss
                + self.tversky_weight
                * self.tversky_loss(
                    logits,
                    targets,
                )
            )

        if (
            self.cldice_enabled
            and self.cldice_weight > 0
        ):

            total_loss = (
                total_loss
                + self.cldice_weight
                * self.cldice_loss(
                    logits,
                    targets,
                )
            )

        return total_loss


# ---------------------------------------------------------------------
# Configuration factory
# ---------------------------------------------------------------------

def build_loss(
    general_params: Dict[str, Any],
    model_config: Optional[
        Dict[str, Any]
    ] = None,
) -> nn.Module:
    """
    Build the Road loss from configuration.

    Parameters
    ----------
    general_params:
        Roads/config/general_params.json

    model_config:
        Optional model-specific configuration.

        Example:
            Roads/config/models/unet.json

        Model-specific loss values override general defaults.

    Returns
    -------
    torch.nn.Module
        Configured Road loss.
    """

    general_loss = dict(
        general_params.get(
            "loss",
            {}
        )
    )

    model_loss = {}

    if model_config is not None:

        model_loss = dict(
            model_config.get(
                "loss",
                {}
            )
        )

    # Model configuration overrides common defaults.
    loss_config = {
        **general_loss,
        **{
            key: value
            for key, value
            in model_loss.items()
            if key
            not in {
                "inherit_general_params"
            }
        },
    }

    loss_name = str(
        loss_config.get(
            "name",
            "bce_dice",
        )
    ).lower()

    if loss_name not in {
        "bce_dice",
        "road_loss",
    }:
        raise ValueError(
            f"Unsupported Road loss: {loss_name!r}"
        )

    road_specific = dict(
        general_loss.get(
            "road_specific_losses",
            {}
        )
    )

    # Allow model-level overrides later if needed.
    model_road_specific = model_loss.get(
        "road_specific_losses",
        {}
    )

    if isinstance(
        model_road_specific,
        dict,
    ):
        road_specific.update(
            model_road_specific
        )

    loss = RoadLoss(
        bce_weight=float(
            loss_config.get(
                "bce_weight",
                0.5,
            )
        ),

        dice_weight=float(
            loss_config.get(
                "dice_weight",
                0.5,
            )
        ),

        smooth=float(
            loss_config.get(
                "smooth",
                1.0,
            )
        ),

        focal_enabled=bool(
            road_specific.get(
                "enable_focal",
                False,
            )
        ),

        focal_weight=float(
            road_specific.get(
                "focal_weight",
                0.0,
            )
        ),

        tversky_enabled=bool(
            road_specific.get(
                "enable_tversky",
                False,
            )
        ),

        tversky_weight=float(
            road_specific.get(
                "tversky_weight",
                0.0,
            )
        ),

        cldice_enabled=bool(
            road_specific.get(
                "enable_cldice",
                False,
            )
        ),

        cldice_weight=float(
            road_specific.get(
                "cldice_weight",
                0.0,
            )
        ),

        pos_weight=(
            float(
                loss_config[
                    "pos_weight"
                ]
            )
            if loss_config.get(
                "pos_weight"
            ) is not None
            else None
        ),
    )

    return loss
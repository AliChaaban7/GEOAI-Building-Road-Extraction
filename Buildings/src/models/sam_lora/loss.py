"""
Buildings/src/models/sam_lora/loss.py

Loss functions for SAM-LoRA building segmentation.

The default training objective combines:

    Binary Cross Entropy with Logits
    +
    Soft Dice Loss

The SAM-LoRA model returns RAW logits:

    [B, 1, H, W]

Ground-truth masks are expected as:

    [B, 1, H, W]

with binary values:

    0 = background
    1 = building

No sigmoid should be applied before BCEWithLogitsLoss.

The Dice component internally converts logits to probabilities using
sigmoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------


@dataclass
class SAMLoRALossResult:
    """
    Container for SAM-LoRA loss values.

    Attributes
    ----------
    total:
        Final weighted training loss.

    bce:
        BCEWithLogits loss.

    dice:
        Soft Dice loss.
    """

    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor

    def to_dict(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Return all losses in dictionary form.
        """

        return {
            "loss": self.total,
            "bce_loss": self.bce,
            "dice_loss": self.dice,
        }


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Validate prediction and ground-truth tensors.
    """

    if not torch.is_tensor(logits):
        raise TypeError(
            "SAM-LoRA loss expects logits to be a torch.Tensor, "
            f"received {type(logits).__name__}."
        )

    if not torch.is_tensor(targets):
        raise TypeError(
            "SAM-LoRA loss expects targets to be a torch.Tensor, "
            f"received {type(targets).__name__}."
        )

    if logits.ndim != 4:
        raise ValueError(
            "SAM-LoRA logits must have shape [B, 1, H, W]. "
            f"Received shape: {tuple(logits.shape)}."
        )

    if targets.ndim not in {
        3,
        4,
    }:
        raise ValueError(
            "SAM-LoRA targets must have shape "
            "[B, H, W] or [B, 1, H, W]. "
            f"Received shape: {tuple(targets.shape)}."
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "SAM-LoRA expects one output channel for binary "
            "building segmentation. "
            f"Received C={logits.shape[1]}."
        )

    if not torch.is_floating_point(logits):
        raise TypeError(
            "SAM-LoRA logits must use floating-point dtype. "
            f"Received dtype={logits.dtype}."
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "SAM-LoRA logits contain NaN or infinite values."
        )

    if not torch.isfinite(targets).all():
        raise ValueError(
            "SAM-LoRA targets contain NaN or infinite values."
        )


def _prepare_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize ground-truth masks to match logits.

    Accepted target shapes:

        [B, H, W]
        [B, 1, H, W]

    Returned shape:

        [B, 1, H, W]
    """

    _validate_logits_and_targets(
        logits=logits,
        targets=targets,
    )

    if targets.ndim == 3:
        targets = targets.unsqueeze(
            dim=1
        )

    if targets.shape != logits.shape:
        raise ValueError(
            "SAM-LoRA prediction and target shapes do not match.\n"
            f"Logits : {tuple(logits.shape)}\n"
            f"Targets: {tuple(targets.shape)}"
        )

    targets = targets.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    minimum = float(
        targets.detach().min().item()
    )

    maximum = float(
        targets.detach().max().item()
    )

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "SAM-LoRA ground-truth masks must contain values "
            "between 0 and 1.\n"
            f"Observed range: [{minimum}, {maximum}]"
        )

    return targets


# ---------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------


class SoftDiceLoss(nn.Module):
    """
    Differentiable Dice loss for binary segmentation.

    The model provides raw logits.

    Internally:

        probabilities = sigmoid(logits)

    Dice coefficient:

                   2 * intersection + smooth
        Dice = -------------------------------------
                prediction + target + smooth

    Dice loss:

        1 - Dice

    Parameters
    ----------
    smooth:
        Numerical smoothing constant.

    eps:
        Small value used for numerical stability.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if smooth < 0.0:
            raise ValueError(
                "Dice smooth must be >= 0."
            )

        if eps <= 0.0:
            raise ValueError(
                "Dice eps must be > 0."
            )

        self.smooth = float(
            smooth
        )

        self.eps = float(
            eps
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batch-averaged soft Dice loss.
        """

        targets = _prepare_targets(
            logits=logits,
            targets=targets,
        )

        probabilities = torch.sigmoid(
            logits
        )

        # --------------------------------------------------------------
        # Flatten each image independently.
        #
        # This prevents one large-building tile from completely
        # dominating the Dice score of the full batch.
        # --------------------------------------------------------------

        probabilities = probabilities.flatten(
            start_dim=1
        )

        targets = targets.flatten(
            start_dim=1
        )

        intersection = (
            probabilities * targets
        ).sum(
            dim=1
        )

        probability_sum = probabilities.sum(
            dim=1
        )

        target_sum = targets.sum(
            dim=1
        )

        numerator = (
            2.0 * intersection
            + self.smooth
        )

        denominator = (
            probability_sum
            + target_sum
            + self.smooth
        )

        dice_score = (
            numerator
            / denominator.clamp_min(
                self.eps
            )
        )

        dice_loss = (
            1.0
            - dice_score
        )

        return dice_loss.mean()


# ---------------------------------------------------------------------
# Combined BCE + Dice
# ---------------------------------------------------------------------


class SAMLoRALoss(nn.Module):
    """
    Default SAM-LoRA building-segmentation training objective.

    Total loss:

        L =
            BCE_weight  * BCEWithLogits
            +
            Dice_weight * DiceLoss

    Parameters
    ----------
    bce_weight:
        Weight assigned to BCE.

    dice_weight:
        Weight assigned to Dice.

    pos_weight:
        Optional positive-class weight used by BCEWithLogitsLoss.

        This can later help if building pixels are strongly
        underrepresented compared with background pixels.

        Leave as None for the baseline experiment.

    dice_smooth:
        Smoothing constant used by SoftDiceLoss.

    normalize_weights:
        If True:

            BCE_weight + Dice_weight = 1

        internally.

        Example:

            bce_weight = 1
            dice_weight = 1

        becomes:

            BCE  = 0.5
            Dice = 0.5
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        dice_smooth: float = 1.0,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()

        if bce_weight < 0.0:
            raise ValueError(
                "bce_weight must be >= 0."
            )

        if dice_weight < 0.0:
            raise ValueError(
                "dice_weight must be >= 0."
            )

        if (
            bce_weight == 0.0
            and dice_weight == 0.0
        ):
            raise ValueError(
                "At least one loss component must have "
                "a weight greater than zero."
            )

        if (
            pos_weight is not None
            and pos_weight <= 0.0
        ):
            raise ValueError(
                "pos_weight must be > 0 when provided."
            )

        self.normalize_weights = bool(
            normalize_weights
        )

        if self.normalize_weights:
            weight_sum = (
                float(bce_weight)
                + float(dice_weight)
            )

            self.bce_weight = (
                float(bce_weight)
                / weight_sum
            )

            self.dice_weight = (
                float(dice_weight)
                / weight_sum
            )

        else:
            self.bce_weight = float(
                bce_weight
            )

            self.dice_weight = float(
                dice_weight
            )

        # --------------------------------------------------------------
        # Optional class weighting.
        #
        # register_buffer() means the tensor automatically follows the
        # loss module when moving between CPU and GPU.
        # --------------------------------------------------------------

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

        self.dice_loss = SoftDiceLoss(
            smooth=dice_smooth
        )

    @property
    def pos_weight(
        self,
    ) -> Optional[torch.Tensor]:
        """
        Return configured BCE positive-class weight.
        """

        return self._pos_weight

    def _bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCEWithLogits dynamically.

        We call torch.nn.functional through PyTorch's loss module logic
        so pos_weight remains on the correct device.
        """

        targets = _prepare_targets(
            logits=logits,
            targets=targets,
        )

        pos_weight = self.pos_weight

        if pos_weight is not None:
            pos_weight = pos_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )

        return nn.functional.binary_cross_entropy_with_logits(
            input=logits,
            target=targets,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def compute_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> SAMLoRALossResult:
        """
        Compute total loss together with individual components.
        """

        targets = _prepare_targets(
            logits=logits,
            targets=targets,
        )

        if self.bce_weight > 0.0:
            bce = self._bce(
                logits=logits,
                targets=targets,
            )

        else:
            bce = logits.new_zeros(
                ()
            )

        if self.dice_weight > 0.0:
            dice = self.dice_loss(
                logits=logits,
                targets=targets,
            )

        else:
            dice = logits.new_zeros(
                ()
            )

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        if not torch.isfinite(
            total
        ):
            raise FloatingPointError(
                "SAM-LoRA loss became NaN or infinite.\n"
                f"BCE loss : {float(bce.detach().item()):.6f}\n"
                f"Dice loss: {float(dice.detach().item()):.6f}"
            )

        return SAMLoRALossResult(
            total=total,
            bce=bce,
            dice=dice,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return only the final scalar loss.

        This standard interface allows:

            loss = criterion(logits, masks)

            loss.backward()
        """

        result = self.compute_components(
            logits=logits,
            targets=targets,
        )

        return result.total


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def build_sam_lora_loss(
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: Optional[float] = None,
    dice_smooth: float = 1.0,
    normalize_weights: bool = True,
) -> SAMLoRALoss:
    """
    Build the default SAM-LoRA loss.

    Kept as a factory because later the shared training engine can
    construct this loss directly from JSON configuration.
    """

    return SAMLoRALoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        dice_smooth=dice_smooth,
        normalize_weights=normalize_weights,
    )
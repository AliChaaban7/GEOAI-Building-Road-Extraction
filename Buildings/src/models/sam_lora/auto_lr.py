"""
Buildings/src/models/sam_lora/auto_lr.py

Automatic learning-rate range test for SAM-LoRA.

Purpose
-------
The final SAM-LoRA learning rate is not guessed manually.

Instead:

    build temporary clean SAM-LoRA
        ↓
    start with a very small learning rate
        ↓
    progressively increase the learning rate
        ↓
    train for a short LR-range test
        ↓
    record smoothed BCE + Dice loss
        ↓
    detect unstable/diverging region
        ↓
    select a stable useful learning rate
        ↓
    discard ALL temporary model weights
        ↓
    return only the resolved learning rate
        ↓
    final training rebuilds a fresh SAM-LoRA model

IMPORTANT
---------
This utility must NEVER operate on the final model instance.

The caller provides a model_factory that constructs a new temporary
SAM-LoRA model.

The loss used here must be the SAME loss used during final training.

For the current thesis implementation:

    BCEWithLogits + Dice

This keeps Auto-LR and final training experimentally consistent.
"""

from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

BatchType = Union[
    Sequence[torch.Tensor],
    Mapping[str, torch.Tensor],
]

ModelFactory = Callable[
    [],
    nn.Module,
]

BatchExtractor = Callable[
    [object],
    Tuple[torch.Tensor, torch.Tensor],
]

OptimizerFactory = Callable[
    [Iterable[nn.Parameter], float],
    Optimizer,
]


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AutoLRResult:
    """
    Result of one SAM-LoRA LR range test.

    Attributes
    ----------
    suggested_lr:
        Learning rate selected for final training.

    steepest_lr:
        Learning rate at the strongest stable loss decrease.

    minimum_loss_lr:
        Learning rate where the lowest smoothed LR-test loss occurred.

    minimum_loss:
        Lowest observed smoothed loss.

    start_lr:
        Initial LR used by the range test.

    requested_end_lr:
        Maximum requested LR.

    actual_end_lr:
        Last LR actually tested.

        This can be smaller than requested_end_lr if divergence causes
        early stopping.

    requested_iterations:
        Maximum requested LR-finder iterations.

    completed_iterations:
        Number of successfully completed optimization steps.

    stopped_early:
        True when divergence was detected before all requested
        iterations were completed.

    stop_reason:
        Human-readable reason for stopping.

    learning_rates:
        LR used at every completed step.

    losses:
        Bias-corrected exponentially smoothed loss at every completed
        step.
    """

    suggested_lr: float

    steepest_lr: float
    minimum_loss_lr: float
    minimum_loss: float

    start_lr: float
    requested_end_lr: float
    actual_end_lr: float

    requested_iterations: int
    completed_iterations: int

    stopped_early: bool
    stop_reason: str

    learning_rates: List[float]
    losses: List[float]

    def to_dict(
        self,
    ) -> Dict[str, object]:
        """
        Convert result into JSON-serializable form.
        """

        return asdict(self)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_lr_range(
    start_lr: float,
    end_lr: float,
    num_iter: int,
) -> None:
    """
    Validate LR finder configuration.
    """

    if start_lr <= 0.0:
        raise ValueError(
            f"start_lr must be > 0, received {start_lr}."
        )

    if end_lr <= start_lr:
        raise ValueError(
            "end_lr must be greater than start_lr.\n"
            f"start_lr={start_lr}\n"
            f"end_lr={end_lr}"
        )

    if num_iter < 10:
        raise ValueError(
            "Auto-LR requires at least 10 iterations. "
            f"Received num_iter={num_iter}."
        )


def _validate_smoothing(
    beta: float,
) -> None:
    """
    Validate exponential loss-smoothing factor.
    """

    if beta < 0.0 or beta >= 1.0:
        raise ValueError(
            "beta must satisfy 0 <= beta < 1. "
            f"Received beta={beta}."
        )


def _validate_divergence_threshold(
    divergence_threshold: float,
) -> None:
    """
    Validate divergence threshold.
    """

    if divergence_threshold <= 1.0:
        raise ValueError(
            "divergence_threshold must be greater than 1. "
            f"Received {divergence_threshold}."
        )


# ---------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------


def default_batch_extractor(
    batch: object,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Extract images and masks from common DataLoader batch structures.

    Supported tuple/list format
    ---------------------------

        batch[0] = images
        batch[1] = masks

    Supported dictionary image keys
    -------------------------------

        images
        image
        x

    Supported dictionary mask keys
    ------------------------------

        masks
        mask
        targets
        target
        y

    A custom batch_extractor can be supplied later if the thesis
    dataset returns another structure.
    """

    # --------------------------------------------------------------
    # Tuple / list
    # --------------------------------------------------------------

    if isinstance(
        batch,
        (tuple, list),
    ):
        if len(batch) < 2:
            raise ValueError(
                "Tuple/list training batches must contain at least "
                "two elements: images and masks."
            )

        images = batch[0]
        masks = batch[1]

        if not torch.is_tensor(images):
            raise TypeError(
                "Batch element 0 must be an image tensor."
            )

        if not torch.is_tensor(masks):
            raise TypeError(
                "Batch element 1 must be a mask tensor."
            )

        return (
            images,
            masks,
        )

    # --------------------------------------------------------------
    # Dictionary / mapping
    # --------------------------------------------------------------

    if isinstance(
        batch,
        Mapping,
    ):
        image_keys = (
            "images",
            "image",
            "x",
        )

        mask_keys = (
            "masks",
            "mask",
            "targets",
            "target",
            "y",
        )

        images = None
        masks = None

        for key in image_keys:
            if key in batch:
                images = batch[key]
                break

        for key in mask_keys:
            if key in batch:
                masks = batch[key]
                break

        if images is None:
            raise KeyError(
                "Could not find image tensor in batch mapping.\n"
                f"Accepted image keys: {list(image_keys)}"
            )

        if masks is None:
            raise KeyError(
                "Could not find mask tensor in batch mapping.\n"
                f"Accepted mask keys: {list(mask_keys)}"
            )

        if not torch.is_tensor(images):
            raise TypeError(
                "Extracted images are not a torch.Tensor."
            )

        if not torch.is_tensor(masks):
            raise TypeError(
                "Extracted masks are not a torch.Tensor."
            )

        return (
            images,
            masks,
        )

    raise TypeError(
        "Unsupported DataLoader batch structure for SAM-LoRA "
        f"Auto-LR: {type(batch).__name__}.\n"
        "Provide a custom batch_extractor if required."
    )


# ---------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------


def _extract_logits(
    model_output: object,
) -> torch.Tensor:
    """
    Extract raw segmentation logits from the SAM-LoRA model.

    Supported forms:

        Tensor

    or:

        {
            "logits": Tensor,
            ...
        }
    """

    if torch.is_tensor(
        model_output
    ):
        return model_output

    if isinstance(
        model_output,
        Mapping,
    ):
        if "logits" not in model_output:
            raise KeyError(
                "SAM-LoRA model returned a mapping without "
                "a 'logits' entry."
            )

        logits = model_output["logits"]

        if not torch.is_tensor(
            logits
        ):
            raise TypeError(
                "SAM-LoRA model output['logits'] is not a tensor."
            )

        return logits

    raise TypeError(
        "Unsupported SAM-LoRA model output type: "
        f"{type(model_output).__name__}."
    )


# ---------------------------------------------------------------------
# Default optimizer
# ---------------------------------------------------------------------


def _build_default_optimizer(
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float],
    eps: float,
) -> Optimizer:
    """
    Build AdamW for the temporary Auto-LR experiment.
    """

    return AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=betas,
        eps=float(eps),
    )


# ---------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------


def _calculate_lr_multiplier(
    start_lr: float,
    end_lr: float,
    num_iter: int,
) -> float:
    """
    Calculate exponential LR increase multiplier.

    LR progression:

        lr_n = start_lr * multiplier^n

    so the final planned iteration approaches end_lr.
    """

    return (
        float(end_lr)
        / float(start_lr)
    ) ** (
        1.0
        / float(
            max(
                num_iter - 1,
                1,
            )
        )
    )


def _set_optimizer_lr(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """
    Set LR for all optimizer parameter groups.
    """

    for group in optimizer.param_groups:
        group["lr"] = float(
            learning_rate
        )


# ---------------------------------------------------------------------
# Loss smoothing
# ---------------------------------------------------------------------


def _update_smoothed_loss(
    raw_loss: float,
    previous_average: float,
    beta: float,
    step_index: int,
) -> Tuple[
    float,
    float,
]:
    """
    Exponentially smooth noisy minibatch losses.

    Uses bias correction similarly to Adam:

        avg = beta * avg + (1-beta) * loss

        corrected =
            avg / (1 - beta^step)

    Returns
    -------
    running_average
    corrected_smoothed_loss
    """

    running_average = (
        beta * previous_average
        + (1.0 - beta) * raw_loss
    )

    correction = (
        1.0
        - beta ** float(
            step_index + 1
        )
    )

    if correction <= 0.0:
        corrected = raw_loss
    else:
        corrected = (
            running_average
            / correction
        )

    return (
        float(running_average),
        float(corrected),
    )


# ---------------------------------------------------------------------
# LR selection
# ---------------------------------------------------------------------


def _select_learning_rate(
    learning_rates: Sequence[float],
    losses: Sequence[float],
    burn_in: int = 5,
    edge_fraction: float = 0.10,
    safety_factor: float = 1.0,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Select a stable LR from completed LR-finder measurements.

    Strategy
    --------
    1. Ignore the earliest warm-up measurements.
    2. Ignore the unstable extreme edges of the LR sweep.
    3. Find the minimum smoothed loss.
    4. Examine only the stable region up to the minimum loss.
    5. Compute:

           d(loss) / d(log10(LR))

    6. Select the LR where loss is decreasing most strongly.

    This is the "steepest descending slope" LR-finder heuristic.

    Parameters
    ----------
    safety_factor:
        Multiplier applied to the selected steepest LR.

        Default:
            1.0

        Values below 1.0 can later make the choice more conservative.

    Returns
    -------
    suggested_lr
    steepest_lr
    minimum_loss_lr
    minimum_loss
    """

    if len(
        learning_rates
    ) != len(
        losses
    ):
        raise ValueError(
            "Learning-rate and loss histories have different lengths."
        )

    number_of_points = len(
        learning_rates
    )

    if number_of_points < 8:
        raise RuntimeError(
            "Auto-LR produced too few valid measurements "
            f"({number_of_points})."
        )

    if safety_factor <= 0.0:
        raise ValueError(
            "safety_factor must be > 0."
        )

    # --------------------------------------------------------------
    # Remove initial warm-up.
    # --------------------------------------------------------------

    first_valid = min(
        max(
            int(burn_in),
            1,
        ),
        number_of_points - 3,
    )

    # --------------------------------------------------------------
    # Avoid unstable sweep edges.
    # --------------------------------------------------------------

    edge_points = max(
        1,
        int(
            round(
                number_of_points
                * float(edge_fraction)
            )
        ),
    )

    first_valid = max(
        first_valid,
        edge_points,
    )

    last_valid_exclusive = (
        number_of_points
        - edge_points
    )

    if (
        last_valid_exclusive
        - first_valid
        < 3
    ):
        first_valid = 1
        last_valid_exclusive = (
            number_of_points - 1
        )

    candidate_indices = list(
        range(
            first_valid,
            last_valid_exclusive,
        )
    )

    if len(candidate_indices) < 3:
        raise RuntimeError(
            "Not enough stable Auto-LR measurements to select "
            "a learning rate."
        )

    # --------------------------------------------------------------
    # Find lowest loss inside stable region.
    # --------------------------------------------------------------

    minimum_index = min(
        candidate_indices,
        key=lambda index: losses[index],
    )

    minimum_loss = float(
        losses[minimum_index]
    )

    minimum_loss_lr = float(
        learning_rates[minimum_index]
    )

    # --------------------------------------------------------------
    # Only search for steepest descent before or at the minimum.
    #
    # Once the loss has already reached its minimum, larger learning
    # rates are less attractive for stable final training.
    # --------------------------------------------------------------

    slope_end = max(
        minimum_index,
        first_valid + 1,
    )

    slope_candidates = list(
        range(
            first_valid,
            slope_end + 1,
        )
    )

    if len(slope_candidates) < 2:
        slope_candidates = (
            candidate_indices
        )

    log_lrs = [
        math.log10(
            float(lr)
        )
        for lr in learning_rates
    ]

    best_slope = None
    best_index = None

    # --------------------------------------------------------------
    # Central finite difference whenever possible.
    # --------------------------------------------------------------

    for index in slope_candidates:

        if (
            index <= 0
            or index >= number_of_points - 1
        ):
            continue

        delta_x = (
            log_lrs[index + 1]
            - log_lrs[index - 1]
        )

        if delta_x == 0.0:
            continue

        slope = (
            losses[index + 1]
            - losses[index - 1]
        ) / delta_x

        if not math.isfinite(
            slope
        ):
            continue

        if (
            best_slope is None
            or slope < best_slope
        ):
            best_slope = float(
                slope
            )

            best_index = int(
                index
            )

    if best_index is None:
        raise RuntimeError(
            "Auto-LR could not calculate a stable loss gradient."
        )

    steepest_lr = float(
        learning_rates[
            best_index
        ]
    )

    suggested_lr = (
        steepest_lr
        * float(safety_factor)
    )

    # --------------------------------------------------------------
    # Never suggest something outside the tested range.
    # --------------------------------------------------------------

    suggested_lr = max(
        float(learning_rates[0]),
        min(
            suggested_lr,
            float(learning_rates[-1]),
        ),
    )

    # --------------------------------------------------------------
    # Additional safety:
    #
    # Do not recommend a value larger than the LR associated with the
    # best stable observed loss.
    # --------------------------------------------------------------

    suggested_lr = min(
        suggested_lr,
        minimum_loss_lr,
    )

    return (
        float(suggested_lr),
        float(steepest_lr),
        float(minimum_loss_lr),
        float(minimum_loss),
    )


# ---------------------------------------------------------------------
# Main LR range test
# ---------------------------------------------------------------------


def run_sam_lora_lr_range_test(
    model_factory: ModelFactory,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: Union[
        str,
        torch.device,
    ] = "cuda",
    start_lr: float = 1e-7,
    end_lr: float = 1e-2,
    num_iter: int = 100,
    beta: float = 0.98,
    divergence_threshold: float = 4.0,
    weight_decay: float = 1e-4,
    betas: Tuple[
        float,
        float,
    ] = (
        0.9,
        0.999,
    ),
    eps: float = 1e-8,
    batch_extractor: Optional[
        BatchExtractor
    ] = None,
    optimizer_factory: Optional[
        OptimizerFactory
    ] = None,
    burn_in: int = 5,
    edge_fraction: float = 0.10,
    safety_factor: float = 1.0,
    verbose: bool = True,
) -> AutoLRResult:
    """
    Resolve a learning rate for SAM-LoRA using a temporary model.

    IMPORTANT
    ---------
    model_factory() MUST create a fresh SAM-LoRA instance.

    The model created here is destroyed after the LR range test.

    Therefore:

        temporary Auto-LR weights

    can never become:

        final training weights

    Parameters
    ----------
    model_factory:
        Function returning a newly initialized SAM-LoRA model.

        Example:

            def model_factory():
                sam, _ = build_sam_lora_model(...)
                return SAMLoRABuildingSegmenter(sam)

    train_loader:
        Training DataLoader.

    criterion:
        SAME criterion used for final SAM-LoRA training.

        Current thesis default:

            SAMLoRALoss(
                bce_weight=0.5,
                dice_weight=0.5,
            )

    device:
        Training device.

    start_lr:
        Small initial LR.

    end_lr:
        Maximum LR to explore.

    num_iter:
        Maximum number of LR-finder optimizer steps.

    beta:
        Loss-smoothing factor.

    divergence_threshold:
        Stop when:

            current_smoothed_loss
                >
            best_smoothed_loss * divergence_threshold

        after the warm-up region.

    weight_decay:
        AdamW weight decay.

        Later this must receive the same value as final training.

    optimizer_factory:
        Optional custom optimizer constructor.

        If omitted:

            AdamW

        is used.

    safety_factor:
        Optional conservative multiplier applied to the selected LR.

        Default:

            1.0

    Returns
    -------
    AutoLRResult
        Contains selected LR and complete LR-test history.
    """

    _validate_lr_range(
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
    )

    _validate_smoothing(
        beta=beta
    )

    _validate_divergence_threshold(
        divergence_threshold
    )

    if len(train_loader) <= 0:
        raise ValueError(
            "SAM-LoRA Auto-LR received an empty training DataLoader."
        )

    device = torch.device(
        device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "SAM-LoRA Auto-LR requested CUDA, but CUDA is not "
            "available."
        )

    if batch_extractor is None:
        batch_extractor = (
            default_batch_extractor
        )

    # --------------------------------------------------------------
    # Temporary objects.
    #
    # They are explicitly cleaned in finally{} regardless of success
    # or failure.
    # --------------------------------------------------------------

    temporary_model: Optional[
        nn.Module
    ] = None

    optimizer: Optional[
        Optimizer
    ] = None

    try:

        # ----------------------------------------------------------
        # Build CLEAN temporary SAM-LoRA.
        # ----------------------------------------------------------

        temporary_model = (
            model_factory()
        )

        if not isinstance(
            temporary_model,
            nn.Module,
        ):
            raise TypeError(
                "model_factory() must return nn.Module, "
                f"received {type(temporary_model).__name__}."
            )

        temporary_model = (
            temporary_model.to(
                device
            )
        )

        temporary_model.train()

        criterion = criterion.to(
            device
        )

        # ----------------------------------------------------------
        # Only optimize parameters that are actually trainable:
        #
        #     LoRA
        #     +
        #     mask decoder
        # ----------------------------------------------------------

        trainable_parameters = [
            parameter
            for parameter
            in temporary_model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "Temporary SAM-LoRA model contains zero trainable "
                "parameters."
            )

        # ----------------------------------------------------------
        # Temporary optimizer.
        # ----------------------------------------------------------

        if optimizer_factory is None:

            optimizer = (
                _build_default_optimizer(
                    parameters=(
                        trainable_parameters
                    ),
                    learning_rate=start_lr,
                    weight_decay=(
                        weight_decay
                    ),
                    betas=betas,
                    eps=eps,
                )
            )

        else:

            optimizer = (
                optimizer_factory(
                    trainable_parameters,
                    float(start_lr),
                )
            )

            if not isinstance(
                optimizer,
                Optimizer,
            ):
                raise TypeError(
                    "optimizer_factory must return a PyTorch "
                    "Optimizer."
                )

        # ----------------------------------------------------------
        # Exponential LR schedule.
        # ----------------------------------------------------------

        lr_multiplier = (
            _calculate_lr_multiplier(
                start_lr=start_lr,
                end_lr=end_lr,
                num_iter=num_iter,
            )
        )

        current_lr = float(
            start_lr
        )

        # ----------------------------------------------------------
        # Histories.
        # ----------------------------------------------------------

        learning_rates: List[
            float
        ] = []

        losses: List[
            float
        ] = []

        running_average = 0.0

        best_smoothed_loss = (
            float("inf")
        )

        stopped_early = False
        stop_reason = (
            "completed_requested_iterations"
        )

        # ----------------------------------------------------------
        # DataLoader iterator.
        #
        # If num_iter > len(train_loader), restart the loader.
        # ----------------------------------------------------------

        data_iterator = iter(
            train_loader
        )

        if verbose:
            print()
            print("=" * 72)
            print("SAM-LoRA Auto-LR")
            print("=" * 72)

            print(
                f"Device                      : "
                f"{device}"
            )

            print(
                f"Start LR                    : "
                f"{start_lr:.3e}"
            )

            print(
                f"Maximum LR                  : "
                f"{end_lr:.3e}"
            )

            print(
                f"Maximum iterations          : "
                f"{num_iter}"
            )

            print(
                f"Trainable parameters        : "
                f"{sum(p.numel() for p in trainable_parameters):,}"
            )

            print(
                "Loss                        : "
                "same criterion as final training"
            )

            print("-" * 72)

        # ----------------------------------------------------------
        # LR range test.
        # ----------------------------------------------------------

        for step_index in range(
            num_iter
        ):

            try:
                batch = next(
                    data_iterator
                )

            except StopIteration:
                data_iterator = iter(
                    train_loader
                )

                batch = next(
                    data_iterator
                )

            images, masks = (
                batch_extractor(
                    batch
                )
            )

            images = images.to(
                device=device,
                non_blocking=True,
            )

            masks = masks.to(
                device=device,
                non_blocking=True,
            )

            _set_optimizer_lr(
                optimizer=optimizer,
                learning_rate=current_lr,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------------
            # Forward.
            # ------------------------------------------------------

            model_output = (
                temporary_model(
                    images
                )
            )

            logits = _extract_logits(
                model_output
            )

            # ------------------------------------------------------
            # SAME BCE + Dice objective as final training.
            # ------------------------------------------------------

            loss_tensor = criterion(
                logits,
                masks,
            )

            if (
                not torch.is_tensor(
                    loss_tensor
                )
                or loss_tensor.ndim != 0
            ):
                raise RuntimeError(
                    "SAM-LoRA Auto-LR criterion must return one "
                    "scalar loss tensor."
                )

            if not torch.isfinite(
                loss_tensor
            ):
                stopped_early = True
                stop_reason = (
                    "non_finite_loss"
                )
                break

            raw_loss = float(
                loss_tensor.detach().item()
            )

            # ------------------------------------------------------
            # Backpropagation BEFORE LR increment.
            # ------------------------------------------------------

            loss_tensor.backward()

            # ------------------------------------------------------
            # Gradient safety.
            #
            # We intentionally do NOT gradient-clip during LR finder.
            #
            # Gradient clipping could hide the LR at which training
            # becomes unstable.
            # ------------------------------------------------------

            gradients_finite = True

            for parameter in (
                trainable_parameters
            ):

                gradient = (
                    parameter.grad
                )

                if gradient is None:
                    continue

                if not torch.isfinite(
                    gradient
                ).all():
                    gradients_finite = (
                        False
                    )
                    break

            if not gradients_finite:
                stopped_early = True
                stop_reason = (
                    "non_finite_gradient"
                )
                break

            optimizer.step()

            # ------------------------------------------------------
            # Smooth noisy minibatch loss.
            # ------------------------------------------------------

            (
                running_average,
                smoothed_loss,
            ) = _update_smoothed_loss(
                raw_loss=raw_loss,
                previous_average=(
                    running_average
                ),
                beta=beta,
                step_index=step_index,
            )

            if not math.isfinite(
                smoothed_loss
            ):
                stopped_early = True
                stop_reason = (
                    "non_finite_smoothed_loss"
                )
                break

            learning_rates.append(
                float(current_lr)
            )

            losses.append(
                float(smoothed_loss)
            )

            # ------------------------------------------------------
            # Track best stable loss.
            # ------------------------------------------------------

            if (
                smoothed_loss
                < best_smoothed_loss
            ):
                best_smoothed_loss = (
                    smoothed_loss
                )

            # ------------------------------------------------------
            # Divergence detection.
            #
            # Skip earliest steps because loss smoothing is still
            # settling.
            # ------------------------------------------------------

            if (
                step_index >= burn_in
                and smoothed_loss
                >
                best_smoothed_loss
                * divergence_threshold
            ):
                stopped_early = True

                stop_reason = (
                    "loss_divergence_detected"
                )

                if verbose:
                    print(
                        "\nAuto-LR stopped early: "
                        "loss divergence detected."
                    )

                break

            # ------------------------------------------------------
            # Optional lightweight progress.
            # ------------------------------------------------------

            if verbose:

                should_print = (
                    step_index == 0
                    or (step_index + 1) % 10 == 0
                    or step_index
                    == num_iter - 1
                )

                if should_print:
                    print(
                        f"Step "
                        f"{step_index + 1:>3}/{num_iter} | "
                        f"LR {current_lr:.3e} | "
                        f"Loss {smoothed_loss:.6f}"
                    )

            # ------------------------------------------------------
            # Increase LR exponentially.
            # ------------------------------------------------------

            current_lr *= (
                lr_multiplier
            )

        # ----------------------------------------------------------
        # Validate completed history.
        # ----------------------------------------------------------

        if len(
            learning_rates
        ) < 8:
            raise RuntimeError(
                "SAM-LoRA Auto-LR stopped before enough valid "
                "measurements were collected.\n"
                f"Completed points: {len(learning_rates)}\n"
                f"Stop reason: {stop_reason}"
            )

        # ----------------------------------------------------------
        # Select LR.
        # ----------------------------------------------------------

        (
            suggested_lr,
            steepest_lr,
            minimum_loss_lr,
            minimum_loss,
        ) = _select_learning_rate(
            learning_rates=(
                learning_rates
            ),
            losses=losses,
            burn_in=burn_in,
            edge_fraction=(
                edge_fraction
            ),
            safety_factor=(
                safety_factor
            ),
        )

        result = AutoLRResult(
            suggested_lr=float(
                suggested_lr
            ),
            steepest_lr=float(
                steepest_lr
            ),
            minimum_loss_lr=float(
                minimum_loss_lr
            ),
            minimum_loss=float(
                minimum_loss
            ),
            start_lr=float(
                start_lr
            ),
            requested_end_lr=float(
                end_lr
            ),
            actual_end_lr=float(
                learning_rates[-1]
            ),
            requested_iterations=int(
                num_iter
            ),
            completed_iterations=int(
                len(
                    learning_rates
                )
            ),
            stopped_early=bool(
                stopped_early
            ),
            stop_reason=str(
                stop_reason
            ),
            learning_rates=[
                float(value)
                for value
                in learning_rates
            ],
            losses=[
                float(value)
                for value
                in losses
            ],
        )

        if verbose:

            print("-" * 72)

            print(
                f"Completed iterations        : "
                f"{result.completed_iterations}"
            )

            print(
                f"Lowest-loss LR             : "
                f"{result.minimum_loss_lr:.3e}"
            )

            print(
                f"Steepest-descent LR        : "
                f"{result.steepest_lr:.3e}"
            )

            print(
                f"Resolved learning rate     : "
                f"{result.suggested_lr:.3e}"
            )

            print(
                f"Stop reason                : "
                f"{result.stop_reason}"
            )

            print("=" * 72)
            print()

        return result

    finally:

        # ----------------------------------------------------------
        # CRITICAL:
        #
        # Destroy temporary LR-finder state.
        #
        # None of these temporary weights may ever be reused for
        # final training.
        # ----------------------------------------------------------

        if optimizer is not None:
            del optimizer

        if temporary_model is not None:
            del temporary_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Convenience resolver
# ---------------------------------------------------------------------


def resolve_sam_lora_learning_rate(
    model_factory: ModelFactory,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: Union[
        str,
        torch.device,
    ] = "cuda",
    start_lr: float = 1e-7,
    end_lr: float = 1e-2,
    num_iter: int = 100,
    beta: float = 0.98,
    divergence_threshold: float = 4.0,
    weight_decay: float = 1e-4,
    betas: Tuple[
        float,
        float,
    ] = (
        0.9,
        0.999,
    ),
    eps: float = 1e-8,
    batch_extractor: Optional[
        BatchExtractor
    ] = None,
    optimizer_factory: Optional[
        OptimizerFactory
    ] = None,
    burn_in: int = 5,
    edge_fraction: float = 0.10,
    safety_factor: float = 1.0,
    verbose: bool = True,
) -> Tuple[
    float,
    AutoLRResult,
]:
    """
    Convenience wrapper used by the future SAM-LoRA trainer.

    Returns
    -------
    resolved_lr:
        Learning rate used for FINAL training.

    result:
        Full LR-range-test metadata.

    Example
    -------

        resolved_lr, auto_lr_result = (
            resolve_sam_lora_learning_rate(
                model_factory=...,
                train_loader=train_loader,
                criterion=criterion,
                device="cuda",
            )
        )

    Then FINAL training must create a NEW model:

        final_model = model_factory()

    The temporary LR-finder model is already destroyed.
    """

    result = (
        run_sam_lora_lr_range_test(
            model_factory=model_factory,
            train_loader=train_loader,
            criterion=criterion,
            device=device,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            beta=beta,
            divergence_threshold=(
                divergence_threshold
            ),
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            batch_extractor=(
                batch_extractor
            ),
            optimizer_factory=(
                optimizer_factory
            ),
            burn_in=burn_in,
            edge_fraction=(
                edge_fraction
            ),
            safety_factor=(
                safety_factor
            ),
            verbose=verbose,
        )
    )

    return (
        float(
            result.suggested_lr
        ),
        result,
    )
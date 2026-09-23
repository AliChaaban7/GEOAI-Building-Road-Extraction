"""
Buildings/src/models/sam_lora/trainer.py

Training engine for SAM-LoRA building segmentation.

Learning-rate policies
----------------------
FIXED:
    Build one clean SAM-LoRA model
        -> AdamW with configured LR
        -> train normally

AUTO:
    Build temporary SAM-LoRA model
        -> LR range test
        -> discard temporary model
        -> build a fresh SAM-LoRA model
        -> train using resolved LR

The fixed mode is important for the current baseline:
    learning_rate.mode  = "fixed"
    learning_rate.value = 0.0001

Auto-LR remains available for future experiments and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
import contextlib
import csv
import gc
import inspect
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESULT
# ============================================================

@dataclass
class SAMLoRATrainingResult:
    best_epoch: int
    best_validation_iou: float
    best_checkpoint: Path
    last_checkpoint: Path
    resolved_learning_rate: float
    learning_rate_mode: str
    epochs_completed: int
    stopped_early: bool
    summary_file: Path
    history_file: Path
    auto_lr_file: Path
    trainable_parameters: int
    total_parameters: int
    training_seconds: float


# ============================================================
# JSON / CSV
# ============================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    return str(value)


def _save_json(
    payload: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_safe(payload),
            file,
            indent=2,
        )


def _save_history_csv(
    rows: list[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# HELPERS
# ============================================================

def _normalize_learning_rate_mode(
    value: Any,
) -> str:
    mode = str(value).strip().lower()

    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "auto_lr": "auto",
        "lr_finder": "auto",
        "fixed": "fixed",
        "manual": "fixed",
        "constant": "fixed",
    }

    if mode not in aliases:
        raise ValueError(
            "learning_rate_mode must be 'fixed' or 'auto'. "
            f"Received: {value!r}"
        )

    return aliases[mode]


def _positive_finite(
    value: Any,
    name: str,
) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be a positive number. "
            f"Received: {value!r}"
        ) from exc

    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"{name} must be finite and > 0. "
            f"Received: {number}"
        )

    return number


def _call_with_supported_kwargs(
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    signature = inspect.signature(
        function
    )

    if any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in signature.parameters.values()
    ):
        return function(
            **kwargs
        )

    filtered = {
        key: value
        for key, value
        in kwargs.items()
        if key
        in signature.parameters
    }

    return function(
        **filtered
    )


def _resolve_model(
    factory_result: Any,
) -> nn.Module:
    result = factory_result

    if isinstance(
        result,
        (tuple, list),
    ):
        modules = [
            item
            for item in result
            if isinstance(
                item,
                nn.Module,
            )
        ]

        if not modules:
            raise TypeError(
                "model_factory returned a tuple/list "
                "without a torch.nn.Module."
            )

        result = modules[0]

    if not isinstance(
        result,
        nn.Module,
    ):
        for attribute in (
            "model",
            "segmenter",
            "network",
        ):
            candidate = getattr(
                result,
                attribute,
                None,
            )

            if isinstance(
                candidate,
                nn.Module,
            ):
                result = candidate
                break

    if not isinstance(
        result,
        nn.Module,
    ):
        raise TypeError(
            "model_factory did not return a torch.nn.Module. "
            f"Received: {type(factory_result).__name__}"
        )

    return result


def _unpack_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(
        batch,
        dict,
    ):
        images = batch.get(
            "image",
            batch.get(
                "images",
            ),
        )

        masks = batch.get(
            "mask",
            batch.get(
                "masks",
                batch.get(
                    "label",
                    batch.get(
                        "labels",
                    ),
                ),
            ),
        )

        if images is None or masks is None:
            raise KeyError(
                "SAM-LoRA batch dictionary must contain "
                "image/images and mask/masks/label/labels."
            )

        return images, masks

    if isinstance(
        batch,
        (tuple, list),
    ) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(
        "Unsupported SAM-LoRA batch type: "
        f"{type(batch).__name__}"
    )


def _extract_logits(
    output: Any,
) -> torch.Tensor:
    if torch.is_tensor(
        output
    ):
        logits = output

    elif isinstance(
        output,
        dict,
    ):
        logits = (
            output.get("logits")
            if output.get("logits") is not None
            else output.get("out")
        )

        if logits is None:
            logits = output.get(
                "masks",
            )

    else:
        logits = getattr(
            output,
            "logits",
            None,
        )

        if logits is None:
            logits = getattr(
                output,
                "out",
                None,
            )

    if not torch.is_tensor(
        logits
    ):
        raise TypeError(
            "Could not extract logits from SAM-LoRA model output."
        )

    if logits.ndim == 3:
        logits = logits.unsqueeze(
            1
        )

    if (
        logits.ndim != 4
        or logits.shape[1] != 1
    ):
        raise ValueError(
            "SAM-LoRA logits must have shape [B,1,H,W]. "
            f"Received: {tuple(logits.shape)}"
        )

    return logits


def _extract_loss(
    criterion_output: Any,
) -> torch.Tensor:
    if torch.is_tensor(
        criterion_output
    ):
        return criterion_output

    if isinstance(
        criterion_output,
        dict,
    ):
        for key in (
            "loss",
            "total_loss",
            "total",
        ):
            candidate = criterion_output.get(
                key,
                None,
            )

            if torch.is_tensor(
                candidate
            ):
                return candidate

    for attribute in (
        "loss",
        "total_loss",
        "total",
    ):
        candidate = getattr(
            criterion_output,
            attribute,
            None,
        )

        if torch.is_tensor(
            candidate
        ):
            return candidate

    if isinstance(
        criterion_output,
        (tuple, list),
    ):
        for item in criterion_output:
            if torch.is_tensor(
                item
            ) and item.ndim == 0:
                return item

    raise TypeError(
        "SAM-LoRA criterion did not return a scalar loss tensor."
    )


def _criterion_loss(
    criterion: Callable[..., Any],
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    output = criterion(
        logits,
        masks,
    )

    loss = _extract_loss(
        output
    )

    if loss.ndim != 0:
        loss = loss.mean()

    return loss


def _count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        int(total),
        int(trainable),
    )


def _trainable_state_dict(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    trainable_names = {
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    full_state = model.state_dict()

    return {
        key: value.detach().cpu()
        for key, value
        in full_state.items()
        if key in trainable_names
    }


def _global_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> Dict[str, float]:
    eps = 1e-12

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
            + eps
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + eps
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + eps
        )
    )

    return {
        "iou": float(iou),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _amp_context(
    enabled: bool,
    device: torch.device,
):
    if (
        enabled
        and device.type == "cuda"
    ):
        try:
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        except Exception:
            return torch.cuda.amp.autocast(
                enabled=True,
            )

    return contextlib.nullcontext()


def _build_grad_scaler(
    enabled: bool,
    device: torch.device,
):
    active = bool(
        enabled
        and device.type == "cuda"
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=active,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=active,
        )


def _optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    name = str(
        optimizer_name
    ).strip().lower()

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise RuntimeError(
            "SAM-LoRA has zero trainable parameters."
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(
                learning_rate
            ),
            weight_decay=float(
                weight_decay
            ),
        )

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=float(
                learning_rate
            ),
            weight_decay=float(
                weight_decay
            ),
        )

    raise ValueError(
        "Unsupported SAM-LoRA optimizer: "
        f"{optimizer_name!r}"
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
):
    name = str(
        scheduler_name
    ).strip().lower()

    if name in (
        "",
        "none",
        "off",
        "false",
    ):
        return None

    if name in (
        "cosine",
        "cosineannealing",
        "cosine_annealing",
    ):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                int(epochs),
                1,
            ),
        )

    if name in (
        "step",
        "steplr",
    ):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(
                int(epochs) // 3,
                1,
            ),
            gamma=0.5,
        )

    raise ValueError(
        "Unsupported SAM-LoRA scheduler: "
        f"{scheduler_name!r}"
    )


# ============================================================
# EPOCHS
# ============================================================

def _train_one_epoch(
    model: nn.Module,
    train_loader: Iterable[Any],
    criterion: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    threshold: float,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()

    loss_total = 0.0
    sample_count = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for batch in train_loader:
        images, masks = _unpack_batch(
            batch
        )

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        ).float()

        optimizer.zero_grad(
            set_to_none=True
        )

        with _amp_context(
            mixed_precision,
            device,
        ):
            logits = _extract_logits(
                model(
                    images
                )
            )

            if (
                logits.shape[-2:]
                != masks.shape[-2:]
            ):
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = _criterion_loss(
                criterion,
                logits,
                masks,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        with torch.no_grad():
            predictions = (
                torch.sigmoid(
                    logits
                )
                >= float(
                    threshold
                )
            )

            targets = (
                masks >= 0.5
            )

            tp += int(
                torch.logical_and(
                    predictions,
                    targets,
                ).sum().item()
            )

            fp += int(
                torch.logical_and(
                    predictions,
                    torch.logical_not(
                        targets
                    ),
                ).sum().item()
            )

            fn += int(
                torch.logical_and(
                    torch.logical_not(
                        predictions
                    ),
                    targets,
                ).sum().item()
            )

            tn += int(
                torch.logical_and(
                    torch.logical_not(
                        predictions
                    ),
                    torch.logical_not(
                        targets
                    ),
                ).sum().item()
            )

        batch_size = int(
            images.shape[0]
        )

        loss_total += (
            float(
                loss.detach().item()
            )
            * batch_size
        )

        sample_count += batch_size

        del images, masks, logits, loss

    metrics = _global_metrics(
        tp,
        fp,
        fn,
        tn,
    )

    metrics["loss"] = (
        loss_total
        / max(
            sample_count,
            1,
        )
    )

    return metrics


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    val_loader: Iterable[Any],
    criterion: Callable[..., Any],
    device: torch.device,
    threshold: float,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.eval()

    loss_total = 0.0
    sample_count = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for batch in val_loader:
        images, masks = _unpack_batch(
            batch
        )

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        ).float()

        with _amp_context(
            mixed_precision,
            device,
        ):
            logits = _extract_logits(
                model(
                    images
                )
            )

            if (
                logits.shape[-2:]
                != masks.shape[-2:]
            ):
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = _criterion_loss(
                criterion,
                logits,
                masks,
            )

        predictions = (
            torch.sigmoid(
                logits
            )
            >= float(
                threshold
            )
        )

        targets = (
            masks >= 0.5
        )

        tp += int(
            torch.logical_and(
                predictions,
                targets,
            ).sum().item()
        )

        fp += int(
            torch.logical_and(
                predictions,
                torch.logical_not(
                    targets
                ),
            ).sum().item()
        )

        fn += int(
            torch.logical_and(
                torch.logical_not(
                    predictions
                ),
                targets,
            ).sum().item()
        )

        tn += int(
            torch.logical_and(
                torch.logical_not(
                    predictions
                ),
                torch.logical_not(
                    targets
                ),
            ).sum().item()
        )

        batch_size = int(
            images.shape[0]
        )

        loss_total += (
            float(
                loss.detach().item()
            )
            * batch_size
        )

        sample_count += batch_size

        del images, masks, logits, loss

    metrics = _global_metrics(
        tp,
        fp,
        fn,
        tn,
    )

    metrics["loss"] = (
        loss_total
        / max(
            sample_count,
            1,
        )
    )

    return metrics


# ============================================================
# CHECKPOINTS
# ============================================================

def _save_best_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    validation_metrics: Dict[str, float],
    resolved_learning_rate: float,
    learning_rate_mode: str,
    total_parameters: int,
    trainable_parameters: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "checkpoint_format":
            "sam_lora_trainable_only_v1",

        "model_type":
            "sam_lora",

        "epoch":
            int(epoch),

        "trained_epoch":
            int(epoch),

        "best_val_iou":
            float(
                validation_metrics[
                    "iou"
                ]
            ),

        "validation_iou":
            float(
                validation_metrics[
                    "iou"
                ]
            ),

        "validation_metrics":
            dict(
                validation_metrics
            ),

        "resolved_learning_rate":
            float(
                resolved_learning_rate
            ),

        "learning_rate_mode":
            str(
                learning_rate_mode
            ),

        "total_parameters":
            int(
                total_parameters
            ),

        "trainable_parameters":
            int(
                trainable_parameters
            ),

        # Official SAM weights are intentionally NOT duplicated here.
        "trainable_state_dict":
            _trainable_state_dict(
                model
            ),
    }

    torch.save(
        payload,
        path,
    )


def _save_last_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    validation_metrics: Dict[str, float],
    resolved_learning_rate: float,
    learning_rate_mode: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "checkpoint_format":
            "sam_lora_training_state_v1",

        "model_type":
            "sam_lora",

        "epoch":
            int(epoch),

        "validation_metrics":
            dict(
                validation_metrics
            ),

        "resolved_learning_rate":
            float(
                resolved_learning_rate
            ),

        "learning_rate_mode":
            str(
                learning_rate_mode
            ),

        "trainable_state_dict":
            _trainable_state_dict(
                model
            ),

        "optimizer_state_dict":
            optimizer.state_dict(),
    }

    try:
        payload[
            "scaler_state_dict"
        ] = scaler.state_dict()
    except Exception:
        payload[
            "scaler_state_dict"
        ] = None

    torch.save(
        payload,
        path,
    )


# ============================================================
# AUTO-LR COMPATIBILITY
# ============================================================

def _resolve_auto_learning_rate(
    model_factory: Callable[[], nn.Module],
    train_loader: Iterable[Any],
    criterion: Callable[..., Any],
    device: torch.device,
    weight_decay: float,
    mixed_precision: bool,
    min_lr: float,
    max_lr: float,
    num_steps: int,
    smoothing_beta: float,
    divergence_threshold: float,
    selection_method: str,
    output_dir: Path,
    verbose: bool,
) -> tuple[float, Dict[str, Any]]:
    """
    Keep the project's existing Auto-LR implementation intact.

    This helper only adapts argument names if auto_lr.py has a slightly
    different signature.
    """

    from .auto_lr import (
        resolve_sam_lora_learning_rate,
    )

    result = _call_with_supported_kwargs(
        resolve_sam_lora_learning_rate,
        model_factory=model_factory,
        train_loader=train_loader,
        dataloader=train_loader,
        criterion=criterion,
        loss_fn=criterion,
        device=device,
        optimizer_name="adamw",
        weight_decay=float(
            weight_decay
        ),
        mixed_precision=bool(
            mixed_precision
        ),
        min_lr=float(
            min_lr
        ),
        max_lr=float(
            max_lr
        ),
        start_lr=float(
            min_lr
        ),
        end_lr=float(
            max_lr
        ),
        num_steps=int(
            num_steps
        ),
        max_iterations=int(
            num_steps
        ),
        smoothing_beta=float(
            smoothing_beta
        ),
        beta=float(
            smoothing_beta
        ),
        divergence_threshold=float(
            divergence_threshold
        ),
        selection_method=str(
            selection_method
        ),
        output_dir=output_dir,
        verbose=bool(
            verbose
        ),
    )

    metadata: Dict[str, Any] = {
        "mode": "auto",
        "raw_result_type":
            type(result).__name__,
    }

    resolved_lr = None

    if isinstance(
        result,
        (float, int),
    ):
        resolved_lr = float(
            result
        )

    elif isinstance(
        result,
        dict,
    ):
        for key in (
            "resolved_learning_rate",
            "selected_lr",
            "learning_rate",
            "best_lr",
            "suggested_lr",
        ):
            if result.get(
                key
            ) is not None:
                resolved_lr = float(
                    result[key]
                )
                break

        metadata.update(
            _json_safe(
                result
            )
        )

    elif isinstance(
        result,
        (tuple, list),
    ):
        for item in result:
            if isinstance(
                item,
                (float, int),
            ):
                resolved_lr = float(
                    item
                )
                break

            if isinstance(
                item,
                dict,
            ):
                for key in (
                    "resolved_learning_rate",
                    "selected_lr",
                    "learning_rate",
                    "best_lr",
                    "suggested_lr",
                ):
                    if item.get(
                        key
                    ) is not None:
                        resolved_lr = float(
                            item[key]
                        )
                        metadata.update(
                            _json_safe(
                                item
                            )
                        )
                        break

            if resolved_lr is not None:
                break

    else:
        for attribute in (
            "resolved_learning_rate",
            "selected_lr",
            "learning_rate",
            "best_lr",
            "suggested_lr",
        ):
            value = getattr(
                result,
                attribute,
                None,
            )

            if value is not None:
                resolved_lr = float(
                    value
                )
                break

        try:
            metadata.update(
                _json_safe(
                    asdict(
                        result
                    )
                )
            )
        except Exception:
            pass

    if resolved_lr is None:
        raise RuntimeError(
            "Auto-LR completed but no selected learning rate "
            "could be resolved from its return value."
        )

    resolved_lr = _positive_finite(
        resolved_lr,
        "Resolved SAM-LoRA Auto-LR",
    )

    return (
        resolved_lr,
        metadata,
    )


# ============================================================
# MAIN TRAINER
# ============================================================

def train_sam_lora(
    model_factory: Callable[[], nn.Module],
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    criterion: Callable[..., Any],
    output_dir: str | Path,
    device: str | torch.device,
    epochs: int = 40,

    optimizer_name: str = "adamw",
    optimizer: Optional[str] = None,
    weight_decay: float = 1e-5,
    scheduler_name: str = "none",
    scheduler: Optional[str] = None,
    mixed_precision: bool = True,

    learning_rate_mode: str = "auto",
    fixed_learning_rate: Optional[float] = None,
    learning_rate: Optional[float] = None,

    validation_threshold: float = 0.5,
    default_threshold: Optional[float] = None,

    auto_lr_min_lr: float = 1e-7,
    auto_lr_max_lr: float = 3e-3,
    auto_lr_num_steps: int = 80,
    auto_lr_smoothing_beta: float = 0.98,
    auto_lr_divergence_threshold: float = 4.0,
    auto_lr_selection_method: str = "steepest_descent",

    early_stopping_patience: Optional[int] = None,
    verbose: bool = True,
    **extra_kwargs: Any,
) -> SAMLoRATrainingResult:
    """
    Train SAM-LoRA.

    Fixed mode:
        learning_rate_mode="fixed"
        fixed_learning_rate=1e-4

    Auto mode:
        learning_rate_mode="auto"
        existing auto_lr.py is used exactly as a development utility.
    """

    del extra_kwargs

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        device
    )

    epochs = int(
        epochs
    )

    if epochs <= 0:
        raise ValueError(
            "epochs must be > 0."
        )

    weight_decay = float(
        weight_decay
    )

    if weight_decay < 0.0:
        raise ValueError(
            "weight_decay must be >= 0."
        )

    if optimizer is not None:
        optimizer_name = str(
            optimizer
        )

    if scheduler is not None:
        scheduler_name = str(
            scheduler
        )

    threshold = float(
        default_threshold
        if default_threshold is not None
        else validation_threshold
    )

    if not (
        0.0
        < threshold
        < 1.0
    ):
        raise ValueError(
            "validation threshold must be between 0 and 1."
        )

    mode = _normalize_learning_rate_mode(
        learning_rate_mode
    )

    auto_lr_file = (
        output_dir
        / "auto_lr.json"
    )

    # ========================================================
    # 1. RESOLVE LEARNING RATE
    # ========================================================

    if mode == "fixed":
        chosen = (
            fixed_learning_rate
            if fixed_learning_rate is not None
            else learning_rate
        )

        resolved_learning_rate = _positive_finite(
            chosen,
            "SAM-LoRA fixed learning rate",
        )

        auto_lr_metadata = {
            "mode": "fixed",
            "auto_lr_used": False,
            "auto_lr_skipped": True,
            "resolved_learning_rate":
                float(
                    resolved_learning_rate
                ),
            "reason":
                "training.learning_rate.mode='fixed'",
        }

        _save_json(
            auto_lr_metadata,
            auto_lr_file,
        )

        if verbose:
            print()
            print("=" * 72)
            print("SAM-LoRA LEARNING RATE")
            print("=" * 72)
            print("Mode                      : FIXED")
            print(
                "Learning Rate             :",
                f"{resolved_learning_rate:.3e}",
            )
            print("Auto-LR Range Test        : SKIPPED")
            print("=" * 72)

    else:
        if verbose:
            print()
            print("=" * 72)
            print("SAM-LoRA AUTO-LR")
            print("=" * 72)
            print("Mode                      : AUTO")

        (
            resolved_learning_rate,
            auto_lr_metadata,
        ) = _resolve_auto_learning_rate(
            model_factory=model_factory,
            train_loader=train_loader,
            criterion=criterion,
            device=device,
            weight_decay=weight_decay,
            mixed_precision=mixed_precision,
            min_lr=auto_lr_min_lr,
            max_lr=auto_lr_max_lr,
            num_steps=auto_lr_num_steps,
            smoothing_beta=auto_lr_smoothing_beta,
            divergence_threshold=auto_lr_divergence_threshold,
            selection_method=auto_lr_selection_method,
            output_dir=output_dir,
            verbose=verbose,
        )

        auto_lr_metadata[
            "resolved_learning_rate"
        ] = float(
            resolved_learning_rate
        )

        _save_json(
            auto_lr_metadata,
            auto_lr_file,
        )

        # The Auto-LR implementation owns its temporary model.
        # Release any cached CUDA blocks before creating the clean
        # final model.
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # 2. BUILD CLEAN FINAL MODEL
    # ========================================================

    if verbose:
        print()
        print("=" * 72)
        print("BUILD FINAL SAM-LoRA MODEL")
        print("=" * 72)

    model = _resolve_model(
        model_factory()
    ).to(
        device
    )

    (
        total_parameters,
        trainable_parameters,
    ) = _count_parameters(
        model
    )

    if trainable_parameters <= 0:
        raise RuntimeError(
            "SAM-LoRA final model has zero trainable parameters."
        )

    if verbose:
        print("Device                      :", device)
        print(
            "Total parameters            :",
            f"{total_parameters:,}",
        )
        print(
            "Trainable parameters        :",
            f"{trainable_parameters:,}",
        )
        print(
            "Resolved learning rate      :",
            f"{resolved_learning_rate:.3e}",
        )
        print(
            "Learning-rate mode          :",
            mode.upper(),
        )
        print(
            "Mixed precision             :",
            "ON"
            if mixed_precision
            else "OFF",
        )
        print("=" * 72)

    optimizer_object = _optimizer(
        model=model,
        learning_rate=resolved_learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

    scheduler_object = _scheduler(
        optimizer=optimizer_object,
        scheduler_name=scheduler_name,
        epochs=epochs,
    )

    scaler = _build_grad_scaler(
        enabled=mixed_precision,
        device=device,
    )

    # ========================================================
    # 3. TRAIN
    # ========================================================

    best_checkpoint = (
        output_dir
        / "best_model.pth"
    )

    last_checkpoint = (
        output_dir
        / "last_checkpoint.pth"
    )

    history_file = (
        output_dir
        / "training_history.json"
    )

    history_csv = (
        output_dir
        / "training_log.csv"
    )

    summary_file = (
        output_dir
        / "training_summary.json"
    )

    best_validation_iou = -1.0
    best_epoch = 0
    epochs_completed = 0
    epochs_without_improvement = 0
    stopped_early = False
    history: list[Dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.perf_counter()

        train_metrics = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer_object,
            scaler=scaler,
            device=device,
            threshold=threshold,
            mixed_precision=mixed_precision,
        )

        validation_metrics = _validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            mixed_precision=mixed_precision,
        )

        if scheduler_object is not None:
            scheduler_object.step()

        current_lr = float(
            optimizer_object.param_groups[0][
                "lr"
            ]
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        row = {
            "epoch":
                int(epoch),

            "train_loss":
                float(
                    train_metrics[
                        "loss"
                    ]
                ),

            "train_iou":
                float(
                    train_metrics[
                        "iou"
                    ]
                ),

            "train_precision":
                float(
                    train_metrics[
                        "precision"
                    ]
                ),

            "train_recall":
                float(
                    train_metrics[
                        "recall"
                    ]
                ),

            "train_f1":
                float(
                    train_metrics[
                        "f1"
                    ]
                ),

            "val_loss":
                float(
                    validation_metrics[
                        "loss"
                    ]
                ),

            "val_iou":
                float(
                    validation_metrics[
                        "iou"
                    ]
                ),

            "val_precision":
                float(
                    validation_metrics[
                        "precision"
                    ]
                ),

            "val_recall":
                float(
                    validation_metrics[
                        "recall"
                    ]
                ),

            "val_f1":
                float(
                    validation_metrics[
                        "f1"
                    ]
                ),

            "learning_rate":
                current_lr,

            "epoch_seconds":
                float(
                    epoch_seconds
                ),
        }

        history.append(
            row
        )

        epochs_completed = int(
            epoch
        )

        improved = (
            float(
                validation_metrics[
                    "iou"
                ]
            )
            > float(
                best_validation_iou
            )
        )

        if improved:
            best_validation_iou = float(
                validation_metrics[
                    "iou"
                ]
            )

            best_epoch = int(
                epoch
            )

            epochs_without_improvement = 0

            _save_best_checkpoint(
                path=best_checkpoint,
                model=model,
                epoch=epoch,
                validation_metrics=validation_metrics,
                resolved_learning_rate=resolved_learning_rate,
                learning_rate_mode=mode,
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
            )

        else:
            epochs_without_improvement += 1

        _save_last_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer_object,
            scaler=scaler,
            epoch=epoch,
            validation_metrics=validation_metrics,
            resolved_learning_rate=resolved_learning_rate,
            learning_rate_mode=mode,
        )

        _save_json(
            {
                "history":
                    history,

                "validation_threshold":
                    threshold,

                "learning_rate_mode":
                    mode,

                "resolved_learning_rate":
                    resolved_learning_rate,
            },
            history_file,
        )

        _save_history_csv(
            history,
            history_csv,
        )

        if verbose:
            marker = (
                " | BEST"
                if improved
                else ""
            )

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {row['train_loss']:.4f} | "
                f"Train IoU: {row['train_iou'] * 100.0:.2f}% | "
                f"Val Loss: {row['val_loss']:.4f} | "
                f"Val IoU: {row['val_iou'] * 100.0:.2f}% | "
                f"P: {row['val_precision'] * 100.0:.2f}% | "
                f"R: {row['val_recall'] * 100.0:.2f}% | "
                f"F1: {row['val_f1'] * 100.0:.2f}% | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_seconds:.2f}s"
                f"{marker}"
            )

        if (
            early_stopping_patience
            is not None
            and int(
                early_stopping_patience
            ) > 0
            and epochs_without_improvement
            >= int(
                early_stopping_patience
            )
        ):
            stopped_early = True

            if verbose:
                print(
                    "Early stopping triggered after",
                    epochs_without_improvement,
                    "epochs without validation-IoU improvement.",
                )

            break

        # Release cached temporary allocations between epochs without
        # changing the model/optimizer state.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    if best_epoch <= 0:
        raise RuntimeError(
            "SAM-LoRA training finished without producing "
            "a best checkpoint."
        )

    # ========================================================
    # 4. SUMMARY
    # ========================================================

    summary = {
        "model_type":
            "sam_lora",

        "training_engine":
            "sam_lora",

        "learning_rate_mode":
            mode,

        "resolved_learning_rate":
            float(
                resolved_learning_rate
            ),

        "auto_lr_used":
            bool(
                mode == "auto"
            ),

        "auto_lr_skipped":
            bool(
                mode == "fixed"
            ),

        "optimizer":
            str(
                optimizer_name
            ).lower(),

        "weight_decay":
            float(
                weight_decay
            ),

        "scheduler":
            str(
                scheduler_name
            ).lower(),

        "mixed_precision":
            bool(
                mixed_precision
            ),

        "validation_threshold":
            float(
                threshold
            ),

        "epochs_requested":
            int(
                epochs
            ),

        "epochs_completed":
            int(
                epochs_completed
            ),

        "stopped_early":
            bool(
                stopped_early
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_iou":
            float(
                best_validation_iou
            ),

        "best_validation_iou":
            float(
                best_validation_iou
            ),

        "best_model_path":
            str(
                best_checkpoint
            ),

        "best_checkpoint":
            str(
                best_checkpoint
            ),

        "last_checkpoint":
            str(
                last_checkpoint
            ),

        "training_history_json":
            str(
                history_file
            ),

        "training_log_csv":
            str(
                history_csv
            ),

        "auto_lr_json":
            str(
                auto_lr_file
            ),

        "total_parameters":
            int(
                total_parameters
            ),

        "trainable_parameters":
            int(
                trainable_parameters
            ),

        "trainable_percentage":
            float(
                100.0
                * trainable_parameters
                / max(
                    total_parameters,
                    1,
                )
            ),

        "training_seconds":
            float(
                training_seconds
            ),

        "external_test_used":
            False,

        "selection_dataset":
            "validation_only",
    }

    _save_json(
        summary,
        summary_file,
    )

    if verbose:
        print()
        print("=" * 72)
        print("SAM-LoRA TRAINING FINISHED")
        print("=" * 72)
        print(
            "Learning Rate Mode          :",
            mode.upper(),
        )
        print(
            "Resolved Learning Rate      :",
            f"{resolved_learning_rate:.3e}",
        )
        print(
            "Best Epoch                  :",
            best_epoch,
        )
        print(
            "Best Validation IoU         :",
            f"{best_validation_iou * 100.0:.2f}%",
        )
        print(
            "Best Model                  :",
            best_checkpoint,
        )
        print(
            "External Test Used          : NO",
        )
        print("=" * 72)
        print()

    return SAMLoRATrainingResult(
        best_epoch=int(
            best_epoch
        ),
        best_validation_iou=float(
            best_validation_iou
        ),
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        resolved_learning_rate=float(
            resolved_learning_rate
        ),
        learning_rate_mode=mode,
        epochs_completed=int(
            epochs_completed
        ),
        stopped_early=bool(
            stopped_early
        ),
        summary_file=summary_file,
        history_file=history_file,
        auto_lr_file=auto_lr_file,
        trainable_parameters=int(
            trainable_parameters
        ),
        total_parameters=int(
            total_parameters
        ),
        training_seconds=float(
            training_seconds
        ),
    )


# ============================================================
# TRAINABLE-CHECKPOINT LOADER
# ============================================================

def load_sam_lora_trainable_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    device: str | torch.device | None = None,
    verbose: bool = True,
    checkpoint: str | Path | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """
    Load a SAM-LoRA trainable-only checkpoint into an already-built model.

    This function is part of the public SAM-LoRA package API and is imported
    by src.models.sam_lora.__init__.

    Supported checkpoint payloads
    -----------------------------
    Preferred:
        {
            "trainable_state_dict": {...},
            ...
        }

    Compatibility:
        {
            "model_state_dict": {...},
            ...
        }

    It also accepts a raw state-dict mapping.

    Notes
    -----
    The official pretrained SAM checkpoint is NOT contained in best_model.pth.
    The caller must first reconstruct the base SAM model from the official
    Meta checkpoint, inject the same LoRA structure, and then call this loader.
    """

    if checkpoint_path is None:
        checkpoint_path = checkpoint

    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required for "
            "load_sam_lora_trainable_checkpoint()."
        )

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            "SAM-LoRA trainable checkpoint was not found:\n"
            f"{path}"
        )

    if map_location is None:
        map_location = (
            torch.device(device)
            if device is not None
            else torch.device("cpu")
        )

    payload = torch.load(
        path,
        map_location=map_location,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "SAM-LoRA checkpoint must contain a dictionary. "
            f"Received: {type(payload).__name__}"
        )

    state_dict = None

    if isinstance(
        payload.get("trainable_state_dict"),
        dict,
    ):
        state_dict = payload["trainable_state_dict"]

    elif isinstance(
        payload.get("model_state_dict"),
        dict,
    ):
        state_dict = payload["model_state_dict"]

    else:
        # Compatibility with a raw state_dict saved directly with torch.save().
        if payload and all(
            torch.is_tensor(value)
            for value in payload.values()
        ):
            state_dict = payload

    if state_dict is None:
        raise KeyError(
            "Could not find a SAM-LoRA state dictionary in checkpoint. "
            "Expected 'trainable_state_dict', 'model_state_dict', "
            "or a raw tensor state_dict."
        )

    current_state = model.state_dict()

    unexpected_keys = [
        key
        for key in state_dict.keys()
        if key not in current_state
    ]

    shape_mismatches = []

    for key, tensor in state_dict.items():
        if key not in current_state:
            continue

        if tuple(tensor.shape) != tuple(current_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": tuple(tensor.shape),
                    "model_shape": tuple(current_state[key].shape),
                }
            )

    if unexpected_keys:
        raise RuntimeError(
            "SAM-LoRA checkpoint contains keys that do not exist "
            "in the reconstructed model:\n- "
            + "\n- ".join(unexpected_keys[:50])
        )

    if shape_mismatches:
        lines = [
            (
                f"{item['key']}: checkpoint "
                f"{item['checkpoint_shape']} != model "
                f"{item['model_shape']}"
            )
            for item in shape_mismatches[:50]
        ]

        raise RuntimeError(
            "SAM-LoRA checkpoint tensor-shape mismatch:\n- "
            + "\n- ".join(lines)
        )

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    checkpoint_names = set(state_dict.keys())

    missing_trainable_keys = sorted(
        trainable_names - checkpoint_names
    )

    if strict and missing_trainable_keys:
        raise RuntimeError(
            "SAM-LoRA trainable checkpoint is missing trainable "
            "parameters required by the reconstructed model:\n- "
            + "\n- ".join(missing_trainable_keys[:50])
        )

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    # Missing frozen base-SAM parameters are expected because best_model.pth
    # intentionally stores only LoRA + trainable mask-decoder parameters.
    unexpected_after_load = list(
        getattr(incompatible, "unexpected_keys", [])
    )

    if unexpected_after_load:
        raise RuntimeError(
            "Unexpected keys remained after loading SAM-LoRA checkpoint:\n- "
            + "\n- ".join(unexpected_after_load[:50])
        )

    if device is not None:
        model.to(torch.device(device))

    info = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_format": payload.get(
            "checkpoint_format",
            "unknown",
        ),
        "epoch": payload.get(
            "epoch",
            payload.get("trained_epoch"),
        ),
        "best_val_iou": payload.get(
            "best_val_iou",
            payload.get("validation_iou"),
        ),
        "resolved_learning_rate": payload.get(
            "resolved_learning_rate",
        ),
        "learning_rate_mode": payload.get(
            "learning_rate_mode",
        ),
        "loaded_tensor_count": len(state_dict),
        "missing_trainable_keys": missing_trainable_keys,
        "strict_trainable_check": bool(strict),
    }

    if verbose:
        print()
        print("=" * 72)
        print("SAM-LoRA TRAINABLE CHECKPOINT LOADED")
        print("=" * 72)
        print("Checkpoint :", path)
        print("Loaded tensors :", len(state_dict))
        print(
            "Missing trainable tensors :",
            len(missing_trainable_keys),
        )
        print("=" * 72)
        print()

    return info

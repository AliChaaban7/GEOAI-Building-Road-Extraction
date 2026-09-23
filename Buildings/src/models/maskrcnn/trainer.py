"""
Mask R-CNN training utilities.

This module preserves the training behavior of the historical
src.instance.maskrcnn_engine while moving model-specific training code into
the Mask R-CNN model package.
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .data import make_loaders
from .evaluation import validation_pixel_iou
from .factory import create_maskrcnn_model


def seed_everything(
    seed: int = 42,
) -> None:
    seed = int(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def make_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
):
    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    name = str(
        optimizer_name
    ).lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(
                learning_rate
            ),
            weight_decay=float(
                weight_decay
            ),
        )

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(
                learning_rate
            ),
            momentum=float(
                momentum
            ),
            weight_decay=float(
                weight_decay
            ),
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


def make_scheduler(
    optimizer,
    scheduler_name: str,
    epochs: int,
):
    name = str(
        scheduler_name
    ).lower()

    if name in {
        "none",
        "off",
        "false",
    }:
        return None

    if name == "step":
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(
                    3,
                    int(
                        epochs
                    )
                    // 3,
                ),
                gamma=0.5,
            )
        )

    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    1,
                    int(
                        epochs
                    ),
                ),
            )
        )

    raise ValueError(
        f"Unsupported scheduler: {scheduler_name}"
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
) -> float:
    model.train()

    running_loss = 0.0
    batches = 0

    for batch_idx, (
        images,
        targets,
    ) in enumerate(
        loader,
        start=1,
    ):
        images = [
            img.to(
                device,
                non_blocking=True,
            )
            for img in images
        ]

        targets = [
            {
                k:
                    v.to(
                        device,
                        non_blocking=True,
                    )
                for k, v
                in target.items()
            }
            for target in targets
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_dict = model(
            images,
            targets,
        )

        loss = sum(
            loss_value
            for loss_value
            in loss_dict.values()
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite Mask R-CNN loss at "
                f"epoch {epoch}, batch {batch_idx}: "
                f"{float(loss.detach().cpu())}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        running_loss += float(
            loss
            .detach()
            .cpu()
        )

        batches += 1

    return (
        running_loss
        / max(
            batches,
            1,
        )
    )


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    val_iou: float,
    params: dict,
    extra_meta: Optional[
        dict
    ] = None,
):
    payload = {
        "epoch":
            int(
                epoch
            ),

        "model_state_dict":
            model.state_dict(),

        "best_val_iou":
            float(
                val_iou
            ),

        "best_validation_iou":
            float(
                val_iou
            ),

        "params":
            dict(
                params
            ),
    }

    if extra_meta:
        payload.update(
            extra_meta
        )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )


def train_model(
    train_pairs,
    val_pairs,
    output_folder: Path,
    epochs: int,
    batch_size: int,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_name: str = "cosine",
    momentum: float = 0.9,
    pretrained: bool = True,
    seed: int = 42,
    train_augmentation=None,
    phase: str = "standard",
    nms_threshold: float = 0.50,
    detections_per_image: int = 300,
) -> dict:
    seed_everything(
        seed
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        train_loader,
        val_loader,
    ) = make_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        batch_size=batch_size,
        train_augmentation=train_augmentation,
    )

    model = (
        create_maskrcnn_model(
            pretrained=pretrained,
            nms_threshold=nms_threshold,
            detections_per_image=detections_per_image,
        )
        .to(
            device
        )
    )

    optimizer = make_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    scheduler = make_scheduler(
        optimizer,
        scheduler_name,
        epochs,
    )

    params = {
        "optimizer":
            optimizer_name,

        "learning_rate":
            float(
                learning_rate
            ),

        "weight_decay":
            float(
                weight_decay
            ),

        "scheduler":
            scheduler_name,

        "momentum":
            float(
                momentum
            ),

        "pretrained":
            bool(
                pretrained
            ),

        "augmentation":
            train_augmentation
            is not None,
    }

    best_val_iou = -1.0
    best_epoch = -1

    history = []

    best_model_path = (
        output_folder
        / "best_model.pth"
    )

    log_path = (
        output_folder
        / "training_log.csv"
    )

    print(
        "\n========== MASK R-CNN TRAINING START =========="
    )
    print(
        f"Phase: {phase}"
    )
    print(
        f"Device: {device}"
    )
    print(
        f"Train Samples: {len(train_pairs)}"
    )
    print(
        f"Validation Samples: {len(val_pairs)}"
    )
    print(
        f"Train Batches: {len(train_loader)}"
    )
    print(
        f"Validation Batches: {len(val_loader)}"
    )
    print(
        f"Batch Size: {batch_size}"
    )
    print(
        f"Epochs: {epochs}"
    )
    print(
        "Augmentation:",
        "ON"
        if train_augmentation is not None
        else "OFF",
    )
    print(
        "Validation Augmentation: OFF"
    )
    print(
        "External Test Used: NO"
    )
    print(
        "================================================\n"
    )

    for epoch in range(
        1,
        int(
            epochs
        )
        + 1,
    ):
        started = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
        )

        val_iou = validation_pixel_iou(
            model,
            val_loader,
            device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        elapsed = (
            time.time()
            - started
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    float(
                        train_loss
                    ),

                "val_iou":
                    float(
                        val_iou
                    ),

                "val_iou_percent":
                    float(
                        val_iou
                        * 100.0
                    ),

                "learning_rate":
                    current_lr,

                "epoch_time_sec":
                    elapsed,
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val IoU: {val_iou * 100:.2f}% | "
            f"LR: {current_lr:.3e}"
        )

        if val_iou > best_val_iou:
            best_val_iou = (
                val_iou
            )

            best_epoch = (
                epoch
            )

            save_checkpoint(
                best_model_path,
                model=model,
                epoch=epoch,
                val_iou=val_iou,
                params=params,
                extra_meta={
                    "phase":
                        phase
                },
            )

            print(
                f"Best model saved at epoch {epoch}"
            )

        pd.DataFrame(
            history
        ).to_csv(
            log_path,
            index=False,
        )

    summary = {
        "phase":
            phase,

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

        "best_validation_iou_percent":
            float(
                best_val_iou
                * 100.0
            ),

        "epochs":
            int(
                epochs
            ),

        "batch_size":
            int(
                batch_size
            ),

        "train_samples":
            int(
                len(
                    train_pairs
                )
            ),

        "validation_samples":
            int(
                len(
                    val_pairs
                )
            ),

        "best_model_path":
            str(
                best_model_path
            ),

        "training_log_csv":
            str(
                log_path
            ),

        "params":
            params,

        "augmentation_enabled":
            train_augmentation
            is not None,

        "validation_augmentation":
            False,

        "external_test_used":
            False,
    }

    (
        output_folder
        / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model
    del optimizer

    if scheduler is not None:
        del scheduler

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


__all__ = [
    "make_optimizer",
    "make_scheduler",
    "save_checkpoint",
    "seed_everything",
    "train_model",
    "train_one_epoch",
]

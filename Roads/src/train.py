"""
Shared native PyTorch training engine for Approach 2 - Road Extraction.

Supported models
----------------
- U-Net
- DeepLabV3+
- SAM-LoRA

NOT handled here
----------------
- ConnectNet
- MultiTaskRoadExtractor

Those two models use the separate ArcGIS Learn backend:

    src/train_arcgis.py

Scientific policy
-----------------
TRAIN:
    learn model parameters

VALIDATION:
    select best epoch by Validation IoU

FINAL TEST:
    never accessed here

Threshold search:
    separate validation-only stage

Post-processing search:
    separate validation-only stage

Checkpoint policy
-----------------
U-Net / DeepLabV3+
    complete state_dict

SAM-LoRA
    only trainable Road-specific parameters:
        - Q/V LoRA
        - SAM mask decoder

    plus:
        - exact model_config
        - exact experiment_config
        - official SAM checkpoint provenance

This makes SAM-LoRA checkpoints much smaller while remaining fully
reconstructable.

Outputs
-------
<output_folder>/
    training_history.csv
    training_history.json
    training_summary.json

<checkpoint_folder>/
    best_model.pth
    last_checkpoint.pth
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from src.checkpointing import (
    save_training_checkpoint,
)

from src.data import (
    RoadSegmentationDataset,
)

from src.losses import (
    build_loss,
)

from src.models import (
    extract_model_logits,
    get_model_parameter_summary,
)


# =====================================================================
# JSON HELPERS
# =====================================================================

def _json_safe(
    value,
):
    """
    Convert common training objects into JSON-safe values.
    """

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    if isinstance(
        value,
        torch.device,
    ):

        return str(
            value
        )

    if isinstance(
        value,
        torch.Tensor,
    ):

        if value.numel() == 1:

            return float(
                value.detach()
                .cpu()
                .item()
            )

        return (
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        return float(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        dict,
    ):

        return {
            str(
                key
            ):
                _json_safe(
                    item
                )

            for (
                key,
                item,
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    return value


def _save_json(
    data,
    path: str | Path,
) -> Path:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                data
            ),
            file,
            indent=2,
        )

    return path


# =====================================================================
# DEVICE
# =====================================================================

def resolve_device(
    device: Optional[
        str | torch.device
    ] = None,
) -> torch.device:
    """
    Resolve training device.
    """

    if isinstance(
        device,
        torch.device,
    ):

        return device

    if device is not None:

        resolved = torch.device(
            str(
                device
            )
        )

        if (
            resolved.type
            == "cuda"
            and not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA device requested but CUDA is not available."
            )

        return resolved

    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def seed_training(
    seed: int,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy and PyTorch.
    """

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

        torch.backends.cudnn.deterministic = bool(
            deterministic
        )

        torch.backends.cudnn.benchmark = not bool(
            deterministic
        )

    except Exception:

        pass


# =====================================================================
# CONFIG HELPERS
# =====================================================================

def _as_dict(
    value,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


def _first_not_none(
    *values,
):

    for value in values:

        if value is not None:

            return value

    return None


def _general_params_from_context(
    context: Dict[str, Any],
) -> Dict[str, Any]:

    general = context.get(
        "general_params"
    )

    if general is None:

        general = context.get(
            "general_config"
        )

    if not isinstance(
        general,
        dict,
    ):

        raise KeyError(
            "Training context requires "
            "context['general_params']."
        )

    return general


# =====================================================================
# TRAINING CONFIGURATION
# =====================================================================

def resolve_training_config(
    general_params: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve common training settings.

    Model-specific configuration overrides general defaults.
    """

    general_training = _as_dict(
        general_params.get(
            "training"
        )
    )

    model_training = _as_dict(
        model_config.get(
            "training"
        )
    )

    # --------------------------------------------------------------
    # Optimizer configuration
    # --------------------------------------------------------------

    general_optimizer = _as_dict(
        general_training.get(
            "optimizer"
        )
    )

    model_optimizer = _as_dict(
        model_training.get(
            "optimizer"
        )
    )

    optimizer_name = str(
        _first_not_none(
            model_optimizer.get(
                "name"
            ),

            model_training.get(
                "optimizer_name"
            ),

            general_optimizer.get(
                "name"
            ),

            general_training.get(
                "optimizer_name"
            ),

            general_training.get(
                "optimizer"
            )
            if isinstance(
                general_training.get(
                    "optimizer"
                ),
                str,
            )
            else None,

            "adamw",
        )
    ).strip().lower()

    learning_rate = float(
        _first_not_none(
            model_training.get(
                "learning_rate"
            ),

            model_training.get(
                "lr"
            ),

            model_optimizer.get(
                "learning_rate"
            ),

            model_optimizer.get(
                "lr"
            ),

            general_training.get(
                "learning_rate"
            ),

            general_training.get(
                "lr"
            ),

            general_optimizer.get(
                "learning_rate"
            ),

            general_optimizer.get(
                "lr"
            ),

            1e-4,
        )
    )

    weight_decay = float(
        _first_not_none(
            model_training.get(
                "weight_decay"
            ),

            model_optimizer.get(
                "weight_decay"
            ),

            general_training.get(
                "weight_decay"
            ),

            general_optimizer.get(
                "weight_decay"
            ),

            1e-5,
        )
    )

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    general_scheduler = _as_dict(
        general_training.get(
            "scheduler"
        )
    )

    model_scheduler = _as_dict(
        model_training.get(
            "scheduler"
        )
    )

    scheduler_name = _first_not_none(
        model_scheduler.get(
            "name"
        ),

        model_training.get(
            "scheduler_name"
        ),

        general_scheduler.get(
            "name"
        ),

        general_training.get(
            "scheduler_name"
        ),
    )

    if scheduler_name is None:

        raw_scheduler = general_training.get(
            "scheduler"
        )

        if isinstance(
            raw_scheduler,
            str,
        ):

            scheduler_name = raw_scheduler

    if scheduler_name is None:

        scheduler_name = "cosine"

    scheduler_name = str(
        scheduler_name
    ).strip().lower()

    # --------------------------------------------------------------
    # AMP
    # --------------------------------------------------------------

    use_amp = bool(
        _first_not_none(
            model_training.get(
                "use_amp"
            ),

            model_training.get(
                "amp"
            ),

            general_training.get(
                "use_amp"
            ),

            general_training.get(
                "amp"
            ),

            True,
        )
    )

    # --------------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------------

    gradient_clipping = _first_not_none(
        model_training.get(
            "gradient_clipping"
        ),

        general_training.get(
            "gradient_clipping"
        ),

        {},
    )

    if isinstance(
        gradient_clipping,
        (
            int,
            float,
        ),
    ):

        gradient_clipping = {
            "enabled":
                True,

            "max_norm":
                float(
                    gradient_clipping
                ),
        }

    gradient_clipping = _as_dict(
        gradient_clipping
    )

    # --------------------------------------------------------------
    # Epoch / batch
    # --------------------------------------------------------------

    epochs = int(
        _first_not_none(
            model_training.get(
                "epochs"
            ),

            general_training.get(
                "epochs"
            ),

            40,
        )
    )

    batch_size = int(
        _first_not_none(
            model_training.get(
                "batch_size"
            ),

            general_training.get(
                "batch_size"
            ),

            8,
        )
    )

    if epochs <= 0:

        raise ValueError(
            "epochs must be > 0."
        )

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be > 0."
        )

    if learning_rate <= 0:

        raise ValueError(
            "learning_rate must be > 0."
        )

    if weight_decay < 0:

        raise ValueError(
            "weight_decay must be >= 0."
        )

    return {
        "epochs":
            epochs,

        "batch_size":
            batch_size,

        "optimizer_name":
            optimizer_name,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "scheduler_name":
            scheduler_name,

        "scheduler_config":
            {
                **general_scheduler,
                **model_scheduler,
            },

        "use_amp":
            use_amp,

        "gradient_clipping":
            gradient_clipping,
    }


# =====================================================================
# DATA LOADER CONFIG
# =====================================================================

def resolve_dataloader_config(
    general_params: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve DataLoader parameters.
    """

    training = _as_dict(
        general_params.get(
            "training"
        )
    )

    data = _as_dict(
        general_params.get(
            "data"
        )
    )

    dataloader = _as_dict(
        general_params.get(
            "dataloader",
            general_params.get(
                "data_loader",
                {},
            ),
        )
    )

    model_training = _as_dict(
        model_config.get(
            "training"
        )
    )

    training_config = (
        resolve_training_config(
            general_params,
            model_config,
        )
    )

    num_workers = int(
        _first_not_none(
            model_training.get(
                "num_workers"
            ),

            dataloader.get(
                "num_workers"
            ),

            training.get(
                "num_workers"
            ),

            0,
        )
    )

    pin_memory = bool(
        _first_not_none(
            model_training.get(
                "pin_memory"
            ),

            dataloader.get(
                "pin_memory"
            ),

            training.get(
                "pin_memory"
            ),

            True,
        )
    )

    image_channels = int(
        _first_not_none(
            data.get(
                "image_channels"
            ),

            data.get(
                "input_channels"
            ),

            3,
        )
    )

    mask_channels = int(
        _first_not_none(
            data.get(
                "mask_channels"
            ),

            data.get(
                "output_channels"
            ),

            1,
        )
    )

    if image_channels != 3:

        raise ValueError(
            "Current Road pipeline expects RGB input_channels=3."
        )

    if mask_channels != 1:

        raise ValueError(
            "Current Road pipeline expects one binary mask channel."
        )

    return {
        "batch_size":
            int(
                training_config[
                    "batch_size"
                ]
            ),

        "num_workers":
            num_workers,

        "pin_memory":
            pin_memory,

        "drop_last":
            False,
    }


# =====================================================================
# DATASETS / LOADERS
# =====================================================================

def _build_road_dataset(
    manifest_csv,
    augmentation,
    tile_size: int,
):
    """
    Build RoadSegmentationDataset while preserving compatibility with
    the current dataset implementation.
    """

    # Current Road dataset API.
    try:

        return RoadSegmentationDataset(
            manifest_csv=manifest_csv,

            augmentation=augmentation,

            expected_tile_size=int(
                tile_size
            ),

            normalize=True,

            return_metadata=False,
        )

    except TypeError:

        pass

    # Compatibility fallback if the implementation uses manifest_path.
    try:

        return RoadSegmentationDataset(
            manifest_path=manifest_csv,

            augmentation=augmentation,

            expected_tile_size=int(
                tile_size
            ),

            normalize=True,

            return_metadata=False,
        )

    except TypeError:

        pass

    # Minimal legacy fallback.
    return RoadSegmentationDataset(
        manifest_csv,
        augmentation=augmentation,
    )


def build_dataloaders(
    context: Dict[str, Any],
    train_augmentation=None,
    validation_augmentation=None,
) -> Dict[str, Any]:
    """
    Build deterministic Train and Validation DataLoaders.
    """

    general_params = (
        _general_params_from_context(
            context
        )
    )

    model_config = context[
        "model_config"
    ]

    experiment_config = context[
        "experiment_config"
    ]

    loader_config = (
        resolve_dataloader_config(
            general_params=general_params,

            model_config=model_config,
        )
    )

    train_manifest = context.get(
        "train_manifest"
    )

    validation_manifest = context.get(
        "validation_manifest"
    )

    if train_manifest is None:

        raise KeyError(
            "context['train_manifest'] is missing."
        )

    if validation_manifest is None:

        raise KeyError(
            "context['validation_manifest'] is missing."
        )

    train_manifest = Path(
        train_manifest
    )

    validation_manifest = Path(
        validation_manifest
    )

    if not train_manifest.exists():

        raise FileNotFoundError(
            "Train manifest not found:\n"
            f"{train_manifest}"
        )

    if not validation_manifest.exists():

        raise FileNotFoundError(
            "Validation manifest not found:\n"
            f"{validation_manifest}"
        )

    tile_size = int(
        experiment_config[
            "tile_size"
        ]
    )

    train_dataset = (
        _build_road_dataset(
            manifest_csv=train_manifest,

            augmentation=train_augmentation,

            tile_size=tile_size,
        )
    )

    validation_dataset = (
        _build_road_dataset(
            manifest_csv=validation_manifest,

            augmentation=validation_augmentation,

            tile_size=tile_size,
        )
    )

    if len(
        train_dataset
    ) == 0:

        raise RuntimeError(
            "Training dataset contains zero samples."
        )

    if len(
        validation_dataset
    ) == 0:

        raise RuntimeError(
            "Validation dataset contains zero samples."
        )

    train_loader = DataLoader(
        train_dataset,

        batch_size=int(
            loader_config[
                "batch_size"
            ]
        ),

        shuffle=True,

        num_workers=int(
            loader_config[
                "num_workers"
            ]
        ),

        pin_memory=bool(
            loader_config[
                "pin_memory"
            ]
        ),

        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=int(
            loader_config[
                "batch_size"
            ]
        ),

        shuffle=False,

        num_workers=int(
            loader_config[
                "num_workers"
            ]
        ),

        pin_memory=bool(
            loader_config[
                "pin_memory"
            ]
        ),

        drop_last=False,
    )

    return {
        "train_dataset":
            train_dataset,

        "validation_dataset":
            validation_dataset,

        "train_loader":
            train_loader,

        "validation_loader":
            validation_loader,

        "train_samples":
            int(
                len(
                    train_dataset
                )
            ),

        "validation_samples":
            int(
                len(
                    validation_dataset
                )
            ),

        "batch_size":
            int(
                loader_config[
                    "batch_size"
                ]
            ),
    }


# Backward-compatible alias.
create_dataloaders = build_dataloaders


# =====================================================================
# BATCH HANDLING
# =====================================================================

def unpack_batch(
    batch,
):
    """
    Normalize supported dataset batch formats to images, masks.
    """

    if isinstance(
        batch,
        dict,
    ):

        image = _first_not_none(
            batch.get(
                "image"
            ),

            batch.get(
                "images"
            ),
        )

        mask = _first_not_none(
            batch.get(
                "mask"
            ),

            batch.get(
                "masks"
            ),

            batch.get(
                "label"
            ),

            batch.get(
                "labels"
            ),
        )

        if image is None or mask is None:

            raise KeyError(
                "Dictionary batch must contain image and mask."
            )

        return (
            image,
            mask,
        )

    if isinstance(
        batch,
        (
            tuple,
            list,
        ),
    ):

        if len(
            batch
        ) < 2:

            raise ValueError(
                "Dataset batch must contain image and mask."
            )

        return (
            batch[
                0
            ],
            batch[
                1
            ],
        )

    raise TypeError(
        "Unsupported Road dataset batch type:\n"
        f"{type(batch).__name__}"
    )


# =====================================================================
# LOSS
# =====================================================================

def build_loss_function(
    general_params: Dict[str, Any],
    model_config: Dict[str, Any],
):
    """
    Compatibility wrapper around src.losses.build_loss().
    """

    return build_loss(
        general_params=general_params,

        model_config=model_config,
    )


def _loss_scalar(
    loss_result,
) -> torch.Tensor:
    """
    Normalize loss implementations that return tensor or dictionary.
    """

    if torch.is_tensor(
        loss_result
    ):

        return loss_result

    if isinstance(
        loss_result,
        dict,
    ):

        for key in (
            "loss",
            "total_loss",
            "combined_loss",
        ):

            value = loss_result.get(
                key
            )

            if torch.is_tensor(
                value
            ):

                return value

    raise TypeError(
        "Road loss function must return a Tensor or a dictionary "
        "containing a loss tensor."
    )


# =====================================================================
# OPTIMIZER
# =====================================================================

def build_optimizer(
    model: nn.Module,
    training_config: Dict[str, Any],
):
    """
    Build optimizer over trainable parameters ONLY.

    This is critical for SAM-LoRA.
    """

    trainable_parameters = [
        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    if not trainable_parameters:

        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    optimizer_name = str(
        training_config[
            "optimizer_name"
        ]
    ).lower()

    learning_rate = float(
        training_config[
            "learning_rate"
        ]
    )

    weight_decay = float(
        training_config[
            "weight_decay"
        ]
    )

    if optimizer_name in {
        "adamw",
        "adam_w",
    }:

        return torch.optim.AdamW(
            trainable_parameters,

            lr=learning_rate,

            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            trainable_parameters,

            lr=learning_rate,

            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":

        return torch.optim.SGD(
            trainable_parameters,

            lr=learning_rate,

            weight_decay=weight_decay,

            momentum=0.9,
        )

    raise ValueError(
        "Unsupported optimizer:\n"
        f"{optimizer_name}"
    )


# =====================================================================
# SCHEDULER
# =====================================================================

def build_scheduler(
    optimizer,
    training_config: Dict[str, Any],
):
    """
    Build LR scheduler.
    """

    scheduler_name = str(
        training_config.get(
            "scheduler_name",
            "cosine",
        )
    ).strip().lower()

    scheduler_config = _as_dict(
        training_config.get(
            "scheduler_config"
        )
    )

    epochs = int(
        training_config[
            "epochs"
        ]
    )

    if scheduler_name in {
        "",
        "none",
        "off",
        "disabled",
    }:

        return None

    if scheduler_name in {
        "cosine",
        "cosineannealing",
        "cosine_annealing",
        "cosineannealinglr",
    }:

        minimum_lr = float(
            _first_not_none(
                scheduler_config.get(
                    "eta_min"
                ),

                scheduler_config.get(
                    "minimum_learning_rate"
                ),

                scheduler_config.get(
                    "min_lr"
                ),

                0.0,
            )
        )

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,

            T_max=max(
                epochs,
                1,
            ),

            eta_min=minimum_lr,
        )

    if scheduler_name in {
        "step",
        "steplr",
    }:

        step_size = int(
            scheduler_config.get(
                "step_size",
                max(
                    epochs // 3,
                    1,
                ),
            )
        )

        gamma = float(
            scheduler_config.get(
                "gamma",
                0.1,
            )
        )

        return torch.optim.lr_scheduler.StepLR(
            optimizer,

            step_size=step_size,

            gamma=gamma,
        )

    if scheduler_name in {
        "plateau",
        "reduce_on_plateau",
        "reducelronplateau",
    }:

        factor = float(
            scheduler_config.get(
                "factor",
                0.5,
            )
        )

        patience = int(
            scheduler_config.get(
                "patience",
                3,
            )
        )

        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,

                mode="max",

                factor=factor,

                patience=patience,
            )
        )

    raise ValueError(
        "Unsupported scheduler:\n"
        f"{scheduler_name}"
    )


# =====================================================================
# AMP
# =====================================================================

def build_grad_scaler(
    use_amp: bool,
):
    """
    Build CUDA GradScaler with compatibility across PyTorch versions.
    """

    enabled = bool(
        use_amp
        and torch.cuda.is_available()
    )

    try:

        return torch.amp.GradScaler(
            "cuda",

            enabled=enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):

        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    """
    AMP context compatible with old/new PyTorch releases.
    """

    enabled = bool(
        enabled
        and device.type
        == "cuda"
    )

    try:

        return torch.amp.autocast(
            device_type="cuda",

            dtype=torch.float16,

            enabled=enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):

        return torch.cuda.amp.autocast(
            enabled=enabled
        )


# =====================================================================
# METRIC ACCUMULATION
# =====================================================================

class _BinaryMetricAccumulator:
    """
    Dataset-wide confusion-count accumulator.

    Metrics are calculated only after all pixels are accumulated.
    """

    def __init__(
        self,
    ):

        self.tp = 0

        self.fp = 0

        self.fn = 0

        self.tn = 0


    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> None:

        predictions = (
            probabilities
            >= float(
                threshold
            )
        )

        targets = (
            targets
            >= 0.5
        )

        predictions = predictions.bool()

        targets = targets.bool()

        self.tp += int(
            (
                predictions
                & targets
            )
            .sum()
            .item()
        )

        self.fp += int(
            (
                predictions
                & ~targets
            )
            .sum()
            .item()
        )

        self.fn += int(
            (
                ~predictions
                & targets
            )
            .sum()
            .item()
        )

        self.tn += int(
            (
                ~predictions
                & ~targets
            )
            .sum()
            .item()
        )


    def compute(
        self,
    ) -> Dict[str, float]:

        tp = float(
            self.tp
        )

        fp = float(
            self.fp
        )

        fn = float(
            self.fn
        )

        tn = float(
            self.tn
        )

        union = (
            tp
            + fp
            + fn
        )

        iou = (
            tp
            / union
            if union > 0.0
            else 1.0
        )

        precision_denominator = (
            tp
            + fp
        )

        precision = (
            tp
            / precision_denominator
            if precision_denominator > 0.0
            else (
                1.0
                if tp + fn == 0.0
                else 0.0
            )
        )

        recall_denominator = (
            tp
            + fn
        )

        recall = (
            tp
            / recall_denominator
            if recall_denominator > 0.0
            else 1.0
        )

        f1_denominator = (
            precision
            + recall
        )

        f1 = (
            2.0
            * precision
            * recall
            / f1_denominator
            if f1_denominator > 0.0
            else 0.0
        )

        total = (
            tp
            + fp
            + fn
            + tn
        )

        pixel_accuracy = (
            (
                tp
                + tn
            )
            / total
            if total > 0.0
            else 0.0
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

            "tp":
                int(
                    self.tp
                ),

            "fp":
                int(
                    self.fp
                ),

            "fn":
                int(
                    self.fn
                ),

            "tn":
                int(
                    self.tn
                ),
        }


# =====================================================================
# clDice
# =====================================================================

def _extract_cldice_number(
    value,
) -> Optional[float]:

    if value is None:

        return None

    if torch.is_tensor(
        value
    ):

        if value.numel() == 1:

            return float(
                value.detach()
                .cpu()
                .item()
            )

        return None

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        return float(
            value
        )

    if isinstance(
        value,
        dict,
    ):

        for key in (
            "cldice",
            "cl_dice",
            "score",
            "value",
        ):

            if key in value:

                result = _extract_cldice_number(
                    value[
                        key
                    ]
                )

                if result is not None:

                    return result

    return None


def _calculate_batch_cldice(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    iterations: int,
) -> Optional[float]:
    """
    Call the project's existing connectivity metric without making the
    training engine depend on one exact historical call signature.
    """

    try:

        from src.evaluation.connectivity_metrics import (
            cldice_score,
        )

    except Exception:

        return None

    predictions = (
        probabilities
        >= float(
            threshold
        )
    ).float()

    targets = (
        targets
        >= 0.5
    ).float()

    attempts = [
        lambda:
            cldice_score(
                predictions=predictions,

                targets=targets,

                iterations=int(
                    iterations
                ),
            ),

        lambda:
            cldice_score(
                prediction=predictions,

                target=targets,

                iterations=int(
                    iterations
                ),
            ),

        lambda:
            cldice_score(
                predictions,
                targets,
                int(
                    iterations
                ),
            ),

        lambda:
            cldice_score(
                predictions,
                targets,
            ),
    ]

    for attempt in attempts:

        try:

            value = attempt()

            number = _extract_cldice_number(
                value
            )

            if number is not None:

                return float(
                    number
                )

        except TypeError:

            continue

        except Exception:

            return None

    return None


# =====================================================================
# GRADIENT CLIPPING
# =====================================================================

def _apply_gradient_clipping(
    model: nn.Module,
    gradient_clipping: Dict[str, Any],
) -> Optional[float]:

    gradient_clipping = _as_dict(
        gradient_clipping
    )

    enabled = bool(
        gradient_clipping.get(
            "enabled",
            False,
        )
    )

    if not enabled:

        return None

    max_norm = float(
        gradient_clipping.get(
            "max_norm",
            gradient_clipping.get(
                "value",
                1.0,
            ),
        )
    )

    if max_norm <= 0:

        raise ValueError(
            "gradient clipping max_norm must be > 0."
        )

    norm_type = float(
        gradient_clipping.get(
            "norm_type",
            2.0,
        )
    )

    total_norm = (
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ],

            max_norm=max_norm,

            norm_type=norm_type,
        )
    )

    return float(
        total_norm.detach()
        .cpu()
        .item()
        if torch.is_tensor(
            total_norm
        )
        else total_norm
    )


# =====================================================================
# TRAIN ONE EPOCH
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler,
    use_amp: bool,
    gradient_clipping: Optional[
        Dict[str, Any]
    ] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Train one epoch and report aggregate Road metrics.
    """

    model.train()

    total_loss = 0.0

    total_samples = 0

    metrics = _BinaryMetricAccumulator()

    gradient_clipping = (
        gradient_clipping
        or {}
    )

    for batch in loader:

        (
            images,
            masks,
        ) = unpack_batch(
            batch
        )

        images = images.to(
            device,
            non_blocking=True,
        ).float()

        masks = masks.to(
            device,
            non_blocking=True,
        ).float()

        optimizer.zero_grad(
            set_to_none=True
        )

        with _autocast_context(
            device=device,

            enabled=use_amp,
        ):

            output = model(
                images
            )

            logits = extract_model_logits(
                output
            )

            loss_result = criterion(
                logits,
                masks,
            )

            loss = _loss_scalar(
                loss_result
            )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training loss detected:\n"
                f"{float(loss.detach().cpu())}"
            )

        scaler.scale(
            loss
        ).backward()

        if bool(
            _as_dict(
                gradient_clipping
            ).get(
                "enabled",
                False,
            )
        ):

            scaler.unscale_(
                optimizer
            )

            _apply_gradient_clipping(
                model=model,

                gradient_clipping=(
                    gradient_clipping
                ),
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = int(
            images.shape[
                0
            ]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_samples += batch_size

        with torch.no_grad():

            probabilities = (
                torch.sigmoid(
                    logits.detach()
                )
            )

            metrics.update(
                probabilities=probabilities,

                targets=masks,

                threshold=threshold,
            )

    if total_samples <= 0:

        raise RuntimeError(
            "Training loader produced zero samples."
        )

    metric_values = (
        metrics.compute()
    )

    return {
        "train_loss":
            float(
                total_loss
                / total_samples
            ),

        "loss":
            float(
                total_loss
                / total_samples
            ),

        **metric_values,
    }


# =====================================================================
# VALIDATE ONE EPOCH
# =====================================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True,
    calculate_cldice: bool = False,
    cldice_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Validate one epoch.

    Selection metric:
        aggregate Validation IoU
    """

    model.eval()

    total_loss = 0.0

    total_samples = 0

    metrics = _BinaryMetricAccumulator()

    cldice_sum = 0.0

    cldice_weight = 0

    for batch in loader:

        (
            images,
            masks,
        ) = unpack_batch(
            batch
        )

        images = images.to(
            device,
            non_blocking=True,
        ).float()

        masks = masks.to(
            device,
            non_blocking=True,
        ).float()

        with _autocast_context(
            device=device,

            enabled=use_amp,
        ):

            output = model(
                images
            )

            logits = extract_model_logits(
                output
            )

            loss_result = criterion(
                logits,
                masks,
            )

            loss = _loss_scalar(
                loss_result
            )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation loss detected."
            )

        probabilities = torch.sigmoid(
            logits
        )

        batch_size = int(
            images.shape[
                0
            ]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
                .item()
            )
            * batch_size
        )

        total_samples += batch_size

        metrics.update(
            probabilities=probabilities,

            targets=masks,

            threshold=threshold,
        )

        if calculate_cldice:

            batch_cldice = (
                _calculate_batch_cldice(
                    probabilities=probabilities,

                    targets=masks,

                    threshold=threshold,

                    iterations=int(
                        cldice_iterations
                    ),
                )
            )

            if batch_cldice is not None:

                cldice_sum += (
                    float(
                        batch_cldice
                    )
                    * batch_size
                )

                cldice_weight += batch_size

    if total_samples <= 0:

        raise RuntimeError(
            "Validation loader produced zero samples."
        )

    metric_values = (
        metrics.compute()
    )

    result = {
        "validation_loss":
            float(
                total_loss
                / total_samples
            ),

        "val_loss":
            float(
                total_loss
                / total_samples
            ),

        "loss":
            float(
                total_loss
                / total_samples
            ),

        **metric_values,
    }

    if calculate_cldice:

        result[
            "cldice"
        ] = (
            float(
                cldice_sum
                / cldice_weight
            )
            if cldice_weight > 0
            else None
        )

    return result


# =====================================================================
# HISTORY
# =====================================================================

def _save_history_csv(
    rows,
    path: str | Path,
) -> Path:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        return path

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        _json_safe(
                            row.get(
                                key
                            )
                        )

                    for key in fieldnames
                }
            )

    return path


# =====================================================================
# TRAIN MODEL
# =====================================================================

def train_model(
    model: nn.Module,
    context: Dict[str, Any],
    train_augmentation=None,
    validation_augmentation=None,
    device: Optional[
        str | torch.device
    ] = None,
) -> Dict[str, Any]:
    """
    Full native Road training workflow.

    Used by:
    - baseline U-Net
    - baseline DeepLabV3+
    - baseline SAM-LoRA
    - general Optuna final retraining
    - Master Optuna final retraining
    """

    # -----------------------------------------------------------------
    # Backend guard
    # -----------------------------------------------------------------

    backend = context.get(
        "model_backend",
        "pytorch",
    )

    if backend != "pytorch":

        raise RuntimeError(
            "train_model() is the native PyTorch engine.\n"
            f"Selected backend: {backend}\n\n"
            "ConnectNet / MultiTaskRoadExtractor must use "
            "src/train_arcgis.py."
        )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    general_params = (
        _general_params_from_context(
            context
        )
    )

    model_config = context[
        "model_config"
    ]

    experiment_config = context[
        "experiment_config"
    ]

    training_config = (
        resolve_training_config(
            general_params=general_params,

            model_config=model_config,
        )
    )

    epochs = int(
        training_config[
            "epochs"
        ]
    )

    resolved_device = resolve_device(
        device
    )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    split_config = _as_dict(
        context.get(
            "split_config"
        )
    )

    seed = int(
        _first_not_none(
            split_config.get(
                "seed"
            ),

            general_params.get(
                "seed"
            ),

            42,
        )
    )

    reproducibility = _as_dict(
        general_params.get(
            "reproducibility"
        )
    )

    deterministic = bool(
        reproducibility.get(
            "deterministic_training",
            False,
        )
    )

    seed_training(
        seed=seed,

        deterministic=deterministic,
    )

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    # checkpoint_folder was introduced after some Road experiment-context
    # builders were already in use. Older/baseline callers may therefore
    # provide only output_folder. Keep both generations compatible:
    #
    #   explicit checkpoint_folder -> use it
    #   missing checkpoint_folder  -> save checkpoints in output_folder
    #
    # This preserves the standard experiment layout:
    #   outputs/experiments/<experiment_id>/best_model.pth
    #   outputs/experiments/<experiment_id>/last_checkpoint.pth
    checkpoint_folder_value = (
        context.get(
            "checkpoint_folder"
        )
        or context.get(
            "output_folder"
        )
    )

    if checkpoint_folder_value is None:
        raise KeyError(
            "Training context requires context['output_folder'] "
            "or context['checkpoint_folder']."
        )

    checkpoint_folder = Path(
        checkpoint_folder_value
    )

    # Normalize the context as well so checkpoint metadata and any
    # downstream caller see one explicit resolved checkpoint location.
    context[
        "checkpoint_folder"
    ] = checkpoint_folder

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        checkpoint_folder
        / "best_model.pth"
    )

    last_checkpoint = (
        checkpoint_folder
        / "last_checkpoint.pth"
    )

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    data = build_dataloaders(
        context=context,

        train_augmentation=(
            train_augmentation
        ),

        validation_augmentation=(
            validation_augmentation
        ),
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = model.to(
        resolved_device
    )

    parameter_summary = (
        get_model_parameter_summary(
            model
        )
    )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    criterion = (
        build_loss_function(
            general_params=general_params,

            model_config=model_config,
        )
    )

    if isinstance(
        criterion,
        nn.Module,
    ):

        criterion = criterion.to(
            resolved_device
        )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = build_optimizer(
        model=model,

        training_config=(
            training_config
        ),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,

        training_config=(
            training_config
        ),
    )

    # -----------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------

    use_amp = bool(
        training_config[
            "use_amp"
        ]
        and resolved_device.type
        == "cuda"
    )

    scaler = build_grad_scaler(
        use_amp
    )

    # -----------------------------------------------------------------
    # Validation threshold
    #
    # This is ONLY the temporary epoch-selection threshold.
    #
    # Final threshold search happens separately on Validation after the
    # best model is selected.
    # -----------------------------------------------------------------

    validation_config = _as_dict(
        general_params.get(
            "validation"
        )
    )

    validation_threshold = float(
        _first_not_none(
            validation_config.get(
                "default_probability_threshold"
            ),

            validation_config.get(
                "threshold"
            ),

            0.5,
        )
    )

    # -----------------------------------------------------------------
    # Connectivity metric
    # -----------------------------------------------------------------

    calculate_validation_cldice = bool(
        _first_not_none(
            validation_config.get(
                "calculate_cldice"
            ),

            validation_config.get(
                "use_cldice"
            ),

            True,
        )
    )

    cldice_iterations = int(
        validation_config.get(
            "cldice_iterations",
            50,
        )
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    model_type = str(
        context.get(
            "model_type",
            experiment_config.get(
                "model_type",
                "unknown",
            ),
        )
    )

    print()
    print(
        "=" * 76
    )

    print(
        "ROAD NATIVE PYTORCH TRAINING"
    )

    print(
        "=" * 76
    )

    print(
        f"Experiment        : "
        f"{experiment_config.get('experiment_id')}"
    )

    print(
        f"Model             : "
        f"{model_type}"
    )

    print(
        f"Device            : "
        f"{resolved_device}"
    )

    print(
        f"Tile size         : "
        f"{experiment_config.get('tile_size')}"
    )

    print(
        f"Train samples     : "
        f"{data['train_samples']}"
    )

    print(
        f"Validation samples: "
        f"{data['validation_samples']}"
    )

    print(
        f"Batch size        : "
        f"{data['batch_size']}"
    )

    print(
        f"Epochs            : "
        f"{epochs}"
    )

    print(
        f"Optimizer         : "
        f"{training_config['optimizer_name']}"
    )

    print(
        f"Learning rate     : "
        f"{training_config['learning_rate']}"
    )

    print(
        f"Weight decay      : "
        f"{training_config['weight_decay']}"
    )

    print(
        f"Scheduler         : "
        f"{training_config['scheduler_name']}"
    )

    print(
        f"AMP               : "
        f"{'ON' if use_amp else 'OFF'}"
    )

    print(
        f"Trainable params  : "
        f"{parameter_summary['trainable_parameters']:,}"
    )

    print(
        f"Total params      : "
        f"{parameter_summary['total_parameters']:,}"
    )

    print(
        f"Epoch threshold   : "
        f"{validation_threshold:.2f}"
    )

    print(
        "Threshold search  : SEPARATE VALIDATION STAGE"
    )

    print(
        "Post-processing   : SEPARATE VALIDATION STAGE"
    )

    print(
        "Independent test  : NOT ACCESSED"
    )

    if model_type.lower() in {
        "sam_lora",
        "sam-lora",
        "samlora",
    }:

        sam_info = parameter_summary.get(
            "sam_lora",
            {},
        )

        if sam_info:

            print(
                f"SAM checkpoint    : "
                f"{sam_info.get('official_sam_checkpoint')}"
            )

            print(
                f"LoRA rank         : "
                f"{sam_info.get('lora_rank')}"
            )

            print(
                f"LoRA alpha        : "
                f"{sam_info.get('lora_alpha')}"
            )

        print(
            "Checkpoint format : trainable Road SAM parameters only"
        )

    else:

        print(
            "Checkpoint format : complete model state"
        )

    print(
        "=" * 76
    )

    # -----------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------

    history = []

    best_epoch = None

    best_validation_iou = float(
        "-inf"
    )

    best_validation_metrics = None

    training_started = time.time()

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_started = time.time()

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        train_result = train_one_epoch(
            model=model,

            loader=data[
                "train_loader"
            ],

            optimizer=optimizer,

            criterion=criterion,

            device=resolved_device,

            scaler=scaler,

            use_amp=use_amp,

            gradient_clipping=(
                training_config[
                    "gradient_clipping"
                ]
            ),

            threshold=0.5,
        )

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        validation_result = (
            validate_one_epoch(
                model=model,

                loader=data[
                    "validation_loader"
                ],

                criterion=criterion,

                device=resolved_device,

                threshold=(
                    validation_threshold
                ),

                use_amp=use_amp,

                calculate_cldice=(
                    calculate_validation_cldice
                ),

                cldice_iterations=(
                    cldice_iterations
                ),
            )
        )

        validation_iou = float(
            validation_result[
                "iou"
            ]
        )

        # -------------------------------------------------------------
        # Best model decision
        # -------------------------------------------------------------

        is_best = (
            validation_iou
            > best_validation_iou
        )

        if is_best:

            best_validation_iou = (
                validation_iou
            )

            best_epoch = int(
                epoch
            )

            best_validation_metrics = (
                deepcopy(
                    validation_result
                )
            )

            save_training_checkpoint(
                checkpoint_path=(
                    best_checkpoint
                ),

                model=model,

                context=context,

                epoch=epoch,

                validation_metrics=(
                    validation_result
                ),

                train_metrics=(
                    train_result
                ),

                optimizer=optimizer,

                scheduler=scheduler,

                scaler=scaler,

                extra={
                    "best_checkpoint":
                        True,

                    "selection_metric":
                        "validation_iou",

                    "validation_threshold_for_epoch_selection":
                        validation_threshold,

                    "final_threshold_search_performed":
                        False,

                    "postprocessing_search_performed":
                        False,

                    "independent_test_used":
                        False,
                },
            )

        # -------------------------------------------------------------
        # Scheduler
        # -------------------------------------------------------------

        if scheduler is not None:

            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):

                scheduler.step(
                    validation_iou
                )

            else:

                scheduler.step()

        # -------------------------------------------------------------
        # Last checkpoint
        # -------------------------------------------------------------

        save_training_checkpoint(
            checkpoint_path=(
                last_checkpoint
            ),

            model=model,

            context=context,

            epoch=epoch,

            validation_metrics=(
                validation_result
            ),

            train_metrics=(
                train_result
            ),

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            extra={
                "best_checkpoint":
                    False,

                "best_epoch_so_far":
                    best_epoch,

                "best_validation_iou_so_far":
                    (
                        best_validation_iou
                    ),

                "independent_test_used":
                    False,
            },
        )

        # -------------------------------------------------------------
        # History
        # -------------------------------------------------------------

        epoch_seconds = float(
            time.time()
            - epoch_started
        )

        history_row = {
            "epoch":
                int(
                    epoch
                ),

            "train_loss":
                float(
                    train_result[
                        "train_loss"
                    ]
                ),

            "train_iou":
                float(
                    train_result[
                        "iou"
                    ]
                ),

            "train_precision":
                float(
                    train_result[
                        "precision"
                    ]
                ),

            "train_recall":
                float(
                    train_result[
                        "recall"
                    ]
                ),

            "train_f1":
                float(
                    train_result[
                        "f1"
                    ]
                ),

            "validation_loss":
                float(
                    validation_result[
                        "validation_loss"
                    ]
                ),

            "validation_iou":
                validation_iou,

            "validation_precision":
                float(
                    validation_result[
                        "precision"
                    ]
                ),

            "validation_recall":
                float(
                    validation_result[
                        "recall"
                    ]
                ),

            "validation_f1":
                float(
                    validation_result[
                        "f1"
                    ]
                ),

            "validation_pixel_accuracy":
                float(
                    validation_result[
                        "pixel_accuracy"
                    ]
                ),

            "validation_cldice":
                validation_result.get(
                    "cldice"
                ),

            "learning_rate":
                current_lr,

            "epoch_seconds":
                epoch_seconds,

            "best":
                bool(
                    is_best
                ),
        }

        history.append(
            history_row
        )

        # -------------------------------------------------------------
        # Persist history after every epoch.
        # -------------------------------------------------------------

        _save_json(
            history,

            output_folder
            / "training_history.json",
        )

        _save_history_csv(
            history,

            output_folder
            / "training_history.csv",
        )

        # -------------------------------------------------------------
        # Console
        # -------------------------------------------------------------

        cldice_value = (
            validation_result.get(
                "cldice"
            )
        )

        cldice_text = (
            f" | clDice: {cldice_value * 100:.2f}%"
            if cldice_value
            is not None
            else ""
        )

        best_text = (
            " | BEST"
            if is_best
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d}"
            f" | Train Loss: {train_result['train_loss']:.4f}"
            f" | Train IoU: {train_result['iou'] * 100:.2f}%"
            f" | Val Loss: {validation_result['validation_loss']:.4f}"
            f" | Val IoU: {validation_iou * 100:.2f}%"
            f" | P: {validation_result['precision'] * 100:.2f}%"
            f" | R: {validation_result['recall'] * 100:.2f}%"
            f" | F1: {validation_result['f1'] * 100:.2f}%"
            f"{cldice_text}"
            f" | LR: {current_lr:.3e}"
            f" | Time: {epoch_seconds:.2f}s"
            f"{best_text}"
        )

    # -----------------------------------------------------------------
    # Finished
    # -----------------------------------------------------------------

    total_seconds = float(
        time.time()
        - training_started
    )

    if best_epoch is None:

        raise RuntimeError(
            "Training completed without producing a best checkpoint."
        )

    if not best_checkpoint.exists():

        raise FileNotFoundError(
            "Best checkpoint was not created:\n"
            f"{best_checkpoint}"
        )

    # -----------------------------------------------------------------
    # Training summary
    # -----------------------------------------------------------------

    checkpoint_format = (
        "road_sam_lora_trainable_v1"

        if model_type.lower()
        in {
            "sam_lora",
            "sam-lora",
            "samlora",
        }

        else "road_full_model_v2"
    )

    summary = {
        "experiment_id":
            experiment_config.get(
                "experiment_id"
            ),

        "approach":
            "roads",

        "model_type":
            model_type,

        "backend":
            "pytorch",

        "dataset_id":
            context.get(
                "dataset_id"
            ),

        "dataset_root":
            str(
                context.get(
                    "dataset_root",
                    ""
                )
            ),

        "source_type":
            experiment_config.get(
                "source_type",
                "mixed",
            ),

        "tile_size":
            int(
                experiment_config[
                    "tile_size"
                ]
            ),

        "train_samples":
            int(
                data[
                    "train_samples"
                ]
            ),

        "validation_samples":
            int(
                data[
                    "validation_samples"
                ]
            ),

        "epochs":
            int(
                epochs
            ),

        "batch_size":
            int(
                data[
                    "batch_size"
                ]
            ),

        "seed":
            int(
                seed
            ),

        "device":
            str(
                resolved_device
            ),

        "use_amp":
            bool(
                use_amp
            ),

        "optimizer":
            training_config[
                "optimizer_name"
            ],

        "learning_rate":
            float(
                training_config[
                    "learning_rate"
                ]
            ),

        "weight_decay":
            float(
                training_config[
                    "weight_decay"
                ]
            ),

        "scheduler":
            training_config[
                "scheduler_name"
            ],

        "validation_threshold_for_epoch_selection":
            float(
                validation_threshold
            ),

        "selection_metric":
            "validation_iou",

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_iou":
            float(
                best_validation_metrics[
                    "iou"
                ]
            ),

        "best_validation_iou_percent":
            float(
                best_validation_metrics[
                    "iou"
                ]
                * 100.0
            ),

        "best_validation_precision":
            float(
                best_validation_metrics[
                    "precision"
                ]
            ),

        "best_validation_recall":
            float(
                best_validation_metrics[
                    "recall"
                ]
            ),

        "best_validation_f1":
            float(
                best_validation_metrics[
                    "f1"
                ]
            ),

        "best_validation_cldice":
            best_validation_metrics.get(
                "cldice"
            ),

        "best_validation_metrics":
            deepcopy(
                best_validation_metrics
            ),

        "parameter_summary":
            parameter_summary,

        "checkpoint_format":
            checkpoint_format,

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
                output_folder
                / "training_history.json"
            ),

        "training_history_csv":
            str(
                output_folder
                / "training_history.csv"
            ),

        "training_seconds":
            total_seconds,

        "threshold_search_performed":
            False,

        "postprocessing_search_performed":
            False,

        "settings_frozen":
            False,

        "independent_final_test_used":
            False,

        "satellite_testing_zone_used":
            False,

        "road_test_label_used":
            False,
    }

    # SAM provenance.
    if model_type.lower() in {
        "sam_lora",
        "sam-lora",
        "samlora",
    }:

        summary[
            "sam_lora"
        ] = deepcopy(
            getattr(
                model,
                "road_sam_lora_build_info",
                {},
            )
        )

        summary[
            "sam_checkpoint_policy"
        ] = {
            "official_sam_base_saved_inside_best_model":
                False,

            "road_trainable_weights_saved":
                True,

            "buildings_trained_checkpoint_used":
                False,

            "buildings_checkpoint_auto_discovery":
                False,
        }

    summary_path = _save_json(
        summary,

        output_folder
        / "training_summary.json",
    )

    # -----------------------------------------------------------------
    # Final console
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "ROAD TRAINING COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best Val IoU      : "
        f"{best_validation_metrics['iou'] * 100:.4f}%"
    )

    print(
        f"Precision         : "
        f"{best_validation_metrics['precision'] * 100:.4f}%"
    )

    print(
        f"Recall            : "
        f"{best_validation_metrics['recall'] * 100:.4f}%"
    )

    print(
        f"F1                : "
        f"{best_validation_metrics['f1'] * 100:.4f}%"
    )

    if (
        best_validation_metrics.get(
            "cldice"
        )
        is not None
    ):

        print(
            f"clDice            : "
            f"{best_validation_metrics['cldice'] * 100:.4f}%"
        )

    print(
        f"Best checkpoint   : "
        f"{best_checkpoint}"
    )

    print(
        f"Checkpoint format : "
        f"{checkpoint_format}"
    )

    print(
        f"Training summary  : "
        f"{summary_path}"
    )

    print(
        "Final test used   : NO"
    )

    print(
        "=" * 76
    )

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Public result
    # -----------------------------------------------------------------

    return {
        "best_checkpoint":
            str(
                best_checkpoint
            ),

        "last_checkpoint":
            str(
                last_checkpoint
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_iou":
            float(
                best_validation_metrics[
                    "iou"
                ]
            ),

        "best_validation_metrics":
            deepcopy(
                best_validation_metrics
            ),

        "history":
            history,

        "training_summary":
            summary,

        "training_summary_path":
            str(
                summary_path
            ),

        "checkpoint_format":
            checkpoint_format,

        "independent_final_test_used":
            False,
    }
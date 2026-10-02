"""
ArcGIS Learn training backend for Approach 2 - Road Extraction.

Supported
---------
- ConnectNet
- MultiTaskRoadExtractor

This backend is intentionally separate from the native PyTorch engine.

Native engine:
    U-Net
    DeepLabV3+
    SAM-LoRA

ArcGIS Learn engine:
    ConnectNet
    MultiTaskRoadExtractor

Data policy
-----------
The same mixed-source Classified Tiles export is used.

ArcGIS Learn performs its own deterministic Train/Validation split via:

    prepare_data(
        val_split_pct=0.20,
        seed=42
    )

Important:
The ratio and seed are matched to the thesis pipeline, but exact sample
membership is NOT assumed to be identical to our native CSV-manifest
split because ArcGIS Learn controls its own splitting implementation.

Independent final-test data is never accessed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.models.connectnet import (
    build_connectnet,
    resolve_connectnet_config,
)

from src.models.multitask_road_extractor import (
    build_multitask_road_extractor,
    resolve_multitask_config,
)

from src.utils import (
    save_json,
)


# ---------------------------------------------------------------------
# ArcGIS
# ---------------------------------------------------------------------

def _get_prepare_data():
    """
    Import ArcGIS Learn only when this backend is executed.
    """

    try:

        from arcgis.learn import (
            prepare_data,
        )

    except Exception as exc:

        raise RuntimeError(
            "ArcGIS Learn could not initialize.\n\n"
            "Your standalone ArcGIS environment currently reports:\n"
            "'The Product License has not been initialized.'\n\n"
            "Run this trainer from an ArcGIS Pro Notebook where the "
            "product license is already initialized."
        ) from exc

    return prepare_data


# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------

def resolve_validation_fraction(
    context: Dict[str, Any],
) -> float:
    """
    Convert configured Validation split to ArcGIS fraction.
    """

    split = context[
        "split_config"
    ]

    value = split.get(
        "validation_percent",
        split.get(
            "validation_fraction",
            0.20,
        ),
    )

    value = float(
        value
    )

    if value > 1.0:

        value = (
            value
            / 100.0
        )

    if not (
        0.0
        < value
        < 1.0
    ):

        raise ValueError(
            "Validation fraction must be between 0 and 1."
        )

    return value


# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------

def effective_arcgis_model_config(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge experiment-specific ArcGIS architecture choices into the
    static model configuration without modifying any source JSON file.

    The shared UI stores the effective ArcGIS values in:

        experiment_config["backbone"]
        experiment_config["arcgis_mtl_model"]

    Examples
    --------
    ArcGIS Default (Hourglass):
        backbone = None
        arcgis_mtl_model = "hourglass"

    ResNet-50 choice:
        backbone = "resnet50"
        arcgis_mtl_model = "linknet"

    The model wrappers remain authoritative for validating the final
    constructor parameters.
    """

    model_config = dict(
        context.get(
            "model_config",
            {},
        )
        or {}
    )

    experiment_config = dict(
        context.get(
            "experiment_config",
            {},
        )
        or {}
    )

    architecture = dict(
        model_config.get(
            "architecture",
            {},
        )
        or {}
    )

    # ---------------------------------------------------------
    # Backbone selected in the shared UI.
    # None is intentional for Hourglass.
    # ---------------------------------------------------------

    if "backbone" in experiment_config:

        selected_backbone = (
            experiment_config.get(
                "backbone"
            )
        )

        model_config[
            "backbone"
        ] = selected_backbone

        architecture[
            "backbone"
        ] = selected_backbone

    # ---------------------------------------------------------
    # ArcGIS road architecture.
    # ---------------------------------------------------------

    selected_mtl_model = (
        experiment_config.get(
            "arcgis_mtl_model"
        )
    )

    if selected_mtl_model:

        selected_mtl_model = (
            str(
                selected_mtl_model
            )
            .strip()
            .lower()
        )

        model_config[
            "mtl_model"
        ] = selected_mtl_model

        architecture[
            "mtl_model"
        ] = selected_mtl_model

        architecture[
            "name"
        ] = selected_mtl_model

    model_config[
        "architecture"
    ] = architecture

    return model_config


def resolve_arcgis_model_training_config(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize training settings for the selected ArcGIS model.
    """

    model_type = context[
        "model_type"
    ]

    effective_config = (
        effective_arcgis_model_config(
            context
        )
    )

    if model_type == "connectnet":

        config = resolve_connectnet_config(
            effective_config
        )

    elif model_type in {
        "multitask_road_extractor",
        "multitask",
    }:

        config = resolve_multitask_config(
            effective_config
        )

    else:

        raise ValueError(
            "ArcGIS Road trainer supports only:\n"
            "- connectnet\n"
            "- multitask_road_extractor"
        )

    return config


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def prepare_arcgis_road_data(
    context: Dict[str, Any],
):
    """
    Build the real ArcGIS Learn DataBunch from Classified Tiles.
    """

    prepare_data = (
        _get_prepare_data()
    )

    dataset_root = Path(
        context[
            "dataset_root"
        ]
    )

    if not dataset_root.exists():

        raise FileNotFoundError(
            "Road training dataset does not exist:\n"
            f"{dataset_root}"
        )

    map_file = (
        dataset_root
        / "map.txt"
    )

    if not map_file.exists():

        raise FileNotFoundError(
            "ArcGIS Classified Tiles map.txt was not found:\n"
            f"{map_file}\n\n"
            "prepare_data() relies on map.txt to infer this exported "
            "dataset format."
        )

    training_config = (
        resolve_arcgis_model_training_config(
            context
        )
    )

    tile_size = int(
        context[
            "experiment_config"
        ][
            "tile_size"
        ]
    )

    validation_fraction = (
        resolve_validation_fraction(
            context
        )
    )

    seed = int(
        context[
            "split_config"
        ][
            "seed"
        ]
    )

    working_dir = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "arcgis_working"
    )

    working_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ARCGIS LEARN DATA PREPARATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Dataset          : "
        f"{dataset_root}"
    )

    print(
        f"Tile size        : "
        f"{tile_size}"
    )

    print(
        f"Batch size       : "
        f"{training_config['batch_size']}"
    )

    print(
        f"Validation       : "
        f"{validation_fraction * 100:.1f}%"
    )

    print(
        f"Seed             : "
        f"{seed}"
    )

    print(
        "Dataset type     : inferred from map.txt"
    )

    print(
        "Sources          : aerial + satellite + drone"
    )

    print(
        "Final test       : NOT ACCESSED"
    )

    print(
        "=" * 72
    )

    data = prepare_data(
        path=str(
            dataset_root
        ),

        chip_size=tile_size,

        val_split_pct=(
            validation_fraction
        ),

        batch_size=int(
            training_config[
                "batch_size"
            ]
        ),

        transforms=None,

        seed=seed,

        dataset_type=None,

        resize_to=None,

        working_dir=str(
            working_dir
        ),
    )

    return data


# ---------------------------------------------------------------------
# Build selected model
# ---------------------------------------------------------------------

def build_arcgis_road_model(
    context: Dict[str, Any],
    data,
):
    """
    Build ConnectNet or MultiTaskRoadExtractor.
    """

    model_type = context[
        "model_type"
    ]

    effective_config = (
        effective_arcgis_model_config(
            context
        )
    )

    if model_type == "connectnet":

        return build_connectnet(
            data=data,

            model_config=effective_config,
        )

    if model_type in {
        "multitask_road_extractor",
        "multitask",
    }:

        return (
            build_multitask_road_extractor(
                data=data,

                model_config=effective_config,
            )
        )

    raise ValueError(
        f"Unsupported ArcGIS model: {model_type}"
    )


# ---------------------------------------------------------------------
# Metrics / history
# ---------------------------------------------------------------------

def get_available_metrics(
    model,
):
    """
    Get Esri model metric names without assuming a specific version.
    """

    try:

        metrics = (
            model.available_metrics
        )

        if callable(
            metrics
        ):

            metrics = metrics()

        return [
            str(
                value
            )
            for value
            in metrics
        ]

    except Exception:

        return []


def choose_monitor_metric(
    model,
) -> str:
    """
    Prefer Road-specific IoU when ArcGIS exposes it.
    """

    metrics = get_available_metrics(
        model
    )

    normalized = {
        value.lower():
            value
        for value
        in metrics
    }

    for candidate in (
        "miou",
        "iou",
        "dice",
        "accuracy",
    ):

        if candidate in normalized:

            return normalized[
                candidate
            ]

    # valid_loss is supported by ArcGISModel.fit even though it may not
    # appear in available_metrics.
    return "valid_loss"


def extract_arcgis_history(
    model,
):
    """
    Best-effort extraction of fastai recorder history.

    Failure to extract history must never break training.
    """

    try:

        learner = getattr(
            model,
            "learn",
            None,
        )

        if learner is None:

            return None

        recorder = getattr(
            learner,
            "recorder",
            None,
        )

        if recorder is None:

            return None

        values = getattr(
            recorder,
            "values",
            None,
        )

        metric_names = getattr(
            recorder,
            "metric_names",
            None,
        )

        if values is None:

            return None

        return {
            "metric_names":
                (
                    [
                        str(
                            item
                        )
                        for item
                        in metric_names
                    ]
                    if metric_names
                    is not None
                    else None
                ),

            "values": [
                [
                    float(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                        ),
                    )
                    else str(
                        value
                    )

                    for value
                    in row
                ]

                for row
                in values
            ],
        }

    except Exception:

        return None


# ---------------------------------------------------------------------
# Validation mIoU
# ---------------------------------------------------------------------

def evaluate_arcgis_miou(
    model,
):
    """
    Run Esri's native validation mIOU when exposed.
    """

    if not hasattr(
        model,
        "mIOU",
    ):

        return None

    try:

        result = model.mIOU()

        if isinstance(
            result,
            dict,
        ):

            return {
                str(
                    key
                ):
                    (
                        float(
                            value
                        )
                        if isinstance(
                            value,
                            (
                                int,
                                float,
                            ),
                        )
                        else str(
                            value
                        )
                    )

                for (
                    key,
                    value,
                ) in result.items()
            }

        try:

            return float(
                result
            )

        except Exception:

            return str(
                result
            )

    except Exception as exc:

        return {
            "error":
                str(
                    exc
                )
        }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_arcgis_road_model(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train the selected ArcGIS Learn Road model.
    """

    if (
        context[
            "model_backend"
        ]
        != "arcgis_learn"
    ):

        raise RuntimeError(
            "train_arcgis_road_model() requires backend='arcgis_learn'."
        )

    training_config = (
        resolve_arcgis_model_training_config(
            context
        )
    )

    data = prepare_arcgis_road_data(
        context
    )

    print()
    print(
        "Building ArcGIS Learn model..."
    )

    model = build_arcgis_road_model(
        context=context,

        data=data,
    )

    available_metrics = (
        get_available_metrics(
            model
        )
    )

    monitor = choose_monitor_metric(
        model
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ARCGIS ROAD TRAINING"
    )

    print(
        "=" * 72
    )

    model_spec = context.get(
        "model_spec"
    )

    display_name = getattr(
        model_spec,
        "display_name",
        None,
    )

    if not display_name:
        display_name = (
            "ConnectNet"
            if context[
                "model_type"
            ] == "connectnet"
            else "MultiTaskRoadExtractor"
        )

    print(
        f"Model            : "
        f"{display_name}"
    )

    print(
        f"Architecture     : "
        f"{training_config['mtl_model']}"
    )

    print(
        f"Backbone         : "
        f"{training_config['backbone']}"
    )

    print(
        f"Epochs           : "
        f"{training_config['epochs']}"
    )

    print(
        f"Learning rate    : "
        f"{training_config['learning_rate']}"
    )

    print(
        f"Weight decay     : "
        f"{training_config['weight_decay']}"
    )

    print(
        f"Monitor          : "
        f"{monitor}"
    )

    print(
        f"Metrics          : "
        f"{available_metrics}"
    )

    print(
        "Checkpoint best  : YES"
    )

    print(
        "Early stopping   : NO"
    )

    print(
        "Final test       : NOT ACCESSED"
    )

    print(
        "=" * 72
    )

    # ArcGISModel.fit supports checkpointing and reloads the best
    # monitored checkpoint at the end of training.
    fit_result = model.fit(
        epochs=int(
            training_config[
                "epochs"
            ]
        ),

        lr=float(
            training_config[
                "learning_rate"
            ]
        ),

        wd=float(
            training_config[
                "weight_decay"
            ]
        ),

        one_cycle=True,

        early_stopping=False,

        checkpoint=True,

        tensorboard=False,

        monitor=monitor,
    )

    validation_miou = (
        evaluate_arcgis_miou(
            model
        )
    )

    # --------------------------------------------------------------
    # Save final/best-loaded Esri package
    # --------------------------------------------------------------

    save_directory = (
        Path(
            context[
                "checkpoint_folder"
            ]
        )
        / "arcgis_model"
    )

    save_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_model_path = model.save(
        str(
            save_directory
        ),

        framework="PyTorch",

        publish=False,

        compute_metrics=True,

        save_optimizer=False,

        save_inference_file=True,
    )

    history = extract_arcgis_history(
        model
    )

    summary = {
        "experiment_id":
            context[
                "experiment_config"
            ][
                "experiment_id"
            ],

        "model_type":
            context[
                "model_type"
            ],

        "backend":
            "arcgis_learn",

        "arcgis_class":
            (
                "ConnectNet"
                if context[
                    "model_type"
                ]
                == "connectnet"
                else "MultiTaskRoadExtractor"
            ),

        "dataset_id":
            context[
                "dataset_id"
            ],

        "dataset_root":
            str(
                context[
                    "dataset_root"
                ]
            ),

        "source_type":
            "mixed",

        "tile_size":
            int(
                context[
                    "experiment_config"
                ][
                    "tile_size"
                ]
            ),

        "split": {
            "engine":
                "arcgis.learn.prepare_data",

            "validation_fraction":
                float(
                    resolve_validation_fraction(
                        context
                    )
                ),

            "seed":
                int(
                    context[
                        "split_config"
                    ][
                        "seed"
                    ]
                ),

            "exact_native_manifest_membership_assumed":
                False,
        },

        "training": {
            "epochs":
                int(
                    training_config[
                        "epochs"
                    ]
                ),

            "batch_size":
                int(
                    training_config[
                        "batch_size"
                    ]
                ),

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

            "monitor":
                monitor,

            "checkpoint_best":
                True,

            "early_stopping":
                False,

            "fit_result":
                (
                    bool(
                        fit_result
                    )
                    if isinstance(
                        fit_result,
                        bool,
                    )
                    else str(
                        fit_result
                    )
                ),
        },

        "model_parameters": {
            "mtl_model":
                training_config[
                    "mtl_model"
                ],

            "backbone":
                training_config[
                    "backbone"
                ],

            "gaussian_thresh":
                training_config[
                    "gaussian_thresh"
                ],

            "orient_bin_size":
                training_config[
                    "orient_bin_size"
                ],

            "orient_theta":
                training_config[
                    "orient_theta"
                ],
        },

        "available_metrics":
            available_metrics,

        "validation_miou":
            validation_miou,

        "saved_model":
            str(
                saved_model_path
            ),

        "history":
            history,

        "selection_dataset":
            "validation",

        "independent_test_used":
            False,

        "satellite_testing_zone_used":
            False,

        "road_test_label_used":
            False,
    }

    summary_path = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "arcgis_training_summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    summary[
        "summary_json"
    ] = str(
        summary_path
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ARCGIS ROAD TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Saved model      : "
        f"{saved_model_path}"
    )

    print(
        f"Validation mIoU  : "
        f"{validation_miou}"
    )

    print(
        "Final test used  : NO"
    )

    print(
        "=" * 72
    )

    return summary
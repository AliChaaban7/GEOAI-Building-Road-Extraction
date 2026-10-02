"""
Validation-only probability-threshold search for Road Extraction.

Purpose
-------
Select the probability threshold that converts model probabilities into
binary Road masks.

Scientific policy
-----------------
Threshold selection is performed ONLY on Validation data.

The selected threshold is then frozen before:

    Aerial final test
    Satellite final test
    Drone final test

Final-test Ground Truth is never accessed by this module.

Primary objective
-----------------
Validation IoU

Secondary reported metrics
--------------------------
- Precision
- Recall
- F1
- clDice for the selected threshold

Important
---------
The model is evaluated only once per validation batch. The same logits
are then evaluated against all threshold candidates. This avoids running
the neural network repeatedly for each threshold.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.data import RoadSegmentationDataset

from src.evaluation.connectivity_metrics import (
    RunningClDice,
)

from src.evaluation.metrics import (
    RunningBinarySegmentationMetrics,
)

from src.inference import (
    restore_model_from_checkpoint,
)

from src.train import (
    autocast_context,
    resolve_training_config,
)

from src.utils import (
    save_json,
)


# ---------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------

def resolve_threshold_search_config(
    general_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve threshold-search settings from general_params.json.

    Supported field aliases allow the configuration file to remain
    readable even if naming changed slightly during development.

    Default search:
        0.30 → 0.70
        step 0.05
    """

    config = dict(
        general_params.get(
            "threshold_search",
            {}
        )
    )

    start = float(
        config.get(
            "start",
            config.get(
                "minimum",
                config.get(
                    "min_threshold",
                    config.get(
                        "threshold_min",
                        0.30,
                    ),
                ),
            ),
        )
    )

    end = float(
        config.get(
            "end",
            config.get(
                "maximum",
                config.get(
                    "max_threshold",
                    config.get(
                        "threshold_max",
                        0.70,
                    ),
                ),
            ),
        )
    )

    step = float(
        config.get(
            "step",
            config.get(
                "threshold_step",
                0.05,
            ),
        )
    )

    objective = str(
        config.get(
            "objective",
            config.get(
                "selection_metric",
                "iou",
            ),
        )
    ).lower()

    if not (
        0.0 <= start <= 1.0
    ):
        raise ValueError(
            f"Threshold-search start must be within [0,1]. "
            f"Received: {start}"
        )

    if not (
        0.0 <= end <= 1.0
    ):
        raise ValueError(
            f"Threshold-search end must be within [0,1]. "
            f"Received: {end}"
        )

    if end < start:
        raise ValueError(
            "Threshold-search end cannot be smaller than start."
        )

    if step <= 0:
        raise ValueError(
            "Threshold-search step must be greater than zero."
        )

    if objective not in {
        "iou",
        "f1",
        "precision",
        "recall",
    }:
        raise ValueError(
            "Unsupported threshold-search objective: "
            f"{objective!r}"
        )

    return {
        "start":
            start,

        "end":
            end,

        "step":
            step,

        "objective":
            objective,

        "validation_only":
            True,
    }


def generate_threshold_candidates(
    start: float,
    end: float,
    step: float,
) -> List[float]:
    """
    Generate deterministic threshold candidates.

    Example
    -------
    start = 0.30
    end   = 0.70
    step  = 0.05

    Returns
    -------
    [
        0.30,
        0.35,
        0.40,
        ...
        0.70
    ]
    """

    start = float(
        start
    )

    end = float(
        end
    )

    step = float(
        step
    )

    if step <= 0:
        raise ValueError(
            "step must be > 0."
        )

    thresholds = []

    current = start

    tolerance = 1e-10

    while current <= (
        end + tolerance
    ):

        thresholds.append(
            round(
                float(
                    current
                ),
                10,
            )
        )

        current += step

    if not thresholds:
        raise RuntimeError(
            "Threshold search generated no candidates."
        )

    return thresholds


# ---------------------------------------------------------------------
# Validation DataLoader
# ---------------------------------------------------------------------

def build_threshold_validation_loader(
    context: Dict[str, Any],
) -> DataLoader:
    """
    Build the Validation DataLoader used for threshold search.

    No augmentation is applied.
    """

    general_params = context[
        "general_params"
    ]

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

    data_loading = general_params.get(
        "data_loading",
        {}
    )

    dataset = RoadSegmentationDataset(
        manifest_csv=context[
            "validation_manifest"
        ],

        augmentation=None,

        expected_tile_size=int(
            experiment_config[
                "tile_size"
            ]
        ),

        # Mandatory numerical image scaling:
        # uint8 0-255 -> float 0-1.
        normalize=True,

        return_metadata=False,
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            8,
        )
    )

    num_workers = int(
        data_loading.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        data_loading.get(
            "pin_memory",
            True,
        )
    )

    persistent_workers = bool(
        data_loading.get(
            "persistent_workers",
            False,
        )
    )

    if num_workers <= 0:
        persistent_workers = False

    return DataLoader(
        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,

        persistent_workers=(
            persistent_workers
        ),

        drop_last=False,
    )


# ---------------------------------------------------------------------
# Threshold ranking
# ---------------------------------------------------------------------

def select_best_threshold(
    results: Sequence[
        Dict[str, Any]
    ],
    objective: str = "iou",
) -> Dict[str, Any]:
    """
    Select the best Validation threshold.

    Ranking
    -------
    1. Requested objective, normally IoU.
    2. F1.
    3. Recall.
    4. Threshold nearest to 0.5.

    The extra criteria only provide deterministic tie-breaking.
    """

    if not results:
        raise ValueError(
            "Threshold-search results are empty."
        )

    objective = str(
        objective
    ).lower()

    if objective not in {
        "iou",
        "f1",
        "precision",
        "recall",
    }:
        raise ValueError(
            f"Unsupported objective: {objective}"
        )

    ranked = sorted(
        results,

        key=lambda row: (
            float(
                row[
                    objective
                ]
            ),
            float(
                row.get(
                    "f1",
                    0.0,
                )
            ),
            float(
                row.get(
                    "recall",
                    0.0,
                )
            ),
            -abs(
                float(
                    row[
                        "threshold"
                    ]
                )
                - 0.5
            ),
        ),

        reverse=True,
    )

    return dict(
        ranked[
            0
        ]
    )


# ---------------------------------------------------------------------
# Core threshold search
# ---------------------------------------------------------------------

@torch.no_grad()
def search_thresholds(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    thresholds: Sequence[float],
    device: torch.device,
    objective: str = "iou",
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all threshold candidates on Validation data.

    Neural-network inference occurs once per batch.

    Each batch's logits are reused across all threshold candidates.
    """

    if not thresholds:
        raise ValueError(
            "At least one threshold candidate is required."
        )

    model = model.to(
        device
    )

    model.eval()

    use_amp = bool(
        use_amp
        and device.type == "cuda"
    )

    meters = {
        float(
            threshold
        ):
            RunningBinarySegmentationMetrics(
                threshold=float(
                    threshold
                ),
                from_logits=True,
            )

        for threshold
        in thresholds
    }

    sample_count = 0

    for (
        images,
        masks,
    ) in validation_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with autocast_context(
            use_amp
        ):

            logits = model(
                images
            )

        if logits.shape != masks.shape:

            raise RuntimeError(
                "Threshold-search prediction/target shape mismatch.\n"
                f"Logits: {tuple(logits.shape)}\n"
                f"Masks : {tuple(masks.shape)}"
            )

        for meter in meters.values():

            meter.update(
                prediction=logits,
                target=masks,
            )

        sample_count += int(
            images.shape[
                0
            ]
        )

    results = []

    for threshold in thresholds:

        threshold = float(
            threshold
        )

        metrics = meters[
            threshold
        ].compute()

        results.append(
            {
                "threshold":
                    threshold,

                "iou":
                    float(
                        metrics[
                            "iou"
                        ]
                    ),

                "precision":
                    float(
                        metrics[
                            "precision"
                        ]
                    ),

                "recall":
                    float(
                        metrics[
                            "recall"
                        ]
                    ),

                "f1":
                    float(
                        metrics[
                            "f1"
                        ]
                    ),

                "pixel_accuracy":
                    float(
                        metrics[
                            "pixel_accuracy"
                        ]
                    ),

                "true_positive":
                    int(
                        metrics[
                            "true_positive"
                        ]
                    ),

                "false_positive":
                    int(
                        metrics[
                            "false_positive"
                        ]
                    ),

                "false_negative":
                    int(
                        metrics[
                            "false_negative"
                        ]
                    ),

                "true_negative":
                    int(
                        metrics[
                            "true_negative"
                        ]
                    ),
            }
        )

    best = select_best_threshold(
        results=results,
        objective=objective,
    )

    return {
        "results":
            results,

        "best":
            best,

        "objective":
            str(
                objective
            ),

        "validation_samples":
            int(
                sample_count
            ),
    }


# ---------------------------------------------------------------------
# Selected-threshold clDice
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_selected_threshold_cldice(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool = True,
    skeleton_iterations: int = 50,
) -> Dict[str, float]:
    """
    Calculate Road-connectivity metrics once for the selected threshold.

    clDice does not determine the threshold in the default workflow.
    It is reported as a secondary structural metric.
    """

    model = model.to(
        device
    )

    model.eval()

    use_amp = bool(
        use_amp
        and device.type == "cuda"
    )

    meter = RunningClDice(
        threshold=float(
            threshold
        ),

        from_logits=True,

        skeleton_iterations=int(
            skeleton_iterations
        ),
    )

    for (
        images,
        masks,
    ) in validation_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with autocast_context(
            use_amp
        ):

            logits = model(
                images
            )

        meter.update(
            prediction=logits,
            target=masks,
        )

    result = meter.compute()

    return {
        "cldice":
            float(
                result[
                    "cldice"
                ]
            ),

        "topology_precision":
            float(
                result[
                    "topology_precision"
                ]
            ),

        "topology_sensitivity":
            float(
                result[
                    "topology_sensitivity"
                ]
            ),
    }


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_threshold_search_results(
    search_result: Dict[str, Any],
    output_folder: str | Path,
) -> Dict[str, Path]:
    """
    Save threshold-search results as CSV + JSON.
    """

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_folder
        / "threshold_search_results.csv"
    )

    json_path = (
        output_folder
        / "threshold_search_results.json"
    )

    selected_path = (
        output_folder
        / "selected_threshold.json"
    )

    rows = search_result[
        "results"
    ]

    if rows:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    save_json(
        search_result,
        json_path,
    )

    selected = {
        "selected_threshold":
            float(
                search_result[
                    "best"
                ][
                    "threshold"
                ]
            ),

        "selection_dataset":
            "validation",

        "selection_metric":
            str(
                search_result[
                    "objective"
                ]
            ),

        "validation_iou":
            float(
                search_result[
                    "best"
                ][
                    "iou"
                ]
            ),

        "validation_precision":
            float(
                search_result[
                    "best"
                ][
                    "precision"
                ]
            ),

        "validation_recall":
            float(
                search_result[
                    "best"
                ][
                    "recall"
                ]
            ),

        "validation_f1":
            float(
                search_result[
                    "best"
                ][
                    "f1"
                ]
            ),

        "validation_cldice":
            (
                float(
                    search_result[
                        "selected_threshold_connectivity"
                    ][
                        "cldice"
                    ]
                )
                if (
                    "selected_threshold_connectivity"
                    in search_result
                )
                else None
            ),

        "frozen_before_final_test":
            True,

        "final_test_used":
            False,
    }

    save_json(
        selected,
        selected_path,
    )

    return {
        "csv":
            csv_path,

        "json":
            json_path,

        "selected_threshold":
            selected_path,
    }


# ---------------------------------------------------------------------
# Complete experiment threshold search
# ---------------------------------------------------------------------

def run_threshold_search(
    context: Dict[str, Any],
    checkpoint_path: Optional[
        str | Path
    ] = None,
    device: Optional[
        str | torch.device
    ] = None,
) -> Dict[str, Any]:
    """
    Run the complete Validation-only threshold search.
    """

    if (
        context[
            "model_backend"
        ]
        != "pytorch"
    ):

        raise RuntimeError(
            "Native threshold-search engine currently supports "
            "PyTorch Road models only.\n"
            f"Model: {context['model_type']}\n"
            f"Backend: {context['model_backend']}"
        )

    if checkpoint_path is None:

        checkpoint_path = (
            Path(
                context[
                    "checkpoint_folder"
                ]
            )
            / "best_model.pth"
        )

    (
        model,
        checkpoint,
        resolved_device,
    ) = restore_model_from_checkpoint(
        checkpoint_path=checkpoint_path,

        context=context,

        device=device,
    )

    search_config = (
        resolve_threshold_search_config(
            context[
                "general_params"
            ]
        )
    )

    thresholds = (
        generate_threshold_candidates(
            start=search_config[
                "start"
            ],

            end=search_config[
                "end"
            ],

            step=search_config[
                "step"
            ],
        )
    )

    validation_loader = (
        build_threshold_validation_loader(
            context
        )
    )

    model_training = context[
        "model_config"
    ].get(
        "training",
        {},
    )

    use_amp = bool(
        model_training.get(
            "use_amp",
            True,
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD VALIDATION THRESHOLD SEARCH"
    )

    print(
        "=" * 72
    )

    print(
        f"Experiment       : "
        f"{context['experiment_config']['experiment_id']}"
    )

    print(
        f"Model            : "
        f"{context['model_spec'].display_name}"
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Search range     : "
        f"{search_config['start']:.2f} "
        f"→ {search_config['end']:.2f}"
    )

    print(
        f"Step             : "
        f"{search_config['step']:.2f}"
    )

    print(
        f"Candidates       : "
        f"{len(thresholds)}"
    )

    print(
        f"Objective        : "
        f"{search_config['objective']}"
    )

    print(
        "Dataset          : VALIDATION ONLY"
    )

    print(
        "Final test used  : NO"
    )

    print(
        "=" * 72
    )

    search_result = search_thresholds(
        model=model,

        validation_loader=validation_loader,

        thresholds=thresholds,

        device=resolved_device,

        objective=search_config[
            "objective"
        ],

        use_amp=use_amp,
    )

    best_threshold = float(
        search_result[
            "best"
        ][
            "threshold"
        ]
    )

    # --------------------------------------------------------------
    # Secondary connectivity evaluation for selected threshold
    # --------------------------------------------------------------

    connectivity = (
        evaluate_selected_threshold_cldice(
            model=model,

            validation_loader=(
                validation_loader
            ),

            device=resolved_device,

            threshold=best_threshold,

            use_amp=use_amp,

            skeleton_iterations=50,
        )
    )

    search_result[
        "selected_threshold_connectivity"
    ] = connectivity

    search_result[
        "experiment_id"
    ] = context[
        "experiment_config"
    ][
        "experiment_id"
    ]

    search_result[
        "model_type"
    ] = context[
        "model_type"
    ]

    search_result[
        "checkpoint"
    ] = str(
        checkpoint_path
    )

    search_result[
        "checkpoint_epoch"
    ] = checkpoint.get(
        "epoch"
    )

    search_result[
        "selection_dataset"
    ] = "validation"

    search_result[
        "final_test_used"
    ] = False

    search_result[
        "frozen_before_final_test"
    ] = True

    output_folder = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "threshold_search"
    )

    paths = save_threshold_search_results(
        search_result=search_result,

        output_folder=output_folder,
    )

    search_result[
        "output_files"
    ] = {
        key: str(
            value
        )
        for key, value
        in paths.items()
    }

    print()
    print(
        "-" * 72
    )

    print(
        f"Selected threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation IoU      : "
        f"{search_result['best']['iou'] * 100:.4f}%"
    )

    print(
        f"Validation F1       : "
        f"{search_result['best']['f1'] * 100:.4f}%"
    )

    print(
        f"Validation clDice   : "
        f"{connectivity['cldice'] * 100:.4f}%"
    )

    print(
        "Threshold frozen    : YES"
    )

    print(
        "Final test used      : NO"
    )

    print(
        "-" * 72
    )

    return search_result
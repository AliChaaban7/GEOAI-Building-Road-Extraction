"""
Validation-only Road post-processing search.

Purpose
-------
Search Road-mask post-processing parameters on Validation data only.

Search dimensions
-----------------
- minimum component size
- closing radius
- bridge-gap radius
- maximum hole size

Scientific policy
-----------------
1. Model checkpoint is already selected from Validation.
2. Probability threshold is already selected from Validation.
3. This module freezes that threshold.
4. Validation masks are predicted once.
5. Candidate post-processing configurations are evaluated on those masks.
6. Best post-processing settings are frozen before final testing.

Final aerial, satellite, and drone test data is never used here.

Primary objective
-----------------
IoU

Tie-break
---------
clDice
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import RoadSegmentationDataset
from src.evaluation.connectivity_metrics import cldice_score
from src.evaluation.metrics import (
    BinaryConfusionCounts,
    metrics_from_counts,
)
from src.inference import restore_model_from_checkpoint
from src.postprocessing.road_mask import (
    RoadPostprocessingConfig,
    postprocess_road_mask,
)
from src.train import (
    autocast_context,
    resolve_training_config,
)
from src.utils import (
    load_optional_config,
    save_json,
)


# ---------------------------------------------------------------------
# Search-space resolution
# ---------------------------------------------------------------------

def _list_from_any(
    value,
    default: Sequence[int],
) -> List[int]:
    """
    Normalize a config value into a list of integers.
    """

    if value is None:
        return [
            int(item)
            for item in default
        ]

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            int(item)
            for item in value
        ]

    return [
        int(
            value
        )
    ]


def resolve_postprocessing_search_space(
    general_params: Dict[str, Any],
    optional_config: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, List[int]]:
    """
    Resolve the Road post-processing candidate grid.

    Defaults
    --------
    component:
        0, 10, 25, 50, 75, 100, 150, 200

    closing:
        0, 1, 2, 3

    holes:
        0, 10, 25, 50, 100, 150

    bridge:
        0, 1, 2, 3
    """

    shared = dict(
        general_params.get(
            "road_postprocessing",
            general_params.get(
                "postprocessing",
                {},
            ),
        )
    )

    optional_config = (
        optional_config
        or {}
    )

    search_space = dict(
        optional_config.get(
            "search_space",
            {}
        )
    )

    if not search_space:
        search_space = dict(
            optional_config.get(
                "grid",
                {}
            )
        )

    component_sizes = _list_from_any(
        search_space.get(
            "minimum_component_size",
            search_space.get(
                "component_sizes",
                shared.get(
                    "component_sizes",
                    [
                        0,
                        10,
                        25,
                        50,
                        75,
                        100,
                        150,
                        200,
                    ],
                ),
            ),
        ),
        [
            0,
            10,
            25,
            50,
            75,
            100,
            150,
            200,
        ],
    )

    closing_radii = _list_from_any(
        search_space.get(
            "closing_radius",
            search_space.get(
                "closing_radii",
                shared.get(
                    "closing_radii",
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                ),
            ),
        ),
        [
            0,
            1,
            2,
            3,
        ],
    )

    hole_sizes = _list_from_any(
        search_space.get(
            "maximum_hole_size",
            search_space.get(
                "hole_sizes",
                shared.get(
                    "hole_sizes",
                    [
                        0,
                        10,
                        25,
                        50,
                        100,
                        150,
                    ],
                ),
            ),
        ),
        [
            0,
            10,
            25,
            50,
            100,
            150,
        ],
    )

    bridge_radii = _list_from_any(
        search_space.get(
            "bridge_gap_radius",
            search_space.get(
                "bridge_gap_radii",
                shared.get(
                    "bridge_gap_radii",
                    [
                        0,
                        1,
                        2,
                        3,
                    ],
                ),
            ),
        ),
        [
            0,
            1,
            2,
            3,
        ],
    )

    return {
        "minimum_component_sizes":
            component_sizes,

        "closing_radii":
            closing_radii,

        "maximum_hole_sizes":
            hole_sizes,

        "bridge_gap_radii":
            bridge_radii,
    }


def generate_postprocessing_candidates(
    search_space: Dict[
        str,
        Sequence[int]
    ],
) -> List[RoadPostprocessingConfig]:
    """
    Generate all post-processing parameter combinations.
    """

    candidates = []

    for (
        component_size,
        closing_radius,
        bridge_radius,
        hole_size,
    ) in itertools.product(
        search_space[
            "minimum_component_sizes"
        ],

        search_space[
            "closing_radii"
        ],

        search_space[
            "bridge_gap_radii"
        ],

        search_space[
            "maximum_hole_sizes"
        ],
    ):

        candidate = RoadPostprocessingConfig(
            minimum_component_size=int(
                component_size
            ),

            closing_radius=int(
                closing_radius
            ),

            bridge_gap_radius=int(
                bridge_radius
            ),

            maximum_hole_size=int(
                hole_size
            ),
        )

        candidate.validate()

        candidates.append(
            candidate
        )

    return candidates


# ---------------------------------------------------------------------
# Validation loader
# ---------------------------------------------------------------------

def build_postprocess_validation_loader(
    context: Dict[str, Any],
) -> DataLoader:
    """
    Build non-augmented Validation DataLoader.
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
        {},
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
# Cache validation predictions once
# ---------------------------------------------------------------------

@torch.no_grad()
def cache_validation_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run the neural network ONCE and cache:

        raw thresholded prediction
        Ground Truth

    as NumPy arrays.

    These arrays are then reused for every post-processing candidate.
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

    model = model.to(
        device
    )

    model.eval()

    use_amp = bool(
        use_amp
        and device.type == "cuda"
    )

    samples = []

    for (
        images,
        masks,
    ) in loader:

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

        probabilities = torch.sigmoid(
            logits
        )

        binary_predictions = (
            probabilities
            >= threshold
        )

        prediction_np = (
            binary_predictions
            .detach()
            .cpu()
            .numpy()
        )

        masks_np = (
            (
                masks > 0.5
            )
            .detach()
            .cpu()
            .numpy()
        )

        batch_size = int(
            prediction_np.shape[
                0
            ]
        )

        for index in range(
            batch_size
        ):

            prediction_mask = (
                prediction_np[
                    index,
                    0
                ]
                .astype(
                    np.uint8
                )
            )

            target_mask = (
                masks_np[
                    index,
                    0
                ]
                .astype(
                    np.uint8
                )
            )

            samples.append(
                (
                    prediction_mask,
                    target_mask,
                )
            )

    return samples


# ---------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------

def _confusion_from_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
) -> BinaryConfusionCounts:
    """
    Pixel confusion counts for NumPy masks.
    """

    prediction = (
        np.asarray(
            prediction
        )
        > 0
    )

    target = (
        np.asarray(
            target
        )
        > 0
    )

    if (
        prediction.shape
        != target.shape
    ):
        raise ValueError(
            "Post-processing prediction/target shape mismatch.\n"
            f"Prediction: {prediction.shape}\n"
            f"Target: {target.shape}"
        )

    tp = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            np.logical_not(
                target
            ),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(
                prediction
            ),
            target,
        ).sum()
    )

    tn = int(
        np.logical_and(
            np.logical_not(
                prediction
            ),
            np.logical_not(
                target
            ),
        ).sum()
    )

    return BinaryConfusionCounts(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )


def evaluate_postprocessing_candidate(
    samples: Sequence[
        Tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    config: RoadPostprocessingConfig,
    calculate_cldice: bool = True,
    skeleton_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate one post-processing configuration over all Validation masks.
    """

    total_counts = (
        BinaryConfusionCounts()
    )

    cldice_values = []

    topology_precision_values = []

    topology_sensitivity_values = []

    for (
        raw_prediction,
        target,
    ) in samples:

        processed = (
            postprocess_road_mask(
                mask=raw_prediction,
                config=config,
            )
        )

        counts = (
            _confusion_from_numpy(
                prediction=processed,
                target=target,
            )
        )

        total_counts.update(
            counts
        )

        if calculate_cldice:

            prediction_tensor = (
                torch.from_numpy(
                    processed
                )
                .float()
                .unsqueeze(
                    0
                )
                .unsqueeze(
                    0
                )
            )

            target_tensor = (
                torch.from_numpy(
                    target
                )
                .float()
                .unsqueeze(
                    0
                )
                .unsqueeze(
                    0
                )
            )

            connectivity = cldice_score(
                prediction=prediction_tensor,

                target=target_tensor,

                threshold=0.5,

                from_logits=False,

                skeleton_iterations=(
                    skeleton_iterations
                ),
            )

            cldice_values.append(
                float(
                    connectivity[
                        "cldice"
                    ]
                )
            )

            topology_precision_values.append(
                float(
                    connectivity[
                        "topology_precision"
                    ]
                )
            )

            topology_sensitivity_values.append(
                float(
                    connectivity[
                        "topology_sensitivity"
                    ]
                )
            )

    metrics = metrics_from_counts(
        total_counts
    )

    result = {
        **config.to_dict(),

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
                total_counts.true_positive
            ),

        "false_positive":
            int(
                total_counts.false_positive
            ),

        "false_negative":
            int(
                total_counts.false_negative
            ),

        "true_negative":
            int(
                total_counts.true_negative
            ),
    }

    if calculate_cldice:

        result[
            "cldice"
        ] = float(
            np.mean(
                cldice_values
            )
            if cldice_values
            else 0.0
        )

        result[
            "topology_precision"
        ] = float(
            np.mean(
                topology_precision_values
            )
            if topology_precision_values
            else 0.0
        )

        result[
            "topology_sensitivity"
        ] = float(
            np.mean(
                topology_sensitivity_values
            )
            if topology_sensitivity_values
            else 0.0
        )

    return result


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def select_best_postprocessing(
    results: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Select best Validation post-processing configuration.

    Ranking
    -------
    1. IoU
    2. clDice
    3. F1
    4. simpler transformation

    The final tie-break prefers less aggressive post-processing.
    """

    if not results:
        raise ValueError(
            "Post-processing results are empty."
        )

    ranked = sorted(
        results,

        key=lambda row: (
            float(
                row[
                    "iou"
                ]
            ),

            float(
                row.get(
                    "cldice",
                    0.0,
                )
            ),

            float(
                row.get(
                    "f1",
                    0.0,
                )
            ),

            -(
                int(
                    row[
                        "minimum_component_size"
                    ]
                )
                +
                int(
                    row[
                        "closing_radius"
                    ]
                )
                +
                int(
                    row[
                        "bridge_gap_radius"
                    ]
                )
                +
                int(
                    row[
                        "maximum_hole_size"
                    ]
                )
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
# Search
# ---------------------------------------------------------------------

def search_postprocessing(
    samples: Sequence[
        Tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    candidates: Sequence[
        RoadPostprocessingConfig
    ],
    calculate_cldice: bool = True,
    skeleton_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate every candidate configuration.
    """

    if not samples:
        raise ValueError(
            "No Validation prediction samples were provided."
        )

    if not candidates:
        raise ValueError(
            "No post-processing candidates were provided."
        )

    results = []

    total = len(
        candidates
    )

    for index, config in enumerate(
        candidates,
        start=1,
    ):

        result = (
            evaluate_postprocessing_candidate(
                samples=samples,
                config=config,
                calculate_cldice=(
                    calculate_cldice
                ),
                skeleton_iterations=(
                    skeleton_iterations
                ),
            )
        )

        result[
            "candidate_index"
        ] = int(
            index
        )

        results.append(
            result
        )

        print(
            f"[{index:03d}/{total:03d}] "
            f"IoU {result['iou'] * 100:.4f}% | "
            f"F1 {result['f1'] * 100:.4f}%"
            +
            (
                f" | clDice "
                f"{result['cldice'] * 100:.4f}%"
                if "cldice"
                in result
                else ""
            )
        )

    best = select_best_postprocessing(
        results
    )

    return {
        "results":
            results,

        "best":
            best,

        "candidate_count":
            int(
                len(
                    results
                )
            ),

        "selection_dataset":
            "validation",

        "primary_objective":
            "iou",

        "tie_break_metric":
            "cldice",

        "final_test_used":
            False,
    }


# ---------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------

def save_postprocessing_search_results(
    search_result: Dict[str, Any],
    output_folder: str | Path,
) -> Dict[str, Path]:
    """
    Save post-processing search CSV + JSON + selected config.
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
        / "postprocessing_search_results.csv"
    )

    json_path = (
        output_folder
        / "postprocessing_search_results.json"
    )

    selected_path = (
        output_folder
        / "selected_postprocessing.json"
    )

    rows = search_result[
        "results"
    ]

    if rows:

        fieldnames = []

        seen = set()

        for row in rows:

            for key in row.keys():

                if key not in seen:

                    fieldnames.append(
                        key
                    )

                    seen.add(
                        key
                    )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    save_json(
        search_result,
        json_path,
    )

    best = search_result[
        "best"
    ]

    selected = {
        "minimum_component_size":
            int(
                best[
                    "minimum_component_size"
                ]
            ),

        "closing_radius":
            int(
                best[
                    "closing_radius"
                ]
            ),

        "bridge_gap_radius":
            int(
                best[
                    "bridge_gap_radius"
                ]
            ),

        "maximum_hole_size":
            int(
                best[
                    "maximum_hole_size"
                ]
            ),

        "validation_iou":
            float(
                best[
                    "iou"
                ]
            ),

        "validation_precision":
            float(
                best[
                    "precision"
                ]
            ),

        "validation_recall":
            float(
                best[
                    "recall"
                ]
            ),

        "validation_f1":
            float(
                best[
                    "f1"
                ]
            ),

        "validation_cldice":
            (
                float(
                    best[
                        "cldice"
                    ]
                )
                if "cldice"
                in best
                else None
            ),

        "parameter_units":
            "pixels",

        "selection_dataset":
            "validation",

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

        "selected_postprocessing":
            selected_path,
    }


# ---------------------------------------------------------------------
# Selected threshold loader
# ---------------------------------------------------------------------

def load_selected_threshold(
    context: Dict[str, Any],
) -> float:
    """
    Load threshold selected by the Validation threshold-search stage.
    """

    path = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "threshold_search"
        / "selected_threshold.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Selected threshold file does not exist:\n"
            f"{path}\n\n"
            "Run run_threshold_search.py first."
        )

    import json

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    threshold = float(
        data[
            "selected_threshold"
        ]
    )

    return threshold


# ---------------------------------------------------------------------
# Complete experiment search
# ---------------------------------------------------------------------

def run_postprocessing_search(
    context: Dict[str, Any],
    checkpoint_path: Optional[
        str | Path
    ] = None,
    device: Optional[
        str | torch.device
    ] = None,
) -> Dict[str, Any]:
    """
    Run complete Validation-only Road post-processing search.
    """

    if (
        context[
            "model_backend"
        ]
        != "pytorch"
    ):

        raise RuntimeError(
            "Current native Road post-processing search supports "
            "PyTorch models only."
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

    threshold = load_selected_threshold(
        context
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

    validation_loader = (
        build_postprocess_validation_loader(
            context
        )
    )

    use_amp = bool(
        context[
            "model_config"
        ].get(
            "training",
            {},
        ).get(
            "use_amp",
            True,
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD VALIDATION POST-PROCESSING SEARCH"
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
        f"Frozen threshold : "
        f"{threshold:.2f}"
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

    # --------------------------------------------------------------
    # Neural network inference ONCE
    # --------------------------------------------------------------

    samples = cache_validation_predictions(
        model=model,

        loader=validation_loader,

        device=resolved_device,

        threshold=threshold,

        use_amp=use_amp,
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    optional_config = {}

    try:

        optional_config = (
            load_optional_config(
                "postprocessing"
            )
        )

    except Exception:

        optional_config = {}

    search_space = (
        resolve_postprocessing_search_space(
            general_params=(
                context[
                    "general_params"
                ]
            ),

            optional_config=(
                optional_config
            ),
        )
    )

    candidates = (
        generate_postprocessing_candidates(
            search_space
        )
    )

    print(
        f"Validation masks : "
        f"{len(samples)}"
    )

    print(
        f"Candidates       : "
        f"{len(candidates)}"
    )

    print()

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------

    search_result = (
        search_postprocessing(
            samples=samples,

            candidates=candidates,

            calculate_cldice=True,

            skeleton_iterations=50,
        )
    )

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
        "frozen_threshold"
    ] = float(
        threshold
    )

    search_result[
        "search_space"
    ] = search_space

    output_folder = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "postprocessing_search"
    )

    paths = (
        save_postprocessing_search_results(
            search_result=search_result,

            output_folder=output_folder,
        )
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

    best = search_result[
        "best"
    ]

    print()
    print(
        "=" * 72
    )

    print(
        "POST-PROCESSING SEARCH COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Min component    : "
        f"{best['minimum_component_size']} px"
    )

    print(
        f"Closing radius   : "
        f"{best['closing_radius']} px"
    )

    print(
        f"Gap bridge       : "
        f"{best['bridge_gap_radius']} px"
    )

    print(
        f"Max hole size    : "
        f"{best['maximum_hole_size']} px"
    )

    print(
        f"Validation IoU   : "
        f"{best['iou'] * 100:.4f}%"
    )

    print(
        f"Validation F1    : "
        f"{best['f1'] * 100:.4f}%"
    )

    if "cldice" in best:

        print(
            f"Validation clDice: "
            f"{best['cldice'] * 100:.4f}%"
        )

    print(
        "Settings frozen  : YES"
    )

    print(
        "Final test used  : NO"
    )

    print(
        "=" * 72
    )

    return search_result
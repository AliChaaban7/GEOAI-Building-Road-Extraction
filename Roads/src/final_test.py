"""
Independent final-test controller for Approach 2 - Road Extraction.

Scientific workflow
-------------------
Frozen checkpoint
+
Frozen threshold
+
Frozen post-processing
        ↓
same settings applied independently to:
        ↓
    Aerial
    Satellite
    Drone

For each configured source:

    held-out raster
        ↓
    frozen inference
        ↓
    georeferenced Road mask
        ↓
    Road polygons
        ↓
    held-out Ground Truth
        ↓
    strict GIS evaluation
        ↓
    IoU / Precision / Recall / F1

Important
---------
This module performs NO:

- training
- model selection
- threshold search
- post-processing search
- source-specific retuning
- Optuna
- Master Optuna

Missing Aerial/Drone paths are not invented. In "all" mode, incomplete
sources are skipped and clearly recorded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from src.evaluation.gis_evaluation import (
    evaluate_polygon_prediction,
)

from src.gis_inference import (
    run_frozen_raster_inference,
)

from src.pipeline.final_settings import (
    load_frozen_final_settings,
)

from src.utils import (
    load_paths_config,
    load_test_zone_config,
    save_json,
)


# ---------------------------------------------------------------------
# Supported sources
# ---------------------------------------------------------------------

FINAL_TEST_SOURCES = (
    "aerial",
    "satellite",
    "drone",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _first_nonempty(
    *values,
):
    """
    Return the first value that is not None/empty.
    """

    for value in values:

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:
                continue

        return value

    return None


def _as_dict(
    value,
) -> Dict[str, Any]:
    """
    Return value when it is a dictionary, otherwise {}.
    """

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


def _source_block_from_config(
    config: Dict[str, Any],
    source_name: str,
) -> Dict[str, Any]:
    """
    Resolve a source block from several supported config layouts.

    Supported examples
    ------------------
    {
        "final_test": {
            "satellite": {...}
        }
    }

    {
        "final_tests": {
            "satellite": {...}
        }
    }

    {
        "sources": {
            "satellite": {...}
        }
    }

    {
        "satellite": {...}
    }
    """

    source_name = str(
        source_name
    ).lower()

    candidate_roots = [
        config,
        _as_dict(
            config.get(
                "final_test"
            )
        ),
        _as_dict(
            config.get(
                "final_tests"
            )
        ),
        _as_dict(
            config.get(
                "sources"
            )
        ),
        _as_dict(
            config.get(
                "test_sources"
            )
        ),
    ]

    for root in candidate_roots:

        value = root.get(
            source_name
        )

        if isinstance(
            value,
            dict,
        ):
            return dict(
                value
            )

    return {}


def _resolve_image_path(
    source_block: Dict[str, Any],
):
    """
    Resolve held-out test raster path.
    """

    return _first_nonempty(
        source_block.get(
            "image_path"
        ),

        source_block.get(
            "raster_path"
        ),

        source_block.get(
            "test_raster"
        ),

        source_block.get(
            "image"
        ),

        source_block.get(
            "raster"
        ),

        source_block.get(
            "input_raster"
        ),
    )


def _resolve_ground_truth_path(
    source_block: Dict[str, Any],
):
    """
    Resolve held-out Ground Truth feature class path.
    """

    return _first_nonempty(
        source_block.get(
            "ground_truth_path"
        ),

        source_block.get(
            "ground_truth"
        ),

        source_block.get(
            "gt_path"
        ),

        source_block.get(
            "gt"
        ),

        source_block.get(
            "labels"
        ),

        source_block.get(
            "label_path"
        ),
    )


def _arcgis_exists(
    path: str,
) -> bool:
    """
    Test an ArcGIS raster/feature-class path.
    """

    try:
        import arcpy

    except ImportError as exc:

        raise RuntimeError(
            "Final GIS testing requires the ArcGIS Pro "
            "Python environment."
        ) from exc

    return bool(
        arcpy.Exists(
            path
        )
    )


def _safe_arcgis_name(
    text: str,
    maximum_length: int = 80,
) -> str:
    """
    Create a conservative ArcGIS-safe dataset prefix.
    """

    text = str(
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    text = text.strip(
        "_"
    )

    if not text:

        text = "road_final_test"

    if text[
        0
    ].isdigit():

        text = (
            "r_"
            + text
        )

    return text[
        :int(
            maximum_length
        )
    ]


# ---------------------------------------------------------------------
# Resolve source configuration
# ---------------------------------------------------------------------

def resolve_final_test_source(
    source_name: str,
    paths_config: Dict[str, Any],
    test_zone_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve one source's held-out raster and Ground Truth.

    test_zone.json values take precedence over paths.json when both
    explicitly provide the same item.
    """

    source_name = str(
        source_name
    ).lower()

    if (
        source_name
        not in FINAL_TEST_SOURCES
    ):

        raise ValueError(
            "Unknown final-test source: "
            f"{source_name!r}"
        )

    paths_block = (
        _source_block_from_config(
            paths_config,
            source_name,
        )
    )

    zone_block = (
        _source_block_from_config(
            test_zone_config,
            source_name,
        )
    )

    image_path = _first_nonempty(
        _resolve_image_path(
            zone_block
        ),

        _resolve_image_path(
            paths_block
        ),
    )

    ground_truth_path = (
        _first_nonempty(
            _resolve_ground_truth_path(
                zone_block
            ),

            _resolve_ground_truth_path(
                paths_block
            ),
        )
    )

    enabled = bool(
        _first_nonempty(
            zone_block.get(
                "enabled"
            ),

            paths_block.get(
                "enabled"
            ),

            True,
        )
    )

    configured_flag = _first_nonempty(
        zone_block.get(
            "configured"
        ),

        paths_block.get(
            "configured"
        ),
    )

    configured = (
        bool(
            configured_flag
        )
        if configured_flag
        is not None
        else bool(
            image_path
            and ground_truth_path
        )
    )

    native_resolution_m = (
        _first_nonempty(
            zone_block.get(
                "native_resolution_m"
            ),

            zone_block.get(
                "resolution_m"
            ),

            paths_block.get(
                "native_resolution_m"
            ),

            paths_block.get(
                "resolution_m"
            ),
        )
    )

    reasons = []

    if not enabled:

        reasons.append(
            "source disabled"
        )

    if not image_path:

        reasons.append(
            "test raster path missing"
        )

    if not ground_truth_path:

        reasons.append(
            "Ground Truth path missing"
        )

    ready = (
        enabled
        and bool(
            image_path
        )
        and bool(
            ground_truth_path
        )
    )

    return {
        "source":
            source_name,

        "enabled":
            enabled,

        "configured":
            configured,

        "ready":
            ready,

        "image_path":
            image_path,

        "ground_truth_path":
            ground_truth_path,

        "native_resolution_m":
            (
                float(
                    native_resolution_m
                )
                if native_resolution_m
                is not None
                else None
            ),

        "reasons_not_ready":
            reasons,

        "paths_config":
            paths_block,

        "test_zone_config":
            zone_block,
    }


# ---------------------------------------------------------------------
# Validate actual ArcGIS datasets
# ---------------------------------------------------------------------

def validate_final_test_source(
    source_config: Dict[str, Any],
) -> None:
    """
    Verify configured final-test raster and Ground Truth exist.
    """

    source = source_config[
        "source"
    ]

    if not source_config[
        "ready"
    ]:

        raise RuntimeError(
            f"Final-test source {source!r} is not fully configured:\n"
            + "\n".join(
                source_config[
                    "reasons_not_ready"
                ]
            )
        )

    image_path = source_config[
        "image_path"
    ]

    ground_truth_path = source_config[
        "ground_truth_path"
    ]

    if not _arcgis_exists(
        image_path
    ):

        raise FileNotFoundError(
            f"{source.capitalize()} final-test raster "
            f"does not exist:\n"
            f"{image_path}"
        )

    if not _arcgis_exists(
        ground_truth_path
    ):

        raise FileNotFoundError(
            f"{source.capitalize()} Road Ground Truth "
            f"does not exist:\n"
            f"{ground_truth_path}"
        )


# ---------------------------------------------------------------------
# Project geodatabase
# ---------------------------------------------------------------------

def resolve_output_geodatabase(
    paths_config: Dict[str, Any],
) -> str:
    """
    Resolve ArcGIS project geodatabase.
    """

    arcgis_config = _as_dict(
        paths_config.get(
            "arcgis"
        )
    )

    geodatabase = _first_nonempty(
        arcgis_config.get(
            "project_gdb"
        ),

        paths_config.get(
            "project_gdb"
        ),
    )

    if not geodatabase:

        raise ValueError(
            "ArcGIS project geodatabase is not configured "
            "in paths.json."
        )

    if not _arcgis_exists(
        geodatabase
    ):

        raise FileNotFoundError(
            "Configured ArcGIS project geodatabase "
            "does not exist:\n"
            f"{geodatabase}"
        )

    return str(
        geodatabase
    )


# ---------------------------------------------------------------------
# One final test
# ---------------------------------------------------------------------

def run_one_final_test(
    context: Dict[str, Any],
    source_name: str,
    source_config: Dict[str, Any],
    frozen_settings: Dict[str, Any],
    output_geodatabase: str,
    overlap: Optional[int] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run independent final inference + GIS evaluation for one source.

    No source-specific settings are changed.
    """

    validate_final_test_source(
        source_config
    )

    source_name = str(
        source_name
    ).lower()

    if (
        source_name
        not in frozen_settings[
            "final_test_sources"
        ]
    ):

        raise RuntimeError(
            f"{source_name!r} is not included in the "
            "frozen final-test source list."
        )

    output_folder = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "final_test"
        / source_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"FINAL ROAD TEST — {source_name.upper()}"
    )

    print(
        "=" * 72
    )

    print(
        f"Raster           : "
        f"{source_config['image_path']}"
    )

    print(
        f"Ground Truth     : "
        f"{source_config['ground_truth_path']}"
    )

    print(
        f"Checkpoint       : FROZEN"
    )

    print(
        f"Threshold        : "
        f"{frozen_settings['inference']['probability_threshold']:.2f}"
    )

    print(
        f"Post-processing  : "
        f"{'ON' if frozen_settings['postprocessing']['enabled'] else 'OFF'}"
    )

    print(
        "Retuning         : NO"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------------
    # Prediction
    #
    # Ground Truth is deliberately NOT passed to inference.
    # --------------------------------------------------------------

    inference_result = (
        run_frozen_raster_inference(
            context=context,

            raster_path=(
                source_config[
                    "image_path"
                ]
            ),

            source_name=(
                source_name
            ),

            output_folder=(
                output_folder
            ),

            output_geodatabase=(
                output_geodatabase
            ),

            overlap=overlap,

            batch_size=int(
                batch_size
            ),

            device=device,

            save_probability_raster=True,

            create_polygon=True,
        )
    )

    prediction_fc = (
        inference_result[
            "polygon_feature_class"
        ]
    )

    if not prediction_fc:

        raise RuntimeError(
            "Final inference did not produce a Road polygon "
            "feature class."
        )

    # --------------------------------------------------------------
    # Strict held-out GIS evaluation
    # --------------------------------------------------------------

    evaluation_name = (
        _safe_arcgis_name(
            f"{context['experiment_config']['experiment_id']}_"
            f"{source_name}_final"
        )
    )

    evaluation_json = (
        output_folder
        / "final_evaluation.json"
    )

    evaluation_result = (
        evaluate_polygon_prediction(
            prediction_feature_class=(
                prediction_fc
            ),

            ground_truth_feature_class=(
                source_config[
                    "ground_truth_path"
                ]
            ),

            output_geodatabase=(
                output_geodatabase
            ),

            evaluation_name=(
                evaluation_name
            ),

            evaluation_raster=(
                source_config[
                    "image_path"
                ]
            ),

            save_json_path=(
                evaluation_json
            ),
        )
    )

    metrics = evaluation_result[
        "metrics"
    ]

    # --------------------------------------------------------------
    # Source result
    # --------------------------------------------------------------

    result = {
        "source":
            source_name,

        "status":
            "completed",

        "native_resolution_m":
            source_config.get(
                "native_resolution_m"
            ),

        "input_raster":
            source_config[
                "image_path"
            ],

        "ground_truth":
            source_config[
                "ground_truth_path"
            ],

        "frozen_settings": {
            "checkpoint":
                frozen_settings[
                    "checkpoint"
                ][
                    "path"
                ],

            "checkpoint_sha256":
                frozen_settings[
                    "checkpoint"
                ][
                    "sha256"
                ],

            "probability_threshold":
                frozen_settings[
                    "inference"
                ][
                    "probability_threshold"
                ],

            "postprocessing":
                frozen_settings[
                    "postprocessing"
                ],
        },

        "inference":
            inference_result,

        "evaluation":
            evaluation_result,

        "metrics": {
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

            "iou_percent":
                float(
                    metrics[
                        "iou_percent"
                    ]
                ),

            "precision_percent":
                float(
                    metrics[
                        "precision_percent"
                    ]
                ),

            "recall_percent":
                float(
                    metrics[
                        "recall_percent"
                    ]
                ),

            "f1_percent":
                float(
                    metrics[
                        "f1_percent"
                    ]
                ),
        },

        "parameter_search_performed":
            False,

        "source_specific_retuning":
            False,

        "final_ground_truth_used_only_for_evaluation":
            True,
    }

    source_summary_path = (
        output_folder
        / "final_test_summary.json"
    )

    save_json(
        result,
        source_summary_path,
    )

    result[
        "summary_json"
    ] = str(
        source_summary_path
    )

    print()
    print(
        "-" * 72
    )

    print(
        f"FINAL {source_name.upper()} RESULTS"
    )

    print(
        "-" * 72
    )

    print(
        f"IoU              : "
        f"{metrics['iou_percent']:.4f}%"
    )

    print(
        f"Precision        : "
        f"{metrics['precision_percent']:.4f}%"
    )

    print(
        f"Recall           : "
        f"{metrics['recall_percent']:.4f}%"
    )

    print(
        f"F1               : "
        f"{metrics['f1_percent']:.4f}%"
    )

    print(
        "Retuning         : NO"
    )

    print(
        "-" * 72
    )

    return result


# ---------------------------------------------------------------------
# Mean source metrics
# ---------------------------------------------------------------------

def calculate_mean_source_metrics(
    completed_results: Iterable[
        Dict[str, Any]
    ],
) -> Optional[
    Dict[str, float]
]:
    """
    Macro-average completed source metrics.

    Each source receives equal weight regardless of raster area.
    """

    results = list(
        completed_results
    )

    if not results:
        return None

    metric_names = (
        "iou",
        "precision",
        "recall",
        "f1",
    )

    output = {}

    for metric_name in metric_names:

        value = mean(
            float(
                result[
                    "metrics"
                ][
                    metric_name
                ]
            )
            for result
            in results
        )

        output[
            metric_name
        ] = float(
            value
        )

        output[
            f"{metric_name}_percent"
        ] = float(
            value * 100.0
        )

    output[
        "source_count"
    ] = int(
        len(
            results
        )
    )

    output[
        "aggregation"
    ] = "macro_mean_across_sources"

    return output


# ---------------------------------------------------------------------
# Complete final-test controller
# ---------------------------------------------------------------------

def run_final_tests(
    context: Dict[str, Any],
    source: str = "all",
    overlap: Optional[int] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one or all configured independent final tests.

    Parameters
    ----------
    source:
        all
        aerial
        satellite
        drone

    Behavior
    --------
    source="all":
        incomplete sources are skipped.

    explicit source:
        incomplete configuration raises an error.
    """

    source = str(
        source
    ).lower()

    if source not in {
        "all",
        *FINAL_TEST_SOURCES,
    }:

        raise ValueError(
            "source must be one of:\n"
            "all, aerial, satellite, drone"
        )

    # --------------------------------------------------------------
    # Frozen settings
    # --------------------------------------------------------------

    frozen_settings = (
        load_frozen_final_settings(
            context=context,

            verify_checkpoint=True,
        )
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    paths_config = (
        load_paths_config()
    )

    test_zone_config = (
        load_test_zone_config()
    )

    output_geodatabase = (
        resolve_output_geodatabase(
            paths_config
        )
    )

    frozen_sources = [
        str(
            item
        ).lower()
        for item in frozen_settings[
            "final_test_sources"
        ]
    ]

    if source == "all":

        requested_sources = [
            item
            for item in FINAL_TEST_SOURCES
            if item in frozen_sources
        ]

    else:

        if source not in frozen_sources:

            raise RuntimeError(
                f"Requested source {source!r} is not part "
                "of frozen final settings."
            )

        requested_sources = [
            source
        ]

    # --------------------------------------------------------------
    # Resolve readiness
    # --------------------------------------------------------------

    source_configs = {
        source_name:
            resolve_final_test_source(
                source_name=source_name,

                paths_config=paths_config,

                test_zone_config=(
                    test_zone_config
                ),
            )

        for source_name
        in requested_sources
    }

    completed = []

    skipped = []

    failed = []

    # --------------------------------------------------------------
    # Run each source independently
    # --------------------------------------------------------------

    for source_name in requested_sources:

        source_config = (
            source_configs[
                source_name
            ]
        )

        if not source_config[
            "ready"
        ]:

            reason = "; ".join(
                source_config[
                    "reasons_not_ready"
                ]
            )

            if source != "all":

                raise RuntimeError(
                    f"{source_name.capitalize()} final test "
                    f"is not ready:\n{reason}"
                )

            print()
            print(
                f"[SKIP] {source_name.upper()}: "
                f"{reason}"
            )

            skipped.append(
                {
                    "source":
                        source_name,

                    "reason":
                        reason,
                }
            )

            continue

        try:

            result = run_one_final_test(
                context=context,

                source_name=(
                    source_name
                ),

                source_config=(
                    source_config
                ),

                frozen_settings=(
                    frozen_settings
                ),

                output_geodatabase=(
                    output_geodatabase
                ),

                overlap=overlap,

                batch_size=(
                    batch_size
                ),

                device=device,
            )

            completed.append(
                result
            )

        except Exception as exc:

            if source != "all":
                raise

            print()
            print(
                f"[FAILED] {source_name.upper()}: "
                f"{exc}"
            )

            failed.append(
                {
                    "source":
                        source_name,

                    "error":
                        str(
                            exc
                        ),
                }
            )

    # --------------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------------

    mean_metrics = (
        calculate_mean_source_metrics(
            completed
        )
    )

    completed_sources = [
        item[
            "source"
        ]
        for item in completed
    ]

    all_frozen_sources_completed = (
        set(
            completed_sources
        )
        == set(
            frozen_sources
        )
    )

    summary = {
        "experiment_id":
            context[
                "experiment_config"
            ][
                "experiment_id"
            ],

        "status":
            (
                "complete"
                if all_frozen_sources_completed
                else "partial"
            ),

        "settings_status":
            "FROZEN",

        "frozen_checkpoint":
            frozen_settings[
                "checkpoint"
            ],

        "frozen_threshold":
            float(
                frozen_settings[
                    "inference"
                ][
                    "probability_threshold"
                ]
            ),

        "frozen_postprocessing":
            frozen_settings[
                "postprocessing"
            ],

        "expected_final_sources":
            frozen_sources,

        "requested_sources":
            requested_sources,

        "completed_sources":
            completed_sources,

        "skipped_sources":
            skipped,

        "failed_sources":
            failed,

        "all_frozen_sources_completed":
            bool(
                all_frozen_sources_completed
            ),

        "source_results":
            completed,

        "mean_metrics":
            mean_metrics,

        "mean_metric_policy":
            (
                "Unweighted macro mean across completed "
                "source-specific final tests."
            ),

        "source_specific_retuning":
            False,

        "final_test_used_for_selection":
            False,
    }

    output_folder = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "final_test"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_folder
        / "final_test_summary.json"
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

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD FINAL TEST SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "Completed        : "
        + (
            ", ".join(
                completed_sources
            )
            if completed_sources
            else "none"
        )
    )

    print(
        "Skipped          : "
        + (
            ", ".join(
                item[
                    "source"
                ]
                for item
                in skipped
            )
            if skipped
            else "none"
        )
    )

    print(
        "Failed           : "
        + (
            ", ".join(
                item[
                    "source"
                ]
                for item
                in failed
            )
            if failed
            else "none"
        )
    )

    if mean_metrics is not None:

        print(
            f"Mean IoU         : "
            f"{mean_metrics['iou_percent']:.4f}%"
        )

        print(
            f"Mean Precision   : "
            f"{mean_metrics['precision_percent']:.4f}%"
        )

        print(
            f"Mean Recall      : "
            f"{mean_metrics['recall_percent']:.4f}%"
        )

        print(
            f"Mean F1          : "
            f"{mean_metrics['f1_percent']:.4f}%"
        )

    print(
        f"Overall status   : "
        f"{summary['status'].upper()}"
    )

    print(
        "Retuning         : NO"
    )

    print(
        "=" * 72
    )

    return summary
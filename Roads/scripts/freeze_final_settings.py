"""
Freeze Road model-selection settings before independent final testing.

This script DOES NOT open:
- Aerial final raster
- Satellite final raster
- Drone final raster
- Ground Truth

It simply freezes the settings selected from Train / Validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROAD_ROOT = Path(__file__).resolve().parents[1]

if str(ROAD_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROAD_ROOT),
    )


from src.pipeline import prepare_experiment


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Freeze Road settings before final test."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(
    path,
):

    path = Path(path)

    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def ensure_checkpoint_folder(
    context,
):
    """
    Normalize Road experiment context for compatibility with older
    prepare_experiment() outputs.

    Preferred:
        context["checkpoint_folder"]

    Backward-compatible fallback:
        context["output_folder"]

    The standard Road experiment layout stores best_model.pth directly in
    outputs/experiments/<experiment_id>/ unless a dedicated checkpoint
    folder is explicitly configured.
    """

    checkpoint_folder = (
        context.get(
            "checkpoint_folder"
        )
        or context.get(
            "output_folder"
        )
    )

    if checkpoint_folder is None:
        raise KeyError(
            "Road experiment context is missing both "
            "'checkpoint_folder' and 'output_folder'."
        )

    checkpoint_folder = Path(
        checkpoint_folder
    )

    context[
        "checkpoint_folder"
    ] = checkpoint_folder

    return checkpoint_folder


def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,
        force_regenerate_manifests=False,
    )

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    # ---------------------------------------------------------
    # Backward-compatible checkpoint context normalization.
    # ---------------------------------------------------------

    checkpoint_folder = ensure_checkpoint_folder(
        context
    )

    final_settings_path = (
        output_folder
        / "final_settings.json"
    )

    if (
        final_settings_path.exists()
        and not args.overwrite
    ):

        raise FileExistsError(
            "final_settings.json already exists.\n"
            "Use --overwrite only if model selection "
            "has intentionally changed."
        )

    checkpoint = (
        Path(
            args.checkpoint
        )
        if args.checkpoint
        else (
            checkpoint_folder
            / "best_model.pth"
        )
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Best checkpoint not found:\n{checkpoint}"
        )

    training_summary_path = (
        output_folder
        / "training_summary.json"
    )

    training_summary = load_json(
        training_summary_path
    )

    if training_summary is None:

        raise FileNotFoundError(
            f"Training summary missing:\n"
            f"{training_summary_path}"
        )

    experiment = context[
        "experiment_config"
    ]

    threshold_summary = load_json(
        output_folder
        / "validation_threshold_summary.json"
    )

    if bool(
        experiment.get(
            "use_threshold_search",
            False,
        )
    ) and threshold_summary is None:

        raise RuntimeError(
            "Experiment requires threshold search, but "
            "validation_threshold_summary.json is missing."
        )

    threshold = (
        float(
            threshold_summary[
                "best_threshold"
            ]
        )
        if threshold_summary
        is not None
        else 0.5
    )

    post_summary = load_json(
        output_folder
        / "validation_postprocess_summary.json"
    )

    if bool(
        experiment.get(
            "use_postprocessing",
            False,
        )
    ) and post_summary is None:

        raise RuntimeError(
            "Experiment requires post-processing, but "
            "validation_postprocess_summary.json is missing."
        )

    post_config = (
        post_summary.get(
            "selected_config"
        )
        if post_summary
        is not None
        else None
    )

    optuna_summary = load_json(
        output_folder
        / "optuna_selection_summary.json"
    )

    master_summary = load_json(
        output_folder
        / "master_optuna_selection_summary.json"
    )

    if bool(
        experiment.get(
            "use_optuna",
            False,
        )
    ) and optuna_summary is None:

        raise RuntimeError(
            "use_optuna=true but optuna_selection_summary.json "
            "is missing."
        )

    if bool(
        experiment.get(
            "use_master_optuna",
            False,
        )
    ) and master_summary is None:

        raise RuntimeError(
            "use_master_optuna=true but "
            "master_optuna_selection_summary.json is missing."
        )

    final_settings = {
        "settings_frozen":
            True,

        "frozen_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "architecture":
            "roads_train_validation_freeze_final_test_v1",

        "experiment_id":
            experiment.get(
                "experiment_id"
            ),

        "model_type":
            context.get(
                "model_type"
            ),

        "model_backend":
            context.get(
                "model_backend"
            ),

        "source_training":
            "mixed",

        "checkpoint":
            str(
                checkpoint
            ),

        "checkpoint_format":
            training_summary.get(
                "checkpoint_format"
            ),

        "tile_size":
            int(
                training_summary.get(
                    "tile_size",
                    experiment[
                        "tile_size"
                    ],
                )
            ),

        "best_epoch":
            training_summary.get(
                "best_epoch"
            ),

        "best_validation_iou":
            training_summary.get(
                "best_validation_iou"
            ),

        "probability_threshold":
            threshold,

        "threshold_source":
            (
                "validation_search"
                if threshold_summary
                else "fixed_0.5"
            ),

        "postprocessing_enabled":
            bool(
                post_config
                is not None
            ),

        "postprocessing":
            post_config,

        "optimization": {
            "general_optuna":
                optuna_summary,

            "master_optuna":
                master_summary,
        },

        "final_test_policy": {
            "sources": [
                "aerial",
                "satellite",
                "drone",
            ],

            "same_checkpoint_for_all_sources":
                True,

            "same_threshold_for_all_sources":
                True,

            "same_postprocessing_for_all_sources":
                True,

            "per_source_retuning":
                False,
        },

        "final_test_accessed":
            False,
    }

    final_settings_path.write_text(
        json.dumps(
            final_settings,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD FINAL SETTINGS FROZEN"
    )

    print(
        "=" * 72
    )

    print(
        f"Checkpoint : {checkpoint}"
    )

    print(
        f"Threshold  : {threshold}"
    )

    print(
        f"Post       : "
        f"{'ON' if post_config else 'OFF'}"
    )

    print(
        "Final test accessed: NO"
    )

    print(
        f"Saved      : {final_settings_path}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()

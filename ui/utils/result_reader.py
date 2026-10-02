"""
ui/utils/result_reader.py

Read-only experiment/result discovery shared by Buildings and Roads.

This module never launches training or inference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ui.utils.config_io import (
    first_json,
    load_json,
    percentage,
)


# ============================================================
# KNOWN RESULT FILES
# ============================================================

CONFIG_FILENAMES = (
    "experiment_config_used.json",
    "config_used.json",
    "resolved_master_winner_experiment_config.json",
)

TRAINING_FILENAMES = (
    "training_summary.json",
    "maskrcnn_training_summary.json",
)

OPTUNA_FILENAMES = (
    "optuna_selection_summary.json",
    "run_optuna_summary.json",
)

MASTER_FILENAMES = (
    "master_optuna_selection_summary.json",
)

THRESHOLD_FILENAMES = (
    "validation_threshold_summary.json",
)

POSTPROCESS_FILENAMES = (
    "validation_postprocess_summary.json",
)


# ============================================================
# EXPERIMENT DISCOVERY
# ============================================================

def experiment_folders(
    experiments_root: Path,
) -> list[Path]:
    """
    Return experiment folders newest first.
    """

    if not experiments_root.exists():
        return []

    return sorted(
        [
            folder
            for folder in experiments_root.iterdir()
            if folder.is_dir()
            and not folder.name.lower().startswith(
                (
                    "smoke_",
                    "ui_smoke_",
                )
            )
        ],
        key=lambda folder: (
            folder.stat().st_mtime
        ),
        reverse=True,
    )


def experiment_config(
    folder: Path,
) -> dict:
    """
    Resolve an experiment configuration saved inside an output folder.
    """

    for filename in CONFIG_FILENAMES:
        payload = load_json(
            folder / filename,
            {},
        )

        if payload.get(
            "experiment_id"
        ):
            return payload

        for key in (
            "experiment",
            "experiment_config",
        ):
            nested = payload.get(
                key
            )

            if (
                isinstance(
                    nested,
                    dict,
                )
                and nested.get(
                    "experiment_id"
                )
            ):
                return nested

    return {}


# ============================================================
# SPLIT
# ============================================================

def split_info(
    folder: Path,
    config: dict,
) -> dict:
    """
    Resolve Train/Validation metadata without inventing missing values.
    """

    summary = first_json(
        folder,
        (
            "split_summary.json",
            "manifest_summary.json",
        ),
    )

    data_split = config.get(
        "data_split",
        {},
    )

    return {
        "train_percent":
            summary.get(
                "train_percent",
                data_split.get(
                    "train_percent"
                ),
            ),

        "validation_percent":
            summary.get(
                "validation_percent",
                data_split.get(
                    "validation_percent"
                ),
            ),

        "seed":
            summary.get(
                "seed",
                data_split.get(
                    "seed"
                ),
            ),

        "train_count":
            summary.get(
                "train_count"
            ),

        "validation_count":
            summary.get(
                "validation_count"
            ),
    }


# ============================================================
# FINAL TEST DISCOVERY
# ============================================================

def final_metric_payloads(
    folder: Path,
    approach: str,
) -> list[tuple[str, dict]]:
    """
    Read completed final metrics.

    Supported layouts:
        <experiment>/final_test/final_metrics.json

    and source-specific Roads layout:
        <experiment>/final_test/aerial/final_metrics.json
        <experiment>/final_test/satellite/final_metrics.json
        <experiment>/final_test/drone/final_metrics.json
    """

    final_root = (
        folder
        / "final_test"
    )

    if not final_root.exists():
        return []

    direct = load_json(
        final_root
        / "final_metrics.json",
        {},
    )

    if direct:
        source = str(
            direct.get(
                "source"
            )
            or experiment_config(
                folder
            ).get(
                "source_type"
            )
            or "final"
        )

        return [
            (
                source,
                direct,
            )
        ]

    rows = []

    for source in (
        "aerial",
        "satellite",
        "drone",
    ):
        payload = load_json(
            final_root
            / source
            / "final_metrics.json",
            {},
        )

        if payload:
            rows.append(
                (
                    source,
                    payload,
                )
            )

    return rows


# ============================================================
# STATUS
# ============================================================

def status_for(
    training: dict,
    optuna: dict,
    master: dict,
    threshold: dict,
    postprocess: dict,
    final_rows,
) -> str:

    if final_rows:
        return "Final Tested"

    if postprocess:
        return "Post-processed"

    if threshold:
        return "Threshold Tuned"

    if master:
        return "Master Optuna Complete"

    if optuna:
        return "Optuna Complete"

    if training:
        return "Trained"

    return "Incomplete"


# ============================================================
# TABLE
# ============================================================

def result_rows(
    approach: str,
    experiments_root: Path,
) -> pd.DataFrame:
    """
    Build one professional result table from experiment artifacts.
    """

    rows = []

    for folder in experiment_folders(
        experiments_root
    ):
        config = experiment_config(
            folder
        )

        split = split_info(
            folder,
            config,
        )

        training = first_json(
            folder,
            TRAINING_FILENAMES,
        )

        optuna = first_json(
            folder,
            OPTUNA_FILENAMES,
        )

        master = first_json(
            folder,
            MASTER_FILENAMES,
        )

        threshold = first_json(
            folder,
            THRESHOLD_FILENAMES,
        )

        postprocess = first_json(
            folder,
            POSTPROCESS_FILENAMES,
        )

        finals = final_metric_payloads(
            folder,
            approach,
        )

        validation_iou = percentage(
            training,
            "best_validation_iou_percent",
            "best_validation_iou",
            "best_val_iou",
            "validation_iou",
        )

        if (
            validation_iou is None
            and master
        ):
            validation_iou = percentage(
                master,
                "fresh_training_best_validation_iou_percent",
                "fresh_training_best_validation_iou",
                "best_trial_validation_iou_percent",
                "best_trial_validation_iou",
                "best_validation_iou_percent",
                "best_validation_iou",
            )

        if (
            validation_iou is None
            and optuna
        ):
            validation_iou = percentage(
                optuna,
                "final_training_best_validation_iou",
                "best_validation_iou_percent",
                "best_validation_iou",
                "best_value",
            )

        train_percent = split.get(
            "train_percent"
        )

        validation_percent = split.get(
            "validation_percent"
        )

        split_text = (
            f"{float(train_percent):.0f}/"
            f"{float(validation_percent):.0f}"
            if (
                train_percent
                is not None
                and validation_percent
                is not None
            )
            else "—"
        )

        threshold_value = (
            threshold.get(
                "best_threshold"
            )
            if threshold
            else None
        )

        common = {
            "Approach":
                approach.title(),

            "Experiment":
                folder.name,

            "Model":
                config.get(
                    "model_type",
                    "—",
                ),

            "Backend":
                config.get(
                    "model_backend",
                    "—",
                ),

            "Source":
                config.get(
                    "source_type",
                    "—",
                ),

            "Tile":
                config.get(
                    "tile_size",
                    "—",
                ),

            "Split":
                split_text,

            "Seed":
                split.get(
                    "seed",
                    "—",
                ),

            "Aug":
                (
                    "ON"
                    if config.get(
                        "use_augmentation",
                        False,
                    )
                    else "OFF"
                ),

            "Optuna":
                (
                    "ON"
                    if (
                        optuna
                        or config.get(
                            "use_optuna",
                            False,
                        )
                    )
                    else "OFF"
                ),

            "Master Optuna":
                (
                    "ON"
                    if (
                        master
                        or config.get(
                            "use_master_optuna",
                            False,
                        )
                    )
                    else "OFF"
                ),

            "Validation IoU":
                validation_iou,

            "Threshold":
                threshold_value,

            "Post-processing":
                (
                    "ON"
                    if postprocess
                    else "OFF"
                ),

            "Status":
                status_for(
                    training,
                    optuna,
                    master,
                    threshold,
                    postprocess,
                    finals,
                ),
        }

        if finals:
            for source, final in finals:
                row = dict(
                    common
                )

                row[
                    "Final Source"
                ] = source

                row[
                    "Test IoU"
                ] = percentage(
                    final,
                    "iou_percent",
                    "iou",
                )

                row[
                    "Precision"
                ] = percentage(
                    final,
                    "precision_percent",
                    "precision",
                )

                row[
                    "Recall"
                ] = percentage(
                    final,
                    "recall_percent",
                    "recall",
                )

                row[
                    "F1"
                ] = percentage(
                    final,
                    "f1_percent",
                    "f1",
                )

                row[
                    "clDice"
                ] = percentage(
                    final,
                    "cldice_percent",
                    "cldice",
                )

                rows.append(
                    row
                )

        else:
            row = dict(
                common
            )

            row.update(
                {
                    "Final Source":
                        "—",

                    "Test IoU":
                        None,

                    "Precision":
                        None,

                    "Recall":
                        None,

                    "F1":
                        None,

                    "clDice":
                        None,
                }
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )

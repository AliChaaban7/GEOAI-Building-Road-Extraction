"""Shared experiment helpers used by model-aware launchers."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import shutil

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_datasets_config,
    get_dataset_info,
    create_experiment_output_folder,
    create_train_val_manifests,
    resolve_train_validation_split,
)



# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_json_if_exists(path: Path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return load_json(path)
    except Exception:
        return None


def save_json(data: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ============================================================
# DATASET IDENTITY HELPERS
# ============================================================

def _normalize_path(path) -> str:
    return os.path.normcase(
        os.path.normpath(
            os.path.abspath(str(path))
        )
    )


def _same_path(path_a, path_b) -> bool:
    try:
        return (
            _normalize_path(path_a)
            == _normalize_path(path_b)
        )
    except Exception:
        return False


def _path_is_inside(
    child_path,
    parent_path,
) -> bool:
    try:
        child = _normalize_path(child_path)
        parent = _normalize_path(parent_path)

        return (
            os.path.commonpath(
                [child, parent]
            )
            == parent
        )
    except Exception:
        return False


def _manifest_belongs_to_dataset(
    manifest_path: Path,
    dataset_path,
    sample_limit: int = 20,
) -> bool:
    """
    Backward-compatible dataset-identity check for old manifest folders.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return False

    image_columns = (
        "image_path",
        "image",
        "image_file",
        "image_filepath",
        "chip_path",
        "raster_path",
    )

    checked = 0

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                return False

            image_column = next(
                (
                    column
                    for column in image_columns
                    if column in reader.fieldnames
                ),
                None,
            )

            if image_column is None:
                return False

            for row in reader:
                value = str(
                    row.get(
                        image_column,
                        "",
                    )
                ).strip()

                if not value:
                    continue

                image_path = Path(value)

                if not image_path.is_absolute():
                    image_path = (
                        Path(dataset_path)
                        / image_path
                    )

                if not _path_is_inside(
                    image_path,
                    dataset_path,
                ):
                    return False

                checked += 1

                if checked >= int(sample_limit):
                    break

    except Exception:
        return False

    return checked > 0


def existing_manifests_match_dataset(
    output_folder: Path,
    dataset_info: dict,
) -> bool:
    """
    Verify that an existing experiment folder belongs to the currently
    requested dataset.

    This prevents stale manifests from a previous tile/source/dataset
    configuration from being silently reused.
    """
    output_folder = Path(output_folder)

    requested_dataset_path = (
        dataset_info.get("path")
        or dataset_info.get("dataset_path")
        or dataset_info.get("folder")
        or dataset_info.get("root")
    )

    if requested_dataset_path is None:
        return False

    quality_summary = load_json_if_exists(
        output_folder
        / "data_quality_summary.json"
    )

    if isinstance(
        quality_summary,
        dict,
    ):
        existing_dataset_path = (
            quality_summary.get(
                "dataset_path"
            )
        )

        if existing_dataset_path:
            return _same_path(
                existing_dataset_path,
                requested_dataset_path,
            )

    train_manifest = (
        output_folder
        / "train_manifest.csv"
    )

    val_manifest = (
        output_folder
        / "val_manifest.csv"
    )

    return (
        _manifest_belongs_to_dataset(
            train_manifest,
            requested_dataset_path,
        )
        and
        _manifest_belongs_to_dataset(
            val_manifest,
            requested_dataset_path,
        )
    )


# ============================================================
# ROLLBACK BACKUP
# ============================================================

def backup_existing_split_files(
    output_folder: Path,
):
    """
    Preserve the first pre-regeneration split/data-quality files.

    This keeps the unified three-model workflow rollback-safe.
    """
    output_folder = Path(output_folder)

    candidates = (
        "train_manifest.csv",
        "val_manifest.csv",
        "manifest_summary.json",
        "split_summary.json",
        "data_quality.csv",
        "data_quality_summary.json",
    )

    existing = [
        output_folder / name
        for name in candidates
        if (output_folder / name).exists()
    ]

    if not existing:
        return None

    backup_folder = (
        output_folder
        / "legacy_split_backup"
    )

    backup_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in existing:
        destination = (
            backup_folder
            / source.name
        )

        if destination.exists():
            continue

        shutil.copy2(
            source,
            destination,
        )

    return backup_folder


# ============================================================
# PREPARE EXPERIMENT SPLIT
# ============================================================

def prepare_experiment_split(
    buildings_root: Path,
    config_path: Path,
):
    """
    Resolve dataset and create/reuse deterministic Train/Validation manifests.

    Dataset type is model-aware and automatic:
        U-Net / DeepLabV3 -> CT
        Mask R-CNN        -> RCNN

    Reuse is allowed ONLY when BOTH match:
        - requested Train/Validation percentages + seed
        - requested dataset identity

    Test data are never included.
    """
    buildings_root = Path(
        buildings_root
    )

    experiment_config = load_experiment_config(
        str(config_path)
    )

    general_params = load_general_params(
        buildings_root
    )

    datasets_config = load_datasets_config(
        buildings_root
    )

    dataset_info = get_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config,
    )

    output_folder = Path(
        create_experiment_output_folder(
            experiment_config=experiment_config,
            root=buildings_root,
        )
    )

    train_manifest = (
        output_folder
        / "train_manifest.csv"
    )

    val_manifest = (
        output_folder
        / "val_manifest.csv"
    )

    split_info = resolve_train_validation_split(
        general_params=general_params
    )

    split_summary = (
        output_folder
        / "split_summary.json"
    )

    regenerate = True

    if (
        train_manifest.exists()
        and val_manifest.exists()
        and split_summary.exists()
    ):
        try:
            existing_split = load_json(
                split_summary
            )

            split_matches = (
                abs(
                    float(
                        existing_split.get(
                            "train_percent"
                        )
                    )
                    - float(
                        split_info[
                            "train_percent"
                        ]
                    )
                )
                < 1e-8

                and

                abs(
                    float(
                        existing_split.get(
                            "validation_percent"
                        )
                    )
                    - float(
                        split_info[
                            "validation_percent"
                        ]
                    )
                )
                < 1e-8

                and

                int(
                    existing_split.get(
                        "seed"
                    )
                )
                == int(
                    split_info[
                        "seed"
                    ]
                )
            )

            dataset_matches = (
                existing_manifests_match_dataset(
                    output_folder=
                        output_folder,

                    dataset_info=
                        dataset_info,
                )
            )

            regenerate = not (
                split_matches
                and dataset_matches
            )

        except Exception:
            regenerate = True

    if regenerate:
        backup_existing_split_files(
            output_folder
        )

        result = create_train_val_manifests(
            dataset_info=dataset_info,
            output_folder=output_folder,
            general_params=general_params,
        )

        train_manifest = Path(
            result[
                "train_manifest_csv"
            ]
        )

        val_manifest = Path(
            result[
                "val_manifest_csv"
            ]
        )

    return {
        "experiment_config":
            experiment_config,

        "general_params":
            general_params,

        "datasets_config":
            datasets_config,

        "dataset_info":
            dataset_info,

        "output_folder":
            output_folder,

        "train_manifest":
            train_manifest,

        "val_manifest":
            val_manifest,

        "split_info":
            split_info,

        "dataset_id":
            dataset_info.get(
                "dataset_id"
            ),

        "dataset_path":
            dataset_info.get(
                "path"
            ),

        "manifests_regenerated":
            bool(regenerate),
    }


# ============================================================
# PIXEL SIZE
# ============================================================

def get_pixel_size_m(
    dataset_info: dict,
    general_params: dict,
    fallback: float = 0.3,
) -> float:
    """
    Resolve pixel size conservatively without using the external test scene.
    """
    candidates = [
        dataset_info.get(
            "pixel_size_m"
        ),
        dataset_info.get(
            "resolution_m"
        ),
        dataset_info.get(
            "spatial_resolution_m"
        ),
        general_params.get(
            "data",
            {},
        ).get(
            "pixel_size_m"
        ),
    ]

    for value in candidates:
        if value is not None:
            try:
                value = float(value)

                if value > 0:
                    return value

            except Exception:
                pass

    return float(
        fallback
    )

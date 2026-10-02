"""Road Classified-Tiles manifest creation.

Purpose
-------
Build deterministic Train / Validation manifests for Road semantic
segmentation from ArcGIS Classified Tiles exports.

Important Road policy
---------------------
ArcGIS may export image chips that contain no Road feature without writing
a corresponding label raster. Those chips are valid background examples,
not automatically corrupted data.

This module therefore supports two pairing modes:

    strict_pairing=True
        Every image chip must have a matching Road label.
        Unmatched files raise an error.

    strict_pairing=False
        Missing-label image chips are treated as background candidates.
        A zero-valued mask is generated for the selected background chips.

The normal Road training pipeline uses non-strict pairing plus controlled
background sampling. All positive Road chips are kept; background-only
chips are sampled deterministically using background_ratio (default 0.5).

Final external Aerial / Satellite / Drone test data are never accessed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import csv
import json
import random

import numpy as np
from PIL import Image


RASTER_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def _json_safe(
    value: Any,
):
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                _json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(item)
            for item
            in value
        ]

    return value


def save_json(
    data: dict,
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
            ensure_ascii=False,
        )

    return path


def _relative_key(
    file_path: Path,
    root_folder: Path,
) -> str:
    """
    Relative path without suffix.

    Examples:
        images/0001.tif        -> 0001
        images/a/0001.tif      -> a/0001
    """

    file_path = Path(
        file_path
    )

    root_folder = Path(
        root_folder
    )

    relative = file_path.relative_to(
        root_folder
    )

    return relative.with_suffix(
        ""
    ).as_posix()


def _discover_rasters(
    folder: Path,
) -> Dict[str, Path]:
    folder = Path(
        folder
    )

    if not folder.exists():
        return {}

    discovered: Dict[
        str,
        Path,
    ] = {}

    for path in sorted(
        folder.rglob(
            "*"
        )
    ):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in RASTER_EXTENSIONS
        ):
            continue

        key = _relative_key(
            path,
            folder,
        )

        discovered[
            key
        ] = path

    return discovered


# ============================================================
# RASTER IO
# ============================================================

def _read_raster_array(
    path: str | Path,
) -> np.ndarray:
    """
    Read ArcGIS-exported masks robustly.

    Fallback order:
        Pillow -> rasterio -> ArcPy
    """

    path = Path(
        path
    )

    errors = []

    try:
        with Image.open(
            path
        ) as image:
            return np.asarray(
                image
            )

    except Exception as error:
        errors.append(
            f"Pillow: {error}"
        )

    try:
        import rasterio

        with rasterio.open(
            path
        ) as dataset:
            array = dataset.read()

        if array.ndim == 3:
            return array[
                0
            ]

        return np.asarray(
            array
        )

    except Exception as error:
        errors.append(
            f"rasterio: {error}"
        )

    try:
        import arcpy

        array = arcpy.RasterToNumPyArray(
            str(
                path
            )
        )

        array = np.asarray(
            array
        )

        if (
            array.ndim == 3
            and array.shape[0] >= 1
        ):
            array = array[
                0
            ]

        return array

    except Exception as error:
        errors.append(
            f"ArcPy: {error}"
        )

    raise RuntimeError(
        "Unable to read raster:\n"
        f"{path}\n\n"
        + "\n".join(
            errors
        )
    )


def _raster_size(
    path: str | Path,
) -> Tuple[int, int]:
    """
    Return width, height with Pillow -> rasterio -> ArcPy fallback.
    """

    path = Path(
        path
    )

    errors = []

    try:
        with Image.open(
            path
        ) as image:
            width, height = image.size

        return (
            int(
                width
            ),
            int(
                height
            ),
        )

    except Exception as error:
        errors.append(
            f"Pillow: {error}"
        )

    try:
        import rasterio

        with rasterio.open(
            path
        ) as dataset:
            return (
                int(
                    dataset.width
                ),
                int(
                    dataset.height
                ),
            )

    except Exception as error:
        errors.append(
            f"rasterio: {error}"
        )

    try:
        import arcpy

        raster = arcpy.Raster(
            str(
                path
            )
        )

        return (
            int(
                raster.width
            ),
            int(
                raster.height
            ),
        )

    except Exception as error:
        errors.append(
            f"ArcPy: {error}"
        )

    raise RuntimeError(
        "Unable to determine raster dimensions:\n"
        f"{path}\n\n"
        + "\n".join(
            errors
        )
    )


def _mask_has_road(
    mask_path: str | Path,
) -> bool:
    array = _read_raster_array(
        mask_path
    )

    array = np.nan_to_num(
        np.asarray(
            array
        )
    )

    return bool(
        np.any(
            array > 0
        )
    )


def _create_empty_mask(
    image_path: str | Path,
    output_mask: str | Path,
) -> Path:
    """
    Create/recreate a zero-valued mask with the same dimensions
    as the source image.
    """

    image_path = Path(
        image_path
    )

    output_mask = Path(
        output_mask
    )

    output_mask.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    width, height = _raster_size(
        image_path
    )

    expected_size = (
        int(
            width
        ),
        int(
            height
        ),
    )

    if output_mask.exists():
        try:
            with Image.open(
                output_mask
            ) as existing:
                if existing.size == expected_size:
                    return output_mask
        except Exception:
            pass

    empty = Image.new(
        "L",
        expected_size,
        0,
    )

    empty.save(
        output_mask
    )

    return output_mask


# ============================================================
# PAIR DISCOVERY
# ============================================================

def discover_classified_tile_pairs(
    dataset_root: str | Path,
    images_folder: str = "images",
    labels_folder: str = "labels",
    strict: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Discover real image/label pairs.

    strict=True:
        raise when images or labels are unmatched.

    strict=False:
        return common pairs only.
        Missing-label images are handled later as background candidates
        by create_train_val_manifests().
    """

    dataset_root = Path(
        dataset_root
    )

    images_root = (
        dataset_root
        / images_folder
    )

    labels_root = (
        dataset_root
        / labels_folder
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            "Road dataset root was not found:\n"
            f"{dataset_root}"
        )

    if not images_root.exists():
        raise FileNotFoundError(
            "Road images folder was not found:\n"
            f"{images_root}"
        )

    if not labels_root.exists():
        raise FileNotFoundError(
            "Road labels folder was not found:\n"
            f"{labels_root}"
        )

    images = _discover_rasters(
        images_root
    )

    labels = _discover_rasters(
        labels_root
    )

    common_keys = sorted(
        set(
            images
        )
        & set(
            labels
        )
    )

    missing_label_keys = sorted(
        set(
            images
        )
        - set(
            labels
        )
    )

    missing_image_keys = sorted(
        set(
            labels
        )
        - set(
            images
        )
    )

    if strict:

        if missing_label_keys:
            examples = "\n".join(
                missing_label_keys[
                    :10
                ]
            )

            raise RuntimeError(
                f"{len(missing_label_keys)} image chip(s) do not have "
                "a corresponding Road label.\n\n"
                "Examples:\n"
                f"{examples}"
            )

        if missing_image_keys:
            examples = "\n".join(
                missing_image_keys[
                    :10
                ]
            )

            raise RuntimeError(
                f"{len(missing_image_keys)} Road label chip(s) do not have "
                "a corresponding image.\n\n"
                "Examples:\n"
                f"{examples}"
            )

    pairs = [
        (
            images[
                key
            ],
            labels[
                key
            ],
        )
        for key
        in common_keys
    ]

    if not pairs:
        raise RuntimeError(
            "No Road image/label pairs were found in:\n"
            f"{dataset_root}"
        )

    return pairs


# ============================================================
# SPLIT
# ============================================================

def split_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    train_percent: float = 80.0,
    validation_percent: float = 20.0,
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    pairs = list(
        pairs
    )

    train_percent = float(
        train_percent
    )

    validation_percent = float(
        validation_percent
    )

    if abs(
        (
            train_percent
            + validation_percent
        )
        - 100.0
    ) > 1e-8:
        raise ValueError(
            "train_percent + validation_percent must equal 100."
        )

    if not pairs:
        raise RuntimeError(
            "Cannot split an empty Road dataset."
        )

    rng = random.Random(
        int(
            seed
        )
    )

    shuffled = list(
        pairs
    )

    rng.shuffle(
        shuffled
    )

    if len(
        shuffled
    ) == 1:
        return (
            shuffled,
            [],
        )

    validation_count = int(
        round(
            len(
                shuffled
            )
            * (
                validation_percent
                / 100.0
            )
        )
    )

    validation_count = max(
        1,
        validation_count,
    )

    validation_count = min(
        validation_count,
        len(
            shuffled
        )
        - 1,
    )

    validation_pairs = shuffled[
        :validation_count
    ]

    train_pairs = shuffled[
        validation_count:
    ]

    return (
        train_pairs,
        validation_pairs,
    )


# ============================================================
# MANIFEST IO
# ============================================================

def write_manifest(
    pairs: Sequence[Tuple[Path, Path]],
    output_csv: str | Path,
    split_name: str,
) -> Path:
    """
    Preserve the Road manifest schema used by the current loaders:

        split
        sample_id
        image_path
        mask_path
    """

    output_csv = Path(
        output_csv
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "sample_id",
                "image_path",
                "mask_path",
            ],
        )

        writer.writeheader()

        for index, (
            image_path,
            mask_path,
        ) in enumerate(
            pairs
        ):

            writer.writerow(
                {
                    "split":
                        str(
                            split_name
                        ),

                    "sample_id":
                        f"{split_name}_{index:06d}",

                    "image_path":
                        str(
                            Path(
                                image_path
                            )
                        ),

                    "mask_path":
                        str(
                            Path(
                                mask_path
                            )
                        ),
                }
            )

    return output_csv


def read_manifest(
    manifest_csv: str | Path,
) -> List[Dict[str, str]]:
    manifest_csv = Path(
        manifest_csv
    )

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}"
        )

    with manifest_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# CONFIG RESOLUTION
# ============================================================

def _resolve_dataset_root(
    dataset_root: str | Path | None,
    dataset_info: dict | None,
) -> Path:
    if dataset_root is not None:
        return Path(
            dataset_root
        )

    dataset_info = dict(
        dataset_info or {}
    )

    value = (
        dataset_info.get(
            "path"
        )
        or dataset_info.get(
            "dataset_path"
        )
        or dataset_info.get(
            "folder"
        )
        or dataset_info.get(
            "root"
        )
    )

    if value is None:
        raise ValueError(
            "Road dataset path could not be resolved."
        )

    return Path(
        value
    )


def _resolve_split(
    train_percent: float | None,
    validation_percent: float | None,
    seed: int | None,
    general_params: dict | None,
) -> Tuple[
    float,
    float,
    int,
]:
    general_params = dict(
        general_params or {}
    )

    split = dict(
        general_params.get(
            "data_split",
            {},
        )
    )

    training = dict(
        general_params.get(
            "training",
            {},
        )
    )

    data = dict(
        general_params.get(
            "data",
            {},
        )
    )

    if train_percent is None:
        train_percent = split.get(
            "train_percent"
        )

    if validation_percent is None:
        validation_percent = split.get(
            "validation_percent"
        )

    if seed is None:
        seed = split.get(
            "seed",
            training.get(
                "seed",
                training.get(
                    "random_seed",
                    42,
                ),
            ),
        )

    if (
        train_percent is None
        and validation_percent is None
    ):
        validation_ratio = (
            training.get(
                "validation_ratio"
            )
        )

        if validation_ratio is None:
            validation_ratio = (
                data.get(
                    "validation_ratio",
                    general_params.get(
                        "validation_ratio",
                        0.20,
                    ),
                )
            )

        validation_ratio = float(
            validation_ratio
        )

        if validation_ratio > 1.0:
            validation_ratio = (
                validation_ratio
                / 100.0
            )

        validation_percent = (
            validation_ratio
            * 100.0
        )

        train_percent = (
            100.0
            - validation_percent
        )

    elif train_percent is None:
        validation_percent = float(
            validation_percent
        )

        train_percent = (
            100.0
            - validation_percent
        )

    elif validation_percent is None:
        train_percent = float(
            train_percent
        )

        validation_percent = (
            100.0
            - train_percent
        )

    train_percent = float(
        train_percent
    )

    validation_percent = float(
        validation_percent
    )

    if abs(
        (
            train_percent
            + validation_percent
        )
        - 100.0
    ) > 1e-8:
        raise ValueError(
            "Road Train/Validation percentages must sum to 100."
        )

    return (
        train_percent,
        validation_percent,
        int(
            seed
        ),
    )


def _resolve_background_ratio(
    background_ratio: float | None,
    general_params: dict | None,
    dataset_info: dict | None,
) -> float:
    if background_ratio is not None:
        return float(
            background_ratio
        )

    general_params = dict(
        general_params or {}
    )

    dataset_info = dict(
        dataset_info or {}
    )

    training = dict(
        general_params.get(
            "training",
            {},
        )
    )

    data = dict(
        general_params.get(
            "data",
            {},
        )
    )

    dataset = dict(
        general_params.get(
            "dataset",
            {},
        )
    )

    candidates = (
        dataset_info.get(
            "background_ratio"
        ),
        training.get(
            "background_ratio"
        ),
        data.get(
            "background_ratio"
        ),
        dataset.get(
            "background_ratio"
        ),
        general_params.get(
            "background_ratio"
        ),
        0.5,
    )

    for value in candidates:
        if value is not None:
            ratio = float(
                value
            )

            if ratio < 0:
                raise ValueError(
                    "background_ratio must be >= 0."
                )

            return ratio

    return 0.5


# ============================================================
# MAIN MANIFEST CREATOR
# ============================================================

def create_train_val_manifests(
    dataset_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    train_percent: float | None = None,
    validation_percent: float | None = None,
    seed: int | None = None,
    images_folder: str = "images",
    labels_folder: str = "labels",
    strict_pairing: bool = False,
    dataset_info: dict | None = None,
    general_params: dict | None = None,
    validation_ratio: float | None = None,
    background_ratio: float | None = None,
    **kwargs,
) -> Dict[str, object]:
    """
    Create Road Train/Validation manifests.

    Non-strict Road workflow:
        1. Keep every real positive Road mask.
        2. Treat unlabeled image chips as background candidates.
        3. Generate zero masks only for selected backgrounds.
        4. Sample backgrounds deterministically using background_ratio.
        5. Split selected development samples 80/20 (or configured values).

    strict_pairing=True is still available for datasets that are guaranteed
    to be one-to-one paired.
    """

    if output_folder is None:
        raise ValueError(
            "output_folder is required."
        )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_root = _resolve_dataset_root(
        dataset_root=
            dataset_root,

        dataset_info=
            dataset_info,
    )

    (
        train_percent,
        validation_percent,
        seed,
    ) = _resolve_split(
        train_percent=
            train_percent,

        validation_percent=
            validation_percent,

        seed=
            seed,

        general_params=
            general_params,
    )

    if (
        validation_ratio is not None
        and train_percent is None
        and validation_percent is None
    ):
        validation_percent = (
            float(
                validation_ratio
            )
            * 100.0
        )

        train_percent = (
            100.0
            - validation_percent
        )

    background_ratio = (
        _resolve_background_ratio(
            background_ratio=
                background_ratio,

            general_params=
                general_params,

            dataset_info=
                dataset_info,
        )
    )

    images_root = (
        dataset_root
        / images_folder
    )

    labels_root = (
        dataset_root
        / labels_folder
    )

    images = _discover_rasters(
        images_root
    )

    labels = _discover_rasters(
        labels_root
    )

    if not images:
        raise RuntimeError(
            "No Road image chips were found in:\n"
            f"{images_root}"
        )

    common_keys = sorted(
        set(
            images
        )
        & set(
            labels
        )
    )

    missing_label_keys = sorted(
        set(
            images
        )
        - set(
            labels
        )
    )

    extra_label_keys = sorted(
        set(
            labels
        )
        - set(
            images
        )
    )

    if strict_pairing:
        # Reuse the explicit strict validator/error messages.
        discover_classified_tile_pairs(
            dataset_root=
                dataset_root,

            images_folder=
                images_folder,

            labels_folder=
                labels_folder,

            strict=
                True,
        )

    # --------------------------------------------------------
    # Classify the real labelled masks
    # --------------------------------------------------------

    positive_pairs: List[
        Tuple[
            Path,
            Path,
        ]
    ] = []

    background_pairs_existing: List[
        Tuple[
            Path,
            Path,
        ]
    ] = []

    unreadable_labels = []

    for key in common_keys:

        image_path = images[
            key
        ]

        label_path = labels[
            key
        ]

        try:
            positive = _mask_has_road(
                label_path
            )

        except Exception as error:
            unreadable_labels.append(
                {
                    "sample":
                        key,

                    "label_path":
                        str(
                            label_path
                        ),

                    "error":
                        str(
                            error
                        ),
                }
            )

            continue

        pair = (
            image_path,
            label_path,
        )

        if positive:
            positive_pairs.append(
                pair
            )
        else:
            background_pairs_existing.append(
                pair
            )

    if unreadable_labels:
        preview = "\n".join(
            item[
                "label_path"
            ]
            for item
            in unreadable_labels[
                :10
            ]
        )

        raise RuntimeError(
            f"{len(unreadable_labels)} Road label mask(s) could not be read.\n\n"
            "Examples:\n"
            f"{preview}"
        )

    if not positive_pairs:
        raise RuntimeError(
            "No positive Road training masks were found. "
            "Training cannot continue."
        )

    # --------------------------------------------------------
    # Missing-label image chips are background candidates only
    # in non-strict mode.
    # --------------------------------------------------------

    available_background_count = (
        len(
            background_pairs_existing
        )
        + len(
            missing_label_keys
        )
    )

    desired_background_count = int(
        round(
            len(
                positive_pairs
            )
            * float(
                background_ratio
            )
        )
    )

    desired_background_count = min(
        desired_background_count,
        available_background_count,
    )

    rng = random.Random(
        int(
            seed
        )
    )

    background_candidates = []

    for image_path, mask_path in background_pairs_existing:
        background_candidates.append(
            (
                "existing",
                image_path,
                mask_path,
                None,
            )
        )

    for key in missing_label_keys:
        background_candidates.append(
            (
                "missing",
                images[
                    key
                ],
                None,
                key,
            )
        )

    rng.shuffle(
        background_candidates
    )

    selected_candidates = background_candidates[
        :desired_background_count
    ]

    generated_root = (
        output_folder
        / "generated_empty_masks"
    )

    selected_background_pairs: List[
        Tuple[
            Path,
            Path,
        ]
    ] = []

    generated_empty_count = 0

    for (
        kind,
        image_path,
        mask_path,
        relative_key,
    ) in selected_candidates:

        if kind == "existing":
            selected_background_pairs.append(
                (
                    image_path,
                    mask_path,
                )
            )

            continue

        relative_path = Path(
            str(
                relative_key
            )
        )

        output_mask = (
            generated_root
            / relative_path.parent
            / (
                relative_path.name
                + "_empty.png"
            )
        )

        empty_mask = _create_empty_mask(
            image_path=
                image_path,

            output_mask=
                output_mask,
        )

        generated_empty_count += 1

        selected_background_pairs.append(
            (
                image_path,
                empty_mask,
            )
        )

    # --------------------------------------------------------
    # Keep every positive + controlled background subset
    # --------------------------------------------------------

    selected_pairs = (
        list(
            positive_pairs
        )
        + list(
            selected_background_pairs
        )
    )

    if not selected_pairs:
        raise RuntimeError(
            "No Road samples were selected for training."
        )

    (
        train_pairs,
        validation_pairs,
    ) = split_pairs(
        pairs=
            selected_pairs,

        train_percent=
            train_percent,

        validation_percent=
            validation_percent,

        seed=
            seed,
    )

    train_manifest = write_manifest(
        pairs=
            train_pairs,

        output_csv=
            output_folder
            / "train_manifest.csv",

        split_name=
            "train",
    )

    val_manifest = write_manifest(
        pairs=
            validation_pairs,

        output_csv=
            output_folder
            / "val_manifest.csv",

        split_name=
            "validation",
    )

    # --------------------------------------------------------
    # Audit outputs
    # --------------------------------------------------------

    split_summary = {
        "dataset_root":
            str(
                dataset_root
            ),

        "images_folder":
            str(
                images_folder
            ),

        "labels_folder":
            str(
                labels_folder
            ),

        "strict_pairing":
            bool(
                strict_pairing
            ),

        "train_percent":
            float(
                train_percent
            ),

        "validation_percent":
            float(
                validation_percent
            ),

        "train_ratio":
            float(
                train_percent
                / 100.0
            ),

        "validation_ratio":
            float(
                validation_percent
                / 100.0
            ),

        "seed":
            int(
                seed
            ),

        "background_ratio":
            float(
                background_ratio
            ),

        "image_count":
            int(
                len(
                    images
                )
            ),

        "label_count":
            int(
                len(
                    labels
                )
            ),

        "paired_count":
            int(
                len(
                    common_keys
                )
            ),

        "positive_road_count":
            int(
                len(
                    positive_pairs
                )
            ),

        "existing_background_label_count":
            int(
                len(
                    background_pairs_existing
                )
            ),

        "missing_label_background_candidates":
            int(
                len(
                    missing_label_keys
                )
            ),

        "extra_label_count":
            int(
                len(
                    extra_label_keys
                )
            ),

        "available_background_count":
            int(
                available_background_count
            ),

        "selected_background_count":
            int(
                len(
                    selected_background_pairs
                )
            ),

        "generated_empty_mask_count":
            int(
                generated_empty_count
            ),

        "total_selected_count":
            int(
                len(
                    selected_pairs
                )
            ),

        "train_count":
            int(
                len(
                    train_pairs
                )
            ),

        "validation_count":
            int(
                len(
                    validation_pairs
                )
            ),

        "test_included_in_split":
            False,

        "train_manifest_csv":
            str(
                train_manifest
            ),

        "val_manifest_csv":
            str(
                val_manifest
            ),
    }

    split_summary_path = save_json(
        split_summary,
        output_folder
        / "split_summary.json",
    )

    data_quality_summary = {
        "dataset_path":
            str(
                dataset_root
            ),

        "image_count":
            int(
                len(
                    images
                )
            ),

        "label_count":
            int(
                len(
                    labels
                )
            ),

        "paired_count":
            int(
                len(
                    common_keys
                )
            ),

        "missing_label_count":
            int(
                len(
                    missing_label_keys
                )
            ),

        "extra_label_count":
            int(
                len(
                    extra_label_keys
                )
            ),

        "positive_road_chips":
            int(
                len(
                    positive_pairs
                )
            ),

        "available_background_chips":
            int(
                available_background_count
            ),

        "selected_background_chips":
            int(
                len(
                    selected_background_pairs
                )
            ),

        "generated_empty_masks":
            int(
                generated_empty_count
            ),

        "background_ratio":
            float(
                background_ratio
            ),

        "strict_pairing":
            bool(
                strict_pairing
            ),
    }

    data_quality_summary_path = save_json(
        data_quality_summary,
        output_folder
        / "data_quality_summary.json",
    )

    print()
    print(
        "=" * 72
    )
    print(
        "ROAD DATASET / MANIFEST SUMMARY"
    )
    print(
        "=" * 72
    )
    print(
        f"Image chips                  : {len(images)}"
    )
    print(
        f"Original Road labels         : {len(labels)}"
    )
    print(
        f"Matched image/label chips    : {len(common_keys)}"
    )
    print(
        f"Positive Road chips          : {len(positive_pairs)}"
    )
    print(
        f"Missing-label backgrounds    : {len(missing_label_keys)}"
    )
    print(
        f"Background ratio             : {background_ratio:.3f}"
    )
    print(
        f"Selected background chips    : {len(selected_background_pairs)}"
    )
    print(
        f"Generated empty masks        : {generated_empty_count}"
    )
    print(
        f"Total selected development   : {len(selected_pairs)}"
    )
    print(
        f"Train / Validation           : {len(train_pairs)} / {len(validation_pairs)}"
    )
    print(
        f"Split seed                   : {seed}"
    )
    print(
        "Final external test included : NO"
    )
    print(
        "=" * 72
    )
    print()

    return {
        "train_manifest":
            train_manifest,

        "train_manifest_csv":
            train_manifest,

        "validation_manifest":
            val_manifest,

        "val_manifest":
            val_manifest,

        "val_manifest_csv":
            val_manifest,

        "split_summary":
            split_summary,

        "split_summary_json":
            split_summary_path,

        "data_quality_summary":
            data_quality_summary,

        "data_quality_summary_json":
            data_quality_summary_path,

        "positive_count":
            int(
                len(
                    positive_pairs
                )
            ),

        "selected_background_count":
            int(
                len(
                    selected_background_pairs
                )
            ),

        "generated_empty_mask_count":
            int(
                generated_empty_count
            ),

        "train_count":
            int(
                len(
                    train_pairs
                )
            ),

        "validation_count":
            int(
                len(
                    validation_pairs
                )
            ),
    }


__all__ = [
    "create_train_val_manifests",
    "discover_classified_tile_pairs",
    "read_manifest",
    "save_json",
    "split_pairs",
    "write_manifest",
]

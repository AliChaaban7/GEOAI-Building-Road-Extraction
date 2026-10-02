"""Road experiment preparation helpers.

This module is Road-specific.

Important architecture
----------------------
Road training manifests are created by:

    Roads/src/data/manifests.py

NOT by Roads/src/utils.py.

The Road manifest API is responsible for:
- image/mask pairing
- deterministic Train/Validation split
- writing train_manifest.csv
- writing val_manifest.csv
- split metadata

External final-test imagery is never used here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import json
import os
import shutil


# ============================================================
# EXISTING ROAD UTILITIES
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_datasets_config,
    get_dataset_info,
    create_experiment_output_folder,
)

# Road-specific manifest implementation.
from src.data.manifests import (
    create_train_val_manifests,
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_json_if_exists(
    path: str | Path,
):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return load_json(
            path
        )
    except Exception:
        return None


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
            _json_safe(
                data
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ============================================================
# SPLIT RESOLUTION
# ============================================================

def apply_runtime_split_overrides(
    experiment_config: dict,
    general_params: dict,
) -> dict:
    """
    Apply the split selected in the Streamlit UI without changing
    config/general_params.json on disk.
    """

    experiment_config = dict(
        experiment_config or {}
    )

    # JSON-style deep copy.
    general_params = json.loads(
        json.dumps(
            general_params or {}
        )
    )

    override = (
        experiment_config.get(
            "data_split"
        )
    )

    if not isinstance(
        override,
        dict,
    ):
        return general_params

    split = dict(
        general_params.get(
            "data_split",
            {},
        )
    )

    for key in (
        "train_percent",
        "validation_percent",
        "seed",
    ):
        value = override.get(
            key
        )

        if value is not None:
            split[key] = value

    general_params[
        "data_split"
    ] = split

    # Use the same seed for stochastic training when the UI explicitly
    # selects a split seed.
    if split.get(
        "seed"
    ) is not None:

        general_params.setdefault(
            "training",
            {},
        )

        general_params[
            "training"
        ][
            "seed"
        ] = int(
            split["seed"]
        )

        general_params[
            "training"
        ][
            "random_seed"
        ] = int(
            split["seed"]
        )

    return general_params


def resolve_train_validation_split(
    general_params: dict | None = None,
) -> dict:
    """
    Road-local split resolver.

    Preferred:
        general_params["data_split"] = {
            "train_percent": 80,
            "validation_percent": 20,
            "seed": 42
        }

    Backward-compatible fallbacks are supported.
    """

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

    train_percent = split.get(
        "train_percent"
    )

    validation_percent = split.get(
        "validation_percent"
    )

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

    # Legacy validation-ratio support.
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
                general_params.get(
                    "validation_ratio",
                    0.20,
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

    if not (
        0.0
        < train_percent
        < 100.0
    ):
        raise ValueError(
            "train_percent must be between 0 and 100. "
            f"Received: {train_percent}"
        )

    if not (
        0.0
        < validation_percent
        < 100.0
    ):
        raise ValueError(
            "validation_percent must be between 0 and 100. "
            f"Received: {validation_percent}"
        )

    if abs(
        (
            train_percent
            + validation_percent
        )
        - 100.0
    ) > 1e-8:
        raise ValueError(
            "Training and Validation percentages must sum to 100. "
            f"Received: {train_percent} + {validation_percent}"
        )

    return {
        "train_percent":
            train_percent,

        "validation_percent":
            validation_percent,

        "train_ratio":
            train_percent
            / 100.0,

        "validation_ratio":
            validation_percent
            / 100.0,

        "seed":
            int(
                seed
            ),
    }


# ============================================================
# DATASET HELPERS
# ============================================================

def _looks_like_path(
    value: Any,
) -> bool:
    if isinstance(
        value,
        Path,
    ):
        return True

    if not isinstance(
        value,
        str,
    ):
        return False

    value = value.strip()

    if not value:
        return False

    return (
        "\\" in value
        or "/" in value
        or (
            len(value) >= 2
            and value[1] == ":"
        )
    )


def normalize_dataset_info(
    raw_dataset_info: Any,
    experiment_config: dict | None = None,
) -> dict:
    """
    Normalize the return value of Roads/src/utils.py::get_dataset_info().

    Supported return styles:
        dict
        (dataset_id, dict)
        (dict, dataset_id)
        (dataset_path, dict)
        (dataset_id, dataset_path, dict)
        list/tuple containing one dataset dictionary

    The current Road helper may return a tuple; the training pipeline itself
    should always work with one normalized dictionary.
    """

    experiment_config = dict(
        experiment_config or {}
    )

    if isinstance(
        raw_dataset_info,
        dict,
    ):
        info = dict(
            raw_dataset_info
        )

        extra_values = []

    elif isinstance(
        raw_dataset_info,
        (tuple, list),
    ):
        values = list(
            raw_dataset_info
        )

        dict_candidates = [
            value
            for value
            in values
            if isinstance(
                value,
                dict,
            )
        ]

        if not dict_candidates:
            raise TypeError(
                "get_dataset_info() returned a tuple/list but it did not "
                "contain a dataset dictionary.\n"
                f"Received: {raw_dataset_info!r}"
            )

        # Prefer the dictionary that contains an actual dataset path.
        info = None

        for candidate in dict_candidates:
            if any(
                candidate.get(
                    key
                )
                for key
                in (
                    "path",
                    "dataset_path",
                    "folder",
                    "root",
                )
            ):
                info = dict(
                    candidate
                )
                break

        if info is None:
            info = dict(
                dict_candidates[0]
            )

        extra_values = [
            value
            for value
            in values
            if not isinstance(
                value,
                dict,
            )
        ]

    else:
        raise TypeError(
            "get_dataset_info() must return a dict or a tuple/list "
            "containing a dict.\n"
            f"Received type: {type(raw_dataset_info)}\n"
            f"Received value: {raw_dataset_info!r}"
        )

    # --------------------------------------------------------
    # Recover path from tuple/list when the dict does not carry it.
    # --------------------------------------------------------

    dataset_path = (
        info.get(
            "path"
        )
        or info.get(
            "dataset_path"
        )
        or info.get(
            "folder"
        )
        or info.get(
            "root"
        )
    )

    if dataset_path is None:
        for value in extra_values:
            if _looks_like_path(
                value
            ):
                dataset_path = value
                break

    if dataset_path is None:
        raise ValueError(
            "Road dataset information was resolved, but no dataset path "
            "could be found.\n"
            "Expected one of: path, dataset_path, folder, root.\n"
            f"Raw get_dataset_info() result: {raw_dataset_info!r}"
        )

    info[
        "path"
    ] = str(
        dataset_path
    )

    # --------------------------------------------------------
    # Recover dataset id from tuple/list when available.
    # --------------------------------------------------------

    dataset_id = (
        info.get(
            "dataset_id"
        )
        or info.get(
            "id"
        )
        or experiment_config.get(
            "dataset_id"
        )
    )

    if dataset_id is None:
        for value in extra_values:

            if isinstance(
                value,
                str,
            ) and not _looks_like_path(
                value
            ):
                dataset_id = value
                break

    if dataset_id is None:
        tile_size = (
            experiment_config.get(
                "tile_size"
            )
        )

        source_type = str(
            experiment_config.get(
                "source_type",
                "mixed",
            )
        ).lower()

        dataset_type = str(
            experiment_config.get(
                "dataset_type",
                "ct",
            )
        ).lower()

        if tile_size is not None:
            dataset_id = (
                f"{source_type}_{dataset_type}_{tile_size}"
            )

    if dataset_id is not None:
        info[
            "dataset_id"
        ] = str(
            dataset_id
        )

    return info


def resolve_dataset_root(
    dataset_info: dict,
) -> Path:
    dataset_info = normalize_dataset_info(
        dataset_info
    )

    dataset_root = (
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

    if dataset_root is None:
        raise ValueError(
            "Road dataset path could not be resolved. "
            "Expected one of: path, dataset_path, folder, root."
        )

    return Path(
        dataset_root
    )


def _same_path(
    first,
    second,
) -> bool:
    try:
        return (
            os.path.normcase(
                os.path.normpath(
                    os.path.abspath(
                        str(
                            first
                        )
                    )
                )
            )
            ==
            os.path.normcase(
                os.path.normpath(
                    os.path.abspath(
                        str(
                            second
                        )
                    )
                )
            )
        )
    except Exception:
        return False


def normalize_output_folder(
    raw_output: Any,
    roads_root: Path,
    experiment_config: dict,
) -> Path:
    """
    Normalize create_experiment_output_folder() return styles.

    Supported:
        path/string
        (path, ...)
        {"output_folder": path}
    """

    if isinstance(
        raw_output,
        dict,
    ):
        for key in (
            "output_folder",
            "path",
            "folder",
        ):
            value = raw_output.get(
                key
            )

            if value:
                return Path(
                    value
                )

    if isinstance(
        raw_output,
        (str, Path),
    ):
        return Path(
            raw_output
        )

    if isinstance(
        raw_output,
        (tuple, list),
    ):
        for value in raw_output:

            if isinstance(
                value,
                (str, Path),
            ) and _looks_like_path(
                value
            ):
                return Path(
                    value
                )

    experiment_id = str(
        experiment_config.get(
            "experiment_id",
            "",
        )
    ).strip()

    if not experiment_id:
        raise ValueError(
            "Could not resolve Road experiment output folder because "
            "experiment_id is missing."
        )

    # Canonical Road fallback.
    return (
        Path(
            roads_root
        )
        / "outputs"
        / "experiments"
        / experiment_id
    )


# ============================================================
# MANIFEST COMPATIBILITY
# ============================================================

def manifests_match_requested_split(
    output_folder: str | Path,
    dataset_info: dict,
    split_info: dict,
) -> bool:
    """
    Reuse existing manifests only when:
    - train_manifest.csv exists
    - val_manifest.csv exists
    - split percentages match
    - seed matches
    - saved dataset path matches
    """

    output_folder = Path(
        output_folder
    )

    train_manifest = (
        output_folder
        / "train_manifest.csv"
    )

    val_manifest = (
        output_folder
        / "val_manifest.csv"
    )

    if (
        not train_manifest.exists()
        or not val_manifest.exists()
    ):
        return False

    split_summary = (
        load_json_if_exists(
            output_folder
            / "split_summary.json"
        )
        or {}
    )

    try:
        split_matches = (
            abs(
                float(
                    split_summary.get(
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
                    split_summary.get(
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
                split_summary.get(
                    "seed"
                )
            )
            == int(
                split_info[
                    "seed"
                ]
            )
        )

    except Exception:
        split_matches = False

    if not split_matches:
        return False

    saved_dataset = (
        load_json_if_exists(
            output_folder
            / "dataset_used.json"
        )
    )

    if not isinstance(
        saved_dataset,
        dict,
    ):
        # Old experiment folders may not have dataset_used.json.
        # Regenerate once so the current Road workflow becomes auditable.
        return False

    requested_root = (
        resolve_dataset_root(
            dataset_info
        )
    )

    saved_root = (
        saved_dataset.get(
            "path"
        )
        or saved_dataset.get(
            "dataset_path"
        )
        or saved_dataset.get(
            "folder"
        )
        or saved_dataset.get(
            "root"
        )
    )

    if saved_root is None:
        return False

    return _same_path(
        saved_root,
        requested_root,
    )


def backup_existing_split_files(
    output_folder: str | Path,
):
    """
    Preserve the first previous split as a rollback baseline.
    """

    output_folder = Path(
        output_folder
    )

    names = (
        "train_manifest.csv",
        "val_manifest.csv",
        "split_summary.json",
        "dataset_used.json",
        "experiment_config_used.json",
        "general_params_used.json",
    )

    existing = [
        output_folder
        / name

        for name
        in names

        if (
            output_folder
            / name
        ).exists()
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
# ROAD MANIFEST API COMPATIBILITY
# ============================================================

def _manifest_path_from_result(
    result: Any,
    output_folder: Path,
    kind: str,
) -> Path:
    """
    Normalize manifest results from different Road manifest API versions.
    """

    output_folder = Path(
        output_folder
    )

    if kind == "train":
        candidates = (
            "train_manifest",
            "train_manifest_csv",
            "train_csv",
        )

        fallback = (
            output_folder
            / "train_manifest.csv"
        )

    else:
        candidates = (
            "validation_manifest",
            "val_manifest",
            "validation_manifest_csv",
            "val_manifest_csv",
            "val_csv",
        )

        fallback = (
            output_folder
            / "val_manifest.csv"
        )

    if isinstance(
        result,
        dict,
    ):
        for key in candidates:

            value = result.get(
                key
            )

            if value:
                return Path(
                    value
                )

    elif isinstance(
        result,
        (tuple, list),
    ):
        path_values = [
            Path(
                value
            )
            for value
            in result
            if isinstance(
                value,
                (str, Path),
            )
            and _looks_like_path(
                value
            )
        ]

        if kind == "train":
            for value in path_values:
                if "train" in value.name.lower():
                    return value
        else:
            for value in path_values:
                name = value.name.lower()

                if (
                    "val" in name
                    or "validation" in name
                ):
                    return value

    return fallback


def call_create_train_val_manifests(
    dataset_info: dict,
    dataset_root: Path,
    output_folder: Path,
    general_params: dict,
    split_info: dict,
    strict_pairing: bool,
):
    """
    Call the installed Roads/src/data/manifests.py implementation using
    its real function signature.

    This avoids hard-coding one historical API version.
    """

    function = (
        create_train_val_manifests
    )

    signature = inspect.signature(
        function
    )

    parameters = signature.parameters

    kwargs = {}

    # --------------------------------------------------------
    # Dataset argument
    # --------------------------------------------------------

    if "dataset_info" in parameters:
        kwargs[
            "dataset_info"
        ] = dataset_info

    elif "dataset_root" in parameters:
        kwargs[
            "dataset_root"
        ] = dataset_root

    elif "dataset_path" in parameters:
        kwargs[
            "dataset_path"
        ] = dataset_root

    elif "root" in parameters:
        kwargs[
            "root"
        ] = dataset_root

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    if "output_folder" in parameters:
        kwargs[
            "output_folder"
        ] = output_folder

    elif "output_dir" in parameters:
        kwargs[
            "output_dir"
        ] = output_folder

    # --------------------------------------------------------
    # Split configuration
    # --------------------------------------------------------

    if "general_params" in parameters:
        kwargs[
            "general_params"
        ] = general_params

    if "train_percent" in parameters:
        kwargs[
            "train_percent"
        ] = float(
            split_info[
                "train_percent"
            ]
        )

    if "validation_percent" in parameters:
        kwargs[
            "validation_percent"
        ] = float(
            split_info[
                "validation_percent"
            ]
        )

    if "validation_ratio" in parameters:
        kwargs[
            "validation_ratio"
        ] = float(
            split_info[
                "validation_ratio"
            ]
        )

    if "seed" in parameters:
        kwargs[
            "seed"
        ] = int(
            split_info[
                "seed"
            ]
        )

    # --------------------------------------------------------
    # Road folder names / pairing policy
    # --------------------------------------------------------

    if "images_folder" in parameters:
        kwargs[
            "images_folder"
        ] = str(
            dataset_info.get(
                "images_folder",
                "images",
            )
        )

    if "labels_folder" in parameters:
        kwargs[
            "labels_folder"
        ] = str(
            dataset_info.get(
                "labels_folder",
                "labels",
            )
        )

    if "strict_pairing" in parameters:
        kwargs[
            "strict_pairing"
        ] = bool(
            strict_pairing
        )

    # --------------------------------------------------------
    # Run exact installed API
    # --------------------------------------------------------

    try:
        return function(
            **kwargs
        )

    except TypeError as error:
        raise TypeError(
            "Road manifest API call failed.\n"
            f"Installed signature: {signature}\n"
            f"Arguments supplied: {sorted(kwargs.keys())}\n"
            f"Original error: {error}"
        ) from error


def write_split_summary(
    output_folder: Path,
    split_info: dict,
    train_manifest: Path,
    val_manifest: Path,
    dataset_info: dict,
) -> Path:
    """
    Ensure the Road experiment always has auditable split metadata,
    even when an older manifest implementation does not create it.
    """

    path = (
        Path(
            output_folder
        )
        / "split_summary.json"
    )

    payload = {
        "train_percent":
            float(
                split_info[
                    "train_percent"
                ]
            ),

        "validation_percent":
            float(
                split_info[
                    "validation_percent"
                ]
            ),

        "train_ratio":
            float(
                split_info[
                    "train_ratio"
                ]
            ),

        "validation_ratio":
            float(
                split_info[
                    "validation_ratio"
                ]
            ),

        "seed":
            int(
                split_info[
                    "seed"
                ]
            ),

        "train_manifest_csv":
            str(
                train_manifest
            ),

        "val_manifest_csv":
            str(
                val_manifest
            ),

        "dataset_id":
            dataset_info.get(
                "dataset_id"
            ),

        "dataset_path":
            dataset_info.get(
                "path"
            ),

        "test_included_in_split":
            False,
    }

    return save_json(
        payload,
        path,
    )


# ============================================================
# MODEL BACKEND
# ============================================================

def resolve_model_backend(
    experiment_config: dict,
) -> str:
    """
    Resolve the execution backend expected by Roads/scripts/run_train.py.

    Current Road model families:
        U-Net                     -> pytorch
        DeepLabV3+                -> pytorch
        SAM-LoRA                  -> pytorch
        ConnectNet                -> arcgis_learn
        Multi Task Road Extractor -> arcgis_learn

    An explicit backend/model_backend value in experiment.json takes
    precedence when present.
    """

    experiment_config = dict(
        experiment_config or {}
    )

    explicit = (
        experiment_config.get(
            "model_backend"
        )
        or experiment_config.get(
            "backend"
        )
        or experiment_config.get(
            "framework"
        )
    )

    if explicit is not None:
        value = (
            str(
                explicit
            )
            .strip()
            .lower()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

        aliases = {
            "torch":
                "pytorch",

            "pytorch":
                "pytorch",

            "arcgis":
                "arcgis_learn",

            "arcgislearn":
                "arcgis_learn",

            "arcgis_learn":
                "arcgis_learn",
        }

        if value in aliases:
            return aliases[
                value
            ]

    model_type = (
        str(
            experiment_config.get(
                "model_type",
                "",
            )
        )
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    aliases = {
        "u_net":
            "unet",

        "deeplab":
            "deeplabv3",

        "deeplabv3+":
            "deeplabv3",

        "deeplabv3plus":
            "deeplabv3",

        "deep_lab_v3":
            "deeplabv3",

        "connect_net":
            "connectnet",

        "multi_task_road_extractor":
            "multitask_road_extractor",

        "multi_task":
            "multitask_road_extractor",

        "samlora":
            "sam_lora",
    }

    model_type = aliases.get(
        model_type,
        model_type,
    )

    if model_type in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:
        return "pytorch"

    if model_type in {
        "connectnet",
        "multitask_road_extractor",
    }:
        return "arcgis_learn"

    raise ValueError(
        "Could not resolve Road model backend for "
        f"model_type={experiment_config.get('model_type')!r}."
    )


# ============================================================
# PREPARE EXPERIMENT
# ============================================================

def prepare_experiment_split(
    roads_root: str | Path,
    config_path: str | Path,
    *,
    force_regenerate_split: bool = False,
    strict_pairing: bool | None = None,
) -> dict:
    """
    Resolve one Road experiment and create/reuse deterministic manifests.

    Final Aerial/Satellite/Drone testing data are not touched here.
    """

    roads_root = Path(
        roads_root
    ).resolve()

    config_path = Path(
        config_path
    ).resolve()

    experiment_config = (
        load_experiment_config(
            str(
                config_path
            )
        )
    )

    general_params_path = (
        roads_root
        / "config"
        / "general_params.json"
    )

    general_params = (
        load_general_params(
            general_params_path
        )
    )

    general_params = (
        apply_runtime_split_overrides(
            experiment_config=
                experiment_config,

            general_params=
                general_params,
        )
    )

    datasets_config_path = (
        roads_root
        / "config"
        / "datasets.json"
    )

    datasets_config = (
        load_datasets_config(
            datasets_config_path
        )
    )

    raw_dataset_info = (
        get_dataset_info(
            experiment_config=
                experiment_config,

            datasets_config=
                datasets_config,
        )
    )

    dataset_info = (
        normalize_dataset_info(
            raw_dataset_info=
                raw_dataset_info,

            experiment_config=
                experiment_config,
        )
    )

    dataset_root = (
        resolve_dataset_root(
            dataset_info
        )
    )

    # --------------------------------------------------------
    # EXPERIMENT OUTPUT FOLDER
    # --------------------------------------------------------
    # Do NOT call src.utils.create_experiment_output_folder() here.
    #
    # The Roads codebase has had multiple incompatible versions of that
    # helper (different arguments / return styles). The canonical Road
    # experiment location is stable and unambiguous, so build it directly:
    #
    #     Roads/outputs/experiments/<experiment_id>
    #
    # This removes the API mismatch completely.
    # --------------------------------------------------------

    experiment_id = str(
        experiment_config.get(
            "experiment_id",
            "",
        )
    ).strip()

    if not experiment_id:
        raise ValueError(
            "experiment.json must contain a non-empty experiment_id."
        )

    output_folder = (
        roads_root
        / "outputs"
        / "experiments"
        / experiment_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_info = (
        resolve_train_validation_split(
            general_params=
                general_params
        )
    )

    if strict_pairing is None:

        strict_pairing = (
            experiment_config.get(
                "strict_pairing"
            )
        )

        if strict_pairing is None:
            strict_pairing = (
                dataset_info.get(
                    "strict_pairing"
                )
            )

        # ----------------------------------------------------
        # ROAD CLASSIFIED-TILES DEFAULT
        # ----------------------------------------------------
        # ArcGIS Road exports can legitimately contain image chips
        # with no label file when the chip has no Road feature.
        #
        # In the Road workflow these are handled as background
        # candidates by src.data.manifests, which generates zero
        # masks for a controlled subset (background_ratio, default 0.5).
        #
        # Explicit experiment/dataset settings still override this.
        # ----------------------------------------------------
        if strict_pairing is None:
            strict_pairing = False

    strict_pairing = bool(
        strict_pairing
    )

    reusable = (
        manifests_match_requested_split(
            output_folder=
                output_folder,

            dataset_info=
                dataset_info,

            split_info=
                split_info,
        )
    )

    regenerate = (
        bool(
            force_regenerate_split
        )
        or not reusable
    )

    if regenerate:

        backup_existing_split_files(
            output_folder
        )

        manifest_result = (
            call_create_train_val_manifests(
                dataset_info=
                    dataset_info,

                dataset_root=
                    dataset_root,

                output_folder=
                    output_folder,

                general_params=
                    general_params,

                split_info=
                    split_info,

                strict_pairing=
                    strict_pairing,
            )
        )

        train_manifest = (
            _manifest_path_from_result(
                manifest_result,
                output_folder,
                "train",
            )
        )

        val_manifest = (
            _manifest_path_from_result(
                manifest_result,
                output_folder,
                "validation",
            )
        )

    else:

        manifest_result = {}

        train_manifest = (
            output_folder
            / "train_manifest.csv"
        )

        val_manifest = (
            output_folder
            / "val_manifest.csv"
        )

    if not train_manifest.exists():
        raise FileNotFoundError(
            "Road training manifest was not created/found:\n"
            f"{train_manifest}"
        )

    if not val_manifest.exists():
        raise FileNotFoundError(
            "Road validation manifest was not created/found:\n"
            f"{val_manifest}"
        )

    write_split_summary(
        output_folder=
            output_folder,

        split_info=
            split_info,

        train_manifest=
            train_manifest,

        val_manifest=
            val_manifest,

        dataset_info=
            dataset_info,
    )

    # Reproducibility/audit files.
    save_json(
        experiment_config,
        output_folder
        / "experiment_config_used.json",
    )

    save_json(
        general_params,
        output_folder
        / "general_params_used.json",
    )

    save_json(
        dataset_info,
        output_folder
        / "dataset_used.json",
    )

    model_backend = (
        resolve_model_backend(
            experiment_config
        )
    )

    context = {
        "roads_root":
            roads_root,

        "config_path":
            config_path,

        "experiment_config":
            experiment_config,

        "model_type":
            experiment_config.get(
                "model_type"
            ),

        "model_backend":
            model_backend,

        "backbone":
            experiment_config.get(
                "backbone"
            ),

        "source_type":
            experiment_config.get(
                "source_type",
                "mixed",
            ),

        "tile_size":
            experiment_config.get(
                "tile_size"
            ),

        "general_params":
            general_params,

        "general_params_path":
            general_params_path,

        "datasets_config":
            datasets_config,

        "datasets_config_path":
            datasets_config_path,

        "dataset_info":
            dataset_info,

        "dataset_id":
            dataset_info.get(
                "dataset_id"
            ),

        "dataset_root":
            dataset_root,

        "dataset_path":
            dataset_root,

        "output_folder":
            output_folder,

        "train_manifest":
            train_manifest,

        "train_manifest_csv":
            train_manifest,

        "val_manifest":
            val_manifest,

        "val_manifest_csv":
            val_manifest,

        "validation_manifest":
            val_manifest,

        "split_info":
            split_info,

        "strict_pairing":
            strict_pairing,

        "manifests_regenerated":
            bool(
                regenerate
            ),

        "manifest_result":
            manifest_result,
    }

    return context


def prepare_experiment(
    config_path: str | Path | None = None,
    roads_root: str | Path | None = None,
    root: str | Path | None = None,
    force_regenerate_split: bool = False,
    strict_pairing: bool | None = None,
    **kwargs,
) -> dict:
    """
    Compatibility entry point used by Road launchers.

    Supported examples:

        prepare_experiment(config_path)

        prepare_experiment(
            config_path=config_path
        )

        prepare_experiment(
            roads_root=ROADS_ROOT,
            config_path=config_path
        )
    """

    if roads_root is None:
        roads_root = root

    if roads_root is None:
        roads_root = (
            Path(
                __file__
            )
            .resolve()
            .parents[2]
        )

    roads_root = Path(
        roads_root
    )

    if config_path is None:
        config_path = (
            roads_root
            / "config"
            / "experiment.json"
        )

    return prepare_experiment_split(
        roads_root=
            roads_root,

        config_path=
            config_path,

        force_regenerate_split=
            force_regenerate_split,

        strict_pairing=
            strict_pairing,
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_experiment_summary(
    context: dict,
) -> None:
    """
    Print a concise Road experiment summary for training/validation launchers.
    """

    experiment = dict(
        context.get(
            "experiment_config",
            {},
        )
    )

    split = dict(
        context.get(
            "split_info",
            {},
        )
    )

    dataset_info = dict(
        context.get(
            "dataset_info",
            {},
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "ROAD EXPERIMENT"
    )
    print(
        "=" * 72
    )

    print(
        "Experiment ID :",
        experiment.get(
            "experiment_id",
            "unknown",
        ),
    )

    print(
        "Model         :",
        experiment.get(
            "model_type",
            "unknown",
        ),
    )

    print(
        "Backend       :",
        context.get(
            "model_backend",
            "unknown",
        ),
    )

    print(
        "Backbone      :",
        experiment.get(
            "backbone",
            "unknown",
        ),
    )

    print(
        "Source        :",
        experiment.get(
            "source_type",
            "mixed",
        ),
    )

    print(
        "Tile Size     :",
        experiment.get(
            "tile_size",
            "unknown",
        ),
    )

    print(
        "Dataset ID    :",
        (
            dataset_info.get(
                "dataset_id"
            )
            or context.get(
                "dataset_id"
            )
            or "unknown"
        ),
    )

    print(
        "Dataset Root  :",
        context.get(
            "dataset_root"
        ),
    )

    print(
        "Train / Val   :",
        (
            f"{float(split.get('train_percent', 0)):.0f}% / "
            f"{float(split.get('validation_percent', 0)):.0f}%"
        ),
    )

    print(
        "Split Seed    :",
        split.get(
            "seed",
            "unknown",
        ),
    )

    print(
        "Pairing       :",
        (
            "STRICT"
            if context.get(
                "strict_pairing",
                True,
            )
            else "ROAD LABELS + CONTROLLED BACKGROUND MASKS"
        ),
    )

    print(
        "Manifests     :",
        (
            "REGENERATED"
            if context.get(
                "manifests_regenerated"
            )
            else "REUSED"
        ),
    )

    print(
        "Train Manifest:",
        context.get(
            "train_manifest"
        ),
    )

    print(
        "Val Manifest  :",
        context.get(
            "val_manifest"
        ),
    )

    print(
        "Final Test    : NOT ACCESSED"
    )

    print(
        "=" * 72
    )
    print()


# ============================================================
# PIXEL SIZE
# ============================================================

def get_pixel_size_m(
    dataset_info: dict,
    general_params: dict,
    fallback: float = 0.3,
) -> float:
    """
    Resolve training-data pixel size without reading external final-test data.
    """

    candidates = (
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
    )

    for value in candidates:

        if value is None:
            continue

        try:
            value = float(
                value
            )

            if value > 0:
                return value

        except Exception:
            continue

    return float(
        fallback
    )


__all__ = [
    "apply_runtime_split_overrides",
    "backup_existing_split_files",
    "create_train_val_manifests",
    "get_pixel_size_m",
    "load_json",
    "load_json_if_exists",
    "manifests_match_requested_split",
    "normalize_dataset_info",
    "normalize_output_folder",
    "resolve_model_backend",
    "prepare_experiment",
    "prepare_experiment_split",
    "print_experiment_summary",
    "resolve_dataset_root",
    "resolve_train_validation_split",
    "save_json",
]

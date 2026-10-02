"""
Shared utilities for Approach 2 - Road Extraction.

This module provides:
- JSON configuration loading/saving
- Project path handling
- Dataset lookup
- Model configuration lookup
- Optional configuration lookup
- Experiment output/checkpoint folder creation
- Reproducibility utilities
- Device selection
- Configuration merging

The Roads module is intentionally independent from the Buildings module.
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------

SRC_ROOT = Path(__file__).resolve().parent
ROADS_ROOT = SRC_ROOT.parent
CONFIG_ROOT = ROADS_ROOT / "config"
MODELS_CONFIG_ROOT = CONFIG_ROOT / "models"
OPTIONAL_CONFIG_ROOT = CONFIG_ROOT / "optional"

OUTPUTS_ROOT = ROADS_ROOT / "outputs"
CHECKPOINTS_ROOT = ROADS_ROOT / "checkpoints"


# ---------------------------------------------------------------------
# Basic JSON utilities
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON configuration file does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a JSON file, but received:\n{path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file:\n{path}\n"
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected the root of {path.name} to be a JSON object."
        )

    return data


def save_json(
    data: Dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save a dictionary as a formatted JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    return path


# ---------------------------------------------------------------------
# Main configuration loaders
# ---------------------------------------------------------------------

def load_experiment_config(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load the active Road experiment configuration.
    """
    if path is None:
        path = CONFIG_ROOT / "experiment.json"

    return load_json(path)


def load_general_params(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load shared Road training and evaluation parameters.
    """
    if path is None:
        path = CONFIG_ROOT / "general_params.json"

    return load_json(path)


def load_datasets_config(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load Road dataset definitions.
    """
    if path is None:
        path = CONFIG_ROOT / "datasets.json"

    return load_json(path)


def load_paths_config(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load project, dataset, checkpoint, and final-test paths.
    """
    if path is None:
        path = CONFIG_ROOT / "paths.json"

    return load_json(path)


def load_test_zone_config(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load the final-test holdout policy.
    """
    if path is None:
        path = CONFIG_ROOT / "test_zone.json"

    return load_json(path)


def load_ui_schema(
    path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load Road Streamlit/UI schema.
    """
    if path is None:
        path = CONFIG_ROOT / "ui_schema.json"

    return load_json(path)


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_CONFIG_FILES = {
    "unet": "unet.json",
    "deeplabv3": "deeplabv3.json",
    "connectnet": "connectnet.json",
    "multitask_road_extractor": "multitask_road_extractor.json",
    "sam_lora": "sam_lora.json",
}


MODEL_ALIASES = {
    "u-net": "unet",
    "u_net": "unet",

    "deeplab": "deeplabv3",
    "deeplabv3+": "deeplabv3",
    "deeplabv3plus": "deeplabv3",

    "connect_net": "connectnet",
    "connect-net": "connectnet",

    "multi-task-road-extractor": "multitask_road_extractor",
    "multi_task_road_extractor": "multitask_road_extractor",
    "multitaskroadextractor": "multitask_road_extractor",

    "sam-lora": "sam_lora",
    "samlora": "sam_lora",
}


def normalize_model_type(model_type: str) -> str:
    """
    Normalize model naming used throughout the Road pipeline.
    """
    if not model_type:
        raise ValueError("model_type cannot be empty.")

    normalized = str(model_type).strip().lower()

    normalized = MODEL_ALIASES.get(
        normalized,
        normalized,
    )

    if normalized not in MODEL_CONFIG_FILES:
        supported = ", ".join(MODEL_CONFIG_FILES.keys())

        raise ValueError(
            f"Unsupported Road model_type: {model_type!r}\n"
            f"Supported models: {supported}"
        )

    return normalized


def load_model_config(
    model_type: str,
) -> Dict[str, Any]:
    """
    Load the configuration file corresponding to a Road model.
    """
    model_type = normalize_model_type(model_type)

    filename = MODEL_CONFIG_FILES[model_type]

    path = MODELS_CONFIG_ROOT / filename

    return load_json(path)


# ---------------------------------------------------------------------
# Optional configuration
# ---------------------------------------------------------------------

OPTIONAL_CONFIG_FILES = {
    "augmentation": "augmentation.json",
    "optuna": "optuna.json",
    "master_optuna": "master_optuna_search_space.json",
    "postprocessing": "postprocessing.json",
    "cross_validation": "cross_validation.json",
    "reporting": "reporting.json",
    "error_analysis": "error_analysis.json",
}


def load_optional_config(
    config_name: str,
) -> Dict[str, Any]:
    """
    Load one of the optional Road configuration files.

    Empty JSON files such as {} are valid and intentionally supported.
    """
    key = str(config_name).strip().lower()

    if key not in OPTIONAL_CONFIG_FILES:
        supported = ", ".join(
            OPTIONAL_CONFIG_FILES.keys()
        )

        raise ValueError(
            f"Unknown optional configuration: {config_name!r}\n"
            f"Supported configurations: {supported}"
        )

    path = OPTIONAL_CONFIG_ROOT / OPTIONAL_CONFIG_FILES[key]

    return load_json(path)


# ---------------------------------------------------------------------
# Configuration merging
# ---------------------------------------------------------------------

def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from `override` replace values from `base`.

    Neither input dictionary is modified.
    """
    result = deepcopy(base)

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )

        else:
            result[key] = deepcopy(value)

    return result


# ---------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------

def get_dataset_info(
    experiment_config: Optional[Dict[str, Any]] = None,
    datasets_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve the active training/validation dataset.

    Primary method:
        experiment.json -> dataset_id

    Fallback:
        tile_size + source_type + dataset_type
    """
    if experiment_config is None:
        experiment_config = load_experiment_config()

    if datasets_config is None:
        datasets_config = load_datasets_config()

    datasets = datasets_config.get("datasets", {})

    if not datasets:
        raise RuntimeError(
            "No datasets are registered in datasets.json."
        )

    # --------------------------------------------------------------
    # Preferred: explicit dataset_id
    # --------------------------------------------------------------

    dataset_id = experiment_config.get("dataset_id")

    if dataset_id:

        if dataset_id not in datasets:
            raise KeyError(
                f"Dataset '{dataset_id}' from experiment.json "
                f"is not registered in datasets.json."
            )

        dataset_info = deepcopy(
            datasets[dataset_id]
        )

        if not dataset_info.get("enabled", True):
            raise RuntimeError(
                f"Dataset '{dataset_id}' is disabled."
            )

        return dataset_id, dataset_info

    # --------------------------------------------------------------
    # Fallback automatic resolution
    # --------------------------------------------------------------

    tile_size = int(
        experiment_config.get("tile_size")
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

    matches = []

    for candidate_id, info in datasets.items():

        if not info.get("enabled", True):
            continue

        if info.get("role") != "train_validation":
            continue

        candidate_tile_size = int(
            info.get("tile_size", -1)
        )

        candidate_source_type = str(
            info.get("source_type", "")
        ).lower()

        candidate_dataset_type = str(
            info.get("dataset_type", "")
        ).lower()

        if candidate_tile_size != tile_size:
            continue

        if candidate_source_type != source_type:
            continue

        if candidate_dataset_type != dataset_type:
            continue

        matches.append(
            (
                candidate_id,
                deepcopy(info),
            )
        )

    if len(matches) == 0:
        raise RuntimeError(
            "No Road dataset matches the active experiment:\n"
            f"tile_size={tile_size}\n"
            f"source_type={source_type}\n"
            f"dataset_type={dataset_type}"
        )

    if len(matches) > 1:
        ids = [
            dataset_id
            for dataset_id, _ in matches
        ]

        raise RuntimeError(
            "Multiple Road datasets match the active experiment:\n"
            + "\n".join(ids)
        )

    return matches[0]


def get_dataset_path(
    experiment_config: Optional[Dict[str, Any]] = None,
    datasets_config: Optional[Dict[str, Any]] = None,
    validate_exists: bool = True,
) -> Path:
    """
    Return the filesystem path of the active training dataset.
    """
    dataset_id, dataset_info = get_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config,
    )

    path_value = dataset_info.get("path")

    if not path_value:
        raise ValueError(
            f"Dataset '{dataset_id}' has no path configured."
        )

    dataset_path = Path(path_value)

    if validate_exists and not dataset_path.exists():
        raise FileNotFoundError(
            f"Configured Road dataset does not exist:\n"
            f"{dataset_path}"
        )

    return dataset_path


# ---------------------------------------------------------------------
# Experiment folders
# ---------------------------------------------------------------------

def get_experiment_id(
    experiment_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return the active experiment identifier.
    """
    if experiment_config is None:
        experiment_config = load_experiment_config()

    experiment_id = experiment_config.get(
        "experiment_id"
    )

    if not experiment_id:
        raise ValueError(
            "experiment.json does not define experiment_id."
        )

    return str(experiment_id)


def create_experiment_output_folder(
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create and return:

        Roads/outputs/experiments/<experiment_id>/
    """
    experiment_id = get_experiment_id(
        experiment_config
    )

    output_folder = (
        OUTPUTS_ROOT
        / "experiments"
        / experiment_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_folder


def create_experiment_checkpoint_folder(
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create and return:

        Roads/checkpoints/<experiment_id>/
    """
    experiment_id = get_experiment_id(
        experiment_config
    )

    checkpoint_folder = (
        CHECKPOINTS_ROOT
        / experiment_id
    )

    checkpoint_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return checkpoint_folder


# ---------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------

def save_experiment_snapshot(
    experiment_config: Optional[Dict[str, Any]] = None,
    output_folder: Optional[str | Path] = None,
) -> Path:
    """
    Save a snapshot of the active experiment and related configuration.

    This ensures every experiment remains reproducible even if the main
    configuration files are changed later.
    """
    if experiment_config is None:
        experiment_config = load_experiment_config()

    if output_folder is None:
        output_folder = create_experiment_output_folder(
            experiment_config
        )

    output_folder = Path(output_folder)

    model_type = normalize_model_type(
        experiment_config["model_type"]
    )

    snapshot = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "experiment": experiment_config,

        "general_params": load_general_params(),

        "model_config": load_model_config(
            model_type
        ),

        "dataset_config": get_dataset_info(
            experiment_config=experiment_config
        )[1],

        "test_zone_policy": load_test_zone_config(),
    }

    snapshot_path = (
        output_folder
        / "experiment_snapshot.json"
    )

    save_json(
        snapshot,
        snapshot_path,
    )

    return snapshot_path


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_random_seed(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    """
    Set Python, NumPy, and PyTorch random seeds.

    Parameters
    ----------
    seed:
        Random seed.

    deterministic:
        When True, requests deterministic PyTorch behavior.
        This may reduce training performance.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except Exception:
            pass

    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------

def get_device(
    preferred: Optional[str] = None,
) -> torch.device:
    """
    Resolve the training device.

    Examples
    --------
    get_device()
        -> cuda:0 when CUDA is available

    get_device("cpu")
        -> cpu

    get_device("cuda:0")
        -> cuda:0
    """
    if preferred:

        preferred = str(
            preferred
        ).strip().lower()

        if preferred.startswith("cuda"):

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but PyTorch "
                    "cannot access a CUDA-capable GPU."
                )

            return torch.device(
                preferred
            )

        if preferred == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Unsupported device: {preferred!r}"
        )

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def get_device_summary(
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Return basic information about the selected computation device.
    """
    if device is None:
        device = get_device()

    summary = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda":

        index = (
            device.index
            if device.index is not None
            else 0
        )

        properties = torch.cuda.get_device_properties(
            index
        )

        summary.update(
            {
                "gpu_name": properties.name,

                "gpu_memory_gb": round(
                    properties.total_memory
                    / (1024 ** 3),
                    2,
                ),

                "cuda_device_index": index,
            }
        )

    return summary


# ---------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------

def ensure_directory(
    path: str | Path,
) -> Path:
    """
    Create a directory when it does not already exist.
    """
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def require_path(
    path: str | Path,
    description: str = "Required path",
) -> Path:
    """
    Validate that a required filesystem path exists.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )

    return path


# ---------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------

def print_section(
    title: str,
    width: int = 72,
) -> None:
    """
    Print a simple consistent console section header.
    """
    title = str(title).strip()

    print()
    print("=" * width)
    print(title)
    print("=" * width)


def print_key_value(
    key: str,
    value: Any,
    key_width: int = 24,
) -> None:
    """
    Print a clean key/value line.
    """
    print(
        f"{key:<{key_width}} : {value}"
    )
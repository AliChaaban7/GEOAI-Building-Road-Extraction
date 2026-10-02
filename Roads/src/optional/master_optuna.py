"""
Master Optuna for Approach 2 - Road Extraction.

Purpose
-------
Master Optuna performs a broader hyperparameter search for ONE fixed
Road model family.

The selected model is fixed by experiment.json.

Approved searchable parameters
------------------------------
- tile_size
- batch_size
- learning_rate
- weight_decay
- bce_weight
- augmentation

Fixed architecture
------------------
U-Net:
    architecture fixed

DeepLabV3+:
    architecture fixed

SAM-LoRA:
    backbone = ViT-B
    LoRA rank = 8
    LoRA alpha = 16
    LoRA targets = Q + V

These architectural values are NOT searched unless this implementation
is explicitly changed in the future.

Scientific policy
-----------------
Master Optuna accesses only Train / Validation data.

It NEVER accesses:
- independent Aerial test
- independent Satellite test
- independent Drone test
- Road_Test_Label
- SatelliteTesingZone

Workflow
--------
Master trial:
    choose tile
    choose batch size
    choose LR
    choose WD
    choose BCE weight
    choose augmentation
        ↓
    prepare matching mixed-source dataset
        ↓
    short training
        ↓
    Validation IoU
        ↓
    pruning

After study:
    best Validation IoU
    clDice tie-break
        ↓
    winning hyperparameters
        ↓
    FRESH model
        ↓
    full 40-epoch training
        ↓
    canonical best_model.pth

Threshold and post-processing optimization happen AFTER this stage.

Configuration policy
--------------------
Hyperparameter search values come ONLY from:

    Roads/config/optional/master_optuna_search_space.json

Numeric ranges and categorical choices are never invented in code.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


from src.checkpointing import (
    save_training_checkpoint,
)

from src.models import (
    build_pytorch_model,
    normalize_model_type,
)

from src.optional.augmentation import (
    build_training_augmentation,
)

from src.pipeline import (
    prepare_experiment,
)

from src.train import (
    build_dataloaders,
    build_grad_scaler,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    resolve_device,
    resolve_training_config,
    seed_training,
    train_model,
    train_one_epoch,
    validate_one_epoch,
)


# =====================================================================
# ROOTS
# =====================================================================

ROAD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_MASTER_CONFIG = (
    ROAD_ROOT
    / "config"
    / "optional"
    / "master_optuna_search_space.json"
)

DATASETS_CONFIG_PATH = (
    ROAD_ROOT
    / "config"
    / "datasets.json"
)

AUGMENTATION_CONFIG_PATH = (
    ROAD_ROOT
    / "config"
    / "optional"
    / "augmentation.json"
)


# =====================================================================
# OPTUNA
# =====================================================================

def _get_optuna():

    try:

        import optuna

    except ImportError as exc:

        raise ImportError(
            "Master Optuna requires the 'optuna' package."
        ) from exc

    return optuna


# =====================================================================
# JSON
# =====================================================================

def _json_safe(
    value,
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
        torch.Tensor,
    ):

        if value.numel() == 1:

            return float(
                value.detach()
                .cpu()
                .item()
            )

        return (
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(
                key
            ):
                _json_safe(
                    item
                )

            for (
                key,
                item,
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    return value


def load_json(
    path: str | Path,
) -> Dict[str, Any]:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        value = json.load(
            file
        )

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            f"Expected JSON object:\n{path}"
        )

    return value


def save_json(
    value,
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
                value
            ),
            file,
            indent=2,
        )

    return path


# =====================================================================
# SMALL HELPERS
# =====================================================================

def _as_dict(
    value,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


def _first_not_none(
    *values,
):

    for value in values:

        if value is not None:

            return value

    return None


# =====================================================================
# LOAD MASTER CONFIG
# =====================================================================

def load_master_config(
    path: Optional[
        str | Path
    ] = None,
) -> Dict[str, Any]:

    if path is None:

        path = DEFAULT_MASTER_CONFIG

    return load_json(
        path
    )


# =====================================================================
# MODEL-SPECIFIC SEARCH SPACE
# =====================================================================

_ALLOWED_PARAMETERS = {
    "tile_size",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "bce_weight",
    "augmentation",
}


def get_model_search_block(
    master_config: Dict[str, Any],
    model_type: str,
) -> Dict[str, Any]:
    """
    Resolve model-specific Master Optuna search block.

    Supports:

        {
            "models": {
                "unet": {
                    "parameters": {...}
                }
            }
        }

    and a global form:

        {
            "parameters": {...}
        }
    """

    model_type = normalize_model_type(
        model_type
    )

    models = _as_dict(
        master_config.get(
            "models"
        )
    )

    if models:

        block = models.get(
            model_type
        )

        if block is None:

            raise ValueError(
                "No Master Optuna search space exists for model:\n"
                f"{model_type}"
            )

        if not isinstance(
            block,
            dict,
        ):

            raise TypeError(
                f"Master model block for '{model_type}' must be an object."
            )

        return copy.deepcopy(
            block
        )

    return copy.deepcopy(
        master_config
    )


def get_parameter_block(
    model_block: Dict[str, Any],
) -> Dict[str, Any]:

    parameters = model_block.get(
        "parameters"
    )

    if isinstance(
        parameters,
        dict,
    ):

        return copy.deepcopy(
            parameters
        )

    parameters = model_block.get(
        "search_space"
    )

    if isinstance(
        parameters,
        dict,
    ):

        return copy.deepcopy(
            parameters
        )

    raise ValueError(
        "Master Optuna requires an explicit 'parameters' "
        "or 'search_space' object."
    )


# =====================================================================
# CATEGORICAL SPECIFICATION
# =====================================================================

def _categorical_values(
    parameters: Dict[str, Any],
    name: str,
) -> list:

    specification = parameters.get(
        name
    )

    if specification is None:

        raise ValueError(
            f"Master Optuna parameter '{name}' is missing."
        )

    if isinstance(
        specification,
        dict,
    ):

        values = _first_not_none(
            specification.get(
                "values"
            ),

            specification.get(
                "choices"
            ),
        )

    elif isinstance(
        specification,
        list,
    ):

        values = specification

    else:

        raise TypeError(
            f"Master parameter '{name}' must define values/choices."
        )

    if not isinstance(
        values,
        list,
    ) or not values:

        raise ValueError(
            f"Master parameter '{name}' requires a non-empty list."
        )

    return list(
        values
    )


# =====================================================================
# NUMERIC SPECIFICATION
# =====================================================================

def _numeric_specification(
    parameters: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """
    Numeric values are required explicitly from JSON.

    Nothing is invented.
    """

    specification = parameters.get(
        name
    )

    if specification is None:

        raise ValueError(
            f"\nMaster Optuna numeric search '{name}' is missing.\n"
            "Define it explicitly in master_optuna_search_space.json."
        )

    if isinstance(
        specification,
        dict,
    ):

        low = _first_not_none(
            specification.get(
                "low"
            ),

            specification.get(
                "min"
            ),
        )

        high = _first_not_none(
            specification.get(
                "high"
            ),

            specification.get(
                "max"
            ),
        )

        log = bool(
            specification.get(
                "log",
                False,
            )
        )

        step = specification.get(
            "step"
        )

    elif (
        isinstance(
            specification,
            list,
        )
        and len(
            specification
        ) == 2
    ):

        low = specification[
            0
        ]

        high = specification[
            1
        ]

        log = False

        step = None

    else:

        raise TypeError(
            f"Invalid numeric Master specification for '{name}'."
        )

    if low is None or high is None:

        raise ValueError(
            f"Master '{name}' must explicitly contain low/min and high/max."
        )

    low = float(
        low
    )

    high = float(
        high
    )

    if not math.isfinite(
        low
    ) or not math.isfinite(
        high
    ):

        raise ValueError(
            f"{name} bounds must be finite."
        )

    if low >= high:

        raise ValueError(
            f"{name}: low must be smaller than high."
        )

    if log and low <= 0:

        raise ValueError(
            f"{name}: log search requires low > 0."
        )

    if step is not None:

        step = float(
            step
        )

        if step <= 0:

            raise ValueError(
                f"{name}: step must be > 0."
            )

        if log:

            raise ValueError(
                f"{name}: step and log cannot be used together."
            )

    return {
        "low":
            low,

        "high":
            high,

        "log":
            log,

        "step":
            step,
    }


# =====================================================================
# RESOLVE APPROVED SEARCH SPACE
# =====================================================================

def resolve_master_search_space(
    master_config: Dict[str, Any],
    model_type: str,
) -> Dict[str, Any]:
    """
    Resolve ONLY the approved Road Master parameters.
    """

    block = get_model_search_block(
        master_config,
        model_type,
    )

    parameters = get_parameter_block(
        block
    )

    unsupported = (
        set(
            parameters.keys()
        )
        - _ALLOWED_PARAMETERS
    )

    if unsupported:

        raise ValueError(
            "\nUnsupported Master Optuna search parameters detected:\n"
            f"{sorted(unsupported)}\n\n"
            "Current Roads Master Optuna deliberately allows only:\n"
            "- tile_size\n"
            "- batch_size\n"
            "- learning_rate\n"
            "- weight_decay\n"
            "- bce_weight\n"
            "- augmentation\n\n"
            "SAM rank/alpha/backbone and model architecture stay fixed."
        )

    tile_sizes = [
        int(
            value
        )
        for value
        in _categorical_values(
            parameters,
            "tile_size",
        )
    ]

    allowed_tiles = {
        128,
        192,
        256,
        384,
        512,
    }

    invalid_tiles = (
        set(
            tile_sizes
        )
        - allowed_tiles
    )

    if invalid_tiles:

        raise ValueError(
            "Unsupported Road tile sizes in Master config:\n"
            f"{sorted(invalid_tiles)}\n\n"
            "Available exported datasets are:\n"
            "128, 192, 256, 384, 512"
        )

    if 320 in tile_sizes:

        raise ValueError(
            "CT_320_Roads_03 does not exist. "
            "Tile size 320 must not be used."
        )

    batch_sizes = [
        int(
            value
        )
        for value
        in _categorical_values(
            parameters,
            "batch_size",
        )
    ]

    for batch_size in batch_sizes:

        if batch_size <= 0:

            raise ValueError(
                "Master batch sizes must be > 0."
            )

    augmentation_values = (
        _categorical_values(
            parameters,
            "augmentation",
        )
    )

    normalized_augmentation = []

    for value in augmentation_values:

        if isinstance(
            value,
            bool,
        ):

            normalized_augmentation.append(
                value
            )

            continue

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip()
                .lower()
            )

            if normalized in {
                "true",
                "yes",
                "on",
                "1",
            }:

                normalized_augmentation.append(
                    True
                )

                continue

            if normalized in {
                "false",
                "no",
                "off",
                "0",
            }:

                normalized_augmentation.append(
                    False
                )

                continue

        raise ValueError(
            "Master augmentation choices must be booleans."
        )

    normalized_augmentation = list(
        dict.fromkeys(
            normalized_augmentation
        )
    )

    search_space = {
        "tile_size":
            {
                "values":
                    tile_sizes,
            },

        "batch_size":
            {
                "values":
                    batch_sizes,
            },

        "learning_rate":
            _numeric_specification(
                parameters,
                "learning_rate",
            ),

        "weight_decay":
            _numeric_specification(
                parameters,
                "weight_decay",
            ),

        "bce_weight":
            _numeric_specification(
                parameters,
                "bce_weight",
            ),

        "augmentation":
            {
                "values":
                    normalized_augmentation,
            },
    }

    bce = search_space[
        "bce_weight"
    ]

    if (
        bce[
            "low"
        ] < 0.0
        or bce[
            "high"
        ] > 1.0
    ):

        raise ValueError(
            "Master bce_weight range must remain within [0,1]."
        )

    return search_space


# =====================================================================
# STUDY SETTINGS
# =====================================================================

def resolve_master_settings(
    master_config: Dict[str, Any],
    model_type: str,
) -> Dict[str, Any]:
    """
    Resolve study mechanics.

    These defaults control the study process, not searched
    hyperparameter values.
    """

    study = _as_dict(
        master_config.get(
            "study"
        )
    )

    pruning = _as_dict(
        master_config.get(
            "pruning",
            master_config.get(
                "pruner",
                {},
            ),
        )
    )

    seed = int(
        _first_not_none(
            study.get(
                "seed"
            ),

            master_config.get(
                "seed"
            ),

            42,
        )
    )

    requested_trials = int(
        _first_not_none(
            study.get(
                "trials"
            ),

            study.get(
                "n_trials"
            ),

            master_config.get(
                "trials"
            ),

            15,
        )
    )

    trial_epochs = int(
        _first_not_none(
            study.get(
                "trial_epochs"
            ),

            master_config.get(
                "trial_epochs"
            ),

            12,
        )
    )

    final_epochs = int(
        _first_not_none(
            study.get(
                "final_epochs"
            ),

            master_config.get(
                "final_epochs"
            ),

            40,
        )
    )

    model_type = normalize_model_type(
        model_type
    )

    # Thesis policy: SAM-LoRA is intentionally limited because each
    # ViT-B trial is considerably more expensive.
    if model_type == "sam_lora":

        trials = min(
            requested_trials,
            5,
        )

    else:

        trials = requested_trials

    if trials <= 0:

        raise ValueError(
            "Master trials must be > 0."
        )

    if trial_epochs <= 0:

        raise ValueError(
            "Master trial_epochs must be > 0."
        )

    if final_epochs <= 0:

        raise ValueError(
            "Master final_epochs must be > 0."
        )

    return {
        "requested_trials":
            requested_trials,

        "trials":
            trials,

        "trial_epochs":
            trial_epochs,

        "final_epochs":
            final_epochs,

        "seed":
            seed,

        "pruning_enabled":
            bool(
                _first_not_none(
                    pruning.get(
                        "enabled"
                    ),

                    True,
                )
            ),

        "startup_trials":
            int(
                _first_not_none(
                    pruning.get(
                        "startup_trials"
                    ),

                    pruning.get(
                        "n_startup_trials"
                    ),

                    3,
                )
            ),

        "warmup_epochs":
            int(
                _first_not_none(
                    pruning.get(
                        "warmup_epochs"
                    ),

                    pruning.get(
                        "n_warmup_steps"
                    ),

                    4,
                )
            ),

        "objective":
            "validation_iou",

        "tie_break":
            "validation_cldice",

        "fresh_winner_training":
            True,
    }


# =====================================================================
# DATASET DISCOVERY
# =====================================================================

def _iter_dataset_entries(
    node,
    key_path: Tuple[str, ...] = (),
) -> Iterable[
    Tuple[
        str,
        Dict[str, Any],
    ]
]:
    """
    Recursively discover actual dataset entries.

    We do not construct or guess dataset IDs.
    """

    if not isinstance(
        node,
        dict,
    ):

        return

    for (
        key,
        value,
    ) in node.items():

        if not isinstance(
            value,
            dict,
        ):

            continue

        new_path = (
            *key_path,
            str(
                key
            ),
        )

        tile = _first_not_none(
            value.get(
                "tile_size"
            ),

            value.get(
                "chip_size"
            ),
        )

        has_data_reference = any(
            name in value
            for name
            in (
                "path",
                "dataset_path",
                "root",
                "folder",
                "images",
                "images_path",
                "image_folder",
            )
        )

        if (
            tile is not None
            and has_data_reference
        ):

            yield (
                str(
                    key
                ),
                copy.deepcopy(
                    value
                ),
            )

        yield from _iter_dataset_entries(
            value,
            new_path,
        )


def _dataset_source_is_mixed(
    info: Dict[str, Any],
) -> bool:

    source = str(
        _first_not_none(
            info.get(
                "source_type"
            ),

            info.get(
                "source"
            ),

            "",
        )
    ).strip().lower()

    if source == "mixed":

        return True

    sources = info.get(
        "sources"
    )

    if isinstance(
        sources,
        list,
    ):

        normalized = {
            str(
                value
            )
            .strip()
            .lower()

            for value
            in sources
        }

        return {
            "aerial",
            "satellite",
            "drone",
        }.issubset(
            normalized
        )

    return False


def find_dataset_for_tile(
    datasets_config: Dict[str, Any],
    tile_size: int,
) -> Tuple[
    str,
    Dict[str, Any],
]:
    """
    Select dataset by actual metadata.

    Dataset identifier strings are NEVER guessed.
    """

    tile_size = int(
        tile_size
    )

    candidates = []

    seen = set()

    for (
        dataset_id,
        info,
    ) in _iter_dataset_entries(
        datasets_config
    ):

        try:

            dataset_tile = int(
                _first_not_none(
                    info.get(
                        "tile_size"
                    ),

                    info.get(
                        "chip_size"
                    ),
                )
            )

        except Exception:

            continue

        if dataset_tile != tile_size:

            continue

        if not _dataset_source_is_mixed(
            info
        ):

            continue

        if info.get(
            "enabled"
        ) is False:

            continue

        identity = (
            dataset_id,
            str(
                _first_not_none(
                    info.get(
                        "path"
                    ),

                    info.get(
                        "dataset_path"
                    ),

                    info.get(
                        "root"
                    ),

                    info.get(
                        "folder"
                    ),

                    "",
                )
            ),
        )

        if identity in seen:

            continue

        seen.add(
            identity
        )

        candidates.append(
            (
                dataset_id,
                info,
            )
        )

    if not candidates:

        raise ValueError(
            "\nNo enabled mixed-source Road dataset was found "
            f"for tile_size={tile_size} in datasets.json.\n\n"
            "Master Optuna will not invent a dataset ID."
        )

    if len(
        candidates
    ) > 1:

        candidate_text = "\n".join(
            f"- {dataset_id}"
            for (
                dataset_id,
                _
            ) in candidates
        )

        raise ValueError(
            "\nMore than one mixed Road dataset matches "
            f"tile_size={tile_size}:\n"
            f"{candidate_text}\n\n"
            "Make the dataset metadata unique instead of allowing "
            "Master Optuna to guess."
        )

    return candidates[
        0
    ]


def validate_master_datasets(
    datasets_config: Dict[str, Any],
    tile_values,
) -> Dict[
    int,
    Dict[str, Any],
]:
    """
    Resolve all configured tile choices before starting the study.
    """

    resolved = {}

    for tile_size in tile_values:

        (
            dataset_id,
            dataset_info,
        ) = find_dataset_for_tile(
            datasets_config,
            int(
                tile_size
            ),
        )

        resolved[
            int(
                tile_size
            )
        ] = {
            "dataset_id":
                dataset_id,

            "dataset_info":
                dataset_info,
        }

    return resolved


# =====================================================================
# SAM ARCHITECTURE SAFEGUARD
# =====================================================================

def validate_fixed_sam_architecture(
    context: Dict[str, Any],
) -> None:
    """
    Ensure Master Optuna cannot silently alter the Road SAM baseline.
    """

    if normalize_model_type(
        context[
            "model_type"
        ]
    ) != "sam_lora":

        return

    model = context[
        "model_config"
    ]

    lora = _as_dict(
        model.get(
            "lora"
        )
    )

    backbone = str(
        _first_not_none(
            model.get(
                "backbone"
            ),

            model.get(
                "default_backbone"
            ),

            _as_dict(
                model.get(
                    "architecture"
                )
            ).get(
                "backbone"
            ),

            "vit_b",
        )
    ).strip().lower()

    aliases = {
        "vit-b":
            "vit_b",

        "vitb":
            "vit_b",
    }

    backbone = aliases.get(
        backbone,
        backbone,
    )

    rank = int(
        _first_not_none(
            lora.get(
                "rank"
            ),

            model.get(
                "lora_rank"
            ),

            8,
        )
    )

    alpha = float(
        _first_not_none(
            lora.get(
                "alpha"
            ),

            model.get(
                "lora_alpha"
            ),

            16,
        )
    )

    if backbone != "vit_b":

        raise ValueError(
            "Road Master Optuna currently requires SAM backbone ViT-B."
        )

    if rank != 8:

        raise ValueError(
            "Road Master Optuna currently keeps SAM LoRA rank fixed at 8."
        )

    if alpha != 16.0:

        raise ValueError(
            "Road Master Optuna currently keeps SAM LoRA alpha fixed at 16."
        )


# =====================================================================
# TRIAL SAMPLING
# =====================================================================

def _suggest_float(
    trial,
    name: str,
    specification: Dict[str, Any],
):

    kwargs = {
        "low":
            float(
                specification[
                    "low"
                ]
            ),

        "high":
            float(
                specification[
                    "high"
                ]
            ),

        "log":
            bool(
                specification.get(
                    "log",
                    False,
                )
            ),
    }

    step = specification.get(
        "step"
    )

    if step is not None:

        kwargs[
            "step"
        ] = float(
            step
        )

    return trial.suggest_float(
        name,
        **kwargs,
    )


def sample_master_parameters(
    trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:

    tile_size = int(
        trial.suggest_categorical(
            "tile_size",
            search_space[
                "tile_size"
            ][
                "values"
            ],
        )
    )

    batch_size = int(
        trial.suggest_categorical(
            "batch_size",
            search_space[
                "batch_size"
            ][
                "values"
            ],
        )
    )

    learning_rate = float(
        _suggest_float(
            trial,
            "learning_rate",
            search_space[
                "learning_rate"
            ],
        )
    )

    weight_decay = float(
        _suggest_float(
            trial,
            "weight_decay",
            search_space[
                "weight_decay"
            ],
        )
    )

    bce_weight = float(
        _suggest_float(
            trial,
            "bce_weight",
            search_space[
                "bce_weight"
            ],
        )
    )

    augmentation = bool(
        trial.suggest_categorical(
            "augmentation",
            search_space[
                "augmentation"
            ][
                "values"
            ],
        )
    )

    return {
        "tile_size":
            tile_size,

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "bce_weight":
            bce_weight,

        "dice_weight":
            float(
                1.0
                - bce_weight
            ),

        "augmentation":
            augmentation,
    }


# =====================================================================
# INTERNAL EXPERIMENT CONFIG
# =====================================================================

def build_master_runtime_experiment(
    base_experiment: Dict[str, Any],
    model_type: str,
    tile_size: int,
    dataset_id: str,
    runtime_experiment_id: str,
) -> Dict[str, Any]:
    """
    Build temporary Master experiment config.

    dataset_id comes from datasets.json metadata discovery.
    """

    experiment = copy.deepcopy(
        base_experiment
    )

    experiment[
        "experiment_id"
    ] = str(
        runtime_experiment_id
    )

    experiment[
        "model_type"
    ] = normalize_model_type(
        model_type
    )

    experiment[
        "source_type"
    ] = "mixed"

    experiment[
        "tile_size"
    ] = int(
        tile_size
    )

    # Supply the exact ID discovered from datasets.json.
    experiment[
        "dataset_id"
    ] = str(
        dataset_id
    )

    experiment[
        "dataset"
    ] = str(
        dataset_id
    )

    experiment[
        "use_optuna"
    ] = False

    experiment[
        "use_master_optuna"
    ] = True

    experiment[
        "use_threshold_search"
    ] = False

    experiment[
        "use_postprocessing"
    ] = False

    experiment[
        "use_cross_validation"
    ] = False

    experiment[
        "use_reporting"
    ] = False

    experiment[
        "use_error_analysis"
    ] = False

    return experiment


# =====================================================================
# PREPARE TILE CONTEXT
# =====================================================================

def prepare_tile_context(
    base_context: Dict[str, Any],
    tile_size: int,
    dataset_id: str,
    master_root: Path,
) -> Dict[str, Any]:
    """
    Let the existing Road preparation pipeline build the correct
    deterministic manifests for the actual selected dataset.
    """

    model_type = normalize_model_type(
        base_context[
            "model_type"
        ]
    )

    runtime_id = (
        f"__master_{base_context['experiment_config']['experiment_id']}"
        f"_{model_type}_{int(tile_size)}"
    )

    experiment = (
        build_master_runtime_experiment(
            base_experiment=(
                base_context[
                    "experiment_config"
                ]
            ),

            model_type=model_type,

            tile_size=int(
                tile_size
            ),

            dataset_id=dataset_id,

            runtime_experiment_id=runtime_id,
        )
    )

    runtime_folder = (
        master_root
        / "runtime_configs"
    )

    runtime_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_path = (
        runtime_folder
        / f"tile_{int(tile_size)}.json"
    )

    save_json(
        experiment,
        runtime_path,
    )

    context = prepare_experiment(
        config_path=str(
            runtime_path
        ),

        force_regenerate_manifests=False,
    )

    # --------------------------------------------------------------
    # Strict dataset verification.
    # --------------------------------------------------------------

    resolved_tile = int(
        context[
            "experiment_config"
        ][
            "tile_size"
        ]
    )

    if resolved_tile != int(
        tile_size
    ):

        raise RuntimeError(
            "Prepared Master context tile mismatch."
        )

    return context


# =====================================================================
# APPLY MASTER PARAMETERS
# =====================================================================

def apply_master_parameters(
    prepared_context: Dict[str, Any],
    params: Dict[str, Any],
    epochs: int,
    output_folder: Path,
    checkpoint_folder: Path,
) -> Dict[str, Any]:
    """
    Apply searched values to one prepared tile-specific context.
    """

    context = copy.deepcopy(
        prepared_context
    )

    context[
        "output_folder"
    ] = Path(
        output_folder
    )

    context[
        "checkpoint_folder"
    ] = Path(
        checkpoint_folder
    )

    # --------------------------------------------------------------
    # General training settings.
    # --------------------------------------------------------------

    general = context.setdefault(
        "general_params",
        {}
    )

    training = general.setdefault(
        "training",
        {}
    )

    training[
        "epochs"
    ] = int(
        epochs
    )

    training[
        "batch_size"
    ] = int(
        params[
            "batch_size"
        ]
    )

    training[
        "learning_rate"
    ] = float(
        params[
            "learning_rate"
        ]
    )

    training[
        "weight_decay"
    ] = float(
        params[
            "weight_decay"
        ]
    )

    # --------------------------------------------------------------
    # Model training overrides.
    #
    # model_config has precedence in the shared trainer.
    # --------------------------------------------------------------

    model = context.setdefault(
        "model_config",
        {}
    )

    model_training = model.setdefault(
        "training",
        {}
    )

    model_training[
        "epochs"
    ] = int(
        epochs
    )

    model_training[
        "batch_size"
    ] = int(
        params[
            "batch_size"
        ]
    )

    model_training[
        "learning_rate"
    ] = float(
        params[
            "learning_rate"
        ]
    )

    model_training[
        "weight_decay"
    ] = float(
        params[
            "weight_decay"
        ]
    )

    # --------------------------------------------------------------
    # Loss.
    # --------------------------------------------------------------

    loss = model.setdefault(
        "loss",
        {}
    )

    loss[
        "bce_weight"
    ] = float(
        params[
            "bce_weight"
        ]
    )

    loss[
        "dice_weight"
    ] = float(
        params[
            "dice_weight"
        ]
    )

    # --------------------------------------------------------------
    # Experiment.
    # --------------------------------------------------------------

    experiment = context.setdefault(
        "experiment_config",
        {}
    )

    experiment[
        "tile_size"
    ] = int(
        params[
            "tile_size"
        ]
    )

    experiment[
        "use_augmentation"
    ] = bool(
        params[
            "augmentation"
        ]
    )

    experiment[
        "use_master_optuna"
    ] = True

    experiment[
        "use_optuna"
    ] = False

    experiment[
        "use_threshold_search"
    ] = False

    experiment[
        "use_postprocessing"
    ] = False

    experiment[
        "use_cross_validation"
    ] = False

    experiment[
        "use_reporting"
    ] = False

    experiment[
        "use_error_analysis"
    ] = False

    return context


# =====================================================================
# AUGMENTATION
# =====================================================================

def build_master_augmentation(
    enabled: bool,
):

    if not enabled:

        return None

    config = load_json(
        AUGMENTATION_CONFIG_PATH
    )

    augmentation = (
        build_training_augmentation(
            config
        )
    )

    if augmentation is None:

        raise RuntimeError(
            "Master Optuna selected augmentation=True but "
            "the augmentation pipeline returned None."
        )

    return augmentation


# =====================================================================
# SCHEDULER
# =====================================================================

def _step_scheduler(
    scheduler,
    validation_iou: float,
):

    if scheduler is None:

        return

    if isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):

        scheduler.step(
            validation_iou
        )

    else:

        scheduler.step()


# =====================================================================
# ONE NATIVE PYTORCH MASTER TRIAL
# =====================================================================

def run_native_master_trial(
    trial,
    prepared_context: Dict[str, Any],
    params: Dict[str, Any],
    settings: Dict[str, Any],
    master_root: Path,
    device=None,
) -> Dict[str, Any]:
    """
    Run one native PyTorch Master Optuna candidate.
    """

    optuna = _get_optuna()

    trial_number = int(
        trial.number
    )

    trial_root = (
        master_root
        / "trials"
        / f"trial_{trial_number:04d}"
    )

    output_folder = (
        trial_root
        / "output"
    )

    checkpoint_folder = (
        trial_root
        / "checkpoints"
    )

    context = apply_master_parameters(
        prepared_context=prepared_context,
        params=params,
        epochs=settings[
            "trial_epochs"
        ],
        output_folder=output_folder,
        checkpoint_folder=checkpoint_folder,
    )

    validate_fixed_sam_architecture(
        context
    )

    seed = int(
        settings[
            "seed"
        ]
    )

    seed_training(
        seed=seed,
        deterministic=False,
    )

    resolved_device = resolve_device(
        device
    )

    augmentation = build_master_augmentation(
        params[
            "augmentation"
        ]
    )

    model = None

    try:

        data = build_dataloaders(
            context=context,
            train_augmentation=augmentation,
            validation_augmentation=None,
        )

        model = build_pytorch_model(
            context=context
        ).to(
            resolved_device
        )

        general_params = context[
            "general_params"
        ]

        model_config = context[
            "model_config"
        ]

        training_config = (
            resolve_training_config(
                general_params=general_params,
                model_config=model_config,
            )
        )

        criterion = build_loss_function(
            general_params=general_params,
            model_config=model_config,
        )

        if isinstance(
            criterion,
            torch.nn.Module,
        ):

            criterion = criterion.to(
                resolved_device
            )

        optimizer = build_optimizer(
            model=model,
            training_config=training_config,
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=training_config,
        )

        use_amp = bool(
            training_config[
                "use_amp"
            ]
            and resolved_device.type == "cuda"
        )

        scaler = build_grad_scaler(
            use_amp
        )

        validation_config = _as_dict(
            general_params.get(
                "validation"
            )
        )

        validation_threshold = float(
            _first_not_none(
                validation_config.get(
                    "default_probability_threshold"
                ),

                validation_config.get(
                    "threshold"
                ),

                0.5,
            )
        )

        cldice_iterations = int(
            validation_config.get(
                "cldice_iterations",
                50,
            )
        )

        best_iou = float(
            "-inf"
        )

        best_cldice = float(
            "-inf"
        )

        best_epoch = None

        best_metrics = None

        best_checkpoint = (
            checkpoint_folder
            / "best_model.pth"
        )

        history = []

        for epoch in range(
            1,
            int(
                settings[
                    "trial_epochs"
                ]
            )
            + 1,
        ):

            train_result = train_one_epoch(
                model=model,

                loader=data[
                    "train_loader"
                ],

                optimizer=optimizer,

                criterion=criterion,

                device=resolved_device,

                scaler=scaler,

                use_amp=use_amp,

                gradient_clipping=(
                    training_config[
                        "gradient_clipping"
                    ]
                ),

                threshold=0.5,
            )

            validation_result = (
                validate_one_epoch(
                    model=model,

                    loader=data[
                        "validation_loader"
                    ],

                    criterion=criterion,

                    device=resolved_device,

                    threshold=(
                        validation_threshold
                    ),

                    use_amp=use_amp,

                    # Master objective uses clDice only as tie-break.
                    calculate_cldice=True,

                    cldice_iterations=(
                        cldice_iterations
                    ),
                )
            )

            current_iou = float(
                validation_result[
                    "iou"
                ]
            )

            current_cldice = (
                validation_result.get(
                    "cldice"
                )
            )

            current_cldice_value = (
                float(
                    current_cldice
                )
                if current_cldice
                is not None
                else float(
                    "-inf"
                )
            )

            # ------------------------------------------------------
            # Current epoch value goes to Optuna's pruner.
            # ------------------------------------------------------

            trial.report(
                current_iou,
                step=epoch,
            )

            # ------------------------------------------------------
            # Best epoch:
            # IoU primary
            # clDice tie-break
            # ------------------------------------------------------

            is_better = False

            if current_iou > best_iou:

                is_better = True

            elif math.isclose(
                current_iou,
                best_iou,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):

                if (
                    current_cldice_value
                    > best_cldice
                ):

                    is_better = True

            if is_better:

                best_iou = current_iou

                best_cldice = (
                    current_cldice_value
                )

                best_epoch = int(
                    epoch
                )

                best_metrics = copy.deepcopy(
                    validation_result
                )

                save_training_checkpoint(
                    checkpoint_path=(
                        best_checkpoint
                    ),

                    model=model,

                    context=context,

                    epoch=epoch,

                    validation_metrics=(
                        validation_result
                    ),

                    train_metrics=(
                        train_result
                    ),

                    optimizer=optimizer,

                    scheduler=scheduler,

                    scaler=scaler,

                    extra={
                        "master_optuna":
                            True,

                        "master_trial":
                            trial_number,

                        "master_parameters":
                            copy.deepcopy(
                                params
                            ),

                        "selection_metric":
                            "validation_iou",

                        "tie_break":
                            "validation_cldice",

                        "independent_test_used":
                            False,
                    },
                )

            history.append(
                {
                    "epoch":
                        int(
                            epoch
                        ),

                    "train_loss":
                        float(
                            train_result[
                                "train_loss"
                            ]
                        ),

                    "train_iou":
                        float(
                            train_result[
                                "iou"
                            ]
                        ),

                    "validation_loss":
                        float(
                            validation_result[
                                "validation_loss"
                            ]
                        ),

                    "validation_iou":
                        current_iou,

                    "validation_cldice":
                        current_cldice,

                    "best":
                        bool(
                            is_better
                        ),
                }
            )

            save_json(
                history,
                output_folder
                / "trial_history.json",
            )

            _step_scheduler(
                scheduler,
                current_iou,
            )

            if (
                settings[
                    "pruning_enabled"
                ]
                and trial.should_prune()
            ):

                save_json(
                    {
                        "trial":
                            trial_number,

                        "status":
                            "PRUNED",

                        "epoch":
                            int(
                                epoch
                            ),

                        "current_validation_iou":
                            current_iou,

                        "best_validation_iou":
                            (
                                best_iou
                                if best_epoch
                                is not None
                                else None
                            ),

                        "parameters":
                            params,

                        "final_test_used":
                            False,
                    },
                    output_folder
                    / "trial_summary.json",
                )

                raise optuna.TrialPruned(
                    f"Master trial {trial_number} pruned."
                )

        if best_epoch is None:

            raise RuntimeError(
                "Master trial completed without a valid best epoch."
            )

        best_cldice_output = (
            None
            if best_cldice
            == float(
                "-inf"
            )
            else float(
                best_cldice
            )
        )

        trial.set_user_attr(
            "best_epoch",
            int(
                best_epoch
            ),
        )

        trial.set_user_attr(
            "best_validation_cldice",
            best_cldice_output,
        )

        trial.set_user_attr(
            "dataset_id",
            str(
                context.get(
                    "dataset_id"
                )
            ),
        )

        summary = {
            "trial":
                trial_number,

            "status":
                "COMPLETE",

            "parameters":
                params,

            "dataset_id":
                context.get(
                    "dataset_id"
                ),

            "dataset_root":
                str(
                    context.get(
                        "dataset_root"
                    )
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_validation_iou":
                float(
                    best_iou
                ),

            "best_validation_cldice":
                best_cldice_output,

            "best_validation_metrics":
                best_metrics,

            "checkpoint":
                str(
                    best_checkpoint
                ),

            "final_test_used":
                False,
        }

        save_json(
            summary,
            output_folder
            / "trial_summary.json",
        )

        return summary

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        save_json(
            {
                "trial":
                    int(
                        trial.number
                    ),

                "status":
                    "PRUNED_OOM",

                "parameters":
                    params,

                "error":
                    str(
                        exc
                    ),
            },
            output_folder
            / "trial_summary.json",
        )

        raise optuna.TrialPruned(
            "CUDA out of memory."
        )

    except RuntimeError as exc:

        if "out of memory" in str(
            exc
        ).lower():

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            raise optuna.TrialPruned(
                "CUDA out of memory."
            )

        raise

    finally:

        try:

            del model

        except Exception:

            pass

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# =====================================================================
# TRIAL CSV
# =====================================================================

def save_master_trials_csv(
    study,
    path: str | Path,
) -> Path:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_names = sorted(
        {
            name

            for trial
            in study.trials

            for name
            in trial.params.keys()
        }
    )

    fieldnames = [
        "trial",
        "state",
        "validation_iou",
        "validation_cldice",
        *parameter_names,
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for trial in study.trials:

            row = {
                "trial":
                    int(
                        trial.number
                    ),

                "state":
                    trial.state.name,

                "validation_iou":
                    trial.value,

                "validation_cldice":
                    trial.user_attrs.get(
                        "best_validation_cldice"
                    ),
            }

            for name in parameter_names:

                row[
                    name
                ] = trial.params.get(
                    name
                )

            writer.writerow(
                row
            )

    return path


# =====================================================================
# WINNER SELECTION
# =====================================================================

def select_master_winner(
    study,
):
    """
    Select best trial by:

        1. Validation IoU
        2. Validation clDice tie-break
    """

    optuna = _get_optuna()

    completed = [
        trial

        for trial
        in study.trials

        if trial.state
        == optuna.trial.TrialState.COMPLETE

        and trial.value
        is not None
    ]

    if not completed:

        raise RuntimeError(
            "Master Optuna completed without any successful trials."
        )

    def ranking(
        trial,
    ):

        cldice = trial.user_attrs.get(
            "best_validation_cldice"
        )

        cldice = (
            float(
                cldice
            )
            if cldice
            is not None
            else float(
                "-inf"
            )
        )

        return (
            float(
                trial.value
            ),
            cldice,
        )

    return max(
        completed,
        key=ranking,
    )


# =====================================================================
# COPY MANIFESTS INTO FINAL EXPERIMENT
# =====================================================================

def copy_final_manifests(
    source_context: Dict[str, Any],
    final_context: Dict[str, Any],
) -> None:
    """
    Make the final full retraining experiment independent from
    Master Optuna's internal runtime folders.
    """

    destination = (
        Path(
            final_context[
                "output_folder"
            ]
        )
        / "manifests"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = {
        "train_manifest":
            "train_manifest.csv",

        "validation_manifest":
            "validation_manifest.csv",
    }

    for (
        context_key,
        filename,
    ) in mapping.items():

        source = source_context.get(
            context_key
        )

        if source is None:

            continue

        source = Path(
            source
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Master manifest does not exist:\n{source}"
            )

        target = (
            destination
            / filename
        )

        shutil.copy2(
            source,
            target,
        )

        final_context[
            context_key
        ] = target


# =====================================================================
# BUILD FINAL WINNER CONTEXT
# =====================================================================

def build_final_winner_context(
    base_context: Dict[str, Any],
    winner_tile_context: Dict[str, Any],
    best_params: Dict[str, Any],
    final_epochs: int,
) -> Dict[str, Any]:
    """
    Build canonical experiment context for fresh winner retraining.
    """

    final_context = copy.deepcopy(
        winner_tile_context
    )

    # Canonical output/checkpoint locations from original experiment.
    final_context[
        "output_folder"
    ] = Path(
        base_context[
            "output_folder"
        ]
    )

    final_context[
        "checkpoint_folder"
    ] = Path(
        base_context[
            "checkpoint_folder"
        ]
    )

    final_context[
        "output_folder"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    final_context[
        "checkpoint_folder"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    original_experiment = copy.deepcopy(
        base_context[
            "experiment_config"
        ]
    )

    selected_experiment = copy.deepcopy(
        final_context[
            "experiment_config"
        ]
    )

    # Preserve original experiment identity.
    selected_experiment[
        "experiment_id"
    ] = original_experiment[
        "experiment_id"
    ]

    selected_experiment[
        "model_type"
    ] = base_context[
        "model_type"
    ]

    selected_experiment[
        "source_type"
    ] = "mixed"

    selected_experiment[
        "tile_size"
    ] = int(
        best_params[
            "tile_size"
        ]
    )

    selected_experiment[
        "use_augmentation"
    ] = bool(
        best_params[
            "augmentation"
        ]
    )

    selected_experiment[
        "use_master_optuna"
    ] = True

    selected_experiment[
        "use_optuna"
    ] = False

    selected_experiment[
        "use_threshold_search"
    ] = False

    selected_experiment[
        "use_postprocessing"
    ] = False

    selected_experiment[
        "use_cross_validation"
    ] = False

    selected_experiment[
        "use_reporting"
    ] = False

    selected_experiment[
        "use_error_analysis"
    ] = False

    final_context[
        "experiment_config"
    ] = selected_experiment

    # Apply training + loss parameters.
    final_context = apply_master_parameters(
        prepared_context=final_context,

        params=best_params,

        epochs=int(
            final_epochs
        ),

        output_folder=(
            final_context[
                "output_folder"
            ]
        ),

        checkpoint_folder=(
            final_context[
                "checkpoint_folder"
            ]
        ),
    )

    copy_final_manifests(
        source_context=winner_tile_context,

        final_context=final_context,
    )

    return final_context


# =====================================================================
# FRESH WINNER TRAINING
# =====================================================================

def train_fresh_master_winner(
    base_context: Dict[str, Any],
    winner_tile_context: Dict[str, Any],
    best_params: Dict[str, Any],
    final_epochs: int,
    device=None,
) -> Dict[str, Any]:
    """
    Train Master winner from fresh initialization.
    """

    final_context = build_final_winner_context(
        base_context=base_context,

        winner_tile_context=(
            winner_tile_context
        ),

        best_params=best_params,

        final_epochs=int(
            final_epochs
        ),
    )

    validate_fixed_sam_architecture(
        final_context
    )

    output_folder = Path(
        final_context[
            "output_folder"
        ]
    )

    # --------------------------------------------------------------
    # Persist exact winning architecture/settings BEFORE training.
    # --------------------------------------------------------------

    save_json(
        final_context[
            "model_config"
        ],

        output_folder
        / "resolved_master_winner_model_config.json",
    )

    save_json(
        final_context[
            "experiment_config"
        ],

        output_folder
        / "resolved_master_winner_experiment_config.json",
    )

    save_json(
        best_params,

        output_folder
        / "resolved_master_winner_parameters.json",
    )

    augmentation = build_master_augmentation(
        best_params[
            "augmentation"
        ]
    )

    # --------------------------------------------------------------
    # Critical:
    # fresh architecture, fresh initialization.
    #
    # Trial weights are NOT loaded.
    # --------------------------------------------------------------

    model = build_pytorch_model(
        context=final_context
    )

    result = train_model(
        model=model,

        context=final_context,

        train_augmentation=augmentation,

        validation_augmentation=None,

        device=device,
    )

    return {
        "context":
            final_context,

        "result":
            result,
    }


# =====================================================================
# MASTER OPTUNA WORKFLOW
# =====================================================================

def run_master_optuna(
    base_context: Dict[str, Any],
    config_path: Optional[
        str | Path
    ] = None,
    device=None,
) -> Dict[str, Any]:
    """
    Execute the complete native Road Master Optuna workflow.
    """

    optuna = _get_optuna()

    # --------------------------------------------------------------
    # Backend
    # --------------------------------------------------------------

    backend = base_context.get(
        "model_backend"
    )

    if backend != "pytorch":

        raise RuntimeError(
            "\nThis Master Optuna engine currently controls the native "
            "PyTorch Road models:\n"
            "- U-Net\n"
            "- DeepLabV3+\n"
            "- SAM-LoRA\n\n"
            "ConnectNet and MultiTaskRoadExtractor use the real "
            "ArcGIS Learn backend and must not be forced through the "
            "PyTorch optimization engine."
        )

    model_type = normalize_model_type(
        base_context[
            "model_type"
        ]
    )

    # --------------------------------------------------------------
    # Experiment switches
    # --------------------------------------------------------------

    experiment = base_context[
        "experiment_config"
    ]

    if not bool(
        experiment.get(
            "use_master_optuna",
            False,
        )
    ):

        raise RuntimeError(
            "experiment.json has use_master_optuna=false."
        )

    if bool(
        experiment.get(
            "use_optuna",
            False,
        )
    ):

        raise RuntimeError(
            "General Optuna and Master Optuna cannot run together."
        )

    validate_fixed_sam_architecture(
        base_context
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    master_config = load_master_config(
        config_path
    )

    search_space = (
        resolve_master_search_space(
            master_config,
            model_type,
        )
    )

    settings = resolve_master_settings(
        master_config,
        model_type,
    )

    # --------------------------------------------------------------
    # Actual Road datasets
    # --------------------------------------------------------------

    datasets_config = load_json(
        DATASETS_CONFIG_PATH
    )

    dataset_map = validate_master_datasets(
        datasets_config=datasets_config,

        tile_values=(
            search_space[
                "tile_size"
            ][
                "values"
            ]
        ),
    )

    experiment_id = str(
        experiment[
            "experiment_id"
        ]
    )

    master_root = (
        ROAD_ROOT
        / "outputs"
        / "master_optuna"
        / experiment_id
    )

    master_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        master_config,
        master_root
        / "master_config_used.json",
    )

    save_json(
        search_space,
        master_root
        / "resolved_search_space.json",
    )

    save_json(
        settings,
        master_root
        / "study_settings.json",
    )

    save_json(
        {
            str(
                tile
            ):
                {
                    "dataset_id":
                        value[
                            "dataset_id"
                        ],

                    "dataset_info":
                        value[
                            "dataset_info"
                        ],
                }

            for (
                tile,
                value,
            ) in dataset_map.items()
        },
        master_root
        / "resolved_datasets.json",
    )

    # --------------------------------------------------------------
    # Prepare each tile dataset ONCE.
    # --------------------------------------------------------------

    prepared_tile_contexts = {}

    for tile in (
        search_space[
            "tile_size"
        ][
            "values"
        ]
    ):

        dataset_id = (
            dataset_map[
                int(
                    tile
                )
            ][
                "dataset_id"
            ]
        )

        prepared_tile_contexts[
            int(
                tile
            )
        ] = prepare_tile_context(
            base_context=base_context,

            tile_size=int(
                tile
            ),

            dataset_id=dataset_id,

            master_root=master_root,
        )

    # --------------------------------------------------------------
    # Persistent study
    # --------------------------------------------------------------

    database_path = (
        master_root
        / "master_optuna_study.db"
    )

    storage = (
        f"sqlite:///"
        f"{database_path.as_posix()}"
    )

    study_name = (
        f"{experiment_id}_{model_type}_master"
    )

    sampler = optuna.samplers.TPESampler(
        seed=int(
            settings[
                "seed"
            ]
        )
    )

    if settings[
        "pruning_enabled"
    ]:

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(
                settings[
                    "startup_trials"
                ]
            ),

            n_warmup_steps=int(
                settings[
                    "warmup_epochs"
                ]
            ),
        )

    else:

        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,

        storage=storage,

        direction="maximize",

        sampler=sampler,

        pruner=pruner,

        load_if_exists=True,
    )

    # --------------------------------------------------------------
    # Objective
    # --------------------------------------------------------------

    def objective(
        trial,
    ):

        params = sample_master_parameters(
            trial,
            search_space,
        )

        tile_size = int(
            params[
                "tile_size"
            ]
        )

        prepared_context = (
            prepared_tile_contexts[
                tile_size
            ]
        )

        result = run_native_master_trial(
            trial=trial,

            prepared_context=prepared_context,

            params=params,

            settings=settings,

            master_root=master_root,

            device=device,
        )

        return float(
            result[
                "best_validation_iou"
            ]
        )

    # --------------------------------------------------------------
    # Resume support.
    #
    # Target is total trial count, not additional count.
    # --------------------------------------------------------------

    existing_trials = len(
        study.trials
    )

    remaining_trials = max(
        0,
        int(
            settings[
                "trials"
            ]
        )
        - existing_trials,
    )

    if remaining_trials > 0:

        study.optimize(
            objective,

            n_trials=remaining_trials,

            gc_after_trial=True,

            show_progress_bar=False,
        )

    # --------------------------------------------------------------
    # Winner
    # --------------------------------------------------------------

    winner = select_master_winner(
        study
    )

    best_params = dict(
        winner.params
    )

    best_params[
        "tile_size"
    ] = int(
        best_params[
            "tile_size"
        ]
    )

    best_params[
        "batch_size"
    ] = int(
        best_params[
            "batch_size"
        ]
    )

    best_params[
        "augmentation"
    ] = bool(
        best_params[
            "augmentation"
        ]
    )

    best_params[
        "dice_weight"
    ] = float(
        1.0
        - float(
            best_params[
                "bce_weight"
            ]
        )
    )

    winner_cldice = (
        winner.user_attrs.get(
            "best_validation_cldice"
        )
    )

    trials_csv = save_master_trials_csv(
        study,

        master_root
        / "master_optuna_trials.csv",
    )

    # --------------------------------------------------------------
    # Save selection BEFORE final retraining.
    # --------------------------------------------------------------

    preliminary = {
        "study_name":
            study_name,

        "model_type":
            model_type,

        "source_type":
            "mixed",

        "objective":
            "validation_iou",

        "tie_break":
            "validation_cldice",

        "requested_trials":
            int(
                settings[
                    "requested_trials"
                ]
            ),

        "effective_trials":
            int(
                settings[
                    "trials"
                ]
            ),

        "trial_epochs":
            int(
                settings[
                    "trial_epochs"
                ]
            ),

        "final_epochs":
            int(
                settings[
                    "final_epochs"
                ]
            ),

        "best_trial_number":
            int(
                winner.number
            ),

        "best_trial_validation_iou":
            float(
                winner.value
            ),

        "best_trial_validation_iou_percent":
            float(
                winner.value
                * 100.0
            ),

        "best_trial_validation_cldice":
            winner_cldice,

        "best_params":
            best_params,

        "selected_dataset_id":
            dataset_map[
                int(
                    best_params[
                        "tile_size"
                    ]
                )
            ][
                "dataset_id"
            ],

        "search_space":
            search_space,

        "trials_csv":
            str(
                trials_csv
            ),

        "winner_policy":
            "fresh_full_retraining",

        "trial_checkpoint_promoted":
            False,

        "threshold_search_inside_master":
            False,

        "postprocessing_inside_master":
            False,

        "cross_validation_inside_master":
            False,

        "final_test_used":
            False,
    }

    save_json(
        preliminary,

        master_root
        / "master_optuna_selection.json",
    )

    # --------------------------------------------------------------
    # FRESH full winner training
    # --------------------------------------------------------------

    winner_tile_context = (
        prepared_tile_contexts[
            int(
                best_params[
                    "tile_size"
                ]
            )
        ]
    )

    fresh_training = train_fresh_master_winner(
        base_context=base_context,

        winner_tile_context=(
            winner_tile_context
        ),

        best_params=best_params,

        final_epochs=int(
            settings[
                "final_epochs"
            ]
        ),

        device=device,
    )

    training_result = fresh_training[
        "result"
    ]

    # --------------------------------------------------------------
    # Canonical final summary in experiment folder.
    # --------------------------------------------------------------

    final_summary = {
        **preliminary,

        "fresh_winner_retrained":
            True,

        "fresh_training_best_epoch":
            int(
                training_result[
                    "best_epoch"
                ]
            ),

        "fresh_training_best_validation_iou":
            float(
                training_result[
                    "best_validation_iou"
                ]
            ),

        "fresh_training_best_validation_iou_percent":
            float(
                training_result[
                    "best_validation_iou"
                ]
                * 100.0
            ),

        "final_checkpoint":
            str(
                training_result[
                    "best_checkpoint"
                ]
            ),

        "resolved_winner_model_config":
            str(
                Path(
                    base_context[
                        "output_folder"
                    ]
                )
                / "resolved_master_winner_model_config.json"
            ),

        "resolved_winner_parameters":
            str(
                Path(
                    base_context[
                        "output_folder"
                    ]
                )
                / "resolved_master_winner_parameters.json"
            ),

        "independent_test_used":
            False,
    }

    canonical_path = (
        Path(
            base_context[
                "output_folder"
            ]
        )
        / "master_optuna_selection_summary.json"
    )

    save_json(
        final_summary,
        canonical_path,
    )

    save_json(
        final_summary,
        master_root
        / "master_optuna_final_summary.json",
    )

    # --------------------------------------------------------------
    # Console
    # --------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "ROAD MASTER OPTUNA COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Model              : {model_type}"
    )

    print(
        f"Best trial         : {winner.number}"
    )

    print(
        f"Trial Val IoU      : {winner.value * 100:.4f}%"
    )

    if winner_cldice is not None:

        print(
            f"Trial clDice       : {float(winner_cldice) * 100:.4f}%"
        )

    print()
    print(
        "WINNING PARAMETERS"
    )

    for (
        key,
        value,
    ) in best_params.items():

        print(
            f"  {key:<20}: {value}"
        )

    print()
    print(
        "FIXED ARCHITECTURE"
    )

    if model_type == "sam_lora":

        print(
            "  backbone            : vit_b"
        )

        print(
            "  lora_rank           : 8"
        )

        print(
            "  lora_alpha          : 16"
        )

        print(
            "  lora_targets        : Q + V"
        )

    else:

        print(
            "  model architecture  : unchanged"
        )

    print()
    print(
        f"Fresh training IoU : "
        f"{training_result['best_validation_iou'] * 100:.4f}%"
    )

    print(
        f"Final checkpoint   : "
        f"{training_result['best_checkpoint']}"
    )

    print(
        "Threshold search   : NOT RUN"
    )

    print(
        "Post-processing    : NOT RUN"
    )

    print(
        "Final test         : NOT ACCESSED"
    )

    print(
        "=" * 78
    )

    return final_summary


# Compatibility aliases.
run_master = run_master_optuna
run_master_optuna_search = run_master_optuna
"""
master_optuna.py

Model-level Master Optuna engine for the Buildings module.

Supported model families:
- U-Net
- DeepLabV3 / DeepLabV3+
- Mask R-CNN
- SAM-LoRA

Scientific policy
-----------------
Master Optuna performs a wider search inside ONE already-selected model family.

Fixed for a study:
- model family
- imagery source
- Train/Validation percentages
- split/training seed
- external geographic test area is never accessed

Every trial is a complete candidate training run. Each trial saves the
checkpoint from its BEST validation IoU. After the study, the exact winning
trial checkpoint is promoted to the normal experiment best_model.pth.

There is NO second training of the winner.

SAM-LoRA policy
---------------
SAM-LoRA uses its dedicated trainer and can search:
- fixed supported backbone list from JSON (currently ViT-B)
- tile size
- batch size
- learning rate
- weight decay
- BCE/Dice balance
- LoRA rank
- LoRA alpha
- augmentation ON/OFF

Auto-LR is disabled inside Master Optuna because Optuna itself selects the
learning rate. The winning SAM-LoRA model configuration is preserved beside
the promoted checkpoint so later validation/inference stages can reconstruct
the exact LoRA architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import copy
import gc
import json
import shutil
import sys

import numpy as np
import torch

try:
    import optuna
except ImportError:
    optuna = None


THIS_FILE = Path(__file__).resolve()
BUILDINGS_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = BUILDINGS_ROOT.parent

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))


from src.pipeline.model_registry import (
    get_model_spec,
    normalize_model_type,
)

from src.utils import (
    create_train_val_manifests,
    get_backbone_info,
    get_dataset_info,
    load_datasets_config,
    load_general_params,
    load_model_config,
    resolve_train_validation_split,
    save_config_used,
    set_random_seed,
)


# ============================================================
# RUNTIME SPLIT OVERRIDE
# ============================================================

def apply_runtime_split_overrides(
    experiment_config,
    general_params,
):
    experiment_config = dict(
        experiment_config or {}
    )

    general_params = json.loads(
        json.dumps(
            general_params or {}
        )
    )

    override = experiment_config.get(
        "data_split"
    )

    if not isinstance(
        override,
        dict,
    ):
        return general_params

    current = dict(
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
        value = override.get(key)

        if value is not None:
            current[key] = value

    general_params[
        "data_split"
    ] = current

    return general_params


# ============================================================
# BASIC HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def load_json(path: Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    payload: dict,
    path: Path,
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
            json_safe(payload),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


def resolve_device():
    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


def resolve_built_model(build_result):
    if isinstance(
        build_result,
        (tuple, list),
    ):
        if not build_result:
            raise RuntimeError(
                "build_model() returned an empty tuple/list."
            )

        model = build_result[0]

    else:
        model = build_result

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "build_model() did not return a torch.nn.Module."
        )

    try:
        device = next(
            model.parameters()
        ).device

    except StopIteration:
        device = resolve_device()
        model = model.to(device)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        model,
        device,
        trainable_params,
    )


def cleanup_cuda(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SEARCH-SPACE RESOLUTION
# ============================================================

def _condition_matches(
    condition: dict | None,
    sampled: dict,
) -> bool:
    if not condition:
        return True

    parameter = condition.get(
        "parameter"
    )

    if parameter not in sampled:
        return False

    current = sampled.get(
        parameter
    )

    if "equals" in condition:
        return (
            current
            == condition[
                "equals"
            ]
        )

    if "in" in condition:
        return (
            current
            in condition[
                "in"
            ]
        )

    if "not_equals" in condition:
        return (
            current
            != condition[
                "not_equals"
            ]
        )

    raise ValueError(
        f"Unsupported Master Optuna condition: {condition}"
    )


def sample_parameters(
    trial,
    parameter_space: dict,
) -> dict:
    sampled: Dict[str, Any] = {}
    pending = dict(
        parameter_space
    )

    while pending:
        progressed = False

        for name in list(
            pending.keys()
        ):
            spec = pending[name]
            condition = spec.get(
                "condition"
            )

            if (
                condition
                and condition.get(
                    "parameter"
                )
                not in sampled
            ):
                if (
                    condition.get(
                        "parameter"
                    )
                    in pending
                ):
                    continue

            if not _condition_matches(
                condition,
                sampled,
            ):
                pending.pop(name)
                progressed = True
                continue

            kind = str(
                spec.get(
                    "type",
                    "categorical",
                )
            ).lower()

            if kind == "categorical":
                values = spec.get(
                    "values",
                    [],
                )

                if not values:
                    raise ValueError(
                        f"Master Optuna parameter '{name}' "
                        "has no categorical values."
                    )

                value = (
                    trial.suggest_categorical(
                        name,
                        values,
                    )
                )

            elif kind == "float":
                kwargs = {
                    "low":
                        float(
                            spec[
                                "low"
                            ]
                        ),

                    "high":
                        float(
                            spec[
                                "high"
                            ]
                        ),

                    "log":
                        bool(
                            spec.get(
                                "log",
                                False,
                            )
                        ),
                }

                if (
                    spec.get(
                        "step"
                    )
                    is not None
                ):
                    if kwargs[
                        "log"
                    ]:
                        raise ValueError(
                            f"Parameter '{name}' cannot use "
                            "both step and log=True."
                        )

                    kwargs[
                        "step"
                    ] = float(
                        spec[
                            "step"
                        ]
                    )

                value = trial.suggest_float(
                    name,
                    **kwargs,
                )

            elif kind == "int":
                kwargs = {
                    "low":
                        int(
                            spec[
                                "low"
                            ]
                        ),

                    "high":
                        int(
                            spec[
                                "high"
                            ]
                        ),

                    "step":
                        int(
                            spec.get(
                                "step",
                                1,
                            )
                        ),

                    "log":
                        bool(
                            spec.get(
                                "log",
                                False,
                            )
                        ),
                }

                value = trial.suggest_int(
                    name,
                    **kwargs,
                )

            else:
                raise ValueError(
                    f"Unsupported Master Optuna parameter type "
                    f"'{kind}' for '{name}'."
                )

            sampled[name] = value
            pending.pop(name)
            progressed = True

        if not progressed:
            raise ValueError(
                "Could not resolve Master Optuna parameter "
                f"conditions. Remaining: {list(pending)}"
            )

    return sampled


def effective_search_space(
    base_config: dict,
    model_space: dict,
    datasets_config: dict,
) -> dict:
    """
    Intersect JSON choices with datasets/backbones that actually exist.
    """
    model_type = normalize_model_type(
        base_config[
            "model_type"
        ]
    )

    source = str(
        base_config[
            "source_type"
        ]
    ).lower()

    dataset_type = str(
        model_space[
            "dataset_type"
        ]
    ).lower()

    effective = copy.deepcopy(
        model_space
    )

    parameters = effective[
        "parameters"
    ]

    # --------------------------------------------------------
    # DATASET / TILE AVAILABILITY
    # --------------------------------------------------------
    if "tile_size" in parameters:
        candidate_tiles = list(
            parameters[
                "tile_size"
            ].get(
                "values",
                [],
            )
        )

        available_tiles = []

        for tile in candidate_tiles:
            candidate_config = copy.deepcopy(
                base_config
            )

            candidate_config[
                "source_type"
            ] = source

            candidate_config[
                "dataset_type"
            ] = dataset_type

            candidate_config[
                "tile_size"
            ] = int(tile)

            try:
                get_dataset_info(
                    candidate_config,
                    datasets_config,
                )

            except Exception:
                continue

            available_tiles.append(
                int(tile)
            )

        if not available_tiles:
            raise ValueError(
                "No Master Optuna tile size from the JSON exists "
                "in datasets.json for "
                f"model={model_type}, source={source}, "
                f"dataset_type={dataset_type}."
            )

        parameters[
            "tile_size"
        ][
            "values"
        ] = available_tiles

    # --------------------------------------------------------
    # BACKBONE AVAILABILITY
    # --------------------------------------------------------
    if "backbone" in parameters:
        candidate_backbones = list(
            parameters[
                "backbone"
            ].get(
                "values",
                [],
            )
        )

        registry_backbones = list(
            get_model_spec(
                model_type
            ).supported_backbones
        )

        available_backbones = []

        if model_type == "sam_lora":
            # SAM-LoRA uses its own model config/factory.  Do not
            # pass it through generic get_backbone_info().
            try:
                from src.models.sam_lora.factory import (
                    load_sam_lora_config,
                )

                sam_config = load_sam_lora_config(
                    BUILDINGS_ROOT
                    / "config"
                    / "models"
                    / "sam_lora.json"
                )

                supported = sam_config.get(
                    "supported_backbones",
                    {},
                )

                for backbone in candidate_backbones:
                    if backbone not in registry_backbones:
                        continue

                    details = supported.get(
                        backbone
                    )

                    if (
                        isinstance(
                            details,
                            dict,
                        )
                        and not bool(
                            details.get(
                                "enabled",
                                True,
                            )
                        )
                    ):
                        continue

                    available_backbones.append(
                        backbone
                    )

            except Exception:
                # Registry is still a safe fallback.
                available_backbones = [
                    backbone
                    for backbone
                    in candidate_backbones
                    if backbone in registry_backbones
                ]

        elif (
            get_model_spec(
                model_type
            ).family
            == "instance"
        ):
            available_backbones = [
                backbone
                for backbone
                in candidate_backbones
                if backbone in registry_backbones
            ]

        else:
            model_config = load_model_config(
                model_type,
                BUILDINGS_ROOT,
            )

            for backbone in candidate_backbones:
                candidate_config = copy.deepcopy(
                    base_config
                )

                candidate_config[
                    "backbone"
                ] = backbone

                try:
                    get_backbone_info(
                        candidate_config,
                        model_config,
                    )

                except Exception:
                    continue

                available_backbones.append(
                    backbone
                )

        if not available_backbones:
            raise ValueError(
                "No Master Optuna backbone from the JSON is "
                f"supported by installed {model_type}. "
                f"Registry options: {registry_backbones}"
            )

        parameters[
            "backbone"
        ][
            "values"
        ] = available_backbones

    return effective


# ============================================================
# FIXED SPLIT / DATASET CACHE
# ============================================================

def build_general_params(
    base_config: dict,
) -> dict:
    general_params = (
        load_general_params(
            BUILDINGS_ROOT
        )
    )

    general_params = (
        apply_runtime_split_overrides(
            experiment_config=base_config,
            general_params=general_params,
        )
    )

    split = (
        resolve_train_validation_split(
            general_params
        )
    )

    general_params.setdefault(
        "training",
        {},
    )

    general_params[
        "training"
    ][
        "random_seed"
    ] = int(
        split[
            "seed"
        ]
    )

    general_params[
        "training"
    ][
        "seed"
    ] = int(
        split[
            "seed"
        ]
    )

    return general_params


def prepare_cached_split(
    trial_config: dict,
    general_params: dict,
    datasets_config: dict,
    split_cache_root: Path,
) -> Tuple[
    Path,
    Path,
    dict,
    dict,
]:
    dataset_info = get_dataset_info(
        experiment_config=trial_config,
        datasets_config=datasets_config,
    )

    dataset_id = str(
        dataset_info[
            "dataset_id"
        ]
    )

    cache_folder = (
        Path(
            split_cache_root
        )
        / dataset_id
    )

    train_manifest = (
        cache_folder
        / "train_manifest.csv"
    )

    val_manifest = (
        cache_folder
        / "val_manifest.csv"
    )

    if (
        not train_manifest.exists()
        or not val_manifest.exists()
    ):
        cache_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        result = (
            create_train_val_manifests(
                dataset_info=dataset_info,
                output_folder=cache_folder,
                general_params=general_params,
            )
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

    split_info = (
        resolve_train_validation_split(
            general_params
        )
    )

    return (
        train_manifest,
        val_manifest,
        dataset_info,
        split_info,
    )


# ============================================================
# PARAMETER APPLICATION
# ============================================================

def build_trial_config(
    base_config: dict,
    params: dict,
    experiment_id: str,
) -> dict:
    config = copy.deepcopy(
        base_config
    )

    config[
        "experiment_id"
    ] = str(
        experiment_id
    )

    config[
        "model_type"
    ] = normalize_model_type(
        base_config[
            "model_type"
        ]
    )

    if "backbone" in params:
        config[
            "backbone"
        ] = params[
            "backbone"
        ]

    if "tile_size" in params:
        config[
            "tile_size"
        ] = int(
            params[
                "tile_size"
            ]
        )

    if "augmentation" in params:
        config[
            "use_augmentation"
        ] = bool(
            params[
                "augmentation"
            ]
        )

    config[
        "use_optuna"
    ] = False

    config[
        "use_master_optuna"
    ] = True

    config[
        "use_threshold_search"
    ] = False

    config[
        "use_postprocessing"
    ] = False

    # Keep SAM-LoRA architecture provenance visible in the
    # experiment metadata as well as in model_config_used.json.
    if (
        config[
            "model_type"
        ]
        == "sam_lora"
    ):
        if "lora_rank" in params:
            config[
                "sam_lora_rank"
            ] = int(
                params[
                    "lora_rank"
                ]
            )

        if "lora_alpha" in params:
            config[
                "sam_lora_alpha"
            ] = float(
                params[
                    "lora_alpha"
                ]
            )

    return config


def apply_semantic_params(
    general_params: dict,
    model_config: dict,
    params: dict,
) -> Tuple[
    dict,
    dict,
]:
    general = copy.deepcopy(
        general_params
    )

    model = copy.deepcopy(
        model_config
    )

    training = general.setdefault(
        "training",
        {},
    )

    if "optimizer" in params:
        training[
            "optimizer"
        ] = str(
            params[
                "optimizer"
            ]
        ).lower()

    if "learning_rate" in params:
        training[
            "learning_rate"
        ] = float(
            params[
                "learning_rate"
            ]
        )

    if "weight_decay" in params:
        training[
            "weight_decay"
        ] = float(
            params[
                "weight_decay"
            ]
        )

    if "momentum" in params:
        training[
            "momentum"
        ] = float(
            params[
                "momentum"
            ]
        )

    model.setdefault(
        "loss",
        {},
    )

    if "loss_type" in params:
        model[
            "loss"
        ][
            "type"
        ] = str(
            params[
                "loss_type"
            ]
        ).lower()

    if "bce_weight" in params:
        bce_weight = float(
            params[
                "bce_weight"
            ]
        )

        model[
            "loss"
        ][
            "bce_weight"
        ] = bce_weight

        model[
            "loss"
        ][
            "dice_weight"
        ] = (
            1.0
            - bce_weight
        )

    return (
        general,
        model,
    )


# ============================================================
# AUGMENTATION
# ============================================================

def build_augmentation_if_enabled(
    enabled: bool,
):
    if not bool(
        enabled
    ):
        return (
            None,
            None,
        )

    augmentation_config_path = (
        BUILDINGS_ROOT
        / "config"
        / "optional"
        / "augmentation.json"
    )

    if not augmentation_config_path.exists():
        raise FileNotFoundError(
            "Master Optuna selected augmentation but "
            f"config is missing: {augmentation_config_path}"
        )

    from src.optional.augmentation import (
        build_augmentation,
    )

    augmentation = (
        build_augmentation(
            config_path=augmentation_config_path,
            enabled_override=True,
        )
    )

    if augmentation is None:
        raise RuntimeError(
            "Could not build the training augmentation pipeline."
        )

    return (
        augmentation,
        augmentation_config_path,
    )


# ============================================================
# U-NET / DEEPLAB TRAINING
# ============================================================

def train_semantic_configuration(
    base_config: dict,
    params: dict,
    general_params: dict,
    datasets_config: dict,
    split_cache_root: Path,
    output_folder: Path,
    epochs: int,
    experiment_id: str,
    final_training: bool = False,
) -> dict:
    from src.models.factory import (
        build_model,
    )

    from src.train import (
        build_loss_function,
        build_optimizer,
        create_dataloaders,
        train_model,
    )

    model_type = normalize_model_type(
        base_config[
            "model_type"
        ]
    )

    trial_config = build_trial_config(
        base_config,
        params,
        experiment_id,
    )

    (
        train_manifest,
        val_manifest,
        dataset_info,
        split_info,
    ) = prepare_cached_split(
        trial_config=trial_config,
        general_params=general_params,
        datasets_config=datasets_config,
        split_cache_root=split_cache_root,
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    if final_training:
        final_manifest_result = (
            create_train_val_manifests(
                dataset_info=dataset_info,
                output_folder=output_folder,
                general_params=general_params,
            )
        )

        train_manifest = Path(
            final_manifest_result[
                "train_manifest_csv"
            ]
        )

        val_manifest = Path(
            final_manifest_result[
                "val_manifest_csv"
            ]
        )

    raw_model_config = (
        load_model_config(
            model_type,
            BUILDINGS_ROOT,
        )
    )

    (
        trial_general,
        trial_model_config,
    ) = apply_semantic_params(
        general_params=general_params,
        model_config=raw_model_config,
        params=params,
    )

    backbone_info = (
        get_backbone_info(
            experiment_config=trial_config,
            model_config=trial_model_config,
        )
    )

    seed = int(
        split_info[
            "seed"
        ]
    )

    set_random_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    (
        augmentation,
        augmentation_path,
    ) = build_augmentation_if_enabled(
        trial_config.get(
            "use_augmentation",
            False,
        )
    )

    batch_size = int(
        params[
            "batch_size"
        ]
    )

    training_cfg = (
        trial_general.get(
            "training",
            {},
        )
    )

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    ) = create_dataloaders(
        train_manifest_csv=train_manifest,
        val_manifest_csv=val_manifest,
        batch_size=batch_size,
        num_workers=int(
            training_cfg.get(
                "num_workers",
                0,
            )
        ),
        pin_memory=bool(
            training_cfg.get(
                "pin_memory",
                True,
            )
        ),
        model_type=model_type,
        train_augmentation=augmentation,
    )

    build_result = build_model(
        experiment_config=trial_config,
        model_config=trial_model_config,
        backbone_info=backbone_info,
        move_to_device=True,
    )

    (
        model,
        device,
        trainable_params,
    ) = resolve_built_model(
        build_result
    )

    loss_fn = build_loss_function(
        trial_model_config
    )

    optimizer = build_optimizer(
        model=model,
        general_params=trial_general,
        model_config=trial_model_config,
    )

    save_json(
        params,
        output_folder
        / "master_params.json",
    )

    save_json(
        trial_config,
        output_folder
        / "experiment_config_used.json",
    )

    save_json(
        trial_general,
        output_folder
        / "general_params_used.json",
    )

    save_json(
        trial_model_config,
        output_folder
        / "model_config_used.json",
    )

    save_json(
        dataset_info,
        output_folder
        / "dataset_used.json",
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MASTER OPTUNA — SEMANTIC CONFIGURATION"
    )
    print(
        "=" * 72
    )
    print(
        f"Model: {model_type}"
    )
    print(
        f"Source: {trial_config['source_type']}"
    )
    print(
        f"Backbone: {trial_config['backbone']}"
    )
    print(
        f"Tile Size: {trial_config['tile_size']}"
    )
    print(
        f"Batch Size: {batch_size}"
    )
    print(
        f"Epochs: {int(epochs)}"
    )
    print(
        f"Split: {split_info['train_percent']:.0f}/"
        f"{split_info['validation_percent']:.0f}"
    )
    print(
        f"Seed: {seed}"
    )
    print(
        "Augmentation:",
        "ON"
        if augmentation is not None
        else "OFF",
    )
    print(
        "Objective Dataset: VALIDATION ONLY"
    )
    print(
        "External Test Used: NO"
    )
    print(
        "=" * 72
        + "\n"
    )

    try:
        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=int(epochs),
            output_folder=output_folder,
            experiment_config=trial_config,
            model_config=trial_model_config,
            backbone_info=backbone_info,
            trainable_params=trainable_params,
            augmentation_enabled=(
                augmentation
                is not None
            ),
            augmentation_config_path=(
                str(
                    augmentation_path
                )
                if augmentation_path
                is not None
                else None
            ),
        )

    finally:
        cleanup_cuda(
            model,
            optimizer,
            train_loader,
            val_loader,
        )

    summary.update(
        {
            "master_optuna":
                True,

            "master_params":
                json_safe(
                    params
                ),

            "model_type":
                model_type,

            "backbone":
                trial_config.get(
                    "backbone"
                ),

            "tile_size":
                int(
                    trial_config[
                        "tile_size"
                    ]
                ),

            "batch_size":
                batch_size,

            "train_samples":
                int(
                    len(
                        train_dataset
                    )
                ),

            "validation_samples":
                int(
                    len(
                        val_dataset
                    )
                ),

            "selection_dataset":
                "validation_manifest",

            "external_test_used":
                False,

            "final_training":
                bool(
                    final_training
                ),

            "training_engine":
                "src.train",
        }
    )

    save_json(
        summary,
        output_folder
        / "training_summary.json",
    )

    if final_training:
        save_config_used(
            experiment_config=trial_config,
            general_params=trial_general,
            model_config=trial_model_config,
            datasets_config=datasets_config,
            output_folder=output_folder,
        )

    return summary


# ============================================================
# SAM-LORA TRAINING
# ============================================================

def build_sam_lora_master_config(
    trial_config: dict,
    params: dict,
    epochs: int,
    output_folder: Path,
) -> Tuple[
    dict,
    Path,
]:
    """
    Create a trial-specific SAM-LoRA config without changing the
    canonical config/models/sam_lora.json.
    """
    from src.models.sam_lora.factory import (
        load_sam_lora_config,
    )

    base_path = (
        BUILDINGS_ROOT
        / "config"
        / "models"
        / "sam_lora.json"
    )

    sam_config = copy.deepcopy(
        load_sam_lora_config(
            base_path
        )
    )

    backbone = str(
        params.get(
            "backbone",
            trial_config.get(
                "backbone",
                sam_config.get(
                    "default_backbone",
                    "vit_b",
                ),
            ),
        )
    ).strip().lower()

    sam_config[
        "default_backbone"
    ] = backbone

    # --------------------------------------------------------
    # LoRA architecture
    # --------------------------------------------------------
    lora = sam_config.setdefault(
        "lora",
        {},
    )

    if "lora_rank" in params:
        lora[
            "rank"
        ] = int(
            params[
                "lora_rank"
            ]
        )

    if "lora_alpha" in params:
        lora[
            "alpha"
        ] = float(
            params[
                "lora_alpha"
            ]
        )

    # Keep Q+V/all-block policy from the canonical SAM-LoRA
    # config unless it is explicitly extended in a future study.

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    training = sam_config.setdefault(
        "training",
        {},
    )

    training[
        "epochs"
    ] = int(
        epochs
    )

    batch_size = int(
        params.get(
            "batch_size",
            training.get(
                "default_batch_size",
                1,
            ),
        )
    )

    training[
        "default_batch_size"
    ] = batch_size

    training.setdefault(
        "recommended_batch_size_by_backbone",
        {},
    )

    training[
        "recommended_batch_size_by_backbone"
    ][
        backbone
    ] = batch_size

    training[
        "optimizer"
    ] = "adamw"

    if "weight_decay" in params:
        training[
            "weight_decay"
        ] = float(
            params[
                "weight_decay"
            ]
        )

    # Master Optuna selects LR directly.
    lr_value = float(
        params.get(
            "learning_rate",
            1e-4,
        )
    )

    lr_cfg = training.setdefault(
        "learning_rate",
        {},
    )

    lr_cfg[
        "mode"
    ] = "fixed"

    lr_cfg[
        "value"
    ] = lr_value

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    loss_cfg = training.setdefault(
        "loss",
        {},
    )

    loss_cfg[
        "type"
    ] = "bce_dice"

    bce_weight = float(
        params.get(
            "bce_weight",
            loss_cfg.get(
                "bce_weight",
                0.5,
            ),
        )
    )

    loss_cfg[
        "bce_weight"
    ] = bce_weight

    loss_cfg[
        "dice_weight"
    ] = (
        1.0
        - bce_weight
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        output_folder
        / "sam_lora_master_config.json"
    )

    save_json(
        sam_config,
        config_path,
    )

    # Keep conventional artifact name too.  This is important
    # when rank/alpha differ from the canonical config.
    save_json(
        sam_config,
        output_folder
        / "model_config_used.json",
    )

    return (
        sam_config,
        config_path,
    )


def train_sam_lora_configuration(
    base_config: dict,
    params: dict,
    general_params: dict,
    datasets_config: dict,
    split_cache_root: Path,
    output_folder: Path,
    epochs: int,
    experiment_id: str,
    final_training: bool = False,
) -> dict:
    from src.train import (
        create_dataloaders,
    )

    from src.models.sam_lora.factory import (
        build_sam_lora_factory_bundle,
    )

    from src.models.sam_lora.trainer import (
        train_sam_lora,
    )

    trial_config = build_trial_config(
        base_config,
        params,
        experiment_id,
    )

    (
        train_manifest,
        val_manifest,
        dataset_info,
        split_info,
    ) = prepare_cached_split(
        trial_config=trial_config,
        general_params=general_params,
        datasets_config=datasets_config,
        split_cache_root=split_cache_root,
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    if final_training:
        final_manifest_result = (
            create_train_val_manifests(
                dataset_info=dataset_info,
                output_folder=output_folder,
                general_params=general_params,
            )
        )

        train_manifest = Path(
            final_manifest_result[
                "train_manifest_csv"
            ]
        )

        val_manifest = Path(
            final_manifest_result[
                "val_manifest_csv"
            ]
        )

    (
        sam_config,
        sam_config_path,
    ) = build_sam_lora_master_config(
        trial_config=trial_config,
        params=params,
        epochs=epochs,
        output_folder=output_folder,
    )

    seed = int(
        split_info[
            "seed"
        ]
    )

    set_random_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    (
        augmentation,
        augmentation_path,
    ) = build_augmentation_if_enabled(
        trial_config.get(
            "use_augmentation",
            False,
        )
    )

    batch_size = int(
        params.get(
            "batch_size",
            1,
        )
    )

    training_cfg = (
        general_params.get(
            "training",
            {},
        )
    )

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    ) = create_dataloaders(
        train_manifest_csv=train_manifest,
        val_manifest_csv=val_manifest,
        batch_size=batch_size,
        num_workers=int(
            training_cfg.get(
                "num_workers",
                0,
            )
        ),
        pin_memory=bool(
            training_cfg.get(
                "pin_memory",
                True,
            )
        ),
        model_type="sam_lora",
        train_augmentation=augmentation,
    )

    device = resolve_device()

    bundle = (
        build_sam_lora_factory_bundle(
            config_path=sam_config_path,
            backbone=trial_config.get(
                "backbone",
                sam_config.get(
                    "default_backbone",
                    "vit_b",
                ),
            ),
            device=None,
            verbose_model_build=False,
        )
    )

    trainer_kwargs = dict(
        bundle.trainer_kwargs
    )

    trainer_kwargs[
        "epochs"
    ] = int(
        epochs
    )

    learning_rate = float(
        params.get(
            "learning_rate",
            1e-4,
        )
    )

    weight_decay = float(
        params.get(
            "weight_decay",
            sam_config.get(
                "training",
                {},
            ).get(
                "weight_decay",
                1e-5,
            ),
        )
    )

    trainer_kwargs[
        "weight_decay"
    ] = weight_decay

    # Compatibility across the current SAM-LoRA trainer/factory
    # revisions.  The trial LR is always fixed.
    trainer_kwargs[
        "learning_rate_mode"
    ] = "fixed"

    trainer_kwargs[
        "fixed_learning_rate"
    ] = learning_rate

    trainer_kwargs[
        "learning_rate"
    ] = learning_rate

    save_json(
        params,
        output_folder
        / "master_params.json",
    )

    save_json(
        trial_config,
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

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MASTER OPTUNA — SAM-LoRA CONFIGURATION"
    )
    print(
        "=" * 72
    )
    print(
        f"Source: {trial_config['source_type']}"
    )
    print(
        f"Backbone: {bundle.backbone}"
    )
    print(
        f"Tile Size: {trial_config['tile_size']}"
    )
    print(
        f"Batch Size: {batch_size}"
    )
    print(
        f"Epochs: {int(epochs)}"
    )
    print(
        f"Learning Rate: {learning_rate}"
    )
    print(
        "Learning Rate Mode: FIXED"
    )
    print(
        "Auto-LR Range Test: SKIPPED"
    )
    print(
        f"Weight Decay: {weight_decay}"
    )
    print(
        "BCE Weight:",
        sam_config[
            "training"
        ][
            "loss"
        ][
            "bce_weight"
        ],
    )
    print(
        "Dice Weight:",
        sam_config[
            "training"
        ][
            "loss"
        ][
            "dice_weight"
        ],
    )
    print(
        "LoRA Rank:",
        sam_config.get(
            "lora",
            {},
        ).get(
            "rank"
        ),
    )
    print(
        "LoRA Alpha:",
        sam_config.get(
            "lora",
            {},
        ).get(
            "alpha"
        ),
    )
    print(
        "LoRA Targets:",
        sam_config.get(
            "lora",
            {},
        ).get(
            "target_projections"
        ),
    )
    print(
        "LoRA Blocks:",
        sam_config.get(
            "lora",
            {},
        ).get(
            "apply_to_transformer_blocks"
        ),
    )
    print(
        f"Split: {split_info['train_percent']:.0f}/"
        f"{split_info['validation_percent']:.0f}"
    )
    print(
        f"Seed: {seed}"
    )
    print(
        "Augmentation:",
        "ON"
        if augmentation is not None
        else "OFF",
    )
    print(
        "Objective Dataset: VALIDATION ONLY"
    )
    print(
        "External Test Used: NO"
    )
    print(
        "=" * 72
        + "\n"
    )

    result = None

    try:
        result = train_sam_lora(
            model_factory=bundle.model_factory,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=bundle.criterion,
            output_dir=output_folder,
            device=device,
            **trainer_kwargs,
        )

    finally:
        cleanup_cuda(
            train_loader,
            val_loader,
        )

    summary_path = (
        output_folder
        / "training_summary.json"
    )

    if summary_path.exists():
        summary = load_json(
            summary_path
        )
    else:
        summary = {}

    best_iou = float(
        getattr(
            result,
            "best_validation_iou",
            summary.get(
                "best_validation_iou",
                summary.get(
                    "best_val_iou",
                    0.0,
                ),
            ),
        )
    )

    if best_iou > 1.0:
        best_iou /= 100.0

    best_epoch = int(
        getattr(
            result,
            "best_epoch",
            summary.get(
                "best_epoch",
                -1,
            ),
        )
    )

    best_checkpoint = str(
        getattr(
            result,
            "best_checkpoint",
            output_folder
            / "best_model.pth",
        )
    )

    summary.update(
        {
            "master_optuna":
                True,

            "master_params":
                json_safe(
                    params
                ),

            "model_type":
                "sam_lora",

            "backbone":
                bundle.backbone,

            "tile_size":
                int(
                    trial_config[
                        "tile_size"
                    ]
                ),

            "batch_size":
                batch_size,

            "best_val_iou":
                best_iou,

            "best_validation_iou":
                best_iou,

            "best_validation_iou_percent":
                best_iou
                * 100.0,

            "best_epoch":
                best_epoch,

            "best_model_path":
                best_checkpoint,

            "train_samples":
                int(
                    len(
                        train_dataset
                    )
                ),

            "validation_samples":
                int(
                    len(
                        val_dataset
                    )
                ),

            "learning_rate":
                learning_rate,

            "resolved_learning_rate":
                learning_rate,

            "learning_rate_mode":
                "fixed",

            "auto_lr_used":
                False,

            "auto_lr_skipped":
                True,

            "weight_decay":
                weight_decay,

            "bce_weight":
                float(
                    sam_config[
                        "training"
                    ][
                        "loss"
                    ][
                        "bce_weight"
                    ]
                ),

            "dice_weight":
                float(
                    sam_config[
                        "training"
                    ][
                        "loss"
                    ][
                        "dice_weight"
                    ]
                ),

            "lora_rank":
                int(
                    sam_config.get(
                        "lora",
                        {},
                    ).get(
                        "rank",
                        8,
                    )
                ),

            "lora_alpha":
                float(
                    sam_config.get(
                        "lora",
                        {},
                    ).get(
                        "alpha",
                        16.0,
                    )
                ),

            "sam_lora_model_config_path":
                str(
                    sam_config_path
                ),

            "augmentation_enabled":
                bool(
                    augmentation
                    is not None
                ),

            "augmentation_config_path":
                (
                    str(
                        augmentation_path
                    )
                    if augmentation_path
                    is not None
                    else None
                ),

            "selection_dataset":
                "validation_manifest",

            "external_test_used":
                False,

            "final_training":
                bool(
                    final_training
                ),

            "training_engine":
                "sam_lora",
        }
    )

    save_json(
        summary,
        summary_path,
    )

    if final_training:
        save_config_used(
            experiment_config=trial_config,
            general_params=general_params,
            model_config=sam_config,
            datasets_config=datasets_config,
            output_folder=output_folder,
        )

    return summary


# ============================================================
# MASK R-CNN TRAINING
# ============================================================

def train_maskrcnn_configuration(
    base_config: dict,
    params: dict,
    general_params: dict,
    datasets_config: dict,
    split_cache_root: Path,
    output_folder: Path,
    epochs: int,
    experiment_id: str,
    final_training: bool = False,
) -> dict:
    from src.instance.maskrcnn_engine import (
        read_manifest_pairs,
        train_model,
    )

    trial_config = build_trial_config(
        base_config,
        params,
        experiment_id,
    )

    (
        train_manifest,
        val_manifest,
        dataset_info,
        split_info,
    ) = prepare_cached_split(
        trial_config=trial_config,
        general_params=general_params,
        datasets_config=datasets_config,
        split_cache_root=split_cache_root,
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    if final_training:
        final_manifest_result = (
            create_train_val_manifests(
                dataset_info=dataset_info,
                output_folder=output_folder,
                general_params=general_params,
            )
        )

        train_manifest = Path(
            final_manifest_result[
                "train_manifest_csv"
            ]
        )

        val_manifest = Path(
            final_manifest_result[
                "val_manifest_csv"
            ]
        )

    train_pairs = (
        read_manifest_pairs(
            train_manifest
        )
    )

    val_pairs = (
        read_manifest_pairs(
            val_manifest
        )
    )

    seed = int(
        split_info[
            "seed"
        ]
    )

    (
        augmentation,
        augmentation_path,
    ) = build_augmentation_if_enabled(
        trial_config.get(
            "use_augmentation",
            False,
        )
    )

    optimizer = str(
        params.get(
            "optimizer",
            "adamw",
        )
    ).lower()

    learning_rate = float(
        params.get(
            "learning_rate",
            1e-4,
        )
    )

    weight_decay = float(
        params.get(
            "weight_decay",
            1e-5,
        )
    )

    momentum = float(
        params.get(
            "momentum",
            0.9,
        )
    )

    scheduler = str(
        params.get(
            "scheduler",
            "cosine",
        )
    ).lower()

    pretrained = bool(
        params.get(
            "pretrained",
            True,
        )
    )

    batch_size = int(
        params[
            "batch_size"
        ]
    )

    save_json(
        params,
        output_folder
        / "master_params.json",
    )

    save_json(
        trial_config,
        output_folder
        / "experiment_config_used.json",
    )

    save_json(
        dataset_info,
        output_folder
        / "dataset_used.json",
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MASTER OPTUNA — MASK R-CNN CONFIGURATION"
    )
    print(
        "=" * 72
    )
    print(
        f"Source: {trial_config['source_type']}"
    )
    print(
        f"Backbone: {trial_config['backbone']}"
    )
    print(
        f"Tile Size: {trial_config['tile_size']}"
    )
    print(
        f"Batch Size: {batch_size}"
    )
    print(
        f"Optimizer: {optimizer}"
    )
    print(
        f"Scheduler: {scheduler}"
    )
    print(
        f"Pretrained: {pretrained}"
    )
    print(
        f"Epochs: {int(epochs)}"
    )
    print(
        f"Split: {split_info['train_percent']:.0f}/"
        f"{split_info['validation_percent']:.0f}"
    )
    print(
        f"Seed: {seed}"
    )
    print(
        "Augmentation:",
        "ON"
        if augmentation is not None
        else "OFF",
    )
    print(
        "Objective Dataset: VALIDATION ONLY"
    )
    print(
        "External Test Used: NO"
    )
    print(
        "=" * 72
        + "\n"
    )

    summary = train_model(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_folder=output_folder,
        epochs=int(epochs),
        batch_size=batch_size,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        momentum=momentum,
        pretrained=pretrained,
        seed=seed,
        train_augmentation=augmentation,
        phase=(
            "master_optuna_final"
            if final_training
            else "master_optuna_trial"
        ),
        nms_threshold=0.50,
        detections_per_image=300,
    )

    summary.update(
        {
            "master_optuna":
                True,

            "master_params":
                json_safe(
                    params
                ),

            "model_type":
                "maskrcnn",

            "backbone":
                trial_config.get(
                    "backbone"
                ),

            "tile_size":
                int(
                    trial_config[
                        "tile_size"
                    ]
                ),

            "selection_dataset":
                "validation_manifest",

            "external_test_used":
                False,

            "final_training":
                bool(
                    final_training
                ),

            "augmentation_config_path":
                (
                    str(
                        augmentation_path
                    )
                    if augmentation_path
                    is not None
                    else None
                ),
        }
    )

    save_json(
        summary,
        output_folder
        / "training_summary.json",
    )

    if final_training:
        save_json(
            trial_config,
            output_folder
            / "experiment_config_used.json",
        )

        save_json(
            general_params,
            output_folder
            / "general_params_used.json",
        )

        save_json(
            datasets_config,
            output_folder
            / "datasets_config_used.json",
        )

    cleanup_cuda()

    return summary


# ============================================================
# UNIFIED CONFIGURATION TRAINER
# ============================================================

def train_configuration(
    base_config: dict,
    params: dict,
    general_params: dict,
    datasets_config: dict,
    split_cache_root: Path,
    output_folder: Path,
    epochs: int,
    experiment_id: str,
    final_training: bool = False,
) -> dict:
    model_type = normalize_model_type(
        base_config[
            "model_type"
        ]
    )

    if model_type == "sam_lora":
        return (
            train_sam_lora_configuration(
                base_config=base_config,
                params=params,
                general_params=general_params,
                datasets_config=datasets_config,
                split_cache_root=split_cache_root,
                output_folder=output_folder,
                epochs=epochs,
                experiment_id=experiment_id,
                final_training=final_training,
            )
        )

    family = (
        get_model_spec(
            model_type
        ).family
    )

    if family == "semantic":
        return (
            train_semantic_configuration(
                base_config=base_config,
                params=params,
                general_params=general_params,
                datasets_config=datasets_config,
                split_cache_root=split_cache_root,
                output_folder=output_folder,
                epochs=epochs,
                experiment_id=experiment_id,
                final_training=final_training,
            )
        )

    return (
        train_maskrcnn_configuration(
            base_config=base_config,
            params=params,
            general_params=general_params,
            datasets_config=datasets_config,
            split_cache_root=split_cache_root,
            output_folder=output_folder,
            epochs=epochs,
            experiment_id=experiment_id,
            final_training=final_training,
        )
    )


# ============================================================
# STUDY HELPERS
# ============================================================

def best_validation_iou(
    summary: dict,
) -> float:
    for key in (
        "best_val_iou",
        "best_validation_iou",
        "validation_iou",
    ):
        if summary.get(
            key
        ) is not None:
            value = float(
                summary[
                    key
                ]
            )

            if value > 1.0:
                value /= 100.0

            return value

    return 0.0


def _copy_if_exists(
    source: Path,
    destination: Path,
) -> None:
    source = Path(
        source
    )

    destination = Path(
        destination
    )

    if source.exists():
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )


# ============================================================
# WINNER PROMOTION
# ============================================================

def promote_winning_trial(
    *,
    base_config: dict,
    best_params: dict,
    best_trial_number: int,
    best_trial_value: float,
    trial_epochs: int,
    trials_root: Path,
    experiment_folder: Path,
    general_params: dict,
    datasets_config: dict,
) -> tuple[
    dict,
    dict,
]:
    """
    Promote the exact Master Optuna winning checkpoint without retraining.
    """
    winner_folder = (
        Path(
            trials_root
        )
        / f"trial_{int(best_trial_number):04d}"
    )

    winner_checkpoint = (
        winner_folder
        / "best_model.pth"
    )

    winner_summary_path = (
        winner_folder
        / "training_summary.json"
    )

    if not winner_checkpoint.exists():
        raise RuntimeError(
            "Winning Master Optuna checkpoint is missing: "
            f"{winner_checkpoint}"
        )

    if not winner_summary_path.exists():
        raise RuntimeError(
            "Winning trial training summary is missing: "
            f"{winner_summary_path}"
        )

    winner_summary = load_json(
        winner_summary_path
    )

    winner_iou = best_validation_iou(
        winner_summary
    )

    if (
        abs(
            float(
                winner_iou
            )
            - float(
                best_trial_value
            )
        )
        > 1e-8
    ):
        raise RuntimeError(
            "Winning trial Validation IoU does not match "
            "the Optuna study value "
            f"({winner_iou} vs {best_trial_value})."
        )

    experiment_folder = Path(
        experiment_folder
    )

    experiment_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_config = build_trial_config(
        base_config=base_config,
        params=best_params,
        experiment_id=base_config[
            "experiment_id"
        ],
    )

    final_config[
        "master_optuna_best_trial"
    ] = int(
        best_trial_number
    )

    final_config[
        "master_optuna_best_params"
    ] = json_safe(
        best_params
    )

    final_config[
        "master_optuna_trial_epochs"
    ] = int(
        trial_epochs
    )

    final_config[
        "master_optuna_winner_promoted_without_retraining"
    ] = True

    dataset_info = get_dataset_info(
        experiment_config=final_config,
        datasets_config=datasets_config,
    )

    create_train_val_manifests(
        dataset_info=dataset_info,
        output_folder=experiment_folder,
        general_params=general_params,
    )

    promoted_checkpoint = (
        experiment_folder
        / "best_model.pth"
    )

    shutil.copy2(
        winner_checkpoint,
        promoted_checkpoint,
    )

    # --------------------------------------------------------
    # COMMON WINNER ARTIFACTS
    # --------------------------------------------------------
    for artifact in (
        "training_log.csv",
        "training_history.json",
        "general_params_used.json",
        "model_config_used.json",
        "dataset_used.json",
        "auto_lr.json",
    ):
        _copy_if_exists(
            winner_folder
            / artifact,
            experiment_folder
            / artifact,
        )

    # --------------------------------------------------------
    # SAM-LoRA exact architecture provenance
    # --------------------------------------------------------
    model_type = normalize_model_type(
        base_config[
            "model_type"
        ]
    )

    sam_selected_config = None

    if model_type == "sam_lora":
        winner_sam_config = (
            winner_folder
            / "sam_lora_master_config.json"
        )

        if not winner_sam_config.exists():
            # model_config_used.json is the equivalent fallback.
            winner_sam_config = (
                winner_folder
                / "model_config_used.json"
            )

        if not winner_sam_config.exists():
            raise RuntimeError(
                "Winning SAM-LoRA trial does not contain its "
                "model configuration. Refusing to promote a "
                "checkpoint whose LoRA rank/alpha cannot be "
                "reconstructed safely."
            )

        sam_selected_config = (
            experiment_folder
            / "sam_lora_selected_config.json"
        )

        shutil.copy2(
            winner_sam_config,
            sam_selected_config,
        )

        # Also keep the standard model_config_used name authoritative.
        shutil.copy2(
            winner_sam_config,
            experiment_folder
            / "model_config_used.json",
        )

        final_config[
            "sam_lora_model_config_path"
        ] = str(
            sam_selected_config
        )

        if "lora_rank" in best_params:
            final_config[
                "sam_lora_rank"
            ] = int(
                best_params[
                    "lora_rank"
                ]
            )

        if "lora_alpha" in best_params:
            final_config[
                "sam_lora_alpha"
            ] = float(
                best_params[
                    "lora_alpha"
                ]
            )

    # --------------------------------------------------------
    # Authoritative final metadata
    # --------------------------------------------------------
    save_json(
        final_config,
        experiment_folder
        / "experiment_config_used.json",
    )

    save_json(
        general_params,
        experiment_folder
        / "general_params_used.json",
    )

    save_json(
        datasets_config,
        experiment_folder
        / "datasets_config_used.json",
    )

    save_json(
        dataset_info,
        experiment_folder
        / "dataset_used.json",
    )

    save_json(
        best_params,
        experiment_folder
        / "master_params.json",
    )

    promoted_summary = copy.deepcopy(
        winner_summary
    )

    promoted_summary.update(
        {
            "experiment_id":
                str(
                    base_config[
                        "experiment_id"
                    ]
                ),

            "master_optuna":
                True,

            "master_optuna_winner":
                True,

            "master_optuna_best_trial":
                int(
                    best_trial_number
                ),

            "master_params":
                json_safe(
                    best_params
                ),

            "best_val_iou":
                float(
                    best_trial_value
                ),

            "best_validation_iou":
                float(
                    best_trial_value
                ),

            "best_validation_iou_percent":
                float(
                    best_trial_value
                    * 100.0
                ),

            "best_model_path":
                str(
                    promoted_checkpoint
                ),

            "training_log_csv":
                str(
                    experiment_folder
                    / "training_log.csv"
                ),

            "training_summary_json":
                str(
                    experiment_folder
                    / "training_summary.json"
                ),

            "selection_dataset":
                "validation_manifest",

            "external_test_used":
                False,

            "retrained_after_master_optuna":
                False,

            "promoted_from_trial_folder":
                str(
                    winner_folder
                ),

            "sam_lora_selected_config":
                (
                    str(
                        sam_selected_config
                    )
                    if sam_selected_config
                    is not None
                    else None
                ),
        }
    )

    save_json(
        promoted_summary,
        experiment_folder
        / "training_summary.json",
    )

    return (
        final_config,
        promoted_summary,
    )


# ============================================================
# MASTER OPTUNA STUDY
# ============================================================

def run_master_optuna_study(
    base_config: dict,
    search_space_path: Path,
    n_trials: int = 30,
    trial_epochs: int = 40,
) -> dict:
    if optuna is None:
        raise ImportError(
            "Optuna is not installed in this Python environment."
        )

    model_type = normalize_model_type(
        base_config.get(
            "model_type",
            "",
        )
    )

    source = str(
        base_config.get(
            "source_type",
            "",
        )
    ).lower().strip()

    if not source:
        raise ValueError(
            "Master Optuna requires source_type."
        )

    experiment_id = str(
        base_config.get(
            "experiment_id",
            "",
        )
    ).strip()

    if not experiment_id:
        raise ValueError(
            "Master Optuna requires experiment_id."
        )

    search_payload = load_json(
        search_space_path
    )

    model_space = (
        search_payload.get(
            "models",
            {},
        ).get(
            model_type
        )
    )

    if not model_space:
        raise ValueError(
            "No Master Optuna search space exists for "
            f"model '{model_type}'."
        )

    # --------------------------------------------------------
    # FIXED SCIENTIFIC CONTEXT
    # --------------------------------------------------------
    base_config = copy.deepcopy(
        base_config
    )

    base_config[
        "experiment_id"
    ] = experiment_id

    base_config[
        "model_type"
    ] = model_type

    base_config[
        "source_type"
    ] = source

    base_config[
        "dataset_type"
    ] = str(
        model_space[
            "dataset_type"
        ]
    ).lower()

    base_config[
        "use_master_optuna"
    ] = True

    base_config[
        "use_optuna"
    ] = False

    base_config[
        "use_threshold_search"
    ] = False

    base_config[
        "use_postprocessing"
    ] = False

    general_params = build_general_params(
        base_config
    )

    datasets_config = (
        load_datasets_config(
            BUILDINGS_ROOT
        )
    )

    split_info = (
        resolve_train_validation_split(
            general_params
        )
    )

    effective_space = (
        effective_search_space(
            base_config=base_config,
            model_space=model_space,
            datasets_config=datasets_config,
        )
    )

    master_root = (
        BUILDINGS_ROOT
        / "outputs"
        / "master_optuna"
        / experiment_id
    )

    trials_root = (
        master_root
        / "trials"
    )

    split_cache_root = (
        master_root
        / "split_cache"
    )

    master_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_folder = (
        BUILDINGS_ROOT
        / "outputs"
        / "experiments"
        / experiment_id
    )

    final_receipt = (
        experiment_folder
        / "final_test"
        / "final_test_receipt.json"
    )

    master_selection = (
        experiment_folder
        / "master_optuna_selection_summary.json"
    )

    if final_receipt.exists():
        raise RuntimeError(
            "This experiment already has a final external-test "
            "receipt. Master Optuna refuses to modify it."
        )

    if master_selection.exists():
        raise RuntimeError(
            "Master Optuna already completed for this experiment. "
            "Use a new experiment ID for a new study."
        )

    save_json(
        search_payload,
        master_root
        / "search_space_requested.json",
    )

    save_json(
        effective_space,
        master_root
        / "search_space_effective.json",
    )

    save_json(
        base_config,
        master_root
        / "base_experiment_config.json",
    )

    save_json(
        split_info,
        master_root
        / "fixed_split.json",
    )

    study_cfg = (
        search_payload.get(
            "study",
            {},
        )
    )

    seed = int(
        split_info[
            "seed"
        ]
    )

    n_startup = int(
        study_cfg.get(
            "n_startup_trials",
            10,
        )
    )

    keep_trial_checkpoints = bool(
        study_cfg.get(
            "keep_trial_checkpoints",
            False,
        )
    )

    storage = (
        "sqlite:///"
        + (
            master_root
            / "master_optuna_study.db"
        ).as_posix()
    )

    study_name = (
        f"{experiment_id}_master_optuna"
    )

    sampler = (
        optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
        )
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    completed_before = [
        trial
        for trial
        in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    remaining = max(
        0,
        int(
            n_trials
        )
        - len(
            completed_before
        ),
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "MASTER OPTUNA START — DIRECT WINNER MODE"
    )
    print(
        "=" * 78
    )
    print(
        f"Experiment ID: {experiment_id}"
    )
    print(
        f"Model: {model_type} (FIXED)"
    )
    print(
        f"Source: {source} (FIXED)"
    )
    print(
        "Train / Validation:",
        f"{split_info['train_percent']:.0f}% / "
        f"{split_info['validation_percent']:.0f}% (FIXED)",
    )
    print(
        f"Seed: {seed} (FIXED)"
    )
    print(
        f"Trials Target: {int(n_trials)}"
    )
    print(
        f"Epochs Per Trial: {int(trial_epochs)}"
    )
    print(
        "Trial Score: BEST Validation IoU reached inside each trial"
    )
    print(
        "Winner Policy: exact highest-IoU trial checkpoint"
    )
    print(
        "Second Training of Winner: NO"
    )
    print(
        "External Test Accessed: NO"
    )

    if model_type == "sam_lora":
        print(
            "SAM-LoRA Auto-LR: OFF inside Master Optuna"
        )
        print(
            "SAM-LoRA LR: Optuna-selected FIXED LR"
        )

    print(
        "=" * 78
        + "\n"
    )

    parameter_space = (
        effective_space[
            "parameters"
        ]
    )

    def objective(trial):
        params = sample_parameters(
            trial,
            parameter_space,
        )

        trial_folder = (
            trials_root
            / f"trial_{trial.number:04d}"
        )

        trial_experiment_id = (
            f"{experiment_id}"
            f"__master_trial_{trial.number:04d}"
        )

        save_json(
            {
                "trial":
                    int(
                        trial.number
                    ),

                "params":
                    params,

                "epochs":
                    int(
                        trial_epochs
                    ),

                "objective":
                    "best_validation_iou",

                "external_test_used":
                    False,

                "winner_checkpoint_policy":
                    "best_model_pth_from_this_trial",
            },
            trial_folder
            / "trial_definition.json",
        )

        print(
            "\n"
            + "-" * 78
        )
        print(
            f"MASTER OPTUNA TRIAL {trial.number}"
        )
        print(
            "-" * 78
        )

        for key, value in params.items():
            print(
                f"{key}: {value}"
            )

        print(
            f"Epoch Budget: {int(trial_epochs)}"
        )
        print(
            "Checkpoint Policy: save BEST Validation-IoU epoch"
        )
        print(
            "External Test Used: NO"
        )
        print(
            "-" * 78
        )

        try:
            summary = train_configuration(
                base_config=base_config,
                params=params,
                general_params=general_params,
                datasets_config=datasets_config,
                split_cache_root=split_cache_root,
                output_folder=trial_folder,
                epochs=int(
                    trial_epochs
                ),
                experiment_id=trial_experiment_id,
                final_training=False,
            )

            value = best_validation_iou(
                summary
            )

            checkpoint = (
                trial_folder
                / "best_model.pth"
            )

            if not checkpoint.exists():
                raise RuntimeError(
                    f"Trial {trial.number} finished without "
                    f"best_model.pth: {checkpoint}"
                )

            trial.set_user_attr(
                "best_validation_iou_percent",
                value
                * 100.0,
            )

            trial.set_user_attr(
                "best_epoch",
                int(
                    summary.get(
                        "best_epoch",
                        -1,
                    )
                ),
            )

            trial.set_user_attr(
                "checkpoint_path",
                str(
                    checkpoint
                ),
            )

            trial.set_user_attr(
                "external_test_used",
                False,
            )

            trial.set_user_attr(
                "training_engine",
                summary.get(
                    "training_engine",
                    model_type,
                ),
            )

            return value

        except torch.cuda.OutOfMemoryError as exc:
            cleanup_cuda()

            save_json(
                {
                    "status":
                        "pruned",

                    "reason":
                        "cuda_out_of_memory",

                    "error":
                        str(
                            exc
                        ),
                },
                trial_folder
                / "trial_failure.json",
            )

            raise optuna.TrialPruned(
                "CUDA out of memory for this configuration."
            )

        except RuntimeError as exc:
            message = str(
                exc
            ).lower()

            if "out of memory" in message:
                cleanup_cuda()

                save_json(
                    {
                        "status":
                            "pruned",

                        "reason":
                            "out_of_memory",

                        "error":
                            str(
                                exc
                            ),
                    },
                    trial_folder
                    / "trial_failure.json",
                )

                raise optuna.TrialPruned(
                    "Out of memory for this configuration."
                )

            raise

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )

    complete_trials = [
        trial
        for trial
        in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    if not complete_trials:
        raise RuntimeError(
            "Master Optuna completed without a successful trial."
        )

    study.trials_dataframe().to_csv(
        master_root
        / "master_optuna_trials.csv",
        index=False,
    )

    best_trial = (
        study.best_trial
    )

    best_params = dict(
        best_trial.params
    )

    best_trial_folder = (
        trials_root
        / f"trial_{best_trial.number:04d}"
    )

    winner_summary = load_json(
        best_trial_folder
        / "training_summary.json"
    )

    winner_best_epoch = int(
        winner_summary.get(
            "best_epoch",
            -1,
        )
    )

    best_trial_payload = {
        "study_name":
            study_name,

        "experiment_id":
            experiment_id,

        "model_type":
            model_type,

        "source_type":
            source,

        "best_trial":
            int(
                best_trial.number
            ),

        "best_epoch":
            winner_best_epoch,

        "best_validation_iou":
            float(
                best_trial.value
            ),

        "best_validation_iou_percent":
            float(
                best_trial.value
                * 100.0
            ),

        "best_params":
            json_safe(
                best_params
            ),

        "best_configuration":
            json_safe(
                best_params
            ),

        "fixed_split":
            split_info,

        "trial_epochs":
            int(
                trial_epochs
            ),

        "retrained_after_selection":
            False,

        "external_test_used":
            False,
    }

    save_json(
        best_trial_payload,
        master_root
        / "master_optuna_best_trial.json",
    )

    (
        final_config,
        promoted_summary,
    ) = promote_winning_trial(
        base_config=base_config,
        best_params=best_params,
        best_trial_number=int(
            best_trial.number
        ),
        best_trial_value=float(
            best_trial.value
        ),
        trial_epochs=int(
            trial_epochs
        ),
        trials_root=trials_root,
        experiment_folder=experiment_folder,
        general_params=general_params,
        datasets_config=datasets_config,
    )

    selection = {
        **best_trial_payload,

        "dataset_type":
            final_config[
                "dataset_type"
            ],

        "selected_backbone":
            final_config.get(
                "backbone"
            ),

        "selected_tile_size":
            int(
                final_config[
                    "tile_size"
                ]
            ),

        "selected_batch_size":
            best_params.get(
                "batch_size"
            ),

        "selected_augmentation":
            bool(
                final_config.get(
                    "use_augmentation",
                    False,
                )
            ),

        "selected_lora_rank":
            (
                best_params.get(
                    "lora_rank"
                )
                if model_type
                == "sam_lora"
                else None
            ),

        "selected_lora_alpha":
            (
                best_params.get(
                    "lora_alpha"
                )
                if model_type
                == "sam_lora"
                else None
            ),

        "sam_lora_selected_config":
            (
                promoted_summary.get(
                    "sam_lora_selected_config"
                )
                if model_type
                == "sam_lora"
                else None
            ),

        "best_model_path":
            str(
                experiment_folder
                / "best_model.pth"
            ),

        "selection_dataset":
            "validation_manifest",

        "winner_policy":
            "exact_best_trial_checkpoint",

        "second_training":
            False,

        "external_test_used":
            False,

        "next_stage":
            "validation_threshold_or_postprocessing_then_final_test",
    }

    save_json(
        selection,
        experiment_folder
        / "master_optuna_selection_summary.json",
    )

    save_json(
        selection,
        master_root
        / "master_optuna_selection_summary.json",
    )

    # --------------------------------------------------------
    # Remove large trial checkpoints only after promotion.
    # --------------------------------------------------------
    if not keep_trial_checkpoints:
        promoted_path = (
            experiment_folder
            / "best_model.pth"
        ).resolve()

        if not promoted_path.exists():
            raise RuntimeError(
                "Promoted Master Optuna winner checkpoint is missing."
            )

        for trial_item in complete_trials:
            checkpoint = (
                trials_root
                / f"trial_{trial_item.number:04d}"
                / "best_model.pth"
            )

            if checkpoint.exists():
                checkpoint.unlink()

    print(
        "\n"
        + "=" * 78
    )
    print(
        "MASTER OPTUNA FINISHED — EXACT WINNER PROMOTED"
    )
    print(
        "=" * 78
    )
    print(
        f"Experiment ID: {experiment_id}"
    )
    print(
        f"Model: {model_type}"
    )
    print(
        f"Source: {source}"
    )
    print(
        f"Best Trial: {best_trial.number}"
    )
    print(
        f"Best Epoch in Winning Trial: {winner_best_epoch}"
    )
    print(
        "BEST VALIDATION IoU:",
        f"{best_trial.value * 100.0:.2f}%",
    )
    print(
        "Selected Backbone:",
        final_config.get(
            "backbone"
        ),
    )
    print(
        "Selected Tile Size:",
        final_config.get(
            "tile_size"
        ),
    )
    print(
        "Selected Batch Size:",
        best_params.get(
            "batch_size"
        ),
    )
    print(
        "Selected Augmentation:",
        (
            "ON"
            if final_config.get(
                "use_augmentation"
            )
            else "OFF"
        ),
    )

    if model_type == "sam_lora":
        print(
            "Selected LoRA Rank:",
            best_params.get(
                "lora_rank"
            ),
        )
        print(
            "Selected LoRA Alpha:",
            best_params.get(
                "lora_alpha"
            ),
        )
        print(
            "Selected SAM-LoRA Config:",
            promoted_summary.get(
                "sam_lora_selected_config"
            ),
        )

    print(
        "Winning Model:",
        experiment_folder
        / "best_model.pth",
    )
    print(
        "Second Training of Winner: NO"
    )
    print(
        "External Test Accessed: NO"
    )
    print(
        "Next: validation threshold/post-processing, then Final Test"
    )
    print(
        "=" * 78
        + "\n"
    )

    return selection

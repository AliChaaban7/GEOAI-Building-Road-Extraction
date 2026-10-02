"""
Focused Optuna optimization for Approach 2 - Road Extraction.

Purpose
-------
General Optuna tunes a SMALL approved hyperparameter set for ONE fixed
Road model and ONE fixed tile dataset.

Searchable here
---------------
- learning_rate
- weight_decay
- bce_weight
- augmentation

NOT searchable here
-------------------
- model family
- tile size
- SAM backbone
- SAM-LoRA rank
- SAM-LoRA alpha
- threshold
- post-processing
- final-test settings
- cross-validation

Those broader choices belong to Master Optuna or remain fixed.

Scientific workflow
-------------------
TRAIN / VALIDATION only:

    trial
      ↓
    train for trial_epochs
      ↓
    report CURRENT validation IoU each epoch
      ↓
    Optuna pruning allowed
      ↓
    choose best trial by Validation IoU
      ↓
    discard trial model
      ↓
    build FRESH model
      ↓
    train winner for final_epochs
      ↓
    best_model.pth

The independent Aerial/Satellite/Drone final tests are NEVER opened here.

Configuration policy
--------------------
Search ranges/options come ONLY from:

    Roads/config/optional/optuna.json

There are NO invented numeric search-range fallbacks.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch


from src.checkpointing import (
    save_training_checkpoint,
)

from src.models import (
    build_pytorch_model,
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
# ROOT
# =====================================================================

ROAD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# =====================================================================
# OPTUNA IMPORT
# =====================================================================

def _get_optuna():

    try:

        import optuna

    except ImportError as exc:

        raise ImportError(
            "Optuna is required for Road hyperparameter optimization."
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
            tuple,
            list,
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


def load_json(
    path: str | Path,
) -> Dict[str, Any]:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"JSON configuration not found:\n{path}"
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


# =====================================================================
# OPTUNA CONFIG
# =====================================================================

def load_optuna_config(
    path: Optional[
        str | Path
    ] = None,
) -> Dict[str, Any]:
    """
    Load Road Optuna configuration.

    Supports either:

        {...}

    or:

        {
            "optuna": {...}
        }
    """

    if path is None:

        path = (
            ROAD_ROOT
            / "config"
            / "optional"
            / "optuna.json"
        )

    config = load_json(
        path
    )

    if (
        "optuna"
        in config
        and isinstance(
            config[
                "optuna"
            ],
            dict,
        )
    ):

        config = dict(
            config[
                "optuna"
            ]
        )

    return config


# =====================================================================
# BASIC CONFIG HELPERS
# =====================================================================

def _first_not_none(
    *values,
):

    for value in values:

        if value is not None:

            return value

    return None


def _as_dict(
    value,
):

    return (
        dict(
            value
        )
        if isinstance(
            value,
            dict,
        )
        else {}
    )


# =====================================================================
# STUDY SETTINGS
# =====================================================================

def resolve_optuna_settings(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve study mechanics.

    Unlike SEARCH RANGES, structural defaults are allowed here because
    they do not create hyperparameter values.
    """

    study = _as_dict(
        config.get(
            "study"
        )
    )

    pruning = _as_dict(
        config.get(
            "pruning",
            config.get(
                "pruner",
                {},
            ),
        )
    )

    sampler = _as_dict(
        config.get(
            "sampler"
        )
    )

    trials = int(
        _first_not_none(
            config.get(
                "trials"
            ),

            config.get(
                "n_trials"
            ),

            study.get(
                "trials"
            ),

            study.get(
                "n_trials"
            ),

            10,
        )
    )

    trial_epochs = int(
        _first_not_none(
            config.get(
                "trial_epochs"
            ),

            study.get(
                "trial_epochs"
            ),

            12,
        )
    )

    final_epochs = int(
        _first_not_none(
            config.get(
                "final_epochs"
            ),

            study.get(
                "final_epochs"
            ),

            40,
        )
    )

    train_final = bool(
        _first_not_none(
            config.get(
                "train_final"
            ),

            study.get(
                "train_final"
            ),

            True,
        )
    )

    seed = int(
        _first_not_none(
            config.get(
                "seed"
            ),

            study.get(
                "seed"
            ),

            sampler.get(
                "seed"
            ),

            42,
        )
    )

    objective_metric = str(
        _first_not_none(
            config.get(
                "objective_metric"
            ),

            study.get(
                "objective_metric"
            ),

            "validation_iou",
        )
    ).strip().lower()

    direction = str(
        _first_not_none(
            config.get(
                "direction"
            ),

            study.get(
                "direction"
            ),

            "maximize",
        )
    ).strip().lower()

    if direction not in {
        "maximize",
        "minimize",
    }:

        raise ValueError(
            "Optuna direction must be maximize or minimize."
        )

    if objective_metric not in {
        "validation_iou",
        "val_iou",
        "iou",
    }:

        raise ValueError(
            "Current Road Optuna objective must be Validation IoU."
        )

    if trials <= 0:

        raise ValueError(
            "Optuna trials must be > 0."
        )

    if trial_epochs <= 0:

        raise ValueError(
            "Optuna trial_epochs must be > 0."
        )

    if final_epochs <= 0:

        raise ValueError(
            "Optuna final_epochs must be > 0."
        )

    pruning_enabled = bool(
        _first_not_none(
            pruning.get(
                "enabled"
            ),

            True,
        )
    )

    startup_trials = int(
        _first_not_none(
            pruning.get(
                "startup_trials"
            ),

            pruning.get(
                "n_startup_trials"
            ),

            3,
        )
    )

    warmup_epochs = int(
        _first_not_none(
            pruning.get(
                "warmup_epochs"
            ),

            pruning.get(
                "n_warmup_steps"
            ),

            4,
        )
    )

    return {
        "trials":
            trials,

        "trial_epochs":
            trial_epochs,

        "final_epochs":
            final_epochs,

        "train_final":
            train_final,

        "seed":
            seed,

        "objective_metric":
            "validation_iou",

        "direction":
            direction,

        "pruning_enabled":
            pruning_enabled,

        "startup_trials":
            startup_trials,

        "warmup_epochs":
            warmup_epochs,
    }


# =====================================================================
# SEARCH-SPACE PARSING
# =====================================================================

_ALLOWED_SEARCH_KEYS = {
    "learning_rate",
    "weight_decay",
    "bce_weight",
    "augmentation",
}


def _search_space_block(
    config: Dict[str, Any],
) -> Dict[str, Any]:

    search_space = config.get(
        "search_space"
    )

    if isinstance(
        search_space,
        dict,
    ):

        return dict(
            search_space
        )

    parameters = config.get(
        "parameters"
    )

    if isinstance(
        parameters,
        dict,
    ):

        return dict(
            parameters
        )

    return {}


def _numeric_parameter_from_config(
    config: Dict[str, Any],
    name: str,
    aliases=(),
) -> Dict[str, Any]:
    """
    Resolve an explicitly configured numeric search parameter.

    Supported JSON forms
    --------------------

    Nested:

        "search_space": {
            "learning_rate": {
                "low":  ...,
                "high": ...,
                "log": true
            }
        }

    Array:

        "learning_rate": [LOW, HIGH]

    Flat:

        "learning_rate_min": ...
        "learning_rate_max": ...
        "learning_rate_log": true

    No numeric values are invented.
    """

    search = _search_space_block(
        config
    )

    candidate_names = (
        name,
        *aliases,
    )

    specification = None

    resolved_name = None

    for candidate in candidate_names:

        if candidate in search:

            specification = search[
                candidate
            ]

            resolved_name = candidate

            break

    # --------------------------------------------------------------
    # Nested / array representation
    # --------------------------------------------------------------

    if specification is not None:

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

            raise ValueError(
                f"Invalid Optuna specification for '{resolved_name}'."
            )

    else:

        # ----------------------------------------------------------
        # Flat representation
        # ----------------------------------------------------------

        low = None
        high = None
        log = False
        step = None

        for candidate in candidate_names:

            candidate_low = _first_not_none(
                config.get(
                    f"{candidate}_min"
                ),

                config.get(
                    f"{candidate}_low"
                ),
            )

            candidate_high = _first_not_none(
                config.get(
                    f"{candidate}_max"
                ),

                config.get(
                    f"{candidate}_high"
                ),
            )

            if (
                candidate_low
                is not None
                or candidate_high
                is not None
            ):

                low = candidate_low
                high = candidate_high

                log = bool(
                    config.get(
                        f"{candidate}_log",
                        False,
                    )
                )

                step = config.get(
                    f"{candidate}_step"
                )

                resolved_name = candidate

                break

    if low is None or high is None:

        raise ValueError(
            f"\nOptuna search range for '{name}' is missing.\n\n"
            "The range must be explicitly configured in "
            "Roads/config/optional/optuna.json.\n"
            "No fallback range will be invented."
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
            f"{name}: logarithmic search requires low > 0."
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
                f"{name}: Optuna cannot use both log=True and step."
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


def _augmentation_parameter_from_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:

    search = _search_space_block(
        config
    )

    specification = search.get(
        "augmentation"
    )

    if specification is None:

        specification = _first_not_none(
            config.get(
                "augmentation_values"
            ),

            config.get(
                "augmentation_choices"
            ),
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

    else:

        values = specification

    if values is None:

        raise ValueError(
            "\nOptuna augmentation choices are missing.\n\n"
            "Explicitly configure them in optuna.json, for example "
            "as a 'values' or 'choices' array.\n"
            "No augmentation choices will be invented."
        )

    if not isinstance(
        values,
        list,
    ):

        raise TypeError(
            "Optuna augmentation choices must be a JSON list."
        )

    normalized = []

    for value in values:

        if isinstance(
            value,
            bool,
        ):

            normalized.append(
                value
            )

        elif isinstance(
            value,
            str,
        ):

            lowered = (
                value.strip()
                .lower()
            )

            if lowered in {
                "true",
                "on",
                "yes",
                "1",
            }:

                normalized.append(
                    True
                )

            elif lowered in {
                "false",
                "off",
                "no",
                "0",
            }:

                normalized.append(
                    False
                )

            else:

                raise ValueError(
                    "augmentation choices must resolve to booleans."
                )

        else:

            raise TypeError(
                "augmentation choices must be booleans."
            )

    normalized = list(
        dict.fromkeys(
            normalized
        )
    )

    if not normalized:

        raise ValueError(
            "augmentation choices cannot be empty."
        )

    return {
        "values":
            normalized
    }


def resolve_search_space(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve ONLY the approved focused Optuna dimensions.
    """

    search = _search_space_block(
        config
    )

    unknown = (
        set(
            search.keys()
        )
        - _ALLOWED_SEARCH_KEYS
    )

    if unknown:

        raise ValueError(
            "General Road Optuna contains unsupported search dimensions:\n"
            f"{sorted(unknown)}\n\n"
            "General Optuna may search only:\n"
            "- learning_rate\n"
            "- weight_decay\n"
            "- bce_weight\n"
            "- augmentation\n\n"
            "Tile size belongs to Master Optuna."
        )

    learning_rate = (
        _numeric_parameter_from_config(
            config,
            "learning_rate",
            aliases=(
                "lr",
            ),
        )
    )

    weight_decay = (
        _numeric_parameter_from_config(
            config,
            "weight_decay",
            aliases=(
                "wd",
            ),
        )
    )

    bce_weight = (
        _numeric_parameter_from_config(
            config,
            "bce_weight",
        )
    )

    if (
        bce_weight[
            "low"
        ] < 0.0
        or bce_weight[
            "high"
        ] > 1.0
    ):

        raise ValueError(
            "bce_weight search must stay within [0,1]."
        )

    augmentation = (
        _augmentation_parameter_from_config(
            config
        )
    )

    return {
        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "bce_weight":
            bce_weight,

        "augmentation":
            augmentation,
    }


# =====================================================================
# OPTUNA SAMPLING
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


def sample_trial_parameters(
    trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:

    learning_rate = _suggest_float(
        trial,
        "learning_rate",
        search_space[
            "learning_rate"
        ],
    )

    weight_decay = _suggest_float(
        trial,
        "weight_decay",
        search_space[
            "weight_decay"
        ],
    )

    bce_weight = _suggest_float(
        trial,
        "bce_weight",
        search_space[
            "bce_weight"
        ],
    )

    augmentation = trial.suggest_categorical(
        "augmentation",
        search_space[
            "augmentation"
        ][
            "values"
        ],
    )

    return {
        "learning_rate":
            float(
                learning_rate
            ),

        "weight_decay":
            float(
                weight_decay
            ),

        "bce_weight":
            float(
                bce_weight
            ),

        "dice_weight":
            float(
                1.0
                - float(
                    bce_weight
                )
            ),

        "augmentation":
            bool(
                augmentation
            ),
    }


# =====================================================================
# TRIAL CONTEXT
# =====================================================================

def apply_optuna_parameters(
    base_context: Dict[str, Any],
    params: Dict[str, Any],
    epochs: int,
    output_folder: str | Path,
    checkpoint_folder: str | Path,
) -> Dict[str, Any]:
    """
    Apply one trial without mutating the base experiment.
    """

    context = copy.deepcopy(
        base_context
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
    # General training config.
    # --------------------------------------------------------------

    general = context.setdefault(
        "general_params",
        {}
    )

    general_training = general.setdefault(
        "training",
        {}
    )

    general_training[
        "epochs"
    ] = int(
        epochs
    )

    general_training[
        "learning_rate"
    ] = float(
        params[
            "learning_rate"
        ]
    )

    general_training[
        "weight_decay"
    ] = float(
        params[
            "weight_decay"
        ]
    )

    # --------------------------------------------------------------
    # Model training config.
    #
    # resolve_training_config() gives model-specific settings priority,
    # therefore these must also be overridden.
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
    # BCE + Dice.
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
    # Experiment switch.
    # --------------------------------------------------------------

    experiment = context.setdefault(
        "experiment_config",
        {}
    )

    experiment[
        "use_augmentation"
    ] = bool(
        params[
            "augmentation"
        ]
    )

    experiment[
        "use_optuna"
    ] = True

    experiment[
        "use_master_optuna"
    ] = False

    # These stages remain separate.
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

def build_trial_augmentation(
    enabled: bool,
):

    if not enabled:

        return None

    config_path = (
        ROAD_ROOT
        / "config"
        / "optional"
        / "augmentation.json"
    )

    config = load_json(
        config_path
    )

    from src.optional.augmentation import (
        build_training_augmentation,
    )

    augmentation = (
        build_training_augmentation(
            config
        )
    )

    if augmentation is None:

        raise RuntimeError(
            "Optuna selected augmentation=True but the Road "
            "augmentation pipeline returned None."
        )

    return augmentation


# =====================================================================
# SCHEDULER STEP
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
# ONE OPTUNA TRIAL
# =====================================================================

def run_optuna_trial(
    trial,
    base_context: Dict[str, Any],
    params: Dict[str, Any],
    settings: Dict[str, Any],
    study_root: Path,
    device=None,
) -> Dict[str, Any]:
    """
    Train one candidate and allow epoch-level pruning.
    """

    optuna = _get_optuna()

    trial_number = int(
        trial.number
    )

    trial_root = (
        study_root
        / "trials"
        / f"trial_{trial_number:03d}"
    )

    output_folder = (
        trial_root
        / "output"
    )

    checkpoint_folder = (
        trial_root
        / "checkpoints"
    )

    context = apply_optuna_parameters(
        base_context=base_context,
        params=params,
        epochs=settings[
            "trial_epochs"
        ],
        output_folder=output_folder,
        checkpoint_folder=checkpoint_folder,
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

    augmentation = build_trial_augmentation(
        params[
            "augmentation"
        ]
    )

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

    training_config = resolve_training_config(
        general_params=general_params,
        model_config=model_config,
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

    best_iou = float(
        "-inf"
    )

    best_epoch = None

    best_metrics = None

    best_checkpoint = (
        checkpoint_folder
        / "best_model.pth"
    )

    history = []

    try:

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

            validation_result = validate_one_epoch(
                model=model,
                loader=data[
                    "validation_loader"
                ],
                criterion=criterion,
                device=resolved_device,
                threshold=validation_threshold,
                use_amp=use_amp,
                calculate_cldice=False,
            )

            current_iou = float(
                validation_result[
                    "iou"
                ]
            )

            # IMPORTANT:
            # Report CURRENT epoch value, not best-so-far.
            trial.report(
                current_iou,
                step=epoch,
            )

            is_best = (
                current_iou
                > best_iou
            )

            if is_best:

                best_iou = current_iou

                best_epoch = int(
                    epoch
                )

                best_metrics = copy.deepcopy(
                    validation_result
                )

                save_training_checkpoint(
                    checkpoint_path=best_checkpoint,
                    model=model,
                    context=context,
                    epoch=epoch,
                    validation_metrics=validation_result,
                    train_metrics=train_result,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    extra={
                        "optuna_trial":
                            trial_number,

                        "optuna_parameters":
                            copy.deepcopy(
                                params
                            ),

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

                    "best_so_far":
                        bool(
                            is_best
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

                        "state":
                            "PRUNED",

                        "pruned_after_epoch":
                            epoch,

                        "current_validation_iou":
                            current_iou,

                        "best_validation_iou":
                            (
                                None
                                if best_epoch
                                is None
                                else best_iou
                            ),

                        "parameters":
                            params,
                    },
                    output_folder
                    / "trial_summary.json",
                )

                raise optuna.TrialPruned(
                    f"Trial {trial_number} pruned after epoch {epoch}."
                )

        if best_epoch is None:

            raise RuntimeError(
                "Optuna trial completed without a valid Validation IoU."
            )

        summary = {
            "trial":
                trial_number,

            "state":
                "COMPLETE",

            "parameters":
                params,

            "best_epoch":
                best_epoch,

            "best_validation_iou":
                best_iou,

            "best_validation_metrics":
                best_metrics,

            "checkpoint":
                str(
                    best_checkpoint
                ),

            "independent_test_used":
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
                    trial_number,

                "state":
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
            f"CUDA OOM in trial {trial_number}."
        )

    except RuntimeError as exc:

        if "out of memory" in str(
            exc
        ).lower():

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            raise optuna.TrialPruned(
                f"OOM in trial {trial_number}."
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
# TRIAL TABLE
# =====================================================================

def save_study_trials_csv(
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
            key
            for trial
            in study.trials
            for key
            in trial.params.keys()
        }
    )

    fieldnames = [
        "trial",
        "state",
        "value",
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
                    str(
                        trial.state.name
                    ),

                "value":
                    trial.value,
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
# FINAL FRESH WINNER
# =====================================================================

def train_fresh_optuna_winner(
    base_context: Dict[str, Any],
    best_params: Dict[str, Any],
    final_epochs: int,
    device=None,
) -> Dict[str, Any]:
    """
    Fresh full training of the winning hyperparameters.

    Trial weights are NEVER promoted directly.
    """

    output_folder = Path(
        base_context[
            "output_folder"
        ]
    )

    checkpoint_folder = Path(
        base_context[
            "checkpoint_folder"
        ]
    )

    final_context = apply_optuna_parameters(
        base_context=base_context,
        params=best_params,
        epochs=int(
            final_epochs
        ),
        output_folder=output_folder,
        checkpoint_folder=checkpoint_folder,
    )

    augmentation = build_trial_augmentation(
        best_params[
            "augmentation"
        ]
    )

    # Fresh architecture / fresh weights.
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

    # Preserve winning hyperparameters separately.
    save_json(
        {
            "parameters":
                best_params,

            "final_epochs":
                int(
                    final_epochs
                ),

            "fresh_retraining":
                True,

            "trial_checkpoint_promoted":
                False,
        },
        output_folder
        / "resolved_optuna_winner.json",
    )

    return result


# =====================================================================
# MAIN OPTUNA WORKFLOW
# =====================================================================

def run_optuna_search(
    context: Dict[str, Any],
    config_path: Optional[
        str | Path
    ] = None,
    device=None,
) -> Dict[str, Any]:
    """
    Execute focused Road Optuna + optional fresh full winner training.
    """

    optuna = _get_optuna()

    if context.get(
        "model_backend"
    ) != "pytorch":

        raise RuntimeError(
            "General Road Optuna currently supports native PyTorch "
            "models only.\n"
            "ConnectNet and MultiTaskRoadExtractor use ArcGIS Learn."
        )

    experiment = context[
        "experiment_config"
    ]

    if bool(
        experiment.get(
            "use_master_optuna",
            False,
        )
    ):

        raise RuntimeError(
            "General Optuna cannot run while use_master_optuna=True."
        )

    config = load_optuna_config(
        config_path
    )

    settings = resolve_optuna_settings(
        config
    )

    search_space = resolve_search_space(
        config
    )

    study_root = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "optuna"
    )

    study_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        config,
        study_root
        / "optuna_config_used.json",
    )

    save_json(
        search_space,
        study_root
        / "resolved_search_space.json",
    )

    # --------------------------------------------------------------
    # Study
    # --------------------------------------------------------------

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

    study_name = (
        f"{experiment['experiment_id']}_optuna"
    )

    study = optuna.create_study(
        study_name=study_name,

        direction=(
            settings[
                "direction"
            ]
        ),

        sampler=sampler,

        pruner=pruner,
    )

    # --------------------------------------------------------------
    # Objective
    # --------------------------------------------------------------

    def objective(
        trial,
    ):

        params = sample_trial_parameters(
            trial,
            search_space,
        )

        trial.set_user_attr(
            "dice_weight",
            params[
                "dice_weight"
            ],
        )

        result = run_optuna_trial(
            trial=trial,

            base_context=context,

            params=params,

            settings=settings,

            study_root=study_root,

            device=device,
        )

        return float(
            result[
                "best_validation_iou"
            ]
        )

    study.optimize(
        objective,

        n_trials=int(
            settings[
                "trials"
            ]
        ),

        gc_after_trial=True,

        show_progress_bar=False,
    )

    # --------------------------------------------------------------
    # Ensure at least one successful trial.
    # --------------------------------------------------------------

    completed = [
        trial
        for trial
        in study.trials
        if trial.state
        == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:

        raise RuntimeError(
            "Optuna completed without any successful trials."
        )

    best_trial = study.best_trial

    best_params = {
        **best_trial.params,
    }

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

    # --------------------------------------------------------------
    # Save study before final training.
    # --------------------------------------------------------------

    trials_csv = save_study_trials_csv(
        study,
        study_root
        / "trials.csv",
    )

    preliminary_summary = {
        "study_name":
            study_name,

        "model_type":
            context.get(
                "model_type"
            ),

        "objective":
            "validation_iou",

        "direction":
            settings[
                "direction"
            ],

        "number_of_requested_trials":
            settings[
                "trials"
            ],

        "number_of_completed_trials":
            len(
                completed
            ),

        "best_trial_number":
            int(
                best_trial.number
            ),

        "best_value":
            float(
                best_trial.value
            ),

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
            best_params,

        "trial_epochs":
            settings[
                "trial_epochs"
            ],

        "final_epochs":
            settings[
                "final_epochs"
            ],

        "search_space":
            search_space,

        "trials_csv":
            str(
                trials_csv
            ),

        "threshold_search_inside_optuna":
            False,

        "postprocessing_search_inside_optuna":
            False,

        "final_test_used":
            False,
    }

    save_json(
        preliminary_summary,
        study_root
        / "study_summary.json",
    )

    # --------------------------------------------------------------
    # Fresh winning retrain
    # --------------------------------------------------------------

    final_training_result = None

    if settings[
        "train_final"
    ]:

        final_training_result = (
            train_fresh_optuna_winner(
                base_context=context,

                best_params=best_params,

                final_epochs=settings[
                    "final_epochs"
                ],

                device=device,
            )
        )

    # --------------------------------------------------------------
    # Canonical summary
    # --------------------------------------------------------------

    selection_summary = {
        **preliminary_summary,

        "train_final":
            bool(
                settings[
                    "train_final"
                ]
            ),

        "fresh_winner_retrained":
            bool(
                settings[
                    "train_final"
                ]
            ),

        "trial_checkpoint_promoted":
            False,

        "final_training_best_epoch":
            (
                final_training_result.get(
                    "best_epoch"
                )
                if final_training_result
                else None
            ),

        "final_training_best_validation_iou":
            (
                final_training_result.get(
                    "best_validation_iou"
                )
                if final_training_result
                else None
            ),

        "final_checkpoint":
            (
                final_training_result.get(
                    "best_checkpoint"
                )
                if final_training_result
                else None
            ),
    }

    canonical_summary = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "optuna_selection_summary.json"
    )

    save_json(
        selection_summary,
        canonical_summary,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "ROAD OPTUNA COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        f"Best trial        : "
        f"{best_trial.number}"
    )

    print(
        f"Trial Val IoU     : "
        f"{best_trial.value * 100:.4f}%"
    )

    print(
        "Best parameters:"
    )

    for (
        key,
        value,
    ) in best_params.items():

        print(
            f"  {key:<18}: {value}"
        )

    print(
        f"Fresh retraining  : "
        f"{'YES' if settings['train_final'] else 'NO'}"
    )

    if final_training_result:

        print(
            f"Final best IoU    : "
            f"{final_training_result['best_validation_iou'] * 100:.4f}%"
        )

        print(
            f"Final checkpoint  : "
            f"{final_training_result['best_checkpoint']}"
        )

    print(
        "Threshold search  : NOT RUN"
    )

    print(
        "Post-processing   : NOT RUN"
    )

    print(
        "Final test        : NOT ACCESSED"
    )

    print(
        "=" * 76
    )

    return selection_summary


# Compatibility aliases.
run_optuna = run_optuna_search
run_full_optuna_workflow = run_optuna_search
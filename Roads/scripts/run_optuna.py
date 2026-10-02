"""
Run focused Optuna for Approach 2 - Road Extraction.

Searches ONLY:
- learning rate
- weight decay
- BCE weight
- augmentation

The model and tile size remain fixed.

General Optuna control
----------------------
The UI / CLI may control:
- number of trials
- epochs per trial
- whether to train a fresh final optimized model
- final optimized-model epochs

Priority:
    CLI / UI values
        ↓
    experiment training_schedule
        ↓
    config/optional/optuna.json
        ↓
    Optuna engine defaults

After the study:
    best hyperparameters
        ↓
    FRESH full training
        ↓
    best_model.pth

No threshold search, post-processing search, or final test occurs here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


# =====================================================================
# PATHS
# =====================================================================

ROAD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    ROAD_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            ROAD_ROOT
        ),
    )


# =====================================================================
# ROAD IMPORTS
# =====================================================================

from src.optional.optuna_search import (
    run_optuna_search,
)

from src.pipeline import (
    prepare_experiment,
    print_experiment_summary,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Run focused Optuna for Road extraction."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional Road experiment.json path."
        ),
    )

    parser.add_argument(
        "--optuna-config",
        type=str,
        default=None,
        help=(
            "Optional optuna.json override."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Example: cuda:0 or cpu."
        ),
    )

    parser.add_argument(
        "--regenerate-manifests",
        action="store_true",
    )

    # -----------------------------------------------------------------
    # General Optuna controls exposed by the shared UI.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Number of General Optuna trials. "
            "Overrides experiment/optuna JSON when supplied."
        ),
    )

    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=None,
        help=(
            "Training epochs per Optuna trial. "
            "Overrides experiment/optuna JSON when supplied."
        ),
    )

    parser.add_argument(
        "--train-final",
        action="store_true",
        help=(
            "After hyperparameter selection, train a fresh final "
            "optimized model."
        ),
    )

    parser.add_argument(
        "--final-epochs",
        type=int,
        default=None,
        help=(
            "Epochs for the fresh final optimized model."
        ),
    )

    return parser.parse_args()


# =====================================================================
# JSON HELPERS
# =====================================================================

def load_json(
    path: Path,
) -> dict:

    path = Path(
        path
    )

    if not path.exists():

        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise TypeError(
            "Optuna configuration must be a JSON object:\n"
            f"{path}"
        )

    return data


def save_json(
    data: dict,
    path: Path,
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
            data,
            file,
            indent=2,
        )

    return path


# =====================================================================
# VALIDATION
# =====================================================================

def validate_positive_integer(
    value,
    name: str,
):

    if value is None:
        return None

    value = int(
        value
    )

    if value <= 0:

        raise ValueError(
            f"{name} must be > 0."
        )

    return value


# =====================================================================
# OPTUNA RUNTIME CONFIG
# =====================================================================

def default_optuna_config_path() -> Path:

    return (
        ROAD_ROOT
        / "config"
        / "optional"
        / "optuna.json"
    )


def resolve_source_optuna_config_path(
    requested_path,
) -> Path:

    if requested_path is None:

        return (
            default_optuna_config_path()
        )

    return Path(
        requested_path
    ).expanduser().resolve()


def _settings_dict(
    config: dict,
) -> dict:
    """
    Preserve the current Road optuna.json structure.

    Supported:
        flat:
            {
                "trials": ...,
                ...
            }

        nested:
            {
                "optuna": {
                    "trials": ...,
                    ...
                }
            }
    """

    nested = config.get(
        "optuna"
    )

    if isinstance(
        nested,
        dict,
    ):

        return nested

    return config


def build_runtime_optuna_config(
    context: dict,
    args,
) -> tuple[Path, dict]:

    experiment = context[
        "experiment_config"
    ]

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    source_path = (
        resolve_source_optuna_config_path(
            args.optuna_config
        )
    )

    if (
        args.optuna_config is not None
        and not source_path.exists()
    ):

        raise FileNotFoundError(
            "Requested Optuna config does not exist:\n"
            f"{source_path}"
        )

    source_config = load_json(
        source_path
    )

    # Work on a copy so the permanent optuna.json is NEVER modified.
    runtime_config = copy.deepcopy(
        source_config
    )

    settings = _settings_dict(
        runtime_config
    )

    schedule = experiment.get(
        "training_schedule",
        {},
    )

    if not isinstance(
        schedule,
        dict,
    ):

        schedule = {}

    # -----------------------------------------------------------------
    # Priority:
    # CLI / UI > experiment training_schedule > optuna.json
    # -----------------------------------------------------------------

    trials = (
        args.trials
        if args.trials is not None
        else schedule.get(
            "trials"
        )
    )

    trial_epochs = (
        args.trial_epochs
        if args.trial_epochs is not None
        else schedule.get(
            "trial_epochs"
        )
    )

    final_epochs = (
        args.final_epochs
        if args.final_epochs is not None
        else schedule.get(
            "final_epochs"
        )
    )

    trials = validate_positive_integer(
        trials,
        "trials",
    )

    trial_epochs = validate_positive_integer(
        trial_epochs,
        "trial_epochs",
    )

    final_epochs = validate_positive_integer(
        final_epochs,
        "final_epochs",
    )

    if trials is not None:

        settings[
            "trials"
        ] = trials

    if trial_epochs is not None:

        settings[
            "trial_epochs"
        ] = trial_epochs

    # --train-final is an explicit request from the UI/CLI.
    # If the flag is not supplied, preserve whatever optuna.json
    # already says.
    if args.train_final:

        settings[
            "train_final"
        ] = True

    if final_epochs is not None:

        settings[
            "final_epochs"
        ] = final_epochs

    # If the source config was empty/flat, settings IS runtime_config.
    # If it was nested, _settings_dict() returned runtime_config["optuna"],
    # so the update is already reflected in runtime_config.

    runtime_path = (
        output_folder
        / "runtime_optuna_config.json"
    )

    save_json(
        runtime_config,
        runtime_path,
    )

    resolved = {
        "trials":
            settings.get(
                "trials"
            ),

        "trial_epochs":
            settings.get(
                "trial_epochs"
            ),

        "train_final":
            bool(
                settings.get(
                    "train_final",
                    False,
                )
            ),

        "final_epochs":
            settings.get(
                "final_epochs"
            ),

        "source_config":
            (
                str(
                    source_path
                )
                if source_path.exists()
                else None
            ),

        "runtime_config":
            str(
                runtime_path
            ),
    }

    return (
        runtime_path,
        resolved,
    )


# =====================================================================
# MAIN
# =====================================================================

def ensure_checkpoint_folder(
    context: dict,
) -> dict:
    """
    Normalize the Road training context for callers created before
    checkpoint_folder became an explicit key.

    General Optuna uses the experiment output folder as the default
    checkpoint folder for the fresh winner training unless a dedicated
    checkpoint folder has already been supplied.
    """

    context = copy.deepcopy(
        context
    )

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
            "Prepared Road Optuna context is missing both "
            "'output_folder' and 'checkpoint_folder'."
        )

    context[
        "checkpoint_folder"
    ] = Path(
        checkpoint_folder
    )

    return context


def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,

        force_regenerate_manifests=(
            args.regenerate_manifests
        ),
    )

    # General Optuna trials can run with their own trial folders, but the
    # fresh final-winner path receives the base experiment context.
    # Normalize that base context before entering optuna_search so
    # train_fresh_optuna_winner() always has checkpoint_folder available.
    context = ensure_checkpoint_folder(
        context
    )

    print_experiment_summary(
        context
    )

    experiment = context[
        "experiment_config"
    ]

    if not bool(
        experiment.get(
            "use_optuna",
            False,
        )
    ):

        raise RuntimeError(
            "experiment.json has use_optuna=false.\n"
            "Enable it before running General Optuna."
        )

    if bool(
        experiment.get(
            "use_master_optuna",
            False,
        )
    ):

        raise RuntimeError(
            "General Optuna and Master Optuna cannot run together."
        )

    if (
        context[
            "model_backend"
        ]
        != "pytorch"
    ):

        raise RuntimeError(
            "General Road Optuna currently supports native PyTorch "
            "models only.\n\n"
            f"Current model   : {context['model_type']}\n"
            f"Current backend : {context['model_backend']}"
        )

    # -----------------------------------------------------------------
    # Resolve UI / CLI overrides without modifying permanent JSON.
    # -----------------------------------------------------------------

    (
        runtime_optuna_config,
        resolved_optuna,
    ) = build_runtime_optuna_config(
        context=context,
        args=args,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "FOCUSED ROAD OPTUNA POLICY"
    )

    print(
        "=" * 76
    )

    print(
        f"Model             : {context['model_type']}"
    )

    print(
        f"Tile size         : "
        f"{experiment['tile_size']} (FIXED)"
    )

    print(
        "Search parameters : LR / WD / BCE / Augmentation"
    )

    print(
        "SAM rank/alpha    : FIXED"
    )

    print(
        "Threshold         : OUTSIDE OPTUNA"
    )

    print(
        "Post-processing   : OUTSIDE OPTUNA"
    )

    print(
        "Final test        : NOT ACCESSED"
    )

    print(
        "-" * 76
    )

    print(
        f"Trials            : {resolved_optuna['trials']}"
    )

    print(
        f"Epochs / Trial    : {resolved_optuna['trial_epochs']}"
    )

    print(
        f"Train Final       : {resolved_optuna['train_final']}"
    )

    print(
        f"Final Epochs      : {resolved_optuna['final_epochs']}"
    )

    print(
        f"Runtime Config    : {runtime_optuna_config}"
    )

    print(
        "=" * 76
    )

    result = run_optuna_search(
        context=context,

        config_path=(
            runtime_optuna_config
        ),

        device=args.device,
    )

    return result


if __name__ == "__main__":

    main()

"""
Standard native PyTorch Road training launcher.

Supported:
- U-Net
- DeepLabV3+
- SAM-LoRA

Not handled here:
- ConnectNet
- MultiTaskRoadExtractor

Those models use:
    Roads/scripts/run_arcgis_train.py

Workflow
--------
Mixed-source labeled data
        ↓
Train / Validation split
        ↓
optional Train augmentation
        ↓
training
        ↓
best epoch selected by Validation IoU

No external final-test raster or Ground Truth is accessed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


ROAD_ROOT = Path(__file__).resolve().parents[1]

if str(ROAD_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROAD_ROOT),
    )


from src.models import build_pytorch_model
from src.optional.augmentation import build_training_augmentation
from src.pipeline import prepare_experiment, print_experiment_summary
from src.train import train_model


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train one native PyTorch Road model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--regenerate-manifests",
        action="store_true",
    )

    return parser.parse_args()


def apply_epoch_override(
    context,
    epochs,
):

    if epochs is None:
        return context

    epochs = int(epochs)

    if epochs <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )

    context = copy.deepcopy(
        context
    )

    context.setdefault(
        "general_params",
        {},
    ).setdefault(
        "training",
        {},
    )[
        "epochs"
    ] = epochs

    context.setdefault(
        "model_config",
        {},
    ).setdefault(
        "training",
        {},
    )[
        "epochs"
    ] = epochs

    return context


def ensure_checkpoint_folder(
    context,
):
    """
    Make the launcher compatible with both Road context generations.

    Newer training code supports a dedicated checkpoint_folder.
    Older prepare_experiment() contexts may expose only output_folder.

    Baseline policy:
        when no dedicated checkpoint folder is configured,
        checkpoints live directly in the experiment output folder.
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
            "Prepared Road experiment context is missing both "
            "'output_folder' and 'checkpoint_folder'."
        )

    context[
        "checkpoint_folder"
    ] = Path(
        checkpoint_folder
    )

    return context


def build_augmentation(
    context,
):

    enabled = bool(
        context[
            "experiment_config"
        ].get(
            "use_augmentation",
            False,
        )
    )

    if not enabled:
        return None

    path = (
        ROAD_ROOT
        / "config"
        / "optional"
        / "augmentation.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Augmentation config missing:\n{path}"
        )

    import json

    config = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    augmentation = build_training_augmentation(
        config
    )

    if augmentation is None:
        raise RuntimeError(
            "use_augmentation=True but augmentation "
            "pipeline returned None."
        )

    return augmentation


def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,
        force_regenerate_manifests=(
            args.regenerate_manifests
        ),
    )

    # Normalize checkpoint location before any model/training stage.
    # This fixes contexts produced before checkpoint_folder became
    # an explicit training-engine requirement.
    context = ensure_checkpoint_folder(
        context
    )

    print_experiment_summary(
        context
    )

    if context[
        "model_backend"
    ] != "pytorch":

        raise RuntimeError(
            "\nThis launcher is for native PyTorch models only.\n\n"
            f"Model   : {context['model_type']}\n"
            f"Backend : {context['model_backend']}\n\n"
            "Use Roads/scripts/run_arcgis_train.py for "
            "ConnectNet or MultiTaskRoadExtractor."
        )

    experiment = context[
        "experiment_config"
    ]

    if bool(
        experiment.get(
            "use_optuna",
            False,
        )
    ):
        raise RuntimeError(
            "use_optuna=True. Run scripts/run_optuna.py instead."
        )

    if bool(
        experiment.get(
            "use_master_optuna",
            False,
        )
    ):
        raise RuntimeError(
            "use_master_optuna=True. "
            "Run scripts/run_master_optuna.py instead."
        )

    context = apply_epoch_override(
        context,
        args.epochs,
    )

    augmentation = build_augmentation(
        context
    )

    model = build_pytorch_model(
        context
    )

    result = train_model(
        model=model,
        context=context,
        train_augmentation=augmentation,
        validation_augmentation=None,
        device=args.device,
    )

    print()
    print(
        f"Best checkpoint: {result['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()

"""
Train the ArcGIS Learn Road models.

Supported:
    ConnectNet
    MultiTaskRoadExtractor

Recommended execution
---------------------
Because standalone Python on this computer currently cannot initialize
the ArcGIS Pro product license, run this script from an ArcGIS Pro
Notebook.

ArcGIS Pro Notebook:

    %run "D:/Ali Chaaban Thesis Project/GeoAI_Thesis_Codebase/Roads/scripts/run_arcgis_train.py"

The current experiment.json determines which ArcGIS model is trained.

No independent final-test data is accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


# ArcGIS/FastAI dependencies in this environment contain legacy string
# literals that trigger Python 3.13 SyntaxWarning messages such as:
#     invalid escape sequence '\\s'
# This warning is external to the thesis code and does not affect training.
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence.*",
    category=SyntaxWarning,
)


ROAD_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
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


from src.pipeline import (
    prepare_experiment,
    print_experiment_summary,
)

from src.train_arcgis import (
    train_arcgis_road_model,
)


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Train ConnectNet or MultiTaskRoadExtractor "
            "using ArcGIS Learn."
        )
    )

    parser.add_argument(
        "--config",

        type=str,

        default=None,

        help=(
            "Optional Road experiment JSON path."
        ),
    )

    # parse_known_args is deliberate:
    # ArcGIS Pro Notebooks may inject kernel arguments.
    args, _ = parser.parse_known_args()

    return args


def _load_json(
    path: Path,
) -> dict:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"JSON configuration not found:\n{path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):

        raise TypeError(
            f"Expected a JSON object in:\n{path}"
        )

    return payload


def normalize_arcgis_training_context(
    context: dict,
    config_path: str | None,
) -> dict:
    """
    Complete the context required by src/train_arcgis.py.

    prepare_experiment() remains authoritative for dataset resolution,
    manifests, backend selection, model registry metadata and output paths.

    This compatibility layer only restores context keys required by the
    ArcGIS Learn trainer when an older/current pipeline package omits them.
    """

    if not isinstance(
        context,
        dict,
    ):

        raise TypeError(
            "prepare_experiment() must return a dictionary context."
        )

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    experiment_config = context.get(
        "experiment_config"
    )

    if not isinstance(
        experiment_config,
        dict,
    ):

        resolved_config_path = (
            Path(
                config_path
            )
            if config_path
            else (
                ROAD_ROOT
                / "config"
                / "experiment.json"
            )
        )

        experiment_config = _load_json(
            resolved_config_path
        )

        context[
            "experiment_config"
        ] = experiment_config

    # --------------------------------------------------------
    # Model type
    # --------------------------------------------------------

    model_type = (
        context.get(
            "model_type"
        )
        or experiment_config.get(
            "model_type"
        )
    )

    if model_type is None:

        raise KeyError(
            "Could not resolve ArcGIS Road model_type."
        )

    model_type = (
        str(
            model_type
        )
        .strip()
        .lower()
    )

    aliases = {
        "multitask":
            "multitask_road_extractor",

        "multi_task_road_extractor":
            "multitask_road_extractor",

        "multitaskroadextractor":
            "multitask_road_extractor",
    }

    model_type = aliases.get(
        model_type,
        model_type,
    )

    context[
        "model_type"
    ] = model_type

    experiment_config[
        "model_type"
    ] = model_type

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------
    #
    # train_arcgis.py directly requires context["model_config"] for
    # resolve_connectnet_config()/resolve_multitask_config() and again
    # when constructing the ArcGIS model.
    # --------------------------------------------------------

    model_config = context.get(
        "model_config"
    )

    if not isinstance(
        model_config,
        dict,
    ) or not model_config:

        candidates = [
            (
                ROAD_ROOT
                / "config"
                / "models"
                / f"{model_type}.json"
            ),
            (
                ROAD_ROOT
                / "config"
                / f"{model_type}.json"
            ),
        ]

        model_config_path = next(
            (
                path
                for path
                in candidates
                if path.exists()
            ),
            None,
        )

        if model_config_path is None:

            raise FileNotFoundError(
                "ArcGIS Road model configuration was not found.\n"
                f"Model: {model_type}\n"
                "Checked:\n"
                + "\n".join(
                    str(
                        path
                    )
                    for path
                    in candidates
                )
            )

        model_config = _load_json(
            model_config_path
        )

        context[
            "model_config"
        ] = model_config

        context[
            "model_config_path"
        ] = model_config_path

    # --------------------------------------------------------
    # Output/checkpoint compatibility
    # --------------------------------------------------------

    output_folder_value = context.get(
        "output_folder"
    )

    if output_folder_value is None:

        raise KeyError(
            "prepare_experiment() did not return 'output_folder'."
        )

    output_folder = Path(
        output_folder_value
    )

    context[
        "output_folder"
    ] = output_folder

    context[
        "checkpoint_folder"
    ] = Path(
        context.get(
            "checkpoint_folder"
        )
        or output_folder
    )

    # --------------------------------------------------------
    # Split compatibility
    # --------------------------------------------------------

    split_config = context.get(
        "split_config"
    )

    if not isinstance(
        split_config,
        dict,
    ):

        split_config = experiment_config.get(
            "data_split",
            {},
        )

        if not isinstance(
            split_config,
            dict,
        ):

            split_config = {}

        split_config = dict(
            split_config
        )

        split_config.setdefault(
            "train_percent",
            80,
        )

        split_config.setdefault(
            "validation_percent",
            20,
        )

        split_config.setdefault(
            "seed",
            42,
        )

        context[
            "split_config"
        ] = split_config

    else:

        split_config.setdefault(
            "train_percent",
            80,
        )

        split_config.setdefault(
            "validation_percent",
            20,
        )

        split_config.setdefault(
            "seed",
            42,
        )

    # --------------------------------------------------------
    # Required ArcGIS trainer context check
    # --------------------------------------------------------

    required = (
        "model_backend",
        "model_type",
        "model_config",
        "dataset_id",
        "dataset_root",
        "experiment_config",
        "split_config",
        "output_folder",
        "checkpoint_folder",
    )

    missing = [
        key
        for key
        in required
        if context.get(
            key
        )
        is None
    ]

    if missing:

        raise KeyError(
            "ArcGIS Road training context is incomplete.\n"
            "Missing:\n"
            + "\n".join(
                f"- {key}"
                for key
                in missing
            )
        )

    return context


def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,

        force_regenerate_manifests=False,
    )

    context = normalize_arcgis_training_context(
        context=context,
        config_path=args.config,
    )

    print_experiment_summary(
        context
    )

    if (
        context[
            "model_backend"
        ]
        != "arcgis_learn"
    ):

        raise RuntimeError(
            "run_arcgis_train.py requires an ArcGIS Learn model.\n\n"
            f"Current model : {context['model_type']}\n"
            f"Backend       : {context['model_backend']}\n\n"
            "Set experiment.json model_type to:\n"
            "    connectnet\n"
            "or\n"
            "    multitask_road_extractor"
        )

    if (
        context[
            "model_type"
        ]
        not in {
            "connectnet",
            "multitask_road_extractor",
        }
    ):

        raise RuntimeError(
            "Unsupported ArcGIS Learn Road model:\n"
            f"{context['model_type']}"
        )

    print()
    print(
        "=" * 72
    )

    print(
        "ARCGIS ROAD MODEL DATA POLICY"
    )

    print(
        "=" * 72
    )

    print(
        "Training export  : mixed aerial + satellite + drone"
    )

    print(
        "Validation       : ArcGIS deterministic split"
    )

    print(
        "SatelliteTesingZone : NOT ACCESSED"
    )

    print(
        "Road_Test_Label     : NOT ACCESSED"
    )

    print(
        "=" * 72
    )

    train_arcgis_road_model(
        context
    )


if __name__ == "__main__":

    main()

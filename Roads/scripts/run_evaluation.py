"""
Standalone strict GIS evaluation for Road predictions.

This script evaluates an already-existing prediction.

It does NOT rerun:
- training
- inference
- threshold search
- post-processing search

Examples
--------
Evaluate the automatically generated Satellite prediction:

$py = "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe"

& $py Roads\\scripts\\run_evaluation.py --source satellite


Evaluate a specific prediction feature class:

& $py Roads\\scripts\\run_evaluation.py `
    --source satellite `
    --prediction "C:\\...\\AliChaabanThesis.gdb\\My_Road_Prediction"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# IMPORTANT:
# Add Roads/ to sys.path, NOT Roads/src/.
#
# Roads/
# └── src/
#     └── ...
#
# This allows:
#
#     from src....
#
# to work correctly.
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from src.evaluation.runner import (
    run_standalone_final_evaluation,
)

from src.pipeline import (
    prepare_experiment,
    print_experiment_summary,
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Strict GIS evaluation of an existing Road prediction."
        )
    )

    parser.add_argument(
        "--source",

        choices=[
            "aerial",
            "satellite",
            "drone",
        ],

        required=True,

        help=(
            "Independent final-test source."
        ),
    )

    parser.add_argument(
        "--prediction",

        type=str,

        default=None,

        help=(
            "Optional existing Road polygon feature class. "
            "If omitted, the prediction from final inference "
            "is used automatically."
        ),
    )

    parser.add_argument(
        "--output-json",

        type=str,

        default=None,

        help=(
            "Optional evaluation JSON output path."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_arguments()

    context = prepare_experiment(
        force_regenerate_manifests=False
    )

    print_experiment_summary(
        context
    )

    run_standalone_final_evaluation(
        context=context,

        source_name=args.source,

        prediction_feature_class=(
            args.prediction
        ),

        output_json=(
            args.output_json
        ),
    )


if __name__ == "__main__":
    main()
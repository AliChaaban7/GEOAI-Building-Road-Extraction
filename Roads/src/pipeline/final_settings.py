"""
Freeze final Road-model settings before independent testing.

Approach 2 - Road Extraction

Purpose
-------
Combine the Validation-selected components into one immutable experiment
record:

    Best checkpoint
    +
    Probability threshold
    +
    Road post-processing parameters
    =
    Frozen final settings

These exact settings must then be used without retuning on:

    Aerial final test
    Satellite final test
    Drone final test

Scientific rule
---------------
This module NEVER reads final-test Ground Truth.

No threshold search, post-processing search, model selection, or
hyperparameter tuning is permitted after the settings are frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import (
    save_json,
)


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def _load_json(
    path: str | Path,
) -> Dict[str, Any]:
    """
    Load and validate a JSON object.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

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
            "Expected JSON object in:\n"
            f"{path}"
        )

    return data


def calculate_file_sha256(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 for a checkpoint.

    This allows us to verify later that the same frozen checkpoint is
    being used for all final tests.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file:\n{path}"
        )

    sha256 = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                int(
                    chunk_size
                )
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ---------------------------------------------------------------------
# Selected threshold
# ---------------------------------------------------------------------

def load_frozen_threshold_source(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load Validation-selected threshold information.
    """

    path = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "threshold_search"
        / "selected_threshold.json"
    )

    data = _load_json(
        path
    )

    if (
        "selected_threshold"
        not in data
    ):
        raise KeyError(
            "selected_threshold.json does not contain "
            "'selected_threshold'."
        )

    threshold = float(
        data[
            "selected_threshold"
        ]
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "Selected threshold must be between 0 and 1."
        )

    if bool(
        data.get(
            "final_test_used",
            False,
        )
    ):
        raise RuntimeError(
            "Threshold source indicates final-test data was used. "
            "Refusing to freeze scientifically invalid settings."
        )

    return {
        "path":
            path,

        "data":
            data,

        "threshold":
            threshold,
    }


# ---------------------------------------------------------------------
# Selected post-processing
# ---------------------------------------------------------------------

def load_frozen_postprocessing_source(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load Validation-selected Road post-processing settings.
    """

    path = (
        Path(
            context[
                "output_folder"
            ]
        )
        / "postprocessing_search"
        / "selected_postprocessing.json"
    )

    data = _load_json(
        path
    )

    required = [
        "minimum_component_size",
        "closing_radius",
        "bridge_gap_radius",
        "maximum_hole_size",
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        raise KeyError(
            "selected_postprocessing.json is missing:\n"
            + "\n".join(
                missing
            )
        )

    if bool(
        data.get(
            "final_test_used",
            False,
        )
    ):
        raise RuntimeError(
            "Post-processing source indicates final-test data was used. "
            "Refusing to freeze scientifically invalid settings."
        )

    config = {
        "minimum_component_size":
            int(
                data[
                    "minimum_component_size"
                ]
            ),

        "closing_radius":
            int(
                data[
                    "closing_radius"
                ]
            ),

        "bridge_gap_radius":
            int(
                data[
                    "bridge_gap_radius"
                ]
            ),

        "maximum_hole_size":
            int(
                data[
                    "maximum_hole_size"
                ]
            ),
    }

    for (
        name,
        value,
    ) in config.items():

        if value < 0:
            raise ValueError(
                f"Invalid frozen post-processing value "
                f"{name}={value}."
            )

    return {
        "path":
            path,

        "data":
            data,

        "config":
            config,
    }


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

def resolve_frozen_checkpoint(
    context: Dict[str, Any],
    checkpoint_path: Optional[
        str | Path
    ] = None,
) -> Dict[str, Any]:
    """
    Resolve and validate the final model checkpoint.
    """

    if checkpoint_path is None:

        checkpoint_path = (
            Path(
                context[
                    "checkpoint_folder"
                ]
            )
            / "best_model.pth"
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Best Road checkpoint does not exist:\n"
            f"{checkpoint_path}\n\n"
            "Run model training first."
        )

    checkpoint_hash = (
        calculate_file_sha256(
            checkpoint_path
        )
    )

    return {
        "path":
            checkpoint_path,

        "sha256":
            checkpoint_hash,
    }


# ---------------------------------------------------------------------
# Optional disabled post-processing mode
# ---------------------------------------------------------------------

def default_no_postprocessing() -> Dict[str, int]:
    """
    Identity Road post-processing configuration.
    """

    return {
        "minimum_component_size":
            0,

        "closing_radius":
            0,

        "bridge_gap_radius":
            0,

        "maximum_hole_size":
            0,
    }


# ---------------------------------------------------------------------
# Optional fixed threshold mode
# ---------------------------------------------------------------------

def resolve_threshold_for_freezing(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve final threshold.

    When threshold search is enabled:
        use selected_threshold.json.

    When disabled:
        use the configured fixed/default threshold.

    This keeps threshold search truly optional.
    """

    experiment_config = context[
        "experiment_config"
    ]

    general_params = context[
        "general_params"
    ]

    if bool(
        experiment_config.get(
            "use_threshold_search",
            False,
        )
    ):

        source = (
            load_frozen_threshold_source(
                context
            )
        )

        return {
            "threshold":
                float(
                    source[
                        "threshold"
                    ]
                ),

            "selection_method":
                "validation_search",

            "source_file":
                str(
                    source[
                        "path"
                    ]
                ),

            "validation_results":
                source[
                    "data"
                ],
        }

    validation_config = (
        general_params.get(
            "validation",
            {}
        )
    )

    threshold = float(
        validation_config.get(
            "default_probability_threshold",
            0.5,
        )
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "Configured fixed threshold must be between 0 and 1."
        )

    return {
        "threshold":
            threshold,

        "selection_method":
            "fixed_configuration",

        "source_file":
            None,

        "validation_results":
            None,
    }


# ---------------------------------------------------------------------
# Optional post-processing mode
# ---------------------------------------------------------------------

def resolve_postprocessing_for_freezing(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve final Road post-processing settings.

    When post-processing is disabled:
        use identity processing.

    When enabled:
        require Validation-selected settings.
    """

    experiment_config = context[
        "experiment_config"
    ]

    if not bool(
        experiment_config.get(
            "use_postprocessing",
            False,
        )
    ):

        return {
            "enabled":
                False,

            "selection_method":
                "disabled",

            "source_file":
                None,

            "parameters":
                default_no_postprocessing(),

            "validation_results":
                None,
        }

    source = (
        load_frozen_postprocessing_source(
            context
        )
    )

    return {
        "enabled":
            True,

        "selection_method":
            "validation_search",

        "source_file":
            str(
                source[
                    "path"
                ]
            ),

        "parameters":
            source[
                "config"
            ],

        "validation_results":
            source[
                "data"
            ],
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_frozen_settings(
    settings: Dict[str, Any],
) -> None:
    """
    Validate the final-settings object before saving.
    """

    if (
        settings.get(
            "selection_dataset"
        )
        != "validation"
    ):
        raise RuntimeError(
            "Frozen settings must be selected without final-test tuning."
        )

    if bool(
        settings.get(
            "final_test_used_for_selection",
            True,
        )
    ):
        raise RuntimeError(
            "Final-test data cannot be used for model-setting selection."
        )

    threshold = float(
        settings[
            "inference"
        ][
            "probability_threshold"
        ]
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "Frozen probability threshold is invalid."
        )

    checkpoint_hash = str(
        settings[
            "checkpoint"
        ][
            "sha256"
        ]
    )

    if len(
        checkpoint_hash
    ) != 64:
        raise ValueError(
            "Frozen checkpoint SHA-256 is invalid."
        )

    final_sources = settings.get(
        "final_test_sources",
        []
    )

    if not final_sources:
        raise ValueError(
            "At least one final-test source must be defined."
        )


# ---------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------

def freeze_final_settings(
    context: Dict[str, Any],
    checkpoint_path: Optional[
        str | Path
    ] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Create the final immutable Road experiment settings.

    Returns
    -------
    dict
        Frozen final-settings document.
    """

    if (
        context[
            "model_backend"
        ]
        != "pytorch"
    ):

        raise RuntimeError(
            "Current freeze implementation is for native PyTorch "
            "Road models. ArcGIS Learn models will use their own "
            "checkpoint/model-package handling."
        )

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    frozen_path = (
        output_folder
        / "frozen_final_settings.json"
    )

    if (
        frozen_path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            "Frozen final settings already exist:\n"
            f"{frozen_path}\n\n"
            "They are intentionally protected against accidental "
            "replacement.\n"
            "Use overwrite=True only if you deliberately want to "
            "invalidate and recreate the frozen experiment."
        )

    checkpoint = (
        resolve_frozen_checkpoint(
            context=context,
            checkpoint_path=checkpoint_path,
        )
    )

    threshold = (
        resolve_threshold_for_freezing(
            context
        )
    )

    postprocessing = (
        resolve_postprocessing_for_freezing(
            context
        )
    )

    experiment_config = context[
        "experiment_config"
    ]

    final_sources = (
        experiment_config.get(
            "final_test_sources",
            [
                "aerial",
                "satellite",
                "drone",
            ],
        )
    )

    # Keep only strings, preserving configured order.
    final_sources = [
        str(
            source
        ).lower()
        for source in final_sources
    ]

    settings = {
        "schema_version":
            1,

        "status":
            "FROZEN",

        "approach":
            "roads",

        "experiment_id":
            experiment_config[
                "experiment_id"
            ],

        "model": {
            "model_type":
                context[
                    "model_type"
                ],

            "model_name":
                context[
                    "model_spec"
                ].display_name,

            "backend":
                context[
                    "model_backend"
                ],

            "tile_size":
                int(
                    experiment_config[
                        "tile_size"
                    ]
                ),

            "input_channels":
                3,

            "output_channels":
                1,

            "task":
                "binary_semantic_segmentation",
        },

        "training_data": {
            "dataset_id":
                context[
                    "dataset_id"
                ],

            "dataset_path":
                str(
                    context[
                        "dataset_root"
                    ]
                ),

            "source_type":
                context[
                    "dataset_info"
                ].get(
                    "source_type"
                ),

            "sources":
                context[
                    "dataset_info"
                ].get(
                    "sources",
                    []
                ),

            "train_manifest":
                str(
                    context[
                        "train_manifest"
                    ]
                ),

            "validation_manifest":
                str(
                    context[
                        "validation_manifest"
                    ]
                ),
        },

        "checkpoint": {
            "path":
                str(
                    checkpoint[
                        "path"
                    ]
                ),

            "sha256":
                checkpoint[
                    "sha256"
                ],
        },

        "inference": {
            "probability_threshold":
                float(
                    threshold[
                        "threshold"
                    ]
                ),

            "threshold_selection_method":
                threshold[
                    "selection_method"
                ],

            "threshold_source_file":
                threshold[
                    "source_file"
                ],
        },

        "postprocessing": {
            "enabled":
                bool(
                    postprocessing[
                        "enabled"
                    ]
                ),

            "selection_method":
                postprocessing[
                    "selection_method"
                ],

            "source_file":
                postprocessing[
                    "source_file"
                ],

            "parameter_units":
                "pixels",

            "parameters":
                postprocessing[
                    "parameters"
                ],
        },

        "selection_dataset":
            "validation",

        "final_test_used_for_selection":
            False,

        "final_test_sources":
            final_sources,

        "final_test_policy": {
            "same_checkpoint_for_all_sources":
                True,

            "same_threshold_for_all_sources":
                True,

            "same_postprocessing_for_all_sources":
                True,

            "source_specific_retuning":
                False,

            "threshold_search_on_final_test":
                False,

            "postprocessing_search_on_final_test":
                False,

            "model_selection_on_final_test":
                False,
        },
    }

    validate_frozen_settings(
        settings
    )

    save_json(
        settings,
        frozen_path,
    )

    return settings


# ---------------------------------------------------------------------
# Frozen settings loader
# ---------------------------------------------------------------------

def load_frozen_final_settings(
    context: Optional[
        Dict[str, Any]
    ] = None,
    path: Optional[
        str | Path
    ] = None,
    verify_checkpoint: bool = True,
) -> Dict[str, Any]:
    """
    Load frozen final settings.

    Optionally verify that the checkpoint file has not changed.
    """

    if path is None:

        if context is None:
            raise ValueError(
                "Either context or path must be provided."
            )

        path = (
            Path(
                context[
                    "output_folder"
                ]
            )
            / "frozen_final_settings.json"
        )

    path = Path(
        path
    )

    settings = _load_json(
        path
    )

    validate_frozen_settings(
        settings
    )

    if verify_checkpoint:

        checkpoint_path = Path(
            settings[
                "checkpoint"
            ][
                "path"
            ]
        )

        expected_hash = settings[
            "checkpoint"
        ][
            "sha256"
        ]

        actual_hash = calculate_file_sha256(
            checkpoint_path
        )

        if actual_hash != expected_hash:

            raise RuntimeError(
                "Frozen checkpoint integrity check FAILED.\n\n"
                f"Checkpoint:\n{checkpoint_path}\n\n"
                f"Expected SHA-256:\n{expected_hash}\n\n"
                f"Actual SHA-256:\n{actual_hash}\n\n"
                "The checkpoint has changed since settings were frozen."
            )

    return settings
"""
test_config_loading.py

Test configuration loading for the Buildings module.

Compatible with:
- DeepLabV3
- U-Net
- get_dataset_info() returning 2 or 3 values
"""

from pathlib import Path
import sys


# ============================================================
# Paths
# ============================================================

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent

sys.path.append(str(BUILDINGS_ROOT))
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.utils import (
    load_experiment_config,
    load_model_config,
    load_paths_config,
    load_datasets_config,
    load_general_params,
    get_dataset_info,
    get_backbone_info,
    get_automatic_batch_size
)


# ============================================================
# Helpers
# ============================================================

def unpack_dataset_result(dataset_result):
    """
    Support different versions of get_dataset_info().

    Possible returns:
        dict
        dataset_id, dataset_info
        dataset_id, dataset_info, dataset_path
        dataset_id, dataset_info, dataset_path, extra...
    """

    # Case 1: get_dataset_info returns dictionary
    if isinstance(dataset_result, dict):
        dataset_info = dataset_result
        dataset_id = dataset_info.get("dataset_id", "unknown_dataset")
        dataset_path = dataset_info.get("path", None)

        return dataset_id, dataset_info, dataset_path

    # Case 2: get_dataset_info returns tuple
    if isinstance(dataset_result, tuple):
        if len(dataset_result) == 2:
            dataset_id, dataset_info = dataset_result
            dataset_path = None

            if isinstance(dataset_info, dict):
                dataset_path = dataset_info.get("path", None)

            return dataset_id, dataset_info, dataset_path

        if len(dataset_result) >= 3:
            dataset_id = dataset_result[0]
            dataset_info = dataset_result[1]
            dataset_path = dataset_result[2]

            return dataset_id, dataset_info, dataset_path

    raise ValueError(
        f"Unexpected get_dataset_info() return format: {type(dataset_result)} -> {dataset_result}"
    )   



    if isinstance(dataset_result, tuple):
        if len(dataset_result) == 2:
            dataset_id, dataset_info = dataset_result
            dataset_path = None

            if isinstance(dataset_info, dict):
                dataset_path = dataset_info.get("path", None)

            return dataset_id, dataset_info, dataset_path

        if len(dataset_result) >= 3:
            dataset_id = dataset_result[0]
            dataset_info = dataset_result[1]
            dataset_path = dataset_result[2]

            return dataset_id, dataset_info, dataset_path

    raise ValueError(
        f"Unexpected get_dataset_info() return format: {type(dataset_result)} -> {dataset_result}"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("\n========== CONFIG LOADING TEST ==========\n")

    # IMPORTANT:
    # load_experiment_config can read the exact experiment.json path.
    experiment_config = load_experiment_config(
        BUILDINGS_ROOT / "config" / "experiment.json"
    )

    # IMPORTANT:
    # These functions expect BUILDINGS_ROOT, not the full json file path.
    paths_config = load_paths_config(
        BUILDINGS_ROOT
    )

    datasets_config = load_datasets_config(
        BUILDINGS_ROOT
    )

    general_params = load_general_params(
        BUILDINGS_ROOT
    )

    model_config = load_model_config(
        experiment_config["model_type"],
        BUILDINGS_ROOT
    )

    dataset_result = get_dataset_info(
        experiment_config,
        datasets_config
    )

    dataset_id, dataset_info, dataset_path = unpack_dataset_result(
        dataset_result
    )

    backbone_info = get_backbone_info(
        experiment_config,
        model_config
    )

    automatic_batch_size = get_automatic_batch_size(
        experiment_config,
        backbone_info
    )

    print("Experiment ID:", experiment_config.get("experiment_id"))
    print("Approach:", experiment_config.get("approach"))
    print("Model Type:", experiment_config.get("model_type"))
    print("Backbone:", experiment_config.get("backbone"))
    print("Source Type:", experiment_config.get("source_type"))
    print("Tile Size:", experiment_config.get("tile_size"))
    print("Dataset Type:", experiment_config.get("dataset_type"))

    print("\n---------- DATASET ----------")
    print("Dataset ID:", dataset_id)

    if dataset_path is not None:
        print("Dataset Path:", dataset_path)

    if isinstance(dataset_info, dict):
        print("Dataset Info Keys:", list(dataset_info.keys()))

    print("\n---------- PATHS CONFIG ----------")
    print("Paths Config Keys:", list(paths_config.keys()))

    print("\n---------- MODEL ----------")
    print("Model Config Type:", model_config.get("model_type"))
    print("Default Backbone:", model_config.get("default_backbone"))
    print("Selected Backbone:", backbone_info.get("name", experiment_config.get("backbone")))

    print("\n---------- TRAINING ----------")
    print("Automatic Batch Size:", automatic_batch_size)

    training_config = general_params.get("training", {})
    print("Epochs:", training_config.get("epochs"))
    print("Optimizer:", training_config.get("optimizer"))
    print("Learning Rate:", training_config.get("learning_rate", training_config.get("lr")))
    print("Weight Decay:", training_config.get("weight_decay"))

    print("\nConfig loading test completed successfully.")
    print("=========================================\n")


if __name__ == "__main__":
    main()
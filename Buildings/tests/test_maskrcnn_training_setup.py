"""
test_maskrcnn_training_setup.py

Test Mask R-CNN dataset loading and one training loss batch.

This script checks:
1. Current experiment.json is Mask R-CNN.
2. Correct RCNN dataset path exists in datasets.json.
3. Image-mask pairs are discovered.
4. Targets contain boxes, labels, masks.
5. Mask R-CNN can calculate training losses for one batch.

It does not train the full model.
"""

from pathlib import Path
import sys
import json

import torch
from torch.utils.data import DataLoader


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

from src.models.factory import build_model
from src.instance_dataset import (
    BuildingMaskRCNNDataset,
    collate_fn,
    create_train_val_manifests
)


# ============================================================
# JSON helpers
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Config loading
# ============================================================

def load_experiment_config():
    return load_json(BUILDINGS_ROOT / "config" / "experiment.json")


def load_datasets_config():
    return load_json(BUILDINGS_ROOT / "config" / "datasets.json")


def load_general_params():
    return load_json(BUILDINGS_ROOT / "config" / "general_params.json")


def load_model_config(model_type):
    return load_json(BUILDINGS_ROOT / "config" / "models" / f"{model_type}.json")


def normalize_model_type(model_type):
    model_type = str(model_type).lower().strip()

    aliases = {
        "mask_rcnn": "maskrcnn",
        "mask-r-cnn": "maskrcnn",
        "rcnn": "maskrcnn"
    }

    return aliases.get(model_type, model_type)


def get_backbone_info(experiment_config, model_config):
    backbone = experiment_config.get(
        "backbone",
        model_config.get("default_backbone", None)
    )

    if backbone is None:
        return {}

    if "supported_backbones" in model_config:
        return model_config["supported_backbones"].get(
            backbone,
            {"name": backbone}
        )

    if "backbones" in model_config:
        return model_config["backbones"].get(
            backbone,
            {"name": backbone}
        )

    return {"name": backbone}


# ============================================================
# Flexible dataset search
# ============================================================

def find_dataset_info_recursive(
    obj,
    source_type,
    tile_size,
    dataset_type,
    current_key=None
):
    """
    Recursively search datasets.json for matching dataset.

    Match by:
    - source_type
    - tile_size
    - dataset_type
    """

    if isinstance(obj, dict):
        has_path = "path" in obj

        if has_path:
            obj_source = str(obj.get("source_type", "")).lower()
            obj_dataset_type = str(obj.get("dataset_type", "")).lower()

            try:
                obj_tile_size = int(obj.get("tile_size"))
            except Exception:
                obj_tile_size = None

            if (
                obj_source == str(source_type).lower()
                and obj_dataset_type == str(dataset_type).lower()
                and obj_tile_size == int(tile_size)
            ):
                result = dict(obj)
                result["dataset_id"] = current_key
                return result

        for key, value in obj.items():
            found = find_dataset_info_recursive(
                value,
                source_type=source_type,
                tile_size=tile_size,
                dataset_type=dataset_type,
                current_key=key
            )

            if found is not None:
                return found

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            found = find_dataset_info_recursive(
                item,
                source_type=source_type,
                tile_size=tile_size,
                dataset_type=dataset_type,
                current_key=str(index)
            )

            if found is not None:
                return found

    return None


def get_dataset_info(experiment_config, datasets_config):
    source_type = experiment_config.get("source_type")
    tile_size = int(experiment_config.get("tile_size"))
    dataset_type = experiment_config.get("dataset_type")

    dataset_info = find_dataset_info_recursive(
        datasets_config,
        source_type=source_type,
        tile_size=tile_size,
        dataset_type=dataset_type
    )

    if dataset_info is None:
        raise ValueError(
            "Dataset not found in datasets.json.\n"
            f"Requested source_type: {source_type}\n"
            f"Requested tile_size: {tile_size}\n"
            f"Requested dataset_type: {dataset_type}\n"
            "Add the matching RCNN dataset path to datasets.json."
        )

    return dataset_info


def get_validation_ratio(general_params):
    if isinstance(general_params, dict):
        if "training" in general_params:
            if "validation_ratio" in general_params["training"]:
                return float(general_params["training"]["validation_ratio"])

        if "validation_ratio" in general_params:
            return float(general_params["validation_ratio"])

    return 0.2


def get_batch_size(model_config):
    if "training" in model_config:
        if "batch_size" in model_config["training"]:
            return int(model_config["training"]["batch_size"])

    return 1


def move_targets_to_device(targets, device):
    moved_targets = []

    for target in targets:
        moved_target = {}

        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value

        moved_targets.append(moved_target)

    return moved_targets


# ============================================================
# Main
# ============================================================

def main():
    experiment_config = load_experiment_config()
    datasets_config = load_datasets_config()
    general_params = load_general_params()

    model_type = normalize_model_type(
        experiment_config.get("model_type")
    )

    if model_type != "maskrcnn":
        raise ValueError(
            "Current experiment.json is not set to Mask R-CNN.\n"
            f"Current model_type: {model_type}"
        )

    model_config = load_model_config(model_type)
    backbone_info = get_backbone_info(
        experiment_config=experiment_config,
        model_config=model_config
    )

    dataset_info = get_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config
    )

    experiment_id = experiment_config.get("experiment_id")
    output_folder = (
        BUILDINGS_ROOT
        / "outputs"
        / "experiments"
        / experiment_id
    )

    validation_ratio = get_validation_ratio(general_params)

    manifest_summary = create_train_val_manifests(
        dataset_path=dataset_info["path"],
        output_folder=output_folder,
        validation_ratio=validation_ratio,
        seed=42
    )

    train_csv = manifest_summary["train_csv"]
    val_csv = manifest_summary["val_csv"]

    train_dataset = BuildingMaskRCNNDataset(
        manifest_csv=train_csv,
        min_object_pixels=5
    )

    val_dataset = BuildingMaskRCNNDataset(
        manifest_csv=val_csv,
        min_object_pixels=5
    )

    batch_size = get_batch_size(model_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, device = build_model(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        move_to_device=True
    )

    print("\n========== MASK R-CNN TRAINING SETUP TEST ==========")
    print(f"Experiment ID: {experiment_id}")
    print(f"Model Type: {model_type}")
    print(f"Backbone: {experiment_config.get('backbone')}")
    print(f"Device: {device}")
    print("----------------------------------------------")
    print(f"Dataset ID: {dataset_info.get('dataset_id')}")
    print(f"Dataset Path: {dataset_info['path']}")
    print(f"Image-Mask Pair Count: {manifest_summary['pair_count']}")
    print(f"Train Count: {manifest_summary['train_count']}")
    print(f"Validation Count: {manifest_summary['val_count']}")
    print(f"Train Manifest: {train_csv}")
    print(f"Val Manifest: {val_csv}")
    print("----------------------------------------------")
    print(f"Batch Size: {batch_size}")

    images, targets = next(iter(train_loader))

    print(f"Loaded Batch Images: {len(images)}")
    print(f"First Image Shape: {tuple(images[0].shape)}")
    print(f"First Target Keys: {list(targets[0].keys())}")
    print(f"First Target Boxes Shape: {tuple(targets[0]['boxes'].shape)}")
    print(f"First Target Labels Shape: {tuple(targets[0]['labels'].shape)}")
    print(f"First Target Masks Shape: {tuple(targets[0]['masks'].shape)}")

    object_counts = [int(t["boxes"].shape[0]) for t in targets]
    print(f"Objects in Batch: {object_counts}")

    images = [img.to(device) for img in images]
    targets = move_targets_to_device(targets, device)

    model.train()

    loss_dict = model(images, targets)
    total_loss = sum(loss for loss in loss_dict.values())

    print("----------------------------------------------")
    print("Loss Dict:")

    for key, value in loss_dict.items():
        print(f"{key}: {float(value.detach().cpu()):.4f}")

    print(f"Total Loss: {float(total_loss.detach().cpu()):.4f}")
    print("Mask R-CNN training setup test completed successfully.")
    print("======================================================\n")


if __name__ == "__main__":
    main()
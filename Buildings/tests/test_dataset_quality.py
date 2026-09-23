"""
test_dataset_quality.py

Flexible dataset quality and manifest creator for the Buildings module.

This script is model-independent.

It works for:
- DeepLabV3
- U-Net
- future semantic segmentation models

It reads:
- experiment.json
- datasets.json
- general_params.json

It creates:
- data_quality.csv
- train_manifest.csv
- val_manifest.csv
- generated_empty_masks/

The manifests are saved to multiple compatible experiment folders so old scripts
and new flexible scripts can both find them.
"""

from pathlib import Path
import sys
import json
import random
import shutil

import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# Paths
# ============================================================

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent

sys.path.append(str(BUILDINGS_ROOT))
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Basic JSON helpers
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Config loading
# ============================================================

def load_experiment_config():
    return load_json(BUILDINGS_ROOT / "config" / "experiment.json")


def load_datasets_config():
    return load_json(BUILDINGS_ROOT / "config" / "datasets.json")


def load_general_params():
    return load_json(BUILDINGS_ROOT / "config" / "general_params.json")


# ============================================================
# Dataset resolver
# ============================================================

def flatten_datasets(obj):
    """
    Recursively find dataset dictionaries inside datasets.json.
    A valid dataset must contain at least:
    - path
    - source_type
    - tile_size
    - dataset_type
    """

    found = []

    if isinstance(obj, dict):
        if "path" in obj and "source_type" in obj and "tile_size" in obj and "dataset_type" in obj:
            found.append(obj)

        for value in obj.values():
            found.extend(flatten_datasets(value))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(flatten_datasets(item))

    return found


def get_flexible_dataset_info(experiment_config, datasets_config):
    """
    Find dataset matching:
    - source_type
    - tile_size
    - dataset_type
    """

    source_type = str(experiment_config.get("source_type")).lower()
    tile_size = int(experiment_config.get("tile_size"))
    dataset_type = str(experiment_config.get("dataset_type")).lower()

    datasets = flatten_datasets(datasets_config)

    for ds in datasets:
        if (
            str(ds.get("source_type")).lower() == source_type
            and int(ds.get("tile_size")) == tile_size
            and str(ds.get("dataset_type")).lower() == dataset_type
        ):
            dataset_info = dict(ds)

            if "dataset_id" not in dataset_info:
                dataset_info["dataset_id"] = f"{source_type}_{dataset_type}_{tile_size}"

            return dataset_info

    raise ValueError(
        "No matching dataset found in datasets.json for:\n"
        f"source_type={source_type}, tile_size={tile_size}, dataset_type={dataset_type}"
    )


# ============================================================
# Folder helpers
# ============================================================

def get_experiment_folder_names(experiment_config):
    """
    Create multiple compatible output folder names.

    This prevents errors between older scripts and newer flexible scripts.
    """

    experiment_id = experiment_config.get("experiment_id", None)
    model_type = experiment_config.get("model_type")
    source_type = experiment_config.get("source_type")
    tile_size = experiment_config.get("tile_size")

    names = []

    if experiment_id:
        names.append(str(experiment_id))

    names.append(f"{model_type}_{tile_size}_{source_type}")
    names.append(f"{model_type}_{tile_size}_{source_type}_baseline")

    # remove duplicates while preserving order
    final_names = []
    for name in names:
        if name not in final_names:
            final_names.append(name)

    return final_names


def get_main_output_folder(experiment_config):
    """
    Main folder used by the current experiment.
    """

    names = get_experiment_folder_names(experiment_config)

    return BUILDINGS_ROOT / "outputs" / "experiments" / names[0]


def get_all_output_folders(experiment_config):
    """
    All compatible output folders.
    """

    names = get_experiment_folder_names(experiment_config)

    return [
        BUILDINGS_ROOT / "outputs" / "experiments" / name
        for name in names
    ]


# ============================================================
# Dataset file discovery
# ============================================================

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_first_existing_folder(base_path, candidates):
    for candidate in candidates:
        folder = base_path / candidate
        if folder.exists() and folder.is_dir():
            return folder

    return None


def collect_files(folder, extensions):
    files = []

    if folder is None or not folder.exists():
        return files

    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))

    return sorted(files)


def find_image_and_label_folders(dataset_path):
    """
    Supports common ArcGIS Export Training Data structures.
    """

    dataset_path = Path(dataset_path)

    image_folder = find_first_existing_folder(
        dataset_path,
        [
            "images",
            "Images",
            "image",
            "chips",
            "Chips",
            "raster",
            "rasters"
        ]
    )

    label_folder = find_first_existing_folder(
        dataset_path,
        [
            "labels",
            "Labels",
            "masks",
            "Masks",
            "label",
            "mask"
        ]
    )

    if image_folder is None:
        image_folder = dataset_path

    if label_folder is None:
        label_folder = dataset_path

    return image_folder, label_folder


def build_label_index(label_files):
    """
    Build label lookup by filename stem.
    """

    label_index = {}

    for label_path in label_files:
        label_index[label_path.stem] = label_path

    return label_index


def find_matching_label(image_path, label_index):
    """
    Match image chip with mask/label by stem.

    Supports:
    - exact stem match
    - common ArcGIS naming similarities
    """

    stem = image_path.stem

    if stem in label_index:
        return label_index[stem]

    # fallback: try simple cleanup
    cleaned = (
        stem.replace("_image", "")
        .replace("_img", "")
        .replace("image_", "")
        .replace("img_", "")
    )

    if cleaned in label_index:
        return label_index[cleaned]

    # fallback: partial contains
    for label_stem, label_path in label_index.items():
        if label_stem in stem or stem in label_stem:
            return label_path

    return None


# ============================================================
# Mask helpers
# ============================================================

def is_positive_mask(mask_path):
    """
    Return True if mask has at least one positive pixel.
    """

    try:
        mask = Image.open(mask_path)
        arr = np.array(mask)

        if arr.ndim == 3:
            arr = arr[:, :, 0]

        return bool(arr.max() > 0)

    except Exception:
        return False


def create_empty_mask_for_image(image_path, empty_masks_folder):
    """
    Create a blank mask with the same width and height as the image.
    """

    empty_masks_folder = Path(empty_masks_folder)
    empty_masks_folder.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    width, height = image.size

    empty_mask = Image.new("L", (width, height), 0)

    output_path = empty_masks_folder / f"{Path(image_path).stem}_empty_mask.png"
    empty_mask.save(output_path)

    return output_path


# ============================================================
# Manifest creation
# ============================================================

def create_dataset_manifests(
    experiment_config,
    dataset_info,
    general_params,
    output_folder
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_info["path"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    image_folder, label_folder = find_image_and_label_folders(dataset_path)

    image_files = collect_files(image_folder, IMAGE_EXTENSIONS)
    label_files = collect_files(label_folder, MASK_EXTENSIONS)

    label_index = build_label_index(label_files)

    empty_masks_folder = output_folder / "generated_empty_masks"
    empty_masks_folder.mkdir(parents=True, exist_ok=True)

    positive_rows = []
    background_rows = []
    problem_rows = []

    paired_count = 0
    missing_label_count = 0

    for image_path in image_files:
        try:
            label_path = find_matching_label(image_path, label_index)

            if label_path is None:
                missing_label_count += 1
                label_path = create_empty_mask_for_image(
                    image_path=image_path,
                    empty_masks_folder=empty_masks_folder
                )
                is_positive = False

            else:
                paired_count += 1
                is_positive = is_positive_mask(label_path)

            row = {
                "image_path": str(image_path),
                "mask_path": str(label_path),
                "is_positive": int(is_positive),
                "source_type": experiment_config.get("source_type"),
                "tile_size": experiment_config.get("tile_size"),
                "dataset_type": experiment_config.get("dataset_type"),
                "model_type": experiment_config.get("model_type"),
                "backbone": experiment_config.get("backbone")
            }

            if is_positive:
                positive_rows.append(row)
            else:
                background_rows.append(row)

        except Exception as e:
            problem_rows.append(
                {
                    "image_path": str(image_path),
                    "error": str(e)
                }
            )

    training_config = general_params.get("training", {})
    dataset_config = general_params.get("dataset", {})

    seed = int(training_config.get("seed", 42))
    validation_ratio = float(training_config.get("validation_ratio", dataset_config.get("validation_ratio", 0.2)))
    background_ratio = float(training_config.get("background_ratio", dataset_config.get("background_ratio", 0.5)))

    random.seed(seed)

    random.shuffle(positive_rows)
    random.shuffle(background_rows)

    selected_background_count = int(round(len(positive_rows) * background_ratio))
    selected_background_count = min(selected_background_count, len(background_rows))

    selected_rows = positive_rows + background_rows[:selected_background_count]
    random.shuffle(selected_rows)

    val_count = int(round(len(selected_rows) * validation_ratio))
    train_count = len(selected_rows) - val_count

    train_rows = selected_rows[:train_count]
    val_rows = selected_rows[train_count:]

    train_manifest = pd.DataFrame(train_rows)
    val_manifest = pd.DataFrame(val_rows)

    data_quality = pd.DataFrame(
        [
            {"metric": "image_count", "value": len(image_files)},
            {"metric": "label_count", "value": len(label_files)},
            {"metric": "paired_count", "value": paired_count},
            {"metric": "missing_label_count", "value": missing_label_count},
            {"metric": "positive_chips", "value": len(positive_rows)},
            {"metric": "background_chips", "value": len(background_rows)},
            {"metric": "unreadable_problem_chips", "value": len(problem_rows)},
            {"metric": "selected_positive_count", "value": len(positive_rows)},
            {"metric": "selected_background_count", "value": selected_background_count},
            {"metric": "total_selected_count", "value": len(selected_rows)},
            {"metric": "train_count", "value": len(train_rows)},
            {"metric": "validation_count", "value": len(val_rows)}
        ]
    )

    train_manifest.to_csv(output_folder / "train_manifest.csv", index=False)
    val_manifest.to_csv(output_folder / "val_manifest.csv", index=False)
    data_quality.to_csv(output_folder / "data_quality.csv", index=False)

    if problem_rows:
        pd.DataFrame(problem_rows).to_csv(output_folder / "problem_chips.csv", index=False)

    summary = {
        "dataset_id": dataset_info.get("dataset_id"),
        "dataset_path": str(dataset_path),
        "image_folder": str(image_folder),
        "label_folder": str(label_folder),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "paired_count": paired_count,
        "missing_label_count": missing_label_count,
        "positive_chips": len(positive_rows),
        "background_chips": len(background_rows),
        "unreadable_problem_chips": len(problem_rows),
        "selected_positive_count": len(positive_rows),
        "selected_background_count": selected_background_count,
        "total_selected_count": len(selected_rows),
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_manifest": str(output_folder / "train_manifest.csv"),
        "val_manifest": str(output_folder / "val_manifest.csv"),
        "data_quality_csv": str(output_folder / "data_quality.csv")
    }

    save_json(summary, output_folder / "data_quality_summary.json")

    return summary


def copy_manifest_outputs(source_folder, target_folder):
    """
    Copy generated manifests and quality files to compatible folders.
    """

    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "train_manifest.csv",
        "val_manifest.csv",
        "data_quality.csv",
        "data_quality_summary.json",
        "problem_chips.csv"
    ]

    for filename in files_to_copy:
        src = source_folder / filename
        if src.exists():
            shutil.copy2(src, target_folder / filename)

    src_empty = source_folder / "generated_empty_masks"
    dst_empty = target_folder / "generated_empty_masks"

    if src_empty.exists():
        if dst_empty.exists():
            shutil.rmtree(dst_empty)

        shutil.copytree(src_empty, dst_empty)


# ============================================================
# Main
# ============================================================

def main():
    print("\n========== DATASET QUALITY ==========\n")

    experiment_config = load_experiment_config()
    datasets_config = load_datasets_config()
    general_params = load_general_params()

    dataset_info = get_flexible_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config
    )

    output_folders = get_all_output_folders(experiment_config)
    main_output_folder = output_folders[0]

    summary = create_dataset_manifests(
        experiment_config=experiment_config,
        dataset_info=dataset_info,
        general_params=general_params,
        output_folder=main_output_folder
    )

    # Copy to all compatible folders so old scripts do not fail.
    for folder in output_folders[1:]:
        copy_manifest_outputs(
            source_folder=main_output_folder,
            target_folder=folder
        )

    print(f"Experiment ID: {experiment_config.get('experiment_id')}")
    print(f"Model Type: {experiment_config.get('model_type')}")
    print(f"Backbone: {experiment_config.get('backbone')}")
    print(f"Dataset ID: {dataset_info.get('dataset_id')}")
    print(f"Dataset Path: {dataset_info.get('path')}")
    print("-------------------------------------")
    print(f"Image Count: {summary['image_count']}")
    print(f"Label Count: {summary['label_count']}")
    print(f"Paired Count: {summary['paired_count']}")
    print(f"Missing Label Count: {summary['missing_label_count']}")
    print(f"Positive Chips: {summary['positive_chips']}")
    print(f"Background Chips: {summary['background_chips']}")
    print(f"Unreadable/Problem Chips: {summary['unreadable_problem_chips']}")
    print(f"Selected Positive Count: {summary['selected_positive_count']}")
    print(f"Selected Background Count: {summary['selected_background_count']}")
    print(f"Total Selected Count: {summary['total_selected_count']}")
    print(f"Train Count: {summary['train_count']}")
    print(f"Validation Count: {summary['validation_count']}")
    print("-------------------------------------")

    print("\nCreated manifests in:")
    for folder in output_folders:
        print(f"- {folder}")

    print("\nDataset quality test completed successfully.")
    print("=====================================\n")


if __name__ == "__main__":
    main()
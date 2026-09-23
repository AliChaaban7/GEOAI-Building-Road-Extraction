"""
utils.py

Shared utility functions for the Building Extraction module.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- Mask R-CNN
- SAM-LoRA

Updated shared behavior:
- Flexible model/backbone config loading.
- Model-type alias normalization, including SAM-LoRA.
- Supports config keys:
    supported_backbones
    backbones
    architectures
- Detects disabled backbones.
- Safe model-config resolution for config/models/<model_type>.json.
- Generic SAM-LoRA batch-size fallback while preserving dedicated
  SAM-LoRA batch-size resolution in its own factory.
- JSON-safe serialization for Path and NumPy values.
- Keeps dataset quality, deterministic Train/Validation manifest
  creation, and the independent-test exclusion policy unchanged.
"""

from pathlib import Path
import json
import csv
import random
import shutil

import numpy as np
from PIL import Image

from src.raster_io import get_raster_size, read_mask_array

try:
    import torch
except Exception:
    torch = None


# =====================================================
# MODEL TYPE NORMALIZATION
# =====================================================

def normalize_model_type(model_type):
    """
    Normalize historical model aliases to canonical config names.

    Canonical names:
        unet
        deeplabv3
        maskrcnn
        sam_lora
    """

    normalized = str(
        model_type
    ).strip().lower()

    aliases = {
        "u-net": "unet",
        "u_net": "unet",

        "deeplab": "deeplabv3",
        "deeplab_v3": "deeplabv3",
        "deep_lab_v3": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",

        "mask_rcnn": "maskrcnn",
        "mask-r-cnn": "maskrcnn",
        "mask r-cnn": "maskrcnn",
        "rcnn": "maskrcnn",

        "sam-lora": "sam_lora",
        "sam lora": "sam_lora",
        "samlora": "sam_lora",
        "sam_lora": "sam_lora",
    }

    return aliases.get(
        normalized,
        normalized,
    )


# =====================================================
# JSON / CONFIG HELPERS
# =====================================================

def _json_safe(obj):
    """
    Recursively convert common project values into JSON-safe values.
    """

    if isinstance(
        obj,
        Path,
    ):
        return str(
            obj
        )

    if isinstance(
        obj,
        dict,
    ):
        return {
            key: _json_safe(
                value
            )
            for key, value
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            _json_safe(
                value
            )
            for value
            in obj
        ]

    if isinstance(
        obj,
        tuple,
    ):
        return [
            _json_safe(
                value
            )
            for value
            in obj
        ]

    if isinstance(
        obj,
        np.integer,
    ):
        return int(
            obj
        )

    if isinstance(
        obj,
        np.floating,
    ):
        return float(
            obj
        )

    if isinstance(
        obj,
        np.bool_,
    ):
        return bool(
            obj
        )

    if (
        torch is not None
        and isinstance(
            obj,
            torch.device,
        )
    ):
        return str(
            obj
        )

    return obj


def load_json(path):
    """
    Load a JSON file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, output_path=None, path=None):
    """
    Save data as JSON.

    Compatible with:
        save_json(data, output_path)
        save_json(data=data, path=output_path)

    Path and NumPy scalar values are converted automatically.
    """

    if output_path is None:
        output_path = path

    if output_path is None:
        raise ValueError(
            "save_json needs an output path."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                data
            ),
            file,
            indent=4,
            ensure_ascii=False,
        )

def load_experiment_config(config_path):
    """
    Load experiment.json.
    """
    return load_json(config_path)


def load_general_params(root):
    """
    Load config/general_params.json.
    """
    root = Path(root)
    return load_json(root / "config" / "general_params.json")


def load_paths_config(root):
    """
    Load config/paths.json.
    """
    root = Path(root)
    return load_json(root / "config" / "paths.json")


def load_datasets_config(root):
    """
    Load config/datasets.json.
    """
    root = Path(root)
    return load_json(root / "config" / "datasets.json")


def load_model_config(model_type, root):
    """
    Load config/models/<canonical_model_type>.json.

    Historical aliases are normalized first, for example:

        sam-lora  -> sam_lora
        mask_rcnn -> maskrcnn
        u-net     -> unet
    """

    root = Path(
        root
    )

    canonical_model_type = (
        normalize_model_type(
            model_type
        )
    )

    model_config_path = (
        root
        / "config"
        / "models"
        / f"{canonical_model_type}.json"
    )

    if not model_config_path.exists():

        config_folder = (
            root
            / "config"
            / "models"
        )

        available = []

        if config_folder.exists():

            available = sorted(
                path.stem
                for path
                in config_folder.glob(
                    "*.json"
                )
            )

        raise FileNotFoundError(
            "Model configuration was not found:\n"
            f"    {model_config_path}\n"
            f"Requested model_type: {model_type!r}\n"
            f"Canonical model_type: {canonical_model_type!r}\n"
            f"Available model configs: {available}"
        )

    return load_json(
        model_config_path
    )

# =====================================================
# EXPERIMENT / DATASET HELPERS
# =====================================================

def get_dataset_id(experiment_config):
    """
    Build dataset id from experiment config.

    Example:
        source_type = satellite
        dataset_type = ct
        tile_size = 256

        dataset_id = satellite_ct_256
    """
    source_type = str(experiment_config.get("source_type", "")).lower()
    dataset_type = str(experiment_config.get("dataset_type", "")).lower()
    tile_size = str(experiment_config.get("tile_size", ""))

    if source_type == "" or dataset_type == "" or tile_size == "":
        raise ValueError(
            "Cannot build dataset_id. experiment.json must contain "
            "source_type, dataset_type, and tile_size."
        )

    return f"{source_type}_{dataset_type}_{tile_size}"


def _find_dataset_recursive(data, dataset_id):
    """
    Recursively search for a dataset id inside datasets.json.
    """
    if not isinstance(data, dict):
        return None

    if dataset_id in data and isinstance(data[dataset_id], dict):
        return data[dataset_id]

    for value in data.values():
        if isinstance(value, dict):
            found = _find_dataset_recursive(value, dataset_id)

            if found is not None:
                return found

    return None


def get_dataset_info(experiment_config, datasets_config):
    """
    Get selected dataset information from datasets.json.

    Compatible with:
        {
          "datasets": {
            "satellite_ct_256": {...}
          }
        }

    or:
        {
          "satellite_ct_256": {...}
        }
    """
    dataset_id = get_dataset_id(experiment_config)

    dataset_info = _find_dataset_recursive(
        datasets_config,
        dataset_id
    )

    if dataset_info is None:
        raise ValueError(
            f"Dataset '{dataset_id}' not found in datasets.json."
        )

    dataset_info = dict(dataset_info)
    dataset_info["dataset_id"] = dataset_id

    dataset_path = (
        dataset_info.get("path")
        or dataset_info.get("dataset_path")
        or dataset_info.get("folder")
        or dataset_info.get("root")
    )

    if dataset_path is None:
        raise ValueError(
            f"Dataset '{dataset_id}' found, but no path/dataset_path/folder/root key exists."
        )

    dataset_info["path"] = dataset_path

    return dataset_info


def get_test_image_path(experiment_config, paths_config):
    """
    Get test image path from paths.json.
    """
    test_image_id = experiment_config.get("test_image_id")

    if test_image_id is None:
        raise ValueError("experiment.json must contain test_image_id.")

    test_images = paths_config.get("test_images", {})

    if test_image_id not in test_images:
        raise ValueError(
            f"Test image id '{test_image_id}' not found in paths.json. "
            f"Available: {list(test_images.keys())}"
        )

    return test_images[test_image_id]


def get_ground_truth_path(experiment_config, paths_config):
    """
    Get ground truth path from paths.json.
    """
    ground_truth_id = experiment_config.get("ground_truth_id")

    if ground_truth_id is None:
        raise ValueError("experiment.json must contain ground_truth_id.")

    ground_truth = paths_config.get("ground_truth", {})

    if ground_truth_id not in ground_truth:
        raise ValueError(
            f"Ground truth id '{ground_truth_id}' not found in paths.json. "
            f"Available: {list(ground_truth.keys())}"
        )

    return ground_truth[ground_truth_id]


# =====================================================
# MODEL / BACKBONE HELPERS
# =====================================================

def get_backbone_info(
    experiment_config,
    model_config,
):
    """
    Get backbone information from model config.

    Compatible with:
        supported_backbones
        backbones
        architectures

    Also supports SAM-LoRA's:
        default_backbone
        supported_backbones

    Disabled backbones are rejected when:
        {"enabled": false}
    """

    model_type = normalize_model_type(
        experiment_config.get(
            "model_type",
            model_config.get(
                "model_type",
                "",
            ),
        )
    )

    requested_backbone = (
        experiment_config.get(
            "backbone",
            model_config.get(
                "default_backbone",
                model_config.get(
                    "backbone",
                    None,
                ),
            ),
        )
    )

    if requested_backbone is None:

        requested_backbone = (
            "resnet50"
        )

    requested_backbone = str(
        requested_backbone
    ).strip()

    supported_backbones = {}

    if isinstance(
        model_config.get(
            "supported_backbones"
        ),
        dict,
    ):
        supported_backbones.update(
            model_config[
                "supported_backbones"
            ]
        )

    if isinstance(
        model_config.get(
            "backbones"
        ),
        dict,
    ):
        supported_backbones.update(
            model_config[
                "backbones"
            ]
        )

    if isinstance(
        model_config.get(
            "architectures"
        ),
        dict,
    ):
        supported_backbones.update(
            model_config[
                "architectures"
            ]
        )

    # ---------------------------------------------------------
    # Resolve backbone key case-insensitively without changing
    # the original key stored in the config.
    # ---------------------------------------------------------

    resolved_key = None

    for key in supported_backbones.keys():

        if (
            str(
                key
            ).strip().lower()
            ==
            requested_backbone.lower()
        ):

            resolved_key = key
            break

    if resolved_key is None:

        raise ValueError(
            f"Backbone '{requested_backbone}' not found "
            f"for model '{model_type}'. "
            f"Available: {list(supported_backbones.keys())}. "
            f"Check Buildings/config/models/{model_type}.json"
        )

    backbone_info = dict(
        supported_backbones[
            resolved_key
        ]
    )

    if (
        backbone_info.get(
            "enabled",
            True,
        )
        is False
    ):

        raise ValueError(
            f"Backbone '{resolved_key}' is disabled "
            f"for model '{model_type}'."
        )

    backbone_info[
        "name"
    ] = str(
        resolved_key
    )

    backbone_info[
        "model_type"
    ] = model_type

    return backbone_info


def get_automatic_batch_size(
    experiment_config,
    backbone_info,
    model_config=None,
):
    """
    Get automatic batch size.

    Priority:
    1. backbone_info["automatic_batch_size"]
    2. backbone_info["batch_size"]
    3. SAM-LoRA training.recommended_batch_size_by_backbone
       when model_config is supplied
    4. Existing model-specific fallback rules

    SAM-LoRA's dedicated factory remains the preferred source during
    actual SAM-LoRA training. This shared fallback mainly protects
    generic callers and future UI integration.
    """

    if isinstance(
        backbone_info,
        dict,
    ):

        if (
            "automatic_batch_size"
            in backbone_info
        ):

            return int(
                backbone_info[
                    "automatic_batch_size"
                ]
            )

        if (
            "batch_size"
            in backbone_info
        ):

            return int(
                backbone_info[
                    "batch_size"
                ]
            )

    tile_size = int(
        experiment_config.get(
            "tile_size",
            256,
        )
    )

    model_type = normalize_model_type(
        experiment_config.get(
            "model_type",
            "",
        )
    )

    # ---------------------------------------------------------
    # SAM-LoRA
    # ---------------------------------------------------------

    if model_type == "sam_lora":

        requested_backbone = str(
            experiment_config.get(
                "backbone",
                (
                    backbone_info.get(
                        "name",
                        "vit_b",
                    )
                    if isinstance(
                        backbone_info,
                        dict,
                    )
                    else "vit_b"
                ),
            )
        ).strip().lower()

        if isinstance(
            model_config,
            dict,
        ):

            training_config = (
                model_config.get(
                    "training",
                    {},
                )
            )

            recommended = (
                training_config.get(
                    "recommended_batch_size_by_backbone",
                    {},
                )
            )

            if isinstance(
                recommended,
                dict,
            ):

                for key, value in (
                    recommended.items()
                ):

                    if (
                        str(
                            key
                        ).strip().lower()
                        ==
                        requested_backbone
                    ):

                        batch_size = int(
                            value
                        )

                        if batch_size <= 0:

                            raise ValueError(
                                "Configured SAM-LoRA batch size "
                                "must be > 0."
                            )

                        return batch_size

            default_batch_size = (
                training_config.get(
                    "default_batch_size",
                    None,
                )
            )

            if default_batch_size is not None:

                batch_size = int(
                    default_batch_size
                )

                if batch_size <= 0:

                    raise ValueError(
                        "Configured SAM-LoRA default_batch_size "
                        "must be > 0."
                    )

                return batch_size

        # Safe generic fallback matching the current SAM-LoRA policy.
        if requested_backbone == "vit_b":
            return 2

        if requested_backbone in {
            "vit_l",
            "vit_h",
        }:
            return 1

        return 1

    # ---------------------------------------------------------
    # Existing standard model fallbacks
    # ---------------------------------------------------------

    if model_type == "deeplabv3":

        if tile_size <= 256:
            return 8

        if tile_size <= 384:
            return 4

        return 2

    if model_type == "unet":

        if tile_size <= 256:
            return 8

        if tile_size <= 512:
            return 4

        return 2

    if model_type == "maskrcnn":

        return 2

    return 4

# =====================================================
# OUTPUT FOLDERS
# =====================================================

def create_experiment_output_folder(experiment_config, root):
    """
    Create output folder for the selected experiment.

    Example:
        Buildings/outputs/experiments/deeplabv3_256_satellite_baseline
    """
    root = Path(root)

    experiment_id = experiment_config.get("experiment_id")

    if experiment_id is None:
        raise ValueError("experiment.json must contain experiment_id.")

    output_folder = root / "outputs" / "experiments" / experiment_id
    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder


def save_config_used(
    experiment_config,
    general_params=None,
    model_config=None,
    output_folder=None,
    paths_config=None,
    datasets_config=None,
    **kwargs
):
    """
    Save the complete configuration used for an experiment.

    Compatible with old and new calls.

    Old style:
        save_config_used(
            experiment_config=experiment_config,
            output_folder=output_folder
        )

    New style:
        save_config_used(
            experiment_config=experiment_config,
            general_params=general_params,
            model_config=model_config,
            datasets_config=datasets_config,
            output_folder=output_folder
        )

    SAM-LoRA:
        model_config_used.json stores sam_lora.json content just like
        the other model configurations, keeping experiment artifacts
        comparable and reproducible.
    """

    if output_folder is None:

        output_folder = kwargs.get(
            "save_folder",
            None,
        )

    if output_folder is None:

        output_folder = kwargs.get(
            "experiment_folder",
            None,
        )

    if output_folder is None:

        raise ValueError(
            "output_folder is required "
            "for save_config_used()."
        )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_config = {
        "experiment_config":
            experiment_config,
    }

    if general_params is not None:

        combined_config[
            "general_params"
        ] = general_params

    if model_config is not None:

        combined_config[
            "model_config"
        ] = model_config

    if paths_config is not None:

        combined_config[
            "paths_config"
        ] = paths_config

    if datasets_config is not None:

        combined_config[
            "datasets_config"
        ] = datasets_config


    config_used_path = (
        output_folder
        / "config_used.json"
    )

    experiment_used_path = (
        output_folder
        / "experiment_config_used.json"
    )

    general_used_path = (
        output_folder
        / "general_params_used.json"
    )

    model_used_path = (
        output_folder
        / "model_config_used.json"
    )

    paths_used_path = (
        output_folder
        / "paths_config_used.json"
    )

    datasets_used_path = (
        output_folder
        / "datasets_config_used.json"
    )


    save_json(
        combined_config,
        config_used_path,
    )

    save_json(
        experiment_config,
        experiment_used_path,
    )


    if general_params is not None:

        save_json(
            general_params,
            general_used_path,
        )


    if model_config is not None:

        save_json(
            model_config,
            model_used_path,
        )


    if paths_config is not None:

        save_json(
            paths_config,
            paths_used_path,
        )


    if datasets_config is not None:

        save_json(
            datasets_config,
            datasets_used_path,
        )


    print(
        f"Configuration saved to: "
        f"{output_folder}"
    )


    return {
        "config_used":
            str(
                config_used_path
            ),

        "experiment_config_used":
            str(
                experiment_used_path
            ),

        "general_params_used":
            (
                str(
                    general_used_path
                )
                if general_params
                is not None
                else None
            ),

        "model_config_used":
            (
                str(
                    model_used_path
                )
                if model_config
                is not None
                else None
            ),

        "paths_config_used":
            (
                str(
                    paths_used_path
                )
                if paths_config
                is not None
                else None
            ),

        "datasets_config_used":
            (
                str(
                    datasets_used_path
                )
                if datasets_config
                is not None
                else None
            ),
    }

def set_random_seed(seed=42):
    """
    Set random seed for Python, NumPy, and Torch.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def print_config_summary(
    experiment_config,
    dataset_info=None,
    backbone_info=None,
    batch_size=None
):
    """
    Print readable experiment configuration summary.
    """
    print("\n========== CONFIG SUMMARY ==========")
    print(f"Experiment ID: {experiment_config.get('experiment_id')}")
    print(f"Model Type: {experiment_config.get('model_type')}")
    print(f"Backbone: {experiment_config.get('backbone')}")
    print(f"Source Type: {experiment_config.get('source_type')}")
    print(f"Dataset Type: {experiment_config.get('dataset_type')}")
    print(f"Tile Size: {experiment_config.get('tile_size')}")

    if dataset_info is not None:
        print(f"Dataset ID: {dataset_info.get('dataset_id')}")
        print(f"Dataset Path: {dataset_info.get('path')}")

    if backbone_info is not None:
        print(f"Backbone Info: {backbone_info}")

    if batch_size is not None:
        print(f"Automatic Batch Size: {batch_size}")

    print("====================================\n")


# =====================================================
# DATASET QUALITY HELPERS
# =====================================================

def is_raster_file(path):
    """
    Check if file is a supported raster/image file.
    """
    path = Path(path)

    return path.suffix.lower() in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    ]


def is_label_file(path):
    """
    Determine whether file is probably a label/mask file.
    """
    path = Path(path)
    lower_path = str(path).lower()
    lower_stem = path.stem.lower()

    label_tokens = [
        "label",
        "labels",
        "mask",
        "masks",
        "annotation",
        "annotations",
        "groundtruth",
        "ground_truth"
    ]

    if any(token in lower_path for token in label_tokens):
        return True

    if lower_stem.endswith("_gt"):
        return True

    return False


def normalize_tile_name(path):
    """
    Normalize tile names so image and label files can be paired.

    Examples:
        tile_001.tif
        tile_001_label.png
        tile_001_mask.tif

        -> tile_001
    """
    stem = Path(path).stem.lower()

    replacements = [
        "_label",
        "-label",
        "_labels",
        "-labels",
        "_mask",
        "-mask",
        "_masks",
        "-masks",
        "_annotation",
        "-annotation",
        "_annotations",
        "-annotations",
        "_gt",
        "-gt"
    ]

    changed = True

    while changed:
        changed = False

        for rep in replacements:
            if stem.endswith(rep):
                stem = stem[: -len(rep)]
                changed = True

    return stem


def scan_dataset_files(dataset_path):
    """
    Scan dataset folder and separate image files and label files.

    Works best with folders such as:
        images/
        labels/
        masks/
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    image_files = []
    label_files = []

    all_files = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and is_raster_file(p)
    ]

    for file_path in all_files:
        lower_parts = [
            part.lower()
            for part in file_path.parts
        ]

        parent_tokens = set(lower_parts)

        in_label_folder = any(
            token in parent_tokens
            for token in [
                "label",
                "labels",
                "mask",
                "masks",
                "annotation",
                "annotations"
            ]
        )

        in_image_folder = any(
            token in parent_tokens
            for token in [
                "image",
                "images",
                "img",
                "imgs",
                "tile",
                "tiles",
                "raster",
                "rasters"
            ]
        )

        if in_label_folder or is_label_file(file_path):
            label_files.append(file_path)

        elif in_image_folder:
            image_files.append(file_path)

        else:
            image_files.append(file_path)

    image_files = sorted(image_files)
    label_files = sorted(label_files)

    return image_files, label_files


def pair_image_label_files(image_files, label_files):
    """
    Pair image and label files by normalized tile name.
    """
    label_lookup = {}

    for label_file in label_files:
        key = normalize_tile_name(label_file)
        label_lookup[key] = label_file

    paired = []
    missing_labels = []

    for image_file in image_files:
        key = normalize_tile_name(image_file)

        if key in label_lookup:
            paired.append(
                {
                    "image_path": str(image_file),
                    "label_path": str(label_lookup[key]),
                    "tile_name": key,
                    "has_label": True
                }
            )

        else:
            missing_labels.append(
                {
                    "image_path": str(image_file),
                    "label_path": None,
                    "tile_name": key,
                    "has_label": False
                }
            )

    return paired, missing_labels


def read_mask_statistics(mask_path):
    """
    Read mask and return statistics.

    Uses the shared raster reader so TIFF labels/masks can be handled
    even when Pillow cannot decode them directly.
    """
    mask_path = Path(mask_path)

    try:
        mask = read_mask_array(
            mask_path
        )

        mask = np.asarray(
            mask
        )

        positive_pixels = int(
            (mask > 0).sum()
        )

        total_pixels = int(
            mask.size
        )

        return {
            "readable": True,
            "positive_pixels": positive_pixels,
            "total_pixels": total_pixels,
            "positive_ratio": (
                0.0
                if total_pixels == 0
                else positive_pixels / total_pixels
            ),
            "is_positive": positive_pixels > 0
        }

    except Exception as e:
        return {
            "readable": False,
            "positive_pixels": 0,
            "total_pixels": 0,
            "positive_ratio": 0.0,
            "is_positive": False,
            "error": str(e)
        }


def create_empty_mask_for_image(image_path, output_folder):
    """
    Create an empty binary mask with exactly the same width/height
    as the corresponding image.

    Important:
    Older experiment folders may already contain generated empty masks
    from a different tile size (for example 512x512). Reusing such a
    mask for a 256x256 experiment causes mixed tensor sizes inside a
    DataLoader batch.

    Therefore:
    - If the generated mask does not exist, create it.
    - If it exists and its size matches the image, reuse it.
    - If it exists but its size is different (or it is unreadable),
      recreate it safely using the current image dimensions.
    """
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    mask_name = f"{image_path.stem}_empty_mask.png"
    output_mask = output_folder / mask_name

    # ---------------------------------------------------------
    # Resolve the CURRENT image dimensions using shared robust
    # raster metadata readers. This supports GeoTIFF chips that
    # Pillow cannot identify.
    # ---------------------------------------------------------
    width, height = get_raster_size(
        image_path
    )

    expected_size = (
        width,
        height,
    )

    # ---------------------------------------------------------
    # Reuse only when the existing generated mask belongs to
    # the current image size.
    # ---------------------------------------------------------
    if output_mask.exists():
        try:
            with Image.open(output_mask) as existing_mask:
                existing_size = existing_mask.size

            if existing_size == expected_size:
                return output_mask

            print(
                "Generated empty mask size mismatch detected. "
                f"Recreating: {output_mask.name} | "
                f"existing={existing_size} | expected={expected_size}"
            )

        except Exception:
            print(
                "Generated empty mask is unreadable. "
                f"Recreating: {output_mask.name}"
            )

    # ---------------------------------------------------------
    # Create/recreate a clean zero-valued mask.
    # ---------------------------------------------------------
    empty_mask = Image.new(
        "L",
        expected_size,
        0
    )

    empty_mask.save(output_mask)

    return output_mask


def save_csv(rows, csv_path):
    """
    Save list of dictionaries to CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def create_data_quality_report(dataset_path, output_folder):
    """
    Scan dataset and create data_quality.csv.

    Returns:
        data_quality_rows, paired_rows, missing_rows
    """
    dataset_path = Path(dataset_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files, label_files = scan_dataset_files(dataset_path)

    paired, missing_labels = pair_image_label_files(
        image_files=image_files,
        label_files=label_files
    )

    generated_empty_masks_folder = output_folder / "generated_empty_masks"
    generated_empty_masks_folder.mkdir(parents=True, exist_ok=True)

    data_quality_rows = []
    positive_rows = []
    background_rows = []

    unreadable_count = 0

    for pair in paired:
        stats = read_mask_statistics(pair["label_path"])

        if not stats.get("readable", False):
            unreadable_count += 1

        row = {
            "tile_name": pair["tile_name"],
            "image_path": pair["image_path"],
            "label_path": pair["label_path"],
            "has_original_label": True,
            "is_generated_empty_mask": False,
            "is_positive": stats["is_positive"],
            "positive_pixels": stats["positive_pixels"],
            "total_pixels": stats["total_pixels"],
            "positive_ratio": stats["positive_ratio"],
            "readable": stats["readable"]
        }

        data_quality_rows.append(row)

        if stats["is_positive"]:
            positive_rows.append(row)
        else:
            background_rows.append(row)

    for missing in missing_labels:
        empty_mask_path = create_empty_mask_for_image(
            image_path=missing["image_path"],
            output_folder=generated_empty_masks_folder
        )

        row = {
            "tile_name": missing["tile_name"],
            "image_path": missing["image_path"],
            "label_path": str(empty_mask_path),
            "has_original_label": False,
            "is_generated_empty_mask": True,
            "is_positive": False,
            "positive_pixels": 0,
            "total_pixels": 0,
            "positive_ratio": 0.0,
            "readable": True
        }

        data_quality_rows.append(row)
        background_rows.append(row)

    data_quality_csv = output_folder / "data_quality.csv"

    save_csv(
        data_quality_rows,
        data_quality_csv
    )

    summary = {
        "dataset_path": str(dataset_path),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "paired_count": len(paired),
        "missing_label_count": len(missing_labels),
        "positive_chips": len(positive_rows),
        "background_chips": len(background_rows),
        "unreadable_problem_chips": unreadable_count,
        "data_quality_csv": str(data_quality_csv),
        "generated_empty_masks_folder": str(generated_empty_masks_folder)
    }

    save_json(
        summary,
        output_folder / "data_quality_summary.json"
    )

    return data_quality_rows, positive_rows, background_rows, summary


def select_positive_and_background_samples(
    positive_rows,
    background_rows,
    background_ratio=0.5,
    seed=42
):
    """
    Select all positive samples and a ratio of background samples.

    background_count = positive_count * background_ratio
    """
    random.seed(int(seed))

    positive_rows = list(positive_rows)
    background_rows = list(background_rows)

    positive_count = len(positive_rows)

    desired_background_count = int(
        round(positive_count * float(background_ratio))
    )

    desired_background_count = min(
        desired_background_count,
        len(background_rows)
    )

    selected_background = random.sample(
        background_rows,
        desired_background_count
    ) if desired_background_count > 0 else []

    selected_rows = positive_rows + selected_background

    random.shuffle(selected_rows)

    return selected_rows, selected_background


def resolve_train_validation_split(
    general_params=None,
    train_percent=None,
    validation_percent=None,
    validation_ratio=None,
    seed=42
):
    """
    Resolve the Train / Validation split while preserving backward compatibility.

    New preferred configuration:
        general_params["data_split"] = {
            "train_percent": 80,
            "validation_percent": 20,
            "seed": 42
        }

    Backward-compatible configuration:
        training.validation_ratio
        dataset.validation_ratio
        general_params.validation_ratio

    Rules:
    - Train + Validation must equal 100%.
    - Test data is NOT part of this percentage.
    - If only one percentage is supplied, the other is inferred.
    - If no new percentage configuration exists, the old validation_ratio
      behavior is preserved.
    """
    if general_params is None:
        general_params = {}

    data_split_params = general_params.get("data_split", {})
    training_params = general_params.get("training", {})
    data_params = general_params.get("data", {})
    dataset_params = general_params.get("dataset", {})

    # ---------------------------------------------------------
    # Resolve seed
    # ---------------------------------------------------------
    resolved_seed = (
        data_split_params.get("seed")
        if data_split_params.get("seed") is not None
        else training_params.get("seed")
        if training_params.get("seed") is not None
        else data_params.get("seed")
        if data_params.get("seed") is not None
        else dataset_params.get("seed")
        if dataset_params.get("seed") is not None
        else general_params.get("seed")
        if general_params.get("seed") is not None
        else seed
    )

    resolved_seed = int(resolved_seed)

    # ---------------------------------------------------------
    # New preferred percentage configuration
    # ---------------------------------------------------------
    if train_percent is None:
        train_percent = data_split_params.get("train_percent")

    if validation_percent is None:
        validation_percent = data_split_params.get("validation_percent")

    if train_percent is not None or validation_percent is not None:
        if train_percent is None:
            train_percent = 100.0 - float(validation_percent)

        if validation_percent is None:
            validation_percent = 100.0 - float(train_percent)

        train_percent = float(train_percent)
        validation_percent = float(validation_percent)

        if train_percent <= 0.0 or train_percent >= 100.0:
            raise ValueError(
                f"train_percent must be between 0 and 100. "
                f"Received: {train_percent}"
            )

        if validation_percent <= 0.0 or validation_percent >= 100.0:
            raise ValueError(
                f"validation_percent must be between 0 and 100. "
                f"Received: {validation_percent}"
            )

        if not np.isclose(
            train_percent + validation_percent,
            100.0,
            atol=1e-8
        ):
            raise ValueError(
                "Train and Validation percentages must sum to 100. "
                f"Received: train={train_percent}, "
                f"validation={validation_percent}"
            )

        resolved_validation_ratio = validation_percent / 100.0

        return {
            "train_percent": train_percent,
            "validation_percent": validation_percent,
            "train_ratio": train_percent / 100.0,
            "validation_ratio": resolved_validation_ratio,
            "seed": resolved_seed,
            "source": "data_split_percentages"
        }

    # ---------------------------------------------------------
    # Backward-compatible validation ratio
    # ---------------------------------------------------------
    if validation_ratio is None:
        validation_ratio = (
            training_params.get("validation_ratio")
            if training_params.get("validation_ratio") is not None
            else data_params.get("validation_ratio")
            if data_params.get("validation_ratio") is not None
            else dataset_params.get("validation_ratio")
            if dataset_params.get("validation_ratio") is not None
            else general_params.get("validation_ratio")
            if general_params.get("validation_ratio") is not None
            else 0.2
        )

    validation_ratio = float(validation_ratio)

    # Allow legacy callers to accidentally supply 20 instead of 0.20.
    if validation_ratio > 1.0:
        validation_ratio = validation_ratio / 100.0

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1 "
            "(or between 0 and 100 if supplied as a percentage). "
            f"Received: {validation_ratio}"
        )

    validation_percent = validation_ratio * 100.0
    train_percent = 100.0 - validation_percent

    return {
        "train_percent": train_percent,
        "validation_percent": validation_percent,
        "train_ratio": train_percent / 100.0,
        "validation_ratio": validation_ratio,
        "seed": resolved_seed,
        "source": "legacy_validation_ratio"
    }


def split_train_val(
    rows,
    validation_ratio=0.2,
    seed=42,
    train_percent=None,
    validation_percent=None
):
    """
    Split selected rows into Train / Validation.

    Backward-compatible usage:
        split_train_val(
            rows,
            validation_ratio=0.2,
            seed=42
        )

    New usage:
        split_train_val(
            rows,
            train_percent=80,
            validation_percent=20,
            seed=42
        )

    Important:
    - Test data is deliberately NOT created here.
    - Test imagery remains a separate source-specific geographic test area.
    """
    rows = list(rows)

    if len(rows) == 0:
        return [], []

    if train_percent is not None or validation_percent is not None:
        split_info = resolve_train_validation_split(
            general_params={},
            train_percent=train_percent,
            validation_percent=validation_percent,
            validation_ratio=validation_ratio,
            seed=seed
        )

        validation_ratio = split_info["validation_ratio"]
        seed = split_info["seed"]

    validation_ratio = float(validation_ratio)

    if validation_ratio > 1.0:
        validation_ratio = validation_ratio / 100.0

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError(
            f"validation_ratio must be between 0 and 1. "
            f"Received: {validation_ratio}"
        )

    rng = random.Random(int(seed))

    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)

    if len(shuffled_rows) == 1:
        return [], shuffled_rows

    val_count = int(
        round(
            len(shuffled_rows)
            * validation_ratio
        )
    )

    # Keep both Train and Validation non-empty whenever possible.
    val_count = max(
        1,
        min(
            val_count,
            len(shuffled_rows) - 1
        )
    )

    val_rows = shuffled_rows[:val_count]
    train_rows = shuffled_rows[val_count:]

    return train_rows, val_rows


def save_manifest(rows, manifest_csv):
    """
    Save manifest CSV with image_path and label_path.

    The existing manifest schema is intentionally preserved so the old
    training pipeline can still use these files.
    """
    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for row in rows:
        manifest_rows.append(
            {
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "tile_name": row.get("tile_name", ""),
                "is_positive": row.get("is_positive", False),
                "has_original_label": row.get("has_original_label", False),
                "is_generated_empty_mask": row.get(
                    "is_generated_empty_mask",
                    False
                )
            }
        )

    save_csv(
        manifest_rows,
        manifest_csv
    )

    return manifest_csv


def create_train_val_manifests(
    dataset_info,
    output_folder,
    general_params=None,
    validation_ratio=None,
    background_ratio=None,
    seed=42,
    train_percent=None,
    validation_percent=None,
    **kwargs
):
    """
    Create train_manifest.csv and val_manifest.csv.

    New preferred split control:
        general_params["data_split"] = {
            "train_percent": 80,
            "validation_percent": 20,
            "seed": 42
        }

    Backward compatibility:
    - Existing validation_ratio configuration still works.
    - Existing callers do not need to change immediately.
    - Existing manifest columns are preserved.
    - Existing positive/background selection behavior is preserved.

    Test data is NOT split from these rows.
    The final test is a separate source-specific geographic area.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset_info, dict):
        dataset_path = (
            dataset_info.get("path")
            or dataset_info.get("dataset_path")
            or dataset_info.get("folder")
            or dataset_info.get("root")
        )
    else:
        dataset_path = dataset_info

    if dataset_path is None:
        raise ValueError("Dataset path could not be resolved.")

    if general_params is None:
        general_params = {}

    training_params = general_params.get("training", {})
    data_params = general_params.get("data", {})
    dataset_params = general_params.get("dataset", {})

    # ---------------------------------------------------------
    # Resolve Train / Validation percentages and seed
    # ---------------------------------------------------------
    split_info = resolve_train_validation_split(
        general_params=general_params,
        train_percent=train_percent,
        validation_percent=validation_percent,
        validation_ratio=validation_ratio,
        seed=seed
    )

    validation_ratio = split_info["validation_ratio"]
    train_percent = split_info["train_percent"]
    validation_percent = split_info["validation_percent"]
    seed = split_info["seed"]

    # ---------------------------------------------------------
    # Preserve old background sampling behavior
    # ---------------------------------------------------------
    if background_ratio is None:
        background_ratio = (
            training_params.get("background_ratio")
            if training_params.get("background_ratio") is not None
            else data_params.get("background_ratio")
            if data_params.get("background_ratio") is not None
            else dataset_params.get("background_ratio")
            if dataset_params.get("background_ratio") is not None
            else general_params.get("background_ratio")
            if general_params.get("background_ratio") is not None
            else 0.5
        )

    background_ratio = float(background_ratio)

    # ---------------------------------------------------------
    # Dataset quality
    # ---------------------------------------------------------
    (
        data_quality_rows,
        positive_rows,
        background_rows,
        quality_summary
    ) = create_data_quality_report(
        dataset_path=dataset_path,
        output_folder=output_folder
    )

    # ---------------------------------------------------------
    # Existing positive/background sampling
    # ---------------------------------------------------------
    selected_rows, selected_background = (
        select_positive_and_background_samples(
            positive_rows=positive_rows,
            background_rows=background_rows,
            background_ratio=background_ratio,
            seed=seed
        )
    )

    # ---------------------------------------------------------
    # Controlled Train / Validation split
    # ---------------------------------------------------------
    train_rows, val_rows = split_train_val(
        rows=selected_rows,
        validation_ratio=validation_ratio,
        seed=seed
    )

    train_manifest_csv = output_folder / "train_manifest.csv"
    val_manifest_csv = output_folder / "val_manifest.csv"

    save_manifest(
        train_rows,
        train_manifest_csv
    )

    save_manifest(
        val_rows,
        val_manifest_csv
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    manifest_summary = {
        **quality_summary,
        "split_type": "train_validation_only",
        "split_method": "deterministic_random",
        "split_config_source": split_info["source"],
        "train_percent": float(train_percent),
        "validation_percent": float(validation_percent),
        "train_ratio": float(split_info["train_ratio"]),
        "validation_ratio": float(validation_ratio),
        "seed": int(seed),
        "test_included_in_split": False,
        "selected_positive_count": len(positive_rows),
        "selected_background_count": len(selected_background),
        "total_selected_count": len(selected_rows),
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "background_ratio": float(background_ratio),
        "train_manifest_csv": str(train_manifest_csv),
        "val_manifest_csv": str(val_manifest_csv)
    }

    save_json(
        manifest_summary,
        output_folder / "manifest_summary.json"
    )

    # Dedicated split metadata for the future UI / final architecture.
    split_summary = {
        "train_percent": float(train_percent),
        "validation_percent": float(validation_percent),
        "train_ratio": float(split_info["train_ratio"]),
        "validation_ratio": float(validation_ratio),
        "seed": int(seed),
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "total_selected_count": len(selected_rows),
        "test_included_in_split": False,
        "test_policy": (
            "Use the separate source-specific geographic test area "
            "only after model and settings are frozen."
        ),
        "train_manifest_csv": str(train_manifest_csv),
        "val_manifest_csv": str(val_manifest_csv)
    }

    save_json(
        split_summary,
        output_folder / "split_summary.json"
    )

    print("\n========== DATASET QUALITY ==========")
    print(f"Image Count: {quality_summary['image_count']}")
    print(f"Label Count: {quality_summary['label_count']}")
    print(f"Paired Count: {quality_summary['paired_count']}")
    print(f"Missing Label Count: {quality_summary['missing_label_count']}")
    print(f"Positive Chips: {quality_summary['positive_chips']}")
    print(f"Background Chips: {quality_summary['background_chips']}")
    print(
        "Unreadable/Problem Chips: "
        f"{quality_summary['unreadable_problem_chips']}"
    )
    print(
        "Selected Positive Count: "
        f"{manifest_summary['selected_positive_count']}"
    )
    print(
        "Selected Background Count: "
        f"{manifest_summary['selected_background_count']}"
    )
    print(
        "Total Selected Count: "
        f"{manifest_summary['total_selected_count']}"
    )
    print("-------------------------------------")
    print(f"Training Percentage: {train_percent:.2f}%")
    print(f"Validation Percentage: {validation_percent:.2f}%")
    print(f"Split Seed: {seed}")
    print(f"Train Count: {manifest_summary['train_count']}")
    print(f"Validation Count: {manifest_summary['validation_count']}")
    print("Test Included in Split: NO")
    print("=====================================\n")

    return {
        "train_manifest_csv": str(train_manifest_csv),
        "val_manifest_csv": str(val_manifest_csv),
        "data_quality_csv": str(output_folder / "data_quality.csv"),
        "manifest_summary_json": str(output_folder / "manifest_summary.json"),
        "split_summary_json": str(output_folder / "split_summary.json"),
        "summary": manifest_summary
    }
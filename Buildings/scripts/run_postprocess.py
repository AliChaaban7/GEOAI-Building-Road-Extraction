"""
run_postprocess.py

Shared validation-side post-processing optimization for semantic
Building Extraction models.

Supported semantic models:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA

Architecture
------------
Post-processing parameters are selected ONLY on validation data.

    TRAIN
      ↓
    best checkpoint
      ↓
    VALIDATION threshold search
      ↓
    frozen-candidate probability threshold
      ↓
    VALIDATION post-processing search
      ↓
    best minimum connected-component area
      ↓
    FREEZE SETTINGS
      ↓
    EXTERNAL FINAL TEST

The source-specific external test area is NOT accessed by this script.

SAM-LoRA
--------
SAM-LoRA is reconstructed from:

    official SAM checkpoint
        +
    sam_lora.json
        +
    best_model.pth

After reconstruction, the same shared validation/post-processing logic
is used as for U-Net and DeepLab.

Selection metric
----------------
Aggregate validation pixel IoU.

The selected minimum area is expressed in square meters and later
applied unchanged to final-test polygons.
"""

from pathlib import Path
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scipy import ndimage
except Exception as exc:
    raise ImportError(
        "scipy is required for validation post-processing. "
        "Install/enable scipy in the ArcGIS Pro Python environment."
    ) from exc


# ============================================================
# PROJECT ROOTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# MODULE IMPORTS
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_model_config,
    get_backbone_info,
    get_automatic_batch_size,
    create_experiment_output_folder,
    normalize_model_type,
    save_json,
)

from src.models.factory import build_model

from src.inference import (
    load_model_checkpoint,
    get_model_output,
    normalize_logits_shape,
)

from src.train import (
    BuildingSegmentationDataset,
    unpack_batch,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Select semantic building post-processing parameters "
            "using ONLY the validation split."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROOT / "config" / "experiment.json"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--validation-manifest",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--threshold-summary",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--area-thresholds",
        type=float,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--allow-smoke-checkpoint",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json_file(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nJSON file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_json_if_exists(path):

    path = Path(path)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_device(requested=None):

    if requested is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(requested)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return device


def resolve_built_model(build_result):

    if isinstance(build_result, (tuple, list)):

        if len(build_result) == 0:
            raise RuntimeError(
                "build_model() returned an empty tuple/list."
            )

        model = build_result[0]

    else:
        model = build_result

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "build_model() did not return torch.nn.Module."
        )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = resolve_device()
        model = model.to(device)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return model, device, trainable_params


def resolve_requested_epochs(training_summary):

    if not isinstance(training_summary, dict):
        return None

    for key in (
        "epochs_requested",
        "epochs",
        "requested_epochs",
    ):
        value = training_summary.get(key)

        if value is not None:
            return int(value)

    return None


def check_checkpoint_is_not_smoke_test(
    output_folder,
    checkpoint_path,
    allow_smoke_checkpoint=False,
):

    output_folder = Path(output_folder)
    checkpoint_path = Path(checkpoint_path)

    default_checkpoint = (
        output_folder / "best_model.pth"
    )

    try:
        same_checkpoint = (
            checkpoint_path.resolve()
            == default_checkpoint.resolve()
        )
    except Exception:
        same_checkpoint = (
            str(checkpoint_path)
            == str(default_checkpoint)
        )

    if not same_checkpoint:
        return

    training_summary = load_json_if_exists(
        output_folder / "training_summary.json"
    )

    requested_epochs = resolve_requested_epochs(
        training_summary
    )

    if requested_epochs is None:
        return

    if (
        requested_epochs <= 2
        and not allow_smoke_checkpoint
    ):
        raise RuntimeError(
            "\nThe current best_model.pth appears to come from "
            f"a {requested_epochs}-epoch smoke-test run.\n"
            "Do not select final post-processing settings on it."
        )


def validate_threshold_summary(summary):

    if not isinstance(summary, dict):
        raise TypeError(
            "Threshold summary must contain a JSON object."
        )

    if bool(
        summary.get(
            "external_test_area_used",
            False,
        )
    ):
        raise RuntimeError(
            "Threshold summary indicates external test-area use."
        )

    selected_on = str(
        summary.get(
            "selection_dataset",
            "",
        )
    ).lower()

    if (
        selected_on
        and "validation" not in selected_on
    ):
        raise RuntimeError(
            "Threshold was not selected on validation data."
        )


# ============================================================
# PIXEL SIZE
# ============================================================

def get_first_validation_image_path(manifest_csv):

    manifest_csv = Path(manifest_csv)

    with manifest_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)
        first_row = next(reader, None)

    if first_row is None:
        raise RuntimeError(
            f"Validation manifest is empty:\n{manifest_csv}"
        )

    image_path = first_row.get(
        "image_path"
    )

    if not image_path:
        raise KeyError(
            "Validation manifest does not contain image_path."
        )

    return Path(image_path)


def infer_pixel_size_m(raster_path):

    raster_path = Path(raster_path)

    try:
        import arcpy

        if not arcpy.Exists(str(raster_path)):
            return None, (
                "ArcPy does not recognize the validation chip "
                "as a georeferenced raster."
            )

        raster = arcpy.Raster(str(raster_path))

        cell_width = abs(float(raster.meanCellWidth))
        cell_height = abs(float(raster.meanCellHeight))

        spatial_reference = raster.spatialReference

        linear_unit_name = str(
            getattr(
                spatial_reference,
                "linearUnitName",
                "",
            )
            or ""
        ).lower()

        if not linear_unit_name:
            return None, (
                "Raster cell size exists but spatial reference "
                "does not expose a linear unit."
            )

        if (
            "meter" in linear_unit_name
            or "metre" in linear_unit_name
        ):
            pixel_size_m = float(
                np.sqrt(
                    cell_width * cell_height
                )
            )

            return pixel_size_m, (
                "inferred from ArcPy raster cell size"
            )

        if (
            "foot" in linear_unit_name
            or "feet" in linear_unit_name
        ):
            pixel_size_m = float(
                np.sqrt(
                    cell_width * cell_height
                )
                * 0.3048
            )

            return pixel_size_m, (
                f"converted from {linear_unit_name}"
            )

        return None, (
            "Unsupported/non-metric raster unit: "
            f"{linear_unit_name}"
        )

    except Exception as exc:
        return None, (
            "Could not infer pixel size with ArcPy: "
            f"{type(exc).__name__}: {exc}"
        )


def resolve_pixel_size_m(
    validation_manifest,
    cli_pixel_size_m=None,
):

    if cli_pixel_size_m is not None:

        value = float(cli_pixel_size_m)

        if value <= 0:
            raise ValueError(
                "--pixel-size-m must be > 0."
            )

        return value, "provided by --pixel-size-m"

    first_image = get_first_validation_image_path(
        validation_manifest
    )

    value, source = infer_pixel_size_m(
        first_image
    )

    if value is None:
        raise RuntimeError(
            "\nCould not determine validation pixel size.\n"
            f"First validation image:\n{first_image}\n"
            f"Reason: {source}\n"
            "Provide the known source resolution, e.g. "
            "--pixel-size-m 0.30"
        )

    return float(value), source


# ============================================================
# POST-PROCESSING
# ============================================================

def remove_small_components(
    binary_mask,
    min_area_m2,
    pixel_area_m2,
):

    binary_mask = np.asarray(
        binary_mask,
        dtype=bool,
    )

    min_area_m2 = float(
        min_area_m2
    )

    if min_area_m2 <= 0.0:
        return binary_mask.copy()

    min_pixels = max(
        int(
            np.ceil(
                min_area_m2
                / float(pixel_area_m2)
            )
        ),
        1,
    )

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    labeled, component_count = ndimage.label(
        binary_mask,
        structure=structure,
    )

    if component_count == 0:
        return binary_mask.copy()

    component_sizes = np.bincount(
        labeled.ravel()
    )

    keep_labels = np.where(
        component_sizes >= min_pixels
    )[0]

    keep_labels = keep_labels[
        keep_labels != 0
    ]

    if len(keep_labels) == 0:
        return np.zeros_like(
            binary_mask,
            dtype=bool,
        )

    return np.isin(
        labeled,
        keep_labels,
    )


def get_area_thresholds(args, general_params):

    if args.area_thresholds is not None:
        values = args.area_thresholds
    else:
        values = (
            general_params
            .get("postprocessing", {})
            .get(
                "area_thresholds_m2",
                [
                    0, 5, 10, 15, 20, 25,
                    30, 35, 40, 50, 75, 100,
                ],
            )
        )

    area_thresholds = sorted(
        {
            float(value)
            for value in values
        }
    )

    if not area_thresholds:
        raise ValueError(
            "No area thresholds were provided."
        )

    if any(
        value < 0
        for value in area_thresholds
    ):
        raise ValueError(
            "Area thresholds must be >= 0 m²."
        )

    return area_thresholds


# ============================================================
# VALIDATION SEARCH
# ============================================================

@torch.no_grad()
def search_postprocessing_on_validation(
    model,
    val_loader,
    device,
    probability_threshold,
    area_thresholds_m2,
    pixel_area_m2,
):

    model.eval()

    totals = {
        area_threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "pred_components_before": 0,
            "pred_components_after": 0,
        }
        for area_threshold in area_thresholds_m2
    }

    total_batches = len(val_loader)

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    for batch_index, batch in enumerate(
        val_loader,
        start=1,
    ):

        images, masks = unpack_batch(batch)

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        logits = get_model_output(
            outputs
        )

        logits = normalize_logits_shape(
            logits,
            target_height=int(
                masks.shape[-2]
            ),
            target_width=int(
                masks.shape[-1]
            ),
        )

        probabilities = torch.sigmoid(
            logits
        )

        raw_predictions = (
            probabilities
            >= float(probability_threshold)
        ).detach().cpu().numpy()

        gt_masks = (
            masks >= 0.5
        ).detach().cpu().numpy()

        for sample_index in range(
            raw_predictions.shape[0]
        ):

            pred_mask = (
                raw_predictions[
                    sample_index,
                    0,
                ]
            ).astype(bool)

            gt_mask = (
                gt_masks[
                    sample_index,
                    0,
                ]
            ).astype(bool)

            _, raw_component_count = (
                ndimage.label(
                    pred_mask,
                    structure=structure,
                )
            )

            for area_threshold in area_thresholds_m2:

                cleaned = remove_small_components(
                    binary_mask=pred_mask,
                    min_area_m2=area_threshold,
                    pixel_area_m2=pixel_area_m2,
                )

                _, cleaned_component_count = (
                    ndimage.label(
                        cleaned,
                        structure=structure,
                    )
                )

                tp = np.logical_and(
                    cleaned,
                    gt_mask,
                ).sum()

                fp = np.logical_and(
                    cleaned,
                    ~gt_mask,
                ).sum()

                fn = np.logical_and(
                    ~cleaned,
                    gt_mask,
                ).sum()

                tn = np.logical_and(
                    ~cleaned,
                    ~gt_mask,
                ).sum()

                totals[area_threshold]["tp"] += int(tp)
                totals[area_threshold]["fp"] += int(fp)
                totals[area_threshold]["fn"] += int(fn)
                totals[area_threshold]["tn"] += int(tn)

                totals[
                    area_threshold
                ][
                    "pred_components_before"
                ] += int(raw_component_count)

                totals[
                    area_threshold
                ][
                    "pred_components_after"
                ] += int(cleaned_component_count)

        if (
            batch_index % 10 == 0
            or batch_index == total_batches
        ):
            print(
                "Processed validation batches:",
                f"{batch_index}/{total_batches}",
            )

    rows = []
    eps = 1e-7

    for area_threshold in area_thresholds_m2:

        values = totals[area_threshold]

        tp = float(values["tp"])
        fp = float(values["fp"])
        fn = float(values["fn"])
        tn = float(values["tn"])

        iou = tp / (
            tp + fp + fn + eps
        )

        precision = tp / (
            tp + fp + eps
        )

        recall = tp / (
            tp + fn + eps
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
                + eps
            )
        )

        min_component_pixels = (
            0
            if area_threshold <= 0.0
            else int(
                np.ceil(
                    area_threshold
                    / pixel_area_m2
                )
            )
        )

        rows.append(
            {
                "min_area_m2":
                    float(area_threshold),

                "min_component_pixels":
                    int(min_component_pixels),

                "iou":
                    float(iou),

                "iou_percent":
                    float(iou * 100.0),

                "precision":
                    float(precision),

                "precision_percent":
                    float(precision * 100.0),

                "recall":
                    float(recall),

                "recall_percent":
                    float(recall * 100.0),

                "f1":
                    float(f1),

                "f1_percent":
                    float(f1 * 100.0),

                "tp_pixels":
                    int(tp),

                "fp_pixels":
                    int(fp),

                "fn_pixels":
                    int(fn),

                "tn_pixels":
                    int(tn),

                "pred_components_before":
                    int(
                        values[
                            "pred_components_before"
                        ]
                    ),

                "pred_components_after":
                    int(
                        values[
                            "pred_components_after"
                        ]
                    ),
            }
        )

    best_row = max(
        rows,
        key=lambda row: (
            row["iou"],
            row["f1"],
            -row["min_area_m2"],
        ),
    )

    return rows, best_row


# ============================================================
# OUTPUT
# ============================================================

def save_results_csv(rows, path):

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "min_area_m2",
        "min_component_pixels",
        "iou",
        "iou_percent",
        "precision",
        "precision_percent",
        "recall",
        "recall_percent",
        "f1",
        "f1_percent",
        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
        "tn_pixels",
        "pred_components_before",
        "pred_components_after",
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

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fieldnames
                }
            )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    experiment_config = load_experiment_config(
        Path(args.config)
    )

    general_params = load_general_params(
        ROOT
    )

    model_type = normalize_model_type(
        experiment_config["model_type"]
    )

    if model_type not in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:
        raise ValueError(
            "run_postprocess.py supports semantic models only: "
            "U-Net, DeepLabV3, SAM-LoRA."
        )

    output_folder = Path(
        create_experiment_output_folder(
            experiment_config=experiment_config,
            root=ROOT,
        )
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else output_folder / "best_model.pth"
    )

    val_manifest_csv = (
        Path(args.validation_manifest)
        if args.validation_manifest is not None
        else output_folder / "val_manifest.csv"
    )

    threshold_summary_path = (
        Path(args.threshold_summary)
        if args.threshold_summary is not None
        else (
            output_folder
            / "validation_threshold_summary.json"
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint_path}"
        )

    if not val_manifest_csv.exists():
        raise FileNotFoundError(
            "\nValidation manifest not found:\n"
            f"{val_manifest_csv}"
        )

    threshold_summary = load_json_file(
        threshold_summary_path
    )

    validate_threshold_summary(
        threshold_summary
    )

    best_threshold = float(
        threshold_summary[
            "best_threshold"
        ]
    )

    if not (
        0.0 <= best_threshold <= 1.0
    ):
        raise ValueError(
            "best_threshold must be between 0 and 1."
        )

    check_checkpoint_is_not_smoke_test(
        output_folder=output_folder,
        checkpoint_path=checkpoint_path,
        allow_smoke_checkpoint=(
            args.allow_smoke_checkpoint
        ),
    )

    (
        pixel_size_m,
        pixel_size_source,
    ) = resolve_pixel_size_m(
        validation_manifest=val_manifest_csv,
        cli_pixel_size_m=args.pixel_size_m,
    )

    pixel_area_m2 = float(
        pixel_size_m * pixel_size_m
    )

    area_thresholds = get_area_thresholds(
        args=args,
        general_params=general_params,
    )

    model_config = load_model_config(
        model_type,
        ROOT,
    )

    requested_device = resolve_device(
        args.device
    )

    checkpoint_epoch = None
    checkpoint_val_iou = None
    official_sam_checkpoint = None
    checkpoint_format = None

    if model_type == "sam_lora":

        from src.models.sam_lora.inference import (
            load_trained_sam_lora,
        )

        from src.models.sam_lora.factory import (
            resolve_sam_lora_batch_size,
        )

        sam_config_path = (
            ROOT
            / "config"
            / "models"
            / "sam_lora.json"
        )

        model, load_info = load_trained_sam_lora(
            config_path=sam_config_path,
            trained_checkpoint_path=checkpoint_path,
            device=requested_device,
            verbose=True,
        )

        device = next(
            model.parameters()
        ).device

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        checkpoint_epoch = (
            load_info.trained_epoch
        )

        checkpoint_format = (
            load_info.checkpoint_format
        )

        official_sam_checkpoint = (
            load_info.pretrained_sam_checkpoint_path
        )

        validation_metrics = (
            load_info.validation_metrics
            or {}
        )

        checkpoint_val_iou = (
            validation_metrics.get(
                "iou",
                validation_metrics.get(
                    "val_iou",
                    validation_metrics.get(
                        "validation_iou",
                        None,
                    ),
                ),
            )
        )

        if args.batch_size is not None:
            batch_size = int(
                args.batch_size
            )
        else:
            batch_size = resolve_sam_lora_batch_size(
                config=model_config,
                tile_size=int(
                    experiment_config.get(
                        "tile_size",
                        256,
                    )
                ),
            )

        model_loader = (
            "sam_lora_reconstruction"
        )

    else:

        backbone_info = get_backbone_info(
            experiment_config=experiment_config,
            model_config=model_config,
        )

        build_result = build_model(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
            move_to_device=True,
        )

        (
            model,
            device,
            trainable_params,
        ) = resolve_built_model(
            build_result
        )

        if device != requested_device:
            model = model.to(
                requested_device
            )
            device = requested_device

        model, checkpoint = load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        if isinstance(checkpoint, dict):
            checkpoint_epoch = checkpoint.get(
                "epoch"
            )
            checkpoint_val_iou = checkpoint.get(
                "best_val_iou",
                checkpoint.get(
                    "best_validation_iou",
                    None,
                ),
            )

        if args.batch_size is not None:
            batch_size = int(
                args.batch_size
            )
        else:
            batch_size = get_automatic_batch_size(
                experiment_config=experiment_config,
                backbone_info=backbone_info,
                model_config=model_config,
            )

        model_loader = (
            "standard_shared_checkpoint_loader"
        )

    model.eval()

    if batch_size <= 0:
        raise ValueError(
            "Validation batch size must be > 0."
        )

    val_dataset = BuildingSegmentationDataset(
        manifest_csv=val_manifest_csv,
        augmentation=None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(
            general_params
            .get("training", {})
            .get("num_workers", 0)
        ),
        pin_memory=bool(
            general_params
            .get("training", {})
            .get("pin_memory", True)
        ),
        drop_last=False,
    )

    postprocess_folder = (
        output_folder
        / "validation_postprocessing"
    )

    postprocess_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_csv = (
        postprocess_folder
        / "validation_postprocess_search.csv"
    )

    detailed_summary_json = (
        postprocess_folder
        / "validation_postprocess_summary.json"
    )

    final_summary_json = (
        output_folder
        / "validation_postprocess_summary.json"
    )

    display_backbone = (
        experiment_config.get(
            "backbone",
            model_config.get(
                "default_backbone",
                "unknown",
            ),
        )
    )

    print()
    print("=" * 76)
    print("VALIDATION POST-PROCESSING OPTIMIZATION")
    print("=" * 76)
    print(
        "Experiment ID:",
        experiment_config["experiment_id"],
    )
    print("Model Type:", model_type)
    print("Backbone:", display_backbone)
    print(
        "Source Type:",
        experiment_config["source_type"],
    )
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint Epoch:", checkpoint_epoch)
    print(
        "Checkpoint Validation IoU:",
        checkpoint_val_iou,
    )
    print("Model Loader:", model_loader)

    if model_type == "sam_lora":
        print(
            "Official SAM Checkpoint:",
            official_sam_checkpoint,
        )
        print(
            "SAM-LoRA Checkpoint Format:",
            checkpoint_format,
        )

    print("Device:", device)
    print(
        "Trainable Parameters:",
        f"{trainable_params:,}",
    )
    print(
        "Validation Manifest:",
        val_manifest_csv,
    )
    print(
        "Validation Samples:",
        len(val_dataset),
    )
    print(
        "Selected Validation Threshold:",
        f"{best_threshold:.4f}",
    )
    print(
        "Pixel Size:",
        f"{pixel_size_m:.6f} m",
    )
    print(
        "Area Thresholds (m²):",
        area_thresholds,
    )
    print("-" * 76)
    print("Selection Dataset: VALIDATION ONLY")
    print(
        "Selection Metric: aggregate validation pixel IoU"
    )
    print("External Test Area Used: NO")
    print("=" * 76)
    print()

    rows, best_post_row = (
        search_postprocessing_on_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            probability_threshold=best_threshold,
            area_thresholds_m2=area_thresholds,
            pixel_area_m2=pixel_area_m2,
        )
    )

    save_results_csv(
        rows=rows,
        path=results_csv,
    )

    area_zero_row = next(
        (
            row
            for row in rows
            if abs(
                float(row["min_area_m2"])
            ) < 1e-12
        ),
        None,
    )

    baseline_iou = (
        float(area_zero_row["iou_percent"])
        if area_zero_row is not None
        else None
    )

    improvement_pp = (
        float(
            best_post_row["iou_percent"]
            - baseline_iou
        )
        if baseline_iou is not None
        else None
    )

    final_summary = {
        "experiment_id":
            experiment_config["experiment_id"],

        "model_type":
            model_type,

        "backbone":
            display_backbone,

        "source_type":
            experiment_config["source_type"],

        "tile_size":
            int(experiment_config["tile_size"]),

        "validation_manifest":
            str(val_manifest_csv),

        "validation_samples":
            int(len(val_dataset)),

        "checkpoint_path":
            str(checkpoint_path),

        "checkpoint_epoch":
            checkpoint_epoch,

        "checkpoint_validation_iou":
            checkpoint_val_iou,

        "model_loader":
            model_loader,

        "official_sam_checkpoint":
            (
                str(official_sam_checkpoint)
                if official_sam_checkpoint
                is not None
                else None
            ),

        "sam_lora_checkpoint_format":
            checkpoint_format,

        "threshold_summary_path":
            str(threshold_summary_path),

        "best_threshold":
            float(best_threshold),

        "postprocessing_method":
            "minimum_connected_component_area",

        "best_min_area_m2":
            float(
                best_post_row["min_area_m2"]
            ),

        "best_min_component_pixels":
            int(
                best_post_row[
                    "min_component_pixels"
                ]
            ),

        "best_validation_iou":
            float(
                best_post_row["iou"]
            ),

        "best_validation_iou_percent":
            float(
                best_post_row[
                    "iou_percent"
                ]
            ),

        "best_precision":
            float(
                best_post_row["precision"]
            ),

        "best_precision_percent":
            float(
                best_post_row[
                    "precision_percent"
                ]
            ),

        "best_recall":
            float(
                best_post_row["recall"]
            ),

        "best_recall_percent":
            float(
                best_post_row[
                    "recall_percent"
                ]
            ),

        "best_f1":
            float(
                best_post_row["f1"]
            ),

        "best_f1_percent":
            float(
                best_post_row[
                    "f1_percent"
                ]
            ),

        "baseline_no_area_filter_iou_percent":
            baseline_iou,

        "postprocessing_improvement_pp":
            improvement_pp,

        "pixel_size_m":
            float(pixel_size_m),

        "pixel_area_m2":
            float(pixel_area_m2),

        "pixel_size_source":
            pixel_size_source,

        "area_thresholds_tested_m2":
            area_thresholds,

        "selection_dataset":
            "validation_manifest",

        "selection_metric":
            "aggregate_validation_pixel_iou",

        "external_test_area_used":
            False,

        "settings_ready_to_freeze":
            True,

        "results_csv":
            str(results_csv),

        "methodology_note":
            (
                "Probability threshold and minimum-area post-processing "
                "were selected exclusively on held-out validation chips. "
                "The external source-specific test area was not accessed. "
                "The selected minimum area is frozen in square meters and "
                "applied unchanged during final testing."
            ),
    }

    save_json(
        final_summary,
        detailed_summary_json,
    )

    save_json(
        final_summary,
        final_summary_json,
    )

    print()
    print("=" * 76)
    print(
        "VALIDATION POST-PROCESSING OPTIMIZATION FINISHED"
    )
    print("=" * 76)
    print(
        "Frozen-Candidate Threshold:",
        f"{best_threshold:.4f}",
    )
    print(
        "Best Minimum Area:",
        f"{best_post_row['min_area_m2']:.2f} m²",
    )
    print(
        "Best Validation IoU:",
        f"{best_post_row['iou_percent']:.2f}%",
    )
    print(
        "Precision:",
        f"{best_post_row['precision_percent']:.2f}%",
    )
    print(
        "Recall:",
        f"{best_post_row['recall_percent']:.2f}%",
    )
    print(
        "F1:",
        f"{best_post_row['f1_percent']:.2f}%",
    )
    print("External Test Area Used: NO")
    print(
        "Settings Ready to Freeze: YES"
    )
    print(
        "Validation Postprocess Summary:",
        final_summary_json,
    )
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()

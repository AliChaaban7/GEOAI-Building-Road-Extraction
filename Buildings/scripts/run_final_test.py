"""
run_final_test.py

Run the one-time final source-specific Building Extraction test using
ONLY settings frozen before the test area is accessed.

Supported semantic models:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA

Protocol
--------
    TRAIN
      ↓
    VALIDATION
      ├─ best checkpoint
      ├─ threshold
      └─ post-processing
      ↓
    freeze_final_settings.py
      ↓
    final_settings.json
      ↓
    SAME independent source-specific test zone
      ↓
    full-scene inference ONCE
      ↓
    apply frozen minimum-area filter ONCE
      ↓
    strict polygon IoU / Precision / Recall / F1 ONCE

No threshold search or post-processing search is allowed here.

SAM-LoRA reconstruction
-----------------------
    official SAM checkpoint
        +
    sam_lora.json
        +
    frozen best_model.pth
        ↓
    trained SAM-LoRA
"""

from pathlib import Path
import sys
import json
import shutil
import argparse
import datetime as dt

import torch
import arcpy


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils import (
    load_experiment_config,
    load_general_params,
    load_paths_config,
    load_model_config,
    get_backbone_info,
    get_test_image_path,
    get_ground_truth_path,
    create_experiment_output_folder,
    normalize_model_type,
    save_json,
)

from src.models.factory import build_model

from src.inference import (
    load_model_checkpoint,
    run_full_arcgis_raster_inference,
)

from shared.metrics import (
    calculate_strict_polygon_iou,
    save_metrics_csv,
    save_metrics_json,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the one-time final test using frozen "
            "validation-selected settings."
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
        "--settings",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--allow-rerun",
        action="store_true",
    )

    return parser.parse_args()


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


# ============================================================
# ARCGIS HELPERS
# ============================================================

def validate_arcgis_dataset(path, label):

    if not arcpy.Exists(str(path)):
        raise FileNotFoundError(
            f"\n{label} was not found:\n{path}"
        )


def ensure_file_gdb(gdb_path):

    gdb_path = Path(gdb_path)

    gdb_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not arcpy.Exists(str(gdb_path)):
        arcpy.management.CreateFileGDB(
            str(gdb_path.parent),
            gdb_path.name,
        )

    return gdb_path


def delete_arcgis_dataset_if_exists(path):

    if arcpy.Exists(str(path)):
        arcpy.management.Delete(str(path))


def get_geometry_area_m2(geometry):

    try:
        return float(
            geometry.getArea(
                "GEODESIC",
                "SQUAREMETERS",
            )
        )
    except Exception:
        try:
            return float(
                geometry.getArea(
                    "PLANAR",
                    "SQUAREMETERS",
                )
            )
        except Exception:
            return float(geometry.area)


def apply_frozen_min_area_filter(
    input_fc,
    output_gdb,
    output_name,
    min_area_m2,
):
    """
    Apply exactly one frozen minimum-area threshold.

    No GT is used and no search occurs here.
    """

    input_fc = str(input_fc)

    validate_arcgis_dataset(
        input_fc,
        "Raw final-test prediction",
    )

    output_gdb = ensure_file_gdb(
        output_gdb
    )

    output_fc = str(
        output_gdb / output_name
    )

    delete_arcgis_dataset_if_exists(
        output_fc
    )

    min_area_m2 = float(
        min_area_m2
    )

    input_count = int(
        arcpy.management.GetCount(
            input_fc
        )[0]
    )

    if min_area_m2 <= 0.0:

        arcpy.management.CopyFeatures(
            input_fc,
            output_fc,
        )

        return {
            "output_fc": output_fc,
            "min_area_m2": 0.0,
            "input_count": input_count,
            "output_count": int(
                arcpy.management.GetCount(
                    output_fc
                )[0]
            ),
        }

    description = arcpy.Describe(
        input_fc
    )

    arcpy.management.CreateFeatureclass(
        out_path=str(output_gdb),
        out_name=output_name,
        geometry_type="POLYGON",
        spatial_reference=(
            description.spatialReference
        ),
    )

    kept_count = 0

    with arcpy.da.SearchCursor(
        input_fc,
        ["SHAPE@"],
    ) as search_cursor:

        with arcpy.da.InsertCursor(
            output_fc,
            ["SHAPE@"],
        ) as insert_cursor:

            for row in search_cursor:

                geometry = row[0]

                if geometry is None:
                    continue

                if (
                    get_geometry_area_m2(
                        geometry
                    )
                    >= min_area_m2
                ):
                    insert_cursor.insertRow(
                        [geometry]
                    )

                    kept_count += 1

    return {
        "output_fc": output_fc,
        "min_area_m2": min_area_m2,
        "input_count": input_count,
        "output_count": kept_count,
    }


# ============================================================
# FROZEN SETTINGS VALIDATION
# ============================================================

def validate_frozen_settings(
    final_settings,
    experiment_config,
):

    if not isinstance(final_settings, dict):
        raise TypeError(
            "final_settings.json must contain a JSON object."
        )

    if not bool(
        final_settings.get(
            "settings_frozen",
            False,
        )
    ):
        raise RuntimeError(
            "final_settings.json is not marked as frozen."
        )

    frozen_experiment = final_settings.get(
        "experiment",
        {},
    )

    current_experiment_id = (
        experiment_config.get(
            "experiment_id"
        )
    )

    frozen_experiment_id = (
        frozen_experiment.get(
            "experiment_id"
        )
    )

    if (
        frozen_experiment_id
        != current_experiment_id
    ):
        raise RuntimeError(
            "\nExperiment mismatch.\n"
            f"Current: {current_experiment_id}\n"
            f"Frozen: {frozen_experiment_id}"
        )

    model_type = normalize_model_type(
        frozen_experiment.get(
            "model_type",
            "",
        )
    )

    if model_type not in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:
        raise NotImplementedError(
            "run_final_test.py supports semantic models "
            "U-Net, DeepLabV3, and SAM-LoRA. "
            "Mask R-CNN uses its detector-specific final-test runner."
        )

    split_info = final_settings.get(
        "split",
        {},
    )

    if bool(
        split_info.get(
            "test_included_in_split",
            True,
        )
    ):
        raise RuntimeError(
            "Frozen settings indicate test data were included "
            "in Train/Validation."
        )

    threshold_info = final_settings.get(
        "threshold",
        {},
    )

    selected_on = str(
        threshold_info.get(
            "selected_on",
            "",
        )
    ).lower()

    if selected_on not in {
        "validation",
        "fixed_default",
    }:
        raise RuntimeError(
            "Frozen threshold was not selected by an allowed "
            "pre-test method."
        )

    post_info = final_settings.get(
        "postprocessing",
        {},
    )

    if bool(
        post_info.get(
            "enabled",
            False,
        )
    ):
        post_selected_on = str(
            post_info.get(
                "selected_on",
                "",
            )
        ).lower()

        if post_selected_on != "validation":
            raise RuntimeError(
                "Frozen post-processing parameter was not "
                "selected on validation."
            )

    final_test_info = final_settings.get(
        "final_test",
        {},
    )

    if not bool(
        final_test_info.get(
            "allowed",
            False,
        )
    ):
        raise RuntimeError(
            "Frozen settings do not allow final testing."
        )

    checkpoint_path = Path(
        final_settings[
            "model_selection"
        ][
            "checkpoint_path"
        ]
    )

    threshold = float(
        threshold_info["value"]
    )

    if not (
        0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            "Frozen threshold must be between 0 and 1."
        )

    min_area_m2 = float(
        post_info.get(
            "min_area_m2",
            0.0,
        )
    )

    if min_area_m2 < 0.0:
        raise ValueError(
            "Frozen minimum area must be >= 0."
        )

    return {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "threshold": threshold,
        "postprocessing_enabled": bool(
            post_info.get(
                "enabled",
                False,
            )
        ),
        "min_area_m2": min_area_m2,
    }


def validate_sam_lora_frozen_metadata(
    final_settings,
    model_config,
):
    """
    Verify the frozen SAM-LoRA architecture still matches sam_lora.json.

    Checkpoint key/shape compatibility is additionally enforced by
    load_trained_sam_lora().
    """

    frozen = final_settings.get(
        "sam_lora",
        {},
    )

    if not bool(
        frozen.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "Frozen settings do not contain SAM-LoRA metadata."
        )

    frozen_backbone = str(
        frozen.get(
            "backbone",
            "",
        )
    ).lower()

    current_backbone = str(
        model_config.get(
            "default_backbone",
            "",
        )
    ).lower()

    if frozen_backbone != current_backbone:
        raise RuntimeError(
            "Frozen SAM-LoRA backbone differs from sam_lora.json."
        )

    frozen_lora = frozen.get(
        "lora",
        {},
    )

    current_lora = model_config.get(
        "lora",
        {},
    )

    for key in (
        "rank",
        "alpha",
        "dropout",
    ):
        if (
            frozen_lora.get(key)
            != current_lora.get(key)
        ):
            raise RuntimeError(
                "Frozen SAM-LoRA LoRA configuration differs "
                f"for '{key}'."
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

    output_folder = Path(
        create_experiment_output_folder(
            experiment_config=experiment_config,
            root=ROOT,
        )
    )

    final_settings_path = (
        Path(args.settings)
        if args.settings is not None
        else output_folder / "final_settings.json"
    )

    final_settings = load_json_file(
        final_settings_path
    )

    frozen = validate_frozen_settings(
        final_settings=final_settings,
        experiment_config=experiment_config,
    )

    checkpoint_path = frozen[
        "checkpoint_path"
    ]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nFrozen checkpoint no longer exists:\n"
            f"{checkpoint_path}"
        )

    final_test_folder = (
        output_folder / "final_test"
    )

    final_test_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    receipt_path = (
        final_test_folder
        / "final_test_receipt.json"
    )

    if (
        receipt_path.exists()
        and not args.allow_rerun
    ):
        raise RuntimeError(
            "\nA final-test receipt already exists:\n"
            f"{receipt_path}\n"
            "Use --allow-rerun only for development/debugging."
        )

    if (
        receipt_path.exists()
        and args.allow_rerun
    ):
        print(
            "\nWARNING: rerun override is active. "
            "This is not a new untouched final test.\n"
        )

    # Only now resolve test paths.
    paths_config = load_paths_config(
        ROOT
    )

    test_image_path = get_test_image_path(
        experiment_config=experiment_config,
        paths_config=paths_config,
    )

    gt_fc = get_ground_truth_path(
        experiment_config=experiment_config,
        paths_config=paths_config,
    )

    validate_arcgis_dataset(
        test_image_path,
        "Final test raster",
    )

    validate_arcgis_dataset(
        gt_fc,
        "Final test ground-truth feature class",
    )

    model_type = frozen[
        "model_type"
    ]

    model_config = load_model_config(
        model_type,
        ROOT,
    )

    requested_device = resolve_device(
        args.device
    )

    model_loader = None
    official_sam_checkpoint = None
    checkpoint_format = None
    checkpoint_epoch = None

    if model_type == "sam_lora":

        validate_sam_lora_frozen_metadata(
            final_settings=final_settings,
            model_config=model_config,
        )

        from src.models.sam_lora.inference import (
            load_trained_sam_lora,
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

        model_loader = (
            "sam_lora_reconstruction"
        )

        official_sam_checkpoint = (
            load_info.pretrained_sam_checkpoint_path
        )

        checkpoint_format = (
            load_info.checkpoint_format
        )

        checkpoint_epoch = (
            load_info.trained_epoch
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

        model_loader = (
            "standard_shared_checkpoint_loader"
        )

    model.eval()

    tile_size = int(
        final_settings[
            "experiment"
        ][
            "tile_size"
        ]
    )

    threshold = float(
        frozen["threshold"]
    )

    min_area_m2 = float(
        frozen["min_area_m2"]
    )

    postprocessing_enabled = bool(
        frozen[
            "postprocessing_enabled"
        ]
    )

    overlap_ratio = float(
        general_params[
            "inference"
        ][
            "overlap_ratio"
        ]
    )

    print()
    print("=" * 80)
    print("FINAL SOURCE-SPECIFIC TEST")
    print("=" * 80)
    print(
        "Experiment ID:",
        final_settings[
            "experiment"
        ][
            "experiment_id"
        ],
    )
    print("Model:", model_type)
    print(
        "Backbone:",
        final_settings[
            "experiment"
        ][
            "backbone"
        ],
    )
    print(
        "Source:",
        final_settings[
            "experiment"
        ][
            "source_type"
        ],
    )
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint Epoch:", checkpoint_epoch)
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
    print("-" * 80)
    print(
        "Frozen Threshold:",
        f"{threshold:.4f}",
    )
    print(
        "Frozen Post-processing:",
        "ON" if postprocessing_enabled else "OFF",
    )
    print(
        "Frozen Minimum Area:",
        f"{min_area_m2:.2f} m²",
    )
    print("Tile Size:", tile_size)
    print("Overlap Ratio:", overlap_ratio)
    print("-" * 80)
    print("Final Test Raster:", test_image_path)
    print("Final Ground Truth:", gt_fc)
    print("-" * 80)
    print("Threshold Search: DISABLED")
    print("Post-processing Search: DISABLED")
    print("Settings: FROZEN")
    print("=" * 80)
    print()

    raw_inference_folder = (
        final_test_folder
        / "raw_inference"
    )

    if (
        raw_inference_folder.exists()
        and args.allow_rerun
    ):
        try:
            arcpy.ClearWorkspaceCache_management()
        except Exception:
            pass

        shutil.rmtree(
            raw_inference_folder,
            ignore_errors=True,
        )

    raw_inference_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    inference_summary = (
        run_full_arcgis_raster_inference(
            model=model,
            device=device,
            test_image_path=test_image_path,
            output_folder=raw_inference_folder,
            threshold=threshold,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
        )
    )

    raw_prediction_fc = (
        inference_summary[
            "final_prediction_fc"
        ]
    )

    validate_arcgis_dataset(
        raw_prediction_fc,
        "Raw final-test prediction",
    )

    final_prediction_gdb = (
        final_test_folder
        / "final_prediction.gdb"
    )

    filter_summary = (
        apply_frozen_min_area_filter(
            input_fc=raw_prediction_fc,
            output_gdb=final_prediction_gdb,
            output_name="final_prediction",
            min_area_m2=(
                min_area_m2
                if postprocessing_enabled
                else 0.0
            ),
        )
    )

    final_prediction_fc = (
        filter_summary["output_fc"]
    )

    validate_arcgis_dataset(
        final_prediction_fc,
        "Final frozen-settings prediction",
    )

    evaluation_folder = (
        final_test_folder
        / "evaluation"
    )

    evaluation_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_metrics = (
        calculate_strict_polygon_iou(
            pred_fc=final_prediction_fc,
            gt_fc=gt_fc,
            output_folder=evaluation_folder,
        )
    )

    final_metrics.update(
        {
            "experiment_id":
                final_settings[
                    "experiment"
                ][
                    "experiment_id"
                ],

            "model_type":
                model_type,

            "backbone":
                final_settings[
                    "experiment"
                ][
                    "backbone"
                ],

            "source_type":
                final_settings[
                    "experiment"
                ][
                    "source_type"
                ],

            "tile_size":
                tile_size,

            "frozen_threshold":
                threshold,

            "frozen_postprocessing_enabled":
                postprocessing_enabled,

            "frozen_min_area_m2":
                (
                    min_area_m2
                    if postprocessing_enabled
                    else 0.0
                ),

            "threshold_search_on_test":
                False,

            "postprocessing_search_on_test":
                False,

            "settings_frozen_before_test":
                True,

            "same_shared_test_zone":
                True,

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

            "final_settings_path":
                str(final_settings_path),

            "checkpoint_path":
                str(checkpoint_path),

            "test_image_path":
                str(test_image_path),

            "ground_truth_path":
                str(gt_fc),

            "raw_prediction_fc":
                str(raw_prediction_fc),

            "final_prediction_fc":
                str(final_prediction_fc),

            "raw_prediction_count":
                int(
                    filter_summary[
                        "input_count"
                    ]
                ),

            "final_prediction_count_after_frozen_filter":
                int(
                    filter_summary[
                        "output_count"
                    ]
                ),

            "evaluation_type":
                "one_time_final_external_test",

            "main_metric":
                "strict_polygon_iou",
        }
    )

    final_metrics[
        "final_iou_percent"
    ] = float(
        final_metrics["iou_percent"]
    )

    final_metrics_json = (
        final_test_folder
        / "final_metrics.json"
    )

    final_metrics_csv = (
        final_test_folder
        / "final_metrics.csv"
    )

    save_metrics_json(
        final_metrics,
        final_metrics_json,
    )

    save_metrics_csv(
        final_metrics,
        final_metrics_csv,
    )

    settings_snapshot_path = (
        final_test_folder
        / "final_settings_used.json"
    )

    save_json(
        final_settings,
        settings_snapshot_path,
    )

    receipt = {
        "final_test_completed":
            True,

        "completed_at_utc":
            dt.datetime.now(
                dt.timezone.utc
            ).isoformat(),

        "experiment_id":
            final_settings[
                "experiment"
            ][
                "experiment_id"
            ],

        "model_type":
            model_type,

        "source_type":
            final_settings[
                "experiment"
            ][
                "source_type"
            ],

        "settings_frozen":
            True,

        "settings_path":
            str(final_settings_path),

        "settings_snapshot":
            str(settings_snapshot_path),

        "checkpoint_path":
            str(checkpoint_path),

        "model_loader":
            model_loader,

        "official_sam_checkpoint":
            (
                str(official_sam_checkpoint)
                if official_sam_checkpoint
                is not None
                else None
            ),

        "threshold":
            threshold,

        "min_area_m2":
            (
                min_area_m2
                if postprocessing_enabled
                else 0.0
            ),

        "test_image_path":
            str(test_image_path),

        "ground_truth_path":
            str(gt_fc),

        "final_prediction_fc":
            str(final_prediction_fc),

        "final_metrics_json":
            str(final_metrics_json),

        "final_metrics_csv":
            str(final_metrics_csv),

        "final_strict_polygon_iou_percent":
            float(
                final_metrics[
                    "iou_percent"
                ]
            ),

        "precision_percent":
            float(
                final_metrics[
                    "precision_percent"
                ]
            ),

        "recall_percent":
            float(
                final_metrics[
                    "recall_percent"
                ]
            ),

        "f1_percent":
            float(
                final_metrics[
                    "f1_percent"
                ]
            ),

        "rerun_override_used":
            bool(args.allow_rerun),

        "methodology_note":
            (
                "The same source-specific external test zone was "
                "evaluated only after model, threshold, and "
                "post-processing settings were frozen. No threshold "
                "or post-processing parameter search was performed "
                "on final-test data."
            ),
    }

    save_json(
        receipt,
        receipt_path,
    )

    print()
    print("=" * 80)
    print("FINAL TEST FINISHED")
    print("=" * 80)
    print(
        "FINAL STRICT POLYGON IoU:",
        f"{final_metrics['iou_percent']:.2f}%",
    )
    print(
        "Precision:",
        f"{final_metrics['precision_percent']:.2f}%",
    )
    print(
        "Recall:",
        f"{final_metrics['recall_percent']:.2f}%",
    )
    print(
        "F1:",
        f"{final_metrics['f1_percent']:.2f}%",
    )
    print(
        "GT Count:",
        final_metrics["gt_count"],
    )
    print(
        "Raw Prediction Count:",
        filter_summary["input_count"],
    )
    print(
        "Final Prediction Count:",
        filter_summary["output_count"],
    )
    print("-" * 80)
    print("Threshold Search on Test: NO")
    print("Post-processing Search on Test: NO")
    print("Settings Frozen Before Test: YES")
    print("Same Shared Test Zone: YES")
    print("-" * 80)
    print(
        "Final Prediction:",
        final_prediction_fc,
    )
    print(
        "Final Metrics:",
        final_metrics_json,
    )
    print(
        "Final Test Receipt:",
        receipt_path,
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

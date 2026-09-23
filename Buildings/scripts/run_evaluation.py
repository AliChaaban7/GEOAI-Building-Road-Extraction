"""
run_evaluation.py

Standalone strict-polygon evaluation for a completed shared semantic
Building Extraction inference.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA
- standard and augmented experiments
- ArcGIS geodatabase feature classes
- inference threshold metadata
- Precision / Recall / F1 / strict polygon IoU

This script evaluates the prediction already recorded by:

    inference_summary.json

For the frozen one-time final-test protocol, run_final_test.py performs
its own final strict evaluation. This standalone script remains useful
for normal/raw inference evaluation and backward compatibility.
"""

from pathlib import Path
import sys
import json
import argparse

import arcpy


ROOT = Path(
    __file__
).resolve().parents[1]

PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils import (
    load_experiment_config,
    load_paths_config,
    get_ground_truth_path,
    create_experiment_output_folder,
    normalize_model_type,
    save_json,
)

from shared.metrics import (
    calculate_strict_polygon_iou,
    save_metrics_csv,
    save_metrics_json,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Building Extraction predictions "
            "using strict polygon IoU."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROOT / "config" / "experiment.json"
        ),
    )

    return parser.parse_args()


def load_inference_summary(output_folder):

    path = (
        Path(output_folder)
        / "inference_summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "\ninference_summary.json was not found:\n"
            f"{path}\n"
            "Run run_inference.py first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_arcgis_dataset(
    dataset_path,
    dataset_name,
):

    if not arcpy.Exists(
        str(dataset_path)
    ):
        raise FileNotFoundError(
            f"\n{dataset_name} was not found:\n"
            f"{dataset_path}"
        )


def get_inference_threshold(summary):

    for key in (
        "inference_threshold",
        "threshold",
    ):
        if key in summary:
            return float(summary[key])

    return 0.5


def main():

    args = parse_args()

    experiment_config = load_experiment_config(
        Path(args.config)
    )

    model_type = normalize_model_type(
        experiment_config["model_type"]
    )

    paths_config = load_paths_config(
        ROOT
    )

    output_folder = Path(
        create_experiment_output_folder(
            experiment_config=experiment_config,
            root=ROOT,
        )
    )

    inference_summary = load_inference_summary(
        output_folder
    )

    pred_fc = inference_summary.get(
        "final_prediction_fc"
    )

    if not pred_fc:
        raise KeyError(
            "final_prediction_fc is missing "
            "from inference_summary.json."
        )

    validate_arcgis_dataset(
        pred_fc,
        "Prediction feature class",
    )

    gt_fc = get_ground_truth_path(
        experiment_config=experiment_config,
        paths_config=paths_config,
    )

    validate_arcgis_dataset(
        gt_fc,
        "Ground-truth feature class",
    )

    threshold = get_inference_threshold(
        inference_summary
    )

    augmentation_enabled = bool(
        experiment_config.get(
            "use_augmentation",
            False,
        )
    )

    display_backbone = (
        inference_summary.get(
            "backbone",
            experiment_config.get(
                "backbone",
                "unknown",
            ),
        )
    )

    print()
    print("=" * 72)
    print("STRICT POLYGON IOU EVALUATION START")
    print("=" * 72)
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
    print(
        "Tile Size:",
        experiment_config["tile_size"],
    )
    print(
        "Training Augmentation:",
        "ON" if augmentation_enabled else "OFF",
    )
    print(
        "Inference Threshold:",
        threshold,
    )
    print(
        "Model Loader:",
        inference_summary.get(
            "model_loader",
            "unknown",
        ),
    )
    print("-" * 72)
    print("Prediction FC:", pred_fc)
    print("Ground Truth FC:", gt_fc)
    print("=" * 72)
    print()

    metrics = calculate_strict_polygon_iou(
        pred_fc=pred_fc,
        gt_fc=gt_fc,
        output_folder=output_folder,
    )

    metrics.update(
        {
            "experiment_id":
                experiment_config["experiment_id"],

            "model_type":
                model_type,

            "backbone":
                display_backbone,

            "source_type":
                experiment_config["source_type"],

            "tile_size":
                int(
                    experiment_config["tile_size"]
                ),

            "dataset_type":
                experiment_config["dataset_type"],

            "threshold":
                float(threshold),

            "augmentation_enabled":
                augmentation_enabled,

            "inference_augmentation":
                False,

            "selected_output_type":
                inference_summary.get(
                    "prediction_stage",
                    "raw_prediction",
                ),

            "selected_postprocessing_method":
                (
                    "shared_frozen_postprocessing"
                    if inference_summary.get(
                        "postprocessing_applied",
                        False,
                    )
                    else "none"
                ),

            "prediction_fc":
                str(pred_fc),

            "ground_truth_fc":
                str(gt_fc),

            "model_loader":
                inference_summary.get(
                    "model_loader"
                ),

            "official_sam_checkpoint":
                inference_summary.get(
                    "official_sam_checkpoint"
                ),

            "sam_lora_checkpoint_format":
                inference_summary.get(
                    "sam_lora_checkpoint_format"
                ),

            "same_shared_test_zone":
                bool(
                    inference_summary.get(
                        "same_shared_test_zone",
                        True,
                    )
                ),
        }
    )

    final_metrics_csv = (
        output_folder / "final_metrics.csv"
    )

    final_metrics_json = (
        output_folder / "final_metrics.json"
    )

    summary_json = (
        output_folder / "summary.json"
    )

    save_metrics_csv(
        metrics,
        final_metrics_csv,
    )

    save_metrics_json(
        metrics,
        final_metrics_json,
    )

    save_json(
        metrics,
        summary_json,
    )

    print()
    print("=" * 72)
    print("STRICT POLYGON IOU EVALUATION FINISHED")
    print("=" * 72)
    print(
        "Prediction Count:",
        metrics["pred_count"],
    )
    print(
        "Ground Truth Count:",
        metrics["gt_count"],
    )
    print(
        "Intersection Count:",
        metrics["intersection_count"],
    )
    print("-" * 72)
    print(
        "STRICT POLYGON IOU:",
        f"{metrics['iou_percent']:.2f}%",
    )
    print(
        "Precision:",
        f"{metrics['precision_percent']:.2f}%",
    )
    print(
        "Recall:",
        f"{metrics['recall_percent']:.2f}%",
    )
    print(
        "F1 Score:",
        f"{metrics['f1_percent']:.2f}%",
    )
    print("-" * 72)
    print(
        "Final Metrics CSV:",
        final_metrics_csv,
    )
    print(
        "Final Metrics JSON:",
        final_metrics_json,
    )
    print(
        "Summary JSON:",
        summary_json,
    )
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()

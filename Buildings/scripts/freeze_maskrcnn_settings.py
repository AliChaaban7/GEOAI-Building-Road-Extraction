"""Freeze validation-selected Mask R-CNN settings before final test."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.experiment_utils import prepare_experiment_split


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze Mask R-CNN settings.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BUILDINGS_ROOT / "config" / "experiment.json"),
    )
    parser.add_argument("--threshold-summary", type=str, default=None)
    parser.add_argument("--postprocess-summary", type=str, default=None)
    parser.add_argument("--optuna-summary", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--min-area-m2", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = prepare_experiment_split(BUILDINGS_ROOT, Path(args.config))
    config = prepared["experiment_config"]
    output_folder = Path(prepared["output_folder"])
    checkpoint = output_folder / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")

    score_threshold = 0.50 if args.score_threshold is None else float(args.score_threshold)
    mask_threshold = 0.50 if args.mask_threshold is None else float(args.mask_threshold)
    min_area_m2 = 0.0 if args.min_area_m2 is None else float(args.min_area_m2)

    threshold_payload = None
    if args.threshold_summary:
        threshold_payload = load_json(Path(args.threshold_summary))
        score_threshold = float(
            threshold_payload.get("best_score_threshold", score_threshold)
        )
        mask_threshold = float(
            threshold_payload.get("best_mask_threshold", mask_threshold)
        )

    post_payload = None
    if args.postprocess_summary:
        post_payload = load_json(Path(args.postprocess_summary))
        min_area_m2 = float(post_payload.get("best_min_area_m2", min_area_m2))
        score_threshold = float(post_payload.get("best_score_threshold", score_threshold))
        mask_threshold = float(post_payload.get("best_mask_threshold", mask_threshold))

    optuna_payload = None
    if args.optuna_summary:
        optuna_payload = load_json(Path(args.optuna_summary))

    training_summary_path = output_folder / "training_summary.json"
    training_summary = load_json(training_summary_path) if training_summary_path.exists() else {}

    final_settings = {
        "settings_frozen": True,
        "experiment_id": config["experiment_id"],
        "model_type": "maskrcnn",
        "backbone": config.get("backbone", "resnet50_fpn"),
        "source_type": config.get("source_type"),
        "tile_size": int(config.get("tile_size", 512)),
        "dataset_type": config.get("dataset_type"),
        "checkpoint": str(checkpoint),
        "best_epoch": training_summary.get("best_epoch"),
        "best_training_validation_iou": training_summary.get(
            "best_validation_iou", training_summary.get("best_val_iou")
        ),
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "min_area_m2": min_area_m2,
        "nms_threshold": 0.50,
        "detections_per_image": 300,
        "augmentation_enabled": bool(config.get("use_augmentation", False)),
        "optuna_enabled": bool(config.get("use_optuna", False)),
        "threshold_summary": args.threshold_summary,
        "postprocess_summary": args.postprocess_summary,
        "optuna_summary": args.optuna_summary,
        "selection_dataset": "validation_only",
        "external_test_accessed_before_freeze": False,
    }

    path = output_folder / "final_settings.json"
    path.write_text(json.dumps(final_settings, indent=2), encoding="utf-8")

    print("\n========== MASK R-CNN SETTINGS FROZEN ==========")
    print(f"Experiment ID: {config['experiment_id']}")
    print(f"Score Threshold: {score_threshold:.2f}")
    print(f"Mask Threshold: {mask_threshold:.2f}")
    print(f"Min Area: {min_area_m2:.2f} m2")
    print(f"Settings: {path}")
    print("External Test Accessed: NO")
    print("Ready for explicit final test: YES")
    print("=================================================\n")


if __name__ == "__main__":
    main()

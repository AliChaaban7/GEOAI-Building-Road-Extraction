"""
Validation-only Mask R-CNN evaluator.

Modes:
- fixed: comparable score at a fixed score/mask threshold
- threshold: search score + mask thresholds on validation only
- post: search minimum-area post-processing on validation only

No external test data are accessed.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.experiment_utils import (
    prepare_experiment_split,
    get_pixel_size_m,
)
from src.instance.maskrcnn_engine import (
    read_manifest_pairs,
    make_validation_loader,
    load_checkpoint_model,
    evaluate_validation,
    search_thresholds,
    search_postprocessing,
)


def parse_float_list(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Mask R-CNN validation optimization.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BUILDINGS_ROOT / "config" / "experiment.json"),
    )
    parser.add_argument(
        "--mode",
        choices=("fixed", "threshold", "post"),
        default="threshold",
    )
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument(
        "--score-thresholds",
        type=str,
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument(
        "--mask-thresholds",
        type=str,
        default="0.30,0.40,0.50,0.60,0.70",
    )
    parser.add_argument("--pixel-size-m", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    prepared = prepare_experiment_split(BUILDINGS_ROOT, Path(args.config))
    config = prepared["experiment_config"]
    output_folder = Path(prepared["output_folder"])

    checkpoint = Path(args.checkpoint) if args.checkpoint else output_folder / "best_model.pth"
    val_pairs = read_manifest_pairs(prepared["val_manifest"])
    val_loader = make_validation_loader(val_pairs)
    model, checkpoint_payload, device = load_checkpoint_model(checkpoint)

    if args.mode == "fixed":
        metrics = evaluate_validation(
            model,
            val_loader,
            device,
            score_threshold=float(args.score_threshold),
            mask_threshold=float(args.mask_threshold),
            min_component_pixels=0,
        )
        payload = {
            **metrics,
            "model_type": "maskrcnn",
            "experiment_id": config["experiment_id"],
            "selection_dataset": "validation_manifest",
            "external_test_used": False,
        }
        path = output_folder / "validation_fixed_summary.json"
        save_json(payload, path)
        print(f"Validation IoU: {metrics['iou_percent']:.2f}%")
        print(f"Summary: {path}")
        return

    if args.mode == "threshold":
        scores = parse_float_list(args.score_thresholds)
        masks = parse_float_list(args.mask_thresholds)
        df, best = search_thresholds(
            model,
            val_loader,
            device,
            score_thresholds=scores,
            mask_thresholds=masks,
        )
        folder = output_folder / "validation_threshold_optimization"
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / "validation_threshold_search.csv"
        df.to_csv(csv_path, index=False)
        payload = {
            "experiment_id": config["experiment_id"],
            "model_type": "maskrcnn",
            "best_score_threshold": float(best["score_threshold"]),
            "best_mask_threshold": float(best["mask_threshold"]),
            "best_threshold": float(best["score_threshold"]),
            "best_validation_iou": float(best["iou"]),
            "best_validation_iou_percent": float(best["iou_percent"]),
            "precision_percent": float(best["precision_percent"]),
            "recall_percent": float(best["recall_percent"]),
            "f1_percent": float(best["f1_percent"]),
            "results_csv": str(csv_path),
            "selection_dataset": "validation_manifest",
            "external_test_used": False,
        }
        path = output_folder / "validation_threshold_summary.json"
        save_json(payload, path)
        print("\nBEST VALIDATION THRESHOLDS")
        print(f"Score: {payload['best_score_threshold']:.2f}")
        print(f"Mask: {payload['best_mask_threshold']:.2f}")
        print(f"IoU: {payload['best_validation_iou_percent']:.2f}%")
        print("External Test Used: NO")
        return

    # POST mode
    threshold_summary_path = output_folder / "validation_threshold_summary.json"
    if not threshold_summary_path.exists():
        raise FileNotFoundError(
            "Run --mode threshold first. Missing: " + str(threshold_summary_path)
        )
    threshold_summary = json.loads(threshold_summary_path.read_text(encoding="utf-8"))
    score = float(threshold_summary["best_score_threshold"])
    mask = float(threshold_summary["best_mask_threshold"])
    areas = prepared["general_params"].get("postprocessing", {}).get(
        "area_thresholds_m2", [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
    )
    pixel_size_m = (
        float(args.pixel_size_m)
        if args.pixel_size_m is not None
        else get_pixel_size_m(
            prepared["dataset_info"],
            prepared["general_params"],
            fallback=0.3,
        )
    )
    df, best = search_postprocessing(
        model,
        val_loader,
        device,
        score_threshold=score,
        mask_threshold=mask,
        area_thresholds_m2=areas,
        pixel_size_m=pixel_size_m,
    )
    folder = output_folder / "validation_postprocess_optimization"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "validation_postprocess_search.csv"
    df.to_csv(csv_path, index=False)
    payload = {
        "experiment_id": config["experiment_id"],
        "model_type": "maskrcnn",
        "best_score_threshold": score,
        "best_mask_threshold": mask,
        "best_threshold": score,
        "best_min_area_m2": float(best["min_area_m2"]),
        "pixel_size_m": pixel_size_m,
        "best_validation_iou": float(best["iou"]),
        "best_validation_iou_percent": float(best["iou_percent"]),
        "precision_percent": float(best["precision_percent"]),
        "recall_percent": float(best["recall_percent"]),
        "f1_percent": float(best["f1_percent"]),
        "results_csv": str(csv_path),
        "selection_dataset": "validation_manifest",
        "external_test_used": False,
    }
    path = output_folder / "validation_postprocess_summary.json"
    save_json(payload, path)
    print("\nBEST VALIDATION POST-PROCESSING")
    print(f"Score Threshold: {score:.2f}")
    print(f"Mask Threshold: {mask:.2f}")
    print(f"Min Area: {payload['best_min_area_m2']:.2f} m2")
    print(f"IoU: {payload['best_validation_iou_percent']:.2f}%")
    print("External Test Used: NO")


if __name__ == "__main__":
    main()

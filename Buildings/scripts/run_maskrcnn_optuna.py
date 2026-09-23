"""
Flexible Mask R-CNN Optuna launcher for the Buildings module.

Reads experiment.json dynamically, prepares the same deterministic
Train/Validation split, runs Optuna with augmentation OFF, trains the final
optimized model, and synchronizes the winning checkpoint into the experiment
folder so the rest of the pipeline is model-agnostic.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.experiment_utils import prepare_experiment_split
from src.instance.maskrcnn_engine import read_manifest_pairs
from src.optional.maskrcnn_optuna_search import run_maskrcnn_optuna_workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN Optuna.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BUILDINGS_ROOT / "config" / "experiment.json"),
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--trial-epochs", type=int, default=12)
    parser.add_argument("--train-final", action="store_true")
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = prepare_experiment_split(BUILDINGS_ROOT, Path(args.config))
    config = prepared["experiment_config"]

    if str(config.get("model_type", "")).lower() not in {
        "maskrcnn", "mask_rcnn", "mask-r-cnn"
    }:
        raise ValueError("run_maskrcnn_optuna.py requires model_type=maskrcnn")
    if not bool(config.get("use_optuna", True)):
        raise RuntimeError("use_optuna must be true for the Optuna stage.")

    train_pairs = read_manifest_pairs(prepared["train_manifest"])
    val_pairs = read_manifest_pairs(prepared["val_manifest"])
    experiment_id = str(config["experiment_id"])
    optuna_folder = (
        BUILDINGS_ROOT / "outputs" / "optuna" / f"{experiment_id}_optuna"
    )
    study_name = f"{experiment_id}_optuna"

    summary = run_maskrcnn_optuna_workflow(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_folder=optuna_folder,
        study_name=study_name,
        n_trials=int(args.trials),
        trial_epochs=int(args.trial_epochs),
        train_final=bool(args.train_final),
        final_epochs=int(args.final_epochs),
        batch_size=int(args.batch_size),
        pretrained=not args.no_pretrained,
        seed=int(prepared["split_info"]["seed"]),
    )

    experiment_folder = Path(prepared["output_folder"])
    if args.train_final:
        for name in ("best_model.pth", "training_log.csv", "training_summary.json"):
            source = optuna_folder / name
            if source.exists():
                shutil.copy2(source, experiment_folder / name)

        selection = {
            "experiment_id": experiment_id,
            "model_type": "maskrcnn",
            "study_name": study_name,
            "best_trial": summary["best_trial"],
            "best_validation_iou": summary["best_validation_iou"],
            "best_validation_iou_percent": summary["best_validation_iou_percent"],
            "best_params": summary["best_params"],
            "optuna_folder": str(optuna_folder),
            "checkpoint_copied_to_experiment": str(experiment_folder / "best_model.pth"),
            "selection_dataset": "validation_manifest",
            "augmentation": False,
            "external_test_used": False,
        }
        (experiment_folder / "optuna_selection_summary.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )

    print("\n========== MASK R-CNN OPTUNA FINISHED ==========")
    print(f"Experiment ID: {experiment_id}")
    print(f"Best Trial: {summary['best_trial']}")
    print(f"Best Validation IoU: {summary['best_validation_iou_percent']:.2f}%")
    print("Augmentation: OFF")
    print("External Test Used: NO")
    if args.train_final:
        print(f"Final checkpoint: {experiment_folder / 'best_model.pth'}")
        print("Next Stage: VALIDATION THRESHOLD SEARCH")
    print("=================================================\n")


if __name__ == "__main__":
    main()

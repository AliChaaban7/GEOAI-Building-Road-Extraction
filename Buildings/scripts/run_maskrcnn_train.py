"""
Professional Mask R-CNN standard training launcher.

Supports:
- deterministic Train/Validation split from the shared dataset resolver
- optional shared augmentation (training only)
- pretrained ResNet50-FPN
- AdamW/SGD
- scheduler
- best checkpoint by validation pixel IoU
- no external test access
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.experiment_utils import prepare_experiment_split
from src.instance.maskrcnn_engine import read_manifest_pairs, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train professional Mask R-CNN.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BUILDINGS_ROOT / "config" / "experiment.json"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = prepare_experiment_split(
        BUILDINGS_ROOT,
        Path(args.config),
    )
    config = prepared["experiment_config"]
    general = prepared["general_params"]
    output_folder = prepared["output_folder"]

    if str(config.get("model_type", "")).lower() not in {
        "maskrcnn", "mask_rcnn", "mask-r-cnn"
    }:
        raise ValueError("run_maskrcnn_train.py requires model_type=maskrcnn")

    print("\n========== MASK R-CNN SPLIT ==========")
    print(f"Training: {prepared['split_info']['train_percent']:.0f}%")
    print(f"Validation: {prepared['split_info']['validation_percent']:.0f}%")
    print(f"Seed: {prepared['split_info']['seed']}")
    print("Test Included: NO")
    print("======================================\n")

    if args.prepare_only:
        print("Manifests prepared. Training not started.")
        return

    train_pairs = read_manifest_pairs(prepared["train_manifest"])
    val_pairs = read_manifest_pairs(prepared["val_manifest"])

    training_cfg = general.get("training", {})
    model_cfg = {}
    try:
        from src.utils import load_model_config
        model_cfg = load_model_config("maskrcnn", BUILDINGS_ROOT)
    except Exception:
        model_cfg = {}

    use_aug = bool(config.get("use_augmentation", False))
    augmentation = None
    if use_aug:
        from src.optional.augmentation import build_augmentation
        aug_path = BUILDINGS_ROOT / "config" / "optional" / "augmentation.json"
        augmentation = build_augmentation(
            config_path=aug_path,
            enabled_override=True,
        )

    optimizer_name = str(
        model_cfg.get("training", {}).get(
            "optimizer",
            training_cfg.get("optimizer", "adamw"),
        )
    ).lower()
    learning_rate = float(
        model_cfg.get("training", {}).get(
            "learning_rate",
            training_cfg.get("learning_rate", 1e-4),
        )
    )
    weight_decay = float(
        model_cfg.get("training", {}).get(
            "weight_decay",
            training_cfg.get("weight_decay", 1e-5),
        )
    )
    scheduler_name = str(
        model_cfg.get("training", {}).get(
            "scheduler",
            training_cfg.get("scheduler", "cosine"),
        )
    ).lower()
    if scheduler_name == "reduce_on_plateau":
        # Detection trainer uses epoch schedulers; cosine is the professional default.
        scheduler_name = "cosine"

    seed = int(prepared["split_info"]["seed"])
    pretrained = bool(
        model_cfg.get("pretrained", model_cfg.get("use_pretrained", True))
    )

    summary = train_model(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_folder=output_folder,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_name=scheduler_name,
        momentum=0.9,
        pretrained=pretrained,
        seed=seed,
        train_augmentation=augmentation,
        phase="augmentation" if use_aug else "professional_standard",
    )

    print("\n========== MASK R-CNN TRAINING FINISHED ==========")
    print(f"Experiment ID: {config['experiment_id']}")
    print(f"Best Validation IoU: {summary['best_validation_iou_percent']:.2f}%")
    print(f"Best Epoch: {summary['best_epoch']}")
    print(f"Augmentation: {'ON' if use_aug else 'OFF'}")
    print("External Test Used: NO")
    print("===================================================\n")


if __name__ == "__main__":
    main()

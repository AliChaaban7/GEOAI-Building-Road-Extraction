"""
Focused Optuna launcher for Buildings semantic segmentation under the
Train -> Validation -> Freeze -> Final Test protocol.

Supported:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA

Focused search space:
- learning rate
- weight decay
- BCE weight
- Dice weight = 1 - BCE weight

For SAM-LoRA, focused Optuna keeps the LoRA architecture fixed and uses
the dedicated SAM-LoRA training engine. The Optuna-selected learning rate
is fixed for each trial; Auto-LR is skipped.

The external test scene is never accessed here.
"""

from pathlib import Path
import sys
import json
import argparse
import shutil


BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.append(str(BUILDINGS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils import (
    load_experiment_config,
    load_general_params,
    load_datasets_config,
    get_dataset_info,
    create_experiment_output_folder,
    create_train_val_manifests,
    resolve_train_validation_split,
)

from src.models.model_registry import normalize_model_type
from src.optional.optuna_search import run_full_optuna_workflow


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def make_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [make_safe(v) for v in obj]
        return obj

    with path.open("w", encoding="utf-8") as file:
        json.dump(make_safe(data), file, indent=2)


def load_optional_optuna_config():
    default_config = {
        "trials": 10,
        "trial_epochs": 12,
        "train_final": False,
        "final_epochs": 40,
        "batch_size": None,
        "objective_metric": "best_training_validation_iou",
        "learning_rate_min": 1e-5,
        "learning_rate_max": 5e-4,
        "weight_decay_min": 1e-6,
        "weight_decay_max": 1e-3,
        "bce_weight_min": 0.3,
        "bce_weight_max": 0.7,
        "seed": 42,
    }

    path = (
        BUILDINGS_ROOT
        / "config"
        / "optional"
        / "optuna.json"
    )

    if path.exists():
        loaded = load_json(path)

        if (
            "optuna" in loaded
            and isinstance(loaded["optuna"], dict)
        ):
            loaded = loaded["optuna"]

        for key, value in loaded.items():
            default_config[key] = value

    return default_config


def resolve_default_batch_size(model_type, experiment_config):
    model_type = normalize_model_type(model_type)

    if model_type != "sam_lora":
        return 8

    from src.models.sam_lora.factory import (
        load_sam_lora_config,
        resolve_sam_lora_batch_size,
    )

    sam_config_path = (
        BUILDINGS_ROOT
        / "config"
        / "models"
        / "sam_lora.json"
    )

    sam_config = load_sam_lora_config(
        sam_config_path
    )

    return int(
        resolve_sam_lora_batch_size(
            config=sam_config,
            tile_size=experiment_config.get("tile_size"),
            backbone=experiment_config.get(
                "backbone",
                sam_config.get("default_backbone", "vit_b"),
            ),
            default=1,
        )
    )


def build_optuna_study_name(experiment_config):
    experiment_id = str(
        experiment_config["experiment_id"]
    ).strip()

    if experiment_id.endswith("_optuna"):
        return experiment_id

    return f"{experiment_id}_optuna"


def build_optuna_output_folder(experiment_config):
    return (
        BUILDINGS_ROOT
        / "outputs"
        / "optuna"
        / build_optuna_study_name(experiment_config)
    )


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def manifests_match_requested_split(
    experiment_output_folder,
    general_params,
):
    experiment_output_folder = Path(
        experiment_output_folder
    )

    train_manifest = (
        experiment_output_folder
        / "train_manifest.csv"
    )
    val_manifest = (
        experiment_output_folder
        / "val_manifest.csv"
    )

    desired = resolve_train_validation_split(
        general_params=general_params
    )

    if (
        not train_manifest.exists()
        or not val_manifest.exists()
    ):
        return False, desired

    summary = load_json_if_exists(
        experiment_output_folder
        / "split_summary.json"
    )

    if summary is None:
        return False, desired

    train_percent = summary.get("train_percent")
    validation_percent = summary.get("validation_percent")
    seed = summary.get("seed")

    if (
        train_percent is None
        or validation_percent is None
        or seed is None
    ):
        return False, desired

    matches = (
        abs(
            float(train_percent)
            - float(desired["train_percent"])
        ) < 1e-8
        and
        abs(
            float(validation_percent)
            - float(desired["validation_percent"])
        ) < 1e-8
        and
        int(seed) == int(desired["seed"])
    )

    return matches, desired


def ensure_manifests(
    experiment_config,
    general_params,
    datasets_config,
    experiment_output_folder,
):
    matches, split_info = manifests_match_requested_split(
        experiment_output_folder=experiment_output_folder,
        general_params=general_params,
    )

    train_manifest = (
        Path(experiment_output_folder)
        / "train_manifest.csv"
    )
    val_manifest = (
        Path(experiment_output_folder)
        / "val_manifest.csv"
    )

    if matches:
        print(
            "Existing Train / Validation manifests "
            "match the requested split."
        )
        return (
            train_manifest,
            val_manifest,
            split_info,
        )

    dataset_info = get_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config,
    )

    print(
        "Creating deterministic Train / Validation "
        "manifests for Optuna..."
    )

    result = create_train_val_manifests(
        dataset_info=dataset_info,
        output_folder=experiment_output_folder,
        general_params=general_params,
    )

    return (
        Path(result["train_manifest_csv"]),
        Path(result["val_manifest_csv"]),
        split_info,
    )


def protect_completed_experiment(
    experiment_output_folder,
):
    receipt = (
        Path(experiment_output_folder)
        / "final_test"
        / "final_test_receipt.json"
    )

    if receipt.exists():
        raise RuntimeError(
            "\nThis experiment already has a completed "
            "final-test receipt:\n"
            f"{receipt}\n\n"
            "Do not overwrite a completed experiment with Optuna.\n"
            "Create a NEW Optuna-specific experiment_id."
        )


def copy_if_exists(source, destination):
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        return None

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)
    return destination


def synchronize_final_optuna_model(
    optuna_output_folder,
    experiment_output_folder,
    workflow_summary,
    split_info,
):
    optuna_output_folder = Path(optuna_output_folder)
    experiment_output_folder = Path(experiment_output_folder)

    best_model_src = optuna_output_folder / "best_model.pth"
    training_log_src = optuna_output_folder / "training_log.csv"
    training_history_src = optuna_output_folder / "training_history.json"
    training_summary_src = optuna_output_folder / "training_summary.json"
    auto_lr_src = optuna_output_folder / "auto_lr.json"
    sam_trial_config_src = (
        optuna_output_folder
        / "sam_lora_trial_config.json"
    )

    if not best_model_src.exists():
        raise FileNotFoundError(
            "\nFinal Optuna best_model.pth was not created:\n"
            f"{best_model_src}"
        )

    if not training_summary_src.exists():
        raise FileNotFoundError(
            "\nFinal Optuna training_summary.json was not created:\n"
            f"{training_summary_src}"
        )

    best_model_dst = experiment_output_folder / "best_model.pth"
    training_log_dst = experiment_output_folder / "training_log.csv"
    training_history_dst = (
        experiment_output_folder
        / "training_history.json"
    )
    training_summary_dst = (
        experiment_output_folder
        / "training_summary.json"
    )
    auto_lr_dst = experiment_output_folder / "auto_lr.json"
    sam_trial_config_dst = (
        experiment_output_folder
        / "sam_lora_optuna_final_config.json"
    )

    copy_if_exists(best_model_src, best_model_dst)
    copy_if_exists(training_log_src, training_log_dst)
    copy_if_exists(training_history_src, training_history_dst)
    copy_if_exists(training_summary_src, training_summary_dst)
    copy_if_exists(auto_lr_src, auto_lr_dst)

    copied_sam_config = copy_if_exists(
        sam_trial_config_src,
        sam_trial_config_dst,
    )

    selection_summary = {
        "study_name": workflow_summary.get("study_name"),
        "experiment_id": workflow_summary.get("experiment_id"),
        "model_type": workflow_summary.get("model_type"),
        "backbone": workflow_summary.get("backbone"),
        "best_value": workflow_summary.get("best_value"),
        "best_value_percent": workflow_summary.get(
            "best_value_percent"
        ),
        "best_trial_number": workflow_summary.get(
            "best_trial_number"
        ),
        "best_params": workflow_summary.get("best_params"),
        "final_summary": workflow_summary.get("final_summary"),
        "train_percent": float(split_info["train_percent"]),
        "validation_percent": float(
            split_info["validation_percent"]
        ),
        "split_seed": int(split_info["seed"]),
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "final_checkpoint_path": str(best_model_dst),
        "optuna_output_folder": str(optuna_output_folder),
        "sam_lora_final_config": (
            str(copied_sam_config)
            if copied_sam_config is not None
            else None
        ),
        "methodology_note": (
            "Focused Optuna hyperparameters were selected using only "
            "the held-out validation split. The external test area was "
            "not accessed during Optuna optimization or final model "
            "training."
        ),
    }

    selection_summary_path = (
        experiment_output_folder
        / "optuna_selection_summary.json"
    )

    save_json(
        selection_summary,
        selection_summary_path,
    )

    return {
        "best_model": best_model_dst,
        "training_log": (
            training_log_dst
            if training_log_dst.exists()
            else None
        ),
        "training_history": (
            training_history_dst
            if training_history_dst.exists()
            else None
        ),
        "training_summary": training_summary_dst,
        "optuna_selection_summary": selection_summary_path,
        "sam_lora_final_config": copied_sam_config,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run focused Optuna optimization for Buildings "
            "semantic segmentation using validation only."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            BUILDINGS_ROOT
            / "config"
            / "experiment.json"
        ),
        help="Path to experiment configuration JSON.",
    )

    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--trial-epochs", type=int, default=None)
    parser.add_argument("--train-final", action="store_true")
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    return parser.parse_args()


def apply_cli_overrides(optuna_config, args):
    if args.trials is not None:
        optuna_config["trials"] = args.trials

    if args.trial_epochs is not None:
        optuna_config["trial_epochs"] = args.trial_epochs

    if args.train_final:
        optuna_config["train_final"] = True

    if args.final_epochs is not None:
        optuna_config["final_epochs"] = args.final_epochs

    if args.batch_size is not None:
        optuna_config["batch_size"] = args.batch_size

    return optuna_config


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()

    experiment_config = load_experiment_config(
        config_path
    )
    general_params = load_general_params(
        BUILDINGS_ROOT
    )
    datasets_config = load_datasets_config(
        BUILDINGS_ROOT
    )

    model_type = normalize_model_type(
        experiment_config.get("model_type", "")
    )

    experiment_config["model_type"] = model_type

    if model_type not in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:
        raise NotImplementedError(
            "Focused Optuna supports U-Net, "
            "DeepLabV3 and SAM-LoRA."
        )

    if not bool(
        experiment_config.get(
            "use_optuna",
            False,
        )
    ):
        raise RuntimeError(
            "\nexperiment.json currently has use_optuna=false.\n"
            "For an Optuna experiment, create a NEW experiment_id "
            "and set use_optuna=true."
        )

    experiment_output_folder = (
        create_experiment_output_folder(
            experiment_config=experiment_config,
            root=BUILDINGS_ROOT,
        )
    )

    protect_completed_experiment(
        experiment_output_folder
    )

    optuna_config = load_optional_optuna_config()
    optuna_config = apply_cli_overrides(
        optuna_config,
        args,
    )

    study_name = build_optuna_study_name(
        experiment_config
    )
    optuna_output_folder = build_optuna_output_folder(
        experiment_config
    )

    optuna_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    optuna_config["study_name"] = study_name

    train_manifest, val_manifest, split_info = ensure_manifests(
        experiment_config=experiment_config,
        general_params=general_params,
        datasets_config=datasets_config,
        experiment_output_folder=experiment_output_folder,
    )

    if optuna_config.get("batch_size") is None:
        optuna_config["batch_size"] = resolve_default_batch_size(
            model_type=model_type,
            experiment_config=experiment_config,
        )

    optuna_config["seed"] = int(
        split_info["seed"]
    )

    print()
    print("=" * 72)
    print("FOCUSED OPTUNA START")
    print("=" * 72)
    print("Study Name:", study_name)
    print("Experiment ID:", experiment_config["experiment_id"])
    print("Model Type:", model_type)
    print("Backbone:", experiment_config.get("backbone"))
    print("Source Type:", experiment_config.get("source_type"))
    print("Tile Size:", experiment_config.get("tile_size"))
    print(
        "Training Augmentation:",
        "ON"
        if experiment_config.get("use_augmentation", False)
        else "OFF",
    )
    print("-" * 72)
    print(
        "Train / Validation:",
        (
            f"{split_info['train_percent']:.0f}% / "
            f"{split_info['validation_percent']:.0f}%"
        ),
    )
    print("Split / Trial Seed:", split_info["seed"])
    print("Test Included:", "NO")
    print("Objective:", "best validation IoU during training")
    print(
        "Focused Search Parameters:",
        "learning_rate, weight_decay, bce_weight, dice_weight",
    )

    if model_type == "sam_lora":
        print(
            "SAM-LoRA Architecture Search:",
            "NO (reserved for Master Optuna)",
        )
        print(
            "SAM-LoRA LR Mode in Trials:",
            "FIXED (Optuna-selected; Auto-LR skipped)",
        )

    print("Trials:", optuna_config["trials"])
    print("Trial Epochs:", optuna_config["trial_epochs"])
    print("Train Final Model:", optuna_config["train_final"])
    print("Final Epochs:", optuna_config["final_epochs"])
    print("Batch Size:", optuna_config["batch_size"])
    print("-" * 72)
    print("Experiment Folder:", experiment_output_folder)
    print("Optuna Folder:", optuna_output_folder)
    print("Train Manifest:", train_manifest)
    print("Validation Manifest:", val_manifest)
    print("=" * 72)
    print()

    summary = run_full_optuna_workflow(
        buildings_root=BUILDINGS_ROOT,
        experiment_config=experiment_config,
        general_params=general_params,
        optuna_config=optuna_config,
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        output_folder=optuna_output_folder,
    )

    save_json(
        summary,
        optuna_output_folder
        / "run_optuna_summary.json",
    )

    synced = None

    if bool(optuna_config.get("train_final", False)):
        synced = synchronize_final_optuna_model(
            optuna_output_folder=optuna_output_folder,
            experiment_output_folder=experiment_output_folder,
            workflow_summary=summary,
            split_info=split_info,
        )

    print()
    print("=" * 72)
    print("FOCUSED OPTUNA FINISHED")
    print("=" * 72)
    print("Best Trial:", summary.get("best_trial_number"))

    print(
        "Best Validation Value:",
        (
            f"{float(summary.get('best_value_percent')):.2f}%"
            if summary.get("best_value_percent") is not None
            else None
        ),
    )

    print("Best Params:", summary.get("best_params"))

    if synced is not None:
        print("-" * 72)
        print(
            "Final Optuna checkpoint copied to:",
            synced["best_model"],
        )
        print(
            "Training summary copied to:",
            synced["training_summary"],
        )

        if synced.get("sam_lora_final_config") is not None:
            print(
                "SAM-LoRA final Optuna config copied to:",
                synced["sam_lora_final_config"],
            )

        print(
            "Optuna selection summary:",
            synced["optuna_selection_summary"],
        )
        print("External Test Used:", "NO")
        print("Next Stage:", "VALIDATION THRESHOLD SEARCH")

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()

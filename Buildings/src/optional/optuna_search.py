"""
Focused Optuna utilities for Buildings semantic segmentation.

Supported:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA

Focused search space:
- learning_rate
- weight_decay
- bce_weight
- dice_weight = 1 - bce_weight

External test data are never accessed here.
SAM-LoRA uses its dedicated trainer and a trial-specific SAM-LoRA JSON config.
Auto-LR is skipped during focused Optuna because Optuna itself selects the LR.
"""

from pathlib import Path
import sys
import copy
import json
import random

import numpy as np
import torch

try:
    import optuna
except ImportError:
    optuna = None

THIS_FILE = Path(__file__).resolve()
BUILDINGS_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = BUILDINGS_ROOT.parent

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.append(str(BUILDINGS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.factory import build_model
from src.models.model_registry import normalize_model_type
from src.train import (
    create_dataloaders,
    build_loss_function,
    build_optimizer,
    train_model,
)
from src.utils import load_model_config, get_backbone_info


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device():
    return torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def resolve_built_model(build_result):
    if isinstance(build_result, (tuple, list)):
        if not build_result:
            raise RuntimeError("build_model() returned an empty tuple/list.")
        model = build_result[0]
    else:
        model = build_result

    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model() did not return a torch.nn.Module.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = resolve_device()
        model = model.to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return model, device, trainable_params


def build_training_augmentation(experiment_config):
    if not bool(experiment_config.get("use_augmentation", False)):
        return None, None

    augmentation_config_path = (
        BUILDINGS_ROOT / "config" / "optional" / "augmentation.json"
    )

    if not augmentation_config_path.exists():
        raise FileNotFoundError(
            f"Augmentation enabled but config not found: {augmentation_config_path}"
        )

    from src.optional.augmentation import build_augmentation

    augmentation = build_augmentation(
        config_path=augmentation_config_path,
        enabled_override=True,
    )
    if augmentation is None:
        raise RuntimeError("Augmentation is enabled but could not be built.")

    return augmentation, augmentation_config_path


# ============================================================
# U-NET / DEEPLAB
# ============================================================

def build_trial_configs(
    experiment_config,
    general_params,
    learning_rate,
    weight_decay,
    bce_weight,
    dice_weight,
):
    model_type = experiment_config["model_type"]
    model_config = load_model_config(model_type, BUILDINGS_ROOT)
    backbone_info = get_backbone_info(
        experiment_config=experiment_config,
        model_config=model_config,
    )

    trial_general_params = copy.deepcopy(general_params)
    trial_general_params.setdefault("training", {})
    trial_general_params["training"]["learning_rate"] = float(learning_rate)
    trial_general_params["training"]["weight_decay"] = float(weight_decay)

    trial_model_config = copy.deepcopy(model_config)
    trial_model_config.setdefault("loss", {})
    trial_model_config["loss"]["bce_weight"] = float(bce_weight)
    trial_model_config["loss"]["dice_weight"] = float(dice_weight)

    return trial_general_params, trial_model_config, backbone_info


def train_standard_semantic_for_optuna(
    experiment_config,
    general_params,
    train_manifest_path,
    val_manifest_path,
    output_folder,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    bce_weight,
    dice_weight,
    seed=42,
):
    set_seed(seed)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    trial_general_params, trial_model_config, backbone_info = (
        build_trial_configs(
            experiment_config,
            general_params,
            learning_rate,
            weight_decay,
            bce_weight,
            dice_weight,
        )
    )

    train_augmentation, augmentation_config_path = (
        build_training_augmentation(experiment_config)
    )

    training_config = trial_general_params.get("training", {})

    train_loader, val_loader, train_dataset, val_dataset = (
        create_dataloaders(
            train_manifest_csv=train_manifest_path,
            val_manifest_csv=val_manifest_path,
            batch_size=int(batch_size),
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", True)),
            train_augmentation=train_augmentation,
        )
    )

    build_result = build_model(
        experiment_config=experiment_config,
        model_config=trial_model_config,
        backbone_info=backbone_info,
        move_to_device=True,
    )
    model, device, trainable_params = resolve_built_model(build_result)

    loss_fn = build_loss_function(trial_model_config)
    optimizer = build_optimizer(
        model=model,
        general_params=trial_general_params,
        model_config=trial_model_config,
    )

    print("\n" + "=" * 72)
    print("OPTUNA TRAINING RUN — SHARED STANDARD ENGINE")
    print("=" * 72)
    print(f"Epochs: {int(epochs)}")
    print(f"Device: {device}")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Batch Size: {int(batch_size)}")
    print(f"Learning Rate: {float(learning_rate)}")
    print(f"Weight Decay: {float(weight_decay)}")
    print(f"BCE Weight: {float(bce_weight)}")
    print(f"Dice Weight: {float(dice_weight)}")
    print("Training Augmentation:", "ON" if train_augmentation is not None else "OFF")
    print("Validation Augmentation: OFF")
    print("External Test Used: NO")
    print("=" * 72 + "\n")

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=int(epochs),
        output_folder=output_folder,
        experiment_config=experiment_config,
        model_config=trial_model_config,
        backbone_info=backbone_info,
        trainable_params=trainable_params,
        augmentation_enabled=(train_augmentation is not None),
        augmentation_config_path=(
            str(augmentation_config_path)
            if augmentation_config_path else None
        ),
    )

    summary.update({
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "bce_weight": float(bce_weight),
        "dice_weight": float(dice_weight),
        "trial_seed": int(seed),
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "training_engine": "src.train",
    })
    save_json(summary, output_folder / "training_summary.json")
    return summary


# ============================================================
# SAM-LORA
# ============================================================

def build_sam_lora_trial_config(
    experiment_config,
    output_folder,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    bce_weight,
    dice_weight,
):
    from src.models.sam_lora.factory import load_sam_lora_config

    base_config_path = (
        BUILDINGS_ROOT / "config" / "models" / "sam_lora.json"
    )

    base_config = load_sam_lora_config(base_config_path)
    trial_config = copy.deepcopy(base_config)

    backbone = str(
        experiment_config.get(
            "backbone",
            trial_config.get("default_backbone", "vit_b"),
        )
    ).strip().lower()

    trial_config["default_backbone"] = backbone

    training = trial_config.setdefault("training", {})
    training["epochs"] = int(epochs)
    training["default_batch_size"] = int(batch_size)
    training.setdefault("recommended_batch_size_by_backbone", {})
    training["recommended_batch_size_by_backbone"][backbone] = int(batch_size)
    training["weight_decay"] = float(weight_decay)

    lr_cfg = training.setdefault("learning_rate", {})
    lr_cfg["mode"] = "fixed"
    lr_cfg["value"] = float(learning_rate)

    loss_cfg = training.setdefault("loss", {})
    loss_cfg["type"] = "bce_dice"
    loss_cfg["bce_weight"] = float(bce_weight)
    loss_cfg["dice_weight"] = float(dice_weight)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    trial_config_path = output_folder / "sam_lora_trial_config.json"
    save_json(trial_config, trial_config_path)

    return trial_config, trial_config_path


def train_sam_lora_for_optuna(
    experiment_config,
    general_params,
    train_manifest_path,
    val_manifest_path,
    output_folder,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    bce_weight,
    dice_weight,
    seed=42,
):
    set_seed(seed)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    trial_config, trial_config_path = build_sam_lora_trial_config(
        experiment_config=experiment_config,
        output_folder=output_folder,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )

    train_augmentation, augmentation_config_path = (
        build_training_augmentation(experiment_config)
    )

    training_config = general_params.get("training", {})

    train_loader, val_loader, train_dataset, val_dataset = (
        create_dataloaders(
            train_manifest_csv=train_manifest_path,
            val_manifest_csv=val_manifest_path,
            batch_size=int(batch_size),
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", True)),
            train_augmentation=train_augmentation,
        )
    )

    from src.models.sam_lora.factory import build_sam_lora_factory_bundle
    from src.models.sam_lora.trainer import train_sam_lora

    device = resolve_device()

    bundle = build_sam_lora_factory_bundle(
        config_path=trial_config_path,
        backbone=experiment_config.get(
            "backbone",
            trial_config.get("default_backbone", "vit_b"),
        ),
        device=None,
        verbose_model_build=False,
    )

    trainer_kwargs = dict(bundle.trainer_kwargs)
    trainer_kwargs["epochs"] = int(epochs)
    trainer_kwargs["weight_decay"] = float(weight_decay)
    trainer_kwargs["learning_rate_mode"] = "fixed"
    trainer_kwargs["fixed_learning_rate"] = float(learning_rate)
    trainer_kwargs["learning_rate"] = float(learning_rate)

    print("\n" + "=" * 72)
    print("OPTUNA TRAINING RUN — SAM-LoRA ENGINE")
    print("=" * 72)
    print(f"Epochs: {int(epochs)}")
    print(f"Device: {device}")
    print(f"Backbone: {bundle.backbone}")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Batch Size: {int(batch_size)}")
    print(f"Learning Rate: {float(learning_rate)}")
    print("Learning Rate Mode: FIXED")
    print("Auto-LR Range Test: SKIPPED")
    print(f"Weight Decay: {float(weight_decay)}")
    print(f"BCE Weight: {float(bce_weight)}")
    print(f"Dice Weight: {float(dice_weight)}")
    print("Training Augmentation:", "ON" if train_augmentation is not None else "OFF")
    print("Validation Augmentation: OFF")
    print("Objective Dataset: VALIDATION ONLY")
    print("External Test Used: NO")
    print("=" * 72 + "\n")

    result = train_sam_lora(
        model_factory=bundle.model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=bundle.criterion,
        output_dir=output_folder,
        device=device,
        **trainer_kwargs,
    )

    summary_path = output_folder / "training_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    best_iou = float(
        getattr(
            result,
            "best_validation_iou",
            summary.get(
                "best_validation_iou",
                summary.get("best_val_iou", 0.0),
            ),
        )
    )

    best_epoch = int(
        getattr(
            result,
            "best_epoch",
            summary.get("best_epoch", -1),
        )
    )

    best_model_path = str(
        getattr(
            result,
            "best_checkpoint",
            output_folder / "best_model.pth",
        )
    )

    summary.update({
        "experiment_id": experiment_config.get("experiment_id"),
        "model_type": "sam_lora",
        "backbone": bundle.backbone,
        "best_val_iou": best_iou,
        "best_validation_iou": best_iou,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "resolved_learning_rate": float(learning_rate),
        "learning_rate_mode": "fixed",
        "auto_lr_used": False,
        "auto_lr_skipped": True,
        "weight_decay": float(weight_decay),
        "bce_weight": float(bce_weight),
        "dice_weight": float(dice_weight),
        "trial_seed": int(seed),
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "training_engine": "sam_lora",
        "augmentation_enabled": bool(train_augmentation is not None),
        "augmentation_config_path": (
            str(augmentation_config_path)
            if augmentation_config_path else None
        ),
        "sam_lora_trial_config": str(trial_config_path),
        "focused_optuna_parameters": [
            "learning_rate",
            "weight_decay",
            "bce_weight",
            "dice_weight",
        ],
        "lora_rank_fixed": trial_config.get("lora", {}).get("rank"),
        "lora_alpha_fixed": trial_config.get("lora", {}).get("alpha"),
    })

    save_json(summary, summary_path)
    return summary


# ============================================================
# DISPATCH
# ============================================================

def train_model_for_optuna(
    experiment_config,
    general_params,
    train_manifest_path,
    val_manifest_path,
    output_folder,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    bce_weight,
    dice_weight,
    seed=42,
):
    model_type = normalize_model_type(
        experiment_config.get("model_type", "")
    )

    if model_type == "sam_lora":
        return train_sam_lora_for_optuna(
            experiment_config,
            general_params,
            train_manifest_path,
            val_manifest_path,
            output_folder,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
            bce_weight,
            dice_weight,
            seed,
        )

    if model_type in {"unet", "deeplabv3"}:
        return train_standard_semantic_for_optuna(
            experiment_config,
            general_params,
            train_manifest_path,
            val_manifest_path,
            output_folder,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
            bce_weight,
            dice_weight,
            seed,
        )

    raise NotImplementedError(
        "Focused semantic Optuna supports U-Net, DeepLabV3 and SAM-LoRA."
    )


# ============================================================
# STUDY
# ============================================================

def run_optuna_study(
    experiment_config,
    general_params,
    optuna_config,
    train_manifest_path,
    val_manifest_path,
    output_folder,
):
    if optuna is None:
        raise ImportError("Optuna is not installed in this Python environment.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model_type = normalize_model_type(
        experiment_config.get("model_type", "")
    )

    trials = int(optuna_config.get("trials", 10))
    trial_epochs = int(optuna_config.get("trial_epochs", 12))
    batch_size = int(
        optuna_config.get(
            "batch_size",
            1 if model_type == "sam_lora" else 8,
        )
    )
    seed = int(optuna_config.get("seed", 42))
    study_name = str(
        optuna_config.get(
            "study_name",
            f"{experiment_config.get('experiment_id')}_optuna",
        )
    )

    lr_min = float(optuna_config.get("learning_rate_min", 1e-5))
    lr_max = float(optuna_config.get("learning_rate_max", 5e-4))
    wd_min = float(optuna_config.get("weight_decay_min", 1e-6))
    wd_max = float(optuna_config.get("weight_decay_max", 1e-3))
    bce_min = float(optuna_config.get("bce_weight_min", 0.3))
    bce_max = float(optuna_config.get("bce_weight_max", 0.7))

    def objective(trial):
        learning_rate = trial.suggest_float(
            "learning_rate", lr_min, lr_max, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", wd_min, wd_max, log=True
        )
        bce_weight = trial.suggest_float(
            "bce_weight", bce_min, bce_max
        )
        dice_weight = 1.0 - bce_weight

        trial_folder = (
            output_folder / "trials" / f"trial_{trial.number:03d}"
        )

        print("\n" + "-" * 72)
        print(f"OPTUNA TRIAL {trial.number}")
        print("-" * 72)
        print(f"Model Type: {model_type}")
        print(f"learning_rate: {learning_rate}")
        print(f"weight_decay: {weight_decay}")
        print(f"bce_weight: {bce_weight}")
        print(f"dice_weight: {dice_weight}")
        print(f"Trial Seed: {seed}")
        print("Objective Dataset: VALIDATION ONLY")
        print("Metric: best validation IoU")
        print("External Test Used: NO")
        print("-" * 72)

        summary = train_model_for_optuna(
            experiment_config=copy.deepcopy(experiment_config),
            general_params=general_params,
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            output_folder=trial_folder,
            epochs=trial_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            seed=seed,
        )

        best_iou = float(
            summary.get(
                "best_val_iou",
                summary.get("best_validation_iou", 0.0),
            )
        )

        trial.set_user_attr(
            "best_epoch",
            int(summary.get("best_epoch", -1)),
        )
        trial.set_user_attr(
            "best_model_path",
            summary.get("best_model_path"),
        )
        trial.set_user_attr(
            "best_validation_iou_percent",
            best_iou * 100.0,
        )
        trial.set_user_attr(
            "training_engine",
            summary.get("training_engine"),
        )
        trial.set_user_attr(
            "external_test_area_used",
            False,
        )

        return best_iou

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=trials,
    )

    trials_csv = output_folder / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    best_params = dict(study.best_trial.params)
    best_params["dice_weight"] = (
        1.0 - float(best_params["bce_weight"])
    )

    best_summary = {
        "study_name": study_name,
        "model_type": model_type,
        "best_value": float(study.best_value),
        "best_value_percent": float(study.best_value) * 100.0,
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "trials_csv": str(trials_csv),
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "seed": seed,
        "training_engine": (
            "sam_lora" if model_type == "sam_lora" else "src.train"
        ),
        "focused_optuna_parameters": [
            "learning_rate",
            "weight_decay",
            "bce_weight",
            "dice_weight",
        ],
        "sam_lora_architecture_parameters_searched": False,
    }

    save_json(
        best_summary,
        output_folder / "optuna_best_summary.json",
    )
    save_json(
        best_params,
        output_folder / "optuna_best_params.json",
    )

    return study, best_summary


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model_with_best_params(
    experiment_config,
    general_params,
    best_params,
    train_manifest_path,
    val_manifest_path,
    output_folder,
    final_epochs,
    batch_size,
    seed=42,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    learning_rate = float(best_params.get("learning_rate", 1e-4))
    weight_decay = float(best_params.get("weight_decay", 1e-5))
    bce_weight = float(best_params.get("bce_weight", 0.5))
    dice_weight = float(
        best_params.get("dice_weight", 1.0 - bce_weight)
    )

    final_training_summary = train_model_for_optuna(
        experiment_config=experiment_config,
        general_params=general_params,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        output_folder=output_folder,
        epochs=int(final_epochs),
        batch_size=int(batch_size),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        seed=int(seed),
    )

    best_validation_iou = float(
        final_training_summary.get(
            "best_val_iou",
            final_training_summary.get("best_validation_iou", 0.0),
        )
    )

    final_summary = {
        "experiment_id": experiment_config.get("experiment_id"),
        "model_type": normalize_model_type(
            experiment_config.get("model_type", "")
        ),
        "backbone": experiment_config.get("backbone"),
        "best_params": best_params,
        "final_epochs": int(final_epochs),
        "batch_size": int(batch_size),
        "best_model_path": final_training_summary.get(
            "best_model_path",
            str(output_folder / "best_model.pth"),
        ),
        "training_log_csv": final_training_summary.get(
            "training_log_csv",
            str(output_folder / "training_log.csv"),
        ),
        "training_summary_json": str(
            output_folder / "training_summary.json"
        ),
        "best_training_validation_iou": best_validation_iou,
        "best_training_validation_iou_percent": (
            best_validation_iou * 100.0
        ),
        "best_epoch": int(
            final_training_summary.get("best_epoch", -1)
        ),
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "training_engine": final_training_summary.get("training_engine"),
        "next_stage": "validation_threshold_search",
    }

    save_json(
        final_summary,
        output_folder / "optuna_final_summary.json",
    )

    print("\n" + "=" * 72)
    print("FINAL OPTUNA MODEL TRAINING FINISHED")
    print("=" * 72)
    print(f"Best Validation IoU: {best_validation_iou * 100.0:.2f}%")
    print(f"Best Epoch: {final_summary['best_epoch']}")
    print(f"Best Model: {final_summary['best_model_path']}")
    print("External Test Used: NO")
    print("Next Stage: VALIDATION THRESHOLD SEARCH")
    print("=" * 72 + "\n")

    return final_summary


# ============================================================
# FULL WORKFLOW
# ============================================================

def run_full_optuna_workflow(
    buildings_root,
    experiment_config,
    general_params,
    optuna_config,
    train_manifest_path,
    val_manifest_path,
    output_folder,
):
    del buildings_root

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model_type = normalize_model_type(
        experiment_config.get("model_type", "")
    )

    if model_type == "maskrcnn":
        raise NotImplementedError(
            "This focused Optuna workflow is for semantic models only."
        )

    if model_type not in {"unet", "deeplabv3", "sam_lora"}:
        raise NotImplementedError(
            "Focused Optuna supports U-Net, DeepLabV3 and SAM-LoRA."
        )

    study, best_summary = run_optuna_study(
        experiment_config=experiment_config,
        general_params=general_params,
        optuna_config=optuna_config,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        output_folder=output_folder,
    )

    final_summary = None

    if bool(optuna_config.get("train_final", False)):
        final_summary = train_final_model_with_best_params(
            experiment_config=experiment_config,
            general_params=general_params,
            best_params=best_summary["best_params"],
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            output_folder=output_folder,
            final_epochs=int(optuna_config.get("final_epochs", 40)),
            batch_size=int(
                optuna_config.get(
                    "batch_size",
                    1 if model_type == "sam_lora" else 8,
                )
            ),
            seed=int(optuna_config.get("seed", 42)),
        )

    workflow_summary = {
        "study_name": optuna_config.get("study_name"),
        "experiment_id": experiment_config.get("experiment_id"),
        "model_type": model_type,
        "backbone": experiment_config.get("backbone"),
        "output_folder": str(output_folder),
        "best_value": best_summary.get("best_value"),
        "best_value_percent": best_summary.get("best_value_percent"),
        "best_trial_number": best_summary.get("best_trial_number"),
        "best_params": best_summary.get("best_params"),
        "final_summary": final_summary,
        "selection_dataset": "validation_manifest",
        "external_test_area_used": False,
        "training_engine": (
            "sam_lora" if model_type == "sam_lora" else "src.train"
        ),
        "focused_optuna_parameters": [
            "learning_rate",
            "weight_decay",
            "bce_weight",
            "dice_weight",
        ],
        "sam_lora_architecture_parameters_searched": False,
    }

    save_json(
        workflow_summary,
        output_folder / "optuna_workflow_summary.json",
    )

    return workflow_summary

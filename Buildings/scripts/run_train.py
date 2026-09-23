"""
run_train.py

Shared training launcher for the Building Extraction module.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- Existing standard models through src.models.factory
- SAM-LoRA through its isolated SAM-LoRA training engine
- Optional shared augmentation
- Controlled Train / Validation percentages
- Reproducible split seed
- Reproducible training/model initialization seed
- Dataset-aware manifest reuse protection
- Automatic manifest validation / regeneration
- Legacy manifest backup before split regeneration
- Validation augmentation always OFF
- Test area completely excluded from training/validation

Important architecture
----------------------
LABELED TRAINING DATA
        |
  Train / Validation
        |
 Train      Validation
   |            |
weights      best epoch

The separate source-specific geographic test area is NOT used here.

SAM-LoRA training policy
------------------------
For model_type == "sam_lora":

    shared manifests
        ->
    shared dataset / augmentation
        ->
    temporary SAM-LoRA model
        ->
    Auto-LR range test
        ->
    temporary model discarded
        ->
    fresh SAM-LoRA model
        ->
    final training
        ->
    best_model.pth

The SAM-LoRA branch is intentionally lazy-imported so the existing
U-Net / DeepLab training path remains independent of SAM dependencies.

PowerShell
----------
    & $py "Buildings\\scripts\\run_train.py" --epochs 1
"""

from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import sys

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ============================================================
# SHARED MODULE IMPORTS
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_datasets_config,
    load_model_config,
    get_dataset_info,
    get_backbone_info,
    get_automatic_batch_size,
    create_experiment_output_folder,
    create_train_val_manifests,
    resolve_train_validation_split,
    set_random_seed,
    save_config_used,
    save_json,
)

from src.models.factory import build_model

from src.train import (
    create_dataloaders,
    build_loss_function,
    build_optimizer,
    train_model,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Building Extraction experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "experiment.json"),
        help="Path to experiment configuration JSON.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional epoch override. "
            "Standard models otherwise use general_params.json; "
            "SAM-LoRA otherwise uses sam_lora.json."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional batch-size override. "
            "If omitted, the normal model-specific automatic policy is used."
        ),
    )

    parser.add_argument(
        "--force-regenerate-split",
        action="store_true",
        help=(
            "Regenerate train/validation manifests even if the existing "
            "split appears compatible. Existing split files are backed up."
        ),
    )

    return parser.parse_args()


# ============================================================
# JSON / PATH HELPERS
# ============================================================

def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_path(path):
    """
    Normalize a filesystem path for reliable comparison on Windows.
    """
    return os.path.normcase(
        os.path.normpath(
            os.path.abspath(str(path))
        )
    )


def _same_path(path_a, path_b):
    try:
        return _normalize_path(path_a) == _normalize_path(path_b)
    except Exception:
        return False


def _path_is_inside(child_path, parent_path):
    """
    Return True when child_path is inside parent_path.

    os.path.commonpath is used instead of simple string-prefix matching,
    so paths such as CT_256 and CT_256_old are not confused.
    """
    try:
        child = _normalize_path(child_path)
        parent = _normalize_path(parent_path)
        return os.path.commonpath([child, parent]) == parent
    except Exception:
        return False


def _manifest_belongs_to_dataset(
    manifest_csv,
    dataset_path,
    sample_limit=20,
):
    """
    Verify that image paths stored in an existing manifest belong to
    the currently requested dataset.

    This is a fallback for older experiment folders whose split metadata
    did not store dataset identity.
    """
    manifest_csv = Path(manifest_csv)

    if not manifest_csv.exists():
        return False

    image_columns = (
        "image_path",
        "image",
        "image_file",
        "image_filepath",
        "chip_path",
        "raster_path",
    )

    checked = 0

    try:
        with manifest_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                return False

            image_column = next(
                (
                    column
                    for column in image_columns
                    if column in reader.fieldnames
                ),
                None,
            )

            if image_column is None:
                return False

            for row in reader:
                value = str(row.get(image_column, "")).strip()

                if not value:
                    continue

                image_path = Path(value)

                if not image_path.is_absolute():
                    image_path = Path(dataset_path) / image_path

                if not _path_is_inside(
                    image_path,
                    dataset_path,
                ):
                    return False

                checked += 1

                if checked >= int(sample_limit):
                    break

    except Exception:
        return False

    return checked > 0


def dataset_identity_matches_existing_manifests(
    output_folder,
    dataset_info,
):
    """
    Protect against stale-manifest reuse when the same experiment_id
    is reused after changing source_type, dataset_type, or tile_size.

    Preferred check:
        data_quality_summary.json -> dataset_path

    Backward-compatible fallback:
        inspect image paths inside train/validation manifests
    """
    output_folder = Path(output_folder)

    requested_dataset_path = (
        dataset_info.get("path")
        or dataset_info.get("dataset_path")
        or dataset_info.get("folder")
        or dataset_info.get("root")
    )

    if requested_dataset_path is None:
        return False

    quality_summary = load_json_if_exists(
        output_folder / "data_quality_summary.json"
    )

    if isinstance(quality_summary, dict):
        existing_dataset_path = quality_summary.get("dataset_path")

        if existing_dataset_path:
            return _same_path(
                existing_dataset_path,
                requested_dataset_path,
            )

    train_manifest = output_folder / "train_manifest.csv"
    val_manifest = output_folder / "val_manifest.csv"

    return (
        _manifest_belongs_to_dataset(
            train_manifest,
            requested_dataset_path,
        )
        and
        _manifest_belongs_to_dataset(
            val_manifest,
            requested_dataset_path,
        )
    )


# ============================================================
# MANIFEST BACKUP
# ============================================================

def backup_existing_manifests(output_folder):
    """
    Preserve the current split/data-quality files before regeneration.

    The first copy is stored under:
        <experiment>/legacy_split_backup/

    Existing backup files are never overwritten.
    """
    output_folder = Path(output_folder)
    backup_folder = output_folder / "legacy_split_backup"

    candidates = [
        "train_manifest.csv",
        "val_manifest.csv",
        "manifest_summary.json",
        "split_summary.json",
        "data_quality.csv",
        "data_quality_summary.json",
    ]

    existing_files = [
        output_folder / name
        for name in candidates
        if (output_folder / name).exists()
    ]

    if not existing_files:
        return None

    backup_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_any = False

    for source_path in existing_files:
        destination_path = backup_folder / source_path.name

        if destination_path.exists():
            continue

        shutil.copy2(
            source_path,
            destination_path,
        )
        copied_any = True

    if copied_any:
        print(
            "Legacy manifest backup created:",
            backup_folder,
        )
    else:
        print(
            "Legacy manifest backup already exists:",
            backup_folder,
        )

    return backup_folder


# ============================================================
# SPLIT + DATASET COMPATIBILITY CHECK
# ============================================================

def manifests_match_requested_split(
    output_folder,
    general_params,
    dataset_info,
):
    """
    Verify BOTH:
    1. Train/Validation percentages + seed
    2. Dataset identity

    Backward compatibility:
    - split_summary.json
    - old manifest_summary.json containing validation_ratio
    - old folders without dataset identity in split metadata
    """
    output_folder = Path(output_folder)

    train_manifest_csv = output_folder / "train_manifest.csv"
    val_manifest_csv = output_folder / "val_manifest.csv"

    desired_split = resolve_train_validation_split(
        general_params=general_params
    )

    if (
        not train_manifest_csv.exists()
        or not val_manifest_csv.exists()
    ):
        return False, desired_split, None

    existing_split = load_json_if_exists(
        output_folder / "split_summary.json"
    )

    if existing_split is None:
        existing_split = load_json_if_exists(
            output_folder / "manifest_summary.json"
        )

    if existing_split is None:
        return False, desired_split, None

    dataset_matches = dataset_identity_matches_existing_manifests(
        output_folder=output_folder,
        dataset_info=dataset_info,
    )

    if not dataset_matches:
        return False, desired_split, existing_split

    existing_train_percent = existing_split.get(
        "train_percent",
        None,
    )

    existing_validation_percent = existing_split.get(
        "validation_percent",
        None,
    )

    existing_validation_ratio = existing_split.get(
        "validation_ratio",
        None,
    )

    existing_seed = existing_split.get(
        "seed",
        None,
    )

    if (
        existing_validation_percent is None
        and existing_validation_ratio is not None
    ):
        existing_validation_percent = (
            float(existing_validation_ratio) * 100.0
        )

    if (
        existing_train_percent is None
        and existing_validation_percent is not None
    ):
        existing_train_percent = (
            100.0 - float(existing_validation_percent)
        )

    if (
        existing_train_percent is None
        or existing_validation_percent is None
        or existing_seed is None
    ):
        return False, desired_split, existing_split

    matches = (
        abs(
            float(existing_train_percent)
            - float(desired_split["train_percent"])
        ) < 1e-8
        and
        abs(
            float(existing_validation_percent)
            - float(desired_split["validation_percent"])
        ) < 1e-8
        and
        int(existing_seed)
        == int(desired_split["seed"])
    )

    return matches, desired_split, existing_split


# ============================================================
# STANDARD MODEL BUILD RESULT
# ============================================================

def resolve_built_model(build_result):
    """
    Make run_train.py compatible with standard model factories returning:
        model
        (model, ...)
    """
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
            "The first value returned by build_model() is not a "
            f"torch.nn.Module. Received: {type(model)}"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model = model.to(device)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return model, device, trainable_params


# ============================================================
# SAM-LORA HELPERS
# ============================================================

def _is_sam_lora(model_type):
    return str(model_type).strip().lower() == "sam_lora"


def _validate_sam_lora_experiment(
    experiment_config,
    sam_config,
    resolved_backbone,
):
    """
    Prevent silent disagreement between experiment.json and sam_lora.json.
    """
    dataset_type = str(
        experiment_config.get("dataset_type", "")
    ).strip().lower()

    sam_dataset_type = str(
        sam_config.get("dataset_type", "ct")
    ).strip().lower()

    if dataset_type and dataset_type != sam_dataset_type:
        raise ValueError(
            "SAM-LoRA dataset_type mismatch.\n"
            f"experiment.json: {dataset_type}\n"
            f"sam_lora.json  : {sam_dataset_type}"
        )

    experiment_backbone = experiment_config.get(
        "backbone",
        None,
    )

    if experiment_backbone is not None:
        if (
            str(experiment_backbone).strip().lower()
            != str(resolved_backbone).strip().lower()
        ):
            raise ValueError(
                "SAM-LoRA backbone mismatch.\n"
                f"experiment.json: {experiment_backbone}\n"
                f"sam_lora.json  : {resolved_backbone}\n"
                "Keep both values equal."
            )


def _sam_lora_backbone_info(
    sam_config,
    backbone,
):
    """
    Build lightweight backbone metadata from sam_lora.json.

    This is intentionally separate from the existing get_backbone_info()
    because SAM-LoRA uses supported_backbones rather than the schema used
    by the existing standard model configs.
    """
    supported = sam_config.get(
        "supported_backbones",
        {},
    )

    info = dict(
        supported.get(
            backbone,
            {},
        )
    )

    info["backbone"] = backbone
    info["name"] = backbone

    return info


def _sam_lora_training_summary(
    result,
    output_folder,
):
    """
    Convert SAMLoRATrainingResult into the common run_train summary shape.

    Supports both SAM-LoRA LR policies:
    - fixed: configured LR is used directly; Auto-LR is skipped.
    - auto: LR-range test is used before the clean final model.
    """
    output_folder = Path(output_folder)

    summary = {}

    result_summary_file = getattr(
        result,
        "summary_file",
        None,
    )

    if result_summary_file is not None:
        existing = load_json_if_exists(
            result_summary_file
        )

        if isinstance(existing, dict):
            summary.update(existing)

    best_checkpoint = getattr(
        result,
        "best_checkpoint",
        output_folder / "best_model.pth",
    )

    learning_rate_mode = str(
        getattr(
            result,
            "learning_rate_mode",
            summary.get(
                "learning_rate_mode",
                "auto",
            ),
        )
    ).strip().lower()

    if learning_rate_mode not in (
        "fixed",
        "auto",
    ):
        learning_rate_mode = "auto"

    training_engine = (
        "sam_lora_fixed_lr"
        if learning_rate_mode == "fixed"
        else "sam_lora_auto_lr"
    )

    summary.update(
        {
            "best_epoch": int(
                getattr(result, "best_epoch")
            ),
            "best_val_iou": float(
                getattr(result, "best_validation_iou")
            ),
            "best_validation_iou": float(
                getattr(result, "best_validation_iou")
            ),
            "best_model_path": str(
                best_checkpoint
            ),
            "resolved_learning_rate": float(
                getattr(result, "resolved_learning_rate")
            ),
            "learning_rate_mode":
                learning_rate_mode,
            "auto_lr_used":
                learning_rate_mode == "auto",
            "auto_lr_skipped":
                learning_rate_mode == "fixed",
            "epochs_completed": int(
                getattr(result, "epochs_completed")
            ),
            "stopped_early": bool(
                getattr(result, "stopped_early")
            ),
            "training_summary_json": str(
                result_summary_file
                if result_summary_file is not None
                else output_folder / "training_summary.json"
            ),
            "training_history_json": str(
                output_folder / "training_history.json"
            ),
            "auto_lr_json": str(
                output_folder / "auto_lr.json"
            ),
            "training_log_csv": str(
                output_folder / "training_log.csv"
            ),
            "training_engine":
                training_engine,
        }
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # ========================================================
    # 1. LOAD CONFIGURATION
    # ========================================================

    experiment_config = load_experiment_config(
        args.config
    )

    general_params = load_general_params(
        ROOT
    )

    datasets_config = load_datasets_config(
        ROOT
    )

    # ========================================================
    # 1A. REPRODUCIBLE TRAINING SEED
    # ========================================================

    training_config = general_params.get(
        "training",
        {},
    )

    training_seed = int(
        training_config.get(
            "random_seed",
            training_config.get(
                "seed",
                general_params.get(
                    "data_split",
                    {},
                ).get(
                    "seed",
                    42,
                ),
            ),
        )
    )

    set_random_seed(
        training_seed
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ========================================================
    # 2. MODEL + DATASET RESOLUTION
    # ========================================================

    model_type = str(
        experiment_config["model_type"]
    ).strip().lower()

    is_sam_lora = _is_sam_lora(
        model_type
    )

    dataset_info = get_dataset_info(
        experiment_config=experiment_config,
        datasets_config=datasets_config,
    )

    sam_bundle = None

    if is_sam_lora:
        # ----------------------------------------------------
        # Lazy import is deliberate.
        #
        # Existing U-Net / DeepLab runs must not depend on SAM
        # installation or SAM-LoRA package initialization.
        # ----------------------------------------------------
        from src.models.sam_lora.factory import (
            build_sam_lora_factory_bundle,
            resolve_sam_lora_batch_size,
        )

        sam_config_path = (
            ROOT
            / "config"
            / "models"
            / "sam_lora.json"
        )

        if not sam_config_path.exists():
            raise FileNotFoundError(
                "SAM-LoRA model configuration was not found:\n"
                f"{sam_config_path}"
            )

        sam_bundle = build_sam_lora_factory_bundle(
            config_path=sam_config_path,
            verbose_model_build=False,
        )

        model_config = sam_bundle.config

        _validate_sam_lora_experiment(
            experiment_config=experiment_config,
            sam_config=model_config,
            resolved_backbone=sam_bundle.backbone,
        )

        backbone_info = _sam_lora_backbone_info(
            sam_config=model_config,
            backbone=sam_bundle.backbone,
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

    else:
        # ----------------------------------------------------
        # ORIGINAL STANDARD PATH
        # ----------------------------------------------------
        model_config = load_model_config(
            model_type,
            ROOT,
        )

        backbone_info = get_backbone_info(
            experiment_config=experiment_config,
            model_config=model_config,
        )

        if args.batch_size is not None:
            batch_size = int(
                args.batch_size
            )
        else:
            batch_size = get_automatic_batch_size(
                experiment_config=experiment_config,
                backbone_info=backbone_info,
            )

    if int(batch_size) <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    output_folder = create_experiment_output_folder(
        experiment_config=experiment_config,
        root=ROOT,
    )

    output_folder = Path(output_folder)

    # ========================================================
    # 3. TRAIN / VALIDATION SPLIT
    # ========================================================

    (
        manifests_match,
        split_info,
        existing_split,
    ) = manifests_match_requested_split(
        output_folder=output_folder,
        general_params=general_params,
        dataset_info=dataset_info,
    )

    print()
    print("=" * 65)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 65)
    print(
        "Training Percentage:",
        f"{split_info['train_percent']:.2f}%",
    )
    print(
        "Validation Percentage:",
        f"{split_info['validation_percent']:.2f}%",
    )
    print("Split Seed:", split_info["seed"])
    print(
        "Split Source:",
        split_info.get("source", "unknown"),
    )
    print(
        "Dataset ID:",
        dataset_info.get("dataset_id", "unknown"),
    )
    print(
        "Dataset Path:",
        dataset_info.get("path"),
    )
    print("Test Included in Split:", "NO")
    print(
        "Final Test Policy:",
        "same separate source-specific geographic test area",
    )
    print("=" * 65)
    print()

    train_manifest_csv = (
        output_folder / "train_manifest.csv"
    )

    val_manifest_csv = (
        output_folder / "val_manifest.csv"
    )

    regenerate_split = (
        bool(args.force_regenerate_split)
        or not manifests_match
    )

    if regenerate_split:
        backup_existing_manifests(
            output_folder
        )

        if args.force_regenerate_split:
            print(
                "Forced Train / Validation split regeneration."
            )
        elif (
            train_manifest_csv.exists()
            or val_manifest_csv.exists()
        ):
            print(
                "Existing manifests do not match the requested "
                "dataset and/or split configuration."
            )
            print(
                "Regenerating deterministic manifests..."
            )
        else:
            print(
                "Train / Validation manifests not found."
            )
            print(
                "Creating deterministic manifests..."
            )

        manifest_result = create_train_val_manifests(
            dataset_info=dataset_info,
            output_folder=output_folder,
            general_params=general_params,
        )

        train_manifest_csv = Path(
            manifest_result["train_manifest_csv"]
        )
        val_manifest_csv = Path(
            manifest_result["val_manifest_csv"]
        )
    else:
        print(
            "Existing manifests match the requested dataset and split."
        )
        print(
            "Reusing existing Train / Validation manifests."
        )

    # ========================================================
    # 4. VERIFY MANIFESTS
    # ========================================================

    if not train_manifest_csv.exists():
        raise FileNotFoundError(
            "\nTrain manifest was not created/found:\n"
            f"{train_manifest_csv}"
        )

    if not val_manifest_csv.exists():
        raise FileNotFoundError(
            "\nValidation manifest was not created/found:\n"
            f"{val_manifest_csv}"
        )

    # ========================================================
    # 5. SAVE CONFIGURATION USED
    # ========================================================

    save_config_used(
        experiment_config=experiment_config,
        general_params=general_params,
        model_config=model_config,
        datasets_config=datasets_config,
        output_folder=output_folder,
    )

    # ========================================================
    # 6. AUGMENTATION
    # ========================================================

    use_augmentation = bool(
        experiment_config.get(
            "use_augmentation",
            False,
        )
    )

    augmentation_config_path = (
        ROOT
        / "config"
        / "optional"
        / "augmentation.json"
    )

    train_augmentation = None

    if use_augmentation:
        from src.optional.augmentation import (
            build_augmentation,
            describe_augmentation,
        )

        if not augmentation_config_path.exists():
            raise FileNotFoundError(
                "\nAugmentation is enabled but augmentation.json "
                "was not found:\n"
                f"{augmentation_config_path}"
            )

        train_augmentation = build_augmentation(
            config_path=augmentation_config_path,
            enabled_override=True,
        )

        if train_augmentation is None:
            raise RuntimeError(
                "use_augmentation=True but the augmentation "
                "pipeline could not be created."
            )

        print()
        print("=" * 50)
        print("TRAINING AUGMENTATION ENABLED")
        print("=" * 50)

        describe_augmentation(
            augmentation_config_path
        )
    else:
        print()
        print("=" * 50)
        print("TRAINING AUGMENTATION DISABLED")
        print("=" * 50)

    # ========================================================
    # 7. DATALOADERS
    # ========================================================

    num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        training_config.get(
            "pin_memory",
            True,
        )
    )

    if is_sam_lora:
        from src.models.sam_lora.data import (
            create_sam_lora_dataloaders,
        )

        data_bundle = create_sam_lora_dataloaders(
            train_manifest_csv=train_manifest_csv,
            val_manifest_csv=val_manifest_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_augmentation=train_augmentation,
            validate_data=True,
            verbose=True,
        )

        train_loader = data_bundle.train_loader
        val_loader = data_bundle.val_loader
        train_dataset = data_bundle.train_dataset
        val_dataset = data_bundle.val_dataset

    else:
        (
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        ) = create_dataloaders(
            train_manifest_csv=train_manifest_csv,
            val_manifest_csv=val_manifest_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_augmentation=train_augmentation,
        )

    # ========================================================
    # 8. BUILD/TRAINING ENGINE RESOLUTION
    # ========================================================

    if is_sam_lora:
        # ----------------------------------------------------
        # IMPORTANT:
        # Do not build the SAM-LoRA model here.
        #
        # train_sam_lora() owns model construction:
        #
        # FIXED LR:
        #   build one clean final model -> train
        #
        # AUTO LR:
        #   temporary LR-range-test model -> discard ->
        #   build a fresh final model -> train
        #
        # This keeps fixed-LR baseline training lightweight while
        # preserving the previous Auto-LR workflow as rollback.
        # ----------------------------------------------------
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        params = None

        trainer_kwargs = dict(
            sam_bundle.trainer_kwargs
        )

        if args.epochs is not None:
            epochs = int(
                args.epochs
            )
        else:
            epochs = int(
                trainer_kwargs.get(
                    "epochs",
                    40,
                )
            )

        trainer_kwargs["epochs"] = int(
            epochs
        )

    else:
        # ----------------------------------------------------
        # ORIGINAL STANDARD MODEL BUILD
        # ----------------------------------------------------
        build_result = build_model(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
            move_to_device=True,
        )

        model, device, params = resolve_built_model(
            build_result
        )

        loss_fn = build_loss_function(
            model_config
        )

        optimizer = build_optimizer(
            model=model,
            general_params=general_params,
            model_config=model_config,
        )

        if args.epochs is not None:
            epochs = int(
                args.epochs
            )
        else:
            epochs = int(
                training_config.get(
                    "epochs",
                    40,
                )
            )

    # ========================================================
    # 9. DISPLAY CONFIGURATION
    # ========================================================

    print()
    print("=" * 65)
    print("TRAINING START")
    print("=" * 65)
    print(
        "Experiment ID:",
        experiment_config["experiment_id"],
    )
    print(
        "Model Type:",
        model_type,
    )

    if is_sam_lora:
        display_backbone = sam_bundle.backbone
    else:
        display_backbone = experiment_config.get(
            "backbone",
            backbone_info.get(
                "backbone",
                "unknown",
            ),
        )

    print(
        "Backbone:",
        display_backbone,
    )
    print(
        "Source Type:",
        experiment_config["source_type"],
    )
    print(
        "Tile Size:",
        experiment_config["tile_size"],
    )
    print(
        "Dataset Type:",
        experiment_config["dataset_type"],
    )
    print(
        "Dataset ID:",
        dataset_info.get("dataset_id", "unknown"),
    )
    print(
        "Dataset Path:",
        dataset_info.get("path"),
    )
    print("Device:", device)

    if is_sam_lora:
        print(
            "Trainable Parameters:",
            "resolved during SAM-LoRA model build",
        )
        print(
            "Learning Rate:",
            "AUTO (LR range test)",
        )
    else:
        print(
            "Trainable Parameters:",
            f"{params:,}",
        )

    print("-" * 65)
    print(
        "Training Percentage:",
        f"{split_info['train_percent']:.2f}%",
    )
    print(
        "Validation Percentage:",
        f"{split_info['validation_percent']:.2f}%",
    )
    print("Split Seed:", split_info["seed"])
    print("Training Seed:", training_seed)
    print(
        "Train Samples:",
        len(train_dataset),
    )
    print(
        "Validation Samples:",
        len(val_dataset),
    )
    print(
        "Final Test:",
        "NOT USED DURING TRAINING",
    )
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)

    if is_sam_lora:
        print(
            "Input Normalization:",
            (
                "shared RGB /255 input; "
                "official SAM normalization applied internally"
            ),
        )
        print(
            "SAM Input Size:",
            model_config.get(
                "input",
                {},
            ).get(
                "sam_input_size",
                1024,
            ),
        )
    else:
        print(
            "Image Normalization:",
            "/255.0 only",
        )

    print(
        "Training Augmentation:",
        "ON"
        if use_augmentation
        else "OFF",
    )
    print(
        "Validation Augmentation:",
        "OFF",
    )

    if use_augmentation:
        print(
            "Augmentation Config:",
            augmentation_config_path,
        )

    print(
        "Train Manifest:",
        train_manifest_csv,
    )
    print(
        "Validation Manifest:",
        val_manifest_csv,
    )
    print(
        "Output Folder:",
        output_folder,
    )
    print("=" * 65)
    print()

    # ========================================================
    # 10. TRAIN
    # ========================================================

    if is_sam_lora:
        from src.models.sam_lora.trainer import (
            train_sam_lora,
        )

        training_result = train_sam_lora(
            model_factory=sam_bundle.model_factory,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=sam_bundle.criterion,
            output_dir=output_folder,
            device=device,
            **trainer_kwargs,
        )

        training_summary = _sam_lora_training_summary(
            result=training_result,
            output_folder=output_folder,
        )

    else:
        training_summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            output_folder=output_folder,
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info,
            trainable_params=params,
            augmentation_enabled=use_augmentation,
            augmentation_config_path=(
                str(augmentation_config_path)
                if use_augmentation
                else None
            ),
        )

    # ========================================================
    # 11. FINAL METADATA
    # ========================================================

    training_summary.update(
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
                experiment_config["tile_size"],

            "dataset_type":
                experiment_config["dataset_type"],

            "dataset_id":
                dataset_info.get("dataset_id"),

            "dataset_path":
                str(dataset_info.get("path")),

            "batch_size":
                int(batch_size),

            "epochs_requested":
                int(epochs),

            "train_percent":
                float(split_info["train_percent"]),

            "validation_percent":
                float(split_info["validation_percent"]),

            "split_seed":
                int(split_info["seed"]),

            "training_seed":
                int(training_seed),

            "deterministic_cudnn":
                bool(torch.cuda.is_available()),

            "split_source":
                split_info.get(
                    "source",
                    "unknown",
                ),

            "test_included_in_split":
                False,

            "final_test_policy":
                "same separate source-specific geographic test area",

            "train_manifest_csv":
                str(train_manifest_csv),

            "val_manifest_csv":
                str(val_manifest_csv),

            "train_samples":
                int(len(train_dataset)),

            "validation_samples":
                int(len(val_dataset)),

            "augmentation_enabled":
                bool(use_augmentation),

            "validation_augmentation":
                False,

            "augmentation_config_path": (
                str(augmentation_config_path)
                if use_augmentation
                else None
            ),
        }
    )

    if not is_sam_lora:
        # Preserve the existing standard summary key.
        training_summary["epochs"] = int(
            epochs
        )

    # ========================================================
    # 12. SAVE SUMMARY
    # ========================================================

    save_json(
        training_summary,
        output_folder / "training_summary.json",
    )

    # ========================================================
    # 13. FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)
    print(
        "Best Validation IoU:",
        f"{training_summary['best_val_iou']:.4f}",
    )
    print(
        "Best Epoch:",
        training_summary["best_epoch"],
    )
    print(
        "Best Model:",
        training_summary["best_model_path"],
    )

    training_log = training_summary.get(
        "training_log_csv",
        None,
    )

    if training_log:
        print(
            "Training Log:",
            training_log,
        )

    training_history = training_summary.get(
        "training_history_json",
        None,
    )

    if training_history:
        print(
            "Training History:",
            training_history,
        )

    if is_sam_lora:
        print(
            "Resolved Auto-LR:",
            (
                f"{training_summary['resolved_learning_rate']:.6e}"
            ),
        )

        print(
            "Auto-LR Results:",
            training_summary.get(
                "auto_lr_json",
            ),
        )

    print(
        "Dataset ID:",
        dataset_info.get("dataset_id", "unknown"),
    )
    print(
        "Train / Validation:",
        (
            f"{split_info['train_percent']:.0f}% / "
            f"{split_info['validation_percent']:.0f}%"
        ),
    )
    print("Split Seed:", split_info["seed"])
    print("Training Seed:", training_seed)
    print("Final Test Used:", "NO")
    print(
        "Augmentation:",
        "ON"
        if use_augmentation
        else "OFF",
    )
    print(
        "Validation Augmentation:",
        "OFF",
    )
    print(
        "Training Summary:",
        output_folder / "training_summary.json",
    )
    print("=" * 65)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

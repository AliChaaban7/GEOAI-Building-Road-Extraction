"""
test_training_setup.py

Test the training setup for the Building Extraction module.

Updated:
- Controlled Train / Validation percentage support.
- Reads the shared split configuration from general_params.json.
- Preserves backward compatibility with existing manifests.
- Regenerates manifests only when the requested split configuration changes.
- Test area remains completely separate from Train / Validation.
- Compatible with tuple batches: images, masks.
- Compatible with dict batches if used later.
- Supports DeepLabV3 auxiliary output.
- Supports main loss + auxiliary loss.
- Robust to different build_model() return formats.
"""

from pathlib import Path
import sys
import json

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


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
)

from src.models.factory import build_model

from src.train import (
    create_dataloaders,
    build_loss_function,
    build_optimizer,
    get_model_output,
    compute_segmentation_loss,
    batch_iou_from_logits,
)


# ============================================================
# BATCH HELPER
# ============================================================

def unpack_batch(batch):
    """
    Support both possible batch styles:

    1. Tuple/list:
        images, masks

    2. Dict:
        {
            "image": images,
            "mask": masks
        }
    """

    if isinstance(batch, dict):
        if "image" in batch:
            images = batch["image"]
        elif "images" in batch:
            images = batch["images"]
        else:
            raise KeyError(
                "Batch dictionary does not contain image/images key."
            )

        if "mask" in batch:
            masks = batch["mask"]
        elif "masks" in batch:
            masks = batch["masks"]
        elif "label" in batch:
            masks = batch["label"]
        elif "labels" in batch:
            masks = batch["labels"]
        else:
            raise KeyError(
                "Batch dictionary does not contain "
                "mask/masks/label/labels key."
            )

        return images, masks

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(
                "Batch tuple/list must contain at least images and masks."
            )

        return batch[0], batch[1]

    raise TypeError(
        f"Unsupported batch type: {type(batch)}"
    )


# ============================================================
# JSON HELPER
# ============================================================

def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# SPLIT CHECK
# ============================================================

def manifests_match_requested_split(
    output_folder,
    general_params,
):
    """
    Check whether existing manifests were generated with the
    currently requested Train / Validation percentages and seed.

    Returns:
        bool, desired_split, existing_split
    """

    output_folder = Path(output_folder)

    train_manifest = (
        output_folder
        / "train_manifest.csv"
    )

    val_manifest = (
        output_folder
        / "val_manifest.csv"
    )

    desired_split = (
        resolve_train_validation_split(
            general_params=general_params
        )
    )

    if (
        not train_manifest.exists()
        or not val_manifest.exists()
    ):
        return (
            False,
            desired_split,
            None,
        )

    split_summary_path = (
        output_folder
        / "split_summary.json"
    )

    existing_split = (
        load_json_if_exists(
            split_summary_path
        )
    )

    # Old manifests can still be used if their old manifest summary
    # proves they match the requested split.
    if existing_split is None:
        existing_split = (
            load_json_if_exists(
                output_folder
                / "manifest_summary.json"
            )
        )

    if existing_split is None:
        return (
            False,
            desired_split,
            None,
        )

    existing_train_percent = (
        existing_split.get(
            "train_percent",
            None,
        )
    )

    existing_validation_percent = (
        existing_split.get(
            "validation_percent",
            None,
        )
    )

    existing_validation_ratio = (
        existing_split.get(
            "validation_ratio",
            None,
        )
    )

    existing_seed = (
        existing_split.get(
            "seed",
            None,
        )
    )

    # Backward compatibility for old manifest_summary.json files
    # that only stored validation_ratio.
    if (
        existing_validation_percent is None
        and existing_validation_ratio is not None
    ):
        existing_validation_percent = (
            float(existing_validation_ratio)
            * 100.0
        )

    if (
        existing_train_percent is None
        and existing_validation_percent is not None
    ):
        existing_train_percent = (
            100.0
            - float(existing_validation_percent)
        )

    if (
        existing_train_percent is None
        or existing_validation_percent is None
        or existing_seed is None
    ):
        return (
            False,
            desired_split,
            existing_split,
        )

    matches = (
        abs(
            float(existing_train_percent)
            - float(desired_split["train_percent"])
        )
        < 1e-8

        and

        abs(
            float(existing_validation_percent)
            - float(desired_split["validation_percent"])
        )
        < 1e-8

        and

        int(existing_seed)
        == int(desired_split["seed"])
    )

    return (
        matches,
        desired_split,
        existing_split,
    )


# ============================================================
# MODEL FACTORY COMPATIBILITY
# ============================================================

def resolve_built_model(
    build_result
):
    """
    Handle different build_model() return styles:

        model
        model, device
        model, device, params
    """

    if isinstance(
        build_result,
        (tuple, list),
    ):
        if len(build_result) == 0:
            raise RuntimeError(
                "build_model() returned an empty tuple/list."
            )

        model = build_result[0]

        returned_device = (
            build_result[1]
            if len(build_result) > 1
            else None
        )

        returned_params = (
            build_result[2]
            if len(build_result) > 2
            else None
        )

    else:
        model = build_result
        returned_device = None
        returned_params = None

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "The first object returned by build_model() "
            "is not a torch.nn.Module."
        )

    if returned_device is not None:
        try:
            device = torch.device(
                str(returned_device)
            )
        except Exception:
            device = next(
                model.parameters()
            ).device
    else:
        try:
            device = next(
                model.parameters()
            ).device
        except StopIteration:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            model = model.to(device)

    if returned_params is None:
        params = sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
    else:
        try:
            params = int(
                returned_params
            )
        except Exception:
            params = sum(
                parameter.numel()
                for parameter
                in model.parameters()
                if parameter.requires_grad
            )

    return (
        model,
        device,
        params,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD CONFIGURATION
    # ========================================================

    config_path = (
        ROOT
        / "config"
        / "experiment.json"
    )

    experiment_config = (
        load_experiment_config(
            config_path
        )
    )

    general_params = (
        load_general_params(
            ROOT
        )
    )

    datasets_config = (
        load_datasets_config(
            ROOT
        )
    )

    model_type = (
        experiment_config[
            "model_type"
        ]
    )

    model_config = (
        load_model_config(
            model_type=model_type,
            root=ROOT,
        )
    )

    dataset_info = (
        get_dataset_info(
            experiment_config=
                experiment_config,
            datasets_config=
                datasets_config,
        )
    )

    backbone_info = (
        get_backbone_info(
            experiment_config=
                experiment_config,
            model_config=
                model_config,
        )
    )

    batch_size = (
        get_automatic_batch_size(
            experiment_config=
                experiment_config,
            backbone_info=
                backbone_info,
        )
    )

    output_folder = (
        create_experiment_output_folder(
            experiment_config=
                experiment_config,
            root=
                ROOT,
        )
    )

    # ========================================================
    # 2. RESOLVE REQUESTED TRAIN / VALIDATION SPLIT
    # ========================================================

    (
        manifests_match,
        desired_split,
        existing_split,
    ) = manifests_match_requested_split(
        output_folder=
            output_folder,
        general_params=
            general_params,
    )

    print()
    print(
        "========== TRAIN / VALIDATION SPLIT =========="
    )
    print(
        f"Training: "
        f"{desired_split['train_percent']:.2f}%"
    )
    print(
        f"Validation: "
        f"{desired_split['validation_percent']:.2f}%"
    )
    print(
        f"Seed: "
        f"{desired_split['seed']}"
    )
    print(
        "Test Included in Split: NO"
    )
    print(
        "Test Policy: separate source-specific geographic test area"
    )
    print(
        "================================================"
    )
    print()

    # ========================================================
    # 3. CREATE / REUSE MANIFESTS
    # ========================================================

    train_manifest_csv = (
        output_folder
        / "train_manifest.csv"
    )

    val_manifest_csv = (
        output_folder
        / "val_manifest.csv"
    )

    if not manifests_match:

        if (
            train_manifest_csv.exists()
            or val_manifest_csv.exists()
        ):
            print(
                "Existing manifests do not match the "
                "requested split configuration."
            )
            print(
                "Regenerating deterministic Train / Validation manifests..."
            )
        else:
            print(
                "Manifests not found. "
                "Creating Train / Validation manifests..."
            )

        manifest_result = (
            create_train_val_manifests(
                dataset_info=
                    dataset_info,
                output_folder=
                    output_folder,
                general_params=
                    general_params,
            )
        )

        train_manifest_csv = Path(
            manifest_result[
                "train_manifest_csv"
            ]
        )

        val_manifest_csv = Path(
            manifest_result[
                "val_manifest_csv"
            ]
        )

    else:
        print(
            "Existing Train / Validation manifests "
            "match the requested split."
        )

        print(
            "Reusing existing manifests."
        )

    # ========================================================
    # 4. DATA LOADERS
    # ========================================================

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    ) = create_dataloaders(
        train_manifest_csv=
            train_manifest_csv,
        val_manifest_csv=
            val_manifest_csv,
        batch_size=
            batch_size,
        num_workers=
            general_params[
                "training"
            ][
                "num_workers"
            ],
        pin_memory=
            general_params[
                "training"
            ][
                "pin_memory"
            ],
    )

    # ========================================================
    # 5. BUILD MODEL
    # ========================================================

    build_result = (
        build_model(
            experiment_config=
                experiment_config,
            model_config=
                model_config,
            backbone_info=
                backbone_info,
            move_to_device=
                True,
        )
    )

    (
        model,
        device,
        params,
    ) = resolve_built_model(
        build_result
    )

    # ========================================================
    # 6. LOSS + OPTIMIZER
    # ========================================================

    loss_fn = (
        build_loss_function(
            model_config
        )
    )

    optimizer = (
        build_optimizer(
            model=
                model,
            general_params=
                general_params,
            model_config=
                model_config,
        )
    )

    # ========================================================
    # 7. ONE TRAINING BATCH TEST
    # ========================================================

    batch = next(
        iter(
            train_loader
        )
    )

    images, masks = (
        unpack_batch(
            batch
        )
    )

    images = images.to(
        device
    )

    masks = masks.to(
        device
    )

    model.train()

    optimizer.zero_grad()

    outputs = model(
        images
    )

    aux_weight = getattr(
        loss_fn,
        "aux_weight",
        0.4,
    )

    loss = (
        compute_segmentation_loss(
            outputs=
                outputs,
            masks=
                masks,
            loss_fn=
                loss_fn,
            aux_weight=
                aux_weight,
        )
    )

    logits = (
        get_model_output(
            outputs
        )
    )

    loss.backward()

    optimizer.step()

    batch_iou = (
        batch_iou_from_logits(
            logits=
                logits,
            masks=
                masks,
            threshold=
                0.5,
        )
    )

    # ========================================================
    # 8. FINAL TEST OUTPUT
    # ========================================================

    print()
    print(
        "========== TRAINING SETUP TEST =========="
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Model Type: "
        f"{experiment_config['model_type']}"
    )

    print(
        f"Backbone: "
        f"{experiment_config['backbone']}"
    )

    print(
        f"Training Percentage: "
        f"{desired_split['train_percent']:.2f}%"
    )

    print(
        f"Validation Percentage: "
        f"{desired_split['validation_percent']:.2f}%"
    )

    print(
        f"Split Seed: "
        f"{desired_split['seed']}"
    )

    print(
        f"Train Samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Samples: "
        f"{len(val_dataset)}"
    )

    print(
        "Final Test Samples: "
        "NOT INCLUDED IN TRAIN / VALIDATION SPLIT"
    )

    print(
        f"Batch Size: "
        f"{batch_size}"
    )

    print(
        f"Trainable Parameters: "
        f"{params:,}"
    )

    print(
        f"Images Shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Masks Shape: "
        f"{tuple(masks.shape)}"
    )

    print(
        f"Logits Shape: "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Image Min: "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image Max: "
        f"{images.max().item():.4f}"
    )

    print(
        f"Mask Min: "
        f"{masks.min().item():.4f}"
    )

    print(
        f"Mask Max: "
        f"{masks.max().item():.4f}"
    )

    print(
        f"Loss: "
        f"{loss.item():.4f}"
    )

    print(
        f"Aux Weight: "
        f"{aux_weight}"
    )

    print(
        f"Batch IoU at threshold 0.5: "
        f"{batch_iou:.4f}"
    )

    if isinstance(
        outputs,
        dict,
    ):
        print(
            f"Model Output Keys: "
            f"{list(outputs.keys())}"
        )

        if (
            "aux" in outputs
            and outputs["aux"] is not None
        ):
            print(
                f"Aux Output Shape: "
                f"{tuple(outputs['aux'].shape)}"
            )

    print(
        "Training setup test completed successfully."
    )

    print(
        "=========================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
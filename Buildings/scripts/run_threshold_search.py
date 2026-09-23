"""
run_threshold_search.py

Canonical validation-threshold optimization runner for semantic
Building Extraction models.

Supported semantic models:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA

Architecture
------------
Threshold selection is performed ONLY on the held-out validation
manifest.

    TRAIN
      ↓
    learn weights
      ↓
    VALIDATION
      ↓
    search probability threshold
      ↓
    select best aggregate validation pixel IoU
      ↓
    save validation-selected threshold
      ↓
    FREEZE SETTINGS
      ↓
    EXTERNAL SOURCE-SPECIFIC TEST

The independent test image and its GT are NOT accessed by this script.

Important
---------
- Validation augmentation is ALWAYS OFF.
- Threshold search is shared across semantic models.
- U-Net / DeepLab use their standard full-model checkpoints.
- SAM-LoRA is reconstructed from:
      official SAM checkpoint
      + sam_lora.json
      + best_model.pth
- Final strict polygon IoU remains a separate final-test stage.
- The selected threshold must be frozen before the independent test.

Outputs
-------
<experiment>/validation_threshold_optimization/
    validation_threshold_search.csv
    validation_threshold_summary.json

<experiment>/
    validation_threshold_summary.json
    threshold_summary.json

`threshold_summary.json` is maintained for compatibility with the
shared inference helper.

PowerShell
----------
Default:
    & $py "Buildings\\scripts\\run_threshold_search.py"

Custom thresholds:
    & $py "Buildings\\scripts\\run_threshold_search.py" `
        --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.90

Custom checkpoint:
    & $py "Buildings\\scripts\\run_threshold_search.py" `
        --checkpoint "D:\\path\\to\\best_model.pth"

Smoke-test override for script testing only:
    & $py "Buildings\\scripts\\run_threshold_search.py" `
        --allow-smoke-checkpoint
"""

from pathlib import Path
import sys
import csv
import json
import argparse

import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOTS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

PROJECT_ROOT = ROOT.parent


if str(ROOT) not in sys.path:

    sys.path.append(
        str(ROOT)
    )


if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )


# ============================================================
# SHARED IMPORTS
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_model_config,
    get_backbone_info,
    get_automatic_batch_size,
    create_experiment_output_folder,
    save_json,
)

from src.models.factory import (
    build_model,
)

from src.inference import (
    load_model_checkpoint,
    get_model_output,
    normalize_logits_shape,
)

from src.train import (
    BuildingSegmentationDataset,
    unpack_batch,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Search the best semantic probability threshold "
            "using ONLY the validation split."
        )
    )


    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROOT
            / "config"
            / "experiment.json"
        ),
        help="Path to experiment configuration JSON.",
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path. "
            "Default: <experiment_output>/best_model.pth"
        ),
    )


    parser.add_argument(
        "--validation-manifest",
        type=str,
        default=None,
        help=(
            "Optional validation manifest path. "
            "Default: <experiment_output>/val_manifest.csv"
        ),
    )


    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Probability thresholds to evaluate. "
            "If omitted, use general_params.json -> "
            "thresholds -> semantic_thresholds."
        ),
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional validation batch-size override. "
            "Otherwise use the model-specific automatic policy."
        ),
    )


    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device override: cuda, cuda:0, cpu. "
            "Default: CUDA when available."
        ),
    )


    parser.add_argument(
        "--allow-smoke-checkpoint",
        action="store_true",
        help=(
            "Allow threshold optimization on a checkpoint produced by "
            "a short <=2 requested-epoch smoke-test run."
        ),
    )


    return parser.parse_args()


# ============================================================
# MODEL TYPE
# ============================================================

def normalize_model_type(
    model_type,
):

    value = str(
        model_type
    ).strip().lower()


    aliases = {
        # U-Net
        "u-net": "unet",
        "u_net": "unet",

        # DeepLab
        "deeplab": "deeplabv3",
        "deeplab_v3": "deeplabv3",
        "deep_lab_v3": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",

        # SAM-LoRA
        "sam-lora": "sam_lora",
        "sam lora": "sam_lora",
        "samlora": "sam_lora",
        "sam_lora": "sam_lora",
    }


    return aliases.get(
        value,
        value,
    )


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    requested_device=None,
):

    if requested_device is None:

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


    device = torch.device(
        requested_device
    )


    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA was requested but CUDA is not available."
        )


    return device


# ============================================================
# STANDARD MODEL BUILD RESULT
# ============================================================

def resolve_built_model(
    build_result,
):
    """
    Handle standard build_model() return formats.

    Supported:
        model
        (model, device)
        (model, ...)
    """

    if isinstance(
        build_result,
        (
            tuple,
            list,
        ),
    ):

        if len(
            build_result
        ) == 0:

            raise RuntimeError(
                "build_model() returned an empty tuple/list."
            )


        model = (
            build_result[0]
        )


    else:

        model = (
            build_result
        )


    if not isinstance(
        model,
        torch.nn.Module,
    ):

        raise TypeError(
            "The first object returned by build_model() "
            "is not a torch.nn.Module. "
            f"Received: {type(model)}"
        )


    try:

        device = next(
            model.parameters()
        ).device


    except StopIteration:

        device = resolve_device()

        model = model.to(
            device
        )


    trainable_params = sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    return (
        model,
        device,
        trainable_params,
    )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json_if_exists(
    path,
):

    path = Path(
        path
    )


    if not path.exists():

        return None


    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# THRESHOLD RESOLUTION
# ============================================================

def get_validation_thresholds(
    args,
    general_params,
):
    """
    Resolve thresholds from CLI or general_params.json.
    """

    if args.thresholds is not None:

        thresholds = (
            args.thresholds
        )


    else:

        thresholds_config = (
            general_params.get(
                "thresholds",
                {},
            )
        )


        thresholds = (
            thresholds_config.get(
                "semantic_thresholds",
                None,
            )
        )


        if thresholds is None:

            thresholds = [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.70,
                0.80,
                0.90,
            ]


    thresholds = sorted(
        {
            round(
                float(
                    value
                ),
                4,
            )

            for value
            in thresholds
        }
    )


    if not thresholds:

        raise ValueError(
            "No thresholds were provided."
        )


    for threshold in thresholds:

        if not (
            0.0
            < threshold
            < 1.0
        ):

            raise ValueError(
                f"Invalid threshold: {threshold}. "
                "Threshold must satisfy 0 < threshold < 1."
            )


    return thresholds


# ============================================================
# SMOKE-TEST PROTECTION
# ============================================================

def resolve_requested_epochs(
    training_summary,
):
    """
    Resolve requested epochs across historical and SAM-LoRA summaries.
    """

    if not isinstance(
        training_summary,
        dict,
    ):

        return None


    for key in (
        "epochs_requested",
        "epochs",
        "requested_epochs",
    ):

        value = (
            training_summary.get(
                key,
                None,
            )
        )


        if value is not None:

            return int(
                value
            )


    return None


def check_checkpoint_is_not_smoke_test(
    output_folder,
    checkpoint_path,
    allow_smoke_checkpoint=False,
):
    """
    Protect final threshold selection from a <=2-epoch temporary run.
    """

    output_folder = Path(
        output_folder
    )


    checkpoint_path = Path(
        checkpoint_path
    )


    default_checkpoint = (
        output_folder
        / "best_model.pth"
    )


    if (
        checkpoint_path.resolve()
        != default_checkpoint.resolve()
    ):

        return


    training_summary = (
        load_json_if_exists(
            output_folder
            / "training_summary.json"
        )
    )


    requested_epochs = (
        resolve_requested_epochs(
            training_summary
        )
    )


    if requested_epochs is None:

        return


    if (
        requested_epochs <= 2
        and not allow_smoke_checkpoint
    ):

        raise RuntimeError(
            "\nThe current experiment best_model.pth appears to come from "
            f"a {requested_epochs}-epoch smoke-test run.\n\n"
            f"Checkpoint:\n{checkpoint_path}\n\n"
            "Do NOT select the final threshold on this temporary model.\n"
            "Train/restore the intended full model first, or pass "
            "--allow-smoke-checkpoint only when testing this script."
        )


# ============================================================
# CSV
# ============================================================

def save_results_csv(
    rows,
    path,
):

    path = Path(
        path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fieldnames = [
        "threshold",

        "iou",
        "iou_percent",

        "precision",
        "precision_percent",

        "recall",
        "recall_percent",

        "f1",
        "f1_percent",

        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
        "tn_pixels",

        "pred_positive_pixels",
        "gt_positive_pixels",
    ]


    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        for row in rows:

            writer.writerow(
                {
                    field:
                        row.get(
                            field
                        )

                    for field
                    in fieldnames
                }
            )


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

@torch.no_grad()
def search_thresholds_on_validation(
    model,
    val_loader,
    device,
    thresholds,
):
    """
    Search thresholds using aggregate validation pixel metrics.

    Counts are accumulated across all validation pixels BEFORE metric
    calculation.

    Model output is normalized through the shared inference adapter:

        Tensor
        {"out": Tensor}
        {"logits": Tensor}
    """

    model.eval()


    totals = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

        for threshold
        in thresholds
    }


    total_batches = len(
        val_loader
    )


    for batch_index, batch in enumerate(
        val_loader,
        start=1,
    ):

        images, masks = (
            unpack_batch(
                batch
            )
        )


        images = images.to(
            device,
            non_blocking=True,
        )


        masks = masks.to(
            device,
            non_blocking=True,
        )


        masks_binary = (
            masks
            >= 0.5
        )


        outputs = model(
            images
        )


        logits = get_model_output(
            outputs
        )


        logits = normalize_logits_shape(
            logits,
            target_height=int(
                masks.shape[
                    -2
                ]
            ),
            target_width=int(
                masks.shape[
                    -1
                ]
            ),
        )


        probabilities = torch.sigmoid(
            logits
        )


        for threshold in thresholds:

            predictions = (
                probabilities
                >= float(
                    threshold
                )
            )


            tp = torch.sum(
                predictions
                & masks_binary
            ).item()


            fp = torch.sum(
                predictions
                & ~masks_binary
            ).item()


            fn = torch.sum(
                ~predictions
                & masks_binary
            ).item()


            tn = torch.sum(
                ~predictions
                & ~masks_binary
            ).item()


            totals[
                threshold
            ][
                "tp"
            ] += int(
                tp
            )


            totals[
                threshold
            ][
                "fp"
            ] += int(
                fp
            )


            totals[
                threshold
            ][
                "fn"
            ] += int(
                fn
            )


            totals[
                threshold
            ][
                "tn"
            ] += int(
                tn
            )


        if (
            batch_index % 10 == 0
            or batch_index == total_batches
        ):

            print(
                "Processed validation batches:",
                f"{batch_index}/{total_batches}",
            )


    rows = []

    eps = 1e-7


    for threshold in thresholds:

        tp = float(
            totals[
                threshold
            ][
                "tp"
            ]
        )


        fp = float(
            totals[
                threshold
            ][
                "fp"
            ]
        )


        fn = float(
            totals[
                threshold
            ][
                "fn"
            ]
        )


        tn = float(
            totals[
                threshold
            ][
                "tn"
            ]
        )


        iou = (
            tp
            / (
                tp
                + fp
                + fn
                + eps
            )
        )


        precision = (
            tp
            / (
                tp
                + fp
                + eps
            )
        )


        recall = (
            tp
            / (
                tp
                + fn
                + eps
            )
        )


        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
                + eps
            )
        )


        row = {
            "threshold":
                float(
                    threshold
                ),

            "iou":
                float(
                    iou
                ),

            "iou_percent":
                float(
                    iou
                    * 100.0
                ),

            "precision":
                float(
                    precision
                ),

            "precision_percent":
                float(
                    precision
                    * 100.0
                ),

            "recall":
                float(
                    recall
                ),

            "recall_percent":
                float(
                    recall
                    * 100.0
                ),

            "f1":
                float(
                    f1
                ),

            "f1_percent":
                float(
                    f1
                    * 100.0
                ),

            "tp_pixels":
                int(
                    tp
                ),

            "fp_pixels":
                int(
                    fp
                ),

            "fn_pixels":
                int(
                    fn
                ),

            "tn_pixels":
                int(
                    tn
                ),

            "pred_positive_pixels":
                int(
                    tp
                    + fp
                ),

            "gt_positive_pixels":
                int(
                    tp
                    + fn
                ),
        }


        rows.append(
            row
        )


    # --------------------------------------------------------
    # Primary criterion:
    #   maximum aggregate validation pixel IoU
    #
    # Tie-breakers:
    #   1. F1
    #   2. threshold closest to 0.5
    # --------------------------------------------------------

    best_row = max(
        rows,

        key=lambda row: (
            row[
                "iou"
            ],

            row[
                "f1"
            ],

            -abs(
                row[
                    "threshold"
                ]
                - 0.5
            ),
        ),
    )


    return (
        rows,
        best_row,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    # ========================================================
    # 1. LOAD CONFIGURATION
    # ========================================================

    experiment_config = (
        load_experiment_config(
            Path(
                args.config
            )
        )
    )


    general_params = (
        load_general_params(
            ROOT
        )
    )


    model_type = normalize_model_type(
        experiment_config[
            "model_type"
        ]
    )


    if model_type not in {
        "unet",
        "deeplabv3",
        "sam_lora",
    }:

        raise ValueError(
            "run_threshold_search.py is for semantic models only "
            "(U-Net / DeepLabV3 / SAM-LoRA). "
            f"Current model_type: {model_type}"
        )


    # ========================================================
    # 2. EXPERIMENT PATHS
    # ========================================================

    output_folder = (
        create_experiment_output_folder(
            experiment_config=
                experiment_config,

            root=
                ROOT,
        )
    )


    output_folder = Path(
        output_folder
    )


    checkpoint_path = (
        Path(
            args.checkpoint
        )

        if args.checkpoint
        is not None

        else (
            output_folder
            / "best_model.pth"
        )
    )


    val_manifest_csv = (
        Path(
            args.validation_manifest
        )

        if args.validation_manifest
        is not None

        else (
            output_folder
            / "val_manifest.csv"
        )
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "\nTrained checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    if not val_manifest_csv.exists():

        raise FileNotFoundError(
            "\nValidation manifest not found:\n"
            f"{val_manifest_csv}"
        )


    # ========================================================
    # 3. SMOKE PROTECTION
    # ========================================================

    check_checkpoint_is_not_smoke_test(
        output_folder=
            output_folder,

        checkpoint_path=
            checkpoint_path,

        allow_smoke_checkpoint=
            args.allow_smoke_checkpoint,
    )


    # ========================================================
    # 4. MODEL CONFIG
    # ========================================================

    model_config = (
        load_model_config(
            model_type,
            ROOT,
        )
    )


    requested_device = (
        resolve_device(
            args.device
        )
    )


    checkpoint_epoch = None

    checkpoint_val_iou = None

    model_loader = None

    official_sam_checkpoint = None

    checkpoint_format = None


    # ========================================================
    # 5. BUILD / RECONSTRUCT MODEL
    # ========================================================

    if model_type == "sam_lora":

        from src.models.sam_lora.inference import (
            load_trained_sam_lora,
        )


        from src.models.sam_lora.factory import (
            resolve_sam_lora_batch_size,
        )


        sam_config_path = (
            ROOT
            / "config"
            / "models"
            / "sam_lora.json"
        )


        (
            model,
            load_info,
        ) = (
            load_trained_sam_lora(
                config_path=
                    sam_config_path,

                trained_checkpoint_path=
                    checkpoint_path,

                device=
                    requested_device,

                verbose=
                    True,
            )
        )


        device = next(
            model.parameters()
        ).device


        trainable_params = sum(
            parameter.numel()

            for parameter
            in model.parameters()

            if parameter.requires_grad
        )


        checkpoint_epoch = (
            load_info.trained_epoch
        )


        validation_metrics = (
            load_info.validation_metrics
            or {}
        )


        checkpoint_val_iou = (
            validation_metrics.get(
                "iou",

                validation_metrics.get(
                    "val_iou",

                    validation_metrics.get(
                        "validation_iou",
                        None,
                    ),
                ),
            )
        )


        official_sam_checkpoint = (
            load_info.pretrained_sam_checkpoint_path
        )


        checkpoint_format = (
            load_info.checkpoint_format
        )


        model_loader = (
            "sam_lora_reconstruction"
        )


        if args.batch_size is not None:

            batch_size = int(
                args.batch_size
            )


        else:

            batch_size = (
                resolve_sam_lora_batch_size(
                    config=
                        model_config,

                    tile_size=
                        int(
                            experiment_config.get(
                                "tile_size",
                                256,
                            )
                        ),
                )
            )


    else:

        backbone_info = (
            get_backbone_info(
                experiment_config=
                    experiment_config,

                model_config=
                    model_config,
            )
        )


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
            trainable_params,
        ) = (
            resolve_built_model(
                build_result
            )
        )


        if (
            device
            != requested_device
        ):

            model = model.to(
                requested_device
            )

            device = (
                requested_device
            )


        (
            model,
            checkpoint,
        ) = (
            load_model_checkpoint(
                model=
                    model,

                checkpoint_path=
                    checkpoint_path,

                device=
                    device,
            )
        )


        if isinstance(
            checkpoint,
            dict,
        ):

            checkpoint_epoch = (
                checkpoint.get(
                    "epoch",
                    None,
                )
            )


            checkpoint_val_iou = (
                checkpoint.get(
                    "best_val_iou",

                    checkpoint.get(
                        "best_validation_iou",
                        None,
                    ),
                )
            )


        model_loader = (
            "standard_shared_checkpoint_loader"
        )


        if args.batch_size is not None:

            batch_size = int(
                args.batch_size
            )


        else:

            batch_size = (
                get_automatic_batch_size(
                    experiment_config=
                        experiment_config,

                    backbone_info=
                        backbone_info,

                    model_config=
                        model_config,
                )
            )


    model.eval()


    if int(
        batch_size
    ) <= 0:

        raise ValueError(
            "Validation batch size must be > 0."
        )


    # ========================================================
    # 6. VALIDATION DATA
    # ========================================================

    val_dataset = (
        BuildingSegmentationDataset(
            manifest_csv=
                val_manifest_csv,

            augmentation=
                None,
        )
    )


    val_loader = DataLoader(
        val_dataset,

        batch_size=
            int(
                batch_size
            ),

        shuffle=
            False,

        num_workers=
            int(
                general_params.get(
                    "training",
                    {},
                ).get(
                    "num_workers",
                    0,
                )
            ),

        pin_memory=
            bool(
                general_params.get(
                    "training",
                    {},
                ).get(
                    "pin_memory",
                    True,
                )
            ),

        drop_last=
            False,
    )


    # ========================================================
    # 7. THRESHOLDS
    # ========================================================

    thresholds = (
        get_validation_thresholds(
            args=
                args,

            general_params=
                general_params,
        )
    )


    # ========================================================
    # 8. OUTPUT PATHS
    # ========================================================

    search_root = (
        output_folder
        / "validation_threshold_optimization"
    )


    search_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_csv = (
        search_root
        / "validation_threshold_search.csv"
    )


    detailed_summary_json = (
        search_root
        / "validation_threshold_summary.json"
    )


    validation_summary_json = (
        output_folder
        / "validation_threshold_summary.json"
    )


    compatibility_summary_json = (
        output_folder
        / "threshold_summary.json"
    )


    # ========================================================
    # 9. DISPLAY
    # ========================================================

    display_backbone = (
        experiment_config.get(
            "backbone",
            model_config.get(
                "default_backbone",
                "unknown",
            ),
        )
    )


    print()

    print(
        "=" * 76
    )

    print(
        "VALIDATION THRESHOLD OPTIMIZATION"
    )

    print(
        "=" * 76
    )


    print(
        "Experiment ID:",
        experiment_config[
            "experiment_id"
        ],
    )


    print(
        "Model Type:",
        model_type,
    )


    print(
        "Backbone:",
        display_backbone,
    )


    print(
        "Source Type:",
        experiment_config[
            "source_type"
        ],
    )


    print(
        "Training Augmentation:",
        (
            "ON"
            if experiment_config.get(
                "use_augmentation",
                False,
            )
            else "OFF"
        ),
    )


    print(
        "Validation Augmentation:",
        "OFF",
    )


    print(
        "Checkpoint:",
        checkpoint_path,
    )


    if checkpoint_epoch is not None:

        print(
            "Checkpoint Epoch:",
            checkpoint_epoch,
        )


    if checkpoint_val_iou is not None:

        print(
            "Checkpoint Validation IoU:",
            f"{float(checkpoint_val_iou):.4f}",
        )


    print(
        "Model Loader:",
        model_loader,
    )


    if model_type == "sam_lora":

        print(
            "Official SAM Checkpoint:",
            official_sam_checkpoint,
        )


        print(
            "SAM-LoRA Checkpoint Format:",
            checkpoint_format,
        )


    print(
        "Device:",
        device,
    )


    print(
        "Trainable Parameters:",
        f"{trainable_params:,}",
    )


    print(
        "Validation Manifest:",
        val_manifest_csv,
    )


    print(
        "Validation Samples:",
        len(
            val_dataset
        ),
    )


    print(
        "Batch Size:",
        batch_size,
    )


    print(
        "Thresholds:",
        thresholds,
    )


    print(
        "-" * 76
    )


    print(
        "Selection Dataset: VALIDATION ONLY"
    )


    print(
        "Selection Metric: aggregate validation pixel IoU"
    )


    print(
        "External Test Area Used: NO"
    )


    print(
        "=" * 76
    )

    print()


    # ========================================================
    # 10. SEARCH
    # ========================================================

    (
        rows,
        best_row,
    ) = (
        search_thresholds_on_validation(
            model=
                model,

            val_loader=
                val_loader,

            device=
                device,

            thresholds=
                thresholds,
        )
    )


    # ========================================================
    # 11. DISPLAY RESULTS
    # ========================================================

    print()

    print(
        "=" * 76
    )

    print(
        "VALIDATION THRESHOLD RESULTS"
    )

    print(
        "=" * 76
    )


    for row in rows:

        print(
            f"Threshold: "
            f"{row['threshold']:.2f} | "

            f"IoU: "
            f"{row['iou_percent']:.2f}% | "

            f"P: "
            f"{row['precision_percent']:.2f}% | "

            f"R: "
            f"{row['recall_percent']:.2f}% | "

            f"F1: "
            f"{row['f1_percent']:.2f}%"
        )


    print(
        "=" * 76
    )

    print()


    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================

    save_results_csv(
        rows=
            rows,

        path=
            results_csv,
    )


    threshold_050 = next(
        (
            row

            for row
            in rows

            if abs(
                row[
                    "threshold"
                ]
                - 0.50
            )
            < 1e-9
        ),
        None,
    )


    improvement_vs_050 = None


    if threshold_050 is not None:

        improvement_vs_050 = (
            best_row[
                "iou_percent"
            ]
            -
            threshold_050[
                "iou_percent"
            ]
        )


    final_summary = {
        "experiment_id":
            experiment_config[
                "experiment_id"
            ],

        "model_type":
            model_type,

        "backbone":
            display_backbone,

        "source_type":
            experiment_config[
                "source_type"
            ],

        "tile_size":
            int(
                experiment_config[
                    "tile_size"
                ]
            ),

        "training_augmentation":
            bool(
                experiment_config.get(
                    "use_augmentation",
                    False,
                )
            ),

        "validation_augmentation":
            False,

        "selection_dataset":
            "validation_manifest",

        "validation_manifest":
            str(
                val_manifest_csv
            ),

        "validation_samples":
            int(
                len(
                    val_dataset
                )
            ),

        "external_test_area_used":
            False,

        "selection_metric":
            "aggregate_validation_pixel_iou",

        "thresholds_tested":
            thresholds,

        "best_threshold":
            float(
                best_row[
                    "threshold"
                ]
            ),

        "best_validation_iou":
            float(
                best_row[
                    "iou"
                ]
            ),

        "best_validation_iou_percent":
            float(
                best_row[
                    "iou_percent"
                ]
            ),

        "best_precision":
            float(
                best_row[
                    "precision"
                ]
            ),

        "best_precision_percent":
            float(
                best_row[
                    "precision_percent"
                ]
            ),

        "best_recall":
            float(
                best_row[
                    "recall"
                ]
            ),

        "best_recall_percent":
            float(
                best_row[
                    "recall_percent"
                ]
            ),

        "best_f1":
            float(
                best_row[
                    "f1"
                ]
            ),

        "best_f1_percent":
            float(
                best_row[
                    "f1_percent"
                ]
            ),

        "threshold_050_iou_percent":
            (
                float(
                    threshold_050[
                        "iou_percent"
                    ]
                )
                if threshold_050
                is not None
                else None
            ),

        "iou_improvement_vs_threshold_050_pp":
            (
                float(
                    improvement_vs_050
                )
                if improvement_vs_050
                is not None
                else None
            ),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "checkpoint_epoch":
            (
                int(
                    checkpoint_epoch
                )
                if checkpoint_epoch
                is not None
                else None
            ),

        "checkpoint_validation_iou":
            (
                float(
                    checkpoint_val_iou
                )
                if checkpoint_val_iou
                is not None
                else None
            ),

        "model_loader":
            model_loader,

        "official_sam_checkpoint":
            (
                str(
                    official_sam_checkpoint
                )
                if official_sam_checkpoint
                is not None
                else None
            ),

        "sam_lora_checkpoint_format":
            checkpoint_format,

        "results_csv":
            str(
                results_csv
            ),

        "methodology_note":
            (
                "Probability threshold selected exclusively on the "
                "held-out validation manifest using aggregate pixel IoU. "
                "Validation augmentation is disabled. The independent "
                "source-specific external test area is not accessed. "
                "The selected threshold must be frozen before final test."
            ),
    }


    save_json(
        final_summary,
        detailed_summary_json,
    )


    save_json(
        final_summary,
        validation_summary_json,
    )


    # Compatibility copy used by src.inference.get_best_threshold().
    save_json(
        final_summary,
        compatibility_summary_json,
    )


    # ========================================================
    # 13. FINAL DISPLAY
    # ========================================================

    print()

    print(
        "=" * 76
    )

    print(
        "VALIDATION THRESHOLD OPTIMIZATION FINISHED"
    )

    print(
        "=" * 76
    )


    print(
        "Best Threshold:",
        f"{best_row['threshold']:.2f}",
    )


    print(
        "Best Validation Pixel IoU:",
        f"{best_row['iou_percent']:.2f}%",
    )


    print(
        "Precision:",
        f"{best_row['precision_percent']:.2f}%",
    )


    print(
        "Recall:",
        f"{best_row['recall_percent']:.2f}%",
    )


    print(
        "F1:",
        f"{best_row['f1_percent']:.2f}%",
    )


    if threshold_050 is not None:

        print(
            "-" * 76
        )


        print(
            "Threshold 0.50 Validation IoU:",
            f"{threshold_050['iou_percent']:.2f}%",
        )


        print(
            "IoU Improvement vs 0.50:",
            f"{improvement_vs_050:+.2f} pp",
        )


    print(
        "-" * 76
    )


    print(
        "External Test Area Used:",
        "NO",
    )


    print(
        "Results CSV:",
        results_csv,
    )


    print(
        "Detailed Summary:",
        detailed_summary_json,
    )


    print(
        "Validation Threshold Summary:",
        validation_summary_json,
    )


    print(
        "Shared Threshold Summary:",
        compatibility_summary_json,
    )


    print(
        "=" * 76
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

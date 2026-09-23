"""
freeze_final_settings.py

Create one frozen final_settings.json for the Building Extraction module.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- Mask R-CNN where applicable
- SAM-LoRA

Purpose
-------
After model/settings selection has finished on TRAIN / VALIDATION only,
this script collects the selected settings into one immutable-style
configuration for the independent source-specific final test.

Architecture
------------
    TRAIN
      ↓
    learn weights
      ↓
    VALIDATION
      ├─ best epoch
      ├─ Optuna (optional)
      ├─ threshold selection (optional)
      └─ post-processing selection (optional)
              ↓
        FREEZE SETTINGS
              ↓
        final_settings.json
              ↓
    EXTERNAL SOURCE-SPECIFIC TEST

Important
---------
- This script DOES NOT run inference.
- This script DOES NOT open the final test raster or GT.
- Threshold/post-processing summaries must come from validation.
- A <=2 requested-epoch smoke-test checkpoint is blocked by default.
- Existing final_settings.json is not overwritten unless --overwrite.
- SAM-LoRA-specific architecture metadata is recorded when applicable.
"""

from pathlib import Path
import sys
import json
import argparse
from datetime import datetime, timezone


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_model_config,
    create_experiment_output_folder,
    resolve_train_validation_split,
    save_json,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the selected training/validation settings "
            "before the final external test."
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
            "Checkpoint to freeze. "
            "Default: <experiment_output>/best_model.pth"
        ),
    )

    parser.add_argument(
        "--training-summary",
        type=str,
        default=None,
        help=(
            "Training summary JSON. "
            "Default: <experiment_output>/training_summary.json"
        ),
    )

    parser.add_argument(
        "--threshold-summary",
        type=str,
        default=None,
        help=(
            "Validation threshold summary JSON. "
            "Default: <experiment_output>/validation_threshold_summary.json"
        ),
    )

    parser.add_argument(
        "--postprocess-summary",
        type=str,
        default=None,
        help=(
            "Validation post-processing summary JSON. "
            "Default: <experiment_output>/validation_postprocess_summary.json"
        ),
    )

    parser.add_argument(
        "--optuna-summary",
        type=str,
        default=None,
        help=(
            "Optional Optuna summary JSON. "
            "Use this when the final selected checkpoint comes from Optuna."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional final settings output path. "
            "Default: <experiment_output>/final_settings.json"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing final_settings.json.",
    )

    parser.add_argument(
        "--allow-smoke-checkpoint",
        action="store_true",
        help=(
            "Allow freezing a <=2 requested-epoch smoke-test checkpoint. "
            "Use only for code testing."
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
        "u-net": "unet",
        "u_net": "unet",

        "deeplab": "deeplabv3",
        "deeplab_v3": "deeplabv3",
        "deep_lab_v3": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",

        "mask_rcnn": "maskrcnn",
        "mask-r-cnn": "maskrcnn",
        "mask r-cnn": "maskrcnn",
        "rcnn": "maskrcnn",

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
# JSON HELPERS
# ============================================================

def load_json_file(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nJSON file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


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
# VALUE HELPERS
# ============================================================

def first_not_none(
    *values,
):

    for value in values:

        if value is not None:

            return value

    return None


def get_default_semantic_threshold(
    experiment_config,
):
    """
    Fallback threshold only when threshold search is disabled.
    """

    threshold = float(
        first_not_none(
            experiment_config.get(
                "inference_threshold"
            ),
            experiment_config.get(
                "threshold"
            ),
            0.5,
        )
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Default inference threshold must be between 0 and 1."
        )

    return threshold


def resolve_best_epoch(
    training_summary,
):

    if not isinstance(
        training_summary,
        dict,
    ):

        return None

    value = first_not_none(
        training_summary.get(
            "best_epoch"
        ),
        training_summary.get(
            "epoch"
        ),
    )

    if value is None:

        return None

    return int(
        value
    )


def resolve_best_validation_iou(
    training_summary,
):

    if not isinstance(
        training_summary,
        dict,
    ):

        return None

    value = first_not_none(
        training_summary.get(
            "best_val_iou"
        ),
        training_summary.get(
            "best_validation_iou"
        ),
        training_summary.get(
            "validation_iou"
        ),
    )

    if value is None:

        percent_value = first_not_none(
            training_summary.get(
                "best_val_iou_percent"
            ),
            training_summary.get(
                "best_validation_iou_percent"
            ),
        )

        if percent_value is None:

            return None

        return (
            float(
                percent_value
            )
            / 100.0
        )

    value = float(
        value
    )

    if value > 1.0:

        value = (
            value
            / 100.0
        )

    return value


def resolve_requested_epochs(
    training_summary,
):
    """
    Resolve requested epochs across standard and SAM-LoRA summaries.
    """

    if not isinstance(
        training_summary,
        dict,
    ):

        return None

    return first_not_none(
        training_summary.get(
            "epochs_requested"
        ),
        training_summary.get(
            "epochs"
        ),
        training_summary.get(
            "requested_epochs"
        ),
    )


# ============================================================
# VALIDATION-SUMMARY SAFETY
# ============================================================

def validate_validation_only_summary(
    summary,
    summary_name,
):
    """
    Reject summaries that indicate use of the external test area.
    """

    if not isinstance(
        summary,
        dict,
    ):

        raise TypeError(
            f"{summary_name} must contain a JSON object."
        )

    external_test_used = summary.get(
        "external_test_area_used",
        None,
    )

    if external_test_used is True:

        raise RuntimeError(
            f"\n{summary_name} indicates external_test_area_used=true.\n"
            "Only validation-selected settings can be frozen."
        )

    selection_dataset = str(
        summary.get(
            "selection_dataset",
            "",
        )
    ).lower()

    methodology_note = str(
        summary.get(
            "methodology_note",
            "",
        )
    ).lower()

    legacy_test_markers = [
        "configured full test scene",
        "optimized test-scene",
        "test-scene evaluation",
    ]

    if any(
        marker in methodology_note
        for marker
        in legacy_test_markers
    ):

        raise RuntimeError(
            f"\n{summary_name} appears to come from the legacy "
            "test-scene optimization workflow.\n"
            "Run the validation-based optimization first."
        )

    if (
        selection_dataset
        and "validation"
        not in selection_dataset
    ):

        raise RuntimeError(
            f"\n{summary_name} was not selected on validation data.\n"
            f"selection_dataset={selection_dataset}"
        )


# ============================================================
# SMOKE-TEST PROTECTION
# ============================================================

def check_smoke_checkpoint(
    checkpoint_path,
    training_summary,
    allow_smoke_checkpoint=False,
):
    """
    Prevent accidental freezing of a temporary <=2 requested-epoch model.
    """

    if allow_smoke_checkpoint:

        return

    requested_epochs = (
        resolve_requested_epochs(
            training_summary
        )
    )

    if requested_epochs is None:

        return

    requested_epochs = int(
        requested_epochs
    )

    if requested_epochs <= 2:

        raise RuntimeError(
            "\nRefusing to freeze the current checkpoint because the "
            f"training summary reports only {requested_epochs} requested "
            "epochs.\n\n"
            f"Checkpoint:\n{checkpoint_path}\n\n"
            "This looks like a temporary smoke-test model. "
            "Restore/retrain the intended full model first.\n"
            "Use --allow-smoke-checkpoint only to test this script."
        )


# ============================================================
# OPTUNA HELPERS
# ============================================================

def extract_optuna_metadata(
    optuna_summary,
):

    if not isinstance(
        optuna_summary,
        dict,
    ):

        return {
            "used": False,
            "summary_path": None,
            "study_name": None,
            "best_trial": None,
            "best_validation_iou": None,
            "best_params": None,
        }

    best_trial = first_not_none(
        optuna_summary.get(
            "best_trial"
        ),
        optuna_summary.get(
            "best_trial_number"
        ),
    )

    best_value = first_not_none(
        optuna_summary.get(
            "best_validation_iou"
        ),
        optuna_summary.get(
            "best_value"
        ),
    )

    if best_value is not None:

        best_value = float(
            best_value
        )

        if best_value > 1.0:

            best_value = (
                best_value
                / 100.0
            )

    best_params = first_not_none(
        optuna_summary.get(
            "best_params"
        ),
        (
            optuna_summary.get(
                "final_summary",
                {}
            ).get(
                "best_params"
            )
            if isinstance(
                optuna_summary.get(
                    "final_summary"
                ),
                dict,
            )
            else None
        ),
    )

    return {
        "used": True,

        "study_name":
            optuna_summary.get(
                "study_name"
            ),

        "best_trial":
            (
                int(
                    best_trial
                )
                if best_trial
                is not None
                else None
            ),

        "best_validation_iou":
            best_value,

        "best_params":
            best_params,
    }


# ============================================================
# SAM-LORA METADATA
# ============================================================

def build_sam_lora_metadata(
    model_type,
    model_config,
    training_summary,
):
    """
    Record the architecture/training settings required to reconstruct
    the frozen SAM-LoRA model later.

    No model is loaded and no external test data is accessed here.
    """

    if model_type != "sam_lora":

        return {
            "enabled":
                False,
        }

    backbone = str(
        model_config.get(
            "default_backbone",
            "vit_b",
        )
    )

    supported_backbones = (
        model_config.get(
            "supported_backbones",
            {},
        )
    )

    backbone_config = (
        supported_backbones.get(
            backbone,
            {},
        )
        if isinstance(
            supported_backbones,
            dict,
        )
        else {}
    )

    lora_config = dict(
        model_config.get(
            "lora",
            {},
        )
    )

    freeze_config = dict(
        model_config.get(
            "freeze",
            {},
        )
    )

    prompt_config = dict(
        model_config.get(
            "prompt_strategy",
            {},
        )
    )

    training_config = dict(
        model_config.get(
            "training",
            {},
        )
    )

    loss_config = dict(
        training_config.get(
            "loss",
            {},
        )
    )

    learning_rate_config = dict(
        training_config.get(
            "learning_rate",
            {},
        )
    )

    return {
        "enabled":
            True,

        "backbone":
            backbone,

        "official_sam_checkpoint":
            backbone_config.get(
                "checkpoint"
            ),

        "checkpoint_storage_mode":
            model_config.get(
                "checkpoint",
                {},
            ).get(
                "save_mode",
                "lora_and_decoder",
            ),

        "lora":
            lora_config,

        "freeze":
            freeze_config,

        "prompt_strategy":
            prompt_config,

        "loss":
            loss_config,

        "learning_rate":
            {
                "configured_mode":
                    learning_rate_config.get(
                        "mode",
                        "auto",
                    ),

                "resolved_learning_rate":
                    first_not_none(
                        training_summary.get(
                            "resolved_learning_rate"
                        ),
                        training_summary.get(
                            "learning_rate"
                        ),
                    ),

                "finder":
                    learning_rate_config.get(
                        "finder"
                    ),
            },

        "sam_input_size":
            model_config.get(
                "input",
                {},
            ).get(
                "sam_input_size",
                1024,
            ),

        "input_channels":
            model_config.get(
                "input",
                {},
            ).get(
                "channels",
                3,
            ),

        "preserve_dataset_tile_size_for_output":
            model_config.get(
                "input",
                {},
            ).get(
                "preserve_dataset_tile_size_for_output",
                True,
            ),
    }


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
        experiment_config.get(
            "model_type"
        )
    )

    model_config = (
        load_model_config(
            model_type,
            ROOT,
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

    output_folder = Path(
        output_folder
    )

    # ========================================================
    # 2. PATHS
    # ========================================================

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

    training_summary_path = (
        Path(
            args.training_summary
        )

        if args.training_summary
        is not None

        else (
            output_folder
            / "training_summary.json"
        )
    )

    threshold_summary_path = (
        Path(
            args.threshold_summary
        )

        if args.threshold_summary
        is not None

        else (
            output_folder
            / "validation_threshold_summary.json"
        )
    )

    postprocess_summary_path = (
        Path(
            args.postprocess_summary
        )

        if args.postprocess_summary
        is not None

        else (
            output_folder
            / "validation_postprocess_summary.json"
        )
    )

    final_settings_path = (
        Path(
            args.output
        )

        if args.output
        is not None

        else (
            output_folder
            / "final_settings.json"
        )
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{checkpoint_path}"
        )

    if (
        final_settings_path.exists()
        and not args.overwrite
    ):

        raise FileExistsError(
            "\nFrozen settings already exist:\n"
            f"{final_settings_path}\n\n"
            "Use --overwrite only if you intentionally want "
            "to replace the frozen settings."
        )

    # ========================================================
    # 3. LOAD TRAINING SUMMARY
    # ========================================================

    training_summary = (
        load_json_file(
            training_summary_path
        )
    )

    check_smoke_checkpoint(
        checkpoint_path=
            checkpoint_path,

        training_summary=
            training_summary,

        allow_smoke_checkpoint=
            args.allow_smoke_checkpoint,
    )

    # ========================================================
    # 4. SPLIT
    # ========================================================

    split_info = (
        resolve_train_validation_split(
            general_params=
                general_params
        )
    )

    split_summary_path = (
        output_folder
        / "split_summary.json"
    )

    split_summary = (
        load_json_if_exists(
            split_summary_path
        )
    )

    # ========================================================
    # 5. OPTIONAL-FEATURE FLAGS
    # ========================================================

    use_optuna = bool(
        experiment_config.get(
            "use_optuna",
            False,
        )
    )

    use_augmentation = bool(
        experiment_config.get(
            "use_augmentation",
            False,
        )
    )

    use_threshold_search = bool(
        experiment_config.get(
            "use_threshold_search",
            True,
        )
    )

    use_postprocessing = bool(
        experiment_config.get(
            "use_postprocessing",
            False,
        )
    )

    use_normalization = bool(
        experiment_config.get(
            "use_normalization",
            False,
        )
    )

    use_resampling = bool(
        experiment_config.get(
            "use_resampling",
            False,
        )
    )

    # ========================================================
    # 6. THRESHOLD SETTINGS
    # ========================================================

    threshold_summary = None

    if use_threshold_search:

        threshold_summary = (
            load_json_file(
                threshold_summary_path
            )
        )

        validate_validation_only_summary(
            threshold_summary,
            "Threshold summary",
        )

        best_threshold = (
            threshold_summary.get(
                "best_threshold",
                None,
            )
        )

        if best_threshold is None:

            raise KeyError(
                "\nbest_threshold is missing from:\n"
                f"{threshold_summary_path}"
            )

        best_threshold = float(
            best_threshold
        )

        if not (
            0.0
            <= best_threshold
            <= 1.0
        ):

            raise ValueError(
                "Frozen threshold must be between 0 and 1."
            )

        threshold_selection_metric = (
            threshold_summary.get(
                "selection_metric",
                "validation_iou",
            )
        )

    else:

        best_threshold = (
            get_default_semantic_threshold(
                experiment_config
            )
        )

        threshold_selection_metric = (
            "fixed_default_threshold"
        )

    # ========================================================
    # 7. POST-PROCESSING SETTINGS
    # ========================================================

    postprocess_summary = None

    if use_postprocessing:

        postprocess_summary = (
            load_json_file(
                postprocess_summary_path
            )
        )

        validate_validation_only_summary(
            postprocess_summary,
            "Post-processing summary",
        )

        postprocess_threshold = (
            postprocess_summary.get(
                "best_threshold",
                None,
            )
        )

        if postprocess_threshold is not None:

            if abs(
                float(
                    postprocess_threshold
                )
                - float(
                    best_threshold
                )
            ) > 1e-8:

                raise RuntimeError(
                    "\nThreshold mismatch between validation threshold "
                    "summary and validation post-processing summary.\n"
                    f"Threshold summary: {best_threshold}\n"
                    f"Postprocess summary: {postprocess_threshold}"
                )

        best_min_area_m2 = float(
            postprocess_summary.get(
                "best_min_area_m2",
                postprocess_summary.get(
                    "min_area_m2",
                    0.0,
                ),
            )
        )

        if best_min_area_m2 < 0.0:

            raise ValueError(
                "Frozen minimum polygon area must be >= 0."
            )

        postprocess_method = (
            postprocess_summary.get(
                "postprocessing_method",
                postprocess_summary.get(
                    "method",
                    "minimum_connected_component_area",
                ),
            )
        )

        postprocess_selection_metric = (
            postprocess_summary.get(
                "selection_metric",
                "validation_iou",
            )
        )

    else:

        best_min_area_m2 = 0.0
        postprocess_method = "none"
        postprocess_selection_metric = None

    # ========================================================
    # 8. OPTUNA METADATA
    # ========================================================

    optuna_metadata = {
        "used": False,
        "summary_path": None,
        "study_name": None,
        "best_trial": None,
        "best_validation_iou": None,
        "best_params": None,
    }

    if args.optuna_summary is not None:

        optuna_summary_path = Path(
            args.optuna_summary
        )

        optuna_summary = (
            load_json_file(
                optuna_summary_path
            )
        )

        optuna_metadata = (
            extract_optuna_metadata(
                optuna_summary
            )
        )

        optuna_metadata[
            "summary_path"
        ] = str(
            optuna_summary_path
        )

    elif use_optuna:

        raise RuntimeError(
            "\nexperiment.json has use_optuna=true, but no "
            "--optuna-summary was provided.\n\n"
            "Explicitly provide the selected Optuna summary so the "
            "exact frozen parameters are recorded."
        )

    # ========================================================
    # 9. BEST EPOCH / VALIDATION METRIC
    # ========================================================

    best_epoch = (
        resolve_best_epoch(
            training_summary
        )
    )

    best_training_validation_iou = (
        resolve_best_validation_iou(
            training_summary
        )
    )

    requested_epochs = (
        resolve_requested_epochs(
            training_summary
        )
    )

    # ========================================================
    # 10. SAM-LORA METADATA
    # ========================================================

    sam_lora_metadata = (
        build_sam_lora_metadata(
            model_type=
                model_type,

            model_config=
                model_config,

            training_summary=
                training_summary,
        )
    )

    # ========================================================
    # 11. TEST REFERENCES
    # ========================================================

    # IDs only. No test raster/GT is opened here.
    test_image_id = (
        experiment_config.get(
            "test_image_id"
        )
    )

    ground_truth_id = (
        experiment_config.get(
            "ground_truth_id"
        )
    )

    # ========================================================
    # 12. BUILD FROZEN SETTINGS
    # ========================================================

    frozen_at_utc = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    display_backbone = (
        experiment_config.get(
            "backbone",
            model_config.get(
                "default_backbone",
                None,
            ),
        )
    )

    final_settings = {
        "settings_frozen":
            True,

        "frozen_at_utc":
            frozen_at_utc,

        "architecture_version":
            "train_validation_freeze_final_test_v2",

        "experiment": {
            "experiment_id":
                experiment_config.get(
                    "experiment_id"
                ),

            "model_type":
                model_type,

            "backbone":
                display_backbone,

            "source_type":
                experiment_config.get(
                    "source_type"
                ),

            "tile_size":
                int(
                    experiment_config.get(
                        "tile_size"
                    )
                ),

            "dataset_type":
                experiment_config.get(
                    "dataset_type"
                ),
        },

        "split": {
            "train_percent":
                float(
                    split_info[
                        "train_percent"
                    ]
                ),

            "validation_percent":
                float(
                    split_info[
                        "validation_percent"
                    ]
                ),

            "seed":
                int(
                    split_info[
                        "seed"
                    ]
                ),

            "train_count":
                (
                    int(
                        split_summary.get(
                            "train_count"
                        )
                    )
                    if isinstance(
                        split_summary,
                        dict,
                    )
                    and split_summary.get(
                        "train_count"
                    )
                    is not None
                    else training_summary.get(
                        "train_samples"
                    )
                ),

            "validation_count":
                (
                    int(
                        split_summary.get(
                            "validation_count"
                        )
                    )
                    if isinstance(
                        split_summary,
                        dict,
                    )
                    and split_summary.get(
                        "validation_count"
                    )
                    is not None
                    else training_summary.get(
                        "validation_samples"
                    )
                ),

            "test_included_in_split":
                False,
        },

        "model_selection": {
            "checkpoint_path":
                str(
                    checkpoint_path
                ),

            "checkpoint_storage":
                (
                    "lora_and_decoder"
                    if model_type
                    == "sam_lora"
                    else "full_model_state_dict"
                ),

            "best_epoch":
                best_epoch,

            "requested_epochs":
                (
                    int(
                        requested_epochs
                    )
                    if requested_epochs
                    is not None
                    else None
                ),

            "best_training_validation_iou":
                best_training_validation_iou,

            "training_summary_path":
                str(
                    training_summary_path
                ),
        },

        "optional_features": {
            "augmentation":
                use_augmentation,

            "normalization":
                use_normalization,

            "resampling":
                use_resampling,

            "optuna":
                use_optuna,

            "threshold_search":
                use_threshold_search,

            "postprocessing":
                use_postprocessing,
        },

        "sam_lora":
            sam_lora_metadata,

        "optuna": {
            **optuna_metadata,
        },

        "threshold": {
            "value":
                float(
                    best_threshold
                ),

            "selected_on":
                (
                    "validation"
                    if use_threshold_search
                    else "fixed_default"
                ),

            "selection_metric":
                threshold_selection_metric,

            "summary_path":
                (
                    str(
                        threshold_summary_path
                    )
                    if use_threshold_search
                    else None
                ),
        },

        "postprocessing": {
            "enabled":
                use_postprocessing,

            "method":
                postprocess_method,

            "min_area_m2":
                float(
                    best_min_area_m2
                ),

            "selected_on":
                (
                    "validation"
                    if use_postprocessing
                    else None
                ),

            "selection_metric":
                postprocess_selection_metric,

            "summary_path":
                (
                    str(
                        postprocess_summary_path
                    )
                    if use_postprocessing
                    else None
                ),
        },

        "final_test": {
            "allowed":
                True,

            "already_run":
                False,

            "source_type":
                experiment_config.get(
                    "source_type"
                ),

            "test_image_id":
                test_image_id,

            "ground_truth_id":
                ground_truth_id,

            "same_shared_test_zone":
                True,

            "policy":
                (
                    "Run the source-specific external test only after "
                    "all settings are frozen. Do not search thresholds, "
                    "hyperparameters, or post-processing parameters on test."
                ),

            "final_metric":
                "strict_polygon_iou",
        },

        "provenance": {
            "experiment_config_path":
                str(
                    Path(
                        args.config
                    ).resolve()
                ),

            "general_params_path":
                str(
                    (
                        ROOT
                        / "config"
                        / "general_params.json"
                    ).resolve()
                ),

            "model_config_path":
                str(
                    (
                        ROOT
                        / "config"
                        / "models"
                        / f"{model_type}.json"
                    ).resolve()
                ),

            "split_summary_path":
                (
                    str(
                        split_summary_path.resolve()
                    )
                    if split_summary_path.exists()
                    else None
                ),
        },
    }

    # ========================================================
    # 13. SAVE
    # ========================================================

    final_settings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        final_settings,
        final_settings_path,
    )

    # ========================================================
    # 14. DISPLAY
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "FINAL SETTINGS FROZEN"
    )

    print(
        "=" * 78
    )

    print(
        "Experiment ID:",
        final_settings[
            "experiment"
        ][
            "experiment_id"
        ],
    )

    print(
        "Model:",
        model_type,
    )

    print(
        "Backbone:",
        display_backbone,
    )

    print(
        "Source:",
        final_settings[
            "experiment"
        ][
            "source_type"
        ],
    )

    print(
        "Train / Validation:",
        (
            f"{final_settings['split']['train_percent']:.0f}% / "
            f"{final_settings['split']['validation_percent']:.0f}%"
        ),
    )

    print(
        "Split Seed:",
        final_settings[
            "split"
        ][
            "seed"
        ],
    )

    print(
        "Best Epoch:",
        best_epoch,
    )

    if best_training_validation_iou is not None:

        print(
            "Best Training Validation IoU:",
            f"{best_training_validation_iou * 100.0:.2f}%",
        )

    if model_type == "sam_lora":

        resolved_lr = (
            sam_lora_metadata.get(
                "learning_rate",
                {},
            ).get(
                "resolved_learning_rate"
            )
        )

        if resolved_lr is not None:

            print(
                "Frozen SAM-LoRA Auto-LR:",
                f"{float(resolved_lr):.6e}",
            )

        print(
            "LoRA Rank:",
            sam_lora_metadata.get(
                "lora",
                {},
            ).get(
                "rank"
            ),
        )

        print(
            "LoRA Alpha:",
            sam_lora_metadata.get(
                "lora",
                {},
            ).get(
                "alpha"
            ),
        )

    print(
        "Frozen Threshold:",
        f"{best_threshold:.4f}",
    )

    print(
        "Frozen Minimum Area:",
        f"{best_min_area_m2:.2f} m²",
    )

    print(
        "Optuna:",
        (
            "ON"
            if use_optuna
            else "OFF"
        ),
    )

    print(
        "Augmentation:",
        (
            "ON"
            if use_augmentation
            else "OFF"
        ),
    )

    print(
        "-" * 78
    )

    print(
        "External Test Accessed:",
        "NO",
    )

    print(
        "Same Shared Final Test Zone:",
        "YES",
    )

    print(
        "Settings Frozen:",
        "YES",
    )

    print(
        "Next Stage:",
        "FINAL SOURCE-SPECIFIC TEST",
    )

    print(
        "Final Settings:",
        final_settings_path,
    )

    print(
        "=" * 78
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

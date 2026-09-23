"""
run_inference.py

Run full ArcGIS raster inference using a trained Building Extraction model.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- Existing standard semantic-segmentation experiments
- SAM-LoRA
- Standard and augmented training experiments
- ArcGIS File Geodatabase raster datasets
- Robust build_model() return handling
- Optional threshold override
- Test-time augmentation OFF

Important
---------
The SAME source-specific independent geographic testing zone is used
for all compatible models.

Standard semantic models:
    build_model()
        +
    best_model.pth
        ->
    trained model

SAM-LoRA:
    official SAM checkpoint
        +
    sam_lora.json
        +
    best_model.pth
        ->
    reconstructed trained SAM-LoRA

After model reconstruction, all compatible models use the same shared:

    threshold policy
    tiled ArcGIS inference
    overlap policy
    raster output
    polygon conversion
    post-processing/evaluation stages

PowerShell:
    & $py "Buildings\\scripts\\run_inference.py"

Example with fixed threshold:
    & $py "Buildings\\scripts\\run_inference.py" --threshold 0.5
"""

from pathlib import Path
import sys
import argparse

import torch
import arcpy


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]


if str(ROOT) not in sys.path:

    sys.path.append(
        str(ROOT)
    )


# ============================================================
# SHARED MODULE IMPORTS
# ============================================================

from src.utils import (
    load_experiment_config,
    load_general_params,
    load_paths_config,
    load_model_config,
    get_backbone_info,
    get_test_image_path,
    create_experiment_output_folder,
    save_json,
)

from src.models.factory import (
    build_model
)

from src.inference import (
    get_best_threshold,
    load_model_checkpoint,
    run_full_arcgis_raster_inference,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run full-scene ArcGIS inference "
            "for a Building Extraction experiment."
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
        help=(
            "Path to experiment configuration JSON."
        ),
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional inference threshold override. "
            "If omitted, use saved best threshold "
            "or default 0.5."
        ),
    )


    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device override, for example "
            "'cuda', 'cuda:0', or 'cpu'. "
            "If omitted, CUDA is used when available."
        ),
    )


    return parser.parse_args()


# ============================================================
# MODEL TYPE NORMALIZATION
# ============================================================

def normalize_model_type(
    model_type,
):
    """
    Normalize historical model aliases.
    """

    normalized = (
        str(
            model_type
        )
        .strip()
        .lower()
    )

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
        "samlora": "sam_lora",
        "sam lora": "sam_lora",
        "sam_lora": "sam_lora",
    }

    return aliases.get(
        normalized,
        normalized,
    )


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    requested_device=None,
):
    """
    Resolve inference device.
    """

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
            "CUDA was requested for inference "
            "but CUDA is not available."
        )

    return device


# ============================================================
# ARCGIS PATH CHECK
# ============================================================

def arcgis_path_exists(
    path
):
    """
    Check whether a path exists.

    Supports:
    - normal filesystem paths
    - rasters inside .gdb
    - feature classes inside .gdb
    - other ArcGIS datasets

    pathlib.Path.exists() cannot reliably check datasets
    stored inside an Esri File Geodatabase, therefore
    arcpy.Exists() is checked first.
    """

    path_string = str(
        path
    )


    try:

        if arcpy.Exists(
            path_string
        ):

            return True

    except Exception:

        pass


    try:

        if Path(
            path_string
        ).exists():

            return True

    except Exception:

        pass


    return False


# ============================================================
# ROBUST STANDARD MODEL FACTORY HANDLING
# ============================================================

def resolve_built_model(
    build_result,
):
    """
    Handle different build_model() return styles.

    Supported:
        model
        model, something
        model, device
        model, device, params

    Device and trainable parameter count are obtained directly
    from the final PyTorch model when necessary.
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
                "build_model() returned "
                "an empty tuple/list."
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
            "The first object returned by "
            "build_model() is not a "
            "torch.nn.Module.\n"
            f"Received: {type(model)}"
        )


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
# SAM-LORA RECONSTRUCTION
# ============================================================

def load_sam_lora_for_inference(
    experiment_config,
    best_model_path,
    device,
):
    """
    Reconstruct trained SAM-LoRA.

    Reconstruction:
        official SAM checkpoint
            +
        sam_lora.json
            +
        best_model.pth
            ->
        trained SAM-LoRA
    """

    from src.models.sam_lora.inference import (
        load_trained_sam_lora,
    )

    sam_config_path = (
        ROOT
        / "config"
        / "models"
        / "sam_lora.json"
    )


    if not sam_config_path.exists():

        raise FileNotFoundError(
            "\nSAM-LoRA configuration was not found:\n"
            f"{sam_config_path}"
        )


    model, load_info = (
        load_trained_sam_lora(
            config_path=sam_config_path,
            trained_checkpoint_path=best_model_path,
            device=device,
            verbose=True,
        )
    )


    experiment_backbone = (
        experiment_config.get(
            "backbone",
            None,
        )
    )


    if experiment_backbone is not None:

        if (
            str(
                experiment_backbone
            )
            .strip()
            .lower()
            !=
            str(
                load_info.backbone
            )
            .strip()
            .lower()
        ):

            raise ValueError(
                "\nSAM-LoRA backbone mismatch.\n"
                f"experiment.json : {experiment_backbone}\n"
                f"loaded model    : {load_info.backbone}"
            )


    trainable_params = sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    return (
        model,
        load_info,
        trainable_params,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    # ========================================================
    # 1. LOAD CONFIGURATION
    # ========================================================

    config_path = Path(
        args.config
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


    paths_config = (
        load_paths_config(
            ROOT
        )
    )


    model_type = (
        normalize_model_type(
            experiment_config[
                "model_type"
            ]
        )
    )


    is_sam_lora = (
        model_type
        == "sam_lora"
    )


    # ========================================================
    # 2. EXPERIMENT OUTPUT FOLDER
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


    # ========================================================
    # 3. TRAINED CHECKPOINT
    # ========================================================

    best_model_path = (
        output_folder
        / "best_model.pth"
    )


    if not best_model_path.exists():

        raise FileNotFoundError(
            "\nTrained model was not found:\n"
            f"{best_model_path}\n\n"
            "Train the experiment before "
            "running inference."
        )


    # ========================================================
    # 4. TEST IMAGE
    # ========================================================

    test_image_path = (
        get_test_image_path(
            experiment_config=
                experiment_config,

            paths_config=
                paths_config,
        )
    )


    if not arcgis_path_exists(
        test_image_path
    ):

        raise FileNotFoundError(
            "\nTest image / ArcGIS raster "
            "dataset was not found:\n"
            f"{test_image_path}\n\n"
            "The path was checked using "
            "both arcpy.Exists() and "
            "Path.exists()."
        )


    # ========================================================
    # 5. INFERENCE THRESHOLD
    # ========================================================

    if args.threshold is not None:

        inference_threshold = float(
            args.threshold
        )


        if not (
            0.0
            <= inference_threshold
            <= 1.0
        ):

            raise ValueError(
                "--threshold must be between "
                "0.0 and 1.0."
            )


        threshold_source = (
            "command_line_override"
        )


    else:

        inference_threshold = float(
            get_best_threshold(
                output_folder=
                    output_folder,

                default_threshold=
                    0.5,
            )
        )


        threshold_source = (
            "saved_threshold_or_default"
        )


    # ========================================================
    # 6. DEVICE
    # ========================================================

    requested_device = (
        resolve_device(
            args.device
        )
    )


    # ========================================================
    # 7. BUILD / RECONSTRUCT MODEL
    # ========================================================

    checkpoint_epoch = None
    checkpoint_val_iou = None
    official_sam_checkpoint = None
    checkpoint_format = None


    if is_sam_lora:

        (
            model,
            sam_load_info,
            params,
        ) = (
            load_sam_lora_for_inference(
                experiment_config=
                    experiment_config,

                best_model_path=
                    best_model_path,

                device=
                    requested_device,
            )
        )


        device = next(
            model.parameters()
        ).device


        checkpoint_epoch = (
            sam_load_info.trained_epoch
        )


        official_sam_checkpoint = (
            sam_load_info.pretrained_sam_checkpoint_path
        )


        checkpoint_format = (
            sam_load_info.checkpoint_format
        )


        validation_metrics = (
            sam_load_info.validation_metrics
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


        model_loader = (
            "sam_lora_reconstruction"
        )


    else:

        # ====================================================
        # ORIGINAL STANDARD MODEL PATH
        # ====================================================

        model_config = (
            load_model_config(
                model_type,
                ROOT
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
        ) = (
            resolve_built_model(
                build_result
            )
        )


        # ----------------------------------------------------
        # Honor explicit --device for standard models too.
        # ----------------------------------------------------

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
                    best_model_path,

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


    model.eval()


    # ========================================================
    # 8. INFERENCE SETTINGS
    # ========================================================

    tile_size = int(
        experiment_config[
            "tile_size"
        ]
    )


    overlap_ratio = float(
        general_params[
            "inference"
        ][
            "overlap_ratio"
        ]
    )


    training_augmentation = bool(
        experiment_config.get(
            "use_augmentation",
            False,
        )
    )


    display_backbone = (
        experiment_config.get(
            "backbone",
            (
                sam_load_info.backbone
                if is_sam_lora
                else "unknown"
            ),
        )
    )


    # ========================================================
    # 9. DISPLAY CONFIGURATION
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FULL ARCGIS INFERENCE START"
    )

    print(
        "=" * 72
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
        "Tile Size:",
        tile_size,
    )


    print(
        "Device:",
        device,
    )


    print(
        "Trainable Parameters:",
        f"{params:,}",
    )


    print(
        "Model Loader:",
        model_loader,
    )


    if is_sam_lora:

        print(
            "Official SAM Checkpoint:",
            official_sam_checkpoint,
        )


        print(
            "SAM-LoRA Checkpoint Format:",
            checkpoint_format,
        )


        print(
            "SAM Input:",
            (
                "shared RGB tile -> "
                "official SAM resize/normalization internally"
            ),
        )


    print(
        "-" * 72
    )


    print(
        "Training Augmentation:",
        (
            "ON"
            if training_augmentation
            else "OFF"
        ),
    )


    print(
        "Inference Augmentation:",
        "OFF",
    )


    print(
        "Best Model:",
        best_model_path,
    )


    if checkpoint_epoch is not None:

        print(
            "Checkpoint Best Epoch:",
            checkpoint_epoch,
        )


    if checkpoint_val_iou is not None:

        print(
            "Checkpoint Validation IoU:",
            f"{float(checkpoint_val_iou):.4f}",
        )


    print(
        "Inference Threshold:",
        inference_threshold,
    )


    print(
        "Threshold Source:",
        threshold_source,
    )


    print(
        "Test Image:",
        test_image_path,
    )


    print(
        "ArcGIS Dataset Exists:",
        arcpy.Exists(
            str(
                test_image_path
            )
        ),
    )


    print(
        "Overlap Ratio:",
        overlap_ratio,
    )


    print(
        "Output Folder:",
        output_folder,
    )


    print(
        "Independent Test Zone:",
        "SAME SHARED SOURCE-SPECIFIC TEST AREA",
    )


    print(
        "=" * 72
    )

    print()


    # ========================================================
    # 10. FULL SHARED ARCGIS RASTER INFERENCE
    # ========================================================
    #
    # IMPORTANT:
    #
    # From this point onward SAM-LoRA should use the SAME shared
    # inference pipeline as the other semantic models.
    #
    # The shared src/inference.py will be updated separately to
    # normalize model outputs such as:
    #
    #     Tensor
    #     {"out": Tensor}
    #     {"logits": Tensor}
    #
    # ========================================================

    summary = (
        run_full_arcgis_raster_inference(
            model=
                model,

            device=
                device,

            test_image_path=
                test_image_path,

            output_folder=
                output_folder,

            threshold=
                inference_threshold,

            tile_size=
                tile_size,

            overlap_ratio=
                overlap_ratio,
        )
    )


    # ========================================================
    # 11. ADD EXPERIMENT METADATA
    # ========================================================

    summary[
        "experiment_id"
    ] = experiment_config[
        "experiment_id"
    ]


    summary[
        "model_type"
    ] = model_type


    summary[
        "backbone"
    ] = display_backbone


    summary[
        "source_type"
    ] = experiment_config[
        "source_type"
    ]


    summary[
        "dataset_type"
    ] = experiment_config[
        "dataset_type"
    ]


    summary[
        "tile_size"
    ] = int(
        tile_size
    )


    summary[
        "overlap_ratio"
    ] = float(
        overlap_ratio
    )


    summary[
        "inference_threshold"
    ] = float(
        inference_threshold
    )


    summary[
        "threshold_source"
    ] = (
        threshold_source
    )


    summary[
        "training_augmentation"
    ] = bool(
        training_augmentation
    )


    summary[
        "inference_augmentation"
    ] = False


    summary[
        "checkpoint_path"
    ] = str(
        best_model_path
    )


    summary[
        "model_loader"
    ] = (
        model_loader
    )


    summary[
        "test_image_path"
    ] = str(
        test_image_path
    )


    summary[
        "same_shared_test_zone"
    ] = True


    if checkpoint_epoch is not None:

        summary[
            "checkpoint_epoch"
        ] = int(
            checkpoint_epoch
        )


    if checkpoint_val_iou is not None:

        summary[
            "checkpoint_best_val_iou"
        ] = float(
            checkpoint_val_iou
        )


    if is_sam_lora:

        summary[
            "official_sam_checkpoint"
        ] = str(
            official_sam_checkpoint
        )


        summary[
            "sam_lora_checkpoint_format"
        ] = (
            checkpoint_format
        )


        summary[
            "sam_lora_reconstruction"
        ] = True


    # ========================================================
    # 12. SAVE SUMMARY
    # ========================================================

    inference_summary_path = (
        output_folder
        / "inference_summary.json"
    )


    save_json(
        summary,
        inference_summary_path,
    )


    # ========================================================
    # 13. FINAL DISPLAY
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FULL ARCGIS INFERENCE FINISHED"
    )

    print(
        "=" * 72
    )


    print(
        "Final Prediction FC:",
        summary[
            "final_prediction_fc"
        ],
    )


    print(
        "Polygon Count:",
        summary[
            "polygon_count"
        ],
    )


    print(
        "Building Pixels:",
        summary[
            "building_pixels"
        ],
    )


    print(
        "Building Ratio:",
        f"{summary['building_ratio']:.6f}",
    )


    print(
        "Binary Raster:",
        summary[
            "binary_raster_path"
        ],
    )


    print(
        "Inference Threshold:",
        inference_threshold,
    )


    print(
        "Inference Augmentation:",
        "OFF",
    )


    print(
        "Same Shared Test Zone:",
        "YES",
    )


    print(
        "Summary JSON:",
        inference_summary_path,
    )


    print(
        "Elapsed Time:",
        f"{summary['elapsed_time_seconds']} seconds",
    )


    print(
        "=" * 72
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

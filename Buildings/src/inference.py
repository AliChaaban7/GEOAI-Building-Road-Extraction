"""
inference.py

Shared inference utilities for semantic building extraction.

Supports:
- U-Net
- DeepLabV3 / DeepLabV3+
- SAM-LoRA
- loading saved best threshold
- loading standard trained-model checkpoints
- predicting probability masks
- applying thresholds
- saving validation prediction previews
- tiled full-scene ArcGIS raster inference
- georeferenced binary raster export
- raster-to-polygon conversion

Shared output contract
----------------------
Semantic models may return:

    Tensor
        -> interpreted directly as logits

    {"out": Tensor}
        -> torchvision / DeepLab-style logits

    {"logits": Tensor}
        -> SAM-LoRA-style logits

All compatible semantic models are normalized to the same logits
interface before sigmoid / thresholding.

Important
---------
SAM-LoRA checkpoint reconstruction is model-specific and is handled by:

    src.models.sam_lora.inference.load_trained_sam_lora

This shared module does NOT duplicate SAM checkpoint logic.

After a trained model has been reconstructed, SAM-LoRA uses the same
shared tiled ArcGIS inference path as U-Net and DeepLab.
"""

from pathlib import Path
from collections.abc import Mapping

import csv
import json

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F


# ============================================================
# THRESHOLD
# ============================================================

def load_threshold_summary(
    output_folder,
):
    """
    Load threshold_summary.json if it exists.
    """

    output_folder = Path(
        output_folder
    )

    threshold_summary_path = (
        output_folder
        / "threshold_summary.json"
    )

    if not threshold_summary_path.exists():

        return None

    with open(
        threshold_summary_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def get_best_threshold(
    output_folder,
    default_threshold=0.5,
):
    """
    Get the selected validation threshold.

    Priority:
        threshold_summary.json -> best_threshold

    Fallback:
        default_threshold
    """

    threshold_summary = (
        load_threshold_summary(
            output_folder
        )
    )

    if threshold_summary is None:

        return float(
            default_threshold
        )

    threshold = float(
        threshold_summary.get(
            "best_threshold",
            default_threshold,
        )
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Saved best_threshold must be between 0 and 1. "
            f"Received: {threshold}"
        )

    return threshold


# ============================================================
# STANDARD CHECKPOINT LOADING
# ============================================================

def load_model_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load a STANDARD model checkpoint.

    Expected standard format:
        {
            "model_state_dict": ...
        }

    Used by:
        U-Net
        DeepLabV3 / DeepLabV3+

    SAM-LoRA intentionally does NOT use this function because its
    best_model.pth stores only trainable LoRA + mask-decoder weights.
    SAM-LoRA must be reconstructed with load_trained_sam_lora().
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        Mapping,
    ):

        raise TypeError(
            "Standard checkpoint must be a dictionary-like object."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        if (
            "trainable_state_dict"
            in checkpoint
        ):

            raise ValueError(
                "This checkpoint appears to be a SAM-LoRA checkpoint "
                "containing 'trainable_state_dict'. "
                "Use load_trained_sam_lora() instead of "
                "load_model_checkpoint()."
            )

        raise KeyError(
            "Standard checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# MANIFEST HELPERS
# ============================================================

def read_manifest_rows(
    manifest_csv,
):
    """
    Read validation manifest CSV.

    Compatible label-column names:
        label_path
        mask_path
        mask
        label
        label_file
        mask_filepath
    """

    manifest_csv = Path(
        manifest_csv
    )

    if not manifest_csv.exists():

        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}"
        )

    rows = []

    image_columns = (
        "image_path",
        "image",
        "image_file",
        "image_filepath",
        "chip_path",
        "raster_path",
    )

    mask_columns = (
        "label_path",
        "mask_path",
        "mask",
        "label",
        "label_file",
        "mask_filepath",
    )

    with open(
        manifest_csv,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:

            return rows

        image_col = next(
            (
                column
                for column
                in image_columns
                if column
                in reader.fieldnames
            ),
            None,
        )

        mask_col = next(
            (
                column
                for column
                in mask_columns
                if column
                in reader.fieldnames
            ),
            None,
        )

        if image_col is None:

            raise ValueError(
                "Could not find an image column in validation manifest. "
                f"Columns: {reader.fieldnames}"
            )

        if mask_col is None:

            raise ValueError(
                "Could not find a mask/label column in validation manifest. "
                f"Columns: {reader.fieldnames}"
            )

        for row in reader:

            image_path = str(
                row.get(
                    image_col,
                    "",
                )
            ).strip()

            mask_path = str(
                row.get(
                    mask_col,
                    "",
                )
            ).strip()

            if (
                not image_path
                or not mask_path
            ):

                continue

            normalized_row = dict(
                row
            )

            # Preserve legacy keys used by preview code.
            normalized_row[
                "image_path"
            ] = image_path

            normalized_row[
                "label_path"
            ] = mask_path

            rows.append(
                normalized_row
            )

    return rows


# ============================================================
# IMAGE / MASK LOADING FOR PREVIEWS
# ============================================================

def load_image_as_tensor(
    image_path,
    device,
):
    """
    Load image as RGB torch tensor normalized to [0, 1].

    Shape:
        [1, 3, H, W]

    This representation is shared by U-Net, DeepLab, and SAM-LoRA.
    SAM-LoRA applies official SAM resizing/normalization internally.
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image_array = (
        np.array(
            image
        )
        .astype(
            np.float32
        )
        / 255.0
    )

    image_tensor = (
        torch.from_numpy(
            image_array
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(
            0
        )
        .contiguous()
    )

    image_tensor = image_tensor.to(
        device
    )

    return image_tensor


def load_mask_as_array(
    mask_path,
):
    """
    Load ground-truth mask as binary array.
    """

    mask = Image.open(
        mask_path
    ).convert(
        "L"
    )

    mask_array = np.array(
        mask
    )

    binary_mask = (
        mask_array
        > 0
    ).astype(
        np.uint8
    )

    return binary_mask


# ============================================================
# SHARED MODEL OUTPUT NORMALIZATION
# ============================================================

def get_model_output(
    model_output,
):
    """
    Extract semantic-segmentation logits from model output.

    Supported output styles
    -----------------------
    Tensor:
        U-Net and other custom semantic models.

    Mapping with "out":
        torchvision DeepLab-style output.

    Mapping with "logits":
        SAM-LoRA-style output.

    The function returns RAW LOGITS.
    Sigmoid is applied later by the shared inference pipeline.
    """

    if torch.is_tensor(
        model_output
    ):

        return model_output


    if isinstance(
        model_output,
        Mapping,
    ):

        if (
            "out"
            in model_output
            and torch.is_tensor(
                model_output[
                    "out"
                ]
            )
        ):

            return model_output[
                "out"
            ]


        if (
            "logits"
            in model_output
            and torch.is_tensor(
                model_output[
                    "logits"
                ]
            )
        ):

            return model_output[
                "logits"
            ]


        available_keys = list(
            model_output.keys()
        )

        raise KeyError(
            "Semantic model output mapping does not contain a supported "
            "logit key ('out' or 'logits'). "
            f"Available keys: {available_keys}"
        )


    raise TypeError(
        "Unsupported semantic model output type. "
        "Expected Tensor or mapping containing 'out'/'logits'. "
        f"Received: {type(model_output)}"
    )


def normalize_logits_shape(
    logits,
    target_height=None,
    target_width=None,
):
    """
    Validate semantic logits and optionally restore spatial size.

    Expected semantic output:
        [B, 1, H, W]

    A spatial resize is only performed when necessary. This leaves the
    existing U-Net/DeepLab path unchanged while protecting shared
    inference from models whose decoder returns a different resolution.
    """

    if not torch.is_tensor(
        logits
    ):

        raise TypeError(
            "Logits must be a torch.Tensor."
        )


    if logits.ndim == 3:

        logits = logits.unsqueeze(
            1
        )


    if logits.ndim != 4:

        raise ValueError(
            "Semantic logits must have shape [B,C,H,W]. "
            f"Received: {tuple(logits.shape)}"
        )


    if int(
        logits.shape[1]
    ) != 1:

        raise ValueError(
            "Shared binary building inference expects one output channel. "
            f"Received C={logits.shape[1]}."
        )


    if not torch.isfinite(
        logits
    ).all():

        raise FloatingPointError(
            "Model produced NaN or Inf logits during inference."
        )


    if (
        target_height is not None
        and target_width is not None
    ):

        target_size = (
            int(
                target_height
            ),
            int(
                target_width
            ),
        )

        if (
            tuple(
                logits.shape[-2:]
            )
            != target_size
        ):

            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )


    return logits


# ============================================================
# PROBABILITY / THRESHOLD
# ============================================================

def predict_probability_mask(
    model,
    image_tensor,
):
    """
    Predict one probability mask from an image tensor.

    Input:
        [B,3,H,W]

    Output:
        H x W float32 numpy array for B=1.
    """

    model.eval()

    with torch.no_grad():

        output = model(
            image_tensor
        )

        logits = get_model_output(
            output
        )

        logits = normalize_logits_shape(
            logits,
            target_height=int(
                image_tensor.shape[-2]
            ),
            target_width=int(
                image_tensor.shape[-1]
            ),
        )

        probs = torch.sigmoid(
            logits
        )


    if int(
        probs.shape[0]
    ) != 1:

        raise ValueError(
            "predict_probability_mask expects batch size 1. "
            f"Received B={probs.shape[0]}."
        )


    prob_mask = (
        probs[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    return prob_mask


def apply_threshold(
    prob_mask,
    threshold,
):
    """
    Convert probability mask to binary prediction.
    """

    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "threshold must be between 0 and 1."
        )

    pred_mask = (
        prob_mask
        >= threshold
    ).astype(
        np.uint8
    )

    return pred_mask


# ============================================================
# PREVIEW VISUALIZATION
# ============================================================

def mask_to_rgb(
    mask_array,
):
    """
    Convert binary mask to RGB visualization.
    """

    mask_vis = (
        mask_array
        * 255
    ).astype(
        np.uint8
    )

    mask_img = Image.fromarray(
        mask_vis
    ).convert(
        "RGB"
    )

    return mask_img


def probability_to_rgb(
    prob_mask,
):
    """
    Convert probability mask to grayscale RGB visualization.
    """

    prob_vis = np.clip(
        prob_mask
        * 255,
        0,
        255,
    ).astype(
        np.uint8
    )

    prob_img = Image.fromarray(
        prob_vis
    ).convert(
        "RGB"
    )

    return prob_img


def add_title(
    image,
    title,
):
    """
    Add a small title above an image.
    """

    image = image.convert(
        "RGB"
    )

    title_height = 24

    new_img = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + title_height,
        ),
        "white",
    )

    new_img.paste(
        image,
        (
            0,
            title_height,
        ),
    )

    draw = ImageDraw.Draw(
        new_img
    )

    draw.text(
        (
            5,
            5,
        ),
        title,
        fill="black",
    )

    return new_img


def save_prediction_preview(
    image_path,
    mask_path,
    prob_mask,
    pred_mask,
    output_path,
    threshold,
):
    """
    Save side-by-side preview:

        original image
        ground truth
        probability
        prediction
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    gt_mask = load_mask_as_array(
        mask_path
    )

    gt_img = mask_to_rgb(
        gt_mask
    )

    prob_img = probability_to_rgb(
        prob_mask
    )

    pred_img = mask_to_rgb(
        pred_mask
    )

    image = image.resize(
        gt_img.size
    )

    prob_img = prob_img.resize(
        gt_img.size
    )

    pred_img = pred_img.resize(
        gt_img.size
    )

    panel_1 = add_title(
        image,
        "Image",
    )

    panel_2 = add_title(
        gt_img,
        "Ground Truth",
    )

    panel_3 = add_title(
        prob_img,
        "Probability",
    )

    panel_4 = add_title(
        pred_img,
        f"Prediction T={threshold}",
    )

    panels = [
        panel_1,
        panel_2,
        panel_3,
        panel_4,
    ]

    total_width = sum(
        panel.width
        for panel
        in panels
    )

    max_height = max(
        panel.height
        for panel
        in panels
    )

    preview = Image.new(
        "RGB",
        (
            total_width,
            max_height,
        ),
        "white",
    )

    x = 0

    for panel in panels:

        preview.paste(
            panel,
            (
                x,
                0,
            ),
        )

        x += panel.width

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview.save(
        output_path
    )

    return output_path


def save_validation_prediction_previews(
    model,
    val_manifest_csv,
    output_folder,
    device,
    threshold,
    max_samples=12,
):
    """
    Save prediction previews for validation samples.

    Test-time augmentation is not used.
    """

    rows = read_manifest_rows(
        val_manifest_csv
    )

    preview_folder = (
        Path(
            output_folder
        )
        / "figures"
        / "validation_prediction_previews"
    )

    preview_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files = []

    selected_rows = rows[
        : int(
            max_samples
        )
    ]

    for idx, row in enumerate(
        selected_rows,
        start=1,
    ):

        image_path = row[
            "image_path"
        ]

        mask_path = row[
            "label_path"
        ]

        image_tensor = load_image_as_tensor(
            image_path,
            device,
        )

        prob_mask = predict_probability_mask(
            model,
            image_tensor,
        )

        pred_mask = apply_threshold(
            prob_mask,
            threshold,
        )

        output_path = (
            preview_folder
            / f"preview_{idx:03d}.png"
        )

        save_prediction_preview(
            image_path=image_path,
            mask_path=mask_path,
            prob_mask=prob_mask,
            pred_mask=pred_mask,
            output_path=output_path,
            threshold=threshold,
        )

        saved_files.append(
            str(
                output_path
            )
        )

    return {
        "preview_folder":
            str(
                preview_folder
            ),

        "saved_count":
            len(
                saved_files
            ),

        "saved_files":
            saved_files,
    }


# ============================================================
# FULL ARCGIS RASTER INFERENCE UTILITIES
# ============================================================

def raster_to_rgb_numpy(
    raster_path,
):
    """
    Read an ArcGIS raster as RGB numpy array.

    Returns
    -------
    image_array:
        H x W x 3 float32 array in range [0,1]

    raster_info:
        dictionary with raster metadata

    Notes
    -----
    For multispectral imagery the shared semantic workflow uses the
    first three bands only. SAM-LoRA therefore receives explicit RGB,
    not a silently mixed 4-band tensor.
    """

    import arcpy

    raster_path = str(
        raster_path
    )

    raster = arcpy.Raster(
        raster_path
    )

    desc = arcpy.Describe(
        raster_path
    )

    arr = arcpy.RasterToNumPyArray(
        raster,
        nodata_to_value=0,
    )

    # --------------------------------------------------------
    # ArcGIS multiband raster usually:
    #     bands x height x width
    # --------------------------------------------------------

    if arr.ndim == 3:

        if arr.shape[0] in [
            1,
            3,
            4,
        ]:

            arr = np.moveaxis(
                arr,
                0,
                -1,
            )


    # --------------------------------------------------------
    # Single-band case
    # --------------------------------------------------------

    if arr.ndim == 2:

        arr = np.stack(
            [
                arr,
                arr,
                arr,
            ],
            axis=-1,
        )


    if arr.ndim != 3:

        raise ValueError(
            "ArcGIS raster could not be converted to HxWxC array. "
            f"Received shape: {arr.shape}"
        )


    # --------------------------------------------------------
    # Explicit RGB selection.
    # --------------------------------------------------------

    if arr.shape[-1] > 3:

        arr = arr[
            :,
            :,
            :3,
        ]


    if arr.shape[-1] == 1:

        arr = np.repeat(
            arr,
            3,
            axis=-1,
        )


    if arr.shape[-1] != 3:

        raise ValueError(
            "Shared semantic inference requires exactly 3 RGB channels "
            f"after conversion. Received: {arr.shape[-1]}"
        )


    arr = arr.astype(
        np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


    # --------------------------------------------------------
    # Preserve the current standard module normalization:
    #
    #     /255.0 only
    #
    # This is the same external input convention used during
    # Buildings semantic training.
    #
    # SAM-LoRA receives [0,1] here and applies official SAM
    # preprocessing internally.
    # --------------------------------------------------------

    max_value = (
        float(
            arr.max()
        )
        if arr.size > 0
        else 0.0
    )


    if max_value > 1.0:

        arr = (
            arr
            / 255.0
        )


    arr = np.clip(
        arr,
        0.0,
        1.0,
    )


    raster_info = {
        "width":
            int(
                raster.width
            ),

        "height":
            int(
                raster.height
            ),

        "band_count":
            int(
                raster.bandCount
            ),

        "channels_used":
            3,

        "channel_policy":
            "RGB_first_three_bands",

        "external_normalization":
            "divide_by_255_only",

        "cell_size_x":
            float(
                raster.meanCellWidth
            ),

        "cell_size_y":
            float(
                raster.meanCellHeight
            ),

        "extent_xmin":
            float(
                raster.extent.XMin
            ),

        "extent_ymin":
            float(
                raster.extent.YMin
            ),

        "extent_xmax":
            float(
                raster.extent.XMax
            ),

        "extent_ymax":
            float(
                raster.extent.YMax
            ),

        "spatial_reference":
            desc.spatialReference.name,
    }

    return (
        arr,
        raster_info,
    )


def make_tile_starts(
    length,
    tile_size,
    step,
):
    """
    Create tile start positions so the full raster is covered.
    """

    length = int(
        length
    )

    tile_size = int(
        tile_size
    )

    step = int(
        step
    )

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0."
        )

    if step <= 0:

        raise ValueError(
            "step must be > 0."
        )

    if length <= tile_size:

        return [
            0
        ]

    starts = list(
        range(
            0,
            length
            - tile_size
            + 1,
            step,
        )
    )

    last_start = (
        length
        - tile_size
    )

    if starts[
        -1
    ] != last_start:

        starts.append(
            last_start
        )

    return starts


def predict_large_raster_array(
    model,
    image_array,
    device,
    tile_size=256,
    overlap_ratio=0.25,
    print_every=25,
):
    """
    Predict a full probability mask using shared tiled inference.

    Compatible models:
        U-Net
        DeepLabV3 / DeepLabV3+
        SAM-LoRA

    Input
    -----
    image_array:
        H x W x 3
        float32
        range [0,1]

    Returns
    -------
    probability_mask:
        H x W float32 probability mask.

    Overlap
    -------
    Overlapping predictions are merged with arithmetic mean
    probability, preserving the existing Buildings inference behavior.
    """

    if image_array.ndim != 3:

        raise ValueError(
            "Expected image_array shape HxWx3. "
            f"Received: {image_array.shape}"
        )

    height, width, channels = (
        image_array.shape
    )

    if channels != 3:

        raise ValueError(
            f"Expected 3 channels, got {channels}"
        )


    tile_size = int(
        tile_size
    )

    overlap_ratio = float(
        overlap_ratio
    )


    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0."
        )


    if not (
        0.0
        <= overlap_ratio
        < 1.0
    ):

        raise ValueError(
            "overlap_ratio must satisfy 0 <= overlap_ratio < 1."
        )


    step = int(
        tile_size
        * (
            1.0
            - overlap_ratio
        )
    )

    step = max(
        step,
        1,
    )


    y_starts = make_tile_starts(
        height,
        tile_size,
        step,
    )

    x_starts = make_tile_starts(
        width,
        tile_size,
        step,
    )


    probability_sum = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    count_sum = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )


    total_tiles = (
        len(
            y_starts
        )
        * len(
            x_starts
        )
    )

    tile_counter = 0


    model.eval()


    with torch.no_grad():

        for y in y_starts:

            for x in x_starts:

                tile_counter += 1


                tile = image_array[
                    y:
                    y + tile_size,

                    x:
                    x + tile_size,

                    :,
                ]


                original_h, original_w = (
                    tile.shape[
                        :2
                    ]
                )


                # ------------------------------------------------
                # Pad only when the source raster is smaller than
                # the configured training tile size.
                # ------------------------------------------------

                if (
                    original_h
                    < tile_size
                    or original_w
                    < tile_size
                ):

                    padded_tile = np.zeros(
                        (
                            tile_size,
                            tile_size,
                            3,
                        ),
                        dtype=np.float32,
                    )

                    padded_tile[
                        :original_h,
                        :original_w,
                        :,
                    ] = tile

                    tile = padded_tile


                tile_tensor = (
                    torch.from_numpy(
                        tile
                    )
                    .permute(
                        2,
                        0,
                        1,
                    )
                    .unsqueeze(
                        0
                    )
                    .contiguous()
                )


                tile_tensor = tile_tensor.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )


                output = model(
                    tile_tensor
                )


                logits = get_model_output(
                    output
                )


                logits = normalize_logits_shape(
                    logits,
                    target_height=int(
                        tile.shape[
                            0
                        ]
                    ),
                    target_width=int(
                        tile.shape[
                            1
                        ]
                    ),
                )


                probs = torch.sigmoid(
                    logits
                )


                if int(
                    probs.shape[
                        0
                    ]
                ) != 1:

                    raise ValueError(
                        "Tiled inference expects model batch size 1. "
                        f"Received B={probs.shape[0]}."
                    )


                prob_tile = (
                    probs[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )


                prob_tile = prob_tile[
                    :original_h,
                    :original_w,
                ]


                probability_sum[
                    y:
                    y + original_h,

                    x:
                    x + original_w,
                ] += prob_tile


                count_sum[
                    y:
                    y + original_h,

                    x:
                    x + original_w,
                ] += 1.0


                if (
                    tile_counter
                    % int(
                        print_every
                    )
                    == 0
                    or tile_counter
                    == total_tiles
                ):

                    print(
                        f"Processed tiles: "
                        f"{tile_counter}/{total_tiles}"
                    )


    if np.any(
        count_sum
        <= 0
    ):

        raise RuntimeError(
            "Tiled inference left one or more raster pixels uncovered."
        )


    probability_mask = (
        probability_sum
        / count_sum
    )


    probability_mask = np.clip(
        probability_mask,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


    return probability_mask


def save_binary_mask_as_arcgis_raster(
    binary_mask,
    reference_raster_path,
    output_raster_path,
):
    """
    Save binary mask as georeferenced ArcGIS raster.

    1 = building
    0 = NoData/background
    """

    import arcpy

    reference_raster_path = str(
        reference_raster_path
    )

    output_raster_path = str(
        output_raster_path
    )

    reference_raster = arcpy.Raster(
        reference_raster_path
    )

    desc = arcpy.Describe(
        reference_raster_path
    )


    mask_array = (
        binary_mask
        .astype(
            np.uint8
        )
    )


    lower_left = arcpy.Point(
        reference_raster.extent.XMin,
        reference_raster.extent.YMin,
    )


    out_raster = (
        arcpy.NumPyArrayToRaster(
            mask_array,
            lower_left,
            reference_raster.meanCellWidth,
            reference_raster.meanCellHeight,
            value_to_nodata=0,
        )
    )


    if arcpy.Exists(
        output_raster_path
    ):

        arcpy.management.Delete(
            output_raster_path
        )


    out_raster.save(
        output_raster_path
    )


    arcpy.management.DefineProjection(
        output_raster_path,
        desc.spatialReference,
    )


    return output_raster_path


def create_file_geodatabase(
    gdb_path,
):
    """
    Create file geodatabase if it does not exist.
    """

    import arcpy

    gdb_path = Path(
        gdb_path
    )


    if not arcpy.Exists(
        str(
            gdb_path
        )
    ):

        arcpy.management.CreateFileGDB(
            str(
                gdb_path.parent
            ),
            gdb_path.name,
        )


    return str(
        gdb_path
    )


def raster_to_polygon_feature_class(
    binary_raster_path,
    final_gdb_path,
    output_fc_name="final_prediction_raw",
):
    """
    Convert binary prediction raster to polygon feature class.

    This is the RAW shared semantic prediction.

    Shared post-processing, when enabled, should operate downstream
    rather than being duplicated per model.
    """

    import arcpy

    binary_raster_path = str(
        binary_raster_path
    )

    final_gdb_path = create_file_geodatabase(
        final_gdb_path
    )


    temp_fc = str(
        Path(
            final_gdb_path
        )
        / "temp_prediction_polygon_all"
    )

    output_fc = str(
        Path(
            final_gdb_path
        )
        / output_fc_name
    )


    if arcpy.Exists(
        temp_fc
    ):

        arcpy.management.Delete(
            temp_fc
        )


    if arcpy.Exists(
        output_fc
    ):

        arcpy.management.Delete(
            output_fc
        )


    arcpy.conversion.RasterToPolygon(
        in_raster=binary_raster_path,
        out_polygon_features=temp_fc,
        simplify="NO_SIMPLIFY",
        raster_field="Value",
    )


    fields = [
        field.name.lower()

        for field
        in arcpy.ListFields(
            temp_fc
        )
    ]


    if "gridcode" in fields:

        value_field = (
            "gridcode"
        )


    elif "value" in fields:

        value_field = (
            "Value"
        )


    else:

        value_field = (
            None
        )


    if value_field is not None:

        where_clause = (
            f"{arcpy.AddFieldDelimiters(temp_fc, value_field)} = 1"
        )

        layer_name = (
            "prediction_building_layer"
        )


        arcpy.management.MakeFeatureLayer(
            temp_fc,
            layer_name,
            where_clause,
        )


        selected_count = int(
            arcpy.management.GetCount(
                layer_name
            )[0]
        )


        if selected_count > 0:

            arcpy.management.CopyFeatures(
                layer_name,
                output_fc,
            )


        else:

            desc = arcpy.Describe(
                binary_raster_path
            )


            arcpy.management.CreateFeatureclass(
                out_path=final_gdb_path,
                out_name=output_fc_name,
                geometry_type="POLYGON",
                spatial_reference=desc.spatialReference,
            )


    else:

        arcpy.management.CopyFeatures(
            temp_fc,
            output_fc,
        )


    if arcpy.Exists(
        temp_fc
    ):

        arcpy.management.Delete(
            temp_fc
        )


    final_count = int(
        arcpy.management.GetCount(
            output_fc
        )[0]
    )


    return (
        output_fc,
        final_count,
    )


def run_full_arcgis_raster_inference(
    model,
    device,
    test_image_path,
    output_folder,
    threshold,
    tile_size=256,
    overlap_ratio=0.25,
):
    """
    Full shared semantic inference on an ArcGIS raster.

    Compatible:
        U-Net
        DeepLabV3 / DeepLabV3+
        SAM-LoRA

    Saves
    -----
    temp_inference/raw_binary_prediction.tif

    final_prediction.gdb/final_prediction_raw

    The returned summary is later extended by run_inference.py.

    Important
    ---------
    This function creates the RAW prediction.

    Shared post-processing and final GT evaluation remain separate
    downstream stages so the same operations can be used for every
    compatible model.
    """

    import arcpy
    import time


    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Inference threshold must be between 0 and 1."
        )


    arcpy.env.overwriteOutput = True


    output_folder = Path(
        output_folder
    )


    temp_folder = (
        output_folder
        / "temp_inference"
    )


    temp_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    start_time = (
        time.time()
    )


    print(
        "\nReading ArcGIS raster..."
    )


    image_array, raster_info = (
        raster_to_rgb_numpy(
            test_image_path
        )
    )


    print(
        f"Raster array shape: "
        f"{image_array.shape}"
    )


    print(
        "Running shared tiled model inference..."
    )


    probability_mask = (
        predict_large_raster_array(
            model=model,
            image_array=image_array,
            device=device,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
        )
    )


    print(
        "Applying threshold..."
    )


    binary_mask = apply_threshold(
        prob_mask=probability_mask,
        threshold=threshold,
    )


    building_pixels = int(
        binary_mask.sum()
    )


    total_pixels = int(
        binary_mask.size
    )


    building_ratio = (
        building_pixels
        / max(
            total_pixels,
            1,
        )
    )


    binary_raster_path = (
        temp_folder
        / "raw_binary_prediction.tif"
    )


    print(
        "Saving georeferenced binary prediction raster..."
    )


    save_binary_mask_as_arcgis_raster(
        binary_mask=binary_mask,
        reference_raster_path=test_image_path,
        output_raster_path=binary_raster_path,
    )


    final_gdb_path = (
        output_folder
        / "final_prediction.gdb"
    )


    print(
        "Converting raster prediction to polygons..."
    )


    (
        final_prediction_fc,
        polygon_count,
    ) = (
        raster_to_polygon_feature_class(
            binary_raster_path=binary_raster_path,
            final_gdb_path=final_gdb_path,
            output_fc_name="final_prediction_raw",
        )
    )


    elapsed_time = (
        time.time()
        - start_time
    )


    summary = {
        "threshold":
            float(
                threshold
            ),

        "tile_size":
            int(
                tile_size
            ),

        "overlap_ratio":
            float(
                overlap_ratio
            ),

        "merge_method":
            "average_probability",

        "raster_info":
            raster_info,

        "building_pixels":
            building_pixels,

        "total_pixels":
            total_pixels,

        "building_ratio":
            float(
                building_ratio
            ),

        "binary_raster_path":
            str(
                binary_raster_path
            ),

        "final_gdb_path":
            str(
                final_gdb_path
            ),

        "final_prediction_fc":
            str(
                final_prediction_fc
            ),

        "polygon_count":
            int(
                polygon_count
            ),

        "postprocessing_applied":
            False,

        "prediction_stage":
            "raw_thresholded_prediction",

        "elapsed_time_seconds":
            round(
                elapsed_time,
                2,
            ),
    }


    return summary

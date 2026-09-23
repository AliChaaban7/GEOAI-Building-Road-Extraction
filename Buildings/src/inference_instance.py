"""
inference_instance.py

Mask R-CNN inference utilities for the Buildings module.

This file supports:
- loading Mask R-CNN checkpoints
- tiled inference on ArcGIS rasters
- score threshold + mask threshold search
- merging instance masks into full-size binary prediction masks
"""

from pathlib import Path

import numpy as np
import torch
import arcpy


# ============================================================
# Checkpoint helpers
# ============================================================

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_maskrcnn_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    return model


# ============================================================
# Model runtime settings
# ============================================================

def apply_maskrcnn_runtime_settings(
    model,
    min_score_threshold=0.05,
    nms_threshold=0.5,
    detections_per_image=300
):
    """
    Lower model score threshold during inference so we can test thresholds later.
    """

    if hasattr(model, "roi_heads"):
        model.roi_heads.score_thresh = float(min_score_threshold)
        model.roi_heads.nms_thresh = float(nms_threshold)
        model.roi_heads.detections_per_img = int(detections_per_image)

    return model


# ============================================================
# Raster helpers
# ============================================================

def raster_to_hwc_array(raster_path):
    arr = arcpy.RasterToNumPyArray(raster_path)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    elif arr.ndim == 3:
        # ArcPy usually returns [bands, rows, cols]
        if arr.shape[0] <= 10 and arr.shape[1] > 10 and arr.shape[2] > 10:
            arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    return arr


def normalize_tile_to_tensor(tile):
    tile = tile.astype(np.float32)
    tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)

    if tile.max() > 1.0:
        tile = np.clip(tile, 0, 255) / 255.0

    tile = np.transpose(tile, (2, 0, 1))

    return torch.from_numpy(tile).float()


def make_starts(length, tile_size, stride):
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)

    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size

    if starts[-1] != last:
        starts.append(last)

    return starts


def pad_image_for_tiles(image, y_starts, x_starts, tile_size):
    height, width, channels = image.shape

    target_height = max(y_starts) + tile_size
    target_width = max(x_starts) + tile_size

    pad_bottom = max(0, target_height - height)
    pad_right = max(0, target_width - width)

    if pad_bottom == 0 and pad_right == 0:
        return image

    padded = np.pad(
        image,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="reflect"
    )

    return padded


# ============================================================
# Threshold mask creation
# ============================================================

def create_empty_mask_dictionary(height, width, score_thresholds, mask_thresholds):
    mask_dict = {}

    for score_threshold in score_thresholds:
        for mask_threshold in mask_thresholds:
            key = (float(score_threshold), float(mask_threshold))
            mask_dict[key] = np.zeros((height, width), dtype=np.uint8)

    return mask_dict


def update_global_masks_from_detection(
    global_masks,
    mask_probability,
    score,
    y,
    x,
    usable_height,
    usable_width,
    score_thresholds,
    mask_thresholds
):
    score = float(score)

    if score < min(score_thresholds):
        return

    mask_probability = mask_probability[:usable_height, :usable_width]

    for mask_threshold in mask_thresholds:
        local_binary = (mask_probability >= float(mask_threshold)).astype(np.uint8)

        if int(local_binary.sum()) == 0:
            continue

        for score_threshold in score_thresholds:
            if score >= float(score_threshold):
                key = (float(score_threshold), float(mask_threshold))

                target_slice = global_masks[key][
                    y:y + usable_height,
                    x:x + usable_width
                ]

                np.maximum(
                    target_slice,
                    local_binary,
                    out=target_slice
                )


# ============================================================
# Tiled Mask R-CNN inference
# ============================================================

@torch.no_grad()
def run_maskrcnn_tiled_threshold_inference(
    model,
    device,
    image_array,
    tile_size,
    overlap_ratio,
    batch_size,
    score_thresholds,
    mask_thresholds,
    nms_threshold=0.5,
    detections_per_image=300
):
    """
    Run Mask R-CNN tiled inference once and create binary masks for all
    score/mask threshold combinations.

    Returns:
        global_masks:
            dict[(score_threshold, mask_threshold)] = binary mask [H, W]

        stats:
            inference statistics
    """

    model.eval()

    height, width, channels = image_array.shape

    score_thresholds = sorted([float(x) for x in score_thresholds])
    mask_thresholds = sorted([float(x) for x in mask_thresholds])

    min_score_threshold = min(score_thresholds)

    apply_maskrcnn_runtime_settings(
        model=model,
        min_score_threshold=min_score_threshold,
        nms_threshold=nms_threshold,
        detections_per_image=detections_per_image
    )

    stride = int(tile_size * (1.0 - overlap_ratio))
    stride = max(1, stride)

    y_starts = make_starts(height, tile_size, stride)
    x_starts = make_starts(width, tile_size, stride)

    padded = pad_image_for_tiles(
        image=image_array,
        y_starts=y_starts,
        x_starts=x_starts,
        tile_size=tile_size
    )

    tile_positions = []

    for y in y_starts:
        for x in x_starts:
            tile_positions.append((y, x))

    total_tiles = len(tile_positions)

    global_masks = create_empty_mask_dictionary(
        height=height,
        width=width,
        score_thresholds=score_thresholds,
        mask_thresholds=mask_thresholds
    )

    total_detections = 0
    kept_detections = 0

    print(f"Raster size: {width} x {height}")
    print(f"Tile size: {tile_size}")
    print(f"Stride: {stride}")
    print(f"Total tiles: {total_tiles}")
    print(f"Score thresholds: {score_thresholds}")
    print(f"Mask thresholds: {mask_thresholds}")

    processed = 0

    for start in range(0, total_tiles, batch_size):
        batch_positions = tile_positions[start:start + batch_size]
        batch_images = []

        for y, x in batch_positions:
            tile = padded[y:y + tile_size, x:x + tile_size, :]
            tile_tensor = normalize_tile_to_tensor(tile)
            batch_images.append(tile_tensor.to(device))

        outputs = model(batch_images)

        for output, (y, x) in zip(outputs, batch_positions):
            scores = output.get("scores", torch.empty(0)).detach().cpu().numpy()
            masks = output.get("masks", torch.empty(0)).detach().cpu().numpy()

            if masks.ndim == 4:
                masks = masks[:, 0, :, :]

            total_detections += int(len(scores))

            usable_height = min(tile_size, height - y)
            usable_width = min(tile_size, width - x)

            for det_idx, score in enumerate(scores):
                if float(score) < min_score_threshold:
                    continue

                if det_idx >= masks.shape[0]:
                    continue

                kept_detections += 1

                update_global_masks_from_detection(
                    global_masks=global_masks,
                    mask_probability=masks[det_idx],
                    score=float(score),
                    y=y,
                    x=x,
                    usable_height=usable_height,
                    usable_width=usable_width,
                    score_thresholds=score_thresholds,
                    mask_thresholds=mask_thresholds
                )

        processed += len(batch_positions)

        if processed % 25 == 0 or processed == total_tiles:
            print(
                f"Processed: {processed} / {total_tiles} | "
                f"Total detections: {total_detections} | "
                f"Kept detections: {kept_detections}"
            )

    stats = {
        "height": height,
        "width": width,
        "tile_size": tile_size,
        "stride": stride,
        "total_tiles": total_tiles,
        "total_detections": total_detections,
        "kept_detections": kept_detections,
        "score_thresholds": score_thresholds,
        "mask_thresholds": mask_thresholds,
        "nms_threshold": float(nms_threshold),
        "detections_per_image": int(detections_per_image)
    }

    return global_masks, stats
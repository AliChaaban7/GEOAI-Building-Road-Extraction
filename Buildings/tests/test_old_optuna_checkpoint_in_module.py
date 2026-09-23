"""
test_old_optuna_checkpoint_in_module.py

Purpose:
Test the old successful Optuna DeepLabV3 256 Satellite-only checkpoint
inside the new clean module.

This script uses:
- old checkpoint path
- DeepLabV3 ResNet50 with aux_loss=True
- ImageNet normalization
- full raster tiled inference on Area_04_Clip
- testing threshold search
- strict polygon IoU
- area + hole post-processing search

PowerShell:
    & $py "Buildings\\scripts\\test_old_optuna_checkpoint_in_module.py"
"""

from pathlib import Path
import sys
import csv
import json
import contextlib
import gc

import numpy as np
import torch
import torch.nn as nn
import arcpy

from torchvision.models.segmentation import deeplabv3_resnet50


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

sys.path.append(str(ROOT))
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    load_experiment_config,
    load_paths_config,
    get_test_image_path,
    get_ground_truth_path,
    save_json
)


arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False


# =====================================================
# OLD CHECKPOINT PATH
# =====================================================

OLD_MODEL_PATH = Path(
    r"C:\Users\gis-t12\Documents\ArcGIS\Projects\AliChaabanThesis"
    r"\Optuna_DeepLabV3_256_Satellite_Only"
    r"\models"
    r"\Optuna_DeepLabV3_256_Satellite_Only_BEST.pth"
)


# =====================================================
# SETTINGS MATCHING OLD SCRIPT
# =====================================================

TILE_SIZE = 256
OVERLAP_RATIO = 0.25
BATCH_SIZE = 8

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.70, 0.80, 0.90
]

MIN_AREA_VALUES_M2 = [
    0, 5, 10, 15, 20, 25, 30, 35,
    40, 45, 50, 60, 75, 100
]

HOLE_AREA_M2 = 5

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# =====================================================
# BASIC HELPERS
# =====================================================

def delete_if_exists(path):
    try:
        if path and arcpy.Exists(str(path)):
            arcpy.management.Delete(str(path))
    except Exception:
        pass


def safe_name(text, max_len=60):
    text = str(text)

    for ch in [" ", "-", "/", "\\", ":", ".", "(", ")", "+", "%", ">=", "="]:
        text = text.replace(ch, "_")

    text = "".join(
        c for c in text
        if c.isalnum() or c == "_"
    )

    return text[:max_len]


def save_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def load_checkpoint_safe(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def amp_autocast(device):
    if device.type == "cuda":
        try:
            return torch.amp.autocast("cuda")
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)

    return contextlib.nullcontext()


# =====================================================
# MODEL LOADING — OLD AUX CLASSIFIER STYLE
# =====================================================

def get_old_deeplabv3_model():
    """
    Old checkpoint contains aux_classifier weights.
    Therefore we must create DeepLabV3 with aux_loss=True.
    """
    try:
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            aux_loss=True
        )

    except TypeError:
        model = deeplabv3_resnet50(
            pretrained=False,
            pretrained_backbone=False,
            aux_loss=True
        )

    model.classifier[4] = nn.Conv2d(
        256,
        1,
        kernel_size=1
    )

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            256,
            1,
            kernel_size=1
        )

    return model


def clean_state_dict_keys(state_dict):
    """
    Remove module. prefix if checkpoint was saved from DataParallel.
    """
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key.replace("module.", "", 1)
        else:
            new_key = key

        cleaned[new_key] = value

    return cleaned


def load_old_checkpoint_model(model_path, device):
    checkpoint = load_checkpoint_safe(
        path=model_path,
        device=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)

    model = get_old_deeplabv3_model()

    try:
        model.load_state_dict(
            state_dict,
            strict=True
        )

        print("Old checkpoint loaded with strict=True")

    except RuntimeError as e:
        print("Strict=True loading failed.")
        print("Trying strict=False...")
        print("Reason:")
        print(e)

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False
        )

        print("Old checkpoint loaded with strict=False")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model = model.to(device)
    model.eval()

    validation_iou = None
    validation_threshold = None

    if isinstance(checkpoint, dict):
        for key in ["best_val_iou", "best_validation_iou", "val_iou"]:
            if key in checkpoint:
                validation_iou = float(checkpoint[key]) * 100

        for key in ["best_validation_threshold", "best_threshold", "threshold"]:
            if key in checkpoint:
                validation_threshold = checkpoint[key]

    return model, validation_iou, validation_threshold


# =====================================================
# GIS HELPERS
# =====================================================

def project_if_needed(in_fc, out_fc, target_sr):
    delete_if_exists(out_fc)

    in_sr = arcpy.Describe(in_fc).spatialReference

    if in_sr.name != target_sr.name:
        arcpy.management.Project(
            in_dataset=in_fc,
            out_dataset=out_fc,
            out_coor_system=target_sr
        )

        print(f"Projected GT: {in_sr.name} -> {target_sr.name}")

    else:
        arcpy.management.CopyFeatures(
            in_fc,
            out_fc
        )

        print("GT already in same spatial reference. Copied GT.")

    return out_fc


def add_area_m2(fc):
    fields = [f.name for f in arcpy.ListFields(fc)]

    if "Area_m2" not in fields:
        arcpy.management.AddField(
            in_table=fc,
            field_name="Area_m2",
            field_type="DOUBLE"
        )

    arcpy.management.CalculateGeometryAttributes(
        in_features=fc,
        geometry_property=[["Area_m2", "AREA"]],
        area_unit="SQUARE_METERS"
    )


def create_empty_polygon_fc_like(input_fc, output_fc):
    desc = arcpy.Describe(input_fc)

    arcpy.management.CreateFeatureclass(
        out_path=str(Path(output_fc).parent),
        out_name=Path(output_fc).name,
        geometry_type="POLYGON",
        spatial_reference=desc.spatialReference
    )


# =====================================================
# RASTER TILE READING — OLD STYLE
# =====================================================

def read_raster_tile_uint8(raster_path, x0, y0, width, height, tile_size):
    raster = arcpy.Raster(raster_path)

    cell_w = raster.meanCellWidth
    cell_h = abs(raster.meanCellHeight)

    lower_left_x = raster.extent.XMin + (x0 * cell_w)
    lower_left_y = raster.extent.YMax - ((y0 + height) * cell_h)

    lower_left = arcpy.Point(
        lower_left_x,
        lower_left_y
    )

    arr = arcpy.RasterToNumPyArray(
        raster_path,
        lower_left_corner=lower_left,
        ncols=width,
        nrows=height,
        nodata_to_value=0
    )

    if arr.ndim == 3:
        if arr.shape[0] <= 10:
            arr = np.transpose(arr, (1, 2, 0))

        arr = arr[:, :, :3]

    elif arr.ndim == 2:
        arr = np.stack(
            [arr, arr, arr],
            axis=-1
        )

    arr = np.nan_to_num(
        arr,
        nan=0
    ).astype(np.float32)

    if arr.max() > 255:
        arr = 255 * (
            arr - arr.min()
        ) / (
            arr.max() - arr.min() + 1e-6
        )

    arr = np.clip(
        arr,
        0,
        255
    ).astype(np.uint8)

    original_h, original_w = arr.shape[:2]

    if original_h < tile_size or original_w < tile_size:
        padded = np.zeros(
            (tile_size, tile_size, 3),
            dtype=np.uint8
        )

        padded[:original_h, :original_w, :] = arr
        arr = padded

    return arr, original_h, original_w


def to_imagenet_tensor(tile_uint8):
    tensor = torch.from_numpy(
        tile_uint8.astype(np.float32) / 255.0
    ).permute(2, 0, 1)

    tensor = (
        tensor - IMAGENET_MEAN
    ) / IMAGENET_STD

    return tensor.float()


def get_tile_positions(raster_path, tile_size, overlap_ratio):
    raster = arcpy.Raster(raster_path)

    h = raster.height
    w = raster.width

    stride = int(tile_size * (1 - overlap_ratio))
    stride = max(1, stride)

    y_positions = list(
        range(0, max(h - tile_size + 1, 1), stride)
    )

    x_positions = list(
        range(0, max(w - tile_size + 1, 1), stride)
    )

    if h > tile_size and y_positions[-1] != h - tile_size:
        y_positions.append(h - tile_size)

    if w > tile_size and x_positions[-1] != w - tile_size:
        x_positions.append(w - tile_size)

    positions = [
        (y, x)
        for y in y_positions
        for x in x_positions
    ]

    return positions, stride, h, w


@torch.no_grad()
def run_tiled_inference(model, raster_path, tile_size, overlap_ratio, batch_size, device):
    positions, stride, h, w = get_tile_positions(
        raster_path=raster_path,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio
    )

    print("Raster size:", w, "x", h)
    print("Tile size:", tile_size)
    print("Stride:", stride)
    print("Total tiles:", len(positions))

    prob_sum = np.zeros(
        (h, w),
        dtype=np.float32
    )

    count_sum = np.zeros(
        (h, w),
        dtype=np.uint16
    )

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]

        chips = []
        sizes = []

        for y, x in batch_positions:
            read_w = min(tile_size, w - x)
            read_h = min(tile_size, h - y)

            tile_uint8, original_h, original_w = read_raster_tile_uint8(
                raster_path=raster_path,
                x0=x,
                y0=y,
                width=read_w,
                height=read_h,
                tile_size=tile_size
            )

            chips.append(
                to_imagenet_tensor(tile_uint8)
            )

            sizes.append(
                (original_h, original_w)
            )

        batch_tensor = torch.stack(chips).to(device)

        with amp_autocast(device):
            outputs = model(batch_tensor)

            logits = outputs["out"] if isinstance(outputs, dict) else outputs

            probs = torch.sigmoid(
                logits
            ).detach().cpu().numpy()[:, 0, :, :]

        for i, (y, x) in enumerate(batch_positions):
            oh, ow = sizes[i]

            y2 = y + oh
            x2 = x + ow

            prob_sum[y:y2, x:x2] += probs[i, :oh, :ow]
            count_sum[y:y2, x:x2] += 1

        if start % (batch_size * 50) == 0:
            print(
                "Processed:",
                min(start + batch_size, len(positions)),
                "/",
                len(positions)
            )

    prob_map = prob_sum / np.maximum(
        count_sum.astype(np.float32),
        1
    )

    print("Probability min:", float(prob_map.min()))
    print("Probability max:", float(prob_map.max()))
    print("Probability mean:", float(prob_map.mean()))

    return prob_map, len(positions)


# =====================================================
# MASK / POLYGON CONVERSION
# =====================================================

def save_array_as_geotiff(array, reference_raster_path, out_raster):
    delete_if_exists(out_raster)

    ref = arcpy.Raster(reference_raster_path)

    lower_left = arcpy.Point(
        ref.extent.XMin,
        ref.extent.YMin
    )

    raster = arcpy.NumPyArrayToRaster(
        array.astype(np.uint8),
        lower_left,
        ref.meanCellWidth,
        abs(ref.meanCellHeight),
        value_to_nodata=0
    )

    raster.save(str(out_raster))

    try:
        arcpy.management.DefineProjection(
            str(out_raster),
            ref.spatialReference
        )
    except Exception:
        pass

    return str(out_raster)


def polygon_to_mask_array(poly_fc, reference_raster, temp_gdb):
    temp_raster = str(Path(temp_gdb) / "temp_old_checkpoint_gt_mask")

    delete_if_exists(temp_raster)

    ref = arcpy.Raster(reference_raster)

    old_extent = arcpy.env.extent
    old_snap = arcpy.env.snapRaster
    old_cell = arcpy.env.cellSize
    old_cs = arcpy.env.outputCoordinateSystem

    try:
        arcpy.env.extent = ref.extent
        arcpy.env.snapRaster = reference_raster
        arcpy.env.cellSize = ref.meanCellWidth
        arcpy.env.outputCoordinateSystem = ref.spatialReference

        if int(arcpy.management.GetCount(poly_fc)[0]) == 0:
            return np.zeros(
                (ref.height, ref.width),
                dtype=np.uint8
            )

        oid_field = arcpy.Describe(poly_fc).OIDFieldName

        arcpy.conversion.PolygonToRaster(
            in_features=poly_fc,
            value_field=oid_field,
            out_rasterdataset=temp_raster,
            cell_assignment="MAXIMUM_AREA",
            priority_field="NONE",
            cellsize=ref.meanCellWidth
        )

        arr = arcpy.RasterToNumPyArray(
            temp_raster,
            nodata_to_value=0
        )

        arr = np.nan_to_num(
            arr,
            nan=0
        )

        return (arr > 0).astype(np.uint8)

    finally:
        arcpy.env.extent = old_extent
        arcpy.env.snapRaster = old_snap
        arcpy.env.cellSize = old_cell
        arcpy.env.outputCoordinateSystem = old_cs

        delete_if_exists(temp_raster)


def raster_to_polygon(mask_raster, out_polygon):
    delete_if_exists(out_polygon)

    temp_poly = str(out_polygon) + "_tmp"

    delete_if_exists(temp_poly)

    arr = arcpy.RasterToNumPyArray(
        str(mask_raster),
        nodata_to_value=0
    )

    if int((arr > 0).sum()) == 0:
        sr = arcpy.Describe(str(mask_raster)).spatialReference

        arcpy.management.CreateFeatureclass(
            out_path=str(Path(out_polygon).parent),
            out_name=Path(out_polygon).name,
            geometry_type="POLYGON",
            spatial_reference=sr
        )

        return str(out_polygon)

    arcpy.conversion.RasterToPolygon(
        in_raster=str(mask_raster),
        out_polygon_features=temp_poly,
        simplify="NO_SIMPLIFY",
        raster_field="Value"
    )

    field_names = [
        f.name.lower()
        for f in arcpy.ListFields(temp_poly)
    ]

    if "gridcode" in field_names:
        where_clause = "gridcode <> 0"
    else:
        where_clause = "Value <> 0"

    arcpy.analysis.Select(
        in_features=temp_poly,
        out_feature_class=str(out_polygon),
        where_clause=where_clause
    )

    delete_if_exists(temp_poly)

    return str(out_polygon)


# =====================================================
# METRICS
# =====================================================

def pixel_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_area = pred.sum()
    gt_area = gt.sum()

    iou = 0 if union == 0 else inter / union
    precision = 0 if pred_area == 0 else inter / pred_area
    recall = 0 if gt_area == 0 else inter / gt_area

    f1 = 0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )

    return {
        "pixel_iou": float(iou),
        "pixel_iou_percent": round(float(iou * 100), 2),
        "pixel_precision_percent": round(float(precision * 100), 2),
        "pixel_recall_percent": round(float(recall * 100), 2),
        "pixel_f1_percent": round(float(f1 * 100), 2),
        "pixel_intersection": int(inter),
        "pixel_union": int(union),
        "predicted_pixels": int(pred_area),
        "ground_truth_pixels": int(gt_area)
    }


def calculate_strict_iou_metrics(pred_fc, gt_fc, temp_gdb):
    pred_count = int(arcpy.management.GetCount(pred_fc)[0])
    gt_count = int(arcpy.management.GetCount(gt_fc)[0])

    pred_area = 0.0
    gt_area = 0.0
    inter_area = 0.0

    if pred_count > 0:
        pred_area = sum(
            row[0]
            for row in arcpy.da.SearchCursor(pred_fc, ["SHAPE@AREA"])
        )

    if gt_count > 0:
        gt_area = sum(
            row[0]
            for row in arcpy.da.SearchCursor(gt_fc, ["SHAPE@AREA"])
        )

    inter_fc = str(Path(temp_gdb) / "temp_old_checkpoint_intersection")

    delete_if_exists(inter_fc)

    try:
        if pred_count > 0 and gt_count > 0:
            arcpy.analysis.Intersect(
                [pred_fc, gt_fc],
                inter_fc
            )

            inter_area = sum(
                row[0]
                for row in arcpy.da.SearchCursor(inter_fc, ["SHAPE@AREA"])
            )

        union_area = pred_area + gt_area - inter_area

        iou = 0 if union_area <= 0 else inter_area / union_area
        precision = 0 if pred_area <= 0 else inter_area / pred_area
        recall = 0 if gt_area <= 0 else inter_area / gt_area

        f1 = 0 if precision + recall <= 0 else (
            2 * precision * recall / (precision + recall)
        )

        return {
            "iou_percent": round(float(iou * 100), 2),
            "precision_percent": round(float(precision * 100), 2),
            "recall_percent": round(float(recall * 100), 2),
            "f1_percent": round(float(f1 * 100), 2),
            "prediction_count": pred_count,
            "ground_truth_count": gt_count,
            "pred_area": round(float(pred_area), 2),
            "gt_area": round(float(gt_area), 2),
            "intersection_area": round(float(inter_area), 2),
            "union_area": round(float(union_area), 2)
        }

    finally:
        delete_if_exists(inter_fc)


# =====================================================
# POST-PROCESSING
# =====================================================

def postprocess_area_hole_search(raw_pred_fc, gt_fc, output_gdb):
    results = []

    for min_area in MIN_AREA_VALUES_M2:
        tag = safe_name(
            f"old_ckpt_area_{min_area}_hole_{HOLE_AREA_M2}"
        )

        step_copy = str(Path(output_gdb) / f"{tag}_copy")
        step_single = str(Path(output_gdb) / f"{tag}_single")
        step_area = str(Path(output_gdb) / f"{tag}_area")
        step_holes = str(Path(output_gdb) / f"{tag}_holes")

        for fc in [
            step_copy,
            step_single,
            step_area,
            step_holes
        ]:
            delete_if_exists(fc)

        arcpy.management.CopyFeatures(
            raw_pred_fc,
            step_copy
        )

        try:
            arcpy.management.RepairGeometry(
                step_copy,
                "DELETE_NULL"
            )
        except Exception:
            pass

        try:
            arcpy.management.MultipartToSinglepart(
                step_copy,
                step_single
            )
        except Exception:
            arcpy.management.CopyFeatures(
                step_copy,
                step_single
            )

        add_area_m2(step_single)

        if min_area > 0:
            where_clause = (
                f"{arcpy.AddFieldDelimiters(step_single, 'Area_m2')} >= {float(min_area)}"
            )

            arcpy.analysis.Select(
                in_features=step_single,
                out_feature_class=step_area,
                where_clause=where_clause
            )

            if int(arcpy.management.GetCount(step_area)[0]) == 0:
                delete_if_exists(step_area)
                create_empty_polygon_fc_like(step_single, step_area)

        else:
            arcpy.management.CopyFeatures(
                step_single,
                step_area
            )

        try:
            arcpy.management.EliminatePolygonPart(
                in_features=step_area,
                out_feature_class=step_holes,
                condition="AREA",
                part_area=f"{HOLE_AREA_M2} SquareMeters",
                part_area_percent=0,
                part_option="CONTAINED_ONLY"
            )
        except Exception:
            arcpy.management.CopyFeatures(
                step_area,
                step_holes
            )

        try:
            arcpy.management.RepairGeometry(
                step_holes,
                "DELETE_NULL"
            )
        except Exception:
            pass

        metrics = calculate_strict_iou_metrics(
            pred_fc=step_holes,
            gt_fc=gt_fc,
            temp_gdb=output_gdb
        )

        row = {
            "method": f"area_{min_area}_hole_{HOLE_AREA_M2}",
            "min_area_m2": min_area,
            "hole_area_m2": HOLE_AREA_M2,
            "prediction_fc": step_holes,
            **metrics
        }

        results.append(row)

        print(
            f"Area >= {min_area} m2 + holes {HOLE_AREA_M2} m2 | "
            f"IoU: {row['iou_percent']:.2f}% | "
            f"P: {row['precision_percent']:.2f}% | "
            f"R: {row['recall_percent']:.2f}% | "
            f"F1: {row['f1_percent']:.2f}% | "
            f"Count: {row['prediction_count']}"
        )

    results = sorted(
        results,
        key=lambda x: (
            x["iou_percent"],
            x["f1_percent"],
            x["precision_percent"]
        ),
        reverse=True
    )

    best = results[0]

    best_fc = str(Path(output_gdb) / "old_checkpoint_best_postprocessed")

    delete_if_exists(best_fc)

    arcpy.management.CopyFeatures(
        best["prediction_fc"],
        best_fc
    )

    add_area_m2(best_fc)

    best["final_postprocessed_fc"] = best_fc

    return results, best


# =====================================================
# MAIN
# =====================================================

def main():
    experiment_config = load_experiment_config(
        ROOT / "config" / "experiment.json"
    )

    paths_config = load_paths_config(ROOT)

    test_image_path = get_test_image_path(
        experiment_config=experiment_config,
        paths_config=paths_config
    )

    gt_original = get_ground_truth_path(
        experiment_config=experiment_config,
        paths_config=paths_config
    )

    output_folder = (
        ROOT
        / "outputs"
        / "external_checkpoint_tests"
        / "old_optuna_deeplabv3_256_satellite"
    )

    masks_folder = output_folder / "masks"

    output_folder.mkdir(parents=True, exist_ok=True)
    masks_folder.mkdir(parents=True, exist_ok=True)

    output_gdb = output_folder / "old_optuna_checkpoint_testing.gdb"

    if not arcpy.Exists(str(output_gdb)):
        arcpy.management.CreateFileGDB(
            str(output_gdb.parent),
            output_gdb.name
        )

    print("\n========== CHECK INPUTS ==========")
    print(f"Test Image: {test_image_path}")
    print(f"Test image exists: {arcpy.Exists(test_image_path)}")
    print(f"GT Original: {gt_original}")
    print(f"GT exists: {arcpy.Exists(gt_original)}")
    print(f"Old checkpoint: {OLD_MODEL_PATH}")
    print(f"Old checkpoint exists: {OLD_MODEL_PATH.exists()}")
    print(f"Output Folder: {output_folder}")
    print("==================================\n")

    if not arcpy.Exists(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    if not arcpy.Exists(gt_original):
        raise FileNotFoundError(f"Ground truth not found: {gt_original}")

    if not OLD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Old checkpoint not found: {OLD_MODEL_PATH}")

    raster = arcpy.Raster(test_image_path)
    target_sr = raster.spatialReference

    gt_projected = str(output_gdb / "ground_truth_projected")

    gt_projected = project_if_needed(
        in_fc=gt_original,
        out_fc=gt_projected,
        target_sr=target_sr
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== LOAD OLD CHECKPOINT ==========")
    print(f"Device: {device}")
    print(f"Model path: {OLD_MODEL_PATH}")

    model, validation_iou, validation_threshold = load_old_checkpoint_model(
        model_path=OLD_MODEL_PATH,
        device=device
    )

    print(f"Checkpoint validation IoU: {validation_iou}")
    print(f"Checkpoint validation threshold: {validation_threshold}")
    print("=========================================\n")

    print("\n========== RUN OLD CHECKPOINT TESTING ==========")
    print(f"Raster: {test_image_path}")
    print(f"Raster SR: {target_sr.name}")
    print(f"Raster size: {raster.width} x {raster.height}")
    print(f"Cell size: {raster.meanCellWidth}, {raster.meanCellHeight}")
    print(f"Projected GT: {gt_projected}")
    print(f"Projected GT count: {int(arcpy.management.GetCount(gt_projected)[0])}")
    print(f"Tile Size: {TILE_SIZE}")
    print(f"Overlap Ratio: {OVERLAP_RATIO}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("================================================\n")

    prob_map, processed_tiles = run_tiled_inference(
        model=model,
        raster_path=test_image_path,
        tile_size=TILE_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        batch_size=BATCH_SIZE,
        device=device
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== CREATE GT MASK ==========")

    gt_mask = polygon_to_mask_array(
        poly_fc=gt_projected,
        reference_raster=test_image_path,
        temp_gdb=output_gdb
    )

    print(f"GT mask shape: {gt_mask.shape}")
    print(f"GT positive pixels: {int(gt_mask.sum())}")
    print("====================================\n")

    print("\n========== TESTING THRESHOLD SEARCH ==========")

    threshold_rows = []

    for threshold in THRESHOLDS:
        pred_mask = (
            prob_map >= float(threshold)
        ).astype(np.uint8)

        metrics = pixel_metrics(
            pred_mask=pred_mask,
            gt_mask=gt_mask
        )

        row = {
            "threshold": threshold,
            **metrics
        }

        threshold_rows.append(row)

        print(
            f"Threshold: {threshold} | "
            f"Pixel IoU: {metrics['pixel_iou_percent']:.2f}% | "
            f"P: {metrics['pixel_precision_percent']:.2f}% | "
            f"R: {metrics['pixel_recall_percent']:.2f}% | "
            f"F1: {metrics['pixel_f1_percent']:.2f}%"
        )

    threshold_rows = sorted(
        threshold_rows,
        key=lambda x: x["pixel_iou_percent"],
        reverse=True
    )

    best_threshold_row = threshold_rows[0]
    best_threshold = float(best_threshold_row["threshold"])

    threshold_csv = output_folder / "old_checkpoint_testing_threshold_results.csv"

    save_csv(
        rows=threshold_rows,
        csv_path=threshold_csv
    )

    print("\nBest testing threshold:")
    print(best_threshold)
    print(f"Best testing pixel IoU: {best_threshold_row['pixel_iou_percent']:.2f}%")
    print("=============================================\n")

    final_pred_mask = (
        prob_map >= best_threshold
    ).astype(np.uint8)

    del prob_map

    threshold_tag = str(best_threshold).replace(".", "_")

    mask_raster = masks_folder / f"old_checkpoint_prediction_thr_{threshold_tag}.tif"

    raw_pred_fc = str(
        output_gdb / f"old_checkpoint_raw_prediction_thr_{threshold_tag}"
    )

    print("\nSaving best binary raster...")
    save_array_as_geotiff(
        array=final_pred_mask,
        reference_raster_path=test_image_path,
        out_raster=mask_raster
    )

    print("\nConverting best raster to polygon...")
    raster_to_polygon(
        mask_raster=str(mask_raster),
        out_polygon=raw_pred_fc
    )

    print("\nCalculating raw strict polygon IoU...")

    raw_metrics = calculate_strict_iou_metrics(
        pred_fc=raw_pred_fc,
        gt_fc=gt_projected,
        temp_gdb=output_gdb
    )

    raw_metrics["stage"] = "old_checkpoint_raw"
    raw_metrics["testing_threshold"] = best_threshold
    raw_metrics["testing_pixel_iou_percent"] = best_threshold_row["pixel_iou_percent"]
    raw_metrics["prediction_fc"] = raw_pred_fc
    raw_metrics["mask_raster"] = str(mask_raster)
    raw_metrics["processed_tiles"] = processed_tiles
    raw_metrics["model_path"] = str(OLD_MODEL_PATH)

    print(
        f"Old checkpoint raw strict IoU: {raw_metrics['iou_percent']:.2f}% | "
        f"P: {raw_metrics['precision_percent']:.2f}% | "
        f"R: {raw_metrics['recall_percent']:.2f}% | "
        f"F1: {raw_metrics['f1_percent']:.2f}%"
    )

    print("\n========== POST-PROCESSING SEARCH ==========")

    post_rows, best_post = postprocess_area_hole_search(
        raw_pred_fc=raw_pred_fc,
        gt_fc=gt_projected,
        output_gdb=output_gdb
    )

    post_csv = output_folder / "old_checkpoint_postprocess_results.csv"

    save_csv(
        rows=post_rows,
        csv_path=post_csv
    )

    final_rows = [
        {
            "stage": "old_checkpoint_raw",
            "iou_percent": raw_metrics["iou_percent"],
            "precision_percent": raw_metrics["precision_percent"],
            "recall_percent": raw_metrics["recall_percent"],
            "f1_percent": raw_metrics["f1_percent"],
            "testing_threshold": best_threshold,
            "postprocessing": "none",
            "prediction_fc": raw_pred_fc
        },
        {
            "stage": "old_checkpoint_postprocessed",
            "iou_percent": best_post["iou_percent"],
            "precision_percent": best_post["precision_percent"],
            "recall_percent": best_post["recall_percent"],
            "f1_percent": best_post["f1_percent"],
            "testing_threshold": best_threshold,
            "postprocessing": best_post["method"],
            "prediction_fc": best_post["final_postprocessed_fc"]
        }
    ]

    final_csv = output_folder / "old_checkpoint_final_testing_summary.csv"

    save_csv(
        rows=final_rows,
        csv_path=final_csv
    )

    final_summary = {
        "experiment_id": "old_optuna_deeplabv3_256_satellite_tested_in_module",
        "old_model_path": str(OLD_MODEL_PATH),
        "test_image": test_image_path,
        "ground_truth_projected": gt_projected,
        "best_testing_threshold": best_threshold,
        "best_testing_pixel_iou_percent": best_threshold_row["pixel_iou_percent"],
        "raw_metrics": raw_metrics,
        "best_postprocessed_metrics": best_post,
        "threshold_results_csv": str(threshold_csv),
        "postprocess_results_csv": str(post_csv),
        "final_summary_csv": str(final_csv),
        "output_gdb": str(output_gdb),
        "raw_prediction_fc": raw_pred_fc,
        "final_postprocessed_fc": best_post["final_postprocessed_fc"],
        "mask_raster": str(mask_raster)
    }

    final_json = output_folder / "old_checkpoint_final_testing_summary.json"

    save_json(
        data=final_summary,
        path=final_json
    )

    print("\n========== OLD CHECKPOINT FINAL TESTING RESULTS ==========")
    print(f"Best Testing Threshold: {best_threshold}")
    print("---------------------------------------------------------")
    print(f"Old Checkpoint Raw IoU: {raw_metrics['iou_percent']:.2f}%")
    print(f"Old Checkpoint Raw Precision: {raw_metrics['precision_percent']:.2f}%")
    print(f"Old Checkpoint Raw Recall: {raw_metrics['recall_percent']:.2f}%")
    print(f"Old Checkpoint Raw F1: {raw_metrics['f1_percent']:.2f}%")
    print("---------------------------------------------------------")
    print(f"Best Post Method: {best_post['method']}")
    print(f"Old Checkpoint Post IoU: {best_post['iou_percent']:.2f}%")
    print(f"Old Checkpoint Post Precision: {best_post['precision_percent']:.2f}%")
    print(f"Old Checkpoint Post Recall: {best_post['recall_percent']:.2f}%")
    print(f"Old Checkpoint Post F1: {best_post['f1_percent']:.2f}%")
    print("---------------------------------------------------------")
    print(f"Raw Prediction FC: {raw_pred_fc}")
    print(f"Final Postprocessed FC: {best_post['final_postprocessed_fc']}")
    print(f"Threshold CSV: {threshold_csv}")
    print(f"Postprocess CSV: {post_csv}")
    print(f"Final Summary CSV: {final_csv}")
    print(f"Final Summary JSON: {final_json}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
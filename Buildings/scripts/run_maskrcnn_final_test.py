"""
One-time frozen Mask R-CNN final test.

This script DOES NOT search thresholds or post-processing settings.
It reads final_settings.json, runs the source-specific geographic test once,
applies the already-frozen score/mask/min-area settings, and reports strict
polygon IoU / Precision / Recall / F1.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

import arcpy
import numpy as np
import torch

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.utils import (
    load_experiment_config,
    load_paths_config,
    get_test_image_path,
    get_ground_truth_path,
    create_experiment_output_folder,
)
from src.instance.maskrcnn_engine import load_checkpoint_model

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False


def utc_now_iso():
    """Return an ISO-like UTC timestamp without depending on datetime imports."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen Mask R-CNN final test.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BUILDINGS_ROOT / "config" / "experiment.json"),
    )
    parser.add_argument("--settings", type=str, default=None)
    parser.add_argument("--allow-rerun", action="store_true")
    return parser.parse_args()


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def delete_if_exists(path):
    if path and arcpy.Exists(str(path)):
        arcpy.management.Delete(str(path))


def get_count(feature_class) -> int:
    return int(arcpy.management.GetCount(str(feature_class))[0])


def same_spatial_reference(sr_a, sr_b) -> bool:
    if sr_a is None or sr_b is None:
        return False
    try:
        if sr_a.factoryCode and sr_b.factoryCode:
            return sr_a.factoryCode == sr_b.factoryCode
    except Exception:
        pass
    return sr_a.name == sr_b.name


def read_arcgis_rgb_raster(raster_path):
    arr = np.asarray(arcpy.RasterToNumPyArray(raster_path, nodata_to_value=0))
    if arr.ndim == 3:
        if arr.shape[0] <= 10:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported raster shape: {arr.shape}")
    arr = np.nan_to_num(arr, nan=0).astype(np.float32)
    max_value = float(arr.max()) if arr.size else 0.0
    if max_value > 255.0:
        min_value = float(arr.min())
        arr = 255.0 * (arr - min_value) / (max_value - min_value + 1e-6)
    return np.clip(arr, 0, 255).astype(np.uint8)


def generate_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


@torch.no_grad()
def run_tiled_inference(
    model,
    image_array,
    tile_size,
    overlap_ratio,
    score_threshold,
    mask_threshold,
    device,
):
    h, w, _ = image_array.shape
    stride = max(1, int(tile_size * (1.0 - overlap_ratio)))
    ys = generate_positions(h, tile_size, stride)
    xs = generate_positions(w, tile_size, stride)
    positions = [(y, x) for y in ys for x in xs]
    prediction = np.zeros((h, w), dtype=bool)

    print(f"Raster: {w} x {h}")
    print(f"Tile Size: {tile_size}")
    print(f"Overlap: {overlap_ratio}")
    print(f"Tiles: {len(positions)}")

    for index, (y, x) in enumerate(positions, start=1):
        y2 = min(y + tile_size, h)
        x2 = min(x + tile_size, w)
        tile = image_array[y:y2, x:x2, :]
        oh, ow = tile.shape[:2]
        if oh != tile_size or ow != tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:oh, :ow, :] = tile
            tile = padded

        tensor = (
            torch.from_numpy(tile.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(device)
        )
        output = model([tensor])[0]
        scores = output.get("scores")
        masks = output.get("masks")
        if scores is not None and masks is not None:
            scores_np = scores.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()[:, 0]
            local = np.zeros((oh, ow), dtype=bool)
            for score, mask in zip(scores_np, masks_np):
                if float(score) >= float(score_threshold):
                    local |= mask[:oh, :ow] >= float(mask_threshold)
            prediction[y:y + oh, x:x + ow] |= local

        if index % 25 == 0 or index == len(positions):
            print(f"Processed {index}/{len(positions)} tiles")

    return prediction.astype(np.uint8)


def save_binary_mask_raster(mask, reference_raster, output_raster):
    delete_if_exists(output_raster)
    ref = arcpy.Raster(reference_raster)
    lower_left = arcpy.Point(ref.extent.XMin, ref.extent.YMin)
    out = arcpy.NumPyArrayToRaster(
        mask.astype(np.uint8),
        lower_left,
        ref.meanCellWidth,
        ref.meanCellHeight,
        value_to_nodata=0,
    )
    out.save(output_raster)
    try:
        arcpy.management.DefineProjection(output_raster, ref.spatialReference)
    except Exception:
        pass
    return output_raster


def raster_to_polygon(mask_raster, output_fc, workspace):
    delete_if_exists(output_fc)
    temp_fc = str(Path(workspace) / "temp_raw_prediction_polygon")
    delete_if_exists(temp_fc)
    arcpy.conversion.RasterToPolygon(
        in_raster=mask_raster,
        out_polygon_features=temp_fc,
        simplify="NO_SIMPLIFY",
        raster_field="Value",
        create_multipart_features="SINGLE_OUTER_PART",
    )
    fields = {f.name.lower(): f.name for f in arcpy.ListFields(temp_fc)}
    value_field = fields.get("gridcode") or fields.get("value")
    if value_field:
        delim = arcpy.AddFieldDelimiters(temp_fc, value_field)
        arcpy.analysis.Select(temp_fc, output_fc, f"{delim} <> 0")
    else:
        arcpy.management.CopyFeatures(temp_fc, output_fc)
    delete_if_exists(temp_fc)
    return output_fc


def filter_min_area(input_fc, output_fc, min_area_m2):
    delete_if_exists(output_fc)
    if float(min_area_m2) <= 0:
        arcpy.management.CopyFeatures(input_fc, output_fc)
        return output_fc
    layer = "maskrcnn_final_area_filter"
    arcpy.management.MakeFeatureLayer(input_fc, layer)
    try:
        arcpy.management.SelectLayerByAttribute(
            layer,
            "NEW_SELECTION",
            f"Shape_Area >= {float(min_area_m2)}",
        )
        arcpy.management.CopyFeatures(layer, output_fc)
    finally:
        try:
            arcpy.management.Delete(layer)
        except Exception:
            pass
    return output_fc


def project_ground_truth(gt_source, test_raster, output_fc):
    delete_if_exists(output_fc)
    raster_sr = arcpy.Raster(test_raster).spatialReference
    gt_sr = arcpy.Describe(gt_source).spatialReference
    if same_spatial_reference(gt_sr, raster_sr):
        arcpy.management.CopyFeatures(gt_source, output_fc)
    else:
        arcpy.management.Project(gt_source, output_fc, raster_sr)
    return output_fc


def strict_polygon_metrics(prediction_fc, gt_fc, workspace):
    pred_count = get_count(prediction_fc)
    gt_count = get_count(gt_fc)
    pred_area = sum(
        float(r[0]) for r in arcpy.da.SearchCursor(prediction_fc, ["SHAPE@AREA"])
    ) if pred_count else 0.0
    gt_area = sum(
        float(r[0]) for r in arcpy.da.SearchCursor(gt_fc, ["SHAPE@AREA"])
    ) if gt_count else 0.0
    inter_fc = str(Path(workspace) / "temp_strict_intersection")
    delete_if_exists(inter_fc)
    inter_area = 0.0
    inter_count = 0
    try:
        if pred_count and gt_count:
            arcpy.analysis.Intersect([prediction_fc, gt_fc], inter_fc)
            inter_count = get_count(inter_fc)
            inter_area = sum(
                float(r[0]) for r in arcpy.da.SearchCursor(inter_fc, ["SHAPE@AREA"])
            )
    finally:
        delete_if_exists(inter_fc)
    union_area = pred_area + gt_area - inter_area
    iou = 0.0 if union_area <= 0 else inter_area / union_area
    precision = 0.0 if pred_area <= 0 else inter_area / pred_area
    recall = 0.0 if gt_area <= 0 else inter_area / gt_area
    f1 = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
    return {
        "prediction_count": pred_count,
        "ground_truth_count": gt_count,
        "intersection_count": inter_count,
        "prediction_area": pred_area,
        "ground_truth_area": gt_area,
        "intersection_area": inter_area,
        "union_area": union_area,
        "iou": iou,
        "iou_percent": iou * 100.0,
        "precision": precision,
        "precision_percent": precision * 100.0,
        "recall": recall,
        "recall_percent": recall * 100.0,
        "f1": f1,
        "f1_percent": f1 * 100.0,
    }


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    output_folder = Path(
        create_experiment_output_folder(
            experiment_config=config,
            root=BUILDINGS_ROOT,
        )
    )
    settings_path = Path(args.settings) if args.settings else output_folder / "final_settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"Frozen settings not found: {settings_path}")
    frozen = json.loads(settings_path.read_text(encoding="utf-8"))
    if not frozen.get("settings_frozen", False):
        raise RuntimeError("Settings are not frozen. Final test blocked.")

    final_root = output_folder / "final_test"
    final_root.mkdir(parents=True, exist_ok=True)

    receipt_path = final_root / "final_test_receipt.json"
    metrics_path = final_root / "final_metrics.json"

    if receipt_path.exists() and not args.allow_rerun:
        raise RuntimeError(
            "Final test receipt already exists. Refusing accidental rerun. "
            "Use --allow-rerun only for debugging, not as a new untouched result."
        )

    # --------------------------------------------------------
    # SAFE RECOVERY
    # --------------------------------------------------------
    # A previous final-test run may have completed inference,
    # polygon evaluation, and final_metrics.json, then failed only
    # while writing the receipt timestamp. In that case we MUST
    # NOT access the external test area again. We validate the
    # existing metrics and create only the missing receipt.
    # --------------------------------------------------------

    if (
        metrics_path.exists()
        and not receipt_path.exists()
        and not args.allow_rerun
    ):
        existing_metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        required_metric_keys = {
            "experiment_id",
            "iou_percent",
            "precision_percent",
            "recall_percent",
            "f1_percent",
            "ground_truth_count",
            "prediction_count",
        }

        same_experiment = (
            existing_metrics.get("experiment_id")
            == config.get("experiment_id")
        )

        metrics_complete = required_metric_keys.issubset(
            existing_metrics.keys()
        )

        if same_experiment and metrics_complete:
            receipt = {
                "final_test_completed": True,
                "completed_at_utc": utc_now_iso(),
                "experiment_id": config["experiment_id"],
                "model_type": "maskrcnn",
                "settings_path": str(settings_path),
                "metrics_path": str(metrics_path),
                "threshold_search_on_test": False,
                "postprocess_search_on_test": False,
                "allow_rerun_used": False,
                "recovered_from_existing_metrics": True,
                "external_test_rerun": False,
            }

            save_json(receipt, receipt_path)

            print("\n========== MASK R-CNN FINAL TEST RECOVERED ==========")
            print("Existing final metrics were found and validated.")
            print("External test inference was NOT rerun.")
            print(
                f"Strict Polygon IoU: "
                f"{existing_metrics['iou_percent']:.2f}%"
            )
            print(
                f"Precision: "
                f"{existing_metrics['precision_percent']:.2f}%"
            )
            print(
                f"Recall: "
                f"{existing_metrics['recall_percent']:.2f}%"
            )
            print(
                f"F1: "
                f"{existing_metrics['f1_percent']:.2f}%"
            )
            print(
                f"GT Count: "
                f"{existing_metrics['ground_truth_count']}"
            )
            print(
                f"Prediction Count: "
                f"{existing_metrics['prediction_count']}"
            )
            print(f"Metrics: {metrics_path}")
            print(f"Receipt: {receipt_path}")
            print("======================================================")
            return

    paths_config = load_paths_config(BUILDINGS_ROOT)
    test_image = get_test_image_path(
        experiment_config=config,
        paths_config=paths_config,
    )
    gt_source = get_ground_truth_path(
        experiment_config=config,
        paths_config=paths_config,
    )
    if not arcpy.Exists(test_image):
        raise FileNotFoundError(f"Final test raster not found: {test_image}")
    if not arcpy.Exists(gt_source):
        raise FileNotFoundError(f"Ground truth not found: {gt_source}")

    gdb = final_root / "maskrcnn_final_test.gdb"
    if not arcpy.Exists(str(gdb)):
        arcpy.management.CreateFileGDB(str(final_root), gdb.name)

    checkpoint = Path(frozen["checkpoint"])
    model, checkpoint_payload, device = load_checkpoint_model(
        checkpoint,
        nms_threshold=float(frozen.get("nms_threshold", 0.50)),
        detections_per_image=int(frozen.get("detections_per_image", 300)),
    )

    score_threshold = float(frozen.get("score_threshold", 0.50))
    mask_threshold = float(frozen.get("mask_threshold", 0.50))
    min_area_m2 = float(frozen.get("min_area_m2", 0.0))
    tile_size = int(frozen.get("tile_size", config.get("tile_size", 512)))
    overlap_ratio = 0.25

    print("\n========== MASK R-CNN FINAL TEST ==========")
    print(f"Experiment ID: {config['experiment_id']}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Score Threshold: {score_threshold:.2f} (FROZEN)")
    print(f"Mask Threshold: {mask_threshold:.2f} (FROZEN)")
    print(f"Min Area: {min_area_m2:.2f} m2 (FROZEN)")
    print("Threshold Search on Test: NO")
    print("Post-processing Search on Test: NO")
    print("===========================================\n")

    image = read_arcgis_rgb_raster(test_image)
    mask = run_tiled_inference(
        model,
        image,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        device=device,
    )

    binary_raster = str(gdb / "final_binary_prediction")
    raw_fc = str(gdb / "raw_prediction")
    final_fc = str(gdb / "final_prediction")
    gt_fc = str(gdb / "ground_truth_projected")

    save_binary_mask_raster(mask, test_image, binary_raster)
    raster_to_polygon(binary_raster, raw_fc, gdb)
    filter_min_area(raw_fc, final_fc, min_area_m2)
    project_ground_truth(gt_source, test_image, gt_fc)
    metrics = strict_polygon_metrics(final_fc, gt_fc, gdb)

    payload = {
        "experiment_id": config["experiment_id"],
        "model_type": "maskrcnn",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch") if isinstance(checkpoint_payload, dict) else None,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "min_area_m2": min_area_m2,
        "tile_size": tile_size,
        "overlap_ratio": overlap_ratio,
        "threshold_search_on_test": False,
        "postprocess_search_on_test": False,
        "final_prediction": final_fc,
        **metrics,
    }
    save_json(payload, metrics_path)

    receipt = {
        "final_test_completed": True,
        "completed_at_utc": utc_now_iso(),
        "experiment_id": config["experiment_id"],
        "model_type": "maskrcnn",
        "settings_path": str(settings_path),
        "metrics_path": str(metrics_path),
        "threshold_search_on_test": False,
        "postprocess_search_on_test": False,
        "allow_rerun_used": bool(args.allow_rerun),
    }
    save_json(receipt, receipt_path)

    print("\n========== FINAL RESULT ==========")
    print(f"Strict Polygon IoU: {metrics['iou_percent']:.2f}%")
    print(f"Precision: {metrics['precision_percent']:.2f}%")
    print(f"Recall: {metrics['recall_percent']:.2f}%")
    print(f"F1: {metrics['f1_percent']:.2f}%")
    print(f"GT Count: {metrics['ground_truth_count']}")
    print(f"Prediction Count: {metrics['prediction_count']}")
    print("Test Threshold/Post Search: NO")
    print(f"Metrics: {metrics_path}")
    print("==================================\n")


if __name__ == "__main__":
    main()

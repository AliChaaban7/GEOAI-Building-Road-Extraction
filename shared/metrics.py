"""
metrics.py

Shared GIS evaluation metrics for the thesis project.

Main metric:
- Strict polygon IoU
"""

from pathlib import Path
import csv
import json


def ensure_same_projection(input_fc, target_fc, output_fc):
    """
    Project input_fc to the spatial reference of target_fc if needed.

    Args:
        input_fc: feature class to project
        target_fc: feature class with desired spatial reference
        output_fc: output projected feature class

    Returns:
        path to projected or copied feature class
    """
    import arcpy

    input_desc = arcpy.Describe(input_fc)
    target_desc = arcpy.Describe(target_fc)

    input_sr = input_desc.spatialReference
    target_sr = target_desc.spatialReference

    if arcpy.Exists(output_fc):
        arcpy.management.Delete(output_fc)

    if input_sr.name == target_sr.name:
        arcpy.management.CopyFeatures(input_fc, output_fc)
    else:
        arcpy.management.Project(
            in_dataset=input_fc,
            out_dataset=output_fc,
            out_coor_system=target_sr
        )

    return output_fc


def repair_geometry_safe(feature_class):
    """
    Repair geometry safely.
    """
    import arcpy

    if arcpy.Exists(feature_class):
        try:
            arcpy.management.RepairGeometry(feature_class)
        except Exception:
            pass


def calculate_total_area(feature_class):
    """
    Calculate total polygon area.
    """
    import arcpy

    if not arcpy.Exists(feature_class):
        return 0.0

    count = int(arcpy.management.GetCount(feature_class)[0])

    if count == 0:
        return 0.0

    total_area = 0.0

    with arcpy.da.SearchCursor(feature_class, ["SHAPE@AREA"]) as cursor:
        for row in cursor:
            if row[0] is not None:
                total_area += float(row[0])

    return total_area


def calculate_strict_polygon_iou(pred_fc, gt_fc, output_folder):
    """
    Calculate strict polygon IoU.

    Strict IoU:
        intersection_area / union_area

    union_area:
        prediction_area + ground_truth_area - intersection_area

    Important:
    This function does not dissolve predictions.
    It evaluates the predicted polygons directly.
    """
    import arcpy

    arcpy.env.overwriteOutput = True

    output_folder = Path(output_folder)
    eval_gdb = output_folder / "evaluation.gdb"

    if not arcpy.Exists(str(eval_gdb)):
        arcpy.management.CreateFileGDB(
            str(eval_gdb.parent),
            eval_gdb.name
        )

    gt_projected = str(eval_gdb / "gt_projected_to_prediction_sr")
    intersection_fc = str(eval_gdb / "strict_intersection")

    ensure_same_projection(
        input_fc=gt_fc,
        target_fc=pred_fc,
        output_fc=gt_projected
    )

    repair_geometry_safe(pred_fc)
    repair_geometry_safe(gt_projected)

    if arcpy.Exists(intersection_fc):
        arcpy.management.Delete(intersection_fc)

    pred_count = int(arcpy.management.GetCount(pred_fc)[0])
    gt_count = int(arcpy.management.GetCount(gt_projected)[0])

    pred_area = calculate_total_area(pred_fc)
    gt_area = calculate_total_area(gt_projected)

    if pred_count > 0 and gt_count > 0:
        arcpy.analysis.Intersect(
            [pred_fc, gt_projected],
            intersection_fc
        )
        intersection_area = calculate_total_area(intersection_fc)
        intersection_count = int(arcpy.management.GetCount(intersection_fc)[0])
    else:
        intersection_area = 0.0
        intersection_count = 0

    union_area = pred_area + gt_area - intersection_area

    if union_area <= 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area

    precision = intersection_area / pred_area if pred_area > 0 else 0.0
    recall = intersection_area / gt_area if gt_area > 0 else 0.0

    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    metrics = {
        "pred_fc": str(pred_fc),
        "gt_fc_original": str(gt_fc),
        "gt_fc_projected": str(gt_projected),
        "intersection_fc": str(intersection_fc),

        "pred_count": pred_count,
        "gt_count": gt_count,
        "intersection_count": intersection_count,

        "pred_area": pred_area,
        "gt_area": gt_area,
        "intersection_area": intersection_area,
        "union_area": union_area,

        "iou": iou,
        "iou_percent": iou * 100,

        "precision": precision,
        "precision_percent": precision * 100,

        "recall": recall,
        "recall_percent": recall * 100,

        "f1": f1,
        "f1_percent": f1 * 100
    }

    return metrics


def save_metrics_csv(metrics, csv_path):
    """
    Save metrics dictionary to CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)


def save_metrics_json(metrics, json_path):
    """
    Save metrics dictionary to JSON.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
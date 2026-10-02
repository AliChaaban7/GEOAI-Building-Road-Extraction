"""
GIS evaluation for Approach 2 - Road Extraction.

Purpose
-------
Evaluate predicted Road-surface polygons against held-out Ground Truth
using GIS geometry rather than chip-level pixel masks.

Primary metrics
---------------
IoU:
    Intersection Area / Union Area

Precision:
    Intersection Area / Predicted Area

Recall:
    Intersection Area / Ground-Truth Area

F1:
    2 * Precision * Recall
    ----------------------
       Precision + Recall

Important
---------
- Ground Truth is never used to tune model parameters here.
- Threshold and post-processing must already be frozen.
- Original prediction and Ground Truth feature classes are never modified.
- Evaluation can optionally be restricted to the footprint of the final
  test raster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# ArcPy
# ---------------------------------------------------------------------

def _get_arcpy():
    """
    Import ArcPy only when GIS evaluation is requested.
    """

    try:
        import arcpy

    except ImportError as exc:
        raise RuntimeError(
            "GIS evaluation requires the ArcGIS Pro Python environment.\n"
            "Run this module with:\n\n"
            "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\"
            "arcgispro-py3\\python.exe"
        ) from exc

    return arcpy


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------

def _require_dataset(
    dataset: str,
    description: str,
) -> str:
    """
    Verify that an ArcGIS dataset exists.
    """

    arcpy = _get_arcpy()

    if not dataset:
        raise ValueError(
            f"{description} path cannot be empty."
        )

    if not arcpy.Exists(
        dataset
    ):
        raise FileNotFoundError(
            f"{description} does not exist:\n"
            f"{dataset}"
        )

    return dataset


def _require_polygon_feature_class(
    feature_class: str,
    description: str,
) -> str:
    """
    Ensure an ArcGIS feature class exists and contains polygon geometry.
    """

    arcpy = _get_arcpy()

    _require_dataset(
        feature_class,
        description,
    )

    description_object = arcpy.Describe(
        feature_class
    )

    shape_type = str(
        description_object.shapeType
    ).lower()

    if shape_type != "polygon":
        raise ValueError(
            f"{description} must contain Polygon geometry.\n"
            f"Received: {description_object.shapeType}\n"
            f"Dataset: {feature_class}"
        )

    return feature_class


# ---------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------

def _delete_if_exists(
    dataset: str,
) -> None:
    """
    Delete an output dataset when it already exists.
    """

    arcpy = _get_arcpy()

    if arcpy.Exists(
        dataset
    ):
        arcpy.management.Delete(
            dataset
        )


def _output_path(
    geodatabase: str,
    name: str,
) -> str:
    """
    Create a File Geodatabase dataset path.
    """

    return str(
        Path(
            geodatabase
        )
        / name
    )


# ---------------------------------------------------------------------
# Spatial reference
# ---------------------------------------------------------------------

def _spatial_reference(
    dataset: str,
):
    """
    Return an ArcGIS SpatialReference object.
    """

    arcpy = _get_arcpy()

    return arcpy.Describe(
        dataset
    ).spatialReference


def _same_spatial_reference(
    first,
    second,
) -> bool:
    """
    Compare two ArcGIS spatial references.
    """

    first_code = getattr(
        first,
        "factoryCode",
        0,
    )

    second_code = getattr(
        second,
        "factoryCode",
        0,
    )

    if (
        first_code
        and second_code
    ):
        return (
            first_code
            == second_code
        )

    return (
        first.name
        == second.name
    )


def _project_if_needed(
    feature_class: str,
    target_spatial_reference,
    output_feature_class: str,
) -> str:
    """
    Project a feature class only when necessary.

    Original source data is never modified.
    """

    arcpy = _get_arcpy()

    source_sr = _spatial_reference(
        feature_class
    )

    if _same_spatial_reference(
        source_sr,
        target_spatial_reference,
    ):
        arcpy.management.CopyFeatures(
            feature_class,
            output_feature_class,
        )

    else:
        arcpy.management.Project(
            feature_class,
            output_feature_class,
            target_spatial_reference,
        )

    return output_feature_class


# ---------------------------------------------------------------------
# Raster footprint
# ---------------------------------------------------------------------

def create_raster_extent_polygon(
    raster: str,
    output_feature_class: str,
) -> str:
    """
    Create a polygon corresponding to the rectangular raster extent.

    This is useful for restricting Ground Truth and predictions to the
    exact held-out raster evaluation area.
    """

    arcpy = _get_arcpy()

    _require_dataset(
        raster,
        "Evaluation raster",
    )

    _delete_if_exists(
        output_feature_class
    )

    raster_description = arcpy.Describe(
        raster
    )

    extent = raster_description.extent

    spatial_reference = (
        raster_description.spatialReference
    )

    lower_left = arcpy.Point(
        extent.XMin,
        extent.YMin,
    )

    lower_right = arcpy.Point(
        extent.XMax,
        extent.YMin,
    )

    upper_right = arcpy.Point(
        extent.XMax,
        extent.YMax,
    )

    upper_left = arcpy.Point(
        extent.XMin,
        extent.YMax,
    )

    polygon = arcpy.Polygon(
        arcpy.Array(
            [
                lower_left,
                lower_right,
                upper_right,
                upper_left,
                lower_left,
            ]
        ),
        spatial_reference,
    )

    output_folder = str(
        Path(
            output_feature_class
        ).parent
    )

    output_name = Path(
        output_feature_class
    ).name

    arcpy.management.CreateFeatureclass(
        out_path=output_folder,
        out_name=output_name,
        geometry_type="POLYGON",
        spatial_reference=(
            spatial_reference
        ),
    )

    with arcpy.da.InsertCursor(
        output_feature_class,
        [
            "SHAPE@",
        ],
    ) as cursor:

        cursor.insertRow(
            [
                polygon,
            ]
        )

    return output_feature_class


# ---------------------------------------------------------------------
# Geometry preparation
# ---------------------------------------------------------------------

def _clip_feature_class(
    input_feature_class: str,
    clip_feature_class: str,
    output_feature_class: str,
) -> str:
    """
    Clip a polygon feature class to the evaluation area.
    """

    arcpy = _get_arcpy()

    _delete_if_exists(
        output_feature_class
    )

    try:
        arcpy.analysis.PairwiseClip(
            input_feature_class,
            clip_feature_class,
            output_feature_class,
        )

    except Exception:
        arcpy.analysis.Clip(
            input_feature_class,
            clip_feature_class,
            output_feature_class,
        )

    return output_feature_class


def _dissolve(
    input_feature_class: str,
    output_feature_class: str,
) -> str:
    """
    Dissolve polygons before area calculation.

    This prevents overlapping polygons inside the same feature class
    from being counted more than once.
    """

    arcpy = _get_arcpy()

    _delete_if_exists(
        output_feature_class
    )

    arcpy.management.Dissolve(
        in_features=input_feature_class,
        out_feature_class=(
            output_feature_class
        ),
        multi_part="MULTI_PART",
        unsplit_lines="DISSOLVE_LINES",
    )

    return output_feature_class


# ---------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------

def feature_area_m2(
    feature_class: str,
) -> float:
    """
    Calculate total non-overlapping polygon area in square meters.

    The feature class should normally already be dissolved.

    Projected CRS:
        planar area

    Geographic CRS:
        geodesic area
    """

    arcpy = _get_arcpy()

    _require_polygon_feature_class(
        feature_class,
        "Area feature class",
    )

    spatial_reference = _spatial_reference(
        feature_class
    )

    sr_type = str(
        spatial_reference.type
    ).lower()

    method = (
        "PLANAR"
        if sr_type == "projected"
        else "GEODESIC"
    )

    total_area = 0.0

    with arcpy.da.SearchCursor(
        feature_class,
        [
            "SHAPE@",
        ],
    ) as cursor:

        for row in cursor:

            geometry = row[
                0
            ]

            if (
                geometry is None
                or geometry.isEmpty
            ):
                continue

            total_area += float(
                geometry.getArea(
                    method,
                    "SQUAREMETERS",
                )
            )

    return float(
        total_area
    )


# ---------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------

def _intersect(
    prediction_feature_class: str,
    ground_truth_feature_class: str,
    output_feature_class: str,
) -> str:
    """
    Generate prediction / Ground-Truth polygon intersection.
    """

    arcpy = _get_arcpy()

    _delete_if_exists(
        output_feature_class
    )

    try:
        arcpy.analysis.PairwiseIntersect(
            [
                prediction_feature_class,
                ground_truth_feature_class,
            ],
            output_feature_class,
        )

    except Exception:
        arcpy.analysis.Intersect(
            [
                prediction_feature_class,
                ground_truth_feature_class,
            ],
            output_feature_class,
            join_attributes="ONLY_FID",
            output_type="INPUT",
        )

    return output_feature_class


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_area_metrics(
    prediction_area_m2: float,
    ground_truth_area_m2: float,
    intersection_area_m2: float,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculate GIS Road evaluation metrics from areas.
    """

    prediction_area_m2 = float(
        prediction_area_m2
    )

    ground_truth_area_m2 = float(
        ground_truth_area_m2
    )

    intersection_area_m2 = float(
        intersection_area_m2
    )

    union_area_m2 = (
        prediction_area_m2
        + ground_truth_area_m2
        - intersection_area_m2
    )

    if union_area_m2 > 0:

        iou = (
            intersection_area_m2
            / (
                union_area_m2
                + epsilon
            )
        )

    else:

        iou = 1.0

    if prediction_area_m2 > 0:

        precision = (
            intersection_area_m2
            / (
                prediction_area_m2
                + epsilon
            )
        )

    else:

        precision = (
            1.0
            if ground_truth_area_m2 <= 0
            else 0.0
        )

    if ground_truth_area_m2 > 0:

        recall = (
            intersection_area_m2
            / (
                ground_truth_area_m2
                + epsilon
            )
        )

    else:

        recall = (
            1.0
            if prediction_area_m2 <= 0
            else 0.0
        )

    f1_denominator = (
        precision
        + recall
    )

    if f1_denominator > 0:

        f1 = (
            2.0
            * precision
            * recall
            / (
                f1_denominator
                + epsilon
            )
        )

    else:

        f1 = 0.0

    return {
        "prediction_area_m2":
            prediction_area_m2,

        "ground_truth_area_m2":
            ground_truth_area_m2,

        "intersection_area_m2":
            intersection_area_m2,

        "union_area_m2":
            float(
                union_area_m2
            ),

        "iou":
            float(
                iou
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "iou_percent":
            float(
                iou * 100.0
            ),

        "precision_percent":
            float(
                precision * 100.0
            ),

        "recall_percent":
            float(
                recall * 100.0
            ),

        "f1_percent":
            float(
                f1 * 100.0
            ),
    }


# ---------------------------------------------------------------------
# Main GIS evaluation
# ---------------------------------------------------------------------

def evaluate_polygon_prediction(
    prediction_feature_class: str,
    ground_truth_feature_class: str,
    output_geodatabase: str,
    evaluation_name: str,
    evaluation_raster: Optional[
        str
    ] = None,
    save_json_path: Optional[
        str | Path
    ] = None,
) -> Dict[str, Any]:
    """
    Evaluate Road prediction polygons against Road Ground Truth.

    Parameters
    ----------
    prediction_feature_class:
        Predicted Road-surface polygons.

    ground_truth_feature_class:
        Held-out Road Ground Truth.

    output_geodatabase:
        Workspace for intermediate and evaluation feature classes.

    evaluation_name:
        Prefix for generated outputs.

        Example:
            unet_satellite_final

    evaluation_raster:
        Optional held-out raster.

        When supplied, prediction and Ground Truth are clipped to the
        exact raster extent before evaluation.

        Example:
            SatelliteTesingZone

    save_json_path:
        Optional path for the final metric JSON.

    Returns
    -------
    dict
        IoU, Precision, Recall, F1 and area statistics.
    """

    arcpy = _get_arcpy()

    arcpy.env.overwriteOutput = True

    _require_polygon_feature_class(
        prediction_feature_class,
        "Prediction feature class",
    )

    _require_polygon_feature_class(
        ground_truth_feature_class,
        "Ground Truth feature class",
    )

    _require_dataset(
        output_geodatabase,
        "Output geodatabase",
    )

    evaluation_name = str(
        evaluation_name
    ).strip()

    if not evaluation_name:
        raise ValueError(
            "evaluation_name cannot be empty."
        )

    # --------------------------------------------------------------
    # Output paths
    # --------------------------------------------------------------

    prediction_projected = _output_path(
        output_geodatabase,
        f"{evaluation_name}_pred_projected",
    )

    ground_truth_projected = _output_path(
        output_geodatabase,
        f"{evaluation_name}_gt_projected",
    )

    raster_extent_fc = _output_path(
        output_geodatabase,
        f"{evaluation_name}_extent",
    )

    prediction_clipped = _output_path(
        output_geodatabase,
        f"{evaluation_name}_pred_clip",
    )

    ground_truth_clipped = _output_path(
        output_geodatabase,
        f"{evaluation_name}_gt_clip",
    )

    prediction_dissolved = _output_path(
        output_geodatabase,
        f"{evaluation_name}_pred_diss",
    )

    ground_truth_dissolved = _output_path(
        output_geodatabase,
        f"{evaluation_name}_gt_diss",
    )

    intersection_fc = _output_path(
        output_geodatabase,
        f"{evaluation_name}_intersection",
    )

    # --------------------------------------------------------------
    # Evaluation spatial reference
    #
    # Prefer raster CRS when a test raster is supplied.
    # Otherwise use Ground Truth CRS.
    # --------------------------------------------------------------

    if evaluation_raster is not None:

        _require_dataset(
            evaluation_raster,
            "Evaluation raster",
        )

        target_sr = _spatial_reference(
            evaluation_raster
        )

    else:

        target_sr = _spatial_reference(
            ground_truth_feature_class
        )

    # --------------------------------------------------------------
    # Project copies
    # --------------------------------------------------------------

    for output in (
        prediction_projected,
        ground_truth_projected,
    ):
        _delete_if_exists(
            output
        )

    _project_if_needed(
        feature_class=(
            prediction_feature_class
        ),
        target_spatial_reference=(
            target_sr
        ),
        output_feature_class=(
            prediction_projected
        ),
    )

    _project_if_needed(
        feature_class=(
            ground_truth_feature_class
        ),
        target_spatial_reference=(
            target_sr
        ),
        output_feature_class=(
            ground_truth_projected
        ),
    )

    # --------------------------------------------------------------
    # Restrict to test raster
    # --------------------------------------------------------------

    if evaluation_raster is not None:

        create_raster_extent_polygon(
            raster=evaluation_raster,
            output_feature_class=(
                raster_extent_fc
            ),
        )

        _clip_feature_class(
            input_feature_class=(
                prediction_projected
            ),
            clip_feature_class=(
                raster_extent_fc
            ),
            output_feature_class=(
                prediction_clipped
            ),
        )

        _clip_feature_class(
            input_feature_class=(
                ground_truth_projected
            ),
            clip_feature_class=(
                raster_extent_fc
            ),
            output_feature_class=(
                ground_truth_clipped
            ),
        )

        prediction_for_eval = (
            prediction_clipped
        )

        ground_truth_for_eval = (
            ground_truth_clipped
        )

    else:

        prediction_for_eval = (
            prediction_projected
        )

        ground_truth_for_eval = (
            ground_truth_projected
        )

    # --------------------------------------------------------------
    # Dissolve
    # --------------------------------------------------------------

    _dissolve(
        input_feature_class=(
            prediction_for_eval
        ),
        output_feature_class=(
            prediction_dissolved
        ),
    )

    _dissolve(
        input_feature_class=(
            ground_truth_for_eval
        ),
        output_feature_class=(
            ground_truth_dissolved
        ),
    )

    # --------------------------------------------------------------
    # Intersection
    # --------------------------------------------------------------

    _intersect(
        prediction_feature_class=(
            prediction_dissolved
        ),
        ground_truth_feature_class=(
            ground_truth_dissolved
        ),
        output_feature_class=(
            intersection_fc
        ),
    )

    # --------------------------------------------------------------
    # Areas
    # --------------------------------------------------------------

    prediction_area = feature_area_m2(
        prediction_dissolved
    )

    ground_truth_area = feature_area_m2(
        ground_truth_dissolved
    )

    intersection_area = (
        feature_area_m2(
            intersection_fc
        )
        if int(
            arcpy.management.GetCount(
                intersection_fc
            )[0]
        ) > 0
        else 0.0
    )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = calculate_area_metrics(
        prediction_area_m2=(
            prediction_area
        ),
        ground_truth_area_m2=(
            ground_truth_area
        ),
        intersection_area_m2=(
            intersection_area
        ),
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    result = {
        "evaluation_name":
            evaluation_name,

        "prediction_feature_class":
            prediction_feature_class,

        "ground_truth_feature_class":
            ground_truth_feature_class,

        "evaluation_raster":
            evaluation_raster,

        "evaluation_type":
            "strict_polygon_area",

        "parameter_search_performed":
            False,

        "metrics":
            metrics,

        "outputs": {
            "prediction_dissolved":
                prediction_dissolved,

            "ground_truth_dissolved":
                ground_truth_dissolved,

            "intersection":
                intersection_fc,

            "evaluation_extent":
                (
                    raster_extent_fc
                    if evaluation_raster
                    is not None
                    else None
                ),
        },
    }

    # --------------------------------------------------------------
    # JSON result
    # --------------------------------------------------------------

    if save_json_path is not None:

        save_json_path = Path(
            save_json_path
        )

        save_json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with save_json_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
            )

        result[
            "result_json"
        ] = str(
            save_json_path
        )

    return result
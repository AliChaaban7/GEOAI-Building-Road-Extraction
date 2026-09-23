"""
postprocess.py

Reusable GIS post-processing utilities for the Building Extraction module.

Current supported post-processing:
    - minimum polygon-area filtering

Designed for:
    - U-Net
    - DeepLabV3
    - Mask R-CNN polygon outputs

Key design rules:
    - ArcGIS-safe feature-class names only
    - candidates are created inside a File Geodatabase
    - temporary candidates are deleted after evaluation
    - only the best post-processed prediction is kept
    - strict polygon IoU is supplied by the caller
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import uuid

import arcpy
import pandas as pd


# ============================================================
# ARCPY ENVIRONMENT
# ============================================================

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False


# ============================================================
# SAFE NAMING
# ============================================================

def safe_number_token(value) -> str:
    """
    Convert numeric values to ArcGIS-safe name tokens.

    Examples:
        0.0  -> 0
        5.0  -> 5
        12.5 -> 12p5
        -2.5 -> m2p5
    """

    number = float(value)

    if math.isclose(
        number,
        round(number),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        text = str(
            int(
                round(number)
            )
        )

    else:
        text = (
            f"{number:.6f}"
            .rstrip("0")
            .rstrip(".")
        )

    text = text.replace(
        "-",
        "m",
    )

    text = text.replace(
        ".",
        "p",
    )

    text = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        text,
    )

    return text


def safe_arcgis_name(
    name: str,
    max_length: int = 120,
) -> str:
    """
    Sanitize an ArcGIS geodatabase object name.
    """

    name = str(
        name
    ).strip()

    name = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    name = name.strip(
        "_"
    )

    if not name:
        name = "output"

    if name[0].isdigit():
        name = (
            f"fc_{name}"
        )

    return name[
        :max_length
    ]


# ============================================================
# BASIC HELPERS
# ============================================================

def delete_if_exists(
    dataset
) -> None:

    dataset = str(
        dataset
    )

    if arcpy.Exists(
        dataset
    ):
        arcpy.management.Delete(
            dataset
        )


def ensure_file_gdb(
    gdb_path
) -> Path:

    gdb_path = Path(
        gdb_path
    )

    gdb_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    return gdb_path


def count_features(
    feature_class
) -> int:

    result = (
        arcpy.management.GetCount(
            str(
                feature_class
            )
        )
    )

    return int(
        result[0]
    )


def validate_polygon_feature_class(
    feature_class,
    label="Feature class",
) -> None:

    feature_class = str(
        feature_class
    )

    if not arcpy.Exists(
        feature_class
    ):

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{feature_class}"
        )

    description = (
        arcpy.Describe(
            feature_class
        )
    )

    shape_type = str(
        getattr(
            description,
            "shapeType",
            "",
        )
    ).lower()

    if shape_type != "polygon":

        raise ValueError(
            f"{label} must be polygon geometry. "
            f"Received: {shape_type}"
        )


# ============================================================
# AREA FIELD
# ============================================================

def get_area_field(
    feature_class
) -> str:

    description = (
        arcpy.Describe(
            str(
                feature_class
            )
        )
    )

    area_field = getattr(
        description,
        "areaFieldName",
        None,
    )

    if area_field:

        return str(
            area_field
        )

    fields = (
        arcpy.ListFields(
            str(
                feature_class
            )
        )
    )

    common_names = {
        "shape_area",
        "shape__area",
    }

    for field in fields:

        if (
            field.name.lower()
            in common_names
        ):

            return field.name

    raise RuntimeError(
        "Could not determine polygon "
        "area field for:\n"
        f"{feature_class}"
    )


# ============================================================
# POLYGON AREA FILTER
# ============================================================

def copy_polygons_by_area(
    input_fc,
    output_fc,
    min_area_m2,
):
    """
    Copy polygons whose geometry area is
    >= min_area_m2.
    """

    input_fc = str(
        input_fc
    )

    output_fc = str(
        output_fc
    )

    min_area_m2 = float(
        min_area_m2
    )

    validate_polygon_feature_class(
        input_fc,
        "Input prediction feature class",
    )

    delete_if_exists(
        output_fc
    )

    # ========================================================
    # NO FILTER
    # ========================================================

    if min_area_m2 <= 0.0:

        arcpy.management.CopyFeatures(
            input_fc,
            output_fc,
        )

        return (
            output_fc,
            count_features(
                output_fc
            ),
        )

    # ========================================================
    # SAFE TEMPORARY LAYER NAME
    # ========================================================

    token = safe_number_token(
        min_area_m2
    )

    layer_name = safe_arcgis_name(
        f"pp_area_{token}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    area_field = get_area_field(
        input_fc
    )

    area_field_sql = (
        arcpy.AddFieldDelimiters(
            input_fc,
            area_field,
        )
    )

    where_clause = (
        f"{area_field_sql} >= "
        f"{min_area_m2:.12f}"
    )

    # ========================================================
    # SELECT
    # ========================================================

    arcpy.management.MakeFeatureLayer(
        input_fc,
        layer_name,
    )

    try:

        arcpy.management.SelectLayerByAttribute(
            layer_name,
            "NEW_SELECTION",
            where_clause,
        )

        arcpy.management.CopyFeatures(
            layer_name,
            output_fc,
        )

    finally:

        try:
            arcpy.management.Delete(
                layer_name
            )

        except Exception:
            pass

    polygon_count = (
        count_features(
            output_fc
        )
    )

    return (
        output_fc,
        polygon_count,
    )


# ============================================================
# AREA FILTER SEARCH
# ============================================================

def run_area_filter_postprocessing(
    raw_prediction_fc,
    gt_fc,
    output_folder,
    area_thresholds_m2,
    calculate_iou_function,
):
    """
    Search minimum polygon-area thresholds
    using strict polygon IoU.
    """

    raw_prediction_fc = str(
        raw_prediction_fc
    )

    gt_fc = str(
        gt_fc
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_polygon_feature_class(
        raw_prediction_fc,
        "Raw prediction feature class",
    )

    validate_polygon_feature_class(
        gt_fc,
        "Ground-truth feature class",
    )

    if calculate_iou_function is None:

        raise ValueError(
            "calculate_iou_function "
            "is required."
        )

    # ========================================================
    # AREA THRESHOLDS
    # ========================================================

    thresholds = sorted(
        {
            float(
                value
            )

            for value
            in area_thresholds_m2
        }
    )

    if not thresholds:

        raise ValueError(
            "area_thresholds_m2 is empty."
        )

    if any(
        threshold < 0

        for threshold
        in thresholds
    ):

        raise ValueError(
            "Area thresholds must be >= 0."
        )

    # ========================================================
    # GEODATABASES
    # ========================================================

    candidate_gdb = (
        ensure_file_gdb(
            output_folder
            / "postprocess_candidates.gdb"
        )
    )

    final_gdb = (
        ensure_file_gdb(
            output_folder
            / "final_postprocessed.gdb"
        )
    )

    final_postprocessed_fc = str(
        final_gdb
        / "final_postprocessed"
    )

    delete_if_exists(
        final_postprocessed_fc
    )

    # ========================================================
    # SEARCH
    # ========================================================

    rows = []

    best_row = None

    best_threshold = None

    print()

    print(
        "========== AREA FILTER SEARCH =========="
    )

    for min_area in thresholds:

        token = safe_number_token(
            min_area
        )

        candidate_name = (
            safe_arcgis_name(
                f"candidate_area_ge_{token}"
            )
        )

        candidate_fc = str(
            candidate_gdb
            / candidate_name
        )

        delete_if_exists(
            candidate_fc
        )

        try:

            _, polygon_count = (
                copy_polygons_by_area(
                    input_fc=
                        raw_prediction_fc,

                    output_fc=
                        candidate_fc,

                    min_area_m2=
                        min_area,
                )
            )

            metrics = (
                calculate_iou_function(
                    pred_fc=
                        candidate_fc,

                    gt_fc=
                        gt_fc,

                    output_folder=
                        output_folder,
                )
            )

            row = {

                "method":
                    (
                        "none"

                        if min_area <= 0

                        else (
                            f"area_ge_"
                            f"{token}_m2"
                        )
                    ),

                "min_area_m2":
                    float(
                        min_area
                    ),

                "polygon_count":
                    int(
                        polygon_count
                    ),

                "iou":
                    float(
                        metrics.get(
                            "iou",

                            metrics[
                                "iou_percent"
                            ]
                            / 100.0,
                        )
                    ),

                "iou_percent":
                    float(
                        metrics[
                            "iou_percent"
                        ]
                    ),

                "precision":
                    float(
                        metrics.get(
                            "precision",

                            metrics[
                                "precision_percent"
                            ]
                            / 100.0,
                        )
                    ),

                "precision_percent":
                    float(
                        metrics[
                            "precision_percent"
                        ]
                    ),

                "recall":
                    float(
                        metrics.get(
                            "recall",

                            metrics[
                                "recall_percent"
                            ]
                            / 100.0,
                        )
                    ),

                "recall_percent":
                    float(
                        metrics[
                            "recall_percent"
                        ]
                    ),

                "f1":
                    float(
                        metrics.get(
                            "f1",

                            metrics[
                                "f1_percent"
                            ]
                            / 100.0,
                        )
                    ),

                "f1_percent":
                    float(
                        metrics[
                            "f1_percent"
                        ]
                    ),

                "pred_count":
                    int(
                        metrics[
                            "pred_count"
                        ]
                    ),

                "gt_count":
                    int(
                        metrics[
                            "gt_count"
                        ]
                    ),

                "intersection_count":
                    int(
                        metrics[
                            "intersection_count"
                        ]
                    ),

                "pred_area":
                    float(
                        metrics[
                            "pred_area"
                        ]
                    ),

                "gt_area":
                    float(
                        metrics[
                            "gt_area"
                        ]
                    ),

                "intersection_area":
                    float(
                        metrics[
                            "intersection_area"
                        ]
                    ),

                "union_area":
                    float(
                        metrics[
                            "union_area"
                        ]
                    ),
            }

            rows.append(
                row
            )

            print(
                f"Area >= {min_area:.1f} m²"
                f" | IoU: "
                f"{row['iou_percent']:.2f}%"
                f" | P: "
                f"{row['precision_percent']:.2f}%"
                f" | R: "
                f"{row['recall_percent']:.2f}%"
                f" | F1: "
                f"{row['f1_percent']:.2f}%"
                f" | Count: "
                f"{row['pred_count']}"
            )

            # =================================================
            # BEST CANDIDATE
            # =================================================

            is_better = False

            if best_row is None:

                is_better = True

            elif (
                row[
                    "iou_percent"
                ]
                >
                best_row[
                    "iou_percent"
                ]
            ):

                is_better = True

            elif math.isclose(
                row[
                    "iou_percent"
                ],

                best_row[
                    "iou_percent"
                ],

                rel_tol=0.0,
                abs_tol=1e-12,
            ):

                if (
                    row[
                        "f1_percent"
                    ]
                    >
                    best_row[
                        "f1_percent"
                    ]
                ):

                    is_better = True

                elif math.isclose(
                    row[
                        "f1_percent"
                    ],

                    best_row[
                        "f1_percent"
                    ],

                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):

                    if (
                        row[
                            "min_area_m2"
                        ]
                        <
                        best_row[
                            "min_area_m2"
                        ]
                    ):

                        is_better = True

            if is_better:

                best_row = dict(
                    row
                )

                best_threshold = float(
                    min_area
                )

        finally:

            # Delete temporary candidate.
            delete_if_exists(
                candidate_fc
            )

    print(
        "========================================"
    )

    print()

    if best_row is None:

        raise RuntimeError(
            "No post-processing candidate "
            "was successfully evaluated."
        )

    # ========================================================
    # CREATE ONLY THE FINAL WINNING FEATURE CLASS
    # ========================================================

    copy_polygons_by_area(
        input_fc=
            raw_prediction_fc,

        output_fc=
            final_postprocessed_fc,

        min_area_m2=
            best_threshold,
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_metrics = (
        calculate_iou_function(
            pred_fc=
                final_postprocessed_fc,

            gt_fc=
                gt_fc,

            output_folder=
                output_folder,
        )
    )

    best_row.update(
        {

            "method":
                (
                    "none"

                    if best_threshold <= 0

                    else (
                        "area_ge_"
                        f"{safe_number_token(best_threshold)}"
                        "_m2"
                    )
                ),

            "min_area_m2":
                float(
                    best_threshold
                ),

            "final_postprocessed_fc":
                str(
                    final_postprocessed_fc
                ),

            "iou":
                float(
                    final_metrics.get(
                        "iou",

                        final_metrics[
                            "iou_percent"
                        ]
                        / 100.0,
                    )
                ),

            "iou_percent":
                float(
                    final_metrics[
                        "iou_percent"
                    ]
                ),

            "precision":
                float(
                    final_metrics.get(
                        "precision",

                        final_metrics[
                            "precision_percent"
                        ]
                        / 100.0,
                    )
                ),

            "precision_percent":
                float(
                    final_metrics[
                        "precision_percent"
                    ]
                ),

            "recall":
                float(
                    final_metrics.get(
                        "recall",

                        final_metrics[
                            "recall_percent"
                        ]
                        / 100.0,
                    )
                ),

            "recall_percent":
                float(
                    final_metrics[
                        "recall_percent"
                    ]
                ),

            "f1":
                float(
                    final_metrics.get(
                        "f1",

                        final_metrics[
                            "f1_percent"
                        ]
                        / 100.0,
                    )
                ),

            "f1_percent":
                float(
                    final_metrics[
                        "f1_percent"
                    ]
                ),

            "pred_count":
                int(
                    final_metrics[
                        "pred_count"
                    ]
                ),

            "gt_count":
                int(
                    final_metrics[
                        "gt_count"
                    ]
                ),

            "intersection_count":
                int(
                    final_metrics[
                        "intersection_count"
                    ]
                ),

            "pred_area":
                float(
                    final_metrics[
                        "pred_area"
                    ]
                ),

            "gt_area":
                float(
                    final_metrics[
                        "gt_area"
                    ]
                ),

            "intersection_area":
                float(
                    final_metrics[
                        "intersection_area"
                    ]
                ),

            "union_area":
                float(
                    final_metrics[
                        "union_area"
                    ]
                ),
        }
    )

    # ========================================================
    # CLEAN CANDIDATE GDB
    # ========================================================

    try:

        arcpy.ClearWorkspaceCache_management()

        if arcpy.Exists(
            str(
                candidate_gdb
            )
        ):

            arcpy.management.Delete(
                str(
                    candidate_gdb
                )
            )

    except Exception:

        # A workspace lock is not a critical failure.
        pass

    return (
        rows,
        best_row,
    )


# ============================================================
# SAVE POST-PROCESS RESULTS
# ============================================================

def save_postprocess_results(
    rows,
    csv_path,
):
    """
    Save compact post-processing search results.
    """

    csv_path = Path(
        csv_path
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        rows
    )

    if not dataframe.empty:

        preferred_columns = [

            "method",

            "min_area_m2",

            "iou_percent",

            "precision_percent",

            "recall_percent",

            "f1_percent",

            "pred_count",

            "gt_count",

            "intersection_count",

            "pred_area",

            "gt_area",

            "intersection_area",

            "union_area",
        ]

        available = [

            column

            for column
            in preferred_columns

            if column
            in dataframe.columns
        ]

        remaining = [

            column

            for column
            in dataframe.columns

            if column
            not in available
        ]

        dataframe = dataframe[
            available
            + remaining
        ]

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    return str(
        csv_path
    )
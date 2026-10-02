"""
Evaluation package for Approach 2 - Road Extraction.

Evaluation levels
-----------------
1. Pixel-level segmentation:
   - IoU
   - Precision
   - Recall
   - F1

2. Road connectivity:
   - clDice
   - Topology Precision
   - Topology Sensitivity

3. GIS polygon evaluation:
   - strict area IoU
   - Precision
   - Recall
   - F1

4. Independent final-test evaluation:
   - uses frozen predictions
   - Ground Truth for evaluation only
"""

from .metrics import (
    BinaryConfusionCounts,
    RunningBinarySegmentationMetrics,
    binary_confusion_counts,
    binary_segmentation_metrics,
    metrics_from_counts,
    metrics_to_percent,
)

from .connectivity_metrics import (
    ClDiceStatistics,
    RunningClDice,
    cldice_from_statistics,
    cldice_score,
    cldice_statistics,
    prepare_binary_prediction,
    prepare_binary_target,
    soft_dilate,
    soft_erode,
    soft_open,
    soft_skeletonize,
)

from .gis_evaluation import (
    calculate_area_metrics,
    create_raster_extent_polygon,
    evaluate_polygon_prediction,
    feature_area_m2,
)


__all__ = [
    "BinaryConfusionCounts",
    "RunningBinarySegmentationMetrics",
    "binary_confusion_counts",
    "binary_segmentation_metrics",
    "metrics_from_counts",
    "metrics_to_percent",

    "ClDiceStatistics",
    "RunningClDice",
    "cldice_from_statistics",
    "cldice_score",
    "cldice_statistics",
    "prepare_binary_prediction",
    "prepare_binary_target",
    "soft_dilate",
    "soft_erode",
    "soft_open",
    "soft_skeletonize",

    "calculate_area_metrics",
    "create_raster_extent_polygon",
    "evaluate_polygon_prediction",
    "feature_area_m2",
]
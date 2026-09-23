"""
test_arcgis_access.py

Test ArcGIS Pro / arcpy access to the selected test raster and ground-truth feature class.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import arcpy

from src.utils import (
    load_experiment_config,
    load_paths_config,
    get_test_image_path,
    get_ground_truth_path
)


def describe_raster(raster_path):
    """
    Print important raster information.
    """
    desc = arcpy.Describe(raster_path)

    raster = arcpy.Raster(raster_path)

    print("\n--- Raster Information ---")
    print(f"Raster Path: {raster_path}")
    print(f"Raster Name: {desc.name}")
    print(f"Width: {raster.width}")
    print(f"Height: {raster.height}")
    print(f"Band Count: {raster.bandCount}")
    print(f"Cell Size X: {raster.meanCellWidth}")
    print(f"Cell Size Y: {raster.meanCellHeight}")

    if hasattr(desc, "spatialReference"):
        print(f"Spatial Reference: {desc.spatialReference.name}")

    print(f"Extent XMin: {raster.extent.XMin}")
    print(f"Extent YMin: {raster.extent.YMin}")
    print(f"Extent XMax: {raster.extent.XMax}")
    print(f"Extent YMax: {raster.extent.YMax}")


def describe_feature_class(feature_class_path):
    """
    Print important feature class information.
    """
    desc = arcpy.Describe(feature_class_path)

    count = int(arcpy.management.GetCount(feature_class_path)[0])

    print("\n--- Ground Truth Feature Class Information ---")
    print(f"Feature Class Path: {feature_class_path}")
    print(f"Feature Class Name: {desc.name}")
    print(f"Shape Type: {desc.shapeType}")
    print(f"Feature Count: {count}")

    if hasattr(desc, "spatialReference"):
        print(f"Spatial Reference: {desc.spatialReference.name}")


def main():
    config_path = ROOT / "config" / "experiment.json"

    experiment_config = load_experiment_config(config_path)
    paths_config = load_paths_config(ROOT)

    test_image_path = get_test_image_path(
        experiment_config=experiment_config,
        paths_config=paths_config
    )

    ground_truth_path = get_ground_truth_path(
        experiment_config=experiment_config,
        paths_config=paths_config
    )

    print("\n========== ARCGIS ACCESS TEST ==========")
    print(f"Experiment ID: {experiment_config['experiment_id']}")
    print(f"Test Image ID: {experiment_config['test_image_id']}")
    print(f"Ground Truth ID: {experiment_config['ground_truth_id']}")
    print("----------------------------------------")

    if not arcpy.Exists(test_image_path):
        raise FileNotFoundError(f"Test image not found by arcpy: {test_image_path}")

    if not arcpy.Exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth not found by arcpy: {ground_truth_path}")

    describe_raster(test_image_path)
    describe_feature_class(ground_truth_path)

    print("\nArcGIS access test completed successfully.")
    print("========================================\n")


if __name__ == "__main__":
    main()
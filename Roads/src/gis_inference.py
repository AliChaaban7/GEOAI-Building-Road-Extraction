"""
Georeferenced raster inference for Approach 2 - Road Extraction.

Native PyTorch models
---------------------
- U-Net
- DeepLabV3+
- SAM-LoRA

All three use the same geospatial inference engine.

Workflow
--------
ArcGIS raster
    ↓
overlapping native-resolution tiles
    ↓
trained model
    ↓
probability mosaicking by overlap averaging
    ↓
frozen threshold
    ↓
optional frozen Road post-processing
    ↓
georeferenced binary raster
    ↓
Road polygons

Scientific safeguards
---------------------
- no Ground Truth is accessed here
- no threshold search is performed here
- no post-processing search is performed here
- source raster is never resized globally
- no silent radiometric min/max stretching
- trained tile size is restored from checkpoint
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.inference import (
    get_restored_tile_size,
    predict_tensor,
    restore_model_from_checkpoint,
)


# =====================================================================
# ARCPY
# =====================================================================

def _get_arcpy():

    try:

        import arcpy

    except Exception as exc:

        raise RuntimeError(
            "GIS inference requires ArcPy and an initialized ArcGIS Pro "
            "license/session."
        ) from exc

    return arcpy


# =====================================================================
# VALIDATION
# =====================================================================

def validate_arcgis_raster(
    raster_path: str | Path,
):
    """
    Validate raster and return ArcPy Raster object.
    """

    arcpy = _get_arcpy()

    raster_path = str(
        raster_path
    )

    if not arcpy.Exists(
        raster_path
    ):

        raise FileNotFoundError(
            "ArcGIS raster not found:\n"
            f"{raster_path}"
        )

    raster = arcpy.Raster(
        raster_path
    )

    width = int(
        raster.width
    )

    height = int(
        raster.height
    )

    band_count = int(
        raster.bandCount
    )

    if width <= 0 or height <= 0:

        raise ValueError(
            "Input raster has invalid dimensions."
        )

    if band_count < 3:

        raise ValueError(
            "Road models require at least 3 raster bands."
        )

    return raster


# =====================================================================
# TILE POSITIONS
# =====================================================================

def _axis_positions(
    length: int,
    tile_size: int,
    stride: int,
):
    """
    Generate positions while guaranteeing coverage of the final edge.
    """

    length = int(
        length
    )

    tile_size = int(
        tile_size
    )

    stride = int(
        stride
    )

    if length <= tile_size:

        return [
            0
        ]

    positions = list(
        range(
            0,
            max(
                length - tile_size + 1,
                1,
            ),
            stride,
        )
    )

    final_position = (
        length
        - tile_size
    )

    if positions[
        -1
    ] != final_position:

        positions.append(
            final_position
        )

    return positions


def generate_tile_positions(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
):
    """
    Generate top-left image-array positions.
    """

    tile_size = int(
        tile_size
    )

    overlap = int(
        overlap
    )

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0."
        )

    if overlap < 0:

        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= tile_size:

        raise ValueError(
            "overlap must be smaller than tile_size."
        )

    stride = (
        tile_size
        - overlap
    )

    y_positions = _axis_positions(
        height,
        tile_size,
        stride,
    )

    x_positions = _axis_positions(
        width,
        tile_size,
        stride,
    )

    return [
        (
            y,
            x,
        )

        for y in y_positions
        for x in x_positions
    ]


# =====================================================================
# ARCGIS WINDOW READ
# =====================================================================

def _read_rgb_window(
    raster,
    top_row: int,
    left_column: int,
    tile_size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    """
    Read one native-resolution raster window using ArcPy.

    Returns
    -------
    image
        H x W x 3 uint8

    real_height
    real_width
    """

    arcpy = _get_arcpy()

    raster_height = int(
        raster.height
    )

    raster_width = int(
        raster.width
    )

    top_row = int(
        top_row
    )

    left_column = int(
        left_column
    )

    tile_size = int(
        tile_size
    )

    real_height = min(
        tile_size,
        raster_height - top_row,
    )

    real_width = min(
        tile_size,
        raster_width - left_column,
    )

    if real_height <= 0 or real_width <= 0:

        raise ValueError(
            "Attempted to read an invalid raster window."
        )

    cell_width = float(
        raster.meanCellWidth
    )

    cell_height = abs(
        float(
            raster.meanCellHeight
        )
    )

    extent = raster.extent

    x_min = (
        float(
            extent.XMin
        )
        + left_column
        * cell_width
    )

    # top_row is measured from raster top.
    y_min = (
        float(
            extent.YMax
        )
        - (
            top_row
            + real_height
        )
        * cell_height
    )

    lower_left = arcpy.Point(
        x_min,
        y_min,
    )

    array = arcpy.RasterToNumPyArray(
        raster,
        lower_left_corner=lower_left,
        ncols=real_width,
        nrows=real_height,
    )

    array = np.asarray(
        array
    )

    # ArcPy multiband raster:
    # [bands, rows, columns]
    if array.ndim == 3:

        if array.shape[
            0
        ] < 3:

            raise ValueError(
                "Raster window contains fewer than 3 bands."
            )

        image = np.transpose(
            array[
                :3,
                :,
                :,
            ],
            (
                1,
                2,
                0,
            ),
        )

    elif array.ndim == 2:

        # Do not silently interpret a single-band image as RGB.
        raise ValueError(
            "RasterToNumPyArray returned a single-band image; "
            "Road models require RGB."
        )

    else:

        raise ValueError(
            "Unexpected ArcPy raster array shape:\n"
            f"{array.shape}"
        )

    image = _strict_rgb_uint8(
        image
    )

    return (
        image,
        real_height,
        real_width,
    )


# =====================================================================
# RADIOMETRY
# =====================================================================

def _strict_rgb_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert valid imagery to uint8 without per-tile contrast stretching.

    Accepted:
    - integer imagery in [0,255]
    - floating imagery in [0,1]
    - floating imagery in [0,255]

    Rejected:
    - imagery outside these ranges

    This avoids silently changing radiometry from tile to tile.
    """

    image = np.asarray(
        image
    )

    if (
        image.ndim != 3
        or image.shape[
            2
        ] != 3
    ):

        raise ValueError(
            "Expected H x W x 3 imagery."
        )

    finite = image[
        np.isfinite(
            image
        )
    ]

    if finite.size == 0:

        raise ValueError(
            "Raster tile contains no finite image values."
        )

    minimum = float(
        finite.min()
    )

    maximum = float(
        finite.max()
    )

    if minimum < 0:

        raise ValueError(
            "Raster contains negative radiometric values."
        )

    if np.issubdtype(
        image.dtype,
        np.floating,
    ):

        if maximum <= 1.0:

            output = (
                image
                * 255.0
            )

        elif maximum <= 255.0:

            output = image

        else:

            raise ValueError(
                "Raster contains floating values above 255.\n"
                "Automatic per-tile stretching is intentionally disabled."
            )

    else:

        if maximum > 255:

            raise ValueError(
                "Raster contains integer values above 255.\n"
                "Automatic per-tile stretching is intentionally disabled. "
                "Use imagery exported consistently with the training data."
            )

        output = image

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.ascontiguousarray(
        np.clip(
            output,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


# =====================================================================
# TILE PADDING
# =====================================================================

def _pad_tile(
    image: np.ndarray,
    tile_size: int,
) -> np.ndarray:

    height, width = image.shape[
        :2
    ]

    if (
        height == tile_size
        and width == tile_size
    ):

        return image

    bottom = (
        tile_size
        - height
    )

    right = (
        tile_size
        - width
    )

    if bottom < 0 or right < 0:

        raise ValueError(
            "Tile exceeds configured tile_size."
        )

    mode = (
        "reflect"
        if height > 1
        and width > 1
        else "edge"
    )

    return np.pad(
        image,
        (
            (
                0,
                bottom,
            ),
            (
                0,
                right,
            ),
            (
                0,
                0,
            ),
        ),
        mode=mode,
    )


# =====================================================================
# POST-PROCESSING
# =====================================================================

def _build_postprocess_config(
    config,
):
    """
    Resolve frozen Road post-processing configuration.

    No searching occurs here.
    """

    if config is None:

        return None

    from src.postprocessing.road_mask import (
        RoadPostprocessingConfig,
    )

    if isinstance(
        config,
        RoadPostprocessingConfig,
    ):

        return config

    if not isinstance(
        config,
        dict,
    ):

        raise TypeError(
            "postprocess_config must be dictionary or "
            "RoadPostprocessingConfig."
        )

    # Allow wrapper objects saved by validation-search/freeze modules.
    for key in (
        "selected_config",
        "best_config",
        "postprocessing",
        "config",
    ):

        nested = config.get(
            key
        )

        if isinstance(
            nested,
            dict,
        ):

            config = nested

            break

    if hasattr(
        RoadPostprocessingConfig,
        "from_dict",
    ):

        try:

            return RoadPostprocessingConfig.from_dict(
                config
            )

        except Exception:

            pass

    accepted = {
        "minimum_component_size":
            config.get(
                "minimum_component_size",
                config.get(
                    "min_component_pixels",
                    0,
                ),
            ),

        "closing_radius":
            config.get(
                "closing_radius",
                config.get(
                    "closing_pixels",
                    0,
                ),
            ),

        "bridge_gap_radius":
            config.get(
                "bridge_gap_radius",
                config.get(
                    "bridge_pixels",
                    0,
                ),
            ),

        "maximum_hole_size":
            config.get(
                "maximum_hole_size",
                config.get(
                    "max_hole_pixels",
                    0,
                ),
            ),
    }

    return RoadPostprocessingConfig(
        **accepted
    )


def apply_frozen_postprocessing(
    binary_mask: np.ndarray,
    postprocess_config,
) -> np.ndarray:
    """
    Apply already-selected validation post-processing.
    """

    resolved = _build_postprocess_config(
        postprocess_config
    )

    if resolved is None:

        return (
            np.asarray(
                binary_mask
            )
            > 0
        ).astype(
            np.uint8
        )

    from src.postprocessing.road_mask import (
        postprocess_road_mask,
    )

    result = postprocess_road_mask(
        mask=(
            np.asarray(
                binary_mask
            )
            > 0
        ).astype(
            np.uint8
        ),

        config=resolved,
    )

    return (
        np.asarray(
            result
        )
        > 0
    ).astype(
        np.uint8
    )


# =====================================================================
# NUMPY -> GEOREFERENCED RASTER
# =====================================================================

def save_georeferenced_array(
    array: np.ndarray,
    source_raster,
    output_path: str | Path,
    pixel_type: Optional[str] = None,
):
    """
    Save array using the exact source grid origin/cell size/projection.
    """

    arcpy = _get_arcpy()

    output_path = str(
        output_path
    )

    output_parent = Path(
        output_path
    ).parent

    output_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lower_left = arcpy.Point(
        float(
            source_raster.extent.XMin
        ),
        float(
            source_raster.extent.YMin
        ),
    )

    raster = arcpy.NumPyArrayToRaster(
        np.asarray(
            array
        ),
        lower_left_corner=lower_left,
        x_cell_size=float(
            source_raster.meanCellWidth
        ),
        y_cell_size=abs(
            float(
                source_raster.meanCellHeight
            )
        ),
    )

    raster.save(
        output_path
    )

    spatial_reference = (
        source_raster.spatialReference
    )

    if spatial_reference is not None:

        try:

            arcpy.management.DefineProjection(
                output_path,
                spatial_reference,
            )

        except Exception:

            pass

    return output_path


# =====================================================================
# RASTER -> ROAD POLYGON
# =====================================================================

def binary_raster_to_road_polygons(
    binary_raster: str | Path,
    output_feature_class: str | Path,
):
    """
    Polygonize Road pixels and keep only raster value == 1.
    """

    arcpy = _get_arcpy()

    binary_raster = str(
        binary_raster
    )

    output_feature_class = str(
        output_feature_class
    )

    output_workspace = str(
        Path(
            output_feature_class
        ).parent
    )

    temp_name = (
        "_tmp_road_all_values"
    )

    temp_fc = os.path.join(
        output_workspace,
        temp_name,
    )

    if arcpy.Exists(
        temp_fc
    ):

        arcpy.management.Delete(
            temp_fc
        )

    if arcpy.Exists(
        output_feature_class
    ):

        arcpy.management.Delete(
            output_feature_class
        )

    arcpy.conversion.RasterToPolygon(
        in_raster=binary_raster,
        out_polygon_features=temp_fc,
        simplify="NO_SIMPLIFY",
        raster_field="Value",
        create_multipart_features="SINGLE_OUTER_PART",
    )

    field_names = {
        field.name.lower():
            field.name

        for field
        in arcpy.ListFields(
            temp_fc
        )
    }

    value_field = None

    for candidate in (
        "gridcode",
        "grid_code",
        "value",
        "rastervalu",
    ):

        if candidate in field_names:

            value_field = field_names[
                candidate
            ]

            break

    if value_field is None:

        arcpy.management.Delete(
            temp_fc
        )

        raise RuntimeError(
            "Could not find raster value field after RasterToPolygon."
        )

    layer_name = (
        "_road_value_selection"
    )

    arcpy.management.MakeFeatureLayer(
        temp_fc,
        layer_name,
    )

    delimiter = arcpy.AddFieldDelimiters(
        temp_fc,
        value_field,
    )

    arcpy.management.SelectLayerByAttribute(
        layer_name,
        "NEW_SELECTION",
        f"{delimiter} = 1",
    )

    arcpy.management.CopyFeatures(
        layer_name,
        output_feature_class,
    )

    arcpy.management.Delete(
        layer_name
    )

    arcpy.management.Delete(
        temp_fc
    )

    return output_feature_class


# =====================================================================
# FULL GEOSPATIAL INFERENCE
# =====================================================================

def run_georeferenced_raster_inference(
    context: Dict[str, Any],
    raster_path: str | Path,
    checkpoint_path: str | Path,
    threshold: float,
    output_folder: str | Path,
    output_geodatabase: Optional[
        str | Path
    ] = None,
    output_name: str = "Road_Prediction",
    postprocess_config=None,
    device: Optional[
        str | torch.device
    ] = None,
    tile_size: Optional[
        int
    ] = None,
    overlap: Optional[
        int
    ] = None,
    save_probability_raster: bool = True,
    polygonize: bool = True,
) -> Dict[str, Any]:
    """
    Full native-resolution GIS inference.

    The same function works for:
    U-Net, DeepLabV3+, SAM-LoRA.
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
            "Frozen threshold must be between 0 and 1."
        )

    source_raster = validate_arcgis_raster(
        raster_path
    )

    (
        model,
        checkpoint,
        resolved_device,
    ) = restore_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        context=context,
        device=device,
        strict=True,
    )

    trained_tile_size = get_restored_tile_size(
        model=model,
        context=context,
    )

    if tile_size is None:

        tile_size = trained_tile_size

    else:

        tile_size = int(
            tile_size
        )

        if tile_size != trained_tile_size:

            raise ValueError(
                "GIS inference tile_size does not match the trained "
                "checkpoint.\n"
                f"Checkpoint tile size : {trained_tile_size}\n"
                f"Requested tile size  : {tile_size}"
            )

    if overlap is None:

        # 25% overlap by default.
        overlap = max(
            int(
                tile_size
                // 4
            ),
            0,
        )

    overlap = int(
        overlap
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    height = int(
        source_raster.height
    )

    width = int(
        source_raster.width
    )

    positions = generate_tile_positions(
        height=height,
        width=width,
        tile_size=tile_size,
        overlap=overlap,
    )

    # --------------------------------------------------------------
    # Disk-backed accumulation.
    #
    # This avoids requiring both full probability arrays to live in RAM.
    # --------------------------------------------------------------

    scratch_folder = (
        output_folder
        / "_inference_scratch"
    )

    scratch_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_sum_path = (
        scratch_folder
        / "probability_sum.dat"
    )

    weight_sum_path = (
        scratch_folder
        / "weight_sum.dat"
    )

    probability_sum = np.memmap(
        probability_sum_path,
        dtype=np.float32,
        mode="w+",
        shape=(
            height,
            width,
        ),
    )

    weight_sum = np.memmap(
        weight_sum_path,
        dtype=np.uint16,
        mode="w+",
        shape=(
            height,
            width,
        ),
    )

    probability_sum[
        :
    ] = 0.0

    weight_sum[
        :
    ] = 0

    print()
    print(
        "=" * 76
    )

    print(
        "ROAD GEOREFERENCED INFERENCE"
    )

    print(
        "=" * 76
    )

    print(
        f"Raster            : {raster_path}"
    )

    print(
        f"Size              : {width} x {height}"
    )

    print(
        f"Model             : "
        f"{getattr(model, '_road_checkpoint_reconstruction', {}).get('model_type')}"
    )

    print(
        f"Tile size         : {tile_size}"
    )

    print(
        f"Overlap           : {overlap}"
    )

    print(
        f"Tiles             : {len(positions)}"
    )

    print(
        f"Threshold         : {threshold:.3f}"
    )

    print(
        f"Device            : {resolved_device}"
    )

    print(
        f"Post-processing   : "
        f"{'ON - frozen settings' if postprocess_config is not None else 'OFF'}"
    )

    print(
        "Ground Truth      : NOT ACCESSED"
    )

    print(
        "=" * 76
    )

    try:

        for tile_index, (
            y,
            x,
        ) in enumerate(
            positions,
            start=1,
        ):

            (
                tile,
                real_height,
                real_width,
            ) = _read_rgb_window(
                raster=source_raster,
                top_row=y,
                left_column=x,
                tile_size=tile_size,
            )

            padded_tile = _pad_tile(
                tile,
                tile_size,
            )

            tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        padded_tile
                    )
                )
                .permute(
                    2,
                    0,
                    1,
                )
            )

            prediction = predict_tensor(
                model=model,
                images=tensor,
                device=resolved_device,
                threshold=threshold,
                use_amp=True,
            )

            probabilities = (
                prediction[
                    "probabilities"
                ][
                    0,
                    0,
                    :real_height,
                    :real_width,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            probability_sum[
                y:
                y + real_height,

                x:
                x + real_width,
            ] += probabilities

            weight_sum[
                y:
                y + real_height,

                x:
                x + real_width,
            ] += 1

            if (
                tile_index % 25 == 0
                or tile_index == len(
                    positions
                )
            ):

                print(
                    f"Processed tiles    : "
                    f"{tile_index}/{len(positions)}"
                )

            del tensor
            del prediction
            del probabilities

            if (
                torch.cuda.is_available()
                and tile_index % 50 == 0
            ):

                torch.cuda.empty_cache()

        # ----------------------------------------------------------
        # Verify complete coverage.
        # ----------------------------------------------------------

        if np.any(
            weight_sum == 0
        ):

            raise RuntimeError(
                "Sliding-window inference left uncovered raster pixels."
            )

        # ----------------------------------------------------------
        # Final probability average.
        # ----------------------------------------------------------

        probability_path = (
            scratch_folder
            / "final_probability.dat"
        )

        final_probability = np.memmap(
            probability_path,
            dtype=np.float32,
            mode="w+",
            shape=(
                height,
                width,
            ),
        )

        # Chunk by rows to keep temporary RAM bounded.
        chunk_rows = 1024

        for row_start in range(
            0,
            height,
            chunk_rows,
        ):

            row_end = min(
                row_start
                + chunk_rows,
                height,
            )

            final_probability[
                row_start:
                row_end
            ] = (
                probability_sum[
                    row_start:
                    row_end
                ]
                /
                weight_sum[
                    row_start:
                    row_end
                ]
            )

        final_probability.flush()

        # ----------------------------------------------------------
        # Frozen threshold.
        # ----------------------------------------------------------

        binary_mask = (
            np.asarray(
                final_probability
            )
            >= threshold
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------------
        # Frozen validation-selected Road morphology.
        # ----------------------------------------------------------

        binary_mask = apply_frozen_postprocessing(
            binary_mask=binary_mask,
            postprocess_config=postprocess_config,
        )

        # ----------------------------------------------------------
        # Save probability raster.
        # ----------------------------------------------------------

        probability_raster = None

        if save_probability_raster:

            probability_raster = (
                output_folder
                / f"{output_name}_probability.tif"
            )

            save_georeferenced_array(
                array=final_probability,
                source_raster=source_raster,
                output_path=probability_raster,
            )

            probability_raster = str(
                probability_raster
            )

        # ----------------------------------------------------------
        # Save binary raster.
        # ----------------------------------------------------------

        binary_raster = (
            output_folder
            / f"{output_name}_binary.tif"
        )

        save_georeferenced_array(
            array=binary_mask,
            source_raster=source_raster,
            output_path=binary_raster,
        )

        binary_raster = str(
            binary_raster
        )

        # ----------------------------------------------------------
        # Polygonization.
        # ----------------------------------------------------------

        prediction_fc = None

        if polygonize:

            if output_geodatabase is None:

                raise ValueError(
                    "output_geodatabase is required when polygonize=True."
                )

            arcpy = _get_arcpy()

            output_geodatabase = str(
                output_geodatabase
            )

            if not arcpy.Exists(
                output_geodatabase
            ):

                raise FileNotFoundError(
                    "Output geodatabase does not exist:\n"
                    f"{output_geodatabase}"
                )

            prediction_fc = os.path.join(
                output_geodatabase,
                output_name,
            )

            binary_raster_to_road_polygons(
                binary_raster=(
                    binary_raster
                ),

                output_feature_class=(
                    prediction_fc
                ),
            )

        reconstruction = getattr(
            model,
            "_road_checkpoint_reconstruction",
            {},
        )

        result = {
            "raster":
                str(
                    raster_path
                ),

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "checkpoint_format":
                reconstruction.get(
                    "checkpoint_format"
                ),

            "model_type":
                reconstruction.get(
                    "model_type"
                ),

            "tile_size":
                int(
                    tile_size
                ),

            "overlap":
                int(
                    overlap
                ),

            "tile_count":
                int(
                    len(
                        positions
                    )
                ),

            "threshold":
                float(
                    threshold
                ),

            "postprocessing_applied":
                bool(
                    postprocess_config
                    is not None
                ),

            "probability_raster":
                probability_raster,

            "binary_raster":
                binary_raster,

            "prediction_fc":
                prediction_fc,

            "prediction_feature_class":
                prediction_fc,

            "device":
                str(
                    resolved_device
                ),

            "ground_truth_used":
                False,

            "threshold_search_on_test":
                False,

            "postprocess_search_on_test":
                False,
        }

        return result

    finally:

        # ----------------------------------------------------------
        # Close disk-backed arrays.
        # ----------------------------------------------------------

        try:

            probability_sum.flush()

        except Exception:

            pass

        try:

            weight_sum.flush()

        except Exception:

            pass

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# =====================================================================
# COMPATIBILITY ALIASES
# =====================================================================

# Keep downstream controllers flexible while one canonical function is
# maintained internally.

run_full_raster_inference = (
    run_georeferenced_raster_inference
)

run_gis_inference = (
    run_georeferenced_raster_inference
)

predict_georeferenced_raster = (
    run_georeferenced_raster_inference
)

run_georeferenced_inference = (
    run_georeferenced_raster_inference
)
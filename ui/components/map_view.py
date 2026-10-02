"""
ui/components/map_view.py

Shared GIS preview for completed final-test outputs.

Visualization only:
- no inference
- no threshold selection
- no post-processing
- no source data modification
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


# ============================================================
# EXISTENCE
# ============================================================

def arcgis_exists(
    path,
) -> bool:

    if not path:
        return False

    try:
        import arcpy

        return bool(
            arcpy.Exists(
                str(
                    path
                )
            )
        )

    except Exception:
        try:
            return Path(
                str(
                    path
                )
            ).exists()
        except Exception:
            return False


# ============================================================
# OVERLAY CREATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=300,
)
def testing_image_with_overlays(
    raster_path: str,
    ground_truth_path: str | None = None,
    prediction_path: str | None = None,
    show_ground_truth: bool = True,
    show_prediction: bool = True,
    max_dimension: int = 1600,
):
    """
    Draw GT/prediction polygon outlines on the real test raster.
    """

    import io

    import arcpy
    import numpy as np
    from PIL import (
        Image,
        ImageDraw,
    )

    raster_path = str(
        raster_path
    )

    if not arcpy.Exists(
        raster_path
    ):
        raise FileNotFoundError(
            f"Testing raster not found: {raster_path}"
        )

    raster = arcpy.Raster(
        raster_path
    )

    raster_sr = (
        raster.spatialReference
    )

    if (
        raster_sr is None
        or raster_sr.name
        in (
            None,
            "",
            "Unknown",
        )
    ):
        raise ValueError(
            "Testing raster has no valid spatial reference."
        )

    extent = (
        raster.extent
    )

    array = np.asarray(
        arcpy.RasterToNumPyArray(
            raster
        )
    )

    # --------------------------------------------------------
    # RASTER -> RGB
    # --------------------------------------------------------

    if array.ndim == 2:

        rgb = np.stack(
            [
                array,
                array,
                array,
            ],
            axis=-1,
        )

    elif array.ndim == 3:

        if array.shape[0] <= 16:

            bands = array[
                :3
            ]

            if bands.shape[0] == 1:
                bands = np.repeat(
                    bands,
                    3,
                    axis=0,
                )

            elif bands.shape[0] == 2:
                bands = np.concatenate(
                    [
                        bands,
                        bands[:1],
                    ],
                    axis=0,
                )

            rgb = np.moveaxis(
                bands,
                0,
                -1,
            )

        else:
            rgb = array[
                ...,
                :3
            ]

    else:
        raise ValueError(
            f"Unsupported raster shape: {array.shape}"
        )

    height, width = (
        rgb.shape[
            :2
        ]
    )

    step = max(
        1,
        int(
            max(
                height,
                width,
            )
            / int(
                max_dimension
            )
        ),
    )

    rgb = rgb[
        ::step,
        ::step,
        :
    ].astype(
        "float32",
        copy=False,
    )

    # --------------------------------------------------------
    # DISPLAY STRETCH
    # --------------------------------------------------------

    stretched = np.zeros(
        rgb.shape,
        dtype="uint8",
    )

    for band_index in range(
        3
    ):
        band = rgb[
            :,
            :,
            band_index,
        ]

        finite = band[
            np.isfinite(
                band
            )
        ]

        if finite.size == 0:
            continue

        low = float(
            np.percentile(
                finite,
                2,
            )
        )

        high = float(
            np.percentile(
                finite,
                98,
            )
        )

        if high <= low:
            low = float(
                np.min(
                    finite
                )
            )

            high = float(
                np.max(
                    finite
                )
            )

        if high <= low:
            continue

        normalized = np.clip(
            (
                band
                - low
            )
            / (
                high
                - low
            ),
            0.0,
            1.0,
        )

        stretched[
            :,
            :,
            band_index,
        ] = (
            normalized
            * 255.0
        ).astype(
            "uint8"
        )

    base = Image.fromarray(
        stretched,
        mode="RGB",
    ).convert(
        "RGBA"
    )

    preview_width, preview_height = (
        base.size
    )

    x_span = float(
        extent.XMax
        - extent.XMin
    )

    y_span = float(
        extent.YMax
        - extent.YMin
    )

    if (
        x_span <= 0
        or y_span <= 0
    ):
        raise ValueError(
            "Testing raster has an invalid extent."
        )

    # --------------------------------------------------------
    # WORLD -> PIXEL
    # --------------------------------------------------------

    def world_to_pixel(
        x,
        y,
    ):
        px = (
            (
                float(
                    x
                )
                - float(
                    extent.XMin
                )
            )
            / x_span
            * preview_width
        )

        py = (
            (
                float(
                    extent.YMax
                )
                - float(
                    y
                )
            )
            / y_span
            * preview_height
        )

        return (
            int(
                round(
                    px
                )
            ),
            int(
                round(
                    py
                )
            ),
        )

    # --------------------------------------------------------
    # FEATURE DRAW
    # --------------------------------------------------------

    def draw_feature_class(
        image,
        feature_class,
        line_color,
        line_width,
    ):

        if (
            not feature_class
            or not arcpy.Exists(
                str(
                    feature_class
                )
            )
        ):
            return (
                image,
                0,
            )

        feature_class = str(
            feature_class
        )

        desc = arcpy.Describe(
            feature_class
        )

        feature_sr = getattr(
            desc,
            "spatialReference",
            None,
        )

        overlay = Image.new(
            "RGBA",
            image.size,
            (
                0,
                0,
                0,
                0,
            ),
        )

        drawer = ImageDraw.Draw(
            overlay
        )

        count = 0

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

                if geometry is None:
                    continue

                try:
                    if (
                        feature_sr
                        is not None
                        and feature_sr.name
                        not in (
                            None,
                            "",
                            "Unknown",
                        )
                        and feature_sr.factoryCode
                        != raster_sr.factoryCode
                    ):
                        geometry = geometry.projectAs(
                            raster_sr
                        )
                except Exception:
                    pass

                geometry_extent = getattr(
                    geometry,
                    "extent",
                    None,
                )

                if geometry_extent is not None:

                    if (
                        geometry_extent.XMax
                        < extent.XMin
                        or geometry_extent.XMin
                        > extent.XMax
                        or geometry_extent.YMax
                        < extent.YMin
                        or geometry_extent.YMin
                        > extent.YMax
                    ):
                        continue

                count += 1

                for part in geometry:

                    ring = []

                    for point in part:

                        if point is None:

                            if len(
                                ring
                            ) >= 2:

                                drawer.line(
                                    ring
                                    + [
                                        ring[0]
                                    ],
                                    fill=line_color,
                                    width=line_width,
                                )

                            ring = []
                            continue

                        ring.append(
                            world_to_pixel(
                                point.X,
                                point.Y,
                            )
                        )

                    if len(
                        ring
                    ) >= 2:

                        drawer.line(
                            ring
                            + [
                                ring[0]
                            ],
                            fill=line_color,
                            width=line_width,
                        )

        return (
            Image.alpha_composite(
                image,
                overlay,
            ),
            count,
        )

    gt_count = 0
    prediction_count = 0

    if show_ground_truth:

        base, gt_count = (
            draw_feature_class(
                base,
                ground_truth_path,
                (
                    220,
                    38,
                    38,
                    255,
                ),
                5,
            )
        )

    if show_prediction:

        base, prediction_count = (
            draw_feature_class(
                base,
                prediction_path,
                (
                    37,
                    99,
                    235,
                    255,
                ),
                3,
            )
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    buffer = io.BytesIO()

    base.convert(
        "RGB"
    ).save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return {
        "image_bytes":
            buffer.getvalue(),

        "ground_truth_count_drawn":
            gt_count,

        "prediction_count_drawn":
            prediction_count,
    }


# ============================================================
# STREAMLIT VIEW
# ============================================================

def render_final_overlay(
    raster_path,
    ground_truth_path,
    prediction_path,
    caption: str,
    key_prefix: str,
) -> None:
    """
    Render one completed final-test overlay.
    """

    if not raster_path:
        st.warning(
            "Final testing raster path is unavailable."
        )
        return

    c1, c2 = st.columns(
        2
    )

    with c1:
        show_gt = st.checkbox(
            "Ground Truth",
            value=True,
            key=(
                f"{key_prefix}_gt"
            ),
        )

    with c2:
        show_prediction = st.checkbox(
            "Final Prediction",
            value=True,
            key=(
                f"{key_prefix}_prediction"
            ),
        )

    if (
        not show_gt
        and not show_prediction
    ):
        st.info(
            "Select Ground Truth, Final Prediction, or both."
        )
        return

    try:
        result = testing_image_with_overlays(
            raster_path=str(
                raster_path
            ),

            ground_truth_path=(
                str(
                    ground_truth_path
                )
                if ground_truth_path
                else None
            ),

            prediction_path=(
                str(
                    prediction_path
                )
                if prediction_path
                else None
            ),

            show_ground_truth=(
                show_gt
                and arcgis_exists(
                    ground_truth_path
                )
            ),

            show_prediction=(
                show_prediction
                and arcgis_exists(
                    prediction_path
                )
            ),
        )

        st.image(
            result[
                "image_bytes"
            ],
            caption=caption,
            use_container_width=True,
        )

        st.caption(
            "Red = Ground Truth Â· Blue = Final Prediction. "
            "Visualization only; inference is not rerun."
        )

    except Exception as exc:
        st.error(
            f"Could not create the GIS overlay: {exc}"
        )


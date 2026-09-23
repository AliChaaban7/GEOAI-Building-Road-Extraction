"""
raster_io.py

Shared raster-reading utilities for the Building Extraction module.

Purpose
-------
Provide one robust reader for common image formats used by the module,
including GeoTIFF chips that Pillow may not be able to decode.

Read order:
    1. Pillow
    2. rasterio
    3. ArcPy

This keeps U-Net / DeepLabV3 compatible with aerial/satellite TIFF chips
without creating a dependency from semantic training code to the
Mask R-CNN engine.

Important
---------
This module does NOT apply model normalization. It only reads arrays.
Training code keeps control of normalization so experiment behavior remains
explicit and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _as_hwc_rgb(array: np.ndarray) -> np.ndarray:
    """
    Convert a raster array to H x W x 3 without changing numeric scale.
    """
    array = np.asarray(array)

    if array.ndim == 2:
        return np.repeat(
            array[:, :, None],
            3,
            axis=2,
        )

    if array.ndim != 3:
        raise ValueError(
            f"Unsupported raster shape for RGB conversion: {array.shape}"
        )

    # Common rasterio / ArcPy layout: bands x rows x cols.
    # Prefer this interpretation when the first dimension looks like bands.
    if (
        array.shape[0] <= 16
        and array.shape[1] > 16
        and array.shape[2] > 16
    ):
        array = np.moveaxis(
            array,
            0,
            -1,
        )

    # H x W x C at this point.
    if array.shape[-1] == 1:
        array = np.repeat(
            array,
            3,
            axis=2,
        )

    elif array.shape[-1] == 2:
        third = array[:, :, :1]
        array = np.concatenate(
            [array, third],
            axis=2,
        )

    elif array.shape[-1] > 3:
        array = array[:, :, :3]

    return array


def _as_2d_mask(array: np.ndarray) -> np.ndarray:
    """
    Convert a raster/mask array to H x W.
    """
    array = np.asarray(array)

    if array.ndim == 2:
        return array

    if array.ndim != 3:
        raise ValueError(
            f"Unsupported raster shape for mask conversion: {array.shape}"
        )

    # bands x rows x cols
    if (
        array.shape[0] <= 16
        and array.shape[1] > 16
        and array.shape[2] > 16
    ):
        return array[0]

    # rows x cols x channels
    return array[:, :, 0]


def read_raster_array(
    path,
    *,
    as_mask: bool = False,
) -> np.ndarray:
    """
    Read a raster using Pillow -> rasterio -> ArcPy fallbacks.

    Parameters
    ----------
    path:
        Raster/image path.

    as_mask:
        If True, return a 2D array.
        If False, return an H x W x 3 RGB-like array.

    Returns
    -------
    np.ndarray
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found: {path}"
        )

    errors = []

    # --------------------------------------------------------
    # 1. Pillow
    # --------------------------------------------------------
    try:
        with Image.open(path) as image:
            if as_mask:
                array = np.asarray(image)
            else:
                array = np.asarray(
                    image.convert("RGB")
                )

        if as_mask:
            return _as_2d_mask(array)

        return _as_hwc_rgb(array)

    except Exception as exc:
        errors.append(
            f"Pillow: {exc}"
        )

    # --------------------------------------------------------
    # 2. rasterio
    # --------------------------------------------------------
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            array = dataset.read()

        if as_mask:
            return _as_2d_mask(array)

        return _as_hwc_rgb(array)

    except Exception as exc:
        errors.append(
            f"rasterio: {exc}"
        )

    # --------------------------------------------------------
    # 3. ArcPy
    # --------------------------------------------------------
    try:
        import arcpy

        array = arcpy.RasterToNumPyArray(
            str(path)
        )

        if as_mask:
            return _as_2d_mask(array)

        return _as_hwc_rgb(array)

    except Exception as exc:
        errors.append(
            f"ArcPy: {exc}"
        )

    raise RuntimeError(
        "Unable to read raster:\n"
        f"{path}\n\n"
        + "\n".join(errors)
    )


def get_raster_size(path) -> tuple[int, int]:
    """
    Return raster size as (width, height), using robust fallbacks.

    This function avoids depending on Pillow for GeoTIFF metadata.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found: {path}"
        )

    errors = []

    # --------------------------------------------------------
    # 1. Pillow
    # --------------------------------------------------------
    try:
        with Image.open(path) as image:
            width, height = image.size

        return int(width), int(height)

    except Exception as exc:
        errors.append(
            f"Pillow: {exc}"
        )

    # --------------------------------------------------------
    # 2. rasterio
    # --------------------------------------------------------
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            return (
                int(dataset.width),
                int(dataset.height),
            )

    except Exception as exc:
        errors.append(
            f"rasterio: {exc}"
        )

    # --------------------------------------------------------
    # 3. ArcPy
    # --------------------------------------------------------
    try:
        import arcpy

        raster = arcpy.Raster(
            str(path)
        )

        width = int(
            arcpy.management.GetRasterProperties(
                raster,
                "COLUMNCOUNT",
            ).getOutput(0)
        )

        height = int(
            arcpy.management.GetRasterProperties(
                raster,
                "ROWCOUNT",
            ).getOutput(0)
        )

        return width, height

    except Exception as exc:
        errors.append(
            f"ArcPy: {exc}"
        )

    raise RuntimeError(
        "Unable to determine raster size:\n"
        f"{path}\n\n"
        + "\n".join(errors)
    )


def read_rgb_array(path) -> np.ndarray:
    """
    Read an image as H x W x 3.
    Numeric scale is preserved.
    """
    return read_raster_array(
        path,
        as_mask=False,
    )


def read_mask_array(path) -> np.ndarray:
    """
    Read a mask as H x W.
    Numeric scale is preserved.
    """
    return read_raster_array(
        path,
        as_mask=True,
    )

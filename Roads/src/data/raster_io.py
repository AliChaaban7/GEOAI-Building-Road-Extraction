"""
Raster and mask I/O utilities for Approach 2 - Road Extraction.

Designed for ArcGIS Pro Classified Tiles exports:

    CT_xxx_Roads_03/
        images/
        labels/
        esri_accumulated_stats.json
        map.txt
        stats.txt

Responsibilities
----------------
- Read exported image chips.
- Read Road label masks.
- Convert imagery to RGB.
- Convert Road labels to binary masks.
- Validate image/mask dimensions.
- Support common PNG/JPEG/TIFF ArcGIS exports.
- Use Pillow first and Rasterio as a fallback when necessary.

The module does not perform augmentation, normalization, train/validation
splitting, or model-specific preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Supported raster formats
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


# ---------------------------------------------------------------------
# Generic raster reading
# ---------------------------------------------------------------------

def _read_with_pillow(
    path: Path,
) -> np.ndarray:
    """
    Read a raster using Pillow.

    Pillow is preferred for normal ArcGIS image chips because it is
    lightweight and handles PNG/JPEG/TIFF exports well.
    """
    with Image.open(path) as image:
        array = np.asarray(image)

    return array


def _read_with_rasterio(
    path: Path,
) -> np.ndarray:
    """
    Read a raster using Rasterio.

    Rasterio is used only as a fallback when Pillow cannot read a file.
    """
    try:
        import rasterio

    except ImportError as exc:
        raise RuntimeError(
            "Pillow could not read the raster and Rasterio is not "
            "available in the current Python environment.\n"
            f"Raster: {path}"
        ) from exc

    with rasterio.open(path) as src:
        array = src.read()

    # Rasterio format:
    #
    #     bands x height x width
    #
    # Convert to:
    #
    #     height x width x bands
    #
    if array.ndim == 3:
        array = np.transpose(
            array,
            (1, 2, 0),
        )

    return array


def read_array(
    path: str | Path,
) -> np.ndarray:
    """
    Read an image or label raster into a NumPy array.

    Pillow is attempted first. Rasterio is used as a fallback.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raster does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected raster file but received:\n{path}"
        )

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported raster extension '{path.suffix}' for:\n"
            f"{path}"
        )

    try:
        array = _read_with_pillow(
            path
        )

    except Exception as pillow_error:

        try:
            array = _read_with_rasterio(
                path
            )

        except Exception as rasterio_error:
            raise RuntimeError(
                f"Unable to read raster:\n{path}\n\n"
                f"Pillow error:\n{pillow_error}\n\n"
                f"Rasterio error:\n{rasterio_error}"
            ) from rasterio_error

    return np.asarray(
        array
    )


# ---------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------

def _convert_to_three_channels(
    array: np.ndarray,
) -> np.ndarray:
    """
    Convert input imagery into H x W x 3 RGB-like format.

    Behavior
    --------
    2D:
        Repeated into three channels.

    1 channel:
        Repeated into three channels.

    2 channels:
        Rejected because there is no unambiguous RGB interpretation.

    3+ channels:
        First three channels are used.

    This is consistent with the current Road models, which expect
    three-channel image input.
    """
    if array.ndim == 2:
        return np.repeat(
            array[..., None],
            3,
            axis=2,
        )

    if array.ndim != 3:
        raise ValueError(
            "Expected a 2D or 3D image array. "
            f"Received shape: {array.shape}"
        )

    channels = int(
        array.shape[2]
    )

    if channels == 1:
        return np.repeat(
            array,
            3,
            axis=2,
        )

    if channels == 2:
        raise ValueError(
            "Two-band imagery cannot be automatically converted "
            "to RGB without an explicit band mapping."
        )

    if channels >= 3:
        return array[:, :, :3]

    raise ValueError(
        f"Unsupported image channel count: {channels}"
    )


def _scale_image_to_uint8(
    array: np.ndarray,
) -> np.ndarray:
    """
    Convert an image array to uint8 [0, 255].

    ArcGIS exported chips are normally already uint8. This function
    also provides safe handling for floating-point and higher-bit-depth
    chips.
    """
    array = np.asarray(
        array
    )

    if array.size == 0:
        raise ValueError(
            "Cannot process an empty image array."
        )

    # Already correct.
    if array.dtype == np.uint8:
        return array

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32
    )

    minimum = float(
        np.min(array)
    )

    maximum = float(
        np.max(array)
    )

    # Common normalized image range.
    if (
        minimum >= 0.0
        and maximum <= 1.0
    ):
        array = array * 255.0

    # Already in approximately uint8 range.
    elif (
        minimum >= 0.0
        and maximum <= 255.0
    ):
        pass

    # Higher bit-depth or unusual numeric range.
    else:

        if maximum <= minimum:
            array = np.zeros_like(
                array,
                dtype=np.float32,
            )

        else:
            array = (
                array - minimum
            ) / (
                maximum - minimum
            )

            array = (
                array * 255.0
            )

    array = np.clip(
        array,
        0,
        255,
    )

    return array.astype(
        np.uint8
    )


def read_rgb_image(
    path: str | Path,
) -> np.ndarray:
    """
    Read an ArcGIS image chip as RGB uint8.

    Returns
    -------
    numpy.ndarray
        Shape:

            H x W x 3

        Data type:

            uint8
    """
    array = read_array(
        path
    )

    array = _convert_to_three_channels(
        array
    )

    array = _scale_image_to_uint8(
        array
    )

    return np.ascontiguousarray(
        array
    )


# ---------------------------------------------------------------------
# Label / mask processing
# ---------------------------------------------------------------------

def _collapse_mask_channels(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Convert an exported label image to one 2D class raster.

    Standard ArcGIS Classified Tiles labels are normally single-band.

    Some visualization/export formats may return three channels.
    When all channels are identical, the first channel is used.

    If channels differ, any non-zero channel is treated as foreground.
    """
    mask = np.asarray(
        mask
    )

    if mask.ndim == 2:
        return mask

    if mask.ndim != 3:
        raise ValueError(
            "Expected a 2D or 3D mask array. "
            f"Received shape: {mask.shape}"
        )

    channels = mask.shape[2]

    if channels == 1:
        return mask[:, :, 0]

    first_channel = mask[:, :, 0]

    channels_identical = True

    for channel_index in range(
        1,
        channels,
    ):
        if not np.array_equal(
            first_channel,
            mask[:, :, channel_index],
        ):
            channels_identical = False
            break

    if channels_identical:
        return first_channel

    # Fallback for RGB-style label visualization.
    return np.max(
        mask,
        axis=2,
    )


def read_binary_mask(
    path: str | Path,
    foreground_values: Optional[
        Sequence[int | float]
    ] = None,
) -> np.ndarray:
    """
    Read an ArcGIS Road label and convert it to a binary mask.

    Parameters
    ----------
    path:
        Label raster.

    foreground_values:
        Optional explicit class values representing Road.

        Example:

            foreground_values=[1]

        If omitted, every non-zero class value is considered Road.

    Returns
    -------
    numpy.ndarray
        Shape:

            H x W

        Values:

            0 = background
            1 = road

        Data type:

            uint8
    """
    mask = read_array(
        path
    )

    mask = _collapse_mask_channels(
        mask
    )

    mask = np.nan_to_num(
        mask,
        nan=0,
        posinf=0,
        neginf=0,
    )

    if foreground_values is None:

        binary = (
            mask > 0
        )

    else:

        values = np.asarray(
            list(foreground_values)
        )

        if values.size == 0:
            raise ValueError(
                "foreground_values cannot be empty."
            )

        binary = np.isin(
            mask,
            values,
        )

    return np.ascontiguousarray(
        binary.astype(
            np.uint8
        )
    )


# ---------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------

def validate_pair(
    image_path: str | Path,
    mask_path: str | Path,
    expected_tile_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Validate one image/mask pair.

    Checks:
    - image can be read
    - mask can be read
    - spatial dimensions match
    - optional expected tile size matches

    Returns
    -------
    tuple
        (height, width)
    """
    image = read_rgb_image(
        image_path
    )

    mask = read_binary_mask(
        mask_path
    )

    image_shape = tuple(
        image.shape[:2]
    )

    mask_shape = tuple(
        mask.shape[:2]
    )

    if image_shape != mask_shape:
        raise ValueError(
            "Image/mask spatial-size mismatch.\n"
            f"Image: {image_path}\n"
            f"Shape: {image_shape}\n\n"
            f"Mask: {mask_path}\n"
            f"Shape: {mask_shape}"
        )

    if expected_tile_size is not None:

        expected_tile_size = int(
            expected_tile_size
        )

        expected_shape = (
            expected_tile_size,
            expected_tile_size,
        )

        if image_shape != expected_shape:
            raise ValueError(
                "Unexpected tile dimensions.\n"
                f"Expected: {expected_shape}\n"
                f"Received: {image_shape}\n"
                f"Image: {image_path}"
            )

    return image_shape


# ---------------------------------------------------------------------
# Mask statistics
# ---------------------------------------------------------------------

def calculate_mask_statistics(
    mask: np.ndarray,
) -> dict:
    """
    Calculate basic statistics for a binary Road mask.

    Useful later for dataset quality checks.
    """
    mask = np.asarray(
        mask
    )

    if mask.ndim != 2:
        raise ValueError(
            "Mask statistics require a 2D binary mask. "
            f"Received shape: {mask.shape}"
        )

    binary = (
        mask > 0
    )

    total_pixels = int(
        binary.size
    )

    road_pixels = int(
        binary.sum()
    )

    background_pixels = (
        total_pixels - road_pixels
    )

    road_fraction = (
        road_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "total_pixels": total_pixels,

        "road_pixels": road_pixels,

        "background_pixels": background_pixels,

        "road_fraction": float(
            road_fraction
        ),

        "road_percentage": float(
            road_fraction * 100.0
        ),

        "contains_road": bool(
            road_pixels > 0
        ),
    }


# ---------------------------------------------------------------------
# Simple save helpers
# ---------------------------------------------------------------------

def save_binary_mask(
    mask: np.ndarray,
    output_path: str | Path,
    foreground_value: int = 255,
) -> Path:
    """
    Save a binary Road mask as an 8-bit image.

    Internal masks remain {0, 1}; saved visualization masks use:

        0   = background
        255 = road
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask = np.asarray(
        mask
    )

    if mask.ndim != 2:
        raise ValueError(
            "save_binary_mask expects a 2D mask. "
            f"Received shape: {mask.shape}"
        )

    output = (
        mask > 0
    ).astype(
        np.uint8
    )

    output = (
        output * int(
            foreground_value
        )
    ).astype(
        np.uint8
    )

    Image.fromarray(
        output
    ).save(
        output_path
    )

    return output_path


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def is_supported_raster(
    path: str | Path,
) -> bool:
    """
    Return True when a path has a supported raster extension.
    """
    path = Path(
        path
    )

    return (
        path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )


def list_raster_files(
    folder: str | Path,
) -> list[Path]:
    """
    Return sorted supported raster files from a directory.
    """
    folder = Path(
        folder
    )

    if not folder.exists():
        raise FileNotFoundError(
            f"Raster folder does not exist:\n{folder}"
        )

    if not folder.is_dir():
        raise ValueError(
            f"Expected a directory:\n{folder}"
        )

    files = [
        path
        for path in folder.iterdir()
        if is_supported_raster(
            path
        )
    ]

    return sorted(
        files,
        key=lambda path: path.name.lower(),
    )
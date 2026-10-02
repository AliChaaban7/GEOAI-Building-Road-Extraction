"""
Morphological utilities for Road post-processing.

Approach 2 - Road Extraction

Supported operations
--------------------
- Remove tiny disconnected components.
- Morphological closing.
- Fill small interior holes.

These operations work on binary Road-surface masks:

    0 = background
    1 = road

Important
---------
No skeletonization is applied to the primary Road-surface prediction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------
# SciPy dependency
# ---------------------------------------------------------------------

def _get_ndimage():
    """
    Import scipy.ndimage only when post-processing is actually used.
    """

    try:
        from scipy import ndimage

    except ImportError as exc:
        raise RuntimeError(
            "Road post-processing requires scipy.\n"
            "Install/enable scipy in the ArcGIS Pro Python environment."
        ) from exc

    return ndimage


# ---------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------

def ensure_binary_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Convert an input mask to uint8 {0,1}.
    """

    mask = np.asarray(
        mask
    )

    if mask.ndim != 2:
        raise ValueError(
            "Road post-processing expects a 2D mask.\n"
            f"Received shape: {mask.shape}"
        )

    return (
        mask > 0
    ).astype(
        np.uint8
    )


# ---------------------------------------------------------------------
# Connected-component removal
# ---------------------------------------------------------------------

def remove_small_components(
    mask: np.ndarray,
    minimum_size: int = 0,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove foreground components smaller than minimum_size pixels.

    Parameters
    ----------
    minimum_size:
        Minimum Road-component area in pixels.

        0 or 1:
            no effective filtering.

    connectivity:
        1 = 4-connected
        2 = 8-connected
    """

    mask = ensure_binary_mask(
        mask
    )

    minimum_size = int(
        minimum_size
    )

    if minimum_size <= 1:
        return mask.copy()

    if connectivity not in {
        1,
        2,
    }:
        raise ValueError(
            "connectivity must be 1 or 2."
        )

    ndimage = _get_ndimage()

    structure = ndimage.generate_binary_structure(
        rank=2,
        connectivity=connectivity,
    )

    labels, component_count = (
        ndimage.label(
            mask,
            structure=structure,
        )
    )

    if component_count == 0:
        return mask.copy()

    component_sizes = np.bincount(
        labels.ravel()
    )

    keep = (
        component_sizes
        >= minimum_size
    )

    # Background label 0 must remain background.
    keep[
        0
    ] = False

    cleaned = keep[
        labels
    ]

    return cleaned.astype(
        np.uint8
    )


# ---------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------

def make_disk_structure(
    radius: int,
) -> np.ndarray:
    """
    Create a circular binary structuring element.
    """

    radius = int(
        radius
    )

    if radius < 0:
        raise ValueError(
            "radius cannot be negative."
        )

    if radius == 0:
        return np.ones(
            (
                1,
                1,
            ),
            dtype=bool,
        )

    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    structure = (
        x * x
        + y * y
        <= radius * radius
    )

    return structure.astype(
        bool
    )


def binary_closing(
    mask: np.ndarray,
    radius: int = 0,
) -> np.ndarray:
    """
    Apply binary morphological closing.

    Closing can:
    - reconnect very small breaks,
    - smooth small discontinuities,
    - close narrow gaps.

    radius = 0:
        no operation.
    """

    mask = ensure_binary_mask(
        mask
    )

    radius = int(
        radius
    )

    if radius <= 0:
        return mask.copy()

    ndimage = _get_ndimage()

    structure = make_disk_structure(
        radius
    )

    result = ndimage.binary_closing(
        mask.astype(
            bool
        ),
        structure=structure,
    )

    return result.astype(
        np.uint8
    )


# ---------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------

def fill_small_holes(
    mask: np.ndarray,
    maximum_hole_size: int = 0,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Fill only interior background holes up to a maximum area.

    Background regions touching an image border are NOT holes and are
    therefore never filled.

    Parameters
    ----------
    maximum_hole_size:
        Maximum hole area in pixels.

        0:
            disabled.
    """

    mask = ensure_binary_mask(
        mask
    )

    maximum_hole_size = int(
        maximum_hole_size
    )

    if maximum_hole_size <= 0:
        return mask.copy()

    if connectivity not in {
        1,
        2,
    }:
        raise ValueError(
            "connectivity must be 1 or 2."
        )

    ndimage = _get_ndimage()

    background = np.logical_not(
        mask.astype(
            bool
        )
    )

    structure = ndimage.generate_binary_structure(
        rank=2,
        connectivity=connectivity,
    )

    labels, count = ndimage.label(
        background,
        structure=structure,
    )

    if count == 0:
        return mask.copy()

    sizes = np.bincount(
        labels.ravel()
    )

    # --------------------------------------------------------------
    # Identify labels touching image borders.
    # These represent the exterior background and must never be filled.
    # --------------------------------------------------------------

    border_labels = np.unique(
        np.concatenate(
            [
                labels[
                    0,
                    :
                ],

                labels[
                    -1,
                    :
                ],

                labels[
                    :,
                    0
                ],

                labels[
                    :,
                    -1
                ],
            ]
        )
    )

    fillable = np.zeros(
        len(
            sizes
        ),
        dtype=bool,
    )

    for label_id in range(
        1,
        len(
            sizes
        ),
    ):

        if (
            label_id
            in border_labels
        ):
            continue

        if (
            sizes[
                label_id
            ]
            <= maximum_hole_size
        ):
            fillable[
                label_id
            ] = True

    output = mask.astype(
        bool
    ).copy()

    output[
        fillable[
            labels
        ]
    ] = True

    return output.astype(
        np.uint8
    )


# ---------------------------------------------------------------------
# Component statistics
# ---------------------------------------------------------------------

def component_statistics(
    mask: np.ndarray,
) -> dict:
    """
    Return simple Road-component statistics.
    """

    mask = ensure_binary_mask(
        mask
    )

    ndimage = _get_ndimage()

    structure = ndimage.generate_binary_structure(
        2,
        2,
    )

    labels, component_count = (
        ndimage.label(
            mask,
            structure=structure,
        )
    )

    sizes = np.bincount(
        labels.ravel()
    )

    foreground_sizes = (
        sizes[
            1:
        ]
        if len(
            sizes
        ) > 1
        else np.array(
            [],
            dtype=np.int64,
        )
    )

    return {
        "foreground_pixels":
            int(
                mask.sum()
            ),

        "component_count":
            int(
                component_count
            ),

        "largest_component_pixels":
            (
                int(
                    foreground_sizes.max()
                )
                if foreground_sizes.size
                else 0
            ),

        "smallest_component_pixels":
            (
                int(
                    foreground_sizes.min()
                )
                if foreground_sizes.size
                else 0
            ),
    }
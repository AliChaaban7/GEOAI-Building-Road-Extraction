"""
Road-connectivity post-processing utilities.

Purpose
-------
Bridge small interruptions in predicted Road networks while avoiding
aggressive changes to the Road-surface mask.

The operation uses directional line-shaped morphological closing in:

- horizontal direction,
- vertical direction,
- main diagonal,
- opposite diagonal.

This is more Road-oriented than applying only a large isotropic kernel.

Important
---------
This is an OPTIONAL validation-selected operation.
"""

from __future__ import annotations

import numpy as np

from .morphology import (
    _get_ndimage,
    ensure_binary_mask,
)


# ---------------------------------------------------------------------
# Directional kernels
# ---------------------------------------------------------------------

def horizontal_structure(
    radius: int,
) -> np.ndarray:

    radius = int(
        radius
    )

    size = (
        2 * radius
        + 1
    )

    return np.ones(
        (
            1,
            size,
        ),
        dtype=bool,
    )


def vertical_structure(
    radius: int,
) -> np.ndarray:

    radius = int(
        radius
    )

    size = (
        2 * radius
        + 1
    )

    return np.ones(
        (
            size,
            1,
        ),
        dtype=bool,
    )


def diagonal_structure(
    radius: int,
    reverse: bool = False,
) -> np.ndarray:
    """
    Create a 45-degree line kernel.
    """

    radius = int(
        radius
    )

    size = (
        2 * radius
        + 1
    )

    structure = np.eye(
        size,
        dtype=bool,
    )

    if reverse:

        structure = np.fliplr(
            structure
        )

    return structure


# ---------------------------------------------------------------------
# Gap bridging
# ---------------------------------------------------------------------

def bridge_small_gaps(
    mask: np.ndarray,
    radius: int = 0,
) -> np.ndarray:
    """
    Attempt to bridge short Road-network gaps.

    radius
    ------
    0:
        disabled

    1:
        tests directional gaps over a 3-pixel neighborhood

    2:
        5-pixel neighborhood

    3:
        7-pixel neighborhood

    The original Road mask is always retained.
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

    source = mask.astype(
        bool
    )

    structures = [
        horizontal_structure(
            radius
        ),

        vertical_structure(
            radius
        ),

        diagonal_structure(
            radius,
            reverse=False,
        ),

        diagonal_structure(
            radius,
            reverse=True,
        ),
    ]

    candidates = [
        source
    ]

    for structure in structures:

        closed = ndimage.binary_closing(
            source,
            structure=structure,
        )

        candidates.append(
            closed
        )

    # Retain pixels accepted by any directional bridge.
    combined = np.logical_or.reduce(
        candidates
    )

    return combined.astype(
        np.uint8
    )
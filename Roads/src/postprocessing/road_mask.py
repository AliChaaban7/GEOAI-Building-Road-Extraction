"""
Road-surface mask post-processing.

This module is the authoritative implementation required by:
- Roads/src/postprocessing/__init__.py
- Roads/scripts/run_postprocess.py
- Road GIS inference when frozen post-processing is enabled.

Operations
----------
1. Remove small connected foreground components.
2. Apply optional binary closing.
3. Bridge short Road gaps with directional closing.
4. Fill small enclosed holes.

All sizes are expressed in PIXELS because validation post-processing is
selected on 256/384/... Road mask chips. The exact selected values are
frozen and reused unchanged during final inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

try:
    from scipy import ndimage
except Exception as exc:
    raise ImportError(
        "scipy is required for Road mask post-processing."
    ) from exc


@dataclass(frozen=True)
class RoadPostprocessingConfig:
    """
    Frozen Road morphology settings.

    Parameters are pixel-domain values selected on Validation only.
    """

    minimum_component_size: int = 0
    closing_radius: int = 0
    maximum_hole_size: int = 0
    bridge_gap_radius: int = 0

    def __post_init__(self) -> None:
        for name in (
            "minimum_component_size",
            "closing_radius",
            "maximum_hole_size",
            "bridge_gap_radius",
        ):
            value = int(getattr(self, name))

            if value < 0:
                raise ValueError(
                    f"{name} must be >= 0. Received: {value}"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

    def to_dict(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in asdict(self).items()
        }


def _binary(mask: Any) -> np.ndarray:
    """
    Convert an input mask to a writable 2-D boolean array.
    """
    array = np.asarray(mask)

    # Accept common singleton channel layouts.
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(
            "Road post-processing expects a 2-D binary mask "
            f"(or singleton-channel mask). Received shape: {array.shape}"
        )

    return np.array(
        array > 0,
        dtype=bool,
        copy=True,
    )


def _disk(radius: int) -> np.ndarray:
    radius = int(radius)

    if radius <= 0:
        return np.ones(
            (1, 1),
            dtype=bool,
        )

    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    return (
        x * x
        + y * y
        <= radius * radius
    )


def _remove_small_components(
    mask: np.ndarray,
    minimum_component_size: int,
) -> np.ndarray:
    minimum_component_size = int(
        minimum_component_size
    )

    if minimum_component_size <= 0:
        return mask.copy()

    structure = np.ones(
        (3, 3),
        dtype=bool,
    )

    labeled, count = ndimage.label(
        mask,
        structure=structure,
    )

    if count == 0:
        return mask.copy()

    sizes = np.bincount(
        labeled.ravel()
    )

    keep = sizes >= minimum_component_size

    if keep.size:
        keep[0] = False

    return keep[
        labeled
    ]


def _binary_closing(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    radius = int(radius)

    if radius <= 0:
        return mask.copy()

    return ndimage.binary_closing(
        mask,
        structure=_disk(radius),
        iterations=1,
        border_value=0,
    )


def _directional_line_structures(
    radius: int,
) -> tuple[np.ndarray, ...]:
    """
    Horizontal, vertical and two diagonal line elements.

    These are used only for the dedicated gap-bridging operation so
    bridging does not become a second isotropic dilation.
    """
    radius = int(radius)

    if radius <= 0:
        one = np.ones(
            (1, 1),
            dtype=bool,
        )
        return (
            one,
        )

    size = (
        2 * radius
        + 1
    )

    horizontal = np.zeros(
        (size, size),
        dtype=bool,
    )
    vertical = np.zeros_like(
        horizontal
    )
    diagonal_a = np.eye(
        size,
        dtype=bool,
    )
    diagonal_b = np.fliplr(
        diagonal_a
    ).copy()

    center = radius

    horizontal[
        center,
        :,
    ] = True

    vertical[
        :,
        center,
    ] = True

    return (
        horizontal,
        vertical,
        diagonal_a,
        diagonal_b,
    )


def _bridge_small_gaps(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Bridge short gaps while preserving the original Road mask.

    A directional binary closing is evaluated in four principal
    orientations and unioned with the original foreground.
    """
    radius = int(radius)

    if radius <= 0:
        return mask.copy()

    bridged = mask.copy()

    for structure in _directional_line_structures(
        radius
    ):
        candidate = ndimage.binary_closing(
            mask,
            structure=structure,
            iterations=1,
            border_value=0,
        )

        bridged = np.logical_or(
            bridged,
            candidate,
        )

    return bridged


def _fill_small_holes(
    mask: np.ndarray,
    maximum_hole_size: int,
) -> np.ndarray:
    maximum_hole_size = int(
        maximum_hole_size
    )

    if maximum_hole_size <= 0:
        return mask.copy()

    # Fill every enclosed hole first; then recover only holes whose
    # connected background area is <= the configured limit.
    fully_filled = ndimage.binary_fill_holes(
        mask
    )

    holes = np.logical_and(
        fully_filled,
        ~mask,
    )

    if not np.any(
        holes
    ):
        return mask.copy()

    structure = np.ones(
        (3, 3),
        dtype=bool,
    )

    labeled, count = ndimage.label(
        holes,
        structure=structure,
    )

    if count == 0:
        return mask.copy()

    sizes = np.bincount(
        labeled.ravel()
    )

    fill = (
        sizes <= maximum_hole_size
    )

    if fill.size:
        fill[0] = False

    selected_holes = fill[
        labeled
    ]

    return np.logical_or(
        mask,
        selected_holes,
    )


def road_postprocessing_config_from_dict(
    payload: Mapping[str, Any] | RoadPostprocessingConfig | None,
) -> RoadPostprocessingConfig:
    """
    Convert a JSON/dict payload to RoadPostprocessingConfig.

    Compatible with either:
        {minimum_component_size: ..., ...}

    or a validation summary fragment:
        {"selected_config": {...}}
    """
    if payload is None:
        return RoadPostprocessingConfig()

    if isinstance(
        payload,
        RoadPostprocessingConfig,
    ):
        return payload

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Road post-processing config must be a mapping, "
            "RoadPostprocessingConfig or None."
        )

    selected = payload.get(
        "selected_config"
    )

    if isinstance(
        selected,
        Mapping,
    ):
        payload = selected

    def first_int(
        *names: str,
        default: int = 0,
    ) -> int:
        for name in names:
            if name in payload and payload[name] is not None:
                return int(
                    payload[name]
                )

        return int(
            default
        )

    return RoadPostprocessingConfig(
        minimum_component_size=first_int(
            "minimum_component_size",
            "minimum_component_pixels",
            "min_component_pixels",
            default=0,
        ),
        closing_radius=first_int(
            "closing_radius",
            "closing",
            default=0,
        ),
        maximum_hole_size=first_int(
            "maximum_hole_size",
            "maximum_hole_pixels",
            "max_hole_pixels",
            default=0,
        ),
        bridge_gap_radius=first_int(
            "bridge_gap_radius",
            "bridge_radius",
            "bridge",
            default=0,
        ),
    )


def postprocess_road_mask(
    mask: Any,
    config: Mapping[str, Any] | RoadPostprocessingConfig | None = None,
) -> np.ndarray:
    """
    Apply the frozen Road morphology pipeline.

    Returns
    -------
    numpy.ndarray
        uint8 binary mask containing only values 0 and 1.
    """
    resolved = road_postprocessing_config_from_dict(
        config
    )

    result = _binary(
        mask
    )

    result = _remove_small_components(
        result,
        resolved.minimum_component_size,
    )

    result = _binary_closing(
        result,
        resolved.closing_radius,
    )

    result = _bridge_small_gaps(
        result,
        resolved.bridge_gap_radius,
    )

    result = _fill_small_holes(
        result,
        resolved.maximum_hole_size,
    )

    return np.asarray(
        result,
        dtype=np.uint8,
    )


def postprocess_with_values(
    mask: Any,
    minimum_component_size: int = 0,
    closing_radius: int = 0,
    maximum_hole_size: int = 0,
    bridge_gap_radius: int = 0,
) -> np.ndarray:
    """
    Convenience wrapper used by search/inference code.
    """
    config = RoadPostprocessingConfig(
        minimum_component_size=minimum_component_size,
        closing_radius=closing_radius,
        maximum_hole_size=maximum_hole_size,
        bridge_gap_radius=bridge_gap_radius,
    )

    return postprocess_road_mask(
        mask=mask,
        config=config,
    )


__all__ = [
    "RoadPostprocessingConfig",
    "road_postprocessing_config_from_dict",
    "postprocess_road_mask",
    "postprocess_with_values",
]

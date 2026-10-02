"""
Road-specific post-processing package.

Primary Road output remains a Road-surface segmentation mask.

Core operations
---------------
- small-component removal
- closing
- small-gap bridging
- small-hole filling

Scientific policy
-----------------
Post-processing settings are selected on Validation only and frozen
before final Aerial / Satellite / Drone testing.

Important package rule
----------------------
Core mask/morphology utilities are imported eagerly.

The older ``search.py`` helper module is NOT imported when the package is
initialized. This prevents stale optional search helpers from breaking
imports such as:

    from src.postprocessing.road_mask import ...

The current validation post-processing launcher
``Roads/scripts/run_postprocess.py`` owns the active validation search.
Legacy search helpers remain lazily accessible for compatibility.
"""

from __future__ import annotations

from importlib import import_module


# ============================================================
# CORE CONNECTIVITY
# ============================================================

from .connectivity import (
    bridge_small_gaps,
    diagonal_structure,
    horizontal_structure,
    vertical_structure,
)


# ============================================================
# CORE MORPHOLOGY
# ============================================================

from .morphology import (
    binary_closing,
    component_statistics,
    ensure_binary_mask,
    fill_small_holes,
    make_disk_structure,
    remove_small_components,
)


# ============================================================
# ROAD MASK PIPELINE
# ============================================================

from .road_mask import (
    RoadPostprocessingConfig,
    postprocess_road_mask,
    postprocess_with_values,
    road_postprocessing_config_from_dict,
)


# ============================================================
# LEGACY / OPTIONAL SEARCH API
# ============================================================
#
# Do NOT import .search eagerly here.
#
# The current Roads/scripts/run_postprocess.py performs the active
# validation-only search itself. Older helper APIs from search.py are
# preserved lazily so importing the core post-processing package cannot
# fail because of a stale optional dependency.
# ============================================================

_SEARCH_EXPORTS = {
    "build_postprocess_validation_loader",
    "cache_validation_predictions",
    "evaluate_postprocessing_candidate",
    "generate_postprocessing_candidates",
    "load_selected_threshold",
    "resolve_postprocessing_search_space",
    "run_postprocessing_search",
    "save_postprocessing_search_results",
    "search_postprocessing",
    "select_best_postprocessing",
}


def __getattr__(
    name: str,
):
    """
    Lazily resolve legacy search helpers only when explicitly requested.

    This keeps core Road mask imports independent from the older search.py
    implementation.
    """

    if name in _SEARCH_EXPORTS:
        module = import_module(
            ".search",
            __name__,
        )

        try:
            value = getattr(
                module,
                name,
            )
        except AttributeError as exc:
            raise AttributeError(
                f"src.postprocessing.search does not define {name!r}."
            ) from exc

        # Cache the resolved symbol in this package for later accesses.
        globals()[
            name
        ] = value

        return value

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(
        set(
            globals().keys()
        )
        | _SEARCH_EXPORTS
    )


__all__ = [
    # Road mask configuration / pipeline
    "RoadPostprocessingConfig",
    "road_postprocessing_config_from_dict",
    "postprocess_road_mask",
    "postprocess_with_values",

    # Morphology
    "ensure_binary_mask",
    "remove_small_components",
    "binary_closing",
    "fill_small_holes",
    "make_disk_structure",
    "component_statistics",

    # Connectivity
    "bridge_small_gaps",
    "horizontal_structure",
    "vertical_structure",
    "diagonal_structure",

    # Legacy/lazy search API
    "resolve_postprocessing_search_space",
    "generate_postprocessing_candidates",
    "build_postprocess_validation_loader",
    "cache_validation_predictions",
    "evaluate_postprocessing_candidate",
    "select_best_postprocessing",
    "search_postprocessing",
    "save_postprocessing_search_results",
    "load_selected_threshold",
    "run_postprocessing_search",
]

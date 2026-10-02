"""
Data package for Approach 2 - Road Extraction.

Public data utilities:
- ArcGIS Classified Tiles raster reading
- Image/mask pairing
- Train/Validation manifest generation
- PyTorch Road segmentation dataset
"""

from .dataset import RoadSegmentationDataset

from .manifests import (
    create_train_val_manifests,
    discover_classified_tile_pairs,
    read_manifest,
    split_pairs,
    write_manifest,
)

from .raster_io import (
    IMAGE_EXTENSIONS,
    calculate_mask_statistics,
    list_raster_files,
    read_array,
    read_binary_mask,
    read_rgb_image,
    save_binary_mask,
    validate_pair,
)


__all__ = [
    "RoadSegmentationDataset",

    "create_train_val_manifests",
    "discover_classified_tile_pairs",
    "read_manifest",
    "split_pairs",
    "write_manifest",

    "IMAGE_EXTENSIONS",
    "calculate_mask_statistics",
    "list_raster_files",
    "read_array",
    "read_binary_mask",
    "read_rgb_image",
    "save_binary_mask",
    "validate_pair",
]
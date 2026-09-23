"""
DeepLabV3 model package for the Buildings module.

The implementation lives in:
    src.models.deeplabv3.deeplabv3

This package re-exports the public DeepLabV3 API so existing imports such as:

    from src.models.deeplabv3 import build_deeplabv3

continue to work after reorganizing the model into its own folder.
"""

from .deeplabv3 import (
    build_deeplabv3,
    build_deeplabv3_model,
    create_deeplabv3_model,
)

__all__ = [
    "build_deeplabv3",
    "build_deeplabv3_model",
    "create_deeplabv3_model",
]

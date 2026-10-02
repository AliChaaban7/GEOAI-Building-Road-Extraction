"""
Road U-Net package exports.

The implementation lives in:
    Roads/src/models/unet/unet.py

The shared model factory imports:
    src.models.unet

so this package must expose the U-Net builder at package level.
"""

from .unet import build_unet

__all__ = [
    "build_unet",
]

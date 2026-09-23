"""
U-Net model package for the Buildings module.

The implementation lives in:
    src.models.unet.unet

This package re-exports the public U-Net API so existing imports such as:

    from src.models.unet import UNet, build_unet

continue to work after reorganizing the model into its own folder.
"""

from .unet import (
    DoubleConv,
    DownBlock,
    UpBlock,
    UNet,
    build_unet,
)

__all__ = [
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "UNet",
    "build_unet",
]

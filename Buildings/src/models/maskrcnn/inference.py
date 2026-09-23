"""
Mask R-CNN trained-checkpoint loading utilities.

The model architecture is reconstructed exactly as in the historical
Mask R-CNN engine, then the saved state_dict is loaded with strict=True.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .factory import create_maskrcnn_model


def load_checkpoint_model(
    checkpoint_path: Path,
    device=None,
    pretrained: bool = False,
    nms_threshold: float = 0.50,
    detections_per_image: int = 300,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        str(
            checkpoint_path
        ),
        map_location=device,
    )

    model = (
        create_maskrcnn_model(
            pretrained=pretrained,
            nms_threshold=nms_threshold,
            detections_per_image=detections_per_image,
        )
        .to(
            device
        )
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return (
        model,
        checkpoint,
        device,
    )


__all__ = [
    "load_checkpoint_model",
]

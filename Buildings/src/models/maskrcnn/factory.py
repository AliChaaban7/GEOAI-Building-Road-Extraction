"""
Mask R-CNN model-specific factory.

Purpose
-------
Provide the historical create_maskrcnn_model() API used by the Mask R-CNN
training, validation, threshold-search, post-processing, and inference code.

The shared Buildings model factory still uses:
    src.models.maskrcnn.maskrcnn.build_maskrcnn

This local factory preserves the exact behavior of the previous
src.instance.maskrcnn_engine.create_maskrcnn_model().
"""

from __future__ import annotations

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def create_maskrcnn_model(
    num_classes: int = 2,
    pretrained: bool = True,
    nms_threshold: float = 0.50,
    detections_per_image: int = 300,
):
    """
    Create the Mask R-CNN model used by the dedicated instance workflow.

    This implementation is intentionally kept behavior-compatible with the
    previous Mask R-CNN engine.
    """

    if pretrained:
        try:
            model = maskrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
        except (TypeError, ValueError):
            model = maskrcnn_resnet50_fpn(
                pretrained=True
            )
    else:
        try:
            model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
            )
        except TypeError:
            model = maskrcnn_resnet50_fpn(
                pretrained=False
            )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            int(num_classes),
        )
    )

    in_features_mask = (
        model.roi_heads
        .mask_predictor
        .conv5_mask
        .in_channels
    )

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            in_features_mask,
            256,
            int(num_classes),
        )
    )

    model.roi_heads.nms_thresh = float(
        nms_threshold
    )

    model.roi_heads.detections_per_img = int(
        detections_per_image
    )

    return model


__all__ = [
    "create_maskrcnn_model",
]

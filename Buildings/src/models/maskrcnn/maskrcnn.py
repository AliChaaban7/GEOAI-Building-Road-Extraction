"""
maskrcnn.py

Mask R-CNN model builder for instance-based building extraction.

This model is different from U-Net / DeepLabV3.

Semantic segmentation:
    image -> one binary mask

Mask R-CNN instance segmentation:
    image -> boxes + labels + scores + masks for each detected building
"""

import torch
import torchvision


def _get_nested(config, section, key, default=None):
    if not isinstance(config, dict):
        return default

    if section in config and isinstance(config[section], dict):
        if key in config[section]:
            return config[section][key]

    if key in config:
        return config[key]

    return default


def _build_maskrcnn_resnet50_fpn(pretrained=False):
    """
    Build Mask R-CNN ResNet50-FPN.

    Supports both older and newer torchvision versions.
    """

    from torchvision.models.detection import maskrcnn_resnet50_fpn

    if pretrained:
        # New torchvision API
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

            return maskrcnn_resnet50_fpn(
                weights=weights
            )

        except Exception as e:
            print("Warning: Could not load pretrained Mask R-CNN weights.")
            print(f"Reason: {e}")
            print("Falling back to random initialization.")

    # New torchvision API without weights
    try:
        return maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None
        )

    except TypeError:
        # Old torchvision API
        return maskrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=False
        )


def _replace_heads(model, num_classes):
    """
    Replace COCO heads with building extraction heads.

    num_classes:
        0 = background
        1 = building
    """

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # Box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features_box,
        num_classes=num_classes
    )

    # Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes
    )

    return model


def _apply_inference_thresholds(model, model_config):
    score_threshold = float(
        _get_nested(
            model_config,
            "output",
            "score_threshold",
            _get_nested(model_config, "training", "score_threshold", 0.5)
        )
    )

    nms_threshold = float(
        _get_nested(
            model_config,
            "output",
            "nms_threshold",
            _get_nested(model_config, "training", "nms_threshold", 0.5)
        )
    )

    detections_per_image = int(
        _get_nested(
            model_config,
            "output",
            "detections_per_image",
            _get_nested(model_config, "training", "detections_per_image", 300)
        )
    )

    model.roi_heads.score_thresh = score_threshold
    model.roi_heads.nms_thresh = nms_threshold
    model.roi_heads.detections_per_img = detections_per_image

    return model


def build_maskrcnn(
    experiment_config,
    model_config,
    backbone_info=None
):
    """
    Build Mask R-CNN model for buildings.

    Expected config:
        model_type = maskrcnn
        backbone = resnet50_fpn
        num_classes = 2
    """

    if backbone_info is None:
        backbone_info = {}

    backbone = experiment_config.get(
        "backbone",
        model_config.get("default_backbone", "resnet50_fpn")
    )

    if backbone not in ["resnet50_fpn"]:
        raise ValueError(
            f"Unsupported Mask R-CNN backbone: {backbone}. "
            "Currently supported: resnet50_fpn"
        )

    num_classes = int(model_config.get("num_classes", 2))

    pretrained = bool(
        experiment_config.get(
            "pretrained",
            backbone_info.get(
                "pretrained",
                model_config.get("pretrained", False)
            )
        )
    )

    model = _build_maskrcnn_resnet50_fpn(
        pretrained=pretrained
    )

    model = _replace_heads(
        model=model,
        num_classes=num_classes
    )

    model = _apply_inference_thresholds(
        model=model,
        model_config=model_config
    )

    return model


def build_model(
    experiment_config,
    model_config,
    backbone_info=None
):
    """
    Compatibility alias.
    """

    return build_maskrcnn(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info
    )
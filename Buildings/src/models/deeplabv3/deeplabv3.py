"""
deeplabv3.py

Standard DeepLabV3 model builder for building extraction.

This is the stable standard version:
- Config-driven backbone
- Config-driven pretrained option
- Config-driven aux_loss architecture
- Binary output channel for building mask
- Compatible with factory.py
"""

import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large
)


def _read_bool(key, default=False, experiment_config=None, model_config=None, backbone_info=None):
    """
    Read boolean setting from configs.

    Priority:
    1. experiment_config
    2. model_config
    3. backbone_info
    4. default
    """

    if isinstance(experiment_config, dict) and key in experiment_config:
        return bool(experiment_config[key])

    if isinstance(model_config, dict) and key in model_config:
        return bool(model_config[key])

    if isinstance(backbone_info, dict) and key in backbone_info:
        return bool(backbone_info[key])

    return bool(default)


def _read_backbone(experiment_config=None, model_config=None, backbone_info=None, default="resnet50"):
    """
    Resolve backbone name.
    """

    backbone = default

    if isinstance(experiment_config, dict):
        backbone = experiment_config.get("backbone", backbone)

    if isinstance(model_config, dict):
        backbone = model_config.get(
            "backbone",
            model_config.get("default_backbone", backbone)
        )

    if isinstance(backbone_info, dict):
        backbone = backbone_info.get("name", backbone)

    return str(backbone).lower()


def _read_num_classes(model_config=None, default=1):
    """
    Resolve output classes.
    """

    if isinstance(model_config, dict):
        return int(model_config.get("num_classes", default))

    return int(default)


def _build_resnet50(pretrained=False, aux_loss=False):
    """
    Build DeepLabV3 ResNet50.

    Standard module default:
        pretrained=False
        aux_loss=False
    """

    if pretrained:
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            print("Using pretrained DeepLabV3 ResNet50 weights.")

            return deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT,
                aux_loss=aux_loss
            )

        except Exception as e:
            print("Could not load pretrained DeepLabV3 ResNet50 weights.")
            print("Falling back to random DeepLabV3 ResNet50 weights.")
            print(f"Reason: {e}")

    print("Using random DeepLabV3 ResNet50 weights.")

    try:
        return deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            aux_loss=aux_loss
        )

    except TypeError:
        return deeplabv3_resnet50(
            pretrained=False,
            pretrained_backbone=False,
            aux_loss=aux_loss
        )


def _build_resnet101(pretrained=False, aux_loss=False):
    """
    Build DeepLabV3 ResNet101.
    """

    if pretrained:
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

            print("Using pretrained DeepLabV3 ResNet101 weights.")

            return deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                aux_loss=aux_loss
            )

        except Exception as e:
            print("Could not load pretrained DeepLabV3 ResNet101 weights.")
            print("Falling back to random DeepLabV3 ResNet101 weights.")
            print(f"Reason: {e}")

    print("Using random DeepLabV3 ResNet101 weights.")

    try:
        return deeplabv3_resnet101(
            weights=None,
            weights_backbone=None,
            aux_loss=aux_loss
        )

    except TypeError:
        return deeplabv3_resnet101(
            pretrained=False,
            pretrained_backbone=False,
            aux_loss=aux_loss
        )


def _build_mobilenet_v3_large(pretrained=False, aux_loss=False):
    """
    Build DeepLabV3 MobileNetV3 Large.
    """

    if pretrained:
        try:
            from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

            print("Using pretrained DeepLabV3 MobileNetV3 Large weights.")

            return deeplabv3_mobilenet_v3_large(
                weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
                aux_loss=aux_loss
            )

        except Exception as e:
            print("Could not load pretrained DeepLabV3 MobileNetV3 Large weights.")
            print("Falling back to random DeepLabV3 MobileNetV3 Large weights.")
            print(f"Reason: {e}")

    print("Using random DeepLabV3 MobileNetV3 Large weights.")

    try:
        return deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            aux_loss=aux_loss
        )

    except TypeError:
        return deeplabv3_mobilenet_v3_large(
            pretrained=False,
            pretrained_backbone=False,
            aux_loss=aux_loss
        )


def _replace_deeplabv3_heads(model, num_classes=1):
    """
    Replace DeepLabV3 classifier heads for binary building extraction.
    """

    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1
    )

    if hasattr(model, "aux_classifier"):
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(
                in_channels=256,
                out_channels=num_classes,
                kernel_size=1
            )

    return model


def build_deeplabv3(
    experiment_config=None,
    model_config=None,
    backbone_info=None,
    num_classes=1,
    backbone="resnet50"
):
    """
    Build DeepLabV3 model.

    Compatible with factory.py:

        build_deeplabv3(
            experiment_config=experiment_config,
            model_config=model_config,
            backbone_info=backbone_info
        )
    """

    backbone = _read_backbone(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        default=backbone
    )

    num_classes = _read_num_classes(
        model_config=model_config,
        default=num_classes
    )

    pretrained = _read_bool(
        key="pretrained",
        default=False,
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info
    )

    aux_loss = _read_bool(
        key="aux_loss",
        default=False,
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info
    )

    print("\n========== BUILD DEEPLABV3 ==========")
    print(f"Backbone: {backbone}")
    print(f"Pretrained: {pretrained}")
    print(f"Aux Loss Architecture: {aux_loss}")
    print(f"Num Classes: {num_classes}")
    print("=====================================\n")

    if backbone == "resnet50":
        model = _build_resnet50(
            pretrained=pretrained,
            aux_loss=aux_loss
        )

    elif backbone == "resnet101":
        model = _build_resnet101(
            pretrained=pretrained,
            aux_loss=aux_loss
        )

    elif backbone in ["mobilenet", "mobilenetv3", "mobilenet_v3_large"]:
        model = _build_mobilenet_v3_large(
            pretrained=pretrained,
            aux_loss=aux_loss
        )

    else:
        raise ValueError(f"Unsupported DeepLabV3 backbone: {backbone}")

    model = _replace_deeplabv3_heads(
        model=model,
        num_classes=num_classes
    )

    return model


def build_deeplabv3_model(*args, **kwargs):
    return build_deeplabv3(*args, **kwargs)


def create_deeplabv3_model(*args, **kwargs):
    return build_deeplabv3(*args, **kwargs)
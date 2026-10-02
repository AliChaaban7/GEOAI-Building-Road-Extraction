"""
Roads/src/models/unet.py

Road-surface U-Net family.

Supported backbones
-------------------
- vanilla_unet
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152

Every model returns one-channel logits with the same H x W as the input.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
except Exception as exc:
    raise ImportError(
        "torchvision is required for the ResNet-backed Road U-Net models."
    ) from exc


RESNET_CHANNELS = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "resnet101": (64, 256, 512, 1024, 2048),
    "resnet152": (64, 256, 512, 1024, 2048),
}


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = _conv_block(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.reduce(x)
        x = torch.cat(
            [x, skip],
            dim=1,
        )
        return self.fuse(x)


class VanillaUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels=(64, 128, 256, 512, 1024),
    ) -> None:
        super().__init__()

        c1, c2, c3, c4, c5 = channels

        self.enc1 = _conv_block(in_channels, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.enc4 = _conv_block(c3, c4)
        self.bottleneck = _conv_block(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.head = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        x = self.up4(b, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        x = self.head(x)

        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


def _weight_enum(backbone: str):
    name = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
    }[backbone]

    return getattr(
        models,
        name,
    )


def _build_resnet(
    backbone: str,
    pretrained: bool,
) -> nn.Module:
    constructor = getattr(
        models,
        backbone,
    )

    if not pretrained:
        return constructor(
            weights=None
        )

    try:
        return constructor(
            weights=_weight_enum(backbone).DEFAULT
        )
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained weights for {backbone}: {exc}. "
            "Falling back to random initialization."
        )
        return constructor(
            weights=None
        )


def _replace_first_conv(
    encoder: nn.Module,
    in_channels: int,
) -> None:
    if in_channels == 3:
        return

    old = encoder.conv1

    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    with torch.no_grad():
        mean_weight = old.weight.mean(
            dim=1,
            keepdim=True,
        )
        new.weight.copy_(
            mean_weight.repeat(
                1,
                in_channels,
                1,
                1,
            )
        )

    encoder.conv1 = new


class ResNetUNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in RESNET_CHANNELS:
            raise ValueError(
                f"Unsupported U-Net backbone '{backbone}'. "
                f"Available: {list(RESNET_CHANNELS)}"
            )

        encoder = _build_resnet(
            backbone,
            pretrained,
        )

        _replace_first_conv(
            encoder,
            in_channels,
        )

        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        c0, c1, c2, c3, c4 = (
            RESNET_CHANNELS[
                backbone
            ]
        )

        self.up3 = UpBlock(c4, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.up0 = UpBlock(64, c0, 64)

        self.refine = _conv_block(
            64,
            64,
        )

        self.head = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.layer1(
            self.maxpool(x0)
        )
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.refine(x)

        return self.head(x)


def _resolve_backbone(
    experiment_config,
    model_config,
    backbone_info,
) -> str:
    experiment_config = experiment_config or {}
    model_config = model_config or {}
    backbone_info = backbone_info or {}

    return str(
        experiment_config.get("backbone")
        or backbone_info.get("name")
        or model_config.get("default_backbone")
        or "vanilla_unet"
    ).strip().lower()


def _resolve_pretrained(
    backbone: str,
    model_config,
    backbone_info,
) -> bool:
    model_config = model_config or {}
    backbone_info = backbone_info or {}

    if "pretrained" in backbone_info:
        return bool(
            backbone_info[
                "pretrained"
            ]
        )

    supported = model_config.get(
        "supported_backbones",
        {},
    )

    if isinstance(
        supported,
        dict,
    ):
        info = supported.get(
            backbone,
            {},
        )

        if (
            isinstance(
                info,
                dict,
            )
            and "pretrained"
            in info
        ):
            return bool(
                info[
                    "pretrained"
                ]
            )

    return (
        backbone
        != "vanilla_unet"
    )


def build_unet(
    experiment_config=None,
    model_config=None,
    backbone_info=None,
    **kwargs,
):
    experiment_config = experiment_config or {}
    model_config = model_config or {}

    backbone = _resolve_backbone(
        experiment_config,
        model_config,
        backbone_info,
    )

    in_channels = int(
        experiment_config.get(
            "in_channels",
            model_config.get(
                "in_channels",
                3,
            ),
        )
    )

    out_channels = int(
        experiment_config.get(
            "output_channels",
            model_config.get(
                "output_channels",
                1,
            ),
        )
    )

    if backbone == "vanilla_unet":
        channels = (
            model_config
            .get(
                "architecture",
                {},
            )
            .get(
                "vanilla_channels",
                [
                    64,
                    128,
                    256,
                    512,
                    1024,
                ],
            )
        )

        return VanillaUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(
                int(value)
                for value
                in channels
            ),
        )

    return ResNetUNet(
        backbone=backbone,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=_resolve_pretrained(
            backbone,
            model_config,
            backbone_info,
        ),
    )


build_model = build_unet
get_model = build_unet

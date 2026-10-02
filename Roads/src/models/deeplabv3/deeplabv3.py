"""
Roads/src/models/deeplabv3.py

DeepLabV3+ for Road-surface segmentation.

Supported backbones
-------------------
- resnet50
- resnet101
- resnet152

The decoder uses low-level layer1 features and returns one-channel logits
at the original image size.
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
        "torchvision is required for Road DeepLabV3+."
    ) from exc


SUPPORTED_BACKBONES = (
    "resnet50",
    "resnet101",
    "resnet152",
)


def _weight_enum(backbone: str):
    name = {
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
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported DeepLabV3+ backbone '{backbone}'. "
            f"Available: {list(SUPPORTED_BACKBONES)}"
        )

    constructor = getattr(
        models,
        backbone,
    )

    kwargs = {
        "replace_stride_with_dilation": [
            False,
            False,
            True,
        ],
    }

    if not pretrained:
        return constructor(
            weights=None,
            **kwargs,
        )

    try:
        return constructor(
            weights=_weight_enum(
                backbone
            ).DEFAULT,
            **kwargs,
        )
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained weights for {backbone}: {exc}. "
            "Falling back to random initialization."
        )
        return constructor(
            weights=None,
            **kwargs,
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


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )


class ASPPPooling(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(
                1
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates=(6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        branches = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
                nn.ReLU(
                    inplace=True
                ),
            )
        ]

        for rate in rates:
            branches.append(
                ASPPConv(
                    in_channels,
                    out_channels,
                    int(rate),
                )
            )

        branches.append(
            ASPPPooling(
                in_channels,
                out_channels,
            )
        )

        self.branches = nn.ModuleList(
            branches
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels
                * len(branches),
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Dropout(
                float(dropout)
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            branch(x)
            for branch
            in self.branches
        ]

        return self.project(
            torch.cat(
                features,
                dim=1,
            )
        )


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        aspp_channels: int = 256,
        aspp_rates=(6, 12, 18),
        low_level_channels: int = 48,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

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
            encoder.maxpool,
        )

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.low_projection = nn.Sequential(
            nn.Conv2d(
                256,
                low_level_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                low_level_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.aspp = ASPP(
            in_channels=2048,
            out_channels=aspp_channels,
            rates=tuple(
                int(value)
                for value
                in aspp_rates
            ),
            dropout=dropout,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                aspp_channels
                + low_level_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                decoder_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                decoder_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Dropout(
                float(dropout)
            ),
        )

        self.classifier = nn.Conv2d(
            decoder_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.stem(x)

        low = self.layer1(x)

        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)

        high = self.aspp(high)

        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low = self.low_projection(
            low
        )

        x = torch.cat(
            [
                high,
                low,
            ],
            dim=1,
        )

        x = self.decoder(x)

        x = self.classifier(x)

        return F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


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
        or "resnet50"
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

    return True


def build_deeplabv3(
    experiment_config=None,
    model_config=None,
    backbone_info=None,
    **kwargs,
):
    experiment_config = experiment_config or {}
    model_config = model_config or {}

    architecture = model_config.get(
        "architecture",
        {},
    )

    backbone = _resolve_backbone(
        experiment_config,
        model_config,
        backbone_info,
    )

    return DeepLabV3Plus(
        backbone=backbone,
        in_channels=int(
            experiment_config.get(
                "in_channels",
                model_config.get(
                    "in_channels",
                    3,
                ),
            )
        ),
        out_channels=int(
            experiment_config.get(
                "output_channels",
                model_config.get(
                    "output_channels",
                    1,
                ),
            )
        ),
        pretrained=_resolve_pretrained(
            backbone,
            model_config,
            backbone_info,
        ),
        aspp_channels=int(
            architecture.get(
                "aspp_channels",
                256,
            )
        ),
        aspp_rates=architecture.get(
            "aspp_rates",
            [
                6,
                12,
                18,
            ],
        ),
        low_level_channels=int(
            architecture.get(
                "low_level_channels",
                48,
            )
        ),
        decoder_channels=int(
            architecture.get(
                "decoder_channels",
                256,
            )
        ),
        dropout=float(
            architecture.get(
                "dropout",
                0.1,
            )
        ),
    )


build_deeplab = build_deeplabv3
build_model = build_deeplabv3
get_model = build_deeplabv3

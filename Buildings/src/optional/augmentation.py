from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as TF


class BuildingAugmentation:
    """
    Shared augmentation pipeline for building-extraction models.

    Supported models:
        - U-Net
        - DeepLabV3
        - Mask R-CNN

    Expected inputs:
        image : torch.Tensor [C, H, W], float, usually in [0, 1]
        mask  : torch.Tensor [H, W] or [1, H, W]

    Notes:
        - Geometric transforms are always applied identically to image and mask.
        - Photometric transforms are applied to the image only.
        - For Mask R-CNN, apply augmentation to the binary mask BEFORE
          connected-components / instance-target generation.
        - Validation and test data should NOT use augmentation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = bool(config.get("enabled", False))

        geometric = config.get("geometric", {})
        photometric = config.get("photometric", {})

        # --------------------------------------------------
        # Geometric augmentation
        # --------------------------------------------------
        hflip = geometric.get("horizontal_flip", {})
        self.hflip_enabled = bool(hflip.get("enabled", True))
        self.hflip_prob = float(hflip.get("probability", 0.5))

        vflip = geometric.get("vertical_flip", {})
        self.vflip_enabled = bool(vflip.get("enabled", True))
        self.vflip_prob = float(vflip.get("probability", 0.5))

        rotation = geometric.get("rotation_90", {})
        self.rotation_enabled = bool(rotation.get("enabled", True))
        self.rotation_prob = float(rotation.get("probability", 0.5))
        self.rotation_angles = list(rotation.get("angles", [90, 180, 270]))

        # --------------------------------------------------
        # Photometric augmentation
        # --------------------------------------------------
        brightness = photometric.get("brightness", {})
        self.brightness_enabled = bool(brightness.get("enabled", True))
        self.brightness_prob = float(brightness.get("probability", 0.3))
        self.brightness_min = float(brightness.get("min_factor", 0.90))
        self.brightness_max = float(brightness.get("max_factor", 1.10))

        contrast = photometric.get("contrast", {})
        self.contrast_enabled = bool(contrast.get("enabled", True))
        self.contrast_prob = float(contrast.get("probability", 0.3))
        self.contrast_min = float(contrast.get("min_factor", 0.90))
        self.contrast_max = float(contrast.get("max_factor", 1.10))

        noise = photometric.get("gaussian_noise", {})
        self.noise_enabled = bool(noise.get("enabled", False))
        self.noise_prob = float(noise.get("probability", 0.15))
        self.noise_std = float(noise.get("std", 0.01))

        self._validate_config()

    def _validate_config(self) -> None:
        probabilities = {
            "horizontal_flip": self.hflip_prob,
            "vertical_flip": self.vflip_prob,
            "rotation_90": self.rotation_prob,
            "brightness": self.brightness_prob,
            "contrast": self.contrast_prob,
            "gaussian_noise": self.noise_prob,
        }

        for name, probability in probabilities.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{name} probability must be between 0 and 1, got {probability}"
                )

        valid_angles = {90, 180, 270}
        invalid_angles = [
            int(angle)
            for angle in self.rotation_angles
            if int(angle) not in valid_angles
        ]

        if invalid_angles:
            raise ValueError(
                "rotation_90 angles may only contain 90, 180, or 270. "
                f"Invalid: {invalid_angles}"
            )

        if self.brightness_min <= 0 or self.brightness_max <= 0:
            raise ValueError("Brightness factors must be > 0.")

        if self.contrast_min <= 0 or self.contrast_max <= 0:
            raise ValueError("Contrast factors must be > 0.")

        if self.brightness_min > self.brightness_max:
            raise ValueError("brightness min_factor cannot exceed max_factor.")

        if self.contrast_min > self.contrast_max:
            raise ValueError("contrast min_factor cannot exceed max_factor.")

        if self.noise_std < 0:
            raise ValueError("gaussian_noise std must be >= 0.")

    @staticmethod
    def _ensure_tensor(
        value: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor before augmentation. "
                f"Received: {type(value)}"
            )

        return value

    @staticmethod
    def _mask_spatial_shape(mask: torch.Tensor) -> Tuple[int, int]:
        if mask.ndim == 2:
            return int(mask.shape[0]), int(mask.shape[1])

        if mask.ndim == 3:
            return int(mask.shape[-2]), int(mask.shape[-1])

        raise ValueError(
            f"Mask must have shape [H,W] or [C,H,W], got {tuple(mask.shape)}"
        )

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        image = self._ensure_tensor(image, "image")
        mask = self._ensure_tensor(mask, "mask")

        if not self.enabled:
            return image, mask

        if image.ndim != 3:
            raise ValueError(
                f"Image must have shape [C,H,W], got {tuple(image.shape)}"
            )

        image_h, image_w = int(image.shape[-2]), int(image.shape[-1])
        mask_h, mask_w = self._mask_spatial_shape(mask)

        if (image_h, image_w) != (mask_h, mask_w):
            raise ValueError(
                "Image and mask spatial sizes do not match before augmentation: "
                f"image={(image_h, image_w)}, mask={(mask_h, mask_w)}"
            )

        # ==================================================
        # 1. Horizontal flip
        # ==================================================
        if self.hflip_enabled and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # ==================================================
        # 2. Vertical flip
        # ==================================================
        if self.vflip_enabled and random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # ==================================================
        # 3. 90-degree rotation
        # ==================================================
        if (
            self.rotation_enabled
            and self.rotation_angles
            and random.random() < self.rotation_prob
        ):
            angle = int(random.choice(self.rotation_angles))
            k = angle // 90

            # Current thesis chips are square. For safety, skip 90/270
            # on non-square inputs because dimensions would swap.
            if image_h == image_w or angle == 180:
                image = torch.rot90(image, k=k, dims=(-2, -1))
                mask = torch.rot90(mask, k=k, dims=(-2, -1))

        # ==================================================
        # 4. Brightness — image only
        # ==================================================
        if (
            self.brightness_enabled
            and random.random() < self.brightness_prob
        ):
            factor = random.uniform(
                self.brightness_min,
                self.brightness_max,
            )
            image = TF.adjust_brightness(image, factor)

        # ==================================================
        # 5. Contrast — image only
        # ==================================================
        if (
            self.contrast_enabled
            and random.random() < self.contrast_prob
        ):
            factor = random.uniform(
                self.contrast_min,
                self.contrast_max,
            )
            image = TF.adjust_contrast(image, factor)

        # ==================================================
        # 6. Mild Gaussian noise — image only, optional
        # ==================================================
        if self.noise_enabled and random.random() < self.noise_prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Keep image valid after photometric transforms.
        if image.is_floating_point():
            image = torch.clamp(image, 0.0, 1.0)

        # torch.rot90 / flips can produce non-contiguous tensors.
        image = image.contiguous()
        mask = mask.contiguous()

        return image, mask


def load_augmentation_config(
    config_path: str | Path,
) -> Dict[str, Any]:
    """
    Read and return augmentation.json.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Augmentation config not found:\n{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Augmentation config must contain a JSON object: {config_path}"
        )

    return config


def build_augmentation(
    config_path: str | Path,
    enabled_override: Optional[bool] = None,
) -> Optional[BuildingAugmentation]:
    """
    Factory used by all model trainers/datasets.

    Returns:
        BuildingAugmentation instance if enabled,
        otherwise None.
    """
    config = load_augmentation_config(config_path)

    if enabled_override is not None:
        config = dict(config)
        config["enabled"] = bool(enabled_override)

    if not bool(config.get("enabled", False)):
        return None

    return BuildingAugmentation(config)


def describe_augmentation(
    config_path: str | Path,
) -> None:
    """
    Print a compact description for experiment logs.
    """
    config = load_augmentation_config(config_path)

    print("========== AUGMENTATION CONFIG ==========")
    print("Enabled:", bool(config.get("enabled", False)))

    geometric = config.get("geometric", {})
    photometric = config.get("photometric", {})

    for name, settings in geometric.items():
        print(
            f"{name}: "
            f"enabled={settings.get('enabled', False)}, "
            f"p={settings.get('probability', '-')}"
        )

    for name, settings in photometric.items():
        print(
            f"{name}: "
            f"enabled={settings.get('enabled', False)}, "
            f"p={settings.get('probability', '-')}"
        )

    print("=========================================")

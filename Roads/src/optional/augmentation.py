"""
Optional Train-only augmentation for Approach 2 - Road Extraction.

Scientific rules
----------------
Geometric transformations:
    MUST be applied identically to:
        - image
        - Road mask

Photometric transformations:
    MUST be applied only to:
        - image

Validation:
    augmentation OFF

Final testing:
    augmentation OFF

Implemented
-----------
Geometric:
    - Horizontal flip
    - Vertical flip
    - Random 90-degree rotation

Photometric:
    - Brightness
    - Contrast
    - Gaussian noise

The baseline configuration uses:
    horizontal flip  p=0.5
    vertical flip    p=0.5
    rotation 90      p=0.5
    brightness       p=0.3
    contrast         p=0.3
    gaussian noise   disabled

This implementation has no Albumentations dependency, keeping
augmentation genuinely optional.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_dict(
    value: Any,
) -> Dict[str, Any]:
    """
    Return value when it is a dictionary, otherwise {}.
    """

    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )

    return {}


def _probability(
    config: Dict[str, Any],
    default: float = 0.0,
) -> float:
    """
    Resolve transform probability.
    """

    probability = float(
        config.get(
            "p",
            config.get(
                "probability",
                default,
            ),
        )
    )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise ValueError(
            "Augmentation probability must be between 0 and 1.\n"
            f"Received: {probability}"
        )

    return probability


def _transform_enabled(
    config: Dict[str, Any],
) -> bool:
    """
    Determine whether one augmentation operation is active.
    """

    if not config:
        return False

    if "enabled" in config:

        if not bool(
            config[
                "enabled"
            ]
        ):
            return False

    return (
        _probability(
            config,
            default=0.0,
        )
        > 0.0
    )


def _limit(
    config: Dict[str, Any],
    name: str,
    default: float,
) -> float:
    """
    Read a positive augmentation magnitude.
    """

    value = float(
        config.get(
            name,
            config.get(
                "limit",
                default,
            ),
        )
    )

    if value < 0:

        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


def _validate_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Verify image and mask spatial dimensions.
    """

    if image.ndim != 3:

        raise ValueError(
            "Road augmentation expects image shape H x W x C.\n"
            f"Received: {image.shape}"
        )

    if image.shape[
        2
    ] != 3:

        raise ValueError(
            "Road augmentation expects exactly 3 image channels.\n"
            f"Received: {image.shape[2]}"
        )

    if mask.ndim not in {
        2,
        3,
    }:

        raise ValueError(
            "Road mask must be H x W or H x W x 1.\n"
            f"Received: {mask.shape}"
        )

    if (
        image.shape[
            0
        ]
        != mask.shape[
            0
        ]
        or image.shape[
            1
        ]
        != mask.shape[
            1
        ]
    ):

        raise ValueError(
            "Image and Road mask spatial dimensions do not match.\n"
            f"Image: {image.shape}\n"
            f"Mask : {mask.shape}"
        )


# ---------------------------------------------------------------------
# Main augmentation class
# ---------------------------------------------------------------------

class RoadTrainingAugmentation:
    """
    Train-only augmentation for binary Road segmentation.

    The callable returns an Albumentations-like dictionary:

        {
            "image": augmented_image,
            "mask": augmented_mask
        }

    so it works directly with RoadSegmentationDataset.
    """

    def __init__(
        self,
        config: Optional[
            Dict[str, Any]
        ] = None,
    ):
        self.config = dict(
            config
            or {}
        )

        self.geometric = _as_dict(
            self.config.get(
                "geometric"
            )
        )

        self.photometric = _as_dict(
            self.config.get(
                "photometric"
            )
        )

        # -------------------------------------------------------------
        # Supported geometric operations
        # -------------------------------------------------------------

        self.horizontal_flip = _as_dict(
            self.geometric.get(
                "horizontal_flip"
            )
        )

        self.vertical_flip = _as_dict(
            self.geometric.get(
                "vertical_flip"
            )
        )

        self.rotation_90 = _as_dict(
            self.geometric.get(
                "rotation_90",
                self.geometric.get(
                    "random_rotation_90"
                ),
            )
        )

        # -------------------------------------------------------------
        # Supported photometric operations
        # -------------------------------------------------------------

        self.brightness = _as_dict(
            self.photometric.get(
                "brightness"
            )
        )

        self.contrast = _as_dict(
            self.photometric.get(
                "contrast"
            )
        )

        self.gaussian_noise = _as_dict(
            self.photometric.get(
                "gaussian_noise"
            )
        )

        self._validate_configuration()


    # -----------------------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------------------

    def _validate_configuration(
        self,
    ) -> None:
        """
        Prevent silently ignoring an enabled transform.

        More complex transforms can be implemented later, but if one is
        explicitly enabled now we stop rather than pretending it ran.
        """

        unsupported_geometric = {
            "random_rotation",
            "arbitrary_rotation",
            "scale",
            "random_scale",
            "elastic",
            "elastic_transform",
        }

        for name in unsupported_geometric:

            transform = _as_dict(
                self.geometric.get(
                    name
                )
            )

            if _transform_enabled(
                transform
            ):

                raise NotImplementedError(
                    f"Road augmentation transform {name!r} is enabled, "
                    "but it has not been implemented yet.\n\n"
                    "Disable it or implement it explicitly before "
                    "starting the experiment."
                )

        # Trigger validation of probabilities now.
        for transform in (
            self.horizontal_flip,
            self.vertical_flip,
            self.rotation_90,
            self.brightness,
            self.contrast,
            self.gaussian_noise,
        ):

            if transform:

                _probability(
                    transform,
                    default=0.0,
                )


    # -----------------------------------------------------------------
    # Random decision
    # -----------------------------------------------------------------

    @staticmethod
    def _apply(
        transform_config: Dict[str, Any],
    ) -> bool:
        """
        Randomly decide whether one operation should run.
        """

        if not _transform_enabled(
            transform_config
        ):

            return False

        probability = _probability(
            transform_config
        )

        return bool(
            np.random.random()
            < probability
        )


    # -----------------------------------------------------------------
    # Geometric transforms
    # -----------------------------------------------------------------

    def _apply_geometric(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Apply exactly the same geometry to image and mask.
        """

        # -------------------------------------------------------------
        # Horizontal flip
        # -------------------------------------------------------------

        if self._apply(
            self.horizontal_flip
        ):

            image = np.flip(
                image,
                axis=1,
            )

            mask = np.flip(
                mask,
                axis=1,
            )

        # -------------------------------------------------------------
        # Vertical flip
        # -------------------------------------------------------------

        if self._apply(
            self.vertical_flip
        ):

            image = np.flip(
                image,
                axis=0,
            )

            mask = np.flip(
                mask,
                axis=0,
            )

        # -------------------------------------------------------------
        # Random 90° rotation
        #
        # When activated:
        #     k = 1 → 90°
        #     k = 2 → 180°
        #     k = 3 → 270°
        #
        # No interpolation occurs, which is ideal for binary masks.
        # -------------------------------------------------------------

        if self._apply(
            self.rotation_90
        ):

            k = int(
                np.random.randint(
                    1,
                    4,
                )
            )

            image = np.rot90(
                image,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

            mask = np.rot90(
                mask,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

        return (
            image,
            mask,
        )


    # -----------------------------------------------------------------
    # Photometric transforms
    # -----------------------------------------------------------------

    def _apply_brightness(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Random multiplicative brightness adjustment.

        With limit = 0.15:

            factor ∈ [0.85, 1.15]
        """

        if not self._apply(
            self.brightness
        ):

            return image

        brightness_limit = _limit(
            self.brightness,
            "brightness_limit",
            default=0.15,
        )

        factor = float(
            np.random.uniform(
                1.0
                - brightness_limit,

                1.0
                + brightness_limit,
            )
        )

        image = (
            image.astype(
                np.float32
            )
            * factor
        )

        return image


    def _apply_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Random contrast adjustment around per-channel image means.

        With limit = 0.15:

            factor ∈ [0.85, 1.15]
        """

        if not self._apply(
            self.contrast
        ):

            return image

        contrast_limit = _limit(
            self.contrast,
            "contrast_limit",
            default=0.15,
        )

        factor = float(
            np.random.uniform(
                1.0
                - contrast_limit,

                1.0
                + contrast_limit,
            )
        )

        image = image.astype(
            np.float32,
            copy=False,
        )

        channel_mean = np.mean(
            image,
            axis=(
                0,
                1,
            ),
            keepdims=True,
        )

        image = (
            (
                image
                - channel_mean
            )
            * factor
            + channel_mean
        )

        return image


    def _apply_gaussian_noise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Optional Gaussian image noise.

        Baseline configuration:
            disabled

        Default standard deviation:
            up to 5% of the uint8 range.
        """

        if not self._apply(
            self.gaussian_noise
        ):

            return image

        std_limit = float(
            self.gaussian_noise.get(
                "std_limit",
                self.gaussian_noise.get(
                    "noise_std_limit",
                    0.05,
                ),
            )
        )

        if std_limit < 0:

            raise ValueError(
                "Gaussian noise std_limit cannot be negative."
            )

        maximum_std = (
            255.0
            * std_limit
            if std_limit <= 1.0
            else std_limit
        )

        noise_std = float(
            np.random.uniform(
                0.0,
                maximum_std,
            )
        )

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=image.shape,
        ).astype(
            np.float32
        )

        image = (
            image.astype(
                np.float32,
                copy=False,
            )
            + noise
        )

        return image


    def _apply_photometric(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply image-only transformations.
        """

        image = self._apply_brightness(
            image
        )

        image = self._apply_contrast(
            image
        )

        image = self._apply_gaussian_noise(
            image
        )

        return image


    # -----------------------------------------------------------------
    # Main call
    # -----------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Augment one image/mask pair.
        """

        image = np.asarray(
            image
        )

        mask = np.asarray(
            mask
        )

        _validate_image_mask(
            image,
            mask,
        )

        # Keep input mask values unchanged except for spatial movement.
        mask_dtype = mask.dtype

        # -------------------------------------------------------------
        # Geometry:
        # image + mask together.
        # -------------------------------------------------------------

        image, mask = (
            self._apply_geometric(
                image=image,
                mask=mask,
            )
        )

        # -------------------------------------------------------------
        # Photometry:
        # image only.
        # -------------------------------------------------------------

        image = (
            self._apply_photometric(
                image
            )
        )

        # -------------------------------------------------------------
        # Return stable arrays
        #
        # np.flip / np.rot90 may create negative-stride views.
        # torch.from_numpy cannot safely consume those, so force
        # contiguous arrays.
        # -------------------------------------------------------------

        image = np.clip(
            image,
            0.0,
            255.0,
        ).round().astype(
            np.uint8
        )

        mask = mask.astype(
            mask_dtype,
            copy=False,
        )

        image = np.ascontiguousarray(
            image
        )

        mask = np.ascontiguousarray(
            mask
        )

        # Final defensive check.
        _validate_image_mask(
            image,
            mask,
        )

        return {
            "image":
                image,

            "mask":
                mask,
        }


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_training_augmentation(
    config: Optional[
        Dict[str, Any]
    ] = None,
) -> RoadTrainingAugmentation:
    """
    Build Train-only Road augmentation.

    Notes
    -----
    experiment.json is the master switch:

        "use_augmentation": true / false

    This factory only defines HOW augmentation behaves when the caller
    has already enabled it.

    Therefore a top-level "enabled": false inside augmentation.json
    does not override experiment.json.
    """

    return RoadTrainingAugmentation(
        config=config
    )


# ---------------------------------------------------------------------
# Description helper
# ---------------------------------------------------------------------

def describe_training_augmentation(
    config: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Produce a concise serializable augmentation description.
    """

    config = dict(
        config
        or {}
    )

    augmentation = (
        RoadTrainingAugmentation(
            config
        )
    )

    transforms = {}

    items = {
        "horizontal_flip":
            augmentation.horizontal_flip,

        "vertical_flip":
            augmentation.vertical_flip,

        "rotation_90":
            augmentation.rotation_90,

        "brightness":
            augmentation.brightness,

        "contrast":
            augmentation.contrast,

        "gaussian_noise":
            augmentation.gaussian_noise,
    }

    for (
        name,
        transform_config,
    ) in items.items():

        transforms[
            name
        ] = {
            "enabled":
                bool(
                    _transform_enabled(
                        transform_config
                    )
                ),

            "probability":
                float(
                    _probability(
                        transform_config,
                        default=0.0,
                    )
                )
                if transform_config
                else 0.0,
        }

    return {
        "scope":
            "training_only",

        "validation_augmentation":
            False,

        "final_test_augmentation":
            False,

        "geometric_image_mask_synchronized":
            True,

        "photometric_mask_modified":
            False,

        "transforms":
            transforms,
    }
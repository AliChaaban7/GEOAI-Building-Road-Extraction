"""
SAM-LoRA semantic Road segmentation model.

Approach 2 - Road Extraction

Architecture
------------
RGB Road tile
      ↓
Resize to SAM encoder size
      ↓
SAM ViT image encoder
    + Q/V LoRA
      ↓
No external prompt
      ↓
Frozen SAM prompt encoder
      ↓
Trainable SAM mask decoder
      ↓
1 Road-surface logit mask
      ↓
Resize back to original tile size

Output
------
[B, 1, H, W] logits

No sigmoid is applied inside the network.

This allows the existing Road:
- BCE + Dice loss
- threshold search
- post-processing
- final inference

to operate exactly like U-Net and DeepLabV3+.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_qkv import (
    count_lora_parameters,
    inject_lora_into_sam_image_encoder,
)


# ---------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------

def _get_sam_registry():
    """
    Lazy import so U-Net/DeepLab do not depend on segment-anything.
    """

    try:

        from segment_anything import (
            sam_model_registry,
        )

    except ImportError as exc:

        raise RuntimeError(
            "Road SAM-LoRA requires the 'segment_anything' package.\n\n"
            "The package is already used by the Buildings SAM-LoRA "
            "implementation, but it must also be available in the "
            "environment used for Roads."
        ) from exc

    return sam_model_registry


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def _as_dict(
    value,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):

        return dict(
            value
        )

    return {}


def _first_nonempty(
    *values,
):

    for value in values:

        if value is None:

            continue

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:

                continue

        return value

    return None


def resolve_sam_lora_settings(
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve Road SAM-LoRA architecture settings.
    """

    model_config = deepcopy(
        model_config
        or {}
    )

    architecture = _as_dict(
        model_config.get(
            "architecture"
        )
    )

    lora = _as_dict(
        model_config.get(
            "lora"
        )
    )

    freezing = _as_dict(
        model_config.get(
            "freezing"
        )
    )

    backbone = str(
        _first_nonempty(
            model_config.get(
                "backbone"
            ),

            model_config.get(
                "default_backbone"
            ),

            architecture.get(
                "backbone"
            ),

            "vit_b",
        )
    ).lower()

    aliases = {
        "vit-b":
            "vit_b",

        "vitb":
            "vit_b",

        "vit_b":
            "vit_b",
    }

    backbone = aliases.get(
        backbone,
        backbone,
    )

    # --------------------------------------------------------------
    # Road baseline is intentionally ViT-B.
    # --------------------------------------------------------------

    if backbone != "vit_b":

        raise ValueError(
            "Current Road SAM-LoRA baseline is fixed to ViT-B.\n"
            f"Received backbone: {backbone}"
        )

    rank = int(
        _first_nonempty(
            lora.get(
                "rank"
            ),

            model_config.get(
                "lora_rank"
            ),

            8,
        )
    )

    alpha = float(
        _first_nonempty(
            lora.get(
                "alpha"
            ),

            model_config.get(
                "lora_alpha"
            ),

            16.0,
        )
    )

    dropout = float(
        _first_nonempty(
            lora.get(
                "dropout"
            ),

            0.0,
        )
    )

    targets = _first_nonempty(
        lora.get(
            "targets"
        ),

        lora.get(
            "target_modules"
        ),

        [
            "q",
            "v",
        ],
    )

    targets = [
        str(
            value
        ).lower()
        for value
        in targets
    ]

    enable_q = (
        "q"
        in targets
        or "query"
        in targets
    )

    enable_v = (
        "v"
        in targets
        or "value"
        in targets
    )

    target_blocks = _first_nonempty(
        lora.get(
            "blocks"
        ),

        lora.get(
            "target_blocks"
        ),

        "all",
    )

    train_mask_decoder = bool(
        _first_nonempty(
            freezing.get(
                "train_mask_decoder"
            ),

            model_config.get(
                "train_mask_decoder"
            ),

            True,
        )
    )

    freeze_prompt_encoder = bool(
        _first_nonempty(
            freezing.get(
                "freeze_prompt_encoder"
            ),

            True,
        )
    )

    freeze_base_image_encoder = bool(
        _first_nonempty(
            freezing.get(
                "freeze_image_encoder"
            ),

            freezing.get(
                "freeze_base_image_encoder"
            ),

            True,
        )
    )

    return {
        "backbone":
            backbone,

        "rank":
            rank,

        "alpha":
            alpha,

        "dropout":
            dropout,

        "enable_q":
            enable_q,

        "enable_v":
            enable_v,

        "target_blocks":
            target_blocks,

        "train_mask_decoder":
            train_mask_decoder,

        "freeze_prompt_encoder":
            freeze_prompt_encoder,

        "freeze_base_image_encoder":
            freeze_base_image_encoder,
    }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class RoadSAMLoRA(nn.Module):
    """
    Prompt-free SAM-LoRA Road-surface segmentation model.
    """

    def __init__(
        self,
        sam: nn.Module,
        settings: Dict[str, Any],
    ):
        super().__init__()

        self.sam = sam

        self.settings = deepcopy(
            settings
        )

        # SAM ViT-B encoder input size is normally 1024.
        self.encoder_size = int(
            getattr(
                self.sam.image_encoder,
                "img_size",
                1024,
            )
        )


    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images:
            [B, 3, H, W] float tensor in [0,1].

        Returns
        -------
        [B, 1, H, W] raw logits.
        """

        if images.ndim != 4:

            raise ValueError(
                "SAM-LoRA expects [B,3,H,W]."
            )

        if images.shape[
            1
        ] != 3:

            raise ValueError(
                "SAM-LoRA requires exactly 3 RGB channels."
            )

        original_size = (
            int(
                images.shape[
                    -2
                ]
            ),
            int(
                images.shape[
                    -1
                ]
            ),
        )

        # -------------------------------------------------------------
        # Resize to SAM encoder input resolution.
        # -------------------------------------------------------------

        resized = F.interpolate(
            images,

            size=(
                self.encoder_size,
                self.encoder_size,
            ),

            mode="bilinear",

            align_corners=False,
        )

        # -------------------------------------------------------------
        # SAM normalization.
        #
        # segment_anything stores pixel_mean/std in 0-255 units.
        # Our Road dataset gives [0,1], therefore multiply by 255.
        # -------------------------------------------------------------

        resized_255 = (
            resized
            * 255.0
        )

        pixel_mean = self.sam.pixel_mean.to(
            dtype=resized_255.dtype,
            device=resized_255.device,
        )

        pixel_std = self.sam.pixel_std.to(
            dtype=resized_255.dtype,
            device=resized_255.device,
        )

        normalized = (
            resized_255
            - pixel_mean
        ) / pixel_std

        # -------------------------------------------------------------
        # Image encoder.
        # -------------------------------------------------------------

        image_embeddings = (
            self.sam.image_encoder(
                normalized
            )
        )

        batch_size = int(
            images.shape[
                0
            ]
        )

        masks = []

        # -------------------------------------------------------------
        # SAM mask decoder currently expects prompt embeddings per
        # image. We therefore decode each image independently.
        # -------------------------------------------------------------

        for index in range(
            batch_size
        ):

            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            low_resolution_masks, _ = (
                self.sam.mask_decoder(
                    image_embeddings=(
                        image_embeddings[
                            index:
                            index + 1
                        ]
                    ),

                    image_pe=(
                        self.sam
                        .prompt_encoder
                        .get_dense_pe()
                    ),

                    sparse_prompt_embeddings=(
                        sparse_embeddings
                    ),

                    dense_prompt_embeddings=(
                        dense_embeddings
                    ),

                    multimask_output=False,
                )
            )

            masks.append(
                low_resolution_masks
            )

        logits = torch.cat(
            masks,
            dim=0,
        )

        # -------------------------------------------------------------
        # Bring SAM mask decoder output back to Road tile resolution.
        # -------------------------------------------------------------

        logits = F.interpolate(
            logits,

            size=original_size,

            mode="bilinear",

            align_corners=False,
        )

        return logits


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def build_sam_lora(
    model_config: Dict[str, Any],
    sam_checkpoint_path: str | Path,
) -> RoadSAMLoRA:
    """
    Build Road SAM-LoRA from the official Meta SAM checkpoint.

    IMPORTANT
    ---------
    sam_checkpoint_path must point to the OFFICIAL pretrained SAM
    checkpoint.

    It must never silently point to a trained Buildings checkpoint.
    """

    checkpoint_path = Path(
        sam_checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Official SAM checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    settings = (
        resolve_sam_lora_settings(
            model_config
        )
    )

    registry = (
        _get_sam_registry()
    )

    backbone = settings[
        "backbone"
    ]

    if backbone not in registry:

        raise KeyError(
            f"SAM registry does not contain {backbone!r}."
        )

    sam = registry[
        backbone
    ](
        checkpoint=str(
            checkpoint_path
        )
    )

    # --------------------------------------------------------------
    # Freeze complete pretrained SAM first.
    # --------------------------------------------------------------

    for parameter in (
        sam.parameters()
    ):

        parameter.requires_grad = False

    # --------------------------------------------------------------
    # Inject LoRA into the image encoder.
    # --------------------------------------------------------------

    selected_blocks = (
        inject_lora_into_sam_image_encoder(
            image_encoder=(
                sam.image_encoder
            ),

            rank=settings[
                "rank"
            ],

            alpha=settings[
                "alpha"
            ],

            dropout=settings[
                "dropout"
            ],

            target_blocks=(
                settings[
                    "target_blocks"
                ]
            ),

            enable_q=(
                settings[
                    "enable_q"
                ]
            ),

            enable_v=(
                settings[
                    "enable_v"
                ]
            ),
        )
    )

    # --------------------------------------------------------------
    # LoRA adapter parameters must be trainable.
    # --------------------------------------------------------------

    for module in (
        sam.image_encoder.modules()
    ):

        if (
            module.__class__.__name__
            == "LoRAQKV"
        ):

            for name in (
                "q_A",
                "q_B",
                "v_A",
                "v_B",
            ):

                layer = getattr(
                    module,
                    name,
                    None,
                )

                if layer is None:

                    continue

                for parameter in (
                    layer.parameters()
                ):

                    parameter.requires_grad = True

    # --------------------------------------------------------------
    # Prompt encoder stays frozen.
    # --------------------------------------------------------------

    if settings[
        "freeze_prompt_encoder"
    ]:

        for parameter in (
            sam.prompt_encoder.parameters()
        ):

            parameter.requires_grad = False

    # --------------------------------------------------------------
    # Mask decoder trainable.
    # --------------------------------------------------------------

    if settings[
        "train_mask_decoder"
    ]:

        for parameter in (
            sam.mask_decoder.parameters()
        ):

            parameter.requires_grad = True

    model = RoadSAMLoRA(
        sam=sam,
        settings=settings,
    )

    model.selected_lora_blocks = (
        selected_blocks
    )

    model.official_sam_checkpoint = str(
        checkpoint_path
    )

    return model


# ---------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------

def sam_lora_parameter_summary(
    model: RoadSAMLoRA,
) -> Dict[str, Any]:
    """
    Report SAM-LoRA parameter counts.
    """

    total = int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )
    )

    trainable = int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
    )

    lora_parameters = int(
        count_lora_parameters(
            model.sam.image_encoder
        )
    )

    mask_decoder_trainable = int(
        sum(
            parameter.numel()
            for parameter
            in model.sam.mask_decoder.parameters()
            if parameter.requires_grad
        )
    )

    return {
        "total_parameters":
            total,

        "trainable_parameters":
            trainable,

        "frozen_parameters":
            int(
                total
                - trainable
            ),

        "trainable_percentage":
            (
                float(
                    trainable
                    / total
                    * 100.0
                )
                if total > 0
                else 0.0
            ),

        "lora_parameters":
            lora_parameters,

        "mask_decoder_trainable_parameters":
            mask_decoder_trainable,

        "backbone":
            model.settings[
                "backbone"
            ],

        "lora_rank":
            model.settings[
                "rank"
            ],

        "lora_alpha":
            model.settings[
                "alpha"
            ],

        "lora_blocks":
            list(
                model.selected_lora_blocks
            ),
    }
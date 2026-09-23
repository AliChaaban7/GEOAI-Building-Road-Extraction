"""
Buildings/src/models/sam_lora/model.py

Training wrapper for SAM-LoRA building segmentation.

This module converts the prompt-based Segment Anything Model into a
semantic building-segmentation model suitable for the thesis training
pipeline.

Input
-----
Tensor:

    [B, 3, H, W]

Output
------
Raw building-mask logits:

    [B, 1, H, W]

The output is intentionally NOT passed through sigmoid because training
losses such as BCEWithLogitsLoss expect raw logits.

Architecture
------------
Input RGB tile
    ↓
Resize longest side to SAM image-encoder size
    ↓
SAM normalization + padding
    ↓
SAM image encoder
        - pretrained SAM weights frozen
        - Q/V LoRA trainable
    ↓
Prompt encoder
        - no point prompt
        - no box prompt
        - no mask prompt
        - frozen
    ↓
Mask decoder
        - trainable
    ↓
Single building-mask logit
    ↓
Resize to original tile size
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


InputRange = Literal[
    "0_1",
    "0_255",
]


class SAMLoRABuildingSegmenter(nn.Module):
    """
    Prompt-free SAM-LoRA wrapper for automatic building segmentation.

    Parameters
    ----------
    sam:
        SAM model already configured by builder.py.

        Expected architecture:

            image_encoder
                original SAM weights frozen
                LoRA adapters trainable

            prompt_encoder
                frozen

            mask_decoder
                trainable

    input_range:
        Numerical range used by the dataset loader.

        "0_1":
            Images are floating-point values in [0, 1].

            They are multiplied by 255 before SAM preprocessing.

        "0_255":
            Images are already represented in SAM's expected
            0-255 intensity scale.

    align_corners:
        align_corners argument used during bilinear interpolation.

        False is the correct default for image resizing.
    """

    def __init__(
        self,
        sam: nn.Module,
        input_range: InputRange = "0_1",
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        self.sam = sam
        self.input_range = self._validate_input_range(
            input_range
        )
        self.align_corners = bool(
            align_corners
        )

        self._validate_sam()

        self.encoder_image_size = int(
            self.sam.image_encoder.img_size
        )

        if self.encoder_image_size <= 0:
            raise ValueError(
                "SAM image_encoder.img_size must be positive, "
                f"received {self.encoder_image_size}."
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input_range(
        input_range: str,
    ) -> InputRange:
        """
        Validate image intensity convention.
        """

        normalized = str(
            input_range
        ).strip().lower()

        supported = {
            "0_1",
            "0_255",
        }

        if normalized not in supported:
            raise ValueError(
                f"Unsupported SAM-LoRA input_range "
                f"'{input_range}'. "
                f"Supported values are: "
                f"{sorted(supported)}."
            )

        return normalized  # type: ignore[return-value]

    def _validate_sam(
        self,
    ) -> None:
        """
        Check that the provided object contains the SAM components
        required by this wrapper.
        """

        required = (
            "image_encoder",
            "prompt_encoder",
            "mask_decoder",
            "preprocess",
            "postprocess_masks",
        )

        for name in required:
            if not hasattr(
                self.sam,
                name,
            ):
                raise AttributeError(
                    "SAM-LoRA model wrapper received an incompatible "
                    "SAM model.\n"
                    f"Missing required component: '{name}'."
                )

        if not hasattr(
            self.sam.image_encoder,
            "img_size",
        ):
            raise AttributeError(
                "SAM image encoder does not expose 'img_size'."
            )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _validate_images(
        self,
        images: torch.Tensor,
    ) -> None:
        """
        Validate one training/inference image batch.
        """

        if not torch.is_tensor(
            images
        ):
            raise TypeError(
                "SAM-LoRA expects images to be a torch.Tensor, "
                f"received {type(images).__name__}."
            )

        if images.ndim != 4:
            raise ValueError(
                "SAM-LoRA expects image tensors shaped "
                "[B, C, H, W]. "
                f"Received shape: {tuple(images.shape)}."
            )

        batch_size, channels, height, width = (
            images.shape
        )

        if batch_size <= 0:
            raise ValueError(
                "SAM-LoRA received an empty image batch."
            )

        if channels != 3:
            raise ValueError(
                "Official pretrained SAM expects RGB imagery with "
                "exactly 3 channels.\n"
                f"Received C={channels}.\n\n"
                "Do not silently remove or reorder bands. "
                "If a source contains 4 bands, RGB band selection "
                "must be performed explicitly in the dataset pipeline."
            )

        if height <= 0 or width <= 0:
            raise ValueError(
                "SAM-LoRA received an invalid spatial size: "
                f"H={height}, W={width}."
            )

        if not torch.is_floating_point(
            images
        ):
            raise TypeError(
                "SAM-LoRA expects floating-point image tensors. "
                f"Received dtype={images.dtype}."
            )

        if not torch.isfinite(
            images
        ).all():
            raise ValueError(
                "SAM-LoRA input contains NaN or infinite values."
            )

    def _convert_input_range(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert dataset imagery to SAM's expected 0-255 intensity
        convention.

        SAM's own preprocess() function then applies its official
        pixel mean and standard deviation.
        """

        if self.input_range == "0_1":
            return images * 255.0

        if self.input_range == "0_255":
            return images

        raise RuntimeError(
            "Unexpected input_range state."
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _calculate_resized_size(
        self,
        height: int,
        width: int,
    ) -> Tuple[int, int]:
        """
        Resize the longest image side to SAM image_encoder.img_size
        while preserving aspect ratio.

        This reproduces the geometric principle used by SAM rather
        than stretching every image directly into a square.
        """

        longest_side = max(
            height,
            width,
        )

        scale = (
            float(self.encoder_image_size)
            / float(longest_side)
        )

        resized_height = max(
            1,
            int(
                round(
                    height * scale
                )
            ),
        )

        resized_width = max(
            1,
            int(
                round(
                    width * scale
                )
            ),
        )

        resized_height = min(
            resized_height,
            self.encoder_image_size,
        )

        resized_width = min(
            resized_width,
            self.encoder_image_size,
        )

        return (
            resized_height,
            resized_width,
        )

    def _resize_for_sam(
        self,
        images: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Tuple[int, int],
    ]:
        """
        Resize images before SAM preprocessing.
        """

        height = int(
            images.shape[-2]
        )

        width = int(
            images.shape[-1]
        )

        resized_size = (
            self._calculate_resized_size(
                height=height,
                width=width,
            )
        )

        if resized_size == (
            height,
            width,
        ):
            return (
                images,
                resized_size,
            )

        resized = F.interpolate(
            images,
            size=resized_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        return (
            resized,
            resized_size,
        )

    # ------------------------------------------------------------------
    # SAM preprocessing
    # ------------------------------------------------------------------

    def _prepare_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Tuple[int, int],
    ]:
        """
        Prepare an image batch for the SAM image encoder.

        Steps:

            dataset intensity range
                ↓
            convert to 0-255
                ↓
            resize longest side
                ↓
            SAM official mean/std normalization
                ↓
            pad to image_encoder.img_size
        """

        sam_images = self._convert_input_range(
            images
        )

        sam_images, resized_size = (
            self._resize_for_sam(
                sam_images
            )
        )

        preprocessed = self.sam.preprocess(
            sam_images
        )

        expected_height = (
            self.encoder_image_size
        )

        expected_width = (
            self.encoder_image_size
        )

        if (
            int(preprocessed.shape[-2])
            != expected_height
            or int(preprocessed.shape[-1])
            != expected_width
        ):
            raise RuntimeError(
                "Unexpected SAM preprocessing output size.\n"
                f"Expected: "
                f"({expected_height}, {expected_width})\n"
                f"Received: "
                f"{tuple(preprocessed.shape[-2:])}"
            )

        return (
            preprocessed,
            resized_size,
        )

    # ------------------------------------------------------------------
    # Prompt-free decoder
    # ------------------------------------------------------------------

    def _get_empty_prompt_embeddings(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Obtain SAM prompt embeddings without supplying any user prompt.

        For our thesis task, the model must automatically extract
        buildings from the complete tile.

        Therefore we do not supply:

            point prompts
            box prompts
            mask prompts

        The task supervision comes from the building ground-truth mask
        during training, not from interactive SAM prompts.
        """

        sparse_embeddings, dense_embeddings = (
            self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        )

        return (
            sparse_embeddings,
            dense_embeddings,
        )

    def _decode_one_embedding(
        self,
        image_embedding: torch.Tensor,
        resized_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Decode one SAM image embedding.

        SAM's official mask decoder is fundamentally organized around
        one image plus its corresponding prompt embeddings.

        We therefore encode the whole image batch together for GPU
        efficiency, then decode each embedding independently.

        This avoids incorrect cross-image prompt broadcasting.
        """

        if image_embedding.ndim != 4:
            raise ValueError(
                "Expected image embedding shape "
                "[1, C, H, W]. "
                f"Received {tuple(image_embedding.shape)}."
            )

        if int(
            image_embedding.shape[0]
        ) != 1:
            raise ValueError(
                "_decode_one_embedding() expects exactly one "
                "image embedding."
            )

        sparse_embeddings, dense_embeddings = (
            self._get_empty_prompt_embeddings()
        )

        low_resolution_masks, iou_predictions = (
            self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=(
                    self.sam.prompt_encoder.get_dense_pe()
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

        if low_resolution_masks.ndim != 4:
            raise RuntimeError(
                "Unexpected SAM mask decoder output shape: "
                f"{tuple(low_resolution_masks.shape)}."
            )

        if int(
            low_resolution_masks.shape[1]
        ) != 1:
            raise RuntimeError(
                "SAM-LoRA expects exactly one mask channel when "
                "multimask_output=False, but received "
                f"{low_resolution_masks.shape[1]}."
            )

        full_resolution_logits = (
            self.sam.postprocess_masks(
                low_resolution_masks,
                input_size=resized_size,
                original_size=original_size,
            )
        )

        return (
            full_resolution_logits,
            iou_predictions,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Run one SAM-LoRA forward pass.

        Parameters
        ----------
        images:
            Image batch:

                [B, 3, H, W]

        return_aux:
            False:
                return only building logits.

            True:
                additionally return SAM's decoder IoU predictions.

        Returns
        -------
        Default:

            logits:
                [B, 1, H, W]

        With return_aux=True:

            {
                "logits": ...,
                "iou_predictions": ...
            }

        Important
        ---------
        Returned mask values are RAW LOGITS.

        Do not apply sigmoid here.

        During training:

            BCEWithLogitsLoss(logits, target)

        During validation/inference:

            probabilities = sigmoid(logits)

        and thresholding is performed afterwards.
        """

        self._validate_images(
            images
        )

        original_size = (
            int(images.shape[-2]),
            int(images.shape[-1]),
        )

        prepared_images, resized_size = (
            self._prepare_images(
                images
            )
        )

        # --------------------------------------------------------------
        # Image encoder.
        #
        # Base SAM parameters remain frozen.
        # Gradients still flow through the trainable LoRA adapters.
        # --------------------------------------------------------------

        image_embeddings = (
            self.sam.image_encoder(
                prepared_images
            )
        )

        if image_embeddings.ndim != 4:
            raise RuntimeError(
                "Unexpected SAM image encoder output shape: "
                f"{tuple(image_embeddings.shape)}."
            )

        if int(
            image_embeddings.shape[0]
        ) != int(
            images.shape[0]
        ):
            raise RuntimeError(
                "SAM image encoder changed the batch size unexpectedly."
            )

        batch_logits = []
        batch_iou_predictions = []

        # --------------------------------------------------------------
        # Decode one image embedding at a time.
        #
        # This is intentional.
        #
        # SAM's mask decoder assumes prompt embeddings correspond to
        # one image, so blindly giving a batched image embedding can
        # create incorrect repeat/broadcast behavior.
        # --------------------------------------------------------------

        for batch_index in range(
            int(
                image_embeddings.shape[0]
            )
        ):
            image_embedding = (
                image_embeddings[
                    batch_index : batch_index + 1
                ]
            )

            logits, iou_predictions = (
                self._decode_one_embedding(
                    image_embedding=(
                        image_embedding
                    ),
                    resized_size=resized_size,
                    original_size=original_size,
                )
            )

            batch_logits.append(
                logits
            )

            batch_iou_predictions.append(
                iou_predictions
            )

        logits = torch.cat(
            batch_logits,
            dim=0,
        )

        if logits.shape != (
            images.shape[0],
            1,
            images.shape[-2],
            images.shape[-1],
        ):
            raise RuntimeError(
                "Unexpected SAM-LoRA final output shape.\n"
                "Expected: "
                f"{(
                    images.shape[0],
                    1,
                    images.shape[-2],
                    images.shape[-1],
                )}\n"
                f"Received: "
                f"{tuple(logits.shape)}"
            )

        if not return_aux:
            return logits

        iou_predictions = torch.cat(
            batch_iou_predictions,
            dim=0,
        )

        return {
            "logits": logits,
            "iou_predictions": iou_predictions,
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return building probabilities in [0, 1].

        Shape:

            [B, 1, H, W]
        """

        logits = self.forward(
            images
        )

        if not torch.is_tensor(
            logits
        ):
            raise RuntimeError(
                "Unexpected forward output."
            )

        return torch.sigmoid(
            logits
        )

    @torch.no_grad()
    def predict_mask(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Convert SAM-LoRA probabilities into a binary building mask.

        This helper is only for inference/debugging.

        Final thesis inference will still use the shared threshold
        search and post-processing pipeline.
        """

        if (
            threshold < 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "threshold must satisfy "
                "0 <= threshold <= 1."
            )

        probabilities = (
            self.predict_proba(
                images
            )
        )

        return (
            probabilities >= threshold
        ).to(
            dtype=probabilities.dtype
        )
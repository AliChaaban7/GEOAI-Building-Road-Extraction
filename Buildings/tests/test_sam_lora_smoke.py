"""
Buildings/scripts/test_sam_lora_smoke.py

Structural smoke test for the thesis SAM-LoRA implementation.

This test DOES NOT train a real experiment.

It verifies that:

1. sam_lora.json loads correctly.
2. Official pretrained SAM checkpoint loads.
3. LoRA is successfully injected.
4. Original SAM image-encoder weights remain frozen.
5. Prompt encoder remains frozen.
6. Mask decoder is trainable.
7. LoRA parameters are trainable.
8. A [1, 3, 256, 256] image can pass through the model.
9. Model output is [1, 1, 256, 256].
10. BCE + Dice produces a finite scalar loss.
11. Backward propagation succeeds.
12. Gradients appear ONLY in:
       - LoRA adapters
       - SAM mask decoder
13. Frozen SAM parameters receive no gradients.

Run from repository root:

    python Buildings/scripts/test_sam_lora_smoke.py

Optional:

    python Buildings/scripts/test_sam_lora_smoke.py --device cuda

    python Buildings/scripts/test_sam_lora_smoke.py \
        --config Buildings/config/models/sam_lora.json
"""

from __future__ import annotations

import argparse
import gc
import sys

from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
)

import torch


# ---------------------------------------------------------------------
# Repository import path
# ---------------------------------------------------------------------


THIS_FILE = Path(
    __file__
).resolve()

REPOSITORY_ROOT = (
    THIS_FILE.parents[2]
)

if str(
    REPOSITORY_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            REPOSITORY_ROOT
        ),
    )


from Buildings.src.models.sam_lora import (  # noqa: E402
    build_sam_lora_factory_bundle,
    count_lora_parameters,
    print_sam_lora_factory_summary,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "Buildings"
    / "config"
    / "models"
    / "sam_lora.json"
)

DEFAULT_TILE_SIZE = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the thesis SAM-LoRA building "
            "segmentation implementation."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG
        ),
        help=(
            "Path to sam_lora.json."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help=(
            "PyTorch device. "
            "Default: cuda when available, otherwise cpu."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=(
            DEFAULT_TILE_SIZE
        ),
        help=(
            "Synthetic input tile size. "
            "Default: 256."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_SEED
        ),
        help=(
            "Random seed."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------


def is_lora_parameter(
    name: str,
) -> bool:
    """
    Return True when a parameter belongs to a Q/V LoRA adapter.
    """

    lora_tokens = (
        ".q_lora_A.",
        ".q_lora_B.",
        ".v_lora_A.",
        ".v_lora_B.",
    )

    return any(
        token in name
        for token in lora_tokens
    )


def is_mask_decoder_parameter(
    name: str,
) -> bool:
    """
    Return True for SAM mask-decoder parameters.
    """

    return (
        name.startswith(
            "sam.mask_decoder."
        )
        or ".sam.mask_decoder." in name
    )


def is_allowed_trainable_parameter(
    name: str,
) -> bool:
    """
    Current thesis SAM-LoRA training policy:

        Image encoder:
            LoRA only

        Prompt encoder:
            frozen

        Mask decoder:
            trainable
    """

    return (
        is_lora_parameter(
            name
        )
        or is_mask_decoder_parameter(
            name
        )
    )


# ---------------------------------------------------------------------
# Parameter inspection
# ---------------------------------------------------------------------


def inspect_parameter_policy(
    model: torch.nn.Module,
) -> Dict[str, object]:
    """
    Verify frozen/trainable parameter architecture.
    """

    total_parameters = 0
    trainable_parameters = 0
    frozen_parameters = 0

    lora_parameters = 0
    mask_decoder_parameters = 0

    trainable_names: List[
        str
    ] = []

    unexpected_trainable: List[
        str
    ] = []

    prompt_encoder_trainable: List[
        str
    ] = []

    unexpected_image_encoder_trainable: List[
        str
    ] = []

    frozen_original_qkv = 0

    for name, parameter in (
        model.named_parameters()
    ):

        count = parameter.numel()

        total_parameters += (
            count
        )

        if parameter.requires_grad:

            trainable_parameters += (
                count
            )

            trainable_names.append(
                name
            )

            if is_lora_parameter(
                name
            ):
                lora_parameters += (
                    count
                )

            elif is_mask_decoder_parameter(
                name
            ):
                mask_decoder_parameters += (
                    count
                )

            else:
                unexpected_trainable.append(
                    name
                )

            if (
                "sam.prompt_encoder."
                in name
            ):
                prompt_encoder_trainable.append(
                    name
                )

            if (
                "sam.image_encoder."
                in name
                and not is_lora_parameter(
                    name
                )
            ):
                unexpected_image_encoder_trainable.append(
                    name
                )

        else:

            frozen_parameters += (
                count
            )

            # ------------------------------------------------------
            # Original qkv Linear is nested inside LoRAQKV as:
            #
            #     attn.qkv.qkv.weight
            #     attn.qkv.qkv.bias
            #
            # These MUST remain frozen.
            # ------------------------------------------------------

            if (
                ".attn.qkv.qkv."
                in name
            ):
                frozen_original_qkv += (
                    count
                )

    if total_parameters <= 0:

        raise RuntimeError(
            "Model contains zero parameters."
        )

    if trainable_parameters <= 0:

        raise RuntimeError(
            "Model contains zero trainable parameters."
        )

    if lora_parameters <= 0:

        raise RuntimeError(
            "No trainable LoRA parameters were detected."
        )

    if mask_decoder_parameters <= 0:

        raise RuntimeError(
            "No trainable SAM mask-decoder parameters were detected."
        )

    if frozen_original_qkv <= 0:

        raise RuntimeError(
            "Could not verify frozen original SAM QKV parameters."
        )

    if unexpected_trainable:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in unexpected_trainable
        )

        raise RuntimeError(
            "Unexpected trainable parameters detected.\n\n"
            "Only LoRA adapters and SAM mask decoder are allowed "
            "to be trainable.\n\n"
            f"{formatted}"
        )

    if prompt_encoder_trainable:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in prompt_encoder_trainable
        )

        raise RuntimeError(
            "SAM prompt encoder contains trainable parameters:\n"
            f"{formatted}"
        )

    if unexpected_image_encoder_trainable:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in unexpected_image_encoder_trainable
        )

        raise RuntimeError(
            "Original SAM image-encoder parameters are unexpectedly "
            "trainable:\n"
            f"{formatted}"
        )

    return {
        "total_parameters": int(
            total_parameters
        ),
        "trainable_parameters": int(
            trainable_parameters
        ),
        "frozen_parameters": int(
            frozen_parameters
        ),
        "lora_parameters": int(
            lora_parameters
        ),
        "mask_decoder_parameters": int(
            mask_decoder_parameters
        ),
        "frozen_original_qkv_parameters": int(
            frozen_original_qkv
        ),
        "trainable_names": (
            trainable_names
        ),
    }


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------


def create_fake_batch(
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create one synthetic RGB tile and one synthetic binary building mask.
    """

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0."
        )

    images = torch.rand(
        (
            DEFAULT_BATCH_SIZE,
            3,
            tile_size,
            tile_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------------
    # Respect model input convention.
    # --------------------------------------------------------------

    input_range = getattr(
        model,
        "input_range",
        "0_1",
    )

    if input_range == "0_255":

        images = (
            images
            * 255.0
        )

    elif input_range != "0_1":

        raise RuntimeError(
            "Unexpected model input_range during smoke test: "
            f"{input_range}"
        )

    # --------------------------------------------------------------
    # Synthetic binary building mask.
    #
    # Roughly 25% positive pixels.
    # --------------------------------------------------------------

    masks = (
        torch.rand(
            (
                DEFAULT_BATCH_SIZE,
                1,
                tile_size,
                tile_size,
            ),
            dtype=torch.float32,
            device=device,
        )
        > 0.75
    ).float()

    return (
        images,
        masks,
    )


# ---------------------------------------------------------------------
# Gradient inspection
# ---------------------------------------------------------------------


def inspect_gradients(
    model: torch.nn.Module,
) -> Dict[str, object]:
    """
    Verify backward-pass gradient policy.

    Frozen parameters must NEVER receive gradients.

    Gradient-bearing trainable parameters must belong only to:

        LoRA
        Mask Decoder

    Note
    ----
    Not every trainable mask-decoder parameter is required to receive a
    gradient during this specific forward pass.

    For example, SAM's IoU prediction head is not part of our BCE+Dice
    segmentation loss.

    Therefore the smoke test checks:

        gradients appear in LoRA
        gradients appear in mask decoder
        no forbidden parameter receives gradients
    """

    gradient_names: List[
        str
    ] = []

    no_gradient_trainable_names: List[
        str
    ] = []

    frozen_with_gradient: List[
        str
    ] = []

    unexpected_gradient_names: List[
        str
    ] = []

    non_finite_gradient_names: List[
        str
    ] = []

    lora_gradient_parameters = 0
    mask_decoder_gradient_parameters = 0

    nonzero_lora_gradient_tensors = 0
    nonzero_mask_decoder_gradient_tensors = 0

    for name, parameter in (
        model.named_parameters()
    ):

        gradient = parameter.grad

        # ----------------------------------------------------------
        # Frozen parameter.
        # ----------------------------------------------------------

        if not parameter.requires_grad:

            if gradient is not None:

                frozen_with_gradient.append(
                    name
                )

            continue

        # ----------------------------------------------------------
        # Trainable but unused in this particular loss graph.
        # ----------------------------------------------------------

        if gradient is None:

            no_gradient_trainable_names.append(
                name
            )

            continue

        gradient_names.append(
            name
        )

        if not torch.isfinite(
            gradient
        ).all():

            non_finite_gradient_names.append(
                name
            )

        if is_lora_parameter(
            name
        ):

            lora_gradient_parameters += (
                parameter.numel()
            )

            if torch.count_nonzero(
                gradient
            ).item() > 0:

                nonzero_lora_gradient_tensors += (
                    1
                )

        elif is_mask_decoder_parameter(
            name
        ):

            mask_decoder_gradient_parameters += (
                parameter.numel()
            )

            if torch.count_nonzero(
                gradient
            ).item() > 0:

                nonzero_mask_decoder_gradient_tensors += (
                    1
                )

        else:

            unexpected_gradient_names.append(
                name
            )

    # --------------------------------------------------------------
    # Safety assertions.
    # --------------------------------------------------------------

    if frozen_with_gradient:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in frozen_with_gradient
        )

        raise RuntimeError(
            "Frozen SAM parameters unexpectedly received gradients:\n"
            f"{formatted}"
        )

    if unexpected_gradient_names:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in unexpected_gradient_names
        )

        raise RuntimeError(
            "Gradients were detected outside LoRA + mask decoder:\n"
            f"{formatted}"
        )

    if non_finite_gradient_names:

        formatted = "\n".join(
            f"    - {name}"
            for name
            in non_finite_gradient_names
        )

        raise FloatingPointError(
            "NaN/Inf gradients detected:\n"
            f"{formatted}"
        )

    if lora_gradient_parameters <= 0:

        raise RuntimeError(
            "Backward pass produced no LoRA gradients."
        )

    if mask_decoder_gradient_parameters <= 0:

        raise RuntimeError(
            "Backward pass produced no mask-decoder gradients."
        )

    if nonzero_lora_gradient_tensors <= 0:

        raise RuntimeError(
            "LoRA gradients exist but all are exactly zero."
        )

    if nonzero_mask_decoder_gradient_tensors <= 0:

        raise RuntimeError(
            "Mask-decoder gradients exist but all are exactly zero."
        )

    return {
        "gradient_parameter_tensors": int(
            len(
                gradient_names
            )
        ),
        "lora_gradient_parameters": int(
            lora_gradient_parameters
        ),
        "mask_decoder_gradient_parameters": int(
            mask_decoder_gradient_parameters
        ),
        "nonzero_lora_gradient_tensors": int(
            nonzero_lora_gradient_tensors
        ),
        "nonzero_mask_decoder_gradient_tensors": int(
            nonzero_mask_decoder_gradient_tensors
        ),
        "trainable_without_gradient_count": int(
            len(
                no_gradient_trainable_names
            )
        ),
        "trainable_without_gradient_names": (
            no_gradient_trainable_names
        ),
    }


# ---------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------


def main() -> None:
    """
    Run SAM-LoRA structural smoke test.
    """

    args = parse_arguments()

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() "
            "returned False."
        )

    config_path = (
        Path(
            args.config
        )
        .expanduser()
        .resolve()
    )

    print()
    print("=" * 72)
    print(
        "SAM-LoRA Smoke Test"
    )
    print("=" * 72)

    print(
        f"Repository                  : "
        f"{REPOSITORY_ROOT}"
    )

    print(
        f"Config                      : "
        f"{config_path}"
    )

    print(
        f"Device                      : "
        f"{device}"
    )

    print(
        f"Fake tile                   : "
        f"1 x 3 x {args.tile_size} x {args.tile_size}"
    )

    print("=" * 72)
    print()

    # ==============================================================
    # STEP 1
    # LOAD FACTORY
    # ==============================================================

    print(
        "[1/8] Loading SAM-LoRA configuration..."
    )

    bundle = (
        build_sam_lora_factory_bundle(
            config_path=config_path,
            verbose_model_build=False,
        )
    )

    print_sam_lora_factory_summary(
        bundle
    )

    # ==============================================================
    # STEP 2
    # BUILD MODEL
    # ==============================================================

    print(
        "[2/8] Building fresh SAM-LoRA model..."
    )

    model = (
        bundle
        .model_factory()
        .to(
            device
        )
    )

    criterion = (
        bundle
        .criterion
        .to(
            device
        )
    )

    model.train()

    # ==============================================================
    # STEP 3
    # PARAMETER POLICY
    # ==============================================================

    print(
        "[3/8] Verifying frozen/trainable parameters..."
    )

    parameter_report = (
        inspect_parameter_policy(
            model
        )
    )

    independent_lora_count = (
        count_lora_parameters(
            model
        )
    )

    if (
        independent_lora_count
        != parameter_report[
            "lora_parameters"
        ]
    ):

        raise RuntimeError(
            "LoRA parameter count mismatch.\n"
            f"Package counter : {independent_lora_count:,}\n"
            f"Smoke test      : "
            f"{parameter_report['lora_parameters']:,}"
        )

    print(
        f"      Total parameters      : "
        f"{parameter_report['total_parameters']:,}"
    )

    print(
        f"      Frozen parameters     : "
        f"{parameter_report['frozen_parameters']:,}"
    )

    print(
        f"      Trainable parameters  : "
        f"{parameter_report['trainable_parameters']:,}"
    )

    print(
        f"      LoRA parameters       : "
        f"{parameter_report['lora_parameters']:,}"
    )

    print(
        f"      Mask decoder params   : "
        f"{parameter_report['mask_decoder_parameters']:,}"
    )

    print(
        "      Parameter policy      : PASS"
    )

    # ==============================================================
    # STEP 4
    # FAKE INPUT
    # ==============================================================

    print(
        "[4/8] Creating synthetic building batch..."
    )

    images, masks = (
        create_fake_batch(
            model=model,
            device=device,
            tile_size=(
                args.tile_size
            ),
        )
    )

    print(
        f"      Images                : "
        f"{tuple(images.shape)}"
    )

    print(
        f"      Masks                 : "
        f"{tuple(masks.shape)}"
    )

    # ==============================================================
    # STEP 5
    # FORWARD
    # ==============================================================

    print(
        "[5/8] Running SAM-LoRA forward pass..."
    )

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        images
    )

    if not torch.is_tensor(
        logits
    ):

        raise TypeError(
            "SAM-LoRA forward did not return a tensor."
        )

    expected_shape = (
        DEFAULT_BATCH_SIZE,
        1,
        args.tile_size,
        args.tile_size,
    )

    if tuple(
        logits.shape
    ) != expected_shape:

        raise RuntimeError(
            "Incorrect SAM-LoRA output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {tuple(logits.shape)}"
        )

    if not torch.isfinite(
        logits
    ).all():

        raise FloatingPointError(
            "SAM-LoRA forward produced NaN/Inf logits."
        )

    print(
        f"      Output logits         : "
        f"{tuple(logits.shape)}"
    )

    print(
        "      Forward pass          : PASS"
    )

    # ==============================================================
    # STEP 6
    # LOSS
    # ==============================================================

    print(
        "[6/8] Computing BCE + Dice loss..."
    )

    loss_result = (
        criterion.compute_components(
            logits=logits,
            targets=masks,
        )
    )

    loss = (
        loss_result.total
    )

    if loss.ndim != 0:

        raise RuntimeError(
            "SAM-LoRA loss is not scalar."
        )

    if not torch.isfinite(
        loss
    ):

        raise FloatingPointError(
            "SAM-LoRA loss is NaN or Inf."
        )

    print(
        f"      BCE                   : "
        f"{loss_result.bce.detach().item():.6f}"
    )

    print(
        f"      Dice                  : "
        f"{loss_result.dice.detach().item():.6f}"
    )

    print(
        f"      Total                 : "
        f"{loss.detach().item():.6f}"
    )

    print(
        "      Loss                  : PASS"
    )

    # ==============================================================
    # STEP 7
    # BACKWARD
    # ==============================================================

    print(
        "[7/8] Running backward pass..."
    )

    loss.backward()

    gradient_report = (
        inspect_gradients(
            model
        )
    )

    print(
        f"      Gradient tensors      : "
        f"{gradient_report['gradient_parameter_tensors']}"
    )

    print(
        f"      LoRA grad parameters  : "
        f"{gradient_report['lora_gradient_parameters']:,}"
    )

    print(
        f"      Decoder grad params   : "
        f"{gradient_report['mask_decoder_gradient_parameters']:,}"
    )

    print(
        f"      LoRA nonzero tensors  : "
        f"{gradient_report['nonzero_lora_gradient_tensors']}"
    )

    print(
        f"      Decoder nonzero       : "
        f"{gradient_report['nonzero_mask_decoder_gradient_tensors']}"
    )

    print(
        f"      Trainable/no gradient : "
        f"{gradient_report['trainable_without_gradient_count']}"
    )

    print(
        "      Gradient policy       : PASS"
    )

    # ==============================================================
    # STEP 8
    # FINAL ARCHITECTURE ASSERTION
    # ==============================================================

    print(
        "[8/8] Final SAM-LoRA architecture check..."
    )

    trainable_percentage = (
        parameter_report[
            "trainable_parameters"
        ]
        / parameter_report[
            "total_parameters"
        ]
        * 100.0
    )

    print()
    print("=" * 72)
    print(
        "SAM-LoRA SMOKE TEST PASSED"
    )
    print("=" * 72)

    print(
        f"Backbone                    : "
        f"{bundle.backbone}"
    )

    print(
        "Image encoder base weights  : "
        "FROZEN"
    )

    print(
        "Image encoder LoRA Q/V       : "
        "TRAINABLE"
    )

    print(
        "Prompt encoder               : "
        "FROZEN"
    )

    print(
        "Mask decoder                 : "
        "TRAINABLE"
    )

    print(
        f"Trainable percentage        : "
        f"{trainable_percentage:.4f}%"
    )

    print(
        f"Forward output              : "
        f"{tuple(logits.shape)}"
    )

    print(
        "BCE + Dice                  : "
        "PASS"
    )

    print(
        "Backward                    : "
        "PASS"
    )

    print(
        "Frozen-gradient protection  : "
        "PASS"
    )

    print("=" * 72)
    print()

    # --------------------------------------------------------------
    # Cleanup.
    # --------------------------------------------------------------

    del loss
    del logits
    del masks
    del images
    del criterion
    del model
    del bundle

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
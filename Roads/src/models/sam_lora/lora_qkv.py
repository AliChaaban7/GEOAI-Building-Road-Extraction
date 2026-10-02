"""
LoRA adaptation for the Segment Anything Model image encoder.

Approach 2 - Road Extraction

The official SAM Vision Transformer uses one combined attention
projection:

    qkv = Linear(dim, 3 * dim)

This module preserves the original frozen SAM QKV projection and adds
low-rank updates only to:

    Q = Query
    V = Value

K = Key remains unchanged.

Equations
---------
Q' = Q + scale * Bq(Aq(x))
V' = V + scale * Bv(Av(x))

scale = alpha / rank

Initialization
--------------
LoRA A:
    Kaiming initialization

LoRA B:
    zeros

Therefore the LoRA residual starts at exactly zero and the model begins
from the original pretrained SAM behavior.

No original SAM package source code is modified.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Q/V LoRA wrapper
# ---------------------------------------------------------------------

class LoRAQKV(nn.Module):
    """
    Wrap SAM's combined QKV projection with LoRA updates.

    Parameters
    ----------
    qkv:
        Original SAM attention qkv projection.

    rank:
        LoRA rank.

    alpha:
        LoRA scaling parameter.

    dropout:
        Dropout applied only to the LoRA branch.

    enable_q:
        Enable Query adaptation.

    enable_v:
        Enable Value adaptation.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        enable_q: bool = True,
        enable_v: bool = True,
    ):
        super().__init__()

        if not isinstance(
            qkv,
            nn.Linear,
        ):

            raise TypeError(
                "LoRAQKV expects nn.Linear."
            )

        rank = int(
            rank
        )

        alpha = float(
            alpha
        )

        dropout = float(
            dropout
        )

        if rank <= 0:

            raise ValueError(
                "LoRA rank must be > 0."
            )

        if alpha <= 0:

            raise ValueError(
                "LoRA alpha must be > 0."
            )

        if not (
            0.0
            <= dropout
            < 1.0
        ):

            raise ValueError(
                "LoRA dropout must satisfy 0 <= dropout < 1."
            )

        self.dim = int(
            qkv.in_features
        )

        expected_output = (
            self.dim
            * 3
        )

        if int(
            qkv.out_features
        ) != expected_output:

            raise ValueError(
                "Unexpected SAM qkv dimensions.\n"
                f"Expected: {expected_output}\n"
                f"Received: {qkv.out_features}"
            )

        self.rank = rank

        self.alpha = alpha

        self.scaling = (
            alpha
            / float(
                rank
            )
        )

        self.enable_q = bool(
            enable_q
        )

        self.enable_v = bool(
            enable_v
        )

        if not (
            self.enable_q
            or self.enable_v
        ):

            raise ValueError(
                "At least Query or Value LoRA must be enabled."
            )

        # -------------------------------------------------------------
        # Original SAM path
        # -------------------------------------------------------------

        self.qkv = qkv

        for parameter in (
            self.qkv.parameters()
        ):

            parameter.requires_grad = False

        # -------------------------------------------------------------
        # LoRA input dropout
        # -------------------------------------------------------------

        if dropout > 0.0:

            self.lora_dropout = (
                nn.Dropout(
                    p=dropout
                )
            )

        else:

            self.lora_dropout = (
                nn.Identity()
            )

        # -------------------------------------------------------------
        # Query LoRA
        # -------------------------------------------------------------

        if self.enable_q:

            self.q_A = nn.Linear(
                self.dim,
                rank,
                bias=False,
            )

            self.q_B = nn.Linear(
                rank,
                self.dim,
                bias=False,
            )

        else:

            self.q_A = None
            self.q_B = None

        # -------------------------------------------------------------
        # Value LoRA
        # -------------------------------------------------------------

        if self.enable_v:

            self.v_A = nn.Linear(
                self.dim,
                rank,
                bias=False,
            )

            self.v_B = nn.Linear(
                rank,
                self.dim,
                bias=False,
            )

        else:

            self.v_A = None
            self.v_B = None

        self.reset_lora_parameters()


    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def reset_lora_parameters(
        self,
    ) -> None:
        """
        Initialize A normally and B to zero.

        This guarantees zero LoRA residual at initialization.
        """

        if self.q_A is not None:

            nn.init.kaiming_uniform_(
                self.q_A.weight,
                a=5 ** 0.5,
            )

            nn.init.zeros_(
                self.q_B.weight
            )

        if self.v_A is not None:

            nn.init.kaiming_uniform_(
                self.v_A.weight,
                a=5 ** 0.5,
            )

            nn.init.zeros_(
                self.v_B.weight
            )


    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply original QKV + LoRA Q/V residuals.
        """

        original_qkv = self.qkv(
            x
        )

        lora_input = (
            self.lora_dropout(
                x
            )
        )

        if self.enable_q:

            q_update = (
                self.q_B(
                    self.q_A(
                        lora_input
                    )
                )
                * self.scaling
            )

            original_qkv[
                ...,
                :self.dim,
            ] = (
                original_qkv[
                    ...,
                    :self.dim,
                ]
                + q_update
            )

        if self.enable_v:

            v_start = (
                self.dim
                * 2
            )

            v_end = (
                self.dim
                * 3
            )

            v_update = (
                self.v_B(
                    self.v_A(
                        lora_input
                    )
                )
                * self.scaling
            )

            original_qkv[
                ...,
                v_start:v_end,
            ] = (
                original_qkv[
                    ...,
                    v_start:v_end,
                ]
                + v_update
            )

        return original_qkv


# ---------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------

def inject_lora_into_sam_image_encoder(
    image_encoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_blocks: Sequence[int]
    | str = "all",
    enable_q: bool = True,
    enable_v: bool = True,
) -> List[int]:
    """
    Inject Q/V LoRA into SAM image-encoder attention blocks.

    Expected SAM structure
    ----------------------
    image_encoder.blocks[i].attn.qkv
    """

    if not hasattr(
        image_encoder,
        "blocks",
    ):

        raise AttributeError(
            "SAM image encoder does not expose '.blocks'."
        )

    number_of_blocks = len(
        image_encoder.blocks
    )

    if target_blocks == "all":

        selected_blocks = list(
            range(
                number_of_blocks
            )
        )

    else:

        selected_blocks = [
            int(
                index
            )
            for index
            in target_blocks
        ]

    if not selected_blocks:

        raise ValueError(
            "No SAM blocks selected for LoRA."
        )

    for index in selected_blocks:

        if not (
            0
            <= index
            < number_of_blocks
        ):

            raise IndexError(
                f"SAM block index {index} is invalid. "
                f"Image encoder has {number_of_blocks} blocks."
            )

        block = (
            image_encoder.blocks[
                index
            ]
        )

        if not hasattr(
            block,
            "attn",
        ):

            raise AttributeError(
                f"SAM block {index} has no 'attn' module."
            )

        attention = block.attn

        if not hasattr(
            attention,
            "qkv",
        ):

            raise AttributeError(
                f"SAM block {index} attention has no qkv projection."
            )

        if isinstance(
            attention.qkv,
            LoRAQKV,
        ):

            raise RuntimeError(
                f"LoRA has already been injected into SAM block {index}."
            )

        attention.qkv = LoRAQKV(
            qkv=attention.qkv,

            rank=rank,

            alpha=alpha,

            dropout=dropout,

            enable_q=enable_q,

            enable_v=enable_v,
        )

    return selected_blocks


# ---------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------

def iter_lora_parameters(
    module: nn.Module,
) -> Iterable[
    nn.Parameter
]:
    """
    Yield only LoRA adapter parameters.
    """

    for submodule in (
        module.modules()
    ):

        if not isinstance(
            submodule,
            LoRAQKV,
        ):

            continue

        for attribute in (
            "q_A",
            "q_B",
            "v_A",
            "v_B",
        ):

            layer = getattr(
                submodule,
                attribute,
                None,
            )

            if layer is None:

                continue

            yield from (
                layer.parameters()
            )


def count_lora_parameters(
    module: nn.Module,
) -> int:
    """
    Count LoRA trainable parameters.
    """

    return int(
        sum(
            parameter.numel()
            for parameter
            in iter_lora_parameters(
                module
            )
        )
    )
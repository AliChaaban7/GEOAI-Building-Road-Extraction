"""
Buildings/src/models/sam_lora/lora_qkv.py

LoRA adaptation for the Segment Anything Model (SAM) image encoder.

The official SAM Vision Transformer uses one combined QKV projection:

    qkv = Linear(dim, 3 * dim)

This module keeps the original SAM QKV projection frozen and adds
Low-Rank Adaptation (LoRA) updates only to:

    Q = Query
    V = Value

K = Key remains unchanged.

Mathematically:

    Q' = Q + scale * Bq(Aq(x))
    V' = V + scale * Bv(Av(x))

where:

    scale = alpha / rank

The LoRA B matrices are initialized to zero, which guarantees that
immediately after insertion the model behaves exactly like the original
pretrained SAM.

No original SAM source file is modified.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """
    Wrap SAM's combined QKV linear projection with LoRA updates
    applied only to Query and Value.

    Expected wrapped layer:

        nn.Linear(dim, dim * 3)

    Parameters
    ----------
    qkv:
        Original SAM Attention.qkv layer.

    rank:
        LoRA rank r.

    alpha:
        LoRA scaling parameter.

    dropout:
        Optional dropout applied to the LoRA input only.
        The original SAM path is never dropped.

    enable_q:
        Whether Query receives a LoRA update.

    enable_v:
        Whether Value receives a LoRA update.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        enable_q: bool = True,
        enable_v: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(qkv, nn.Linear):
            raise TypeError(
                "LoRAQKV expects SAM Attention.qkv to be nn.Linear, "
                f"but received {type(qkv).__name__}."
            )

        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, received {rank}.")

        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, received {alpha}.")

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(
                f"LoRA dropout must satisfy 0 <= dropout < 1, received {dropout}."
            )

        self.dim = int(qkv.in_features)

        if int(qkv.out_features) != self.dim * 3:
            raise ValueError(
                "Unexpected SAM QKV projection dimensions. "
                f"Expected out_features = 3 * {self.dim} = {self.dim * 3}, "
                f"but received {qkv.out_features}."
            )

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.enable_q = bool(enable_q)
        self.enable_v = bool(enable_v)

        if not self.enable_q and not self.enable_v:
            raise ValueError(
                "At least one LoRA target must be enabled: Query and/or Value."
            )

        # ------------------------------------------------------------------
        # Original pretrained SAM QKV projection.
        # ------------------------------------------------------------------
        self.qkv = qkv

        # The original SAM QKV weights are always frozen here.
        for parameter in self.qkv.parameters():
            parameter.requires_grad = False

        # ------------------------------------------------------------------
        # LoRA dropout.
        # ------------------------------------------------------------------
        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=float(dropout))
        else:
            self.lora_dropout = nn.Identity()

        # ------------------------------------------------------------------
        # Query LoRA.
        #
        # x:
        #   [..., dim]
        #
        # A:
        #   dim -> rank
        #
        # B:
        #   rank -> dim
        # ------------------------------------------------------------------
        if self.enable_q:
            self.q_lora_A = nn.Linear(
                self.dim,
                self.rank,
                bias=False,
            )

            self.q_lora_B = nn.Linear(
                self.rank,
                self.dim,
                bias=False,
            )
        else:
            self.q_lora_A = None
            self.q_lora_B = None

        # ------------------------------------------------------------------
        # Value LoRA.
        # ------------------------------------------------------------------
        if self.enable_v:
            self.v_lora_A = nn.Linear(
                self.dim,
                self.rank,
                bias=False,
            )

            self.v_lora_B = nn.Linear(
                self.rank,
                self.dim,
                bias=False,
            )
        else:
            self.v_lora_A = None
            self.v_lora_B = None

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """
        Initialize LoRA adapters.

        A matrices:
            Kaiming initialization.

        B matrices:
            Zero initialization.

        Zero-initialized B means:

            delta_Q = 0
            delta_V = 0

        at initialization, so inserting LoRA does not immediately alter
        the pretrained SAM output.
        """

        if self.enable_q:
            assert self.q_lora_A is not None
            assert self.q_lora_B is not None

            nn.init.kaiming_uniform_(
                self.q_lora_A.weight,
                a=5**0.5,
            )

            nn.init.zeros_(
                self.q_lora_B.weight
            )

        if self.enable_v:
            assert self.v_lora_A is not None
            assert self.v_lora_B is not None

            nn.init.kaiming_uniform_(
                self.v_lora_A.weight,
                a=5**0.5,
            )

            nn.init.zeros_(
                self.v_lora_B.weight
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        SAM passes tensors shaped approximately:

            [B, H, W, C]

        into Attention.qkv.

        nn.Linear operates on the final C dimension, so the LoRA layers
        work without reshaping the spatial dimensions.
        """

        # Original frozen SAM QKV.
        base_qkv = self.qkv(x)

        # Separate Query, Key, Value.
        q, k, v = torch.split(
            base_qkv,
            self.dim,
            dim=-1,
        )

        # LoRA sees exactly the same input tensor.
        lora_input = self.lora_dropout(x)

        # --------------------------------------------------------------
        # Query update.
        # --------------------------------------------------------------
        if self.enable_q:
            assert self.q_lora_A is not None
            assert self.q_lora_B is not None

            delta_q = self.q_lora_B(
                self.q_lora_A(lora_input)
            )

            q = q + self.scaling * delta_q

        # --------------------------------------------------------------
        # Value update.
        # --------------------------------------------------------------
        if self.enable_v:
            assert self.v_lora_A is not None
            assert self.v_lora_B is not None

            delta_v = self.v_lora_B(
                self.v_lora_A(lora_input)
            )

            v = v + self.scaling * delta_v

        # Key remains exactly as produced by pretrained SAM.
        return torch.cat(
            [q, k, v],
            dim=-1,
        )

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        """
        Yield only trainable LoRA parameters.
        """

        if self.enable_q:
            assert self.q_lora_A is not None
            assert self.q_lora_B is not None

            yield from self.q_lora_A.parameters()
            yield from self.q_lora_B.parameters()

        if self.enable_v:
            assert self.v_lora_A is not None
            assert self.v_lora_B is not None

            yield from self.v_lora_A.parameters()
            yield from self.v_lora_B.parameters()


def _normalize_target_projections(
    target_projections: Sequence[str],
) -> tuple[bool, bool]:
    """
    Convert configuration such as:

        ["q", "v"]

    into:

        enable_q = True
        enable_v = True
    """

    targets = {
        str(target).strip().lower()
        for target in target_projections
    }

    allowed = {"q", "v"}

    unsupported = targets - allowed

    if unsupported:
        raise ValueError(
            "Unsupported SAM-LoRA target projections: "
            f"{sorted(unsupported)}. "
            "Current implementation supports only ['q', 'v']."
        )

    enable_q = "q" in targets
    enable_v = "v" in targets

    if not enable_q and not enable_v:
        raise ValueError(
            "SAM-LoRA target_projections must contain 'q', 'v', "
            "or both."
        )

    return enable_q, enable_v


def inject_lora_into_sam_image_encoder(
    image_encoder: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_projections: Sequence[str] = ("q", "v"),
    transformer_blocks: Union[str, Sequence[int]] = "all",
) -> List[int]:
    """
    Inject LoRA wrappers into SAM image-encoder Transformer blocks.

    Parameters
    ----------
    image_encoder:
        SAM image_encoder.

    rank:
        LoRA rank.

    alpha:
        LoRA alpha.

    dropout:
        LoRA dropout.

    target_projections:
        Any subset of:
            ["q", "v"]

    transformer_blocks:
        "all"
        or
        explicit block indices, e.g.:
            [0, 1, 2, 3]

    Returns
    -------
    List[int]
        Transformer block indices into which LoRA was inserted.
    """

    if not hasattr(image_encoder, "blocks"):
        raise AttributeError(
            "SAM image encoder does not contain a 'blocks' attribute. "
            "The installed SAM architecture may be incompatible with "
            "this LoRA implementation."
        )

    blocks = image_encoder.blocks

    num_blocks = len(blocks)

    if num_blocks == 0:
        raise RuntimeError(
            "SAM image encoder contains zero Transformer blocks."
        )

    enable_q, enable_v = _normalize_target_projections(
        target_projections
    )

    # --------------------------------------------------------------
    # Resolve blocks.
    # --------------------------------------------------------------
    if isinstance(transformer_blocks, str):
        if transformer_blocks.lower().strip() != "all":
            raise ValueError(
                "transformer_blocks string must be 'all', "
                f"received '{transformer_blocks}'."
            )

        selected_blocks = list(range(num_blocks))

    else:
        selected_blocks = sorted(
            {
                int(index)
                for index in transformer_blocks
            }
        )

        if not selected_blocks:
            raise ValueError(
                "transformer_blocks cannot be empty."
            )

        for index in selected_blocks:
            if index < 0 or index >= num_blocks:
                raise IndexError(
                    f"Transformer block index {index} is invalid. "
                    f"SAM image encoder has {num_blocks} blocks "
                    f"(valid indices 0-{num_blocks - 1})."
                )

    injected_blocks: List[int] = []

    # --------------------------------------------------------------
    # Replace Attention.qkv with LoRAQKV wrapper.
    # --------------------------------------------------------------
    for index in selected_blocks:
        block = blocks[index]

        if not hasattr(block, "attn"):
            raise AttributeError(
                f"SAM Transformer block {index} has no 'attn' module."
            )

        attention = block.attn

        if not hasattr(attention, "qkv"):
            raise AttributeError(
                f"SAM Transformer block {index} attention "
                "has no 'qkv' projection."
            )

        current_qkv = attention.qkv

        # Prevent accidental double injection.
        if isinstance(current_qkv, LoRAQKV):
            raise RuntimeError(
                f"LoRA has already been injected into "
                f"SAM Transformer block {index}."
            )

        if not isinstance(current_qkv, nn.Linear):
            raise TypeError(
                f"Expected block {index}.attn.qkv to be nn.Linear, "
                f"received {type(current_qkv).__name__}."
            )

        attention.qkv = LoRAQKV(
            qkv=current_qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_q=enable_q,
            enable_v=enable_v,
        )

        injected_blocks.append(index)

    return injected_blocks


def iter_lora_modules(
    module: nn.Module,
) -> Iterable[LoRAQKV]:
    """
    Yield all LoRAQKV modules contained in a model.
    """

    for child in module.modules():
        if isinstance(child, LoRAQKV):
            yield child


def iter_lora_parameters(
    module: nn.Module,
) -> Iterable[nn.Parameter]:
    """
    Yield all LoRA adapter parameters contained in a model.
    """

    for lora_module in iter_lora_modules(module):
        yield from lora_module.lora_parameters()


def count_lora_parameters(
    module: nn.Module,
) -> int:
    """
    Count trainable LoRA parameters.
    """

    return sum(
        parameter.numel()
        for parameter in iter_lora_parameters(module)
    )
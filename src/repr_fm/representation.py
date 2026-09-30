"""Replaceable multimodal representation with an optional research loss."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .interfaces import TokenBatch


@dataclass(frozen=True)
class RepresentationConfig:
    dim: int = 256
    kind: str = "transformer"
    depth: int = 2
    heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"transformer", "identity"}:
            raise ValueError("representation.kind must be transformer or identity")
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads:
            raise ValueError("representation.dim must be positive and divisible by heads")
        if self.depth < 1 or not 0 <= self.dropout < 1:
            raise ValueError("representation.depth must be positive and dropout in [0, 1)")


class Representation(nn.Module):
    """Map TokenBatch to TokenBatch without pooling away modality tokens.

    Subclass this module to implement a proposed representation. Preserve the
    output width and mask contract; auxiliary_loss is the separate loss hook.
    The default transformer is a small trainable fusion baseline, not a
    pretrained VLA backbone.
    """

    def __init__(self, config: RepresentationConfig) -> None:
        super().__init__()
        self.config = config
        if config.kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.heads,
                dim_feedforward=4 * config.dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(
                layer, num_layers=config.depth,
                norm=nn.LayerNorm(config.dim), enable_nested_tensor=False,
            )
        else:
            self.backbone = nn.Identity()

    def forward(self, encoded: TokenBatch) -> TokenBatch:
        encoded.validate()
        if encoded.tokens.shape[-1] != self.config.dim:
            raise ValueError("Representation input token width does not match config.dim")
        tokens = encoded.tokens.masked_fill(encoded.padding_mask[..., None], 0)
        if self.config.kind == "transformer":
            tokens = self.backbone(tokens, src_key_padding_mask=encoded.padding_mask)
        else:
            tokens = self.backbone(tokens)
        # Padded query positions can acquire values through attention; keep the
        # boundary unambiguous for custom pooling/auxiliary loss implementations.
        tokens = tokens.masked_fill(encoded.padding_mask[..., None], 0)
        return TokenBatch(tokens=tokens, padding_mask=encoded.padding_mask)

    def auxiliary_loss(
        self, encoded: TokenBatch, represented: TokenBatch, batch: dict[str, Any],
    ) -> torch.Tensor:
        """Return one scalar; override for the proposed representation objective.

        batch may include supervision such as actions. Only observation tokens
        enter forward(), so this training hook does not leak actions at inference.
        """
        return represented.tokens.sum() * 0.0

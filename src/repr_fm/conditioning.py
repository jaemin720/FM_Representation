"""Representation-to-action-head interface, independent of the flow solver."""

from dataclasses import dataclass

import torch
from torch import nn

from .interfaces import TokenBatch


@dataclass(frozen=True)
class ConditioningConfig:
    input_dim: int = 256
    output_dim: int = 2048

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("conditioning dimensions must be positive")


class Conditioning(nn.Module):
    """Masked mean readout from [B,N,D] to one [B,C] global condition.

    Pooling happens after representation fusion. Keep this module identical
    between representation ablations so head width/capacity do not change.
    """

    def __init__(self, config: ConditioningConfig) -> None:
        super().__init__()
        self.config = config
        self.readout = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(self, represented: TokenBatch) -> torch.Tensor:
        represented.validate()
        if represented.tokens.shape[-1] != self.config.input_dim:
            raise ValueError("Conditioning input width does not match config.input_dim")
        values = represented.tokens.masked_fill(represented.padding_mask[..., None], 0)
        count = (~represented.padding_mask).sum(dim=1, keepdim=True).to(values.dtype)
        pooled = values.sum(dim=1) / count
        return self.readout(pooled)

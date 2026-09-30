"""Tensor contract shared by observation encoders and representations."""

from dataclasses import dataclass

import torch


@dataclass
class TokenBatch:
    """Batch-first tokens; True in padding_mask means an invalid token."""

    tokens: torch.Tensor  # [batch, tokens, width]
    padding_mask: torch.Tensor  # bool [batch, tokens]

    def validate(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, D]")
        if self.padding_mask.shape != self.tokens.shape[:2]:
            raise ValueError("padding_mask must have shape [B, N]")
        if self.padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean (True means padding)")
        if self.padding_mask.device != self.tokens.device:
            raise ValueError("tokens and padding_mask must be on the same device")
        if self.padding_mask.all(dim=1).any():
            raise ValueError("Every observation must have at least one valid token")

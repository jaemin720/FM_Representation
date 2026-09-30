"""Checkpointed action statistics fitted only on training demonstrations."""

import torch
from torch import nn


class ActionNormalizer(nn.Module):
    """Per-dimension min/max mapping between raw actions and [-1, 1]."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.register_buffer("minimum", torch.zeros(action_dim))
        self.register_buffer("maximum", torch.ones(action_dim))
        self.register_buffer("scale", torch.ones(action_dim))
        self.register_buffer("bias", torch.zeros(action_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, minimum: torch.Tensor, maximum: torch.Tensor) -> None:
        minimum = torch.as_tensor(minimum, device=self.minimum.device, dtype=self.minimum.dtype)
        maximum = torch.as_tensor(maximum, device=self.maximum.device, dtype=self.maximum.dtype)
        if minimum.shape != self.minimum.shape or maximum.shape != self.maximum.shape:
            raise ValueError("Action statistics must have shape [action_dim]")
        if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
            raise ValueError("Action statistics must be finite")
        ranges = maximum - minimum
        if torch.any(ranges <= 1e-6):
            raise ValueError("Every action dimension needs a nonzero range")
        self.minimum.copy_(minimum)
        self.maximum.copy_(maximum)
        self.scale.copy_(2 / ranges)
        self.bias.copy_(-1 - minimum * self.scale)
        self.fitted.fill_(True)

    def _validate(self, actions: torch.Tensor) -> None:
        if not self.fitted:
            raise RuntimeError("ActionNormalizer is not fitted; load or fit training statistics")
        if actions.ndim < 1 or actions.shape[-1] != self.scale.shape[0]:
            raise ValueError("Last action dimension does not match normalizer")

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        self._validate(actions)
        return actions * self.scale + self.bias

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        self._validate(actions)
        return (actions - self.bias) / self.scale

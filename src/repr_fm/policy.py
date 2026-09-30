"""Orchestration of the four public research boundaries."""

import math
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .action_head import FMActionHead, FMHeadConfig
from .conditioning import Conditioning, ConditioningConfig
from .encoders import EncoderConfig, ObservationEncoder
from .interfaces import TokenBatch
from .normalization import ActionNormalizer
from .representation import Representation, RepresentationConfig


class FlowMatchingPolicy(nn.Module):
    """Encoder -> representation -> conditioning -> FM action head.

    Training loss receives a batch including actions. Encoding explicitly
    whitelists observation fields, so action supervision cannot enter the
    inference path by accidentally forwarding the complete training batch.
    sample() returns actions in the dataset's original units. Call eval()
    before evaluation to disable any representation dropout.
    """

    observation_keys = (
        "images", "vision_features", "instructions", "text_features",
        "text_padding_mask", "proprio",
    )

    def __init__(
        self, encoder: ObservationEncoder, representation: Representation,
        conditioning: Conditioning, action_head: FMActionHead,
        action_normalizer: ActionNormalizer | None = None,
        representation_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(representation_loss_weight) or representation_loss_weight < 0:
            raise ValueError("representation_loss_weight must be finite and nonnegative")
        self.encoder = encoder
        self.representation = representation
        self.conditioning = conditioning
        self.action_head = action_head
        self.action_normalizer = (
            ActionNormalizer(action_head.config.action_dim)
            if action_normalizer is None else action_normalizer
        )
        self.representation_loss_weight = representation_loss_weight

    def encode(self, batch: dict[str, Any]) -> tuple[TokenBatch, TokenBatch, torch.Tensor]:
        """Compute each observation stage once and expose representation tokens."""
        observation = {key: batch[key] for key in self.observation_keys if key in batch}
        encoded = self.encoder(observation)
        represented = self.representation(encoded)
        condition = self.conditioning(represented)
        return encoded, represented, condition

    def loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return scalar total, flow, and auxiliary representation losses."""
        if "actions" not in batch:
            raise ValueError("Training batch must include actions")
        encoded, represented, condition = self.encode(batch)
        normalized_actions = self.action_normalizer.normalize(batch["actions"])
        flow_loss = self.action_head.loss(normalized_actions, condition)
        representation_loss = self.representation.auxiliary_loss(encoded, represented, batch)
        if not isinstance(representation_loss, torch.Tensor) or representation_loss.ndim != 0:
            raise ValueError("representation.auxiliary_loss must return a scalar tensor")
        loss = flow_loss + self.representation_loss_weight * representation_loss
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "representation_loss": representation_loss,
        }

    @torch.no_grad()
    def sample(
        self, batch: dict[str, Any], generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Encode once, integrate the action flow, and denormalize [B,H,A]."""
        _, _, condition = self.encode(batch)
        normalized_actions = self.action_head.sample(
            condition, generator=generator, inference_steps=inference_steps,
        )
        return self.action_normalizer.unnormalize(normalized_actions)

    def checkpoint_config(self) -> dict[str, Any]:
        """Serialize the full model recipe including configuration defaults."""
        return {
            "encoder": asdict(self.encoder.config),
            "representation": asdict(self.representation.config),
            "conditioning": asdict(self.conditioning.config),
            "action_head": asdict(self.action_head.config),
            "representation_loss_weight": self.representation_loss_weight,
        }


def build_policy(model_config: dict[str, Any]) -> FlowMatchingPolicy:
    """Build independent stages from a model config (without loading weights)."""
    allowed = {"encoder", "representation", "conditioning", "action_head", "representation_loss_weight"}
    unexpected = set(model_config) - allowed
    if unexpected:
        raise ValueError(f"Unknown model configuration keys: {sorted(unexpected)}")
    encoder = EncoderConfig(**model_config.get("encoder", {}))
    representation = RepresentationConfig(**model_config.get("representation", {}))
    conditioning = ConditioningConfig(**model_config.get("conditioning", {}))
    action_head = FMHeadConfig(**model_config.get("action_head", {}))
    if encoder.dim != representation.dim:
        raise ValueError("encoder.dim must equal representation.dim")
    if representation.dim != conditioning.input_dim:
        raise ValueError("representation.dim must equal conditioning.input_dim")
    if conditioning.output_dim != action_head.condition_dim:
        raise ValueError("conditioning.output_dim must equal action_head.condition_dim")
    return FlowMatchingPolicy(
        encoder=ObservationEncoder(encoder),
        representation=Representation(representation),
        conditioning=Conditioning(conditioning),
        action_head=FMActionHead(action_head),
        representation_loss_weight=float(model_config.get("representation_loss_weight", 0.0)),
    )

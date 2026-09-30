"""Modular multimodal representation research with Flow Matching actions."""

from .action_head import ConditionalUnet1D, FMActionHead, FMHeadConfig
from .conditioning import Conditioning, ConditioningConfig
from .encoders import EncoderConfig, ObservationEncoder
from .interfaces import TokenBatch
from .normalization import ActionNormalizer
from .policy import FlowMatchingPolicy, build_policy
from .representation import Representation, RepresentationConfig

__all__ = [
    "ActionNormalizer", "ConditionalUnet1D", "Conditioning", "ConditioningConfig",
    "EncoderConfig", "FMActionHead", "FMHeadConfig", "FlowMatchingPolicy",
    "ObservationEncoder", "Representation", "RepresentationConfig", "TokenBatch",
    "build_policy",
]

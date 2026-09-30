"""Versioned checkpoints including fine-tuned, but excluding frozen, backbones.

Training-state handling is adapted from practice/DP/scripts/train.py. This
schema intentionally cannot load the legacy task-ID DP/FM checkpoints.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA = "representation_fm_v1"


def _backbone_prefixes(model: nn.Module) -> tuple[str, ...]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "frozen_backbone_prefixes"):
        return ()
    return tuple("encoder." + value.rstrip(".") + "." for value in encoder.frozen_backbone_prefixes())


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def persistent_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    prefixes = _backbone_prefixes(model)
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in parameters and not name.startswith(prefixes)
    }


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if parameters.keys() != self.shadow.keys():
            raise ValueError("Trainable parameter set changed after EMA initialization")
        for name, parameter in parameters.items():
            self.shadow[name].lerp_(parameter.detach(), 1 - self.decay)


def _validate_tensors(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], label: str) -> None:
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"Checkpoint {label} keys differ: missing={missing}, unexpected={extra}")
    for name, value in actual.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"Checkpoint {label} shape mismatch for {name}: {shape} != {tuple(expected[name].shape)}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint {label} contains non-finite values: {name}")


def validate_schema(payload: dict[str, Any]) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Expected {CHECKPOINT_SCHEMA!r} checkpoint; legacy DP/FM checkpoints "
            "use a different architecture and cannot be restored here."
        )


def make_model_payload(model: nn.Module, ema: EMA | None = None) -> dict[str, Any]:
    encoder = getattr(model, "encoder", None)
    if encoder is not None and encoder.config.vision_trainable and encoder.vision.backbone is None:
        raise ValueError("Initialize the trainable vision backbone before creating a checkpoint")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": model.checkpoint_config(),
        "model_trainable": trainable_state(model),
        "model_buffers": persistent_buffers(model),
    }
    _validate_tensors(payload["model_trainable"], payload["model_trainable"], "parameters")
    _validate_tensors(payload["model_buffers"], payload["model_buffers"], "buffers")
    if ema is not None:
        _validate_tensors(ema.shadow, payload["model_trainable"], "EMA")
        payload["ema"] = {name: value.detach().cpu().clone() for name, value in ema.shadow.items()}
        payload["ema_decay"] = ema.decay
    return payload


@torch.no_grad()
def load_model_state(model: nn.Module, payload: dict[str, Any], weights: str = "raw") -> None:
    """Validate every key/shape before modifying the policy.

    Call before freezing the whole policy for evaluation. Frozen backbone
    modules may be absent (lazy) or loaded; their weights never enter this state.
    Trainable backbones are initialized before validating/restoring their state.
    """
    validate_schema(payload)
    # v1 frozen checkpoints predate the optional fine-tuning flag.
    saved_config = payload.get("model_config", {})
    saved_config = {**saved_config, "encoder": {"vision_trainable": False, **saved_config.get("encoder", {})}}
    if saved_config != model.checkpoint_config():
        raise ValueError("Checkpoint model configuration differs from the constructed policy")
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be 'raw' or 'ema'")
    key = "model_trainable" if weights == "raw" else "ema"
    if key not in payload:
        raise ValueError(f"Checkpoint has no {weights} parameters")
    model.encoder.initialize_trainable_backbones(next(model.parameters()).device)
    expected_parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    expected_buffers = persistent_buffers(model)
    state, buffers = payload[key], payload.get("model_buffers", {})
    _validate_tensors(state, expected_parameters, weights)
    _validate_tensors(buffers, expected_buffers, "buffers")
    for name, parameter in expected_parameters.items():
        parameter.copy_(state[name].to(parameter))
    model_buffers = dict(model.named_buffers())
    for name, value in buffers.items():
        model_buffers[name].copy_(value.to(model_buffers[name]))


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    # Training checkpoints contain optimizer and Python/NumPy RNG state.
    # Only load checkpoints from trusted sources.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping")
    validate_schema(payload)
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    validate_schema(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def update_latest(destination: Path) -> None:
    """Maintain a stable resume path without duplicating checkpoint storage."""
    latest = destination.parent / "latest.pt"
    temporary = destination.parent / "latest.pt.tmp"
    temporary.unlink(missing_ok=True)
    os.link(destination, temporary)
    temporary.replace(latest)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; cannot restore saved CUDA RNG states")
        torch.cuda.set_rng_state_all(state["cuda"])

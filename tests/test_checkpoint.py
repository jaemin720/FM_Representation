"""Checkpoint contracts for representation experiments and EMA evaluation."""

from __future__ import annotations

import copy

import pytest
import torch

from repr_fm.checkpoint import (
    EMA, load_model_state, make_model_payload, read_checkpoint, save_checkpoint,
)
from test_policy import make_batch, make_policy, small_cpu_thread_pool


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_checkpoint_roundtrip_preserves_predictions_and_statistics(tmp_path, weights):
    source = make_policy().eval()
    source.action_normalizer.fit(torch.full((7,), -2.0), torch.full((7,), 3.0))
    source.encoder.set_proprio_statistics(torch.arange(9).float(), torch.full((9,), 2.0))
    ema = EMA(source, decay=0.8)
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05)
    ema.update(source)

    expected_policy = copy.deepcopy(source)
    if weights == "ema":
        with torch.no_grad():
            for name, parameter in expected_policy.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(ema.shadow[name])

    path = tmp_path / "checkpoint.pt"
    payload = make_model_payload(source, ema=ema)
    save_checkpoint(path, payload)
    loaded = read_checkpoint(path)
    restored = make_policy().eval()
    load_model_state(restored, loaded, weights=weights)

    assert restored.checkpoint_config() == source.checkpoint_config()
    expected = expected_policy.sample(make_batch(), generator=torch.Generator().manual_seed(53))
    actual = restored.sample(make_batch(), generator=torch.Generator().manual_seed(53))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, expected_value in expected_policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing_parameter", "shape", "missing_buffer", "config", "schema"])
def test_incompatible_checkpoint_is_rejected_without_partial_load(corruption):
    source = make_policy()
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.add_(0.1)
    payload = copy.deepcopy(make_model_payload(source))
    if corruption == "missing_parameter":
        payload["model_trainable"].pop(next(iter(payload["model_trainable"])))
    elif corruption == "shape":
        name = next(iter(payload["model_trainable"]))
        payload["model_trainable"][name] = torch.zeros(1)
    elif corruption == "missing_buffer":
        payload["model_buffers"].pop(next(iter(payload["model_buffers"])))
    elif corruption == "config":
        payload["model_config"]["representation"]["depth"] += 1
    else:
        payload["schema"] = "old_task_id_policy"

    target = make_policy()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        load_model_state(target, payload)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_ema_request_does_not_silently_fall_back_to_raw_weights():
    payload = make_model_payload(make_policy())
    with pytest.raises((ValueError, RuntimeError)):
        load_model_state(make_policy(), payload, weights="ema")


@pytest.mark.parametrize("source_loaded", [False, True])
def test_lazy_frozen_backbones_do_not_change_checkpoint_contract(source_loaded):
    source, target = make_policy(), make_policy()
    loaded = source if source_loaded else target
    for encoder in (loaded.encoder.vision, loaded.encoder.text):
        # Stand in for locally loaded pretrained weights, without downloading.
        encoder.backbone = torch.nn.Linear(8, 8).requires_grad_(False)
        encoder.backbone.register_buffer("pretrained_stat", torch.tensor([123.0]))
    before = copy.deepcopy(loaded.state_dict())

    payload = make_model_payload(source)
    frozen_prefixes = ("encoder.vision.backbone.", "encoder.text.backbone.")
    for state_name in ("model_trainable", "model_buffers"):
        assert not any(name.startswith(frozen_prefixes) for name in payload[state_name])
    load_model_state(target, payload)
    for name, value in loaded.state_dict().items():
        if name.startswith(frozen_prefixes):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_legacy_checkpoint_is_not_misinterpreted_as_modular_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"policy_type": "flow_matching", "model": {}}, path)
    with pytest.raises((ValueError, RuntimeError)):
        read_checkpoint(path)

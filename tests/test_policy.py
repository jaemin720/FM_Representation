"""CPU contract tests using synthetic cached features, without model downloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from repr_fm.policy import build_policy


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config() -> dict:
    return {
        "encoder": {
            "dim": 16,
            "num_views": 2,
            "vision_feature_dim": 8,
            "vision_grid_size": 2,
            "text_feature_dim": 12,
            "text_max_length": 8,
            "proprio_dim": 9,
        },
        "representation": {
            "kind": "transformer", "dim": 16, "heads": 4,
            "depth": 1, "dropout": 0.0,
        },
        "conditioning": {"input_dim": 16, "output_dim": 32},
        "action_head": {
            "condition_dim": 32, "timestep_dim": 8,
            "down_dims": [16, 32], "kernel_size": 3, "num_groups": 4,
            "horizon": 6, "action_dim": 7, "inference_steps": 4,
        },
        "representation_loss_weight": 0.0,
    }


def make_policy(config: dict | None = None):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        policy = build_policy(config or tiny_config())
    policy.action_normalizer.fit(torch.full((7,), -1.0), torch.ones(7))
    policy.encoder.set_proprio_statistics(torch.zeros(9), torch.ones(9))
    return policy


def make_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    return {
        "vision_features": torch.randn(2, 2, 4, 8, generator=generator),
        "text_features": torch.randn(2, 5, 12, generator=generator),
        "text_padding_mask": torch.tensor([
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]),
        "proprio": torch.randn(2, 9, generator=generator),
        "actions": torch.rand(2, 6, 7, generator=generator) * 2 - 1,
    }


def assert_module_receives_gradient(module):
    gradients = [
        parameter.grad for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients, "No gradient reached this trainable boundary"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_flow_loss_trains_all_four_boundaries():
    policy = make_policy().train()
    losses = policy.loss(make_batch())
    assert set(("loss", "flow_loss", "representation_loss")) <= losses.keys()
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    for module in (
        policy.encoder, policy.representation, policy.conditioning,
        policy.action_head,
    ):
        assert_module_receives_gradient(module)

    # Each modality must participate, not merely an encoder-global embedding.
    for modality in ("vision", "text", "proprio"):
        gradients = [
            parameter.grad for name, parameter in policy.encoder.named_parameters()
            if modality in name and "proj" in name and parameter.requires_grad
        ]
        assert gradients, f"No trainable {modality} projection found"
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_padding_contents_do_not_change_condition_or_actions():
    policy = make_policy().eval()
    batch = make_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["text_features"][batch["text_padding_mask"]] = 10000

    with torch.no_grad():
        condition = policy.encode(batch)[2]
        changed_condition = policy.encode(changed)[2]
    torch.testing.assert_close(condition, changed_condition, rtol=0, atol=1e-6)

    expected = policy.sample(batch, generator=torch.Generator().manual_seed(9))
    actual = policy.sample(changed, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(expected, actual, rtol=0, atol=2e-6)


def test_language_features_condition_policy_without_task_ids():
    policy = make_policy().eval()
    batch = make_batch()
    with torch.no_grad():
        condition = policy.encode(batch)[2]
        with_task_ids = policy.encode({**batch, "task_id": torch.tensor([99, -1])})[2]
        with_other_targets = policy.encode({**batch, "actions": torch.full_like(batch["actions"], 999)})[2]
        different_language = {
            **batch, "text_features": -batch["text_features"],
        }
        different_condition = policy.encode(different_language)[2]
    torch.testing.assert_close(condition, with_task_ids, rtol=0, atol=0)
    torch.testing.assert_close(condition, with_other_targets, rtol=0, atol=0)
    assert not torch.allclose(condition, different_condition)


def test_sampling_encodes_once_and_is_reproducible():
    policy = make_policy().eval()
    batch = make_batch()
    counts = {name: 0 for name in ("encoder", "representation", "conditioning", "velocity")}

    def count(name):
        def hook(_module, _args, _result):
            counts[name] += 1
        return hook

    handles = [
        getattr(policy, name).register_forward_hook(count(name))
        for name in ("encoder", "representation", "conditioning")
    ]
    handles.append(policy.action_head.velocity_net.register_forward_hook(count("velocity")))
    try:
        first = policy.sample(batch, generator=torch.Generator().manual_seed(42))
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"encoder": 1, "representation": 1, "conditioning": 1, "velocity": 4}
    second = policy.sample(batch, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 6, 7)
    assert torch.isfinite(first).all()
    assert not first.requires_grad
    assert first.min() >= -1 and first.max() <= 1


@pytest.mark.parametrize("shape", [(2, 5, 7), (2, 6, 8), (2, 7)])
def test_action_shape_is_checked_before_training(shape):
    policy = make_policy()
    batch = {**make_batch(), "actions": torch.zeros(shape)}
    with pytest.raises(ValueError):
        policy.loss(batch)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 3.0])
def test_invalid_normalized_action_targets_are_rejected(invalid):
    policy = make_policy()
    batch = make_batch()
    batch["actions"][0, 0, 0] = invalid
    with pytest.raises(ValueError):
        policy.loss(batch)


def test_action_statistics_are_used_in_training_and_sampling():
    policy = make_policy().eval()
    policy.action_normalizer.fit(torch.zeros(7), torch.full((7,), 4.0))
    batch = make_batch()
    # Raw actions outside [-1, 1] are valid when training statistics map them in.
    batch["actions"] = (batch["actions"] + 1) * 2
    assert torch.isfinite(policy.loss(batch)["loss"])
    sampled = policy.sample(batch, generator=torch.Generator().manual_seed(5))
    assert sampled.min() >= 0 and sampled.max() <= 4


def test_single_condition_unet_preserves_legacy_numerics(monkeypatch):
    """The boundary refactor must not silently change the inherited action head."""
    legacy_path = Path(__file__).resolve().parents[2] / "DP/src/dp/model.py"
    if not legacy_path.is_file():
        pytest.skip("Sibling legacy DP source is unavailable")
    spec = importlib.util.spec_from_file_location("legacy_dp_model_for_test", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, legacy)
    spec.loader.exec_module(legacy)

    config = tiny_config()["action_head"]
    # Legacy U-Net concatenated two half-width conditions inside forward().
    old_config = legacy.FlowMatchingPolicyConfig(**{**config, "condition_dim": 16})
    old = legacy.ConditionalUnet1D(old_config).eval()
    new = make_policy().action_head.velocity_net.eval()
    new.load_state_dict(old.state_dict(), strict=True)
    actions = make_batch()["actions"]
    times = torch.tensor([0.125, 0.875])
    condition = torch.randn(2, 32)
    with torch.no_grad():
        expected = old(actions, times, condition[:, :16], condition[:, 16:])
        actual = new(actions, times, condition)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

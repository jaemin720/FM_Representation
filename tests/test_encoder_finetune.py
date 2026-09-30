"""Fine-tuning must update raw-image weights and preserve them on restore."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from repr_fm.checkpoint import EMA, load_model_state, make_model_payload
from repr_fm.encoders import EncoderConfig
from test_policy import make_batch, make_policy, small_cpu_thread_pool, tiny_config


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 14, stride=14)
        self.mask_token = nn.Parameter(torch.zeros(1, 8))
        self.register_buffer("pretrained_stat", torch.tensor([3.0]))

    def forward_features(self, images):
        return {"x_norm_patchtokens": self.patch_embed(images).flatten(2).transpose(1, 2)}


@pytest.fixture
def vision_loader(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: TinyVision())


def finetune_config():
    config = tiny_config()
    config["encoder"].update(vision_trainable=True, vision_image_size=28)
    return config


def raw_batch():
    batch = make_batch()
    del batch["vision_features"]
    batch["images"] = torch.randint(0, 256, (2, 2, 3, 28, 28), dtype=torch.uint8,
                                    generator=torch.Generator().manual_seed(23))
    return batch


def test_optimizer_updates_vision_and_keeps_text_frozen(vision_loader):
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_finetune_test", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    model = make_policy(finetune_config())
    model.encoder.text.backbone = nn.Linear(12, 12).requires_grad_(False)
    assert model.encoder.vision.backbone is None
    optimizer = train.build_optimizer(model, {"learning_rate": 1e-3, "vision_learning_rate": 1e-4})
    ema = EMA(model, 0.9)
    model.train()
    backbone = model.encoder.vision.backbone
    assert backbone.training
    assert not model.encoder.text.backbone.training
    assert not backbone.mask_token.requires_grad
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    before = backbone.patch_embed.weight.detach().clone()
    frozen_before = copy.deepcopy(model.encoder.text.backbone.state_dict())
    model.loss(raw_batch())["loss"].backward()
    assert backbone.patch_embed.weight.grad.isfinite().all()
    assert backbone.patch_embed.weight.grad.abs().sum() > 0
    optimizer.step()
    ema.update(model)
    assert not torch.equal(backbone.patch_embed.weight, before)
    assert "encoder.vision.backbone.patch_embed.weight" in ema.shadow
    for name, value in model.encoder.text.backbone.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.encoder.text.backbone.parameters())
    model.eval()
    assert not backbone.training
    with torch.no_grad():
        assert not model.encoder.vision(raw_batch()["images"]).requires_grad


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_finetuned_backbone_checkpoint_restores_into_lazy_policy(vision_loader, weights):
    source = make_policy(finetune_config()).eval()
    source.encoder.initialize_trainable_backbones(torch.device("cpu"))
    ema = EMA(source, 0.8)
    with torch.no_grad():
        source.encoder.vision.backbone.patch_embed.weight.add_(0.15)
        source.encoder.vision.backbone.pretrained_stat.add_(1)
    ema.update(source)
    payload = make_model_payload(source, ema)
    assert "encoder.vision.backbone.patch_embed.weight" in payload["model_trainable"]
    assert "encoder.vision.backbone.pretrained_stat" in payload["model_buffers"]
    assert not any(name.startswith("encoder.text.backbone.") for name in payload["model_trainable"])
    target = make_policy(finetune_config()).eval()
    assert target.encoder.vision.backbone is None
    load_model_state(target, payload, weights=weights)
    expected = copy.deepcopy(source)
    if weights == "ema":
        with torch.no_grad():
            for name, parameter in expected.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(ema.shadow[name])
    actual = target.sample(raw_batch(), generator=torch.Generator().manual_seed(47))
    prediction = expected.sample(raw_batch(), generator=torch.Generator().manual_seed(47))
    torch.testing.assert_close(actual, prediction, rtol=0, atol=0)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


def test_finetuning_rejects_cached_visual_features():
    model = make_policy(finetune_config())
    with pytest.raises(ValueError, match="Image feature caches"):
        model.loss(make_batch())
    with pytest.raises(ValueError, match="Image feature caches"):
        model.encoder.config.cache_signature()
    with pytest.raises(ValueError, match="Initialize"):
        make_model_payload(model)


def test_frozen_vision_remains_gradient_free_and_old_checkpoints_load(vision_loader):
    config = tiny_config()
    config["encoder"]["vision_image_size"] = 28
    model = make_policy(config).train()
    features = model.encoder.vision(raw_batch()["images"])
    assert not features.requires_grad
    assert not model.encoder.vision.backbone.training
    assert all(not p.requires_grad for p in model.encoder.vision.backbone.parameters())
    payload = make_model_payload(model)
    del payload["model_config"]["encoder"]["vision_trainable"]
    target = make_policy(config)
    load_model_state(target, payload)
    assert "vision_trainable" not in EncoderConfig().cache_signature()

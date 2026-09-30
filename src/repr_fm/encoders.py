"""Current-observation tokens with optional DINOv2 fine-tuning and frozen BERT.

Image preprocessing is adapted from ../DP/src/dp/model.py. Unlike that source,
this module preserves camera/spatial tokens and encodes actual language.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .interfaces import TokenBatch


@dataclass(frozen=True)
class EncoderConfig:
    dim: int = 256
    num_views: int = 2
    proprio_dim: int = 9
    vision_model_name: str = "dinov2_vitb14"
    vision_repository: str = "facebookresearch/dinov2"
    vision_pretrained: bool = True
    vision_trainable: bool = False
    vision_image_size: int = 224
    vision_feature_dim: int = 768
    vision_grid_size: int = 8
    text_model_name: str = "google-bert/bert-base-uncased"
    text_revision: str = "main"
    text_feature_dim: int = 768
    text_max_length: int = 64
    text_local_files_only: bool = True

    def __post_init__(self) -> None:
        for key in ("dim", "num_views", "proprio_dim", "vision_image_size",
                    "vision_feature_dim", "vision_grid_size", "text_feature_dim", "text_max_length"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        dimensions = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                      "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
        if self.vision_model_name not in dimensions:
            raise ValueError("Only standard DINOv2 /14 models are supported")
        if not self.vision_pretrained:
            raise ValueError("Frozen backbone weights are external to checkpoints; vision_pretrained must be True")
        if self.vision_image_size % 14 or self.vision_grid_size > self.vision_image_size // 14:
            raise ValueError("DINO image size must be divisible by 14 and grid cannot exceed native patches")

    def cache_signature(self) -> dict[str, Any]:
        """Only frozen feature extraction settings, not trainable token widths."""
        if self.vision_trainable:
            raise ValueError("Image feature caches cannot be used with a trainable vision encoder")
        return {k: v for k, v in asdict(self).items()
                if (k.startswith(("vision_", "text_")) or k == "num_views")
                and k != "vision_trainable"}


class VisionTokens(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone: nn.Module | None = None

    def _load(self, device: torch.device) -> nn.Module:
        if self.backbone is None:
            # Native attention also supports CPU validation; xFormers is optional.
            os.environ.setdefault("XFORMERS_DISABLED", "1")
            repo = str(Path(self.config.vision_repository).expanduser())
            source = "local" if Path(repo).is_dir() else "github"
            # Constructing a pretrained module can initialize temporary random
            # parameters. Keep that from changing FM noise or resume RNG state.
            with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="xFormers is .*", category=UserWarning)
                self.backbone = torch.hub.load(
                    repo, self.config.vision_model_name,
                    pretrained=self.config.vision_pretrained, source=source,
                )
            self.backbone.requires_grad_(self.config.vision_trainable)
            # This policy never masks image patches, so this token is unused.
            if hasattr(self.backbone, "mask_token"):
                self.backbone.mask_token.requires_grad_(False)
        return self.backbone.to(device=device, dtype=torch.float32).train(
            self.training if self.config.vision_trainable else False,
        )

    def train(self, mode: bool = True) -> VisionTokens:
        super().train(mode if self.config.vision_trainable else False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 RGB [B,V,3,H,W] -> float32 [B,V,grid**2,Dv]."""
        cfg = self.config
        if images.ndim != 5 or images.shape[1:3] != (cfg.num_views, 3):
            raise ValueError("images must be [B, num_views, 3, H, W]")
        if images.dtype != torch.uint8:
            raise ValueError("images must be uint8 RGB in [0, 255]")
        backbone = self._load(images.device)
        batch, views = images.shape[:2]
        with torch.autocast(device_type=images.device.type, enabled=False):
            values = F.interpolate(
                images.flatten(0, 1).float() / 255.0,
                size=(cfg.vision_image_size, cfg.vision_image_size),
                mode="bicubic", align_corners=False, antialias=False,
            )
            mean = values.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = values.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            values = (values - mean) / std
        # Frozen features preserve the original float32 cache contract. During
        # fine-tuning, respect the trainer's autocast and the caller's grad mode.
        precision = nullcontext() if cfg.vision_trainable else torch.autocast(
            device_type=images.device.type, enabled=False,
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and cfg.vision_trainable), precision:
            features = backbone.forward_features(values)["x_norm_patchtokens"]
            native = cfg.vision_image_size // 14
            if features.shape[1:] != (native * native, cfg.vision_feature_dim):
                raise ValueError("vision_feature_dim does not match the loaded DINO backbone")
            spatial = features.transpose(1, 2).reshape(batch * views, cfg.vision_feature_dim, native, native)
            spatial = F.adaptive_avg_pool2d(spatial, (cfg.vision_grid_size, cfg.vision_grid_size))
            return spatial.flatten(2).transpose(1, 2).reshape(
                batch, views, cfg.vision_grid_size ** 2, cfg.vision_feature_dim,
            ).contiguous().float()


class FrozenTextTokens(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone: nn.Module | None = None
        self.tokenizer: Any = None
        # Cache frozen outputs only. Trainable projections run on every call.
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _load(self, device: torch.device) -> nn.Module:
        if self.backbone is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise ImportError("Raw instructions require transformers; install this project's dependencies") from error
            kwargs = {"revision": self.config.text_revision,
                      "local_files_only": self.config.text_local_files_only}
            with torch.random.fork_rng(devices=[]):
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name, **kwargs)
                self.backbone = AutoModel.from_pretrained(self.config.text_model_name, **kwargs)
            self.backbone.requires_grad_(False).eval()
        return self.backbone.to(device=device, dtype=torch.float32).eval()

    def train(self, mode: bool = True) -> FrozenTextTokens:
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, instructions: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not instructions or any(not isinstance(s, str) or not s.strip() for s in instructions):
            raise ValueError("instructions must contain nonempty natural-language strings")
        missing = list(dict.fromkeys(s for s in instructions if s not in self._cache))
        if missing:
            backbone = self._load(device)
            values = self.tokenizer(
                missing, padding="max_length", truncation=True,
                max_length=self.config.text_max_length, return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                features = backbone(**values).last_hidden_state.float()
            if features.shape[-1] != self.config.text_feature_dim:
                raise ValueError("text_feature_dim does not match the loaded text encoder")
            padding = ~values["attention_mask"].bool()
            for sentence, feature, mask in zip(missing, features.cpu(), padding.cpu()):
                self._cache[sentence] = (feature.clone(), mask.clone())
        result = [self._cache[s] for s in instructions]
        for sentence in instructions:
            self._cache.move_to_end(sentence)
        while len(self._cache) > 1024:
            self._cache.popitem(last=False)
        return (torch.stack([x[0] for x in result]).to(device),
                torch.stack([x[1] for x in result]).to(device))


class ObservationEncoder(nn.Module):
    """Configurable vision, frozen text, and trainable projections/embeddings."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = VisionTokens(config)
        self.text = FrozenTextTokens(config)
        self.vision_projection = nn.Linear(config.vision_feature_dim, config.dim)
        self.text_projection = nn.Linear(config.text_feature_dim, config.dim)
        self.proprio_projection = nn.Sequential(nn.Linear(config.proprio_dim, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.camera_embedding = nn.Embedding(config.num_views, config.dim)
        self.spatial_embedding = nn.Embedding(config.vision_grid_size ** 2, config.dim)
        self.modality_embedding = nn.Embedding(3, config.dim)
        self.register_buffer("proprio_mean", torch.zeros(config.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(config.proprio_dim))

    def frozen_backbone_prefixes(self) -> tuple[str, ...]:
        return ("text.backbone.",) if self.config.vision_trainable else ("vision.backbone.", "text.backbone.")

    def initialize_trainable_backbones(self, device: torch.device) -> None:
        """Materialize trainable weights before optimizer/EMA creation or restore."""
        if self.config.vision_trainable:
            self.vision._load(device)

    @torch.no_grad()
    def set_proprio_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.proprio_mean.shape or std.shape != self.proprio_std.shape:
            raise ValueError("proprio statistics have the wrong shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("proprio statistics must be finite and standard deviations nonnegative")
        self.proprio_mean.copy_(mean)
        self.proprio_std.copy_(std.clamp_min(1e-6))

    def forward(self, observation: Mapping[str, Any]) -> TokenBatch:
        cfg = self.config
        proprio = observation["proprio"]
        if proprio.ndim != 2 or proprio.shape[1] != cfg.proprio_dim:
            raise ValueError("proprio must be [B, proprio_dim]")
        batch = proprio.shape[0]
        vision = observation.get("vision_features")
        if vision is not None and cfg.vision_trainable:
            raise ValueError("Image feature caches cannot be used with a trainable vision encoder; provide images")
        if vision is None:
            vision = self.vision(observation["images"])
        expected = (batch, cfg.num_views, cfg.vision_grid_size ** 2, cfg.vision_feature_dim)
        if tuple(vision.shape) != expected:
            raise ValueError(f"vision_features must have shape {expected}; legacy averaged CLS caches are incompatible")
        text = observation.get("text_features")
        if text is None:
            instructions = observation.get("instructions")
            if isinstance(instructions, str) or instructions is None or len(instructions) != batch:
                raise ValueError("Provide one instruction string per observation")
            text, text_mask = self.text(instructions, proprio.device)
        else:
            text_mask = observation.get("text_padding_mask")
        if text.ndim != 3 or text.shape[0] != batch or text.shape[-1] != cfg.text_feature_dim:
            raise ValueError("text_features must be [B, L, text_feature_dim]")
        if text.shape[1] < 1 or text.shape[1] > cfg.text_max_length:
            raise ValueError("text feature length exceeds the configured maximum")
        if text_mask is None or text_mask.dtype != torch.bool or text_mask.shape != text.shape[:2]:
            raise ValueError("text_padding_mask must be bool [B,L], True for padding")
        if text_mask.all(dim=1).any():
            raise ValueError("Every instruction must contain a valid token")

        visual = self.vision_projection(vision)
        visual = (visual + self.camera_embedding.weight[None, :, None, :]
                  + self.spatial_embedding.weight[None, None, :, :]
                  + self.modality_embedding.weight[0])
        visual = visual.flatten(1, 2)
        language = self.text_projection(text.masked_fill(text_mask[..., None], 0))
        language = language + self.modality_embedding.weight[1]
        state = self.proprio_projection((proprio - self.proprio_mean) / self.proprio_std)
        state = state[:, None, :] + self.modality_embedding.weight[2]
        tokens = torch.cat((visual, language, state), dim=1)
        padding = torch.cat((
            torch.zeros(batch, visual.shape[1], dtype=torch.bool, device=proprio.device),
            text_mask,
            torch.zeros(batch, 1, dtype=torch.bool, device=proprio.device),
        ), dim=1)
        result = TokenBatch(tokens.masked_fill(padding[..., None], 0), padding)
        result.validate()
        return result

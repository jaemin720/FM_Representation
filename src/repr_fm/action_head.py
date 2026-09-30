"""Conditional action U-Net and straight-line Flow Matching objective.

The numerical U-Net architecture is extracted from ../DP/src/dp/model.py.
The two old condition arguments are now one global_cond vector of their
combined width. State-dict keys and computations of the U-Net are retained.
No observation, language, or representation modules belong in this file.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FMHeadConfig:
    condition_dim: int = 2048
    timestep_dim: int = 256
    down_dims: tuple[int, ...] = (512, 1024, 1536)
    kernel_size: int = 5
    num_groups: int = 8
    horizon: int = 6
    action_dim: int = 7
    inference_steps: int = 4
    noise_std: float = 1.0
    clip_actions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        if min(self.condition_dim, self.horizon, self.action_dim, self.inference_steps) < 1:
            raise ValueError("condition_dim/horizon/action_dim/inference_steps must be positive")
        if self.timestep_dim < 4 or self.timestep_dim % 2:
            raise ValueError("timestep_dim must be even and at least 4")
        if self.num_groups < 1 or len(self.down_dims) < 2:
            raise ValueError("num_groups must be positive; down_dims needs at least two widths")
        if any(dim < 1 or dim % self.num_groups for dim in self.down_dims):
            raise ValueError("down_dims must be positive and divisible by num_groups")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        if not math.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be finite and positive")

    @property
    def temporal_multiple(self) -> int:
        return 2 ** (len(self.down_dims) - 1)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=times.device) / (half - 1)
        )
        angles = times.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, out), nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ResidualBlock(nn.Module):
    def __init__(self, inp: int, out: int, condition_dim: int, config: FMHeadConfig) -> None:
        super().__init__()
        self.first = ConvBlock(inp, out, config.kernel_size, config.num_groups)
        self.second = ConvBlock(out, out, config.kernel_size, config.num_groups)
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, 2 * out))
        self.residual = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()
        self.out = out

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.first(value)
        scale, shift = self.condition(condition).view(-1, 2, self.out, 1).unbind(1)
        return self.second(scale * hidden + shift) + self.residual(value)


class ConditionalUnet1D(nn.Module):
    """Predict [B,H,A] velocity from noisy actions, float flow time and [B,C]."""

    def __init__(self, config: FMHeadConfig) -> None:
        super().__init__()
        self.config = config
        cdim = config.timestep_dim + config.condition_dim
        self.time = nn.Sequential(
            SinusoidalEmbedding(config.timestep_dim),
            nn.Linear(config.timestep_dim, 4 * config.timestep_dim), nn.Mish(),
            nn.Linear(4 * config.timestep_dim, config.timestep_dim),
        )
        pairs = list(zip((config.action_dim,) + config.down_dims[:-1], config.down_dims))
        self.down = nn.ModuleList([
            nn.ModuleList((
                ResidualBlock(inp, out, cdim, config),
                ResidualBlock(out, out, cdim, config),
                nn.Identity() if index == len(pairs) - 1
                else nn.Conv1d(out, out, 3, stride=2, padding=1),
            )) for index, (inp, out) in enumerate(pairs)
        ])
        last = config.down_dims[-1]
        self.mid = nn.ModuleList((
            ResidualBlock(last, last, cdim, config), ResidualBlock(last, last, cdim, config),
        ))
        self.up = nn.ModuleList([
            nn.ModuleList((
                ResidualBlock(2 * skip, out, cdim, config),
                ResidualBlock(out, out, cdim, config),
                nn.ConvTranspose1d(out, out, 4, stride=2, padding=1),
            )) for out, skip in reversed(pairs[1:])
        ])
        self.final = nn.Sequential(
            ConvBlock(config.down_dims[0], config.down_dims[0], config.kernel_size, config.num_groups),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(
        self, actions: torch.Tensor, times: torch.Tensor | float,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dim:
            raise ValueError("actions must have shape [B,H,action_dim]")
        batch, horizon, _ = actions.shape
        if horizon < 1 or global_cond.shape != (batch, self.config.condition_dim):
            raise ValueError("horizon must be positive and global_cond must have shape [B,condition_dim]")
        if not isinstance(times, torch.Tensor):
            times = torch.full((batch,), float(times), device=actions.device, dtype=actions.dtype)
        if times.shape != (batch,) or not times.is_floating_point():
            raise ValueError("Flow times must be floating point with shape [B]")
        condition = torch.cat((self.time(times), global_cond), dim=-1)
        padding = (-horizon) % self.config.temporal_multiple
        value = F.pad(actions.transpose(1, 2), (0, padding))
        skips = []
        for first, second, down in self.down:
            value = second(first(value, condition), condition)
            skips.append(value)
            value = down(value)
        for layer in self.mid:
            value = layer(value, condition)
        for first, second, up in self.up:
            value = torch.cat((value, skips.pop()), dim=1)
            value = up(second(first(value, condition), condition))
        return self.final(value).transpose(1, 2)[:, :horizon]


class FMActionHead(nn.Module):
    """Own the flow objective and solver; consume only condition and actions.

    loss() takes normalized target action chunks. sample() returns normalized
    action chunks. The policy, not this module, owns action normalization.
    """

    def __init__(self, config: FMHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.velocity_net = ConditionalUnet1D(config)

    def loss(
        self, normalized_actions: torch.Tensor, global_cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        expected = (global_cond.shape[0], self.config.horizon, self.config.action_dim)
        if normalized_actions.shape != expected:
            raise ValueError(f"Expected normalized actions of shape {expected}")
        if not torch.isfinite(normalized_actions).all():
            raise ValueError("Flow Matching action targets must be finite")
        if not torch.all((normalized_actions >= -1.00001) & (normalized_actions <= 1.00001)):
            raise ValueError("Flow Matching action targets must lie in [-1, 1]")
        source = torch.randn(
            normalized_actions.shape, device=normalized_actions.device,
            dtype=normalized_actions.dtype, generator=generator,
        ) * self.config.noise_std
        times = torch.rand(
            normalized_actions.shape[0], device=normalized_actions.device,
            dtype=normalized_actions.dtype, generator=generator,
        )
        time = times[:, None, None]
        interpolant = (1 - time) * source + time * normalized_actions
        target_velocity = normalized_actions - source
        predicted_velocity = self.velocity_net(interpolant, times, global_cond)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self, global_cond: torch.Tensor, generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        steps = self.config.inference_steps if inference_steps is None else inference_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("inference_steps must be a positive integer")
        if global_cond.ndim != 2 or global_cond.shape[1] != self.config.condition_dim:
            raise ValueError("global_cond must have shape [B,condition_dim]")
        parameter = next(self.velocity_net.parameters())
        actions = torch.randn(
            (global_cond.shape[0], self.config.horizon, self.config.action_dim),
            device=parameter.device, dtype=parameter.dtype, generator=generator,
        ) * self.config.noise_std
        step_size = 1.0 / steps
        for step in range(steps):
            times = torch.full(
                (actions.shape[0],), step * step_size,
                device=actions.device, dtype=actions.dtype,
            )
            actions = actions + step_size * self.velocity_net(actions, times, global_cond)
        if not torch.isfinite(actions).all():
            raise RuntimeError("Flow Matching sampler produced non-finite actions")
        return actions.clamp(-1, 1) if self.config.clip_actions else actions

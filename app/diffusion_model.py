"""Small unconditional CIFAR-10 DDPM with explicitly managed training gradients.

The network predicts additive Gaussian noise, not a class label. The cosine
schedule follows Nichol & Dhariwal (2021); reverse sampling uses the DDPM
posterior mean with a clipped clean-image estimate (Ho et al., 2020).
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_VERSION = 1
IMAGE_SIZE = 32
DEFAULT_TIMESTEPS = 100
DEFAULT_BASE_CHANNELS = 16
MODEL_NAME = "assignment4-cifar10-diffusion"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/cifar10_diffusion.pt"


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int = 64,
                         max_period: float = 10000.0) -> torch.Tensor:
    """Interleaved sin/cos, omega_i = max_period ** (-2*i/dimension)."""
    if dimension < 2 or dimension % 2:
        raise ValueError("dimension must be a positive even integer")
    frequencies = torch.exp(-math.log(max_period) * torch.arange(
        dimension // 2, device=timesteps.device, dtype=torch.float32
    ) / (dimension // 2))
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class DiffusionUNet(nn.Module):
    """32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32, with skip connections."""
    def __init__(self, base_channels: int = DEFAULT_BASE_CHANNELS):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight")
        self.base_channels = base_channels
        c, time_dim = base_channels, base_channels * 4
        self.time_mlp = nn.Sequential(nn.Linear(64, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.encoder1 = TimeResidualBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.encoder2 = TimeResidualBlock(c * 2, c * 2, time_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.middle1 = TimeResidualBlock(c * 4, c * 4, time_dim)
        self.middle2 = TimeResidualBlock(c * 4, c * 4, time_dim)
        self.up2 = nn.Conv2d(c * 4, c * 2, 3, padding=1)
        self.decoder2 = TimeResidualBlock(c * 4, c * 2, time_dim)
        self.up1 = nn.Conv2d(c * 2, c, 3, padding=1)
        self.decoder1 = TimeResidualBlock(c * 2, c, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(sinusoidal_embedding(timesteps))
        skip1 = self.encoder1(self.input(noisy_images), time)
        skip2 = self.encoder2(self.down1(skip1), time)
        h = self.middle2(self.middle1(self.down2(skip2), time), time)
        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.decoder2(torch.cat((h, skip2), dim=1), time)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(self.decoder1(torch.cat((h, skip1), dim=1), time))


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = DEFAULT_TIMESTEPS):
        super().__init__()
        if type(timesteps) is not int or not 2 <= timesteps <= 1000:
            raise ValueError("timesteps must be an integer from 2 to 1000")
        self.timesteps = timesteps
        grid = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        curve = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        for name, value in {
            "betas": betas, "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "posterior_variance": betas * (1 - previous) / (1 - alpha_bar),
            "posterior_x0_coefficient": betas * previous.sqrt() / (1 - alpha_bar),
            "posterior_xt_coefficient": (1 - previous) * alphas.sqrt() / (1 - alpha_bar),
        }.items():
            self.register_buffer(name, value.float())

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (self.sqrt_alpha_bar[timesteps, None, None, None] * clean
                + self.sqrt_one_minus_alpha_bar[timesteps, None, None, None] * noise)


class DiffusionUnavailableError(RuntimeError):
    """The trained CIFAR-10 diffusion model is missing or invalid."""


@lru_cache(maxsize=1)
def load_diffusion() -> tuple[DiffusionUNet, DiffusionSchedule]:
    path = Path(os.environ.get("DIFFUSION_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["architecture_version"] != ARCHITECTURE_VERSION
                or checkpoint["epochs"] < 1 or checkpoint["train_count"] != 50000
                or checkpoint["prediction_type"] != "epsilon"):
            raise ValueError("Incompatible or untrained diffusion checkpoint")
        with torch.device("meta"):
            model = DiffusionUNet(checkpoint["base_channels"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
        if any(not value.isfinite().all().item() for value in model.state_dict().values()):
            raise ValueError("Non-finite weights")
        model.eval().requires_grad_(False)
        schedule = DiffusionSchedule(checkpoint["timesteps"])
    except Exception as exc:
        raise DiffusionUnavailableError(
            "The trained CIFAR-10 diffusion checkpoint is unavailable or incompatible. "
            "Restore models/cifar10_diffusion.pt or run scripts/train_cifar10_diffusion.py."
        ) from exc
    return model, schedule


@torch.inference_mode()
def sample_diffusion(model: DiffusionUNet, schedule: DiffusionSchedule,
                     count: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Ancestral DDPM sampling with a private CPU RNG; no global RNG mutation.

    Model and schedule must already be on `device` in evaluation mode. Returns
    CPU images in [-1, 1]. The seed chooses noise, not a CIFAR-10 class.
    """
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("count must be an integer from 1 to 16")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 4294967295")
    if model.training:
        raise ValueError("The diffusion model must be in evaluation mode")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    shape = (count, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = torch.randn(shape, generator=rng).to(device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((count,), step, dtype=torch.long, device=device)
        predicted_noise = model(x, t)
        x0 = ((x - schedule.sqrt_one_minus_alpha_bar[step] * predicted_noise)
              / schedule.sqrt_alpha_bar[step]).clamp(-1, 1)
        mean = (schedule.posterior_x0_coefficient[step] * x0
                + schedule.posterior_xt_coefficient[step] * x)
        if step:
            noise = torch.randn(shape, generator=rng).to(device)
            x = mean + schedule.posterior_variance[step].sqrt() * noise
        else:
            x = mean
    if not x.isfinite().all().item():
        raise DiffusionUnavailableError("The diffusion model produced non-finite images")
    return x.clamp(-1, 1).cpu()

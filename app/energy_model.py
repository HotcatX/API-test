"""CIFAR-10 energy model and finite-step, input-gradient image sampling.

The model returns E(x): lower energies are preferred. Sampling differentiates
with respect to the current image, while learning differentiates with respect
to the CNN weights. A fixed quadratic neighbor prior suppresses fine pixel
noise. This is a small teaching model, not a calibrated density.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path

import torch
from torch import nn

ARCHITECTURE_VERSION = 2
MODEL_NAME = "assignment4-cifar10-energy"
BASE_CHANNELS = 32
DEFAULT_STEPS = 100
STEP_SIZE = 10.0
NOISE_STD = 0.005
GRADIENT_CLIP = 0.03
SMOOTHNESS_WEIGHT = 0.002
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "cifar10_energy.pt"


class EnergyUnavailableError(RuntimeError):
    """A compatible trained energy checkpoint could not be loaded or sampled."""


class EnergyModel(nn.Module):
    """CNN plus fixed quadratic neighbor prior -> one energy per RGB image.

    The reference term discourages residual high-frequency pixel noise that a
    small strided CNN alone can miss. It is fixed, not learned image realism.
    """

    def __init__(self, base_channels: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.base_channels = base_channels
        widths = [base_channels // 2, base_channels, base_channels * 2, base_channels * 2]
        self.layers = nn.Sequential(
            nn.Conv2d(3, widths[0], 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(widths[2], widths[3], 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(widths[3] * 2 * 2, widths[3]), nn.SiLU(),
            nn.Linear(widths[3], 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).squeeze(-1) + SMOOTHNESS_WEIGHT * spatial_quadratic(images)


def spatial_quadratic(images: torch.Tensor) -> torch.Tensor:
    """Sum squared horizontal/vertical neighbor differences per RGB image."""
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).square().sum((1, 2, 3))
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).square().sum((1, 2, 3))
    return horizontal + vertical


def langevin_sample(
    model: nn.Module,
    initial: torch.Tensor,
    *,
    steps: int = DEFAULT_STEPS,
    step_size: float = STEP_SIZE,
    noise_std: float = NOISE_STD,
    generator: torch.Generator | None = None,
    gradient_clip: float | None = GRADIENT_CLIP,
) -> torch.Tensor:
    """Update images using a clipped, finite-step Langevin-style chain.

    x <- clamp(x - step_size * clip(dE/dx) + noise_std * N(0,I), -1,1)

    The small noise and chosen drift are practical short-run hyperparameters,
    not a claim of exact equilibrium draws from exp(-E). No parameter values,
    parameter .grad fields, training modes, or requires_grad flags are changed.
    This works inside no_grad; inference_mode is deliberately unsupported
    because the sampler needs an autograd graph for its input images.
    """
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("steps must be an integer from 1 to 1000")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("gradient_clip must be finite and positive, or None")
    if torch.is_inference_mode_enabled():
        raise ValueError("Energy sampling requires autograd; use no_grad, not inference_mode")
    if not initial.is_floating_point() or not torch.isfinite(initial).all().item():
        raise ValueError("initial images must be finite floating-point tensors")
    # Clone so callers' tensors (including replay-buffer slices) are unchanged.
    images = initial.detach().clone().clamp(-1, 1)
    random_device = generator.device if generator is not None else images.device
    for _ in range(steps):
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)
            energies = model(images)
            # Only the input derivative is requested; parameter .grad stays intact.
            gradient, = torch.autograd.grad(energies.sum(), images, create_graph=False)
        with torch.no_grad():
            if gradient_clip is not None:
                gradient = gradient.clamp(-gradient_clip, gradient_clip)
            images = images - step_size * gradient
            if noise_std:
                noise = torch.randn(images.shape, device=random_device,
                                    dtype=images.dtype, generator=generator).to(images.device)
                images = images + noise_std * noise
            images.clamp_(-1, 1)
    return images.detach()


@lru_cache(maxsize=1)
def load_energy_model() -> EnergyModel:
    """Load CPU weights once without consuming global random numbers."""
    path = Path(os.environ.get("ENERGY_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    try:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if (checkpoint["architecture_version"] != ARCHITECTURE_VERSION
                or checkpoint["base_channels"] != BASE_CHANNELS
                or checkpoint["smoothness_weight"] != SMOOTHNESS_WEIGHT
                or checkpoint["epochs"] < 1
                or checkpoint["image_shape"] != [3, 32, 32]):
            raise ValueError("Incompatible trained energy model")
        with torch.device("meta"):
            model = EnergyModel()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
        if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()):
            raise ValueError("Non-finite energy weights")
        model.eval().requires_grad_(False)
    except Exception as exc:
        raise EnergyUnavailableError(
            "The trained CIFAR-10 energy checkpoint is unavailable or incompatible. "
            "Restore models/cifar10_energy.pt or run "
            "'uv run python scripts/train_cifar10_energy.py'."
        ) from exc
    return model


def sample_energy(count: int = 1, seed: int = 42, steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Generate RGB images from fresh uniform noise using a private CPU RNG."""
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("count must be an integer from 1 to 16")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1")
    model = load_energy_model()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.rand(count, 3, 32, 32, generator=generator) * 2 - 1
    output = langevin_sample(model, initial, steps=steps, generator=generator)
    if output.shape != (count, 3, 32, 32) or not torch.isfinite(output).all().item():
        raise EnergyUnavailableError("The energy model produced invalid images")
    with torch.no_grad():
        if not torch.isfinite(model(output)).all().item():
            raise EnergyUnavailableError("The energy model produced non-finite output energies")
    return output

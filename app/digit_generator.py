"""Generate MNIST-like images using the trained Assignment 3 GAN on CPU."""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import secrets

from PIL import Image
import torch

from app.gan_model import ARCHITECTURE_VERSION, LATENT_DIM, Generator

MODEL_NAME = "assignment3-mnist-gan"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "mnist_gan.pt"
MAX_IMAGES = 64
MAX_SEED = 2**32 - 1


class GeneratorUnavailableError(RuntimeError):
    """The trained checkpoint cannot provide valid generated images."""


@dataclass(frozen=True)
class GeneratedDigits:
    png: bytes
    seed: int
    count: int
    width: int
    height: int


@lru_cache(maxsize=1)
def load_generator() -> Generator:
    """Load once, without training or downloading models during a request."""
    path = Path(os.environ.get("GAN_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["architecture_version"] != ARCHITECTURE_VERSION:
            raise ValueError("Unexpected GAN architecture")
        if checkpoint["latent_dim"] != LATENT_DIM or checkpoint["epochs"] < 1:
            raise ValueError("A compatible trained checkpoint is required")
        # Construct without random CPU initialization, then attach the trained
        # CPU tensors. Cold loading therefore leaves the global RNG unchanged.
        with torch.device("meta"):
            model = Generator()
        model.load_state_dict(checkpoint["generator_state_dict"], strict=True, assign=True)
        for value in model.state_dict().values():
            if not torch.isfinite(value).all().item():
                raise ValueError("Non-finite model state")
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if (module.running_var < 0).any().item():
                    raise ValueError("Invalid BatchNorm running variance")
        model.eval()
        model.requires_grad_(False)
    except Exception as exc:
        raise GeneratorUnavailableError(
            "The trained MNIST GAN checkpoint is unavailable or incompatible. "
            "Restore models/mnist_gan.pt or run "
            "'uv run python scripts/train_mnist_gan.py'."
        ) from exc
    return model


def generate_digits(count: int = 1, seed: int | None = None) -> GeneratedDigits:
    """Return an unconditioned digit or a grid of independent generated samples.

    A private CPU random generator avoids changing the process-wide RNG. The
    same seed and count reproduce the same PNG with the same checkpoint/runtime.
    The seed controls noise, not which digit class is generated.
    """
    if type(count) is not int or not 1 <= count <= MAX_IMAGES:
        raise ValueError(f"count must be an integer from 1 to {MAX_IMAGES}.")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= MAX_SEED):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}.")
    used_seed = secrets.randbelow(MAX_SEED + 1) if seed is None else seed
    model = load_generator()
    rng = torch.Generator(device="cpu").manual_seed(used_seed)
    noise = torch.randn(count, LATENT_DIM, generator=rng)
    with torch.inference_mode():
        output = model(noise)
    if output.shape != (count, 1, 28, 28) or not torch.isfinite(output).all().item():
        raise GeneratorUnavailableError("The generator produced invalid images.")
    # Tanh output [-1, 1] becomes a grayscale uint8 PNG, with no per-image stretch.
    pixels = ((output[:, 0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    width, height = columns * 28 + (columns - 1) * 2, rows * 28 + (rows - 1) * 2
    grid = Image.new("L", (width, height), color=0)
    for index, sample in enumerate(pixels):
        grid.paste(Image.fromarray(sample), ((index % columns) * 30, (index // columns) * 30))
    buffer = BytesIO()
    grid.save(buffer, format="PNG")
    return GeneratedDigits(buffer.getvalue(), used_seed, count, width, height)

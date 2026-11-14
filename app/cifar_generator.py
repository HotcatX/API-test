"""PNG responses for the trained Assignment 4 CIFAR-10 generators."""

from dataclasses import dataclass
from io import BytesIO
import math
import secrets
from typing import Literal

from PIL import Image
import torch

from app.diffusion_model import load_diffusion, sample_diffusion
from app.energy_model import sample_energy

MAX_IMAGES = 16
MAX_SEED = 2**32 - 1
GeneratorKind = Literal["diffusion", "energy"]


@dataclass(frozen=True)
class GeneratedImages:
    png: bytes
    seed: int
    count: int
    model: str


def rgb_grid_png(images: torch.Tensor) -> bytes:
    """Encode [-1,1] RGB samples without stretching their individual ranges."""
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("Expected a batch of 32 x 32 RGB images.")
    if not 1 <= len(images) <= MAX_IMAGES or not torch.isfinite(images).all().item():
        raise ValueError("The generator returned an invalid image batch.")
    pixels = ((images.detach().cpu().clamp(-1, 1) + 1) * 127.5).round()
    pixels = pixels.to(torch.uint8).permute(0, 2, 3, 1).numpy()
    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * 34 - 2, rows * 34 - 2))
    for index, sample in enumerate(pixels):
        grid.paste(Image.fromarray(sample), ((index % columns) * 34, (index // columns) * 34))
    buffer = BytesIO()
    grid.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_cifar_images(
    kind: GeneratorKind, count: int = 1, seed: int | None = None,
) -> GeneratedImages:
    """Generate from random noise using trained weights, with request-local RNG."""
    if kind not in ("diffusion", "energy"):
        raise ValueError("Unknown generator.")
    if type(count) is not int or not 1 <= count <= MAX_IMAGES:
        raise ValueError(f"count must be an integer from 1 to {MAX_IMAGES}.")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= MAX_SEED):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}.")
    used_seed = secrets.randbelow(MAX_SEED + 1) if seed is None else seed
    if kind == "diffusion":
        model, schedule = load_diffusion()
        samples = sample_diffusion(model, schedule, count=count, seed=used_seed, device="cpu")
    else:
        # EBM sampling must enable input gradients internally. Do not enclose
        # this call in inference_mode: model weights are frozen, images are not.
        samples = sample_energy(count=count, seed=used_seed)
    return GeneratedImages(
        png=rgb_grid_png(samples), seed=used_seed, count=count,
        model=f"assignment4-cifar10-{kind}",
    )

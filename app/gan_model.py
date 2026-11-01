"""The exact MNIST GAN architecture specified in Assignment 3.

The generator maps standard-normal noise to images in [-1, 1]. The
discriminator returns a probability, so training uses binary cross-entropy.
"""

from __future__ import annotations

import torch
from torch import nn

LATENT_DIM = 100
ARCHITECTURE_VERSION = 1


class Generator(nn.Module):
    """Unconditional generator: (B, 100) -> (B, 1, 28, 28)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 7 * 7 * 128)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        features = self.fc(noise).reshape(-1, 128, 7, 7)
        return self.layers(features)


class Discriminator(nn.Module):
    """Discriminator: (B, 1, 28, 28) -> (B, 1) real-image probability."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def initialize_weights(module: nn.Module) -> None:
    """Initialize trainable layers using the standard DCGAN scale."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)

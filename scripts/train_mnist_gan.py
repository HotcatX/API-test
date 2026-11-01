"""Train the Assignment 3 GAN on all 60,000 MNIST training images.

From the repository root:
    .venv/bin/python scripts/train_mnist_gan.py --epochs 20

The download is cached in ignored data/. Labels are not used because the
specified GAN is unconditional. No test images are used for training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gan_model import (
    ARCHITECTURE_VERSION, LATENT_DIM, Discriminator, Generator,
    initialize_weights,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume the local training state in tmp/mnist_gan/.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive; batch-size must be at least 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = PROJECT_ROOT / "assignment3"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = PROJECT_ROOT / "models" / "mnist_gan.pt"
    resume_path = PROJECT_ROOT / "tmp" / "mnist_gan" / "training_state.pt"
    print(f"Device={device}; seed={args.seed}; batch_size={args.batch_size}; epochs={args.epochs}", flush=True)
    dataset = MNIST(root=str(PROJECT_ROOT / "data"), train=True, download=True)
    # Exactly equivalent to ToTensor() followed by Normalize((0.5,), (0.5,));
    # preprocessing once avoids repeated PIL conversion during every epoch.
    images = dataset.data.unsqueeze(1).float().div_(127.5).sub_(1.0)
    train_loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True,
                        generator=train_loader_generator, num_workers=0, drop_last=False)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise_seed = 2026
    fixed_noise = torch.randn(64, LATENT_DIM, generator=torch.Generator().manual_seed(fixed_noise_seed)).to(device)
    history = []
    start_epoch = 1
    started_at = datetime.now(timezone.utc).isoformat()
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key, expected in [("seed", args.seed), ("batch_size", args.batch_size),
                              ("learning_rate", args.learning_rate), ("architecture_version", ARCHITECTURE_VERSION)]:
            if state[key] != expected:
                parser.error(f"resume requires matching {key}: {state[key]!r}")
        generator.load_state_dict(state["generator_state_dict"])
        discriminator.load_state_dict(state["discriminator_state_dict"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        torch.set_rng_state(state["cpu_rng_state"])
        train_loader_generator.set_state(state["loader_rng_state"])
        if device.type == "mps" and "mps_rng_state" in state:
            torch.mps.set_rng_state(state["mps_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        history = state["history"]
        start_epoch = state["epochs"] + 1
        started_at = state["trained_at_utc"]
        if args.epochs < start_epoch:
            parser.error("--epochs must exceed the completed epoch count when resuming")
        print(f"Resuming from completed epoch {start_epoch - 1}", flush=True)

    print(f"Training images={len(dataset)}; batches/epoch={len(loader)}; G parameters="
          f"{sum(p.numel() for p in generator.parameters())}; D parameters="
          f"{sum(p.numel() for p in discriminator.parameters())}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        generator.train()
        discriminator.train()
        sums = torch.zeros(4, device=device)
        count = 0
        for (real,) in loader:
            real = real.to(device)
            size = real.size(0)
            real_labels = torch.ones(size, 1, device=device)
            fake_labels = torch.zeros(size, 1, device=device)

            # D learns to separate real images from detached generated images.
            discriminator.requires_grad_(True)
            optimizer_d.zero_grad(set_to_none=True)
            fake = generator(torch.randn(size, LATENT_DIM, device=device))
            real_probability = discriminator(real)
            fake_probability = discriminator(fake.detach())
            loss_d = criterion(real_probability, real_labels) + criterion(fake_probability, fake_labels)
            loss_d.backward()
            optimizer_d.step()

            # Non-saturating G objective: generated images should be labeled real.
            # Freezing D's parameters still allows gradients through D into G.
            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            generated_probability = discriminator(fake)
            loss_g = criterion(generated_probability, real_labels)
            loss_g.backward()
            optimizer_g.step()
            sums += torch.stack((loss_d.detach(), loss_g.detach(),
                                 real_probability.detach().mean(), fake_probability.detach().mean())) * size
            count += size
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        values = (sums / count).cpu().tolist()
        if not all(np.isfinite(v) for v in values):
            raise RuntimeError(f"Non-finite loss/probability at epoch {epoch}: {values}")
        record = {"epoch": epoch, "images_seen": count,
                  "discriminator_loss": values[0], "generator_loss": values[1],
                  "mean_d_real": values[2], "mean_d_fake_before_g_update": values[3],
                  "duration_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        checkpoint = {
            "generator_state_dict": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
            "discriminator_state_dict": {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
            "latent_dim": LATENT_DIM, "architecture_version": ARCHITECTURE_VERSION,
            "epochs": epoch, "seed": args.seed, "train_count": len(dataset),
            "dataset": "MNIST official training split", "normalization": "uint8 / 127.5 - 1",
            "noise_distribution": "standard normal N(0,1)", "conditional": False,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "adam_betas": [0.5, 0.999], "trained_at_utc": started_at,
            "pytorch_version": str(torch.__version__), "training_device": str(device),
        }
        save_checkpoint(checkpoint_path, checkpoint)
        state = dict(checkpoint, optimizer_g=optimizer_g.state_dict(), optimizer_d=optimizer_d.state_dict(),
                     cpu_rng_state=torch.get_rng_state(), loader_rng_state=train_loader_generator.get_state(),
                     history=history)
        if device.type == "mps":
            state["mps_rng_state"] = torch.mps.get_rng_state()
        elif device.type == "cuda":
            state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        save_checkpoint(resume_path, state)
        generator.eval()
        with torch.inference_mode():
            fixed_images = generator(fixed_noise).cpu().add(1.0).mul(0.5).clamp(0, 1)
        save_image(fixed_images, output_dir / "generated_samples.png", nrow=8, padding=2, pad_value=0.0)
        if epoch in (1, 5, 10, 20):
            save_image(fixed_images, output_dir / f"generated_samples_epoch_{epoch:02d}.png", nrow=8, padding=2, pad_value=0.0)
        write_json(output_dir / "training_history.json", history)
        metrics = {k: v for k, v in checkpoint.items() if not k.endswith("state_dict")}
        metrics.update({"generator_parameters": sum(p.numel() for p in generator.parameters()),
                        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
                        "batches_per_epoch": len(loader), "training_seconds": sum(r["duration_seconds"] for r in history),
                        "final_losses": record, "sample_grid_count": 64,
                        "sample_grid_noise_seed": fixed_noise_seed, "sample_selection": "First 64 noise draws using fixed CPU seed 2026; no outcome-based selection.",
                        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                        "limitations": ["Unconditional GAN: the API cannot request a particular digit class.",
                                        "Losses are adversarial training diagnostics, not classification accuracy or proof of image quality.",
                                        "The sample grid is a qualitative check; no FID, precision/recall, or coverage guarantee is claimed.",
                                        "A fixed seed helps reproducibility, but floating-point/device differences may change training results."]})
        write_json(output_dir / "training_metrics.json", metrics)
        print(f"Epoch {epoch:02d}/{args.epochs}: D={values[0]:.4f}, G={values[1]:.4f}, "
              f"D(real)={values[2]:.3f}, D(fake)={values[3]:.3f}, "
              f"images={count}, seconds={record['duration_seconds']:.1f}", flush=True)
    print(f"Saved portable checkpoint to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()

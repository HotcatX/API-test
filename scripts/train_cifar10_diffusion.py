"""Train a compact epsilon-predicting DDPM on all 50,000 CIFAR-10 train images.

Run from the project virtual environment:
    .venv/bin/python scripts/train_cifar10_diffusion.py --epochs 20

No test image is used for gradients. A fixed 1,024-image test diagnostic is
reported after training; it is denoising MSE, not generative image quality.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.diffusion_model import (
    ARCHITECTURE_VERSION, DEFAULT_BASE_CHANNELS, DEFAULT_TIMESTEPS,
    DiffusionUNet, DiffusionSchedule, sample_diffusion,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_state(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, schedule, images, device, batch_size):
    rng = torch.Generator().manual_seed(7341)
    total, zero_total, count = 0.0, 0.0, 0
    model.eval()
    for clean_cpu in images.split(batch_size):
        clean = clean_cpu.to(device)
        noise = torch.randn(clean_cpu.shape, generator=rng).to(device)
        timesteps = torch.randint(schedule.timesteps, (len(clean),), generator=rng).to(device)
        predicted = model(schedule.add_noise(clean, noise, timesteps), timesteps)
        total += F.mse_loss(predicted, noise).item() * len(clean)
        zero_total += noise.square().mean().item() * len(clean)
        count += len(clean)
    return {"epsilon_mse": total / count, "zero_predictor_mse": zero_total / count,
            "count": count, "noise_seed": 7341}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--probe-batches", type=int, default=0, help="Time a short probe; do not overwrite trained checkpoint.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.probe_batches < 0:
        parser.error("epochs, batch-size, learning-rate must be positive; probe-batches nonnegative")
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = CIFAR10(str(ROOT / "data"), train=True, download=True)
    images = torch.from_numpy(data.data).permute(0, 3, 1, 2).contiguous().float().div_(127.5).sub_(1)
    loader_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True,
                        generator=loader_rng, num_workers=0, drop_last=False)
    if args.probe_batches > len(loader):
        parser.error(f"--probe-batches cannot exceed one epoch ({len(loader)} batches)")
    model = DiffusionUNet(args.base_channels).to(device)
    schedule = DiffusionSchedule(args.timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = 0.995
    ema_params, params = list(ema.parameters()), list(model.parameters())
    output = ROOT / "assignment4"
    checkpoint_path = ROOT / "models/cifar10_diffusion.pt"
    resume_path = ROOT / "tmp/cifar10_diffusion/training_state.pt"
    history, start_epoch = [], 1
    started_at = datetime.now(timezone.utc).isoformat()
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key, expected in (("seed", args.seed), ("batch_size", args.batch_size),
                              ("learning_rate", args.learning_rate), ("base_channels", args.base_channels),
                              ("timesteps", args.timesteps)):
            if state[key] != expected:
                parser.error(f"resume must match {key}={state[key]}")
        model.load_state_dict(state["raw_state_dict"])
        ema.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        loader_rng.set_state(state["loader_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, start_epoch = state["history"], state["epochs"] + 1
        started_at = state["trained_at_utc"]
        if args.epochs < start_epoch:
            parser.error("--epochs must exceed completed epochs when resuming")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"DDPM: device={device}, n={len(images)}, params={parameter_count}, "
          f"batch={args.batch_size}, T={args.timesteps}, epochs={args.epochs}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        begin = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device)
        count, batches = 0, 0
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device)
            # CIFAR-10 classes are ignored. Horizontal flips are training-only.
            flip = torch.rand(len(clean), 1, 1, 1, device=device) < 0.5
            clean = torch.where(flip, clean.flip(-1), clean)
            timesteps = torch.randint(args.timesteps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = schedule.add_noise(clean, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            # Images/noise have requires_grad=False; gradients train weights.
            loss = F.mse_loss(model(noisy, timesteps), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_params, ema_decay)
                torch._foreach_add_(ema_params, params, alpha=1 - ema_decay)
            total += loss.detach() * len(clean)
            count += len(clean)
            batches += 1
            if args.probe_batches and batches == args.probe_batches:
                synchronize(device)
                seconds = time.perf_counter() - begin
                probe = {"batches": batches, "seconds": seconds, "images_seen": count,
                         "estimated_epoch_seconds": seconds * len(loader) / batches,
                         "loss": (total / count).item(), "base_channels": args.base_channels,
                         "batch_size": args.batch_size, "parameters": parameter_count,
                         "device": str(device)}
                write_json(output / "diffusion_probe.json", probe)
                print(json.dumps(probe), flush=True)
                return
        synchronize(device)
        value = (total / count).item()
        if not np.isfinite(value):
            raise RuntimeError("Non-finite training loss")
        row = {"epoch": epoch, "images_seen": count, "epsilon_mse": value,
               "duration_seconds": time.perf_counter() - begin}
        history.append(row)
        checkpoint = {
            "model_state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "architecture_version": ARCHITECTURE_VERSION, "base_channels": args.base_channels,
            "prediction_type": "epsilon", "timesteps": args.timesteps, "schedule": "cosine, s=0.008, max_beta=0.999",
            "epochs": epoch, "train_count": len(images), "dataset": "CIFAR-10 official training split",
            "image_size": 32, "conditional": False, "normalization": "uint8 / 127.5 - 1",
            "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "optimizer": "AdamW", "weight_decay": 1e-4, "ema_decay": ema_decay,
            "trained_at_utc": started_at, "training_device": str(device), "pytorch_version": str(torch.__version__),
        }
        save_state(checkpoint_path, checkpoint)
        resume = dict(checkpoint, raw_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), loader_rng=loader_rng.get_state(), history=history)
        if device.type == "mps":
            resume["mps_rng"] = torch.mps.get_rng_state()
        if device.type == "cuda":
            resume["cuda_rng"] = torch.cuda.get_rng_state_all()
        save_state(resume_path, resume)
        write_json(output / "diffusion_history.json", history)
        metrics = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        metrics.update({"parameters": parameter_count, "batches_per_epoch": len(loader),
                        "training_seconds": sum(r["duration_seconds"] for r in history),
                        "final_training_mse": value, "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                        "sample_seed": 2026, "sample_count": 16,
                        "sample_selection": "First 16 samples from private seed 2026; no outcome-based selection.",
                        "limitations": ["Compact short training run; samples may be blurry or not recognizable.",
                                        "Unconditional model; no class selection or coverage guarantee.",
                                        "Denoising MSE measures noise-prediction performance, not perceptual quality; FID not measured.",
                                        "Reproducibility is within the same checkpoint, device and runtime; not bitwise across hardware."]})
        write_json(output / "diffusion_metrics.json", metrics)
        print(f"Epoch {epoch:02d}/{args.epochs}: epsilon MSE={value:.5f}, "
              f"images={count}, seconds={row['duration_seconds']:.1f}", flush=True)
        if epoch == 1 or epoch == args.epochs:
            samples = sample_diffusion(ema, schedule, 16, 2026, device=device)
            save_image((samples + 1) / 2, output / f"diffusion_samples_epoch_{epoch:02d}.png", nrow=4)
            save_image((samples + 1) / 2, output / "diffusion_samples.png", nrow=4)
    test_data = CIFAR10(str(ROOT / "data"), train=False, download=True)
    test_images = torch.from_numpy(test_data.data[:1024]).permute(0, 3, 1, 2).contiguous().float().div_(127.5).sub_(1)
    diagnostics = evaluate(ema, schedule, test_images, device, args.batch_size)
    metrics["held_out_denoising"] = dict(diagnostics, split="first 1024 official test images; evaluated after training")
    write_json(output / "diffusion_metrics.json", metrics)
    print(f"Held-out diagnostic: {diagnostics}", flush=True)
    print(f"Saved trained DDPM checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()

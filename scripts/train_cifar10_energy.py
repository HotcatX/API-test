"""Train a small CIFAR-10 EBM with persistent, finite-step contrastive divergence.

Run from the project root: uv run python scripts/train_cifar10_energy.py
The complete 50,000-image CIFAR-10 training split is used. No test examples or
image labels enter training, replay initialization, or generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.energy_model import (
    ARCHITECTURE_VERSION, BASE_CHANNELS, DEFAULT_STEPS, GRADIENT_CLIP,
    NOISE_STD, STEP_SIZE, SMOOTHNESS_WEIGHT, EnergyModel, langevin_sample,
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sampling_steps, args.replay_size) < 1:
        parser.error("counts must be positive")
    if args.replay_size < args.batch_size or args.sampling_steps > 1000:
        parser.error("replay size must fit a batch; sampling steps <=1000")
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    output = ROOT / "assignment4"
    output.mkdir(exist_ok=True)
    checkpoint_path = ROOT / "models/cifar10_energy.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    dataset = CIFAR10(str(ROOT / "data"), train=True, download=True)
    # Keep uint8 data in CPU memory; normalization is done for each selected batch.
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.0, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    replay = torch.rand(args.replay_size, 3, 32, 32, generator=generator) * 2 - 1
    history = []
    start = time.perf_counter()
    print(f"Energy model: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"device={device}; training images={len(data)}; epochs={args.epochs}", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = torch.randperm(len(data), generator=generator)
        sums = {name: 0.0 for name in ("loss", "contrastive_loss", "regularization", "real_energy", "negative_energy")}
        count = 0
        for batch_index, offset in enumerate(range(0, len(data), args.batch_size), 1):
            real = data[order[offset:offset + args.batch_size]].float().div_(127.5).sub_(1)
            n = len(real)
            # Dequantization/smoothing noise discourages sensitivity to byte levels.
            real = (real + NOISE_STD * torch.randn(real.shape, generator=generator)).clamp(-1, 1).to(device)
            indices = torch.randperm(args.replay_size, generator=generator)[:n]
            initial = replay[indices].clone()
            refresh = torch.rand(n, generator=generator) < 0.05
            fresh = torch.rand(n, 3, 32, 32, generator=generator) * 2 - 1
            initial[refresh] = fresh[refresh]

            # Phase A: the sampler optimizes x, with all network weights fixed.
            model.eval().requires_grad_(False)
            negative = langevin_sample(model, initial.to(device), steps=args.sampling_steps,
                                       generator=generator)
            replay[indices] = negative.cpu()
            # Phase B: stop the sampling graph and optimize only network weights.
            model.train().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            real_energy, negative_energy = model(torch.cat((real, negative.detach()), 0)).split(n)
            contrastive = real_energy.mean() - negative_energy.mean()
            # These are the full energies, including the fixed spatial prior.
            regularization = 0.1 * (real_energy.square().mean() + negative_energy.square().mean())
            loss = contrastive + regularization
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss; refusing to save invalid weights")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            values = dict(loss=loss.item(), contrastive_loss=contrastive.item(),
                          regularization=regularization.item(), real_energy=real_energy.mean().item(),
                          negative_energy=negative_energy.mean().item())
            for key, value in values.items():
                sums[key] += n * value
            count += n
            if batch_index % 100 == 0:
                print(f"epoch={epoch}/{args.epochs} batch={batch_index}/391 "
                      f"loss={sums['loss']/count:.4f} real={sums['real_energy']/count:.3f} "
                      f"negative={sums['negative_energy']/count:.3f}", flush=True)
        sync(device)
        record = {"epoch": epoch, "examples": count, "learning_rate": optimizer.param_groups[0]["lr"],
                  **{key: value / count for key, value in sums.items()},
                  "duration_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        scheduler.step()
        write_json(output / "energy_training_history.json", history)
        checkpoint = {
            "architecture_version": ARCHITECTURE_VERSION, "base_channels": BASE_CHANNELS,
            "smoothness_weight": SMOOTHNESS_WEIGHT,
            "image_shape": [3, 32, 32], "dataset": "CIFAR-10", "train_count": len(data),
            "epochs": epoch, "seed": args.seed, "sampling_steps_training": args.sampling_steps,
            "sampling_steps_inference": DEFAULT_STEPS, "sampling_step_size": STEP_SIZE,
            "sampling_noise_std": NOISE_STD, "sampling_gradient_clip": GRADIENT_CLIP,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "pytorch_version": str(torch.__version__), "trained_at_utc": started_at,
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        # Fixed, fresh-noise samples, not replay samples or selected successes.
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            model.eval().requires_grad_(False)
            sample_rng = torch.Generator().manual_seed(2026)
            initial = torch.rand(16, 3, 32, 32, generator=sample_rng) * 2 - 1
            samples = langevin_sample(model, initial.to(device), steps=DEFAULT_STEPS, generator=sample_rng).cpu()
            save_image((samples + 1) / 2, output / f"energy_samples_epoch_{epoch:02d}.png", nrow=4, padding=2)
        print(json.dumps(record), flush=True)

    # Diagnostic energy differences are not an image-quality or likelihood score.
    model.eval().requires_grad_(False)
    sample_rng = torch.Generator().manual_seed(2026)
    initial = torch.rand(16, 3, 32, 32, generator=sample_rng) * 2 - 1
    samples = langevin_sample(model, initial.to(device), steps=DEFAULT_STEPS, generator=sample_rng)
    with torch.no_grad():
        initial_energy = model(initial.to(device)).mean().item()
        sample_energy = model(samples).mean().item()
        reference_energy = model(data[:128].float().div(127.5).sub(1).to(device)).mean().item()
    save_image((samples.cpu() + 1) / 2, output / "energy_generated_samples.png", nrow=4, padding=2)
    metrics = {
        "dataset": "CIFAR-10", "dataset_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "train_count": len(data), "test_examples_used": 0, "epochs": args.epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()), "batch_size": args.batch_size,
        "base_channels": BASE_CHANNELS, "image_shape": [3, 32, 32], "seed": args.seed,
        "architecture_version": ARCHITECTURE_VERSION,
        "energy": "E_theta(x) = CNN_theta(x) + 0.002*sum_over_horizontal_and_vertical_neighbors((x_i-x_j)^2)",
        "smoothness_weight": SMOOTHNESS_WEIGHT,
        "smoothness_prior_is_learned": False,
        "regularization_applies_to": "Full energy including the fixed quadratic spatial prior",
        "optimizer": "Adam", "learning_rate": args.learning_rate, "betas": [0.0, 0.999],
        "learning_rate_decay_per_epoch": 0.97, "energy_regularization": 0.1,
        "parameter_gradient_norm_clip": 10.0, "replay_size": args.replay_size,
        "fresh_noise_probability": 0.05, "training_sampling_steps": args.sampling_steps,
        "inference_sampling_steps": DEFAULT_STEPS, "step_size": STEP_SIZE,
        "noise_std": NOISE_STD, "input_gradient_clip": GRADIENT_CLIP,
        "loss": "mean(E(real)) - mean(E(stop_gradient(negative))) + 0.1*(mean(E(real)^2)+mean(E(negative)^2))",
        "sampling": "finite clipped Langevin-style input updates; initial images are uniform[-1,1]",
        "sample_seed": 2026, "samples_from_fresh_noise": True,
        "diagnostics": {"initial_noise_mean_energy": initial_energy, "generated_mean_energy": sample_energy,
                        "first_128_training_images_mean_energy": reference_energy},
        "device": str(device), "pytorch_version": str(torch.__version__),
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "limitations": "Small teaching model trained for a limited number of epochs; finite clipped chains are biased approximate samples, not verified equilibrium draws. Smoothness partly comes from the explicitly fixed quadratic neighbor prior, not necessarily learned object structure. Energy reduction does not establish image realism; no FID or Inception score was measured. MPS floating-point training is not guaranteed bitwise reproducible across hardware or library versions.",
        "method_reference": "https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html",
        "autograd_reference": "https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html",
    }
    write_json(output / "energy_training_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

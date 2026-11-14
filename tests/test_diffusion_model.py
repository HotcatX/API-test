"""Numerical and gradient-boundary checks for the Assignment 4 DDPM."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from app import diffusion_model as dm


def test_sinusoidal_values_and_timestep_changes():
    output = dm.sinusoidal_embedding(torch.tensor([0, 1]), 8)
    assert torch.equal(output[0], torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]))
    expected = torch.tensor([0.841470985, 0.540302306, 0.099833417, 0.995004165,
                             0.009999833, 0.999950000, 0.0009999998, 0.999999500])
    assert torch.allclose(output[1], expected, atol=1e-7)


def test_forward_noise_and_denoising_loss_train_only_model_weights():
    torch.manual_seed(42)
    model = dm.DiffusionUNet(base_channels=8)
    schedule = dm.DiffusionSchedule(10)
    clean, noise = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0, 9])
    noisy = schedule.add_noise(clean, noise, timesteps)
    assert torch.allclose(noisy[0], schedule.alpha_bar[0].sqrt() * clean[0]
                          + (1 - schedule.alpha_bar[0]).sqrt() * noise[0], atol=1e-6)
    prediction = model(noisy, timesteps)
    assert prediction.shape == clean.shape
    loss = F.mse_loss(prediction, noise)
    loss.backward()
    assert not clean.requires_grad and clean.grad is None
    assert not noise.requires_grad and noise.grad is None
    assert model.output[-1].weight.grad.abs().sum().item() > 0
    assert model.time_mlp[0].weight.grad.abs().sum().item() > 0
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


def test_posterior_at_final_step_is_the_clean_estimate():
    schedule = dm.DiffusionSchedule(100)
    assert schedule.posterior_variance[0] == 0
    assert schedule.posterior_xt_coefficient[0] == 0
    assert schedule.posterior_x0_coefficient[0] == 1
    assert (schedule.betas > 0).all() and (schedule.betas < 1).all()
    assert (schedule.alpha_bar[:-1] > schedule.alpha_bar[1:]).all()
    assert schedule.alpha_bar[-1] < 1e-6


def test_sampling_is_private_rng_and_does_not_accumulate_parameter_gradients():
    model = dm.DiffusionUNet(base_channels=8).eval().requires_grad_(False)
    schedule = dm.DiffusionSchedule(3)
    state = torch.get_rng_state().clone()
    first = dm.sample_diffusion(model, schedule, count=2, seed=2026)
    second = dm.sample_diffusion(model, schedule, count=2, seed=2026)
    other = dm.sample_diffusion(model, schedule, count=2, seed=2027)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(first, second) and not torch.equal(first, other)
    assert first.shape == (2, 3, 32, 32)
    assert first.isfinite().all() and -1 <= first.min() <= first.max() <= 1
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("count,seed", [(0, 1), (17, 1), (True, 1), (1, -1), (1, 2**32), (1, True)])
def test_sampler_rejects_unbounded_or_noninteger_requests(count, seed):
    with pytest.raises(ValueError):
        dm.sample_diffusion(dm.DiffusionUNet(8).eval(), dm.DiffusionSchedule(3), count, seed)


def test_missing_checkpoint_reports_unavailability(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFUSION_MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(dm.DiffusionUnavailableError):
        dm.load_diffusion.__wrapped__()


def test_trained_checkpoint_and_cold_loading_are_valid():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models/cifar10_diffusion.pt"
    metrics = json.loads((root / "assignment4/diffusion_metrics.json").read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert metrics["train_count"] == 50000 and metrics["epochs"] >= 1
    state = torch.get_rng_state().clone()
    model, schedule = dm.load_diffusion.__wrapped__()
    assert torch.equal(state, torch.get_rng_state())
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert schedule.timesteps == metrics["timesteps"]

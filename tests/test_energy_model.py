"""Energy sampling must optimize the input without changing learned parameters."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from app import energy_model


class QuadraticEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * x.square().flatten(1).sum(1)


def test_input_gradient_descent_preserves_weights_grads_flags_and_original():
    model = QuadraticEnergy()
    model.scale.grad = torch.tensor(7.0)
    initial = torch.full((2, 3, 4, 4), 0.8)
    with torch.no_grad():
        result = energy_model.langevin_sample(model, initial, steps=5, step_size=0.1,
                                              noise_std=0.0, gradient_clip=None)
    # For E=sum(x^2), x <- x - .1*2x = .8x on each step.
    assert torch.allclose(result, initial * 0.8**5)
    assert torch.equal(initial, torch.full_like(initial, 0.8))
    assert model.scale.item() == 1 and model.scale.grad.item() == 7
    assert model.scale.requires_grad and model.training
    assert not result.requires_grad and result.grad_fn is None


def test_sampling_randomness_is_private_and_reproducible():
    model = QuadraticEnergy().requires_grad_(False)
    initial = torch.zeros(2, 3, 4, 4)
    before = torch.get_rng_state().clone()
    def sample(seed):
        return energy_model.langevin_sample(model, initial, steps=4, step_size=0.1,
                                            generator=torch.Generator().manual_seed(seed))
    result = sample(42)
    assert torch.equal(result, sample(42))
    assert not torch.equal(result, sample(43))
    assert torch.equal(before, torch.get_rng_state())


def test_sampling_noise_and_clamping_are_bounded():
    model = QuadraticEnergy()
    output = energy_model.langevin_sample(model, torch.ones(1, 3, 2, 2), steps=4,
                                          noise_std=10, generator=torch.Generator().manual_seed(1))
    assert output.min() >= -1 and output.max() <= 1
    assert model.scale.grad is None


def test_model_architecture_and_weight_training_after_detached_samples():
    model = energy_model.EnergyModel()
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    assert model(x).shape == (2,)
    fake = energy_model.langevin_sample(model, x, steps=2, generator=torch.Generator().manual_seed(42))
    assert all(p.grad is None for p in model.parameters())
    loss = model(x).mean() - model(fake).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_fixed_neighbor_prior_has_expected_value_and_input_gradient():
    x = torch.zeros(1, 3, 3, 3)
    x[0, 0, 1, 1] = 1
    x.requires_grad_(True)
    energy = energy_model.SMOOTHNESS_WEIGHT * energy_model.spatial_quadratic(x)
    # Four adjacent edges of squared difference1. The center derivative is8.
    assert energy.item() == pytest.approx(4 * energy_model.SMOOTHNESS_WEIGHT)
    gradient, = torch.autograd.grad(energy.sum(), x)
    assert gradient[0, 0, 1, 1].item() == pytest.approx(8 * energy_model.SMOOTHNESS_WEIGHT)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": True}, {"step_size": -1},
                                   {"noise_std": float("nan")}, {"gradient_clip": 0}])
def test_invalid_sampling_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        energy_model.langevin_sample(QuadraticEnergy(), torch.zeros(1, 1, 1, 1), **kwargs)


def test_real_checkpoint_provenance_and_cold_loading():
    root = Path(__file__).resolve().parents[1]
    path = root / "models/cifar10_energy.pt"
    metrics = json.loads((root / "assignment4/energy_training_metrics.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert metrics["train_count"] == 50_000 and metrics["test_examples_used"] == 0
    before = torch.get_rng_state().clone()
    model = energy_model.load_energy_model.__wrapped__()
    assert torch.equal(before, torch.get_rng_state())
    assert not model.training
    assert all(not p.requires_grad and p.device.type == "cpu" for p in model.parameters())
    x = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(42)) * 2 - 1
    result = energy_model.langevin_sample(model, x, steps=2, generator=torch.Generator().manual_seed(43))
    assert result.shape == x.shape and torch.isfinite(result).all()
    assert all(p.grad is None for p in model.parameters())


def test_missing_checkpoint_does_not_fall_back_to_random_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ENERGY_MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(energy_model.EnergyUnavailableError):
        energy_model.load_energy_model.__wrapped__()


def test_finite_pixels_with_overflowed_energy_are_rejected(monkeypatch):
    class OverflowEnergy(nn.Module):
        def forward(self, x):
            # Differentiable zero gradient, but an invalid infinite energy.
            return x.flatten(1).sum(1) * 0 + float("inf")
    monkeypatch.setattr(energy_model, "load_energy_model", lambda: OverflowEnergy())
    with pytest.raises(energy_model.EnergyUnavailableError, match="non-finite output energies"):
        energy_model.sample_energy(count=1, seed=42, steps=1)

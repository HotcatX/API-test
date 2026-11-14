"""Reproduce and assert the numerical/autograd answers for Assignment 4.

Run from the repository root with:
    uv run python scripts/check_assignment4_theory.py

This script uses no model checkpoints and does not alter existing artifacts.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embedding(
    t: float, dimensions: int = 8, max_period: float = 10000.0
) -> torch.Tensor:
    """Interleave sin/cos pairs using omega[k] = max_period ** (-2*k/d)."""
    if dimensions < 2 or dimensions % 2:
        raise ValueError("dimensions must be a positive even integer of at least 2")
    k = torch.arange(dimensions // 2, dtype=torch.float64)
    angles = t * torch.exp(-math.log(max_period) * 2 * k / dimensions)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten()


def expected_backward_error(value: torch.Tensor) -> str:
    """Require the no-autograd-history error rather than hiding any failure."""
    try:
        value.backward()
    except RuntimeError as exc:
        message = str(exc)
        assert "does not require grad" in message, message
        return message
    raise AssertionError("backward() unexpectedly succeeded")


def run_checks(verbose: bool = True) -> dict:
    """Return observed results after checking every numeric/code question."""
    results = {}

    # Q2: eight dimensions, t=1, all angles in radians.
    embedding = sinusoidal_embedding(1.0)
    expected = torch.tensor(
        [
            math.sin(1.0), math.cos(1.0),
            math.sin(0.1), math.cos(0.1),
            math.sin(0.01), math.cos(0.01),
            math.sin(0.001), math.cos(0.001),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(embedding, expected, rtol=0, atol=1e-15)
    results["q2_embedding"] = [round(value, 9) for value in embedding.tolist()]

    # Q4: kernel 3, stride 2, padding 1 gives the assumed exact halving.
    spatial_sizes = [64]
    for _ in range(3):
        spatial_sizes.append((spatial_sizes[-1] + 2 * 1 - (3 - 1) - 1) // 2 + 1)
    assert spatial_sizes == [64, 32, 16, 8]
    results["q4_spatial_sizes"] = spatial_sizes

    # Q6: derivatives with respect to x.
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2 + 3 * x
    y.backward()
    torch.testing.assert_close(x.grad, torch.tensor([7.0]))
    results["q6_x_grad"] = x.grad.tolist()
    x_false = torch.tensor([2.0], requires_grad=False)
    y_false = x_false**2 + 3 * x_false
    results["q6_false_error"] = expected_backward_error(y_false)
    assert x_false.grad is None
    x_default = torch.tensor([2.0])
    assert x_default.requires_grad is False
    results["q6_default_requires_grad"] = x_default.requires_grad

    # Q7(a): constants w participate in the value, but their gradients are off.
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    y = w[0] * x**2 + w[1] * x
    y.backward()
    torch.testing.assert_close(x.grad, torch.tensor([7.0]))
    assert w.grad is None
    results["q7_original_w_grad"] = w.grad

    # Q7(b): build a new graph after enabling gradients on the weights.
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    y = w[0] * x**2 + w[1] * x
    y.backward()
    torch.testing.assert_close(x.grad, torch.tensor([7.0]))
    torch.testing.assert_close(w.grad, torch.tensor([4.0, 2.0]))
    results["q7_corrected_x_grad"] = x.grad.tolist()
    results["q7_corrected_w_grad"] = w.grad.tolist()

    # Q8: detach disconnects the graph; clone preserves the differentiable path.
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.detach()
    w = z * 2
    results["q8_detach_error"] = expected_backward_error(w)
    assert x.grad is None
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.clone()
    w = z * 2
    w.backward()
    torch.testing.assert_close(x.grad, torch.tensor([6.0]))
    results["q8_corrected_x_grad"] = x.grad.tolist()

    # Re-enabling gradients on detached z does not restore its old history.
    x = torch.tensor([1.0], requires_grad=True)
    z = (x * 3).detach().requires_grad_(True)
    (z * 2).backward()
    assert x.grad is None
    torch.testing.assert_close(z.grad, torch.tensor([2.0]))
    results["q8_detached_leaf_x_grad"] = x.grad
    results["q8_detached_leaf_z_grad"] = z.grad.tolist()

    # Q9: backward adds to the current gradient; resetting isolates the next one.
    x = torch.tensor([1.0], requires_grad=True)
    y1 = x * 2
    y1.backward()
    first = x.grad.clone()
    y2 = x * 3
    y2.backward()
    second = x.grad.clone()
    x.grad = None
    y3 = x * 3
    y3.backward()
    after_reset = x.grad.clone()
    torch.testing.assert_close(first, torch.tensor([2.0]))
    torch.testing.assert_close(second, torch.tensor([5.0]))
    torch.testing.assert_close(after_reset, torch.tensor([3.0]))
    results["q9_first_grad"] = first.tolist()
    results["q9_second_grad"] = second.tolist()
    results["q9_after_reset_grad"] = after_reset.tolist()

    if verbose:
        print(f"PyTorch: {torch.__version__}")
        for name, value in results.items():
            print(f"{name}: {value}")
        print("All Assignment 4 theory checks passed.")
    return results


if __name__ == "__main__":
    run_checks()

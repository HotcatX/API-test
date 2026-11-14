"""Real inference and input validation for the two Assignment 4 API routes."""

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

from app import diffusion_model, energy_model
from app.cifar_generator import rgb_grid_png, generate_cifar_images
from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("kind", ["diffusion", "energy"])
def test_trained_generator_contract_and_rng_isolation(kind):
    before = torch.get_rng_state().clone()
    response = client.get(f"/generate-{kind}?seed=42")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-seed"] == "42"
    assert response.headers["x-image-count"] == "1"
    assert response.headers["x-model"] == f"assignment4-cifar10-{kind}"
    with Image.open(BytesIO(response.content)) as im:
        assert im.format == "PNG" and im.mode == "RGB" and im.size == (32, 32)
        assert any(lo < hi for lo, hi in im.getextrema())
    assert torch.equal(before, torch.get_rng_state())
    assert client.get(f"/generate-{kind}?seed=42").content == response.content
    assert client.get(f"/generate-{kind}?seed=43").content != response.content
    if kind == "diffusion":
        model, _ = diffusion_model.load_diffusion()
    else:
        model = energy_model.load_energy_model()
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("kind", ["diffusion", "energy"])
def test_grid_inference_and_omitted_seed_replay(kind):
    response = client.post(f"/generate-{kind}", json={"count": 4})
    assert response.status_code == 200
    assert response.headers["x-image-count"] == "4"
    seed = int(response.headers["x-seed"])
    assert 0 <= seed <= 2**32 - 1
    with Image.open(BytesIO(response.content)) as im:
        assert im.size == (66, 66) and im.mode == "RGB"
    assert client.post(f"/generate-{kind}", json={"count": 4, "seed": seed}).content == response.content


@pytest.mark.parametrize("kind", ["diffusion", "energy"])
@pytest.mark.parametrize("payload", [
    {"count": 0}, {"count": 17}, {"count": 1.5}, {"count": True}, {"count": "4"},
    {"seed": -1}, {"seed": 2**32}, {"seed": False}, {"seed": 1.5}, {"class": "cat"},
])
def test_invalid_requests_are_rejected(kind, payload):
    assert client.post(f"/generate-{kind}", json=payload).status_code == 422


@pytest.mark.parametrize("kind", ["diffusion", "energy"])
@pytest.mark.parametrize("seed", ["-1", "4294967296", "abc", "1.5"])
def test_invalid_query_seed(kind, seed):
    assert client.get(f"/generate-{kind}?seed={seed}").status_code == 422


@pytest.mark.parametrize("count,size", [(1, (32, 32)), (3, (66, 66)), (16, (134, 134))])
def test_rgb_png_scaling_and_grid_padding(count, size):
    images = torch.zeros(count, 3, 32, 32)
    images[:, 0] = -1
    images[:, 2] = 1
    with Image.open(BytesIO(rgb_grid_png(images))) as im:
        assert im.mode == "RGB" and im.size == size
        assert im.getpixel((0, 0)) == (0, 128, 255)
        if count == 3:
            assert im.getpixel((34, 34)) == (0, 0, 0)


@pytest.mark.parametrize("kind,module,envname,loader", [
    ("diffusion", diffusion_model, "DIFFUSION_MODEL_PATH", "load_diffusion"),
    ("energy", energy_model, "ENERGY_MODEL_PATH", "load_energy_model"),
])
@pytest.mark.parametrize("problem", ["missing", "corrupt", "incompatible", "nonfinite"])
def test_bad_checkpoint_returns_503_without_fallback(monkeypatch, tmp_path, kind, module, envname, loader, problem):
    path = tmp_path / "bad.pt"
    if problem == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif problem == "incompatible":
        torch.save({"architecture_version": -1}, path)
    elif problem == "nonfinite":
        root = Path(__file__).resolve().parents[1]
        checkpoint = torch.load(root / f"models/cifar10_{kind}.pt", weights_only=True, map_location="cpu")
        first_weight = next(iter(checkpoint["model_state_dict"].values()))
        first_weight.flatten()[0] = float("nan")
        torch.save(checkpoint, path)
    monkeypatch.setenv(envname, str(path))
    cached_loader = getattr(module, loader)
    cached_loader.cache_clear()
    try:
        result = client.get(f"/generate-{kind}?seed=42")
        assert result.status_code == 503
        assert client.get("/").json() == {"Hello": "World"}
    finally:
        cached_loader.cache_clear()


def test_openapi_exposes_both_methods_and_png_contract():
    paths = client.get("/openapi.json").json()["paths"]
    for kind in ("diffusion", "energy"):
        for method in ("get", "post"):
            responses = paths[f"/generate-{kind}"][method]["responses"]
            assert {"200", "422", "503"} <= responses.keys()
            assert "image/png" in responses["200"]["content"]


def test_direct_function_rejects_unknown_generator():
    with pytest.raises(ValueError, match="Unknown"):
        generate_cifar_images("unknown")

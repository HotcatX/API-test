"""Verify the assignment architecture and trained GAN image-generation contract."""

from io import BytesIO
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch
from torch import nn

from app import digit_generator
from app.gan_model import Generator, Discriminator, LATENT_DIM
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_required_architecture_and_gradient_flow():
    generator, discriminator = Generator(), Discriminator()
    assert (generator.fc.in_features, generator.fc.out_features) == (100, 6272)
    assert [type(layer) for layer in generator.layers] == [
        nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU, nn.ConvTranspose2d, nn.Tanh,
    ]
    assert [type(layer) for layer in discriminator.layers] == [
        nn.Conv2d, nn.LeakyReLU, nn.Conv2d, nn.BatchNorm2d,
        nn.LeakyReLU, nn.Flatten, nn.Linear, nn.Sigmoid,
    ]
    convolutions = [generator.layers[0], generator.layers[3], discriminator.layers[0], discriminator.layers[2]]
    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (128, 64), (64, 1), (1, 64), (64, 128),
    ]
    for layer in convolutions:
        assert layer.kernel_size == (4, 4)
        assert layer.stride == (2, 2)
        assert layer.padding == (1, 1)
    assert discriminator.layers[1].negative_slope == discriminator.layers[4].negative_slope == 0.2
    assert sum(p.numel() for p in generator.parameters()) == 765_761
    assert sum(p.numel() for p in discriminator.parameters()) == 138_817
    images = generator(torch.randn(2, LATENT_DIM))
    probabilities = discriminator(images)
    assert images.shape == (2, 1, 28, 28)
    assert probabilities.shape == (2, 1)
    assert -1 <= images.min() <= images.max() <= 1
    assert 0 <= probabilities.min() <= probabilities.max() <= 1
    loss = nn.BCELoss()(probabilities, torch.ones_like(probabilities))
    loss.backward()
    assert torch.isfinite(loss)
    assert generator.fc.weight.grad.abs().sum() > 0
    assert discriminator.layers[0].weight.grad.abs().sum() > 0


def test_saved_checkpoint_provenance():
    path = ROOT / "models/mnist_gan.pt"
    metrics = json.loads((ROOT / "assignment3/training_metrics.json").read_text())
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert checkpoint["epochs"] == metrics["epochs"] >= 1
    assert checkpoint["latent_dim"] == 100
    assert checkpoint["architecture_version"] == 1
    assert metrics["train_count"] == 60_000
    assert not digit_generator.load_generator().training


def test_cold_model_loading_does_not_consume_global_randomness():
    before = torch.get_rng_state().clone()
    model = digit_generator.load_generator.__wrapped__()
    assert torch.equal(before, torch.get_rng_state())
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    assert not model.training


def test_single_png_is_actual_generator_output_and_deterministic():
    response = client.get("/generate-digit?seed=42")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-seed"] == "42"
    assert response.headers["x-image-count"] == "1"
    assert response.headers["x-model"] == "assignment3-mnist-gan"
    model = digit_generator.load_generator()
    with torch.inference_mode():
        noise = torch.randn(1, 100, generator=torch.Generator().manual_seed(42))
        expected = ((model(noise)[0, 0] + 1) * 127.5).round().to(torch.uint8)
    with Image.open(BytesIO(response.content)) as image:
        assert image.format == "PNG" and image.mode == "L" and image.size == (28, 28)
        assert image.tobytes() == expected.numpy().tobytes()
        assert image.getextrema()[0] < image.getextrema()[1]
    assert client.get("/generate-digit?seed=42").content == response.content
    assert client.get("/generate-digit?seed=43").content != response.content


@pytest.mark.parametrize("count,size", [(1, (28, 28)), (3, (58, 58)), (16, (118, 118)), (64, (238, 238))])
def test_grid_and_request_local_randomness(count, size):
    model = digit_generator.load_generator()
    state = {name: value.clone() for name, value in model.named_buffers()}
    rng_before = torch.get_rng_state().clone()
    response = client.post("/generate-digits", json={"count": count, "seed": 2026})
    assert response.status_code == 200
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert response.headers["x-image-count"] == str(count)
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "L" and image.size == size
    for name, value in model.named_buffers():
        assert torch.equal(state[name], value), "API inference must not update BatchNorm"


def test_omitted_seed_is_reported_and_can_be_replayed():
    result = client.get("/generate-digit")
    assert result.status_code == 200
    seed = int(result.headers["x-seed"])
    assert 0 <= seed <= digit_generator.MAX_SEED
    assert client.get(f"/generate-digit?seed={seed}").content == result.content


@pytest.mark.parametrize("payload", [
    {"count": 0}, {"count": 65}, {"count": 1.5}, {"count": True}, {"count": "4"},
    {"seed": -1}, {"seed": 2**32}, {"seed": False}, {"seed": 1.5}, {"digit": 7},
])
def test_invalid_grid_requests_are_422(payload):
    assert client.post("/generate-digits", json=payload).status_code == 422


@pytest.mark.parametrize("seed", ["-1", "4294967296", "abc", "1.5"])
def test_invalid_single_image_seed_is_422(seed):
    assert client.get(f"/generate-digit?seed={seed}").status_code == 422


@pytest.mark.parametrize("kind", ["missing", "corrupt", "incompatible", "nonfinite"])
def test_unavailable_checkpoint_is_503_without_random_fallback(monkeypatch, tmp_path, kind):
    path = tmp_path / "bad.pt"
    if kind == "corrupt":
        path.write_bytes(b"invalid checkpoint")
    elif kind == "incompatible":
        torch.save({"architecture_version": -1}, path)
    elif kind == "nonfinite":
        checkpoint = torch.load(ROOT / "models/mnist_gan.pt", weights_only=True, map_location="cpu")
        checkpoint["generator_state_dict"]["fc.weight"][0, 0] = float("nan")
        torch.save(checkpoint, path)
    monkeypatch.setenv("GAN_MODEL_PATH", str(path))
    monkeypatch.setattr(digit_generator, "load_generator", digit_generator.load_generator.__wrapped__)
    response = client.get("/generate-digit?seed=42")
    assert response.status_code == 503
    assert "trained MNIST GAN checkpoint" in response.json()["detail"]
    assert client.get("/").json() == {"Hello": "World"}


def test_openapi_documents_png_and_both_routes():
    paths = client.get("/openapi.json").json()["paths"]
    for route, method in [("/generate-digit", "get"), ("/generate-digits", "post")]:
        endpoint = paths[route][method]
        assert {"200", "422", "503"} <= endpoint["responses"].keys()
        assert "image/png" in endpoint["responses"]["200"]["content"]

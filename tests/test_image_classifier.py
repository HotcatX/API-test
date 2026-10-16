"""Check the specified architecture, real checkpoint, and image-upload contract."""

from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient

from app import image_classifier
from app.cnn_model import AssignmentCNN, CLASSES, preprocess
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def encoded_image(mode="RGB", fmt="PNG", size=(17, 23)):
    buffer = BytesIO()
    Image.new(mode, size, 80).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(data, filename="image.png", content_type="image/png"):
    return client.post("/classify", files={"file": (filename, data, content_type)})


def test_architecture_matches_every_required_layer_and_shape():
    model = AssignmentCNN()
    layers = [*model.features, *model.classifier]
    assert [type(layer) for layer in layers] == [
        nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU,
        nn.MaxPool2d, nn.Flatten, nn.Linear, nn.ReLU, nn.Linear,
    ]
    assert layers[0].kernel_size == layers[3].kernel_size == (3, 3)
    assert layers[0].stride == layers[3].stride == (1, 1)
    assert layers[0].padding == layers[3].padding == (1, 1)
    assert layers[2].kernel_size == layers[2].stride == 2
    assert layers[5].kernel_size == layers[5].stride == 2
    expected = [
        (2, 16, 64, 64), (2, 16, 64, 64), (2, 16, 32, 32),
        (2, 32, 32, 32), (2, 32, 32, 32), (2, 32, 16, 16),
        (2, 8192), (2, 100), (2, 100), (2, 10),
    ]
    x = torch.zeros(2, 3, 64, 64)
    with torch.inference_mode():
        for layer, shape in zip(layers, expected):
            x = layer(x)
            assert tuple(x.shape) == shape
    assert sum(p.numel() for p in model.parameters()) == 825_398


def test_model_supports_backpropagation_and_has_no_mode_dependent_layers():
    model = AssignmentCNN()
    inputs = torch.randn(2, 3, 64, 64)
    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 9]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.features[0].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert torch.equal(logits.detach(), model(inputs))


def test_preprocessing_matches_documented_resize_and_normalization():
    image = Image.fromarray(np.arange(19 * 27 * 3, dtype=np.uint8).reshape(19, 27, 3))
    resized = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR)).copy()
    expected = (torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
    assert torch.equal(preprocess(image), expected)


def test_trained_checkpoint_matches_recorded_metrics_and_api_predictions():
    metrics = json.loads((ROOT / "assignment2/training_metrics.json").read_text())
    path = ROOT / "models/cifar10_cnn.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert metrics["train_count"] == 45000
    assert metrics["validation_count"] == 5000
    assert metrics["test_count"] == 10000
    assert metrics["test_evaluations"] == 1
    assert 0 < metrics["test"]["accuracy"] <= 1
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    reference = AssignmentCNN().eval()
    reference.load_state_dict(checkpoint["model_state_dict"])
    data = (ROOT / "assignment2/sample_cat.png").read_bytes()
    with torch.inference_mode(), Image.open(BytesIO(data)) as image:
        expected = reference(preprocess(image).unsqueeze(0)).softmax(dim=1)[0]
    response = upload(data)
    assert response.status_code == 200
    result = response.json()
    assert result["class_id"] == expected.argmax().item()
    assert result["predicted_class"] == CLASSES[result["class_id"]]
    assert result["input_shape"] == [1, 3, 64, 64]
    assert result["model"] == "assignment2-cifar10-cnn"
    assert list(result["probabilities"]) == list(CLASSES)
    assert list(result["probabilities"].values()) == pytest.approx(expected.tolist())
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))
    assert upload(data).json() == result


@pytest.mark.parametrize("mode,fmt", [("RGB", "JPEG"), ("RGBA", "PNG"), ("L", "PNG"), ("RGB", "WEBP")])
def test_supported_formats_and_color_modes(mode, fmt):
    # A misleading MIME type must not override valid decoded image content.
    response = upload(encoded_image(mode, fmt), "upload.bin", "application/octet-stream")
    assert response.status_code == 200
    assert response.json()["input_shape"] == [1, 3, 64, 64]


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_rejects_empty_or_invalid_image(data):
    assert upload(data).status_code == 422


def test_requires_file_and_rejects_unsupported_format():
    assert client.post("/classify").status_code == 422
    assert upload(encoded_image(fmt="GIF")).status_code == 415


def test_rejects_oversized_file_and_decoded_image(monkeypatch):
    assert upload(b"x" * (image_classifier.MAX_UPLOAD_BYTES + 1)).status_code == 413
    monkeypatch.setattr(image_classifier, "MAX_IMAGE_PIXELS", 16)
    assert upload(encoded_image(size=(5, 5))).status_code == 413


@pytest.mark.parametrize("kind", ["missing", "corrupt", "wrong_classes"])
def test_missing_or_corrupt_checkpoint_is_503_without_random_fallback(monkeypatch, tmp_path, kind):
    path = tmp_path / "checkpoint.pt"
    if kind == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif kind == "wrong_classes":
        torch.save({"classes": ["wrong"]}, path)
    monkeypatch.setenv("CNN_MODEL_PATH", str(path))
    monkeypatch.setattr(image_classifier, "load_classifier", image_classifier.load_classifier.__wrapped__)
    response = upload(encoded_image())
    assert response.status_code == 503
    assert "trained CIFAR-10 checkpoint" in response.json()["detail"]
    assert client.get("/").status_code == 200


def test_openapi_exposes_multipart_upload_and_error_responses():
    endpoint = client.get("/openapi.json").json()["paths"]["/classify"]["post"]
    assert "multipart/form-data" in endpoint["requestBody"]["content"]
    assert {"200", "413", "415", "422", "503"} <= endpoint["responses"].keys()

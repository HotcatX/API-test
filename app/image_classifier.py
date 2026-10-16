"""Load the trained Assignment 2 CNN and classify uploaded images on CPU."""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from app.cnn_model import AssignmentCNN, CLASSES, PREPROCESSING, preprocess

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MODEL_NAME = "assignment2-cifar10-cnn"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "cifar10_cnn.pt"


class ClassifierUnavailableError(RuntimeError):
    """The bundled trained checkpoint is missing or incompatible."""


class InvalidImageError(ValueError):
    """The uploaded file cannot be decoded as an image."""


class ImageTooLargeError(ValueError):
    """The upload exceeds its byte or decoded pixel limit."""


class UnsupportedImageError(ValueError):
    """The file uses an unsupported image format."""


@dataclass(frozen=True)
class LoadedClassifier:
    model: AssignmentCNN
    checkpoint: Path


@lru_cache(maxsize=1)
def load_classifier() -> LoadedClassifier:
    """Reuse one trained CPU model; requests never download or train anything."""
    path = Path(os.environ.get("CNN_MODEL_PATH", str(DEFAULT_CHECKPOINT)))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Unexpected class order")
        if checkpoint["preprocessing"] != PREPROCESSING:
            raise ValueError("The checkpoint preprocessing does not match the API")
        model = AssignmentCNN()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("Non-finite trained parameters")
        model.eval()
    except Exception as exc:
        # A missing or bad model must never silently become random predictions.
        raise ClassifierUnavailableError(
            "The trained CIFAR-10 checkpoint is unavailable or incompatible. "
            "Restore models/cifar10_cnn.pt from the repository or run "
            "'uv run python scripts/train_cifar10.py'."
        ) from exc
    return LoadedClassifier(model=model, checkpoint=path)


def decode_image(data: bytes) -> Image.Image:
    """Validate the actual image content instead of trusting its filename."""
    if not data:
        raise InvalidImageError("Upload a non-empty PNG, JPEG, or WebP image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageTooLargeError("The image file must be at most 5 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageTooLargeError("The decoded image must have at most 16 million pixels.")
                if image.format not in SUPPORTED_FORMATS:
                    raise UnsupportedImageError("Supported image formats are PNG, JPEG, and WebP.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("The decoded image is too large.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The upload is not a valid PNG, JPEG, or WebP image.") from exc


def classify_image(data: bytes) -> dict:
    """Return the predicted CIFAR-10 class and all ten softmax scores."""
    image = decode_image(data)
    loaded = load_classifier()
    batch = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        scores = loaded.model(batch).softmax(dim=1)[0]
    if scores.shape != (10,) or not torch.isfinite(scores).all().item():
        raise ClassifierUnavailableError("The classifier produced invalid predictions.")
    class_id = int(scores.argmax().item())
    return {
        "predicted_class": CLASSES[class_id],
        "class_id": class_id,
        "confidence": float(scores[class_id].item()),
        "probabilities": dict(zip(CLASSES, scores.tolist())),
        "model": MODEL_NAME,
        "input_shape": list(batch.shape),
    }

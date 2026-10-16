"""The exact CNN required by Assignment 2 and its shared image preprocessing."""

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
IMAGE_SIZE = 64
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)
PREPROCESSING = {
    "color_mode": "RGB",
    "image_size": [IMAGE_SIZE, IMAGE_SIZE],
    "resize_interpolation": "bilinear",
    "tensor_range_before_normalization": [0.0, 1.0],
    "normalization_mean": list(NORMALIZATION_MEAN),
    "normalization_std": list(NORMALIZATION_STD),
}
ARCHITECTURE = "AssignmentCNN: Conv3-16-ReLU-Pool2-Conv16-32-ReLU-Pool2-Flatten8192-FC100-ReLU-FC10"


class AssignmentCNN(nn.Module):
    """Accept N x 3 x 64 x 64 tensors and return N x 10 unnormalized logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    """Apply the same deterministic preprocessing used for validation and testing."""
    return _preprocess(image.convert("RGB"))


def training_transform() -> transforms.Compose:
    """Training-only spatial augmentation followed by the shared normalization."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(IMAGE_SIZE, padding=8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])

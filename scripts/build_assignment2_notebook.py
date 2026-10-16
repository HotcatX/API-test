"""Build the editable Assignment 2 notebook from source and recorded results.

Run after training has produced assignment2/training_metrics.json. This script
does not train a model, execute notebook cells, or download a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def clean(text: str) -> str:
    return textwrap.dedent(text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Assignment2_code.ipynb")
    args = parser.parse_args()

    # A completed training record is required; never fill missing results with
    # hypothetical values or silently substitute another checkpoint.
    metrics_path = ROOT / "assignment2/training_metrics.json"
    history_path = ROOT / "assignment2/training_history.json"
    sample_path = ROOT / "assignment2/sample_metadata.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert metrics["parameter_count"] == 825398
    assert metrics["test"]["count"] == metrics["test_count"] == 10000
    assert metrics["test_evaluations"] == 1
    assert len(history) == metrics["epochs"]
    assert sample["index"] == 0 and sample["true_class"] == "cat"
    assert (ROOT / metrics["checkpoint"]).is_file()
    assert (ROOT / sample["image"]).is_file()

    cells = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(clean(text)))

    def code(text: str) -> None:
        cells.append(nbformat.v4.new_code_cell(clean(text)))

    md("""
    # Assignment 2: Convolutional Neural Networks and Image Classification

    Name: He Xiao\x20\x20
    UNI: hx2425\x20\x20
    GitHub repository: [HotcatX/API-test](https://github.com/HotcatX/API-test)

    This submission implements the required PyTorch CNN, trains it on CIFAR-10,
    adds an image classification endpoint to the existing FastAPI application,
    and answers the six CNN arithmetic questions.

    The notebook uses the saved model and recorded training results. Running its
    cells performs small verification and inference checks; it does not retrain
    the model or download the full dataset.
    """)
    md(r"""
    ## Part I. Practice: CNN Architecture

    The required input is a 64 × 64 RGB image. PyTorch represents a batch as
    `(N, 3, 64, 64)`, where `N` is the batch size. The final layer produces ten
    logits, one for each CIFAR-10 class. During training, these logits are passed
    directly to `CrossEntropyLoss`. Softmax is applied only when reporting
    prediction scores.

    Run the notebook with the project's `.venv` Python kernel. From the repository
    root, the environment can be prepared with:

    ```bash
    uv sync --locked --group dev --group notebook
    uv run --group notebook python -m ipykernel install --sys-prefix \
      --name api-test --display-name "API-test (.venv)"
    ```
    """)
    code("""
    from pathlib import Path
    import hashlib
    import inspect
    import json
    import math
    import os
    import sys

    import torch
    from torch import nn
    from IPython.display import Markdown, display
    from PIL import Image

    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next(path for path in candidates
                if (path / "app/cnn_model.py").is_file())
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from app.cnn_model import (
        ARCHITECTURE, CLASSES, PREPROCESSING, AssignmentCNN, preprocess,
    )

    torch.set_num_threads(min(2, torch.get_num_threads()))
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Using project .venv:", Path(sys.prefix).resolve() == (ROOT / ".venv").resolve())
    print("Classes:", ", ".join(CLASSES))
    """)
    md("""
    ### Model implementation

    The following cell displays the actual class imported by the training script
    and API. This keeps the submitted implementation and deployed model consistent.
    """)
    code("""
    print(inspect.getsource(AssignmentCNN))
    """)
    md(r"""
    ### Output shapes and parameters

    Both convolutions use kernel size 3, stride 1, and padding 1, preserving
    height and width. Each 2 × 2 max-pooling layer halves them. After two pooling
    layers, the feature tensor is $(N,32,16,16)$, so the flattened size is

    $$32\times16\times16=8192.$$

    | Layer | Output shape | Trainable parameters |
    | --- | --- | ---: |
    | Input | $(N,3,64,64)$ | 0 |
    | Conv2d, 3 → 16, 3 × 3 | $(N,16,64,64)$ | 448 |
    | ReLU | $(N,16,64,64)$ | 0 |
    | MaxPool2d, 2 × 2 | $(N,16,32,32)$ | 0 |
    | Conv2d, 16 → 32, 3 × 3 | $(N,32,32,32)$ | 4,640 |
    | ReLU | $(N,32,32,32)$ | 0 |
    | MaxPool2d, 2 × 2 | $(N,32,16,16)$ | 0 |
    | Flatten | $(N,8192)$ | 0 |
    | Linear, 8192 → 100 | $(N,100)$ | 819,300 |
    | ReLU | $(N,100)$ | 0 |
    | Linear, 100 → 10 | $(N,10)$ | 1,010 |
    | Total | $(N,10)$ | 825,398 |

    Including one bias per output channel or unit,

    $$
    \begin{aligned}
    P_{\mathrm{conv1}} &=16(3\times3\times3+1)=448,\\
    P_{\mathrm{conv2}} &=32(16\times3\times3+1)=4{,}640,\\
    P_{\mathrm{fc1}} &=8192\times100+100=819{,}300,\\
    P_{\mathrm{fc2}} &=100\times10+10=1{,}010,\\
    P_{\mathrm{total}} &=448+4{,}640+819{,}300+1{,}010\\
    &=825{,}398.
    \end{aligned}
    $$
    """)
    code("""
    # This untrained instance is used only to verify the architecture.
    model = AssignmentCNN()
    traces = []

    def record_shape(name):
        def hook(module, inputs, output):
            traces.append((name, type(module).__name__, tuple(output.shape)))
        return hook

    handles = [
        module.register_forward_hook(record_shape(name))
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 64, 64))
    for handle in handles:
        handle.remove()

    expected = [
        (1, 16, 64, 64), (1, 16, 64, 64), (1, 16, 32, 32),
        (1, 32, 32, 32), (1, 32, 32, 32), (1, 32, 16, 16),
        (1, 8192), (1, 100), (1, 100), (1, 10),
    ]
    assert [shape for _, _, shape in traces] == expected
    for name, kind, shape in traces:
        print(f"{name:15} {kind:12} {shape}")

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameter_count == 825_398
    print(f"Trainable parameters: {parameter_count:,}")
    """)
    md("""
    ## Part II. Practice: Training and API Deployment

    ### Data and preprocessing

    CIFAR-10 contains 60,000 RGB images of size 32 × 32 in ten classes: 50,000
    official training images and 10,000 official test images. The official
    training set is split into 45,000 training and 5,000 validation examples.
    The checkpoint with the highest validation accuracy is selected, then
    evaluated once on the held-out test set.

    All images are converted to RGB and resized to 64 × 64 using bilinear
    interpolation to match the required architecture. Pixel values are scaled
    to [0, 1] and normalized with channel-wise mean and standard deviation of
    0.5, giving values in approximately [-1, 1]. Training additionally uses
    random cropping and horizontal flipping. Validation, testing, and API
    inference use the same deterministic preprocessing function.

    [CIFAR-10 dataset source](https://www.cs.toronto.edu/~kriz/cifar.html)
    """)
    code("""
    metrics = json.loads((ROOT / "assignment2/training_metrics.json").read_text())
    history = json.loads((ROOT / "assignment2/training_history.json").read_text())
    sample_metadata = json.loads((ROOT / "assignment2/sample_metadata.json").read_text())

    assert metrics["classes"] == list(CLASSES)
    assert metrics["architecture"] == ARCHITECTURE
    assert metrics["preprocessing"] == PREPROCESSING
    assert metrics["parameter_count"] == parameter_count
    assert len(history) == metrics["epochs"]
    assert metrics["train_count"] + metrics["validation_count"] == 50_000
    assert metrics["test"]["count"] == metrics["test_count"] == 10_000
    assert metrics["test_evaluations"] == 1

    config_keys = [
        "dataset", "train_count", "validation_count", "test_count", "seed",
        "epochs", "batch_size", "optimizer", "initial_learning_rate",
        "weight_decay", "scheduler", "minimum_learning_rate", "loss",
        "device", "pytorch_version", "training_augmentation",
    ]
    table = ["| Setting | Recorded value |", "| --- | --- |"]
    table.extend(f"| {key} | {metrics[key]} |" for key in config_keys)
    display(Markdown("\\n".join(table)))
    """)
    md("""
    ### Recorded training results

    The results below are read from the training artifacts produced by
    `scripts/train_cifar10.py`. Training accuracy uses augmented inputs, while
    validation accuracy uses deterministic preprocessing. Seeds are fixed, but
    hardware, backend, and library differences can still affect floating-point
    results. The test set is not used to select the epoch.
    """)
    code("""
    table = [
        "| Epoch | Learning rate | Train loss | Train accuracy | Val. loss | Val. accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in history:
        table.append(
            f"| {record['epoch']} | {record['learning_rate']:.6f} | "
            f"{record['train_loss']:.4f} | {record['train_accuracy']:.2%} | "
            f"{record['validation_loss']:.4f} | {record['validation_accuracy']:.2%} |"
        )
    display(Markdown("\\n".join(table)))

    selected = next(record for record in history
                    if record["epoch"] == metrics["best_epoch"])
    assert math.isclose(selected["validation_accuracy"],
                        metrics["best_validation_accuracy"], abs_tol=1e-12)
    assert math.isclose(metrics["best_validation_accuracy"],
                        max(row["validation_accuracy"] for row in history), abs_tol=1e-12)
    print("Selected checkpoint epoch:", metrics["best_epoch"])
    print(f"Best validation accuracy: {metrics['best_validation_accuracy']:.2%}")
    print(f"Held-out test accuracy: {metrics['test']['accuracy']:.2%}")
    print(f"Held-out test loss: {metrics['test']['loss']:.4f}")
    print(f"Recorded elapsed time: {metrics['elapsed_seconds'] / 60:.2f} minutes")
    """)
    code("""
    test = metrics["test"]
    confusion = test["confusion_matrix_rows_true_columns_predicted"]
    correct = sum(confusion[i][i] for i in range(len(CLASSES)))
    count = sum(sum(row) for row in confusion)
    assert count == 10_000
    assert math.isclose(correct / count, test["accuracy"], abs_tol=1e-12)

    table = ["| Class | Test accuracy |", "| --- | ---: |"]
    for name in CLASSES:
        table.append(f"| {name} | {test['per_class_accuracy'][name]:.2%} |")
    display(Markdown("\\n".join(table)))
    print(f"Correct test predictions: {correct:,} / {count:,}")
    """)
    md("""
    ### Loading the saved checkpoint

    The repository includes the trained checkpoint at `models/cifar10_cnn.pt`.
    Its state dictionary is loaded on the CPU for portable inference. The
    following cell verifies its SHA-256 hash against the training record and
    checks that its classes and preprocessing match the implementation.
    """)
    code("""
    checkpoint_path = ROOT / metrics["checkpoint"]
    actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert actual_sha256 == metrics["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["classes"] == list(CLASSES)
    assert checkpoint["preprocessing"] == PREPROCESSING
    assert checkpoint["architecture"] == ARCHITECTURE
    assert checkpoint["epoch"] == metrics["best_epoch"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print("Loaded:", metrics["checkpoint"])
    print("SHA-256:", actual_sha256)
    print("Selected epoch:", checkpoint["epoch"])
    """)
    md("""
    ### Example image and direct inference

    The example is the original image at index 0 of the official CIFAR-10 test
    split, whose label is `cat`. The index was fixed independently of the model's
    prediction. The single example illustrates the interface; aggregate test
    accuracy is reported above.
    """)
    code("""
    sample_path = ROOT / sample_metadata["image"]
    assert sample_metadata["split"] == "test"
    assert sample_metadata["index"] == 0
    assert sample_metadata["true_class"] == "cat"
    with Image.open(sample_path) as image:
        example = image.convert("RGB")
    print("Original image dimensions:", example.size)
    display(example.resize((128, 128), Image.Resampling.NEAREST))
    image_tensor = preprocess(example).unsqueeze(0)
    assert tuple(image_tensor.shape) == (1, 3, 64, 64)
    with torch.inference_mode():
        direct_scores = model(image_tensor).softmax(dim=1)[0]
    direct_id = int(direct_scores.argmax())
    print("True class:", sample_metadata["true_class"])
    print("Predicted class:", CLASSES[direct_id])
    print(f"Softmax score: {float(direct_scores[direct_id]):.4f}")
    """)
    md("""
    ### FastAPI interface

    `POST /classify` accepts an image as a multipart upload with the field name
    `file`. It returns `predicted_class`, `class_id`, `confidence`, probabilities
    for all ten classes, the model name, and the preprocessed tensor shape.
    `confidence` is the largest softmax score; it is not a guarantee of correctness
    or a calibrated probability.

    The existing `/`, `/generate`, and `/embedding` routes remain available.
    Interactive endpoint documentation is at `http://localhost:8000/docs`.

    Supported uploads are PNG, JPEG, and WEBP. Invalid images return an error
    instead of a prediction. The endpoint uses the saved trained weights and
    the same preprocessing as validation and testing.
    """)
    code("""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*"
        )
        from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with sample_path.open("rb") as image_file:
            response = client.post(
                "/classify",
                files={"file": (sample_path.name, image_file, "image/png")},
            )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predicted_class"] == CLASSES[direct_id]
    assert result["class_id"] == direct_id
    assert result["input_shape"] == [1, 3, 64, 64]
    assert set(result["probabilities"]) == set(CLASSES)
    assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-6)
    assert math.isclose(result["confidence"],
                        float(direct_scores[direct_id]), abs_tol=1e-6)
    print("HTTP status:", response.status_code)
    print(json.dumps(result, indent=2))
    """)
    md("""
    ### Running and reproducing the project

    Run the API locally from the repository root using the project environment:

    ```bash
    uv sync --locked --group dev --group notebook
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

    In another terminal, upload the included image:

    ```bash
    curl -X POST http://localhost:8000/classify \\
      -F "file=@assignment2/sample_cat.png"
    ```

    Alternatively, build and start the Docker deployment:

    ```bash
    docker build -t api-test:assignment2 .
    docker run --rm -p 8000:80 api-test:assignment2
    ```

    The same curl request works against the container. The trained checkpoint
    is included in the image, so serving predictions does not require training
    or a CIFAR-10 download.

    Training is an explicit, separate operation. The command below recreates
    the recorded configuration and downloads CIFAR-10 into `data/` when needed:
    """)
    train_command = (
        "uv run python scripts/train_cifar10.py "
        f"--epochs {metrics['epochs']} --batch-size {metrics['batch_size']} "
        f"--learning-rate {metrics['initial_learning_rate']} --seed {metrics['seed']}"
    )
    md(f"""
    ```bash
    {train_command}
    ```

    This overwrites the saved checkpoint and result artifacts with the new run.
    To run the automated checks without retraining:

    ```bash
    uv run --group dev pytest -q
    ```
    """)

    theory = (ROOT / "assignment2/theory_solutions.md").read_text(encoding="utf-8")
    # Keep each question as a separate editable Markdown cell.
    for section in re.split(r"(?=^### Question [1-6]$)", theory, flags=re.MULTILINE):
        if section.strip():
            md(section)
    md("""
    ### Numerical checks for Questions 1-5

    These calculations independently verify the spatial-size arithmetic.
    """)
    code("""
    def spatial_size(n, kernel, stride=1, padding=0):
        return (n + 2 * padding - kernel) // stride + 1

    q1 = spatial_size(32, 5)
    q2 = spatial_size(32, 5, padding=2)
    q3 = spatial_size(64, 3, stride=2)
    q4 = spatial_size(16, 2, stride=2)
    q5_first = spatial_size(128, 3, padding=1)
    q5_second = spatial_size(q5_first, 3, padding=1)
    assert (q1, q2, q3, q4, q5_first, q5_second) == (28, 32, 31, 8, 128, 128)
    print("Q1:", (q1, q1, 8))
    print("Q2:", (q2, q2, 8))
    print("Q3 spatial size:", (q3, q3))
    print("Q4 spatial size:", (q4, q4))
    print("Q5 spatial size:", (q5_second, q5_second))
    """)
    md("""
    ### Training-mode check for Question 6

    This check uses the current network to verify that switching between
    training and evaluation modes does not change its forward output. It also
    checks separately that evaluation mode leaves automatic differentiation
    enabled when gradient tracking is otherwise active.
    """)
    code("""
    assert AssignmentCNN().training is True
    model.train()
    with torch.inference_mode():
        training_output = model(image_tensor)
    model.eval()
    with torch.inference_mode():
        evaluation_output = model(image_tensor)
    assert torch.equal(training_output, evaluation_output)
    print("Same forward output in both modes:",
          torch.equal(training_output, evaluation_output))

    with torch.enable_grad():
        evaluation_with_grad = model(image_tensor)
    assert evaluation_with_grad.requires_grad
    print("Evaluation mode still allows gradients:",
          evaluation_with_grad.requires_grad)
    with torch.no_grad():
        without_grad = model(image_tensor)
    assert not without_grad.requires_grad
    print("no_grad disables gradient tracking:", not without_grad.requires_grad)
    """)
    md("""
    ## References

    - [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
      convolution arguments and output-shape formula.
    - [PyTorch MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html):
      max-pooling behavior and output dimensions.
    - [PyTorch modules](https://docs.pytorch.org/docs/stable/notes/modules.html):
      module construction and the default training mode.
    - [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html):
      `train()` and `eval()`.
    - [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html):
      the distinction between evaluation mode and gradient tracking.
    - [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html):
      classification loss applied to unnormalized logits.
    - [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html):
      image dimensions, classes, and official train/test split.
    - Alex Krizhevsky (2009), [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
    """)

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "API-test (.venv)",
                "language": "python",
                "name": "api-test",
            },
            "language_info": {"name": "python", "file_extension": ".py"},
        },
    )
    nbformat.validate(notebook)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print(f"Wrote {output_path} ({len(cells)} cells; not executed)")


if __name__ == "__main__":
    main()

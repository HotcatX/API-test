"""Train AssignmentCNN on CIFAR-10 and save a portable inference checkpoint.

Run from the repository root: uv run python scripts/train_cifar10.py
The official training set is split into 45,000 training and 5,000 validation
examples. The test set is evaluated once, after selecting the best validation
checkpoint. Data are downloaded to data/, which is excluded from Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cnn_model import (
    ARCHITECTURE, CLASSES, PREPROCESSING, AssignmentCNN,
    preprocess, training_transform,
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model, loader, criterion, device, with_confusion=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    first_prediction = None
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            count += targets.size(0)
            if with_confusion:
                if first_prediction is None:
                    probabilities = logits[0].softmax(dim=0).cpu().tolist()
                    first_prediction = {
                        "predicted_class": CLASSES[predictions[0].item()],
                        "predicted_class_index": predictions[0].item(),
                        "confidence": max(probabilities),
                        "probabilities": dict(zip(CLASSES, probabilities)),
                    }
                indices = targets.cpu() * 10 + predictions.cpu()
                confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
    result = {"loss": total_loss / count, "accuracy": correct / count, "count": count}
    if with_confusion:
        result["confusion_matrix_rows_true_columns_predicted"] = confusion.tolist()
        result["per_class_accuracy"] = {
            name: confusion[i, i].item() / confusion[i].sum().item()
            for i, name in enumerate(CLASSES)
        }
        result["first_image_prediction"] = first_prediction
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers must be nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Stable CPU runs; MPS results can still differ slightly across hardware/PyTorch.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = PROJECT_ROOT / "data"
    checkpoint_path = PROJECT_ROOT / "models" / "cifar10_cnn.pt"
    output_dir = PROJECT_ROOT / "assignment2"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Device: {device}; seed={args.seed}; epochs={args.epochs}; batch_size={args.batch_size}", flush=True)
    print("Loading official CIFAR-10 training data (download is cached in data/).", flush=True)
    training_data = CIFAR10(root=str(data_dir), train=True, download=True, transform=training_transform())
    validation_data = CIFAR10(root=str(data_dir), train=True, download=False, transform=preprocess)
    split_generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(training_data), generator=split_generator).tolist()
    training_indices, validation_indices = indices[:45000], indices[45000:]
    split_digest = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          persistent_workers=args.workers > 0,
                          worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(training_data, training_indices), shuffle=True,
                              generator=loader_generator, **loader_options)
    val_loader = DataLoader(Subset(validation_data, validation_indices), shuffle=False, **loader_options)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    history = []
    best_accuracy = -1.0
    training_start = time.perf_counter()
    print("Data split: 45,000 training / 5,000 validation. Test data are held out.", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for batch_number, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.size(0)
            if batch_number % 100 == 0:
                elapsed = time.perf_counter() - epoch_start
                print(f"Epoch {epoch:02d}/{args.epochs}: batch {batch_number}/{len(train_loader)}, "
                      f"loss={total_loss/count:.4f}, accuracy={correct/count:.2%}, "
                      f"elapsed={elapsed:.1f}s", flush=True)
        validation = evaluate(model, val_loader, criterion, device)
        duration = time.perf_counter() - epoch_start
        record = {"epoch": epoch, "learning_rate": lr,
                  "train_loss": total_loss / count, "train_accuracy": correct / count,
                  "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                  "duration_seconds": duration}
        history.append(record)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            checkpoint = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "classes": list(CLASSES), "preprocessing": PREPROCESSING,
                "architecture": ARCHITECTURE, "epoch": epoch,
                "train_count": 45000, "validation_count": 5000, "seed": args.seed,
                "split_sha256": split_digest, "validation_accuracy": best_accuracy,
                "pytorch_version": str(torch.__version__), "trained_at_utc": started_at,
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
        write_json(output_dir / "training_history.json", history)
        scheduler.step()
        mean_duration = (time.perf_counter() - training_start) / epoch
        remaining = mean_duration * (args.epochs - epoch)
        print(f"Epoch {epoch:02d}/{args.epochs} complete: train_loss={record['train_loss']:.4f}, "
              f"train_acc={record['train_accuracy']:.2%}, val_loss={validation['loss']:.4f}, "
              f"val_acc={validation['accuracy']:.2%}, best_val={best_accuracy:.2%}, "
              f"duration={duration:.1f}s, remaining~{remaining/60:.1f}min", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Evaluating best validation checkpoint (epoch {checkpoint['epoch']}) once on the 10,000 test images.", flush=True)
    test_data = CIFAR10(root=str(data_dir), train=False, download=False, transform=preprocess)
    # The demonstration image is fixed in advance, never chosen for a correct result.
    from PIL import Image
    Image.fromarray(test_data.data[0]).save(output_dir / "sample_cat.png")
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    test = evaluate(model, test_loader, criterion, device, with_confusion=True)
    sample = {
        "image": "assignment2/sample_cat.png", "dataset": "CIFAR-10",
        "split": "test", "index": 0, "true_class_index": int(test_data.targets[0]),
        "true_class": CLASSES[test_data.targets[0]],
        "selection": "Fixed test index 0, independent of model prediction",
        **test.pop("first_image_prediction"),
    }
    write_json(output_dir / "sample_metadata.json", sample)
    metrics = {
        "dataset": "CIFAR-10", "dataset_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "classes": list(CLASSES), "architecture": ARCHITECTURE,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "preprocessing": PREPROCESSING,
        "training_augmentation": "Resize 64x64 bilinear; random crop 64x64 with reflect padding 8; random horizontal flip",
        "train_count": 45000, "validation_count": 5000, "test_count": 10000,
        "split": "torch.randperm(50000, generator=torch.Generator().manual_seed(seed)); first 45000 train, last 5000 validation",
        "split_sha256": split_digest, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "optimizer": "Adam", "initial_learning_rate": args.learning_rate,
        "weight_decay": 0.0001, "scheduler": "CosineAnnealingLR", "minimum_learning_rate": 0.00001,
        "loss": "CrossEntropyLoss on logits", "device": str(device),
        "pytorch_version": str(torch.__version__),
        "determinism_note": "Seeds are fixed. Hardware/backend/library differences may change floating-point results, especially on MPS.",
        "best_epoch": checkpoint["epoch"], "best_validation_accuracy": best_accuracy,
        "checkpoint_selection": "Highest validation accuracy; test set not used for selection",
        "test_evaluations": 1, "test": test,
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - training_start,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    write_json(output_dir / "training_metrics.json", metrics)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "best_validation_accuracy": best_accuracy,
                      "test_accuracy": test["accuracy"], "test_loss": test["loss"],
                      "checkpoint": str(checkpoint_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

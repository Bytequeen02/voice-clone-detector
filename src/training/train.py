"""
Training script for the CNN baseline.

Run:
    python -m src.training.train --config configs/config.yaml

Handles:
  - class imbalance (via weighted loss, computed from the actual manifest)
  - early stopping on validation loss
  - checkpointing the best model
  - saving a metrics log so Azad can use it for documentation/testing
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.training.dataset import VoiceCloneDataset, LABEL_TO_IDX
from src.models.cnn_baseline import CNNBaseline


def get_device(cfg):
    pref = cfg["training"].get("device", "auto")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(manifest_path):
    with open(manifest_path) as f:
        items = json.load(f)
    counts = Counter(item["label"] for item in items)
    total = sum(counts.values())
    weights = torch.zeros(len(LABEL_TO_IDX))
    for label, idx in LABEL_TO_IDX.items():
        c = counts.get(label, 1)
        weights[idx] = total / (len(LABEL_TO_IDX) * c)
    return weights


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main(cfg):
    device = get_device(cfg)
    print(f"Using device: {device}")

    splits_dir = Path(cfg["data"]["splits_dir"])
    train_manifest = splits_dir / "train.json"
    val_manifest = splits_dir / "val.json"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            "Manifests not found. Run: python -m src.preprocessing.prepare_dataset --config configs/config.yaml "
            "after placing audio under data/raw/real and data/raw/cloned."
        )

    train_ds = VoiceCloneDataset(str(train_manifest), cfg)
    val_ds = VoiceCloneDataset(str(val_manifest), cfg)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Train or validation set is empty. Add more audio data before training."
        )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNNBaseline(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    class_weights = compute_class_weights(train_manifest).to(device)
    print(f"Class weights (real, cloned): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - start

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_dir / "best_model.pt")
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_loss={best_val_loss:.4f}. "
          f"Checkpoint: {ckpt_dir / 'best_model.pt'}")
    print("NOTE: Do not report accuracy numbers anywhere (README, PPT, demo) "
          "until you have actually run this and looked at the real numbers here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)

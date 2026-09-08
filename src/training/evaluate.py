"""
Evaluation script. Run AFTER training:

    python -m src.training.evaluate --config configs/config.yaml

Reports on the held-out TEST split only (never look at test results
until you are done tuning - otherwise you are leaking test info back
into your decisions).

Outputs:
  - accuracy, precision, recall, F1, confusion matrix
  - a basic calibration check (are predicted probabilities trustworthy?)

All numbers here are computed live from your actual trained model -
nothing is hardcoded. Do not copy numbers from this file's comments
into your report; only use what you see printed when you run it.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

from src.utils.config_loader import load_config
from src.training.dataset import VoiceCloneDataset, IDX_TO_LABEL
from src.models.cnn_baseline import CNNBaseline


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = CNNBaseline(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train the model first.")

    model, saved_cfg = load_model(ckpt_path, device)

    test_manifest = Path(cfg["data"]["splits_dir"]) / "test.json"
    test_ds = VoiceCloneDataset(str(test_manifest), saved_cfg)
    if len(test_ds) == 0:
        raise RuntimeError("Test set is empty - cannot evaluate.")
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # P(cloned)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== Test Set Results (real numbers, computed just now) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 score : {f1:.4f}")
    print("Confusion matrix (rows=true, cols=predicted), order = [real, cloned]:")
    print(cm)

    # --- Basic calibration check ---
    # Bucket predictions by predicted P(cloned) and compare against actual
    # fraction of cloned samples in each bucket. If a model is well
    # calibrated, "70% confident" predictions should be right ~70% of
    # the time. This is a cheap sanity check, not a full calibration fit.
    print("\n=== Calibration check (predicted P(cloned) vs actual cloned rate) ===")
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    bins = np.linspace(0, 1, 6)
    for i in range(len(bins) - 1):
        mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
        if mask.sum() == 0:
            continue
        actual_rate = labels_arr[mask].mean()
        print(f"  Predicted [{bins[i]:.1f}-{bins[i+1]:.1f}): "
              f"n={mask.sum():3d}, actual cloned rate={actual_rate:.2f}")
    print(
        "If these numbers diverge a lot (e.g. 'confident' bucket has a low "
        "actual rate), consider adding temperature scaling before using "
        "these probabilities directly in the risk score."
    )

    results = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm.tolist(),
    }
    out_path = Path(cfg["training"]["checkpoint_dir"]) / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)

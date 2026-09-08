"""
Dataset preparation script.

WHAT THIS EXPECTS ON DISK (you place files here yourself):

    data/raw/real/<speaker_id>/*.wav
    data/raw/cloned/<speaker_id>/*.wav

Notes on datasets (do NOT blindly trust any single number you read
online - always re-check the current state of a dataset before
relying on it):

  - A commonly used public benchmark for this exact task ("bonafide"
    vs "spoofed" speech, including TTS and voice-conversion attacks)
    is the ASVspoof series (e.g. ASVspoof2019 - Logical Access track).
    It is free for research use but you must register/download it
    yourself from the official ASVspoof site - we do not embed or
    fetch it here.
  - If ASVspoof's folder layout doesn't match the one above, write a
    tiny one-off script that copies/symlinks files into this layout
    (real/<speaker_id>/ and cloned/<speaker_id>/) - do not restructure
    this whole module for one dataset's quirks.
  - For a quick MVP demo you can also SELF-GENERATE a small "cloned"
    set: record a few of your own team members ("real"), then run any
    free TTS/voice-cloning tool on the same sentences to get "cloned"
    samples of a synthetic voice. This is enough to demonstrate the
    pipeline end-to-end even before the full dataset is ready.

WHY SPLIT BY SPEAKER (this is the single most important thing to get
right - most beginner mistakes come from skipping this):

  If the same speaker's clips appear in both train and test, the model
  can learn to recognise "the sound of this specific person's room and
  microphone" instead of "the sound of synthetic speech in general".
  That gives a fake, inflated accuracy that collapses on new speakers.
  So we split speakers into train/val/test, never individual files.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from src.utils.config_loader import load_config


def scan_speakers(raw_dir: Path, class_name: str):
    """Returns {speaker_id: [file_path, ...]} for one class folder."""
    class_dir = raw_dir / class_name
    speakers = defaultdict(list)
    if not class_dir.exists():
        return speakers
    for speaker_dir in sorted(class_dir.iterdir()):
        if speaker_dir.is_dir():
            files = sorted(
                p for p in speaker_dir.glob("*")
                if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")
            )
            if files:
                speakers[speaker_dir.name] = files
    return speakers


def split_speakers(speaker_ids, train_ratio, val_ratio, seed):
    speaker_ids = list(speaker_ids)
    random.Random(seed).shuffle(speaker_ids)
    n = len(speaker_ids)
    n_train = max(1, int(n * train_ratio)) if n > 0 else 0
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    train = speaker_ids[:n_train]
    val = speaker_ids[n_train:n_train + n_val]
    test = speaker_ids[n_train + n_val:]
    return train, val, test


def build_manifest(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["data"]["classes"]
    seed = cfg["data"]["random_seed"]

    manifest = {"train": [], "val": [], "test": []}
    per_class_speaker_counts = {}

    for class_name in classes:
        speakers = scan_speakers(raw_dir, class_name)
        per_class_speaker_counts[class_name] = len(speakers)

        if len(speakers) == 0:
            print(f"[WARN] No speakers found for class '{class_name}' under {raw_dir/class_name}. "
                  f"Add data before training.")
            continue

        train_sp, val_sp, test_sp = split_speakers(
            speakers.keys(),
            cfg["data"]["train_ratio"],
            cfg["data"]["val_ratio"],
            seed,
        )

        for split_name, sp_list in [("train", train_sp), ("val", val_sp), ("test", test_sp)]:
            for sp in sp_list:
                for f in speakers[sp]:
                    manifest[split_name].append({
                        "path": str(f),
                        "label": class_name,
                        "speaker_id": sp,
                    })

    # Simple class balance report (does NOT auto-fix - see note below)
    for split_name in ["train", "val", "test"]:
        counts = defaultdict(int)
        for item in manifest[split_name]:
            counts[item["label"]] += 1
        print(f"[{split_name}] total={len(manifest[split_name])} | per-class={dict(counts)}")

    print(f"Speakers per class: {per_class_speaker_counts}")
    print(
        "[INFO] If one class has far fewer clips than the other, handle it at "
        "training time via class_weight in the loss function (see train.py) "
        "rather than deleting real data."
    )

    for split_name, items in manifest.items():
        out_path = splits_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"Wrote {out_path} ({len(items)} items)")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val/test manifests, split by speaker.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_manifest(cfg)

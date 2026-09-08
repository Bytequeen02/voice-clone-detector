"""
Core detection module. This is the main integration point for the
rest of the team:

  - Chahat (backend) calls detect_voice_clone(path) from the API layer.
  - Bhavya's speaker-verification module plugs its similarity score
    into risk_scoring/risk_score.py (see that file for the exact hook).
  - Nitin's dashboard consumes the JSON this returns.

Output contract (as specified in the problem statement):
{
  "prediction": "real" | "cloned",
  "cloned_probability": float in [0,1],
  "confidence": float in [0,1]
}
This module also attaches risk_score + explanation, since those are
part of "my" (Kashish's) core AI responsibility.
"""
from pathlib import Path

import torch

from src.utils.config_loader import load_config
from src.preprocessing.audio_preprocessing import (
    load_audio, trim_silence, normalize, fix_length,
    log_mel_spectrogram, basic_quality_flags,
)
from src.models.cnn_baseline import CNNBaseline
from src.risk_scoring.risk_score import compute_risk_score
from src.explainability.explain import build_explanation

_MODEL_CACHE = {}


def _get_model(cfg, device):
    """Loads the model once and reuses it (important for an API server -
    do NOT reload the model from disk on every request)."""
    key = str(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt")
    if key not in _MODEL_CACHE:
        ckpt = torch.load(key, map_location=device)
        model = CNNBaseline(
            num_classes=ckpt["config"]["model"]["num_classes"],
            dropout=ckpt["config"]["model"]["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        _MODEL_CACHE[key] = (model, ckpt["config"])
    return _MODEL_CACHE[key]


def detect_voice_clone(
    audio_path: str,
    cfg: dict = None,
    speaker_similarity: float = None,
    attempt_count: int = 0,
) -> dict:
    """
    Main entry point.

    speaker_similarity: optional float in [0,1] from Bhavya's module
        (1.0 = matches claimed speaker perfectly). Leave as None until
        that module is integrated - the risk scorer handles that case.
    attempt_count: optional int, number of recent suspicious attempts
        for this session/user, if such tracking exists later.
    """
    cfg = cfg or load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = _get_model(cfg, device)

    a_cfg = model_cfg["audio"]
    f_cfg = model_cfg["features"]

    y = load_audio(audio_path, target_sr=a_cfg["sample_rate"])
    quality_flags = basic_quality_flags(y, a_cfg["sample_rate"])

    if a_cfg.get("trim_silence", True):
        y = trim_silence(y, top_db=a_cfg.get("top_db", 30))
    y = normalize(y)
    target_len = int(a_cfg["duration_seconds"] * a_cfg["sample_rate"])
    y = fix_length(y, target_len)

    features = log_mel_spectrogram(
        y, sr=a_cfg["sample_rate"], n_mels=f_cfg["n_mels"],
        n_fft=f_cfg["n_fft"], hop_length=f_cfg["hop_length"],
        win_length=f_cfg["win_length"],
    )
    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,n_mels,time)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    cloned_probability = float(probs[1].item())
    prediction = "cloned" if cloned_probability >= 0.5 else "real"
    confidence = float(max(probs).item())

    risk = compute_risk_score(
        cloned_probability=cloned_probability,
        speaker_similarity=speaker_similarity,
        quality_flags=quality_flags,
        attempt_count=attempt_count,
        cfg=cfg,
    )

    explanation = build_explanation(
        prediction=prediction,
        cloned_probability=cloned_probability,
        speaker_similarity=speaker_similarity,
        quality_flags=quality_flags,
        risk_level=risk["risk_level"],
    )

    return {
        "prediction": prediction,
        "cloned_probability": round(cloned_probability, 4),
        "confidence": round(confidence, 4),
        "risk_score": risk,
        "explanation": explanation,
        "quality_flags": quality_flags,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    args = parser.parse_args()

    result = detect_voice_clone(args.audio_path)
    print(json.dumps(result, indent=2))

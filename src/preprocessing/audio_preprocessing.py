"""
Audio preprocessing module.

This is intentionally written as a set of small, pure functions
(input -> output, no hidden state) so that:
  1. It is easy to unit test.
  2. Vanshika can later swap/extend individual steps (e.g. denoising,
     VAD-based trimming) without touching the rest of the pipeline.
  3. The exact same functions are used at training time AND at
     inference time -> no train/serve mismatch.

Pipeline order:
    load -> resample -> mono -> trim_silence -> normalize -> fix_length -> log_mel
"""

import numpy as np
import librosa


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file and resample it. Returns mono float32 waveform."""
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Remove leading/trailing silence. Keeps internal silence intact
    (internal pauses can be a genuine speech characteristic)."""
    if len(y) == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if len(y_trimmed) > 0 else y


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1, 1] to remove loudness as a
    confounding factor (loudness should not affect real-vs-cloned)."""
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 1e-8:
        y = y / peak
    return y


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or center-crop so every sample fed to the model
    has the exact same length (required for fixed-size CNN input)."""
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    pad_total = target_len - len(y)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> np.ndarray:
    """Convert waveform -> log-mel spectrogram (n_mels x time_frames).

    Why log-mel: it compresses raw audio into a compact time-frequency
    picture that highlights the kind of unnatural spectral artifacts
    (buzziness, missing harmonics, odd formant transitions) that many
    TTS/voice-conversion systems leave behind - and it is cheap enough
    to run on a CPU/laptop GPU.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to roughly [0, 1] for stable training
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def preprocess_file(path: str, cfg: dict) -> np.ndarray:
    """
    Full pipeline for a single audio file, driven entirely by config.yaml.
    Returns a (n_mels, time_frames) numpy array ready to be turned into
    a tensor by the model / dataset class.
    """
    a_cfg = cfg["audio"]
    f_cfg = cfg["features"]

    y = load_audio(path, target_sr=a_cfg["sample_rate"])

    if a_cfg.get("trim_silence", True):
        y = trim_silence(y, top_db=a_cfg.get("top_db", 30))

    y = normalize(y)

    target_len = int(a_cfg["duration_seconds"] * a_cfg["sample_rate"])
    y = fix_length(y, target_len)

    features = log_mel_spectrogram(
        y,
        sr=a_cfg["sample_rate"],
        n_mels=f_cfg["n_mels"],
        n_fft=f_cfg["n_fft"],
        hop_length=f_cfg["hop_length"],
        win_length=f_cfg["win_length"],
    )
    return features


def basic_quality_flags(y: np.ndarray, sr: int) -> dict:
    """
    Cheap, explainable audio-quality signals used later by the risk
    scorer (NOT by the classifier itself). These are heuristics, not
    ML predictions, so they are safe to compute even with zero training.
    """
    if len(y) == 0:
        return {"low_energy": True, "clipping": False, "very_short": True}

    rms = float(np.sqrt(np.mean(y ** 2)))
    clipping_ratio = float(np.mean(np.abs(y) > 0.99))
    duration = len(y) / sr

    return {
        "low_energy": rms < 0.01,
        "clipping": clipping_ratio > 0.01,
        "very_short": duration < 1.0,
    }

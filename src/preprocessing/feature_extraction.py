"""
feature_extraction.py
----------------------
Extracts model-ready acoustic features from a preprocessed audio segment:
MFCCs, and spectral features (centroid, bandwidth, rolloff, zero-crossing
rate, chroma, flatness).

Driven entirely by configs/config.yaml.
"""

import numpy as np
import librosa


def extract_mfcc(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Returns MFCCs with delta and delta-delta stacked -> (N_MFCC*3, T)."""
    f_cfg = cfg["features"]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=f_cfg["n_mfcc"],
        n_fft=f_cfg["n_fft"],
        hop_length=f_cfg["hop_length"],
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_spectral_features(audio: np.ndarray, sr: int, cfg: dict) -> dict:
    """Scalar-summarized spectral features (mean + std), useful for
    risk-scoring / explainability."""
    f_cfg = cfg["features"]
    n_fft = f_cfg["n_fft"]
    hop_length = f_cfg["hop_length"]

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)

    def stats(feat):
        return {"mean": float(np.mean(feat)), "std": float(np.std(feat))}

    return {
        "spectral_centroid": stats(centroid),
        "spectral_bandwidth": stats(bandwidth),
        "spectral_rolloff": stats(rolloff),
        "zero_crossing_rate": stats(zcr),
        "chroma_mean": float(np.mean(chroma)),
        "spectral_flatness": stats(flatness),
    }


def extract_all_features(audio: np.ndarray, sr: int, cfg: dict) -> dict:
    """Runs all extractors on one segment, packaged for the model /
    explainability layer."""
    mfcc = extract_mfcc(audio, sr, cfg)
    spectral = extract_spectral_features(audio, sr, cfg)

    return {
        "mfcc": mfcc,
        "spectral_features": spectral,
    }
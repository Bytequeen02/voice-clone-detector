"""
feature_extraction.py
----------------------
Extracts model-ready acoustic features from a preprocessed audio segment:
MFCCs, Mel-spectrogram, and spectral features (centroid, bandwidth,
rolloff, zero-crossing rate, chroma).

Author: Vanshika (Audio Processing & Feature Extraction module)
"""

import numpy as np
import librosa

from . import config


def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns MFCCs with delta and delta-delta (acceleration) coefficients
    stacked together -> shape: (N_MFCC*3, time_steps)
    This richer representation generally helps cloning-artifact detection.
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns log-scaled Mel-spectrogram -> shape: (N_MELS, time_steps)
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def extract_spectral_features(audio: np.ndarray, sr: int) -> dict:
    """
    Returns scalar-summarized spectral features (mean + std across time),
    useful as auxiliary signals for risk-scoring / explainability
    (e.g. showing "why flagged as suspicious" on Nitin's dashboard).
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=config.HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    def stats(feat):
        return {"mean": float(np.mean(feat)), "std": float(np.std(feat))}

    return {
        "spectral_centroid": stats(centroid),
        "spectral_bandwidth": stats(bandwidth),
        "spectral_rolloff": stats(rolloff),
        "zero_crossing_rate": stats(zcr),
        "chroma_mean": float(np.mean(chroma)),
        "spectral_flatness": stats(flatness),   # low flatness = tonal (often synthetic voices are more tonal/uniform)
    }


def extract_all_features(audio: np.ndarray, sr: int) -> dict:
    """
    Runs all feature extractors on a single audio segment and packages
    them into one dict — this is the exact object handed off to
    Kashish's core model per segment.
    """
    mfcc = extract_mfcc(audio, sr)
    mel_spec = extract_mel_spectrogram(audio, sr)
    spectral = extract_spectral_features(audio, sr)

    return {
        "mfcc": mfcc,                 # np.ndarray (120, T)  -> for CNN/RNN input
        "mel_spectrogram": mel_spec,  # np.ndarray (80, T)   -> for CNN input
        "spectral_features": spectral,  # dict of scalars     -> for explainability / auxiliary features
    }

"""
preprocessing.py
-----------------
Handles: loading audio, mono conversion, resampling, noise reduction,
silence trimming, and segmentation into fixed-length chunks.

Author: Vanshika (Audio Processing & Feature Extraction module)
Team: SIH26104 - AI-Powered Real-Time Detection of Voice Cloning Attacks
"""

import numpy as np
import librosa
import soundfile as sf

try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    _NOISEREDUCE_AVAILABLE = False

from . import config


def load_audio(file_path: str):
    """
    Load an audio file (wav/mp3/flac/etc.), convert to mono,
    and resample to the target sample rate.

    Returns:
        audio (np.ndarray): 1D float32 waveform
        sr (int): sample rate (always config.TARGET_SAMPLE_RATE)
    """
    audio, sr = librosa.load(
        file_path,
        sr=config.TARGET_SAMPLE_RATE,
        mono=config.MONO,
    )
    audio = audio.astype(np.float32)
    return audio, sr


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral-gating noise reduction to clean background noise.
    Falls back to the raw signal if the `noisereduce` package isn't installed.
    """
    if not config.APPLY_NOISE_REDUCTION:
        return audio

    if not _NOISEREDUCE_AVAILABLE:
        print("[preprocessing] Warning: 'noisereduce' not installed, skipping denoising.")
        return audio

    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=config.NOISE_REDUCE_PROP_DECREASE,
    )
    return cleaned.astype(np.float32)


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """
    Remove leading/trailing silence from the waveform.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=config.TRIM_TOP_DB)
    return trimmed


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalize the waveform to [-1, 1] range to remove volume bias
    between different recordings/devices.
    """
    peak = np.max(np.abs(audio)) if audio.size > 0 else 0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def segment_audio(audio: np.ndarray, sr: int):
    """
    Split a (potentially long) waveform into fixed-length overlapping segments.
    This is what actually gets fed into feature extraction, one segment at a time,
    which also enables real-time/streaming-style processing later.

    Returns:
        List[dict]: [{"segment_id": int, "start_time": float,
                       "end_time": float, "audio": np.ndarray}, ...]
    """
    seg_len = int(config.SEGMENT_DURATION_SEC * sr)
    hop_len = int((config.SEGMENT_DURATION_SEC - config.SEGMENT_OVERLAP_SEC) * sr)
    min_len = int(config.MIN_SEGMENT_DURATION_SEC * sr)

    segments = []
    start = 0
    seg_id = 0

    # If audio shorter than one segment, just pad it and return single segment
    if len(audio) <= seg_len:
        padded = np.pad(audio, (0, max(0, seg_len - len(audio))))
        segments.append({
            "segment_id": 0,
            "start_time": 0.0,
            "end_time": len(audio) / sr,
            "audio": padded,
        })
        return segments

    while start < len(audio):
        end = start + seg_len
        chunk = audio[start:end]

        if len(chunk) < min_len:
            break  # discard too-short trailing chunk

        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        segments.append({
            "segment_id": seg_id,
            "start_time": round(start / sr, 3),
            "end_time": round(min(end, len(audio)) / sr, 3),
            "audio": chunk,
        })

        seg_id += 1
        start += hop_len

    return segments


def preprocess_pipeline(file_path: str):
    """
    Full preprocessing pipeline: load -> denoise -> trim silence -> normalize -> segment.

    Returns:
        sr (int), segments (List[dict])
    """
    audio, sr = load_audio(file_path)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    segments = segment_audio(audio, sr)
    return sr, segments


def save_wav(audio: np.ndarray, sr: int, out_path: str):
    """Utility to save a processed waveform back to disk (useful for debugging)."""
    sf.write(out_path, audio, sr)

"""
pipeline.py
-----------
This is the SINGLE ENTRY POINT for the Audio Processing & Feature
Extraction module. Kashish's core model and Chahat's backend API
should only ever call `process_audio_file()` from here.

============================================================
 INPUT
============================================================
    file_path: str
        Path to a raw audio file (.wav, .mp3, .flac, .m4a, etc.)
        Any sample rate / channel count is accepted — it gets
        standardized internally.

============================================================
 OUTPUT  (dict) — CONTRACT FOR INTEGRATION
============================================================
{
    "sample_rate": 16000,
    "duration_sec": 7.2,
    "num_segments": 3,
    "segments": [
        {
            "segment_id": 0,
            "start_time": 0.0,
            "end_time": 3.0,
            "mfcc": np.ndarray shape (120, T),          # model input option A
            "mel_spectrogram": np.ndarray shape (80, T),# model input option B
            "spectral_features": {                       # for risk-score explainability
                "spectral_centroid": {"mean": ..., "std": ...},
                "spectral_bandwidth": {...},
                "spectral_rolloff": {...},
                "zero_crossing_rate": {...},
                "chroma_mean": ...,
                "spectral_flatness": {...}
            }
        },
        ...
    ]
}

Kashish's model can pick either "mfcc" or "mel_spectrogram" (or both) as
the tensor input per segment, and use "spectral_features" as extra
signals for the risk-scoring logic / explanation text on the dashboard.

For an API/JSON-safe version (no raw numpy arrays), use
`process_audio_file_json_safe()` which converts arrays to nested lists.
"""

import os
import numpy as np

from . import config
from .preprocessing import preprocess_pipeline
from .feature_extraction import extract_all_features


def process_audio_file(file_path: str) -> dict:
    """
    Full pipeline: preprocessing -> segmentation -> feature extraction.
    Returns a dict with numpy arrays (fast, for in-process model calls).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sr, segments = preprocess_pipeline(file_path)

    if len(segments) == 0:
        raise ValueError("No usable audio detected after preprocessing "
                          "(file may be silent, empty, or corrupted).")

    total_duration = segments[-1]["end_time"]
    output_segments = []

    for seg in segments:
        features = extract_all_features(seg["audio"], sr)
        output_segments.append({
            "segment_id": seg["segment_id"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "mfcc": features["mfcc"],
            "mel_spectrogram": features["mel_spectrogram"],
            "spectral_features": features["spectral_features"],
        })

    return {
        "sample_rate": sr,
        "duration_sec": total_duration,
        "num_segments": len(output_segments),
        "segments": output_segments,
    }


def process_audio_file_json_safe(file_path: str) -> dict:
    """
    Same as process_audio_file(), but converts numpy arrays to plain
    Python lists so the result can be sent directly as JSON over
    Chahat's API layer (e.g., audio-upload -> preprocessing -> response).
    """
    result = process_audio_file(file_path)
    for seg in result["segments"]:
        seg["mfcc"] = seg["mfcc"].tolist()
        seg["mel_spectrogram"] = seg["mel_spectrogram"].tolist()
    return result


def save_features_npz(file_path: str, out_dir: str = None) -> str:
    """
    Runs the pipeline and saves all segment features to a single .npz file
    on disk — useful for Kashish's offline model training/evaluation
    instead of repeatedly recomputing features from raw audio.

    Returns the path to the saved .npz file.
    """
    out_dir = out_dir or config.FEATURE_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    result = process_audio_file(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_features.npz")

    mfcc_stack = np.stack([s["mfcc"] for s in result["segments"]])
    mel_stack = np.stack([s["mel_spectrogram"] for s in result["segments"]])
    meta = [
        {"segment_id": s["segment_id"], "start_time": s["start_time"],
         "end_time": s["end_time"], "spectral_features": s["spectral_features"]}
        for s in result["segments"]
    ]

    np.savez_compressed(
        out_path,
        mfcc=mfcc_stack,
        mel_spectrogram=mel_stack,
        sample_rate=result["sample_rate"],
        duration_sec=result["duration_sec"],
        meta=meta,
    )
    return out_path

"""
demo.py
-------
Quick demo of the Audio Processing & Feature Extraction module.

Usage:
    python demo.py path/to/audio.wav
"""

import sys
import json
from audio_processing.pipeline import process_audio_file, process_audio_file_json_safe


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\nProcessing: {file_path}\n")
    result = process_audio_file(file_path)

    print(f"Sample rate      : {result['sample_rate']} Hz")
    print(f"Duration         : {result['duration_sec']} sec")
    print(f"Number of segments: {result['num_segments']}\n")

    for seg in result["segments"]:
        print(f"--- Segment {seg['segment_id']} ({seg['start_time']}s - {seg['end_time']}s) ---")
        print(f"  MFCC shape           : {seg['mfcc'].shape}")
        print(f"  Mel-spectrogram shape: {seg['mel_spectrogram'].shape}")
        print(f"  Spectral features    : {json.dumps(seg['spectral_features'], indent=2)}")
        print()

    # Example: JSON-safe version for API testing (Chahat's backend)
    json_result = process_audio_file_json_safe(file_path)
    with open("sample_output.json", "w") as f:
        json.dump(json_result, f, indent=2)
    print("Saved JSON-safe output -> sample_output.json")


if __name__ == "__main__":
    main()

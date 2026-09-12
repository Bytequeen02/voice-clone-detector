"""
config.py
----------
Central configuration for the Audio Processing & Feature Extraction module.
Keep these values consistent with what Kashish's model expects as input,
so preprocessing + feature extraction always match model requirements.
"""

# ---- Audio standardization ----
TARGET_SAMPLE_RATE = 16000      # Hz - standard for most voice/speaker models
MONO = True                     # convert all audio to mono

# ---- Silence trimming ----
TRIM_TOP_DB = 25                # higher = more aggressive silence trimming

# ---- Segmentation ----
SEGMENT_DURATION_SEC = 3.0      # length of each audio chunk fed to the model
SEGMENT_OVERLAP_SEC = 0.5       # overlap between consecutive segments
MIN_SEGMENT_DURATION_SEC = 1.0  # discard trailing segments shorter than this

# ---- Noise reduction ----
APPLY_NOISE_REDUCTION = True
NOISE_REDUCE_PROP_DECREASE = 0.75   # 0.0 (none) - 1.0 (aggressive)

# ---- Feature extraction ----
N_MFCC = 40
N_FFT = 400          # 25 ms window at 16kHz
HOP_LENGTH = 160     # 10 ms hop at 16kHz
N_MELS = 80

# ---- Output ----
FEATURE_OUTPUT_DIR = "processed_features"

"""
Basic unit tests that do NOT require a trained model or real audio
files - they test the pure logic pieces (Azad: extend this file with
more cases as you test the pipeline).

Run:
    pytest tests/
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.audio_preprocessing import (
    normalize, fix_length, trim_silence, basic_quality_flags,
)
from src.risk_scoring.risk_score import compute_risk_score
from src.utils.config_loader import load_config

CFG = load_config()


def test_normalize_peak_is_one():
    y = np.array([0.0, 0.5, -1.5, 0.2], dtype=np.float32)
    out = normalize(y)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_normalize_silent_audio_no_crash():
    y = np.zeros(100, dtype=np.float32)
    out = normalize(y)
    assert np.allclose(out, 0.0)


def test_fix_length_pads_short_audio():
    y = np.ones(10, dtype=np.float32)
    out = fix_length(y, 20)
    assert len(out) == 20


def test_fix_length_crops_long_audio():
    y = np.ones(30, dtype=np.float32)
    out = fix_length(y, 20)
    assert len(out) == 20


def test_quality_flags_detects_short_clip():
    y = np.random.randn(500).astype(np.float32)  # far under 1 second at 16kHz
    flags = basic_quality_flags(y, sr=16000)
    assert flags["very_short"] is True


def test_risk_score_mvp_no_optional_signals():
    """With no speaker verification / attempt history, weight should
    fully redistribute onto cloned_probability."""
    result = compute_risk_score(
        cloned_probability=0.9,
        speaker_similarity=None,
        quality_flags={"low_energy": False, "clipping": False, "very_short": False},
        attempt_count=0,
        cfg=CFG,
    )
    assert result["effective_weights"]["speaker_mismatch"] == 0.0
    assert result["effective_weights"]["repeated_attempts"] == 0.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= result["risk_score"] <= 1.0


def test_risk_score_high_when_all_signals_bad():
    result = compute_risk_score(
        cloned_probability=0.95,
        speaker_similarity=0.1,     # big mismatch
        quality_flags={"low_energy": True, "clipping": True, "very_short": False},
        attempt_count=6,
        cfg=CFG,
    )
    assert result["risk_level"] == "HIGH"


def test_risk_score_low_when_all_signals_good():
    result = compute_risk_score(
        cloned_probability=0.02,
        speaker_similarity=0.98,
        quality_flags={"low_energy": False, "clipping": False, "very_short": False},
        attempt_count=0,
        cfg=CFG,
    )
    assert result["risk_level"] == "LOW"

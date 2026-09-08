"""
Dynamic impersonation risk score.

WHY THIS IS NOT JUST "= cloned_probability":
  cloned_probability is a single model's confidence about ONE signal
  (does this audio sound synthetic?). A real impersonation attack
  scenario has other independent signals too - e.g. does the voice
  match the person it claims to be (speaker verification), is the
  audio suspiciously low-quality/edited, has this same
  caller/session tried repeatedly and failed before. The risk score
  fuses these into one operational number so a downstream
  system (or a human) can decide what to DO, not just what the
  classifier thinks about the audio in isolation. It is also designed
  to keep working (in a reduced form) even when some signals aren't
  available yet - which matters a lot for an MVP built before every
  teammate's module exists.

FORMULA (weighted sum, each term normalized to [0,1]):

    risk = w1 * cloned_probability
         + w2 * speaker_mismatch          (0 if speaker verification not yet integrated)
         + w3 * audio_quality_flag        (0 or 1, heuristic-based)
         + w4 * repeated_attempts_signal  (0 if no attempt-history tracking yet)

    weights (w1..w4) are read from configs/config.yaml -> risk_score.weights
    and sum to 1.0, so `risk` stays in [0, 1].

    IMPORTANT: when speaker_similarity / attempt_count are not
    available (None / 0), their weight is NOT silently dropped -
    it is redistributed onto cloned_probability so the score still
    spans the full [0,1] range for the MVP. This is done explicitly
    in `_effective_weights()` below, not hidden inside the formula.

SCORE RANGE: 0.0 (very likely a genuine, low-risk call) to
             1.0 (very likely a cloned-voice impersonation attempt).

RISK LEVELS (thresholds also in config.yaml, tune during testing):
    LOW    : risk <= 0.34
    MEDIUM : 0.34 < risk <= 0.69
    HIGH   : risk > 0.69

HOW THIS UPDATES WHEN NEW MODULES ARRIVE:
  - Bhavya's speaker verification -> pass `speaker_similarity` (0..1)
    into compute_risk_score(); speaker_mismatch = 1 - speaker_similarity.
  - A "repeated attempts" tracker (e.g. in Chahat's backend / a DB) ->
    pass `attempt_count`; it's converted to a 0..1 signal via a simple
    saturating function (more attempts -> higher signal, capped at 1).
  - Any FUTURE signal (e.g. call-metadata anomalies) can be added the
    same way: add a weight in config.yaml, add a term here, done -
    no other module needs to change.
"""


def _quality_signal(quality_flags: dict) -> float:
    """Turns the heuristic quality flags into a single 0/1-ish signal.
    Any one red flag present -> flag the audio as quality-suspicious.
    (Kept deliberately simple/explainable for the MVP.)"""
    if not quality_flags:
        return 0.0
    return 1.0 if any(quality_flags.values()) else 0.0


def _attempts_signal(attempt_count: int) -> float:
    """Saturating signal: 0 attempts -> 0.0, grows toward 1.0 as
    repeated suspicious attempts pile up. Capped at 5 attempts = 1.0."""
    if not attempt_count or attempt_count <= 0:
        return 0.0
    return min(attempt_count / 5.0, 1.0)


def _effective_weights(cfg, has_speaker_signal: bool, has_attempt_signal: bool):
    """
    Redistributes weight from unavailable signals onto cloned_probability,
    so the MVP (no speaker verification, no attempt history yet) still
    produces a meaningful, full-range score using only the detector.
    """
    w = dict(cfg["risk_score"]["weights"])  # copy
    redistributed = 0.0

    if not has_speaker_signal:
        redistributed += w["speaker_mismatch"]
        w["speaker_mismatch"] = 0.0
    if not has_attempt_signal:
        redistributed += w["repeated_attempts"]
        w["repeated_attempts"] = 0.0

    w["cloned_probability"] += redistributed
    return w


def compute_risk_score(
    cloned_probability: float,
    speaker_similarity: float = None,
    quality_flags: dict = None,
    attempt_count: int = 0,
    cfg: dict = None,
) -> dict:
    weights = _effective_weights(
        cfg,
        has_speaker_signal=speaker_similarity is not None,
        has_attempt_signal=bool(attempt_count),
    )

    speaker_mismatch = (1.0 - speaker_similarity) if speaker_similarity is not None else 0.0
    quality_signal = _quality_signal(quality_flags)
    attempts_signal = _attempts_signal(attempt_count)

    risk = (
        weights["cloned_probability"] * cloned_probability
        + weights["speaker_mismatch"] * speaker_mismatch
        + weights["audio_quality_flag"] * quality_signal
        + weights["repeated_attempts"] * attempts_signal
    )
    risk = max(0.0, min(1.0, risk))

    thresholds = cfg["risk_score"]["thresholds"]
    if risk <= thresholds["low_max"]:
        level = "LOW"
    elif risk <= thresholds["medium_max"]:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "risk_score": round(risk, 4),
        "risk_level": level,
        "components": {
            "cloned_probability": round(cloned_probability, 4),
            "speaker_mismatch": round(speaker_mismatch, 4),
            "quality_signal": quality_signal,
            "attempts_signal": round(attempts_signal, 4),
        },
        "effective_weights": {k: round(v, 3) for k, v in weights.items()},
    }

"""
Explainability module.

Produces plain-language reasons for the dashboard/report. This is
RULE-BASED (thresholds on numbers we already computed), not a
separate ML model - simple, transparent, and easy to defend in a
hackathon Q&A ("how did you get this explanation?" -> "here is the
exact rule").

IMPORTANT LIMITATION (state this honestly in the demo):
  This system says whether audio LOOKS synthetic/mismatched and how
  risky that combination is. It does NOT identify which specific
  voice-cloning tool or algorithm was used - the model is not trained
  for that task, so we do not claim it.
"""


def build_explanation(
    prediction: str,
    cloned_probability: float,
    speaker_similarity: float,
    quality_flags: dict,
    risk_level: str,
) -> list:
    reasons = []

    if prediction == "cloned":
        if cloned_probability >= 0.85:
            reasons.append(
                "The audio strongly contains characteristics commonly associated with synthetic speech."
            )
        else:
            reasons.append(
                "The audio contains some characteristics commonly associated with synthetic speech, "
                "though the signal is not extremely strong."
            )
    else:
        reasons.append(
            "The audio's characteristics are consistent with natural human speech based on the current model."
        )

    if speaker_similarity is not None:
        if speaker_similarity < 0.5:
            reasons.append(
                "The detected voice does not sufficiently match the reference speaker."
            )
        else:
            reasons.append(
                "The detected voice sufficiently matches the reference speaker."
            )
    else:
        reasons.append(
            "Speaker verification was not available for this check, so identity matching was not assessed."
        )

    if quality_flags:
        if quality_flags.get("very_short"):
            reasons.append("The audio clip is very short, which reduces detection reliability.")
        if quality_flags.get("low_energy"):
            reasons.append("The audio has unusually low energy/volume, which can affect analysis quality.")
        if quality_flags.get("clipping"):
            reasons.append("The audio shows signs of clipping/distortion, which can affect analysis quality.")

    reasons.append(f"The system has assigned a {risk_level.lower()} impersonation risk.")

    return reasons

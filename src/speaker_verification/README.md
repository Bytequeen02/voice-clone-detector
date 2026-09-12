# Speaker Verification Module

Speaker verification system for detecting whether an incoming voice matches a
registered/reference speaker, and for flagging likely impersonation or
cloned-voice attempts. Built on a pretrained ECAPA-TDNN speaker-embedding
model (SpeechBrain, VoxCeleb-trained).

## Files

- `speaker_verification_module.py` — core module: embedding generation,
  speaker registration, similarity scoring, verification decision, batch
  testing utility, and integration adapter for a core risk-scoring model.
- `speaker_verification_evaluation.py` — evaluation extension: audio
  condition augmentation (noise, gain, phone codec, speech rate), metrics
  (FAR, FRR, EER, AUC-ROC, accuracy, precision, recall, F1), threshold
  calibration, and systematic multi-condition testing. Imports from
  `speaker_verification_module.py`, so keep both files in the same
  directory (or same Colab session).
- `requirements.txt` — Python package dependencies.

## Setup (Google Colab or local)

```bash
pip install -r requirements.txt
```

`speaker_verification_evaluation.py` uses `torchaudio.sox_effects` for the
`fast_speech` / `slow_speech` conditions, which requires the `sox` binary on
the system. On Colab / Debian-based systems:

```bash
apt-get -qq install -y sox
```

If `sox` is not available, remove or skip the `fast_speech` / `slow_speech`
conditions when calling `SystematicConditionTester`.

## Quick start

```python
from speaker_verification_module import SpeakerVerificationModule

sv_module = SpeakerVerificationModule()

sv_module.register_speaker("user_123", ["ref_clip1.wav", "ref_clip2.wav"])

result = sv_module.verify(
    reference_audio="ref_clip1.wav",
    test_audio="incoming_call.wav",
)
print(result.to_json())
```

Standardized output fields:

- `speaker_similarity_score` — normalized similarity, 0.0–1.0
- `speaker_verified` — True if the incoming voice matches the reference
- `speaker_mismatch` — True if it's likely a different/impersonated speaker
- `confidence` — 0.0–1.0, how confident the module is in the decision

## Integrating with a core risk-scoring model

```python
from speaker_verification_module import integrate_with_core_risk_model

def my_core_risk_model(evidence_bundle):
    weight = evidence_bundle["speaker_verification"]["risk_weight"]
    return {"final_risk_score": 0.5 + weight}

combined = integrate_with_core_risk_model(result, core_risk_score_fn=my_core_risk_model)
print(combined)
```

## Threshold calibration and metrics

```python
from speaker_verification_evaluation import ThresholdCalibrator

calibrator = ThresholdCalibrator(sv_module)

genuine_pairs = [("speakerA_ref1.wav", "speakerA_ref2.wav")]
impostor_pairs = [("speakerA_ref1.wav", "speakerB_ref2.wav")]

calibration_result = calibrator.calibrate(genuine_pairs, impostor_pairs, apply_to_module=True)
print(calibration_result["eer"], calibration_result["chosen_threshold"])
```

`calibrate()` computes EER and AUC-ROC from genuine/impostor score
distributions, and (with `apply_to_module=True`) updates
`sv_module.config.VERIFIED_THRESHOLD` / `MISMATCH_THRESHOLD` automatically.
Pass `target_far=0.01` to instead pick the threshold that keeps the false
accept rate at or below 1%.

## Systematic testing across conditions

```python
from speaker_verification_evaluation import SystematicConditionTester, ConditionTestCase

tester = SystematicConditionTester(sv_module)

test_cases = [
    ConditionTestCase(
        speaker_id="speakerA",
        reference_audio="speakerA_ref1.wav",
        base_test_audio="speakerA_test.wav",
        is_genuine=True,
        conditions=["clean", "noisy_10db", "noisy_0db", "phone_codec"],
    ),
    ConditionTestCase(
        speaker_id="speakerA",
        reference_audio="speakerA_ref1.wav",
        base_test_audio="speakerB_test.wav",
        is_genuine=False,
        conditions=["clean", "noisy_10db", "noisy_0db", "phone_codec"],
    ),
]

report = tester.run(test_cases)
tester.print_report(report)
```

This runs verification across every listed condition for every test case and
reports per-condition FAR, FRR, EER, AUC-ROC, accuracy, precision, recall,
and F1 — useful for validating robustness across accents, noise levels, and
recording/codec conditions.

## Notes on threshold defaults

`VerificationConfig.VERIFIED_THRESHOLD` (0.75) and `MISMATCH_THRESHOLD`
(0.55) are starting points, not production values. Always recalibrate on
your own genuine-vs-impostor (including cloned-voice) validation set using
`ThresholdCalibrator` before deploying, since acceptable FAR/FRR trade-offs
are highly deployment-specific — a voice-authentication gate for financial
transactions typically needs a much lower FAR than a convenience feature.

# Voice Clone Detector — Core AI Module
### SIH26104 — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

This repo contains **Kashish's core AI/ML module**: voice-cloning detection,
risk scoring, and explainability. It is built to be a clean integration
point for the rest of the team.

```
Audio Input → Preprocessing → Detection Model → [Speaker Verification*] → Risk Scoring → Explanation → API/Dashboard
```
\* integrated later by Bhavya; the pipeline already works without it.

---

## 1. Technical decisions (why this design)

| Question | Decision |
|---|---|
| **Exact problem** | Binary classification: is this speech clip **real (human)** or **cloned/synthetic (TTS / voice-conversion)**? This is the well-known "spoof / anti-spoofing" detection task. |
| **MVP** | Upload/record a short audio clip → get `{prediction, cloned_probability, confidence}` + a risk score + a plain-language explanation, via a local script and a simple API. |
| **Baseline model** | Small CNN on log-mel spectrograms (`src/models/cnn_baseline.py`). Trains on CPU or a laptop GPU, no license issues, well-understood for this task. |
| **Advanced (optional, only if time permits)** | Swap the hand-built CNN features for a pretrained self-supervised speech model (wav2vec2 / WavLM) as a frozen feature extractor + a small classifier head on top. Only attempt this after the baseline works end-to-end. |
| **Datasets** | Publicly known benchmark for exactly this task: the **ASVspoof** series (e.g. ASVspoof2019 Logical Access), free for research use — you must register/download it yourselves from the official source; we do not embed or fetch it here, and you should re-verify current availability/link since dataset hosting changes over time. For a fast demo before the full dataset is ready, you can self-generate a small set (record your own team as "real", run any free TTS tool on the same sentences for "cloned"). |
| **No GPU / no paid API required** | Yes — CNN baseline trains fine on CPU for a small dataset; nothing in the MVP calls a paid API. |
| **What we're deliberately NOT building (Advanced/Optional)** | Real-time streaming detection, multi-language robustness, identifying *which* cloning tool was used, continuous/online learning, a mobile app, multi-speaker diarization. These are listed as stretch goals only — do not attempt them before the MVP is solid. |

**No fabricated numbers anywhere in this repo.** Accuracy/precision/recall/F1
are computed live by `evaluate.py` only after you actually train a model on
real data — until then, don't quote any performance figures in your PPT or demo.

---

## 2. Folder structure

```
voice-clone-detector/
├── configs/config.yaml          # single source of truth for all settings
├── data/
│   ├── raw/real/<speaker_id>/*.wav      # you add these
│   ├── raw/cloned/<speaker_id>/*.wav    # you add these
│   ├── processed/                       # (optional cache, currently unused by default pipeline)
│   └── splits/{train,val,test}.json     # auto-generated manifests
├── checkpoints/                 # trained model + training history (git-ignored)
├── src/
│   ├── preprocessing/           # audio_preprocessing.py, prepare_dataset.py
│   ├── models/                  # cnn_baseline.py
│   ├── training/                # dataset.py, train.py, evaluate.py
│   ├── inference/               # detect.py  <-- main integration entry point
│   ├── risk_scoring/            # risk_score.py
│   ├── explainability/          # explain.py
│   ├── api/                     # app.py (Flask API for backend/dashboard)
│   └── utils/                   # config_loader.py
├── tests/test_core_module.py    # unit tests, run with `pytest tests/`
└── requirements.txt
```

## 3. Audio format requirements

- Any common format librosa can read (`.wav`, `.flac`, `.mp3`, `.ogg`) — everything gets converted internally.
- Internally standardized to: **16 kHz, mono, 4-second fixed length** (configurable in `configs/config.yaml`).
- Organize raw data as `data/raw/<real|cloned>/<speaker_id>/<file>.wav` — the `speaker_id` folder is required for leakage-safe splitting (see below).

## 4. Preventing data leakage (read this before training)

`prepare_dataset.py` splits **by speaker**, not by individual file. If the
same speaker appeared in both train and test, the model could learn to
recognize *that person's microphone/room*, not synthetic-speech artifacts in
general — giving a fake, inflated accuracy. Always keep this speaker-level
split even if it means fewer, "less clean" numbers.

Class imbalance (e.g. many more real than cloned clips) is **not** fixed by
deleting data — `train.py` computes class weights from your actual manifest
and feeds them into the loss function automatically.

## 5. How to run

```bash
pip install -r requirements.txt

# 1. Put your audio under data/raw/real/<speaker>/*.wav and data/raw/cloned/<speaker>/*.wav

# 2. Build train/val/test manifests (speaker-level split)
python -m src.preprocessing.prepare_dataset --config configs/config.yaml

# 3. Train the baseline CNN
python -m src.training.train --config configs/config.yaml

# 4. Evaluate on the held-out test set (real numbers only, printed live)
python -m src.training.evaluate --config configs/config.yaml

# 5. Run detection on a single file
python -m src.inference.detect path/to/clip.wav

# 6. Run the API (for Chahat's backend / Nitin's dashboard to call)
python -m src.api.app
# then: curl -X POST -F "audio=@clip.wav" http://localhost:5000/detect
```

Run the tests any time (they don't need a trained model):
```bash
pytest tests/
```

## 6. Output contract (for the rest of the team)

`POST /detect` (or `detect_voice_clone()` in Python) returns:

```json
{
  "prediction": "real | cloned",
  "cloned_probability": 0.0,
  "confidence": 0.0,
  "risk_score": {
    "risk_score": 0.0,
    "risk_level": "LOW | MEDIUM | HIGH",
    "components": { "...": "..." },
    "effective_weights": { "...": "..." }
  },
  "explanation": ["human-readable reason", "..."],
  "quality_flags": { "low_energy": false, "clipping": false, "very_short": false }
}
```

### Integration hooks for teammates
- **Bhavya (speaker verification):** call `detect_voice_clone(path, speaker_similarity=<0..1>)`. Until this is passed, the risk score automatically redistributes that weight onto the detector's own probability — the MVP still works.
- **Chahat (backend):** either call the Flask API directly, or import `from src.inference.detect import detect_voice_clone` straight into your backend service. You can also pass `attempt_count` once you track repeated suspicious attempts.
- **Nitin (dashboard):** consume the JSON above; `explanation` is ready-to-display text, `risk_level` is a ready-made badge value (LOW/MEDIUM/HIGH).
- **Azad (testing/docs):** `tests/test_core_module.py` is a starting point — extend it as modules get integrated; `checkpoints/test_results.json` and `training_history.json` are auto-generated and safe to pull into documentation.

## 7. Risk score explained (short version — full formula in `risk_score.py`)

```
risk = w1·cloned_probability + w2·speaker_mismatch + w3·audio_quality_flag + w4·repeated_attempts
```
Weights sum to 1.0 and live in `config.yaml`. Any signal that isn't available
yet (speaker verification, attempt history) has its weight automatically
folded back into `cloned_probability`, so the score always spans [0,1].
Thresholds: LOW ≤ 0.34, MEDIUM ≤ 0.69, HIGH > 0.69 (tune these once you have
real test data).

This is intentionally **not** the same number as model confidence — it's a
decision-oriented fusion of multiple independent signals, designed so it
keeps improving as teammates' modules come online without changing its
public interface.

---

## 8. What's next after this module works

Once you confirm this module runs end-to-end on your machine (even with a
tiny self-recorded dataset), the next steps are:
1. Get real ASVspoof data in and re-train.
2. Have Bhavya wire in `speaker_similarity`.
3. Have Chahat wire the Flask API into the real backend.
4. Have Nitin build the dashboard against the JSON contract above.
5. (Optional/Advanced, only if time remains) try the wav2vec2/WavLM feature extractor as a drop-in replacement for the CNN.

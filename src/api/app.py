"""
Minimal Flask API exposing the core AI module.

This is the integration boundary for the rest of the team:
  - Chahat (backend) can call this directly, or reimplement the same
    endpoint in his own backend framework by importing
    src.inference.detect.detect_voice_clone directly (see below).
  - Nitin (dashboard) can call POST /detect from the frontend and
    render the JSON response.

Run:
    python -m src.api.app

Test:
    curl -X POST -F "audio=@sample.wav" http://localhost:5000/detect
"""
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.utils.config_loader import load_config
from src.inference.detect import detect_voice_clone

app = Flask(__name__)
CORS(app)  # allow Nitin's dashboard (different port/origin) to call this during dev

CFG = load_config()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Expects multipart/form-data with:
      - "audio": the audio file
    Optional form fields (filled in once other modules exist):
      - "speaker_similarity": float 0..1 from Bhavya's module
      - "attempt_count": int, from Chahat's attempt-history tracking
    """
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file provided"}), 400

    audio_file = request.files["audio"]
    speaker_similarity = request.form.get("speaker_similarity", type=float)
    attempt_count = request.form.get("attempt_count", default=0, type=int)

    suffix = Path(audio_file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        audio_file.save(tmp.name)
        try:
            result = detect_voice_clone(
                tmp.name,
                cfg=CFG,
                speaker_similarity=speaker_similarity,
                attempt_count=attempt_count,
            )
        except FileNotFoundError as e:
            return jsonify({"error": f"Model not ready: {e}"}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

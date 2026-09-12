import os
import json
import numpy as np
import torch
import torchaudio
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")

from speechbrain.inference.speaker import EncoderClassifier


class VerificationConfig:
    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    MODEL_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"

    TARGET_SAMPLE_RATE = 16000

    VERIFIED_THRESHOLD = 0.75
    MISMATCH_THRESHOLD = 0.55

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SpeakerVerificationResult:
    speaker_similarity_score: float
    speaker_verified: bool
    speaker_mismatch: bool
    confidence: float
    raw_cosine_similarity: float
    reference_audio: str
    test_audio: str
    decision_zone: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SpeakerVerificationModule:
    def __init__(self, config: VerificationConfig = VerificationConfig()):
        self.config = config
        self.encoder = EncoderClassifier.from_hparams(
            source=config.MODEL_SOURCE,
            savedir=config.MODEL_SAVEDIR,
            run_opts={"device": config.DEVICE},
        )
        self._registered_speakers: Dict[str, torch.Tensor] = {}

    def _load_audio(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.config.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.config.TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        return waveform

    def generate_embedding(self, audio_path: str) -> torch.Tensor:
        waveform = self._load_audio(audio_path)
        with torch.no_grad():
            embedding = self.encoder.encode_batch(waveform)
        embedding = embedding.squeeze()
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        return embedding

    def register_speaker(self, speaker_id: str, reference_audio_paths: List[str]) -> torch.Tensor:
        embeddings = [self.generate_embedding(p) for p in reference_audio_paths]
        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
        avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=0)
        self._registered_speakers[speaker_id] = avg_embedding
        return avg_embedding

    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return float(sim.item())

    @staticmethod
    def _normalize_score(raw_cosine: float) -> float:
        return float((raw_cosine + 1.0) / 2.0)

    def _confidence_from_score(self, normalized_score: float) -> float:
        v = self.config.VERIFIED_THRESHOLD
        m = self.config.MISMATCH_THRESHOLD
        mid = (v + m) / 2.0

        if normalized_score >= v:
            confidence = 0.7 + 0.3 * min((normalized_score - v) / max(1e-6, (1.0 - v)), 1.0)
        elif normalized_score < m:
            confidence = 0.7 + 0.3 * min((m - normalized_score) / max(1e-6, m), 1.0)
        else:
            distance_from_mid = abs(normalized_score - mid)
            half_zone = max(1e-6, (v - m) / 2.0)
            confidence = 0.3 * (distance_from_mid / half_zone)

        return round(float(min(max(confidence, 0.0), 1.0)), 4)

    def verify(
        self,
        reference_audio: str,
        test_audio: str,
        registered_speaker_id: Optional[str] = None,
    ) -> SpeakerVerificationResult:
        if registered_speaker_id and registered_speaker_id in self._registered_speakers:
            ref_embedding = self._registered_speakers[registered_speaker_id]
        else:
            ref_embedding = self.generate_embedding(reference_audio)

        test_embedding = self.generate_embedding(test_audio)

        raw_sim = self.cosine_similarity(ref_embedding, test_embedding)
        norm_score = self._normalize_score(raw_sim)
        confidence = self._confidence_from_score(norm_score)

        if norm_score >= self.config.VERIFIED_THRESHOLD:
            verified, mismatch, zone = True, False, "verified"
        elif norm_score < self.config.MISMATCH_THRESHOLD:
            verified, mismatch, zone = False, True, "mismatch"
        else:
            verified, mismatch, zone = False, False, "uncertain"

        return SpeakerVerificationResult(
            speaker_similarity_score=round(norm_score, 4),
            speaker_verified=verified,
            speaker_mismatch=mismatch,
            confidence=confidence,
            raw_cosine_similarity=round(raw_sim, 4),
            reference_audio=reference_audio if not registered_speaker_id else f"registered:{registered_speaker_id}",
            test_audio=test_audio,
            decision_zone=zone,
        )


class SpeakerVerificationTester:
    def __init__(self, module: SpeakerVerificationModule):
        self.module = module

    def run(self, test_cases: List[Dict]) -> Dict:
        results = []
        correct = 0

        for case in test_cases:
            result = self.module.verify(case["reference_audio"], case["test_audio"])
            predicted_label = "genuine" if result.speaker_verified else "impostor"
            is_correct = predicted_label == case.get("expected_label")
            correct += int(is_correct)

            results.append({
                "condition": case.get("condition", "unspecified"),
                "expected_label": case.get("expected_label"),
                "predicted_label": predicted_label,
                "correct": is_correct,
                **result.to_dict(),
            })

        accuracy = correct / len(test_cases) if test_cases else 0.0

        summary = {
            "total_cases": len(test_cases),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "results": results,
        }
        return summary

    @staticmethod
    def print_summary(summary: Dict):
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {summary['correct']}/{summary['total_cases']} correct "
              f"(accuracy={summary['accuracy']*100:.2f}%)")
        print(f"{'='*60}")
        for r in summary["results"]:
            status = "PASS" if r["correct"] else "FAIL"
            print(f"[{status}] condition={r['condition']:<20} "
                  f"expected={r['expected_label']:<10} predicted={r['predicted_label']:<10} "
                  f"score={r['speaker_similarity_score']:.3f} conf={r['confidence']:.3f} "
                  f"zone={r['decision_zone']}")


def build_risk_evidence(verification_result: SpeakerVerificationResult) -> Dict:
    if verification_result.speaker_mismatch:
        risk_signal = "high_risk"
        risk_weight = round(0.9 * verification_result.confidence, 4)
    elif verification_result.speaker_verified:
        risk_signal = "low_risk"
        risk_weight = round(-0.6 * verification_result.confidence, 4)
    else:
        risk_signal = "medium_risk"
        risk_weight = round(0.4 * (1 - verification_result.confidence) + 0.2, 4)

    evidence = {
        "module": "speaker_verification",
        "risk_signal": risk_signal,
        "risk_weight": risk_weight,
        "evidence": verification_result.to_dict(),
        "explanation": (
            f"Voice similarity score {verification_result.speaker_similarity_score} "
            f"(confidence {verification_result.confidence}) suggests the incoming voice is "
            f"{'the expected registered speaker' if verification_result.speaker_verified else 'NOT confidently the expected speaker' if verification_result.speaker_mismatch else 'inconclusive relative to the expected speaker'}."
        ),
    }
    return evidence


def integrate_with_core_risk_model(
    verification_result: SpeakerVerificationResult,
    core_risk_score_fn=None,
) -> Dict:
    evidence = build_risk_evidence(verification_result)

    if core_risk_score_fn is not None:
        combined_result = core_risk_score_fn({"speaker_verification": evidence})
        return {
            "speaker_verification_evidence": evidence,
            "core_risk_model_output": combined_result,
        }

    return {"speaker_verification_evidence": evidence}


if __name__ == "__main__":
    sv_module = SpeakerVerificationModule()

    print("Module ready.")

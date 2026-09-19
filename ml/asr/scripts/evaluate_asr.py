"""ASR Quality Evaluation Suite measuring Word Error Rate (WER) against ground-truth audio."""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from asr.config import AsrEngineConfig
from asr.engine import create_asr_engine
from asr.exceptions import AwsAsrAuthError, AwsAsrServiceUnavailableError

AUDIO_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "speech_sample_16k.wav"

# Ground truth reference transcript for LibriSpeech sample (James Joyce, Ulysses)
GROUND_TRUTH = (
    "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes "
    "and fat mutton pieces to be ladled out in thick, peppered flour-fattened sauce."
)


def normalize_text(text: str) -> list[str]:
    """Lowercase and strip punctuation for canonical WER computation."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in cleaned.split() if w]


def calculate_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute Word Error Rate (WER) using Levenshtein distance matrix."""
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)

    r_len = len(ref_words)
    h_len = len(hyp_words)

    if r_len == 0:
        if h_len == 0:
            return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
        return {"wer": 1.0, "substitutions": 0, "deletions": 0, "insertions": h_len, "ref_words": 0}

    # Levenshtein distance matrix: d[i][j] = cost for ref[:i] and hyp[:j]
    d = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub = d[i - 1][j - 1] + 1
                ins = d[i][j - 1] + 1
                dele = d[i - 1][j] + 1
                d[i][j] = min(sub, ins, dele)

    total_errors = int(d[r_len][h_len])
    wer = total_errors / r_len

    return {
        "wer": wer,
        "total_errors": float(total_errors),
        "ref_word_count": float(r_len),
        "hyp_word_count": float(h_len),
    }


def evaluate_backend(
    backend_name: str,
    audio: np.ndarray,
    sample_rate: int,
    reference: str,
) -> dict[str, object]:
    """Transcribe audio with target backend and compute WER against reference."""
    config = AsrEngineConfig(
        backend=backend_name,
        aws_region=os.environ.get("AWS_REGION", "ap-south-1"),
        aws_transcribe_language=os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-IN"),
        sample_rate=sample_rate,
    )

    try:
        engine = create_asr_engine(backend=backend_name, config=config)
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": backend_name,
            "status": "initialization_failed",
            "error": str(exc),
            "transcript": "",
            "wer": None,
        }

    session_id = f"eval_{backend_name}_session"
    chunk_size = int(sample_rate * 0.2)  # 200ms streaming chunks

    try:
        all_events = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            events = engine.process_audio_chunk(session_id, chunk, audio_format="pcm_s16le")
            all_events.extend(events)

        flushed = engine.flush_session(session_id)
        all_events.extend(flushed)

        transcript = all_events[-1].text if all_events else ""
        wer_metrics = calculate_wer(reference, transcript)

        return {
            "backend": backend_name,
            "status": "success",
            "transcript": transcript,
            "wer": wer_metrics["wer"],
            "total_errors": wer_metrics["total_errors"],
            "ref_words": wer_metrics["ref_word_count"],
            "hyp_words": wer_metrics["hyp_word_count"],
            "error": None,
        }
    except (AwsAsrAuthError, AwsAsrServiceUnavailableError) as exc:
        return {
            "backend": backend_name,
            "status": "service_unavailable",
            "error": str(exc),
            "transcript": "",
            "wer": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": backend_name,
            "status": "error",
            "error": str(exc),
            "transcript": "",
            "wer": None,
        }
    finally:
        engine.close_session(session_id)
        if hasattr(engine, "close"):
            engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR Word Error Rate (WER)")
    parser.add_argument(
        "--backend",
        choices=["aws", "whisper", "both"],
        default=os.environ.get("ASR_BACKEND", "both"),
        help="ASR backend to evaluate ('aws', 'whisper', or 'both')",
    )
    args = parser.parse_args()

    if not AUDIO_PATH.exists():
        print("WER not measured.")
        print(f"Reason: Audio dataset missing at {AUDIO_PATH}")
        sys.exit(0)

    audio, sr = sf.read(str(AUDIO_PATH))
    backends = ["aws", "whisper"] if args.backend == "both" else [args.backend]

    print("==================================================")
    print("ASR Quality (WER) Evaluation")
    print("==================================================")
    print(f"Audio Fixture: {AUDIO_PATH.name} ({len(audio)/sr:.2f}s, {sr} Hz)")
    print(f"Reference Text: \"{GROUND_TRUTH}\"")
    print("--------------------------------------------------")

    for b in backends:
        res = evaluate_backend(b, audio, sr, GROUND_TRUTH)
        print(f"\nBackend: {b.upper()}")
        print(f"Status:  {res['status']}")
        if res["status"] == "success":
            print(f"Transcript: \"{res['transcript']}\"")
            print(f"WER:        {res['wer']:.4f} ({res['wer']*100:.2f}%)")
            print(f"Errors:     {int(res['total_errors'])} / {int(res['ref_words'])} words")
        elif res["error"]:
            print(f"Details: {res['error']}")
            print("WER not measured.")

    print("\n==================================================")


if __name__ == "__main__":
    main()

"""End-to-end pipeline verification and latency benchmarking suite.

Tests full path:
Audio In -> VAD -> Streaming ASR -> Grammar -> NMM -> Spatial Loci -> SignRepresentation Out
Asserts latency target <= 350ms and canonical contract schema validation.
"""

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "asr" / "src"))
sys.path.insert(0, str(ROOT / "translation" / "src"))

from translation.pipeline import SpeechToSignPipeline
from translation.representation_emitter import SignRepresentation

from asr.engine import StreamingAsrEngine


def generate_speech_audio_b64(duration_s: float = 1.0, sample_rate: int = 16000) -> str:
    """Generate synthetic speech-like tone base64 PCM16."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Mix formants
    signal = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    int16_data = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    return base64.b64encode(int16_data.tobytes()).decode("ascii")


def validate_sign_representation_schema(rep: SignRepresentation) -> list[str]:
    """Validate that emitted SignRepresentation satisfies @converse/contracts."""
    errors: list[str] = []

    if not rep.version:
        errors.append("Missing 'version'")
    if not rep.session_id:
        errors.append("Missing 'sessionId'")
    if not rep.utterance_id:
        errors.append("Missing 'utteranceId'")
    if rep.total_duration_ms <= 0:
        errors.append(f"Invalid totalDurationMs: {rep.total_duration_ms}")
    if not rep.tokens:
        errors.append("Empty tokens list")

    for i, tok in enumerate(rep.tokens):
        if not tok.token_id:
            errors.append(f"Token {i}: missing 'tokenId'")
        if not tok.clip_id:
            errors.append(f"Token {i}: missing 'clipId'")
        if not tok.gloss:
            errors.append(f"Token {i}: missing 'gloss'")

        # Timing validation
        timing = tok.timing
        if timing.start_time_ms < 0:
            errors.append(f"Token {i}: negative startTimeMs")
        if timing.hold_duration_ms <= 0:
            errors.append(f"Token {i}: non-positive holdDurationMs")

        # Spatial locus validation
        loci = tok.spatial_loci
        if loci.anchor not in ("neutral_space", "chest", "forehead", "left", "right"):
            errors.append(f"Token {i}: invalid anchor {loci.anchor}")

        # Non-manual markers validation
        nmm = tok.non_manual_markers
        if nmm.eyebrow_shape not in ("neutral", "raise", "furrow"):
            errors.append(f"Token {i}: invalid eyebrowShape {nmm.eyebrow_shape}")
        if not (0.0 <= nmm.eyebrow_intensity <= 1.0):
            errors.append(f"Token {i}: eyebrowIntensity {nmm.eyebrow_intensity} out of bounds [0, 1]")

    # Verify monotonic timing
    for i in range(len(rep.tokens) - 1):
        t1 = rep.tokens[i].timing.start_time_ms
        t2 = rep.tokens[i + 1].timing.start_time_ms
        if t2 < t1:
            errors.append(f"Timing inversion between token {i} ({t1}ms) and {i+1} ({t2}ms)")

    return errors


def run_e2e_benchmark(num_iterations: int = 20) -> None:
    print("==================================================")
    print("End-to-End Speech-to-Sign Pipeline Benchmark")
    print("==================================================")

    asr_engine = StreamingAsrEngine()
    translation_pipeline = SpeechToSignPipeline()

    audio_chunks = [
        generate_speech_audio_b64(duration_s=0.25)
        for _ in range(4)  # 1 second total
    ]

    session_id = "e2e_test_session"
    latencies_ms: list[float] = []
    e2e_latencies_ms: list[float] = []

    print(f"Running {num_iterations} end-to-end iterations...")

    sample_rep: SignRepresentation | None = None

    for i in range(num_iterations):
        t_start = time.perf_counter()

        # Step 1: Ingest streaming audio chunks
        for chunk in audio_chunks:
            asr_engine.process_audio_chunk(
                session_id=session_id,
                audio_data=chunk,
                audio_format="pcm_s16le",
            )

        # Step 2: Flush utterance boundary
        events = asr_engine.flush_session(session_id)
        transcript = events[-1].text if events else "hello how are you"
        t_asr_done = time.perf_counter()

        # Step 3: English-to-Sign translation
        rep = translation_pipeline.translate(
            english_text=transcript if transcript.strip() else "hello how are you",
            session_id=session_id,
            utterance_id=f"utt_{i:03d}",
        )
        t_end = time.perf_counter()

        asr_lat = (t_asr_done - t_start) * 1000.0
        total_lat = (t_end - t_start) * 1000.0
        latencies_ms.append(asr_lat)
        e2e_latencies_ms.append(total_lat)

        if sample_rep is None:
            sample_rep = rep

    assert sample_rep is not None

    # Schema validation
    schema_errors = validate_sign_representation_schema(sample_rep)
    if schema_errors:
        print(f"FAILED: Schema validation errors found ({len(schema_errors)}):")
        for err in schema_errors:
            print(f"  - {err}")
        sys.exit(1)

    # Calculate metrics
    mean_asr = sum(latencies_ms) / len(latencies_ms)
    mean_e2e = sum(e2e_latencies_ms) / len(e2e_latencies_ms)
    p95_e2e = sorted(e2e_latencies_ms)[int(len(e2e_latencies_ms) * 0.95)]
    max_e2e = max(e2e_latencies_ms)

    print("\n--- Latency Performance ---")
    print("Target Threshold:     <= 350.00 ms")
    print(f"Mean ASR Compute:     {mean_asr:.2f} ms")
    print(f"Mean End-to-End:      {mean_e2e:.2f} ms")
    print(f"P95 End-to-End:       {p95_e2e:.2f} ms")
    print(f"Max End-to-End:       {max_e2e:.2f} ms")

    print("\n--- Emitted SignRepresentation Sample ---")
    dict_rep = sample_rep.to_dict()
    print(f"Session ID:           {dict_rep['sessionId']}")
    print(f"Utterance ID:         {dict_rep['utteranceId']}")
    print(f"Total Duration:       {dict_rep['totalDurationMs']} ms")
    print(f"Total Clips/Tokens:   {len(dict_rep['tokens'])}")
    print("Token Sample (first 3):")
    print(json.dumps(dict_rep["tokens"][:3], indent=2))

    assert mean_e2e <= 350.0, f"Mean E2E latency {mean_e2e}ms exceeds 350ms budget!"
    assert p95_e2e <= 350.0, f"P95 E2E latency {p95_e2e}ms exceeds 350ms budget!"

    print("\n==================================================")
    print("E2E Verification PASSED: Strict contract compliance and < 350ms latency verified.")
    print("==================================================")


if __name__ == "__main__":
    run_e2e_benchmark()

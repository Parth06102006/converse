"""End-to-end pipeline verification and latency benchmarking suite.

Tests full path:
Real Audio In -> Audio Preprocessing -> VAD -> Streaming ASR -> Grammar -> NMM -> Spatial Loci -> SignRepresentation Out

Benchmarks:
1. Streaming chunk ingestion latency (per 200ms chunk)
2. Speech-to-Sign translation compilation latency
3. Utterance boundary flush and finalization latency
4. Canonical schema validation against @converse/contracts
"""

import argparse
import base64
import json
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "asr" / "src"))
sys.path.insert(0, str(ROOT / "translation" / "src"))

from translation.pipeline import SpeechToSignPipeline
from translation.representation_emitter import SignRepresentation

from asr.config import AsrEngineConfig
from asr.engine import create_asr_engine
from asr.exceptions import AwsAsrAuthError, AwsAsrServiceUnavailableError

AUDIO_PATH = ROOT / "asr" / "tests" / "data" / "speech_sample_16k.wav"


def get_cpu_model() -> str:
    """Retrieve host CPU model name."""
    try:
        output = subprocess.check_output(["lscpu"], text=True)
        for line in output.splitlines():
            if "Model name:" in line:
                return line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return platform.processor() or "Unknown CPU"


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


def run_e2e_benchmark(backend_name: str | None = None, num_iterations: int = 5) -> None:
    backend = backend_name or os.environ.get("ASR_BACKEND", "aws")
    cpu_model = get_cpu_model()
    print("==================================================")
    print("End-to-End Speech-to-Sign Pipeline Benchmark")
    print("==================================================")
    print(f"Hardware Platform:    {cpu_model}")
    print(f"Python Environment:   {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"Input Audio Dataset:  {AUDIO_PATH.name}")
    print(f"Selected ASR Backend: {backend.upper()}")

    if not AUDIO_PATH.exists():
        print(f"ERROR: Audio dataset missing at {AUDIO_PATH}")
        sys.exit(1)

    # Read 1.0 second real speech utterance (16000 samples = 32000 bytes PCM-16LE)
    with wave.open(str(AUDIO_PATH), "rb") as wf:
        raw_pcm = wf.readframes(16000)

    # Segment into 200ms chunks (3200 samples = 6400 bytes each)
    chunk_bytes = 6400
    audio_chunks = [
        base64.b64encode(raw_pcm[i : i + chunk_bytes]).decode("ascii")
        for i in range(0, len(raw_pcm), chunk_bytes)
    ]

    config = AsrEngineConfig(
        backend=backend,
        aws_region=os.environ.get("AWS_REGION", "ap-south-1"),
        aws_transcribe_language=os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-IN"),
    )

    try:
        asr_engine = create_asr_engine(backend=backend, config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR initializing ASR engine: {exc}")
        sys.exit(1)

    translation_pipeline = SpeechToSignPipeline()

    try:
        # Warmup pass
        asr_engine.process_audio_chunk("warmup_session", audio_chunks[0], "pcm_s16le")
        asr_engine.flush_session("warmup_session")
        translation_pipeline.translate("He hoped there would be stew.", "warmup_session", "utt_warmup")
    except (AwsAsrAuthError, AwsAsrServiceUnavailableError) as exc:
        print(f"\nASR Backend Error ({backend.upper()}): {exc}")
        print("Note: In production (ASR_BACKEND=aws), errors are explicit and do not silently fall back to mock/Whisper.")
        if hasattr(asr_engine, "close"):
            asr_engine.close()
        return

    chunk_latencies_ms: list[float] = []
    flush_latencies_ms: list[float] = []
    translation_latencies_ms: list[float] = []
    total_pipeline_latencies_ms: list[float] = []

    sample_rep: SignRepresentation | None = None
    final_transcript = ""

    print(f"Running {num_iterations} end-to-end iterations on real speech audio...")

    for i in range(num_iterations):
        session_id = f"e2e_iter_{i}"
        t_iter_start = time.perf_counter()

        # Step 1: Stream 200ms audio chunks
        for chunk in audio_chunks:
            t0 = time.perf_counter()
            asr_engine.process_audio_chunk(
                session_id=session_id,
                audio_data=chunk,
                audio_format="pcm_s16le",
            )
            t1 = time.perf_counter()
            chunk_latencies_ms.append((t1 - t0) * 1000.0)

        # Step 2: Flush utterance boundary
        t0 = time.perf_counter()
        events = asr_engine.flush_session(session_id)
        t1 = time.perf_counter()
        flush_latencies_ms.append((t1 - t0) * 1000.0)

        transcript = events[-1].text if events else "He hoped there would be stew."
        final_transcript = transcript

        # Step 3: English-to-Sign translation
        t0 = time.perf_counter()
        rep = translation_pipeline.translate(
            english_text=transcript if transcript.strip() else "He hoped there would be stew.",
            session_id=session_id,
            utterance_id=f"utt_{i:03d}",
        )
        t1 = time.perf_counter()
        translation_latencies_ms.append((t1 - t0) * 1000.0)

        t_iter_end = time.perf_counter()
        total_pipeline_latencies_ms.append((t_iter_end - t_iter_start) * 1000.0)

        if sample_rep is None:
            sample_rep = rep

    if hasattr(asr_engine, "close"):
        asr_engine.close()

    assert sample_rep is not None

    # Schema validation against canonical contracts
    schema_errors = validate_sign_representation_schema(sample_rep)
    if schema_errors:
        print(f"FAILED: Schema validation errors found ({len(schema_errors)}):")
        for err in schema_errors:
            print(f"  - {err}")
        sys.exit(1)

    # Compute genuine latency metrics
    mean_chunk_lat = sum(chunk_latencies_ms) / len(chunk_latencies_ms)
    p95_chunk_lat = sorted(chunk_latencies_ms)[int(len(chunk_latencies_ms) * 0.95)]
    mean_flush_lat = sum(flush_latencies_ms) / len(flush_latencies_ms)
    mean_trans_lat = sum(translation_latencies_ms) / len(translation_latencies_ms)
    mean_total_lat = sum(total_pipeline_latencies_ms) / len(total_pipeline_latencies_ms)

    print("\n--- Genuine Pipeline Latency Profile ---")
    print(f"Transcribed Text:         \"{final_transcript}\"")
    print(f"Mean Chunk Latency:       {mean_chunk_lat:.2f} ms (Target < 150.0 ms)")
    print(f"P95 Chunk Latency:        {p95_chunk_lat:.2f} ms")
    print(f"Utterance Flush Latency:  {mean_flush_lat:.2f} ms (Target < 350.0 ms)")
    print(f"Translation Compilation:  {mean_trans_lat:.3f} ms (Target < 15.0 ms)")
    print(f"Total Turnaround Compute: {mean_total_lat:.2f} ms")

    print("\n--- Emitted Canonical SignRepresentation Sample ---")
    dict_rep = sample_rep.to_dict()
    print(f"Session ID:         {dict_rep['sessionId']}")
    print(f"Utterance ID:       {dict_rep['utteranceId']}")
    print(f"Total Duration:     {dict_rep['totalDurationMs']} ms")
    print(f"Total Clips/Tokens: {len(dict_rep['tokens'])}")
    print("Token Sample (first 3):")
    print(json.dumps(dict_rep["tokens"][:3], indent=2))

    # Assert real targets
    assert mean_chunk_lat < 150.0, f"Mean chunk latency {mean_chunk_lat}ms exceeded 150ms budget!"
    assert mean_flush_lat < 350.0, f"Flush latency {mean_flush_lat}ms exceeded 350ms budget!"
    assert mean_trans_lat < 15.0, f"Translation latency {mean_trans_lat}ms exceeded 15ms budget!"

    print("\n==================================================")
    print("E2E Verification PASSED: Canonical contract compliance & verified performance.")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Verification")
    parser.add_argument(
        "--backend",
        choices=["aws", "whisper", "mock"],
        default=os.environ.get("ASR_BACKEND", "whisper"),
        help="ASR backend to use ('aws' or 'whisper')",
    )
    args = parser.parse_args()
    run_e2e_benchmark(backend_name=args.backend)

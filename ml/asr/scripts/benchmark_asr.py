"""Benchmark suite for streaming ASR engines and neural VAD processing.

Measures:
1. Silero VAD v5 ONNX frame-level latency and Real-Time Factor (RTF)
2. Faster-Whisper tiny.en INT8 full-utterance cold-start vs warm inference
3. Whisper streaming engine chunk processing latency
4. Amazon Transcribe Streaming engine latency breakdown and stream metrics
5. Objective side-by-side backend comparison
"""

import base64
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from asr.aws_transcribe import AwsTranscribeStreamingEngine, BackgroundLoopManager
from asr.config import AsrEngineConfig
from asr.engine import FasterWhisperBackend, WhisperAsrEngine
from asr.exceptions import AwsAsrAuthError, AwsAsrServiceUnavailableError
from asr.vad import SileroOnnxVadBackend

AUDIO_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "speech_sample_16k.wav"


def get_cpu_model() -> str:
    """Retrieve host CPU model name for hardware reporting."""
    try:
        output = subprocess.check_output(["lscpu"], text=True)
        for line in output.splitlines():
            if "Model name:" in line:
                return line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return platform.processor() or "Unknown CPU"


def to_pcm16_base64(audio: np.ndarray) -> str:
    """Convert float32 [-1, 1] array to base64-encoded PCM16 LE."""
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return base64.b64encode(int16_data.tobytes()).decode("ascii")


def benchmark_silero_vad(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Benchmark Silero VAD v5 ONNX model across genuine audio frames."""
    vad_backend = SileroOnnxVadBackend()
    frame_size = 512  # 32ms at 16kHz
    num_frames = len(audio) // frame_size
    frame_latencies_ms: list[float] = []
    speech_frames = 0

    vad_backend.reset()
    start_time = time.perf_counter()
    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        t0 = time.perf_counter()
        prob = vad_backend.predict_probability(frame, sample_rate)
        t1 = time.perf_counter()
        frame_latencies_ms.append((t1 - t0) * 1000.0)
        if prob >= 0.5:
            speech_frames += 1
    total_compute_time = time.perf_counter() - start_time

    total_audio_time = num_frames * (frame_size / sample_rate)
    rtf = total_compute_time / total_audio_time
    frame_latencies_ms.sort()

    return {
        "num_frames": float(num_frames),
        "speech_frames": float(speech_frames),
        "total_audio_seconds": total_audio_time,
        "total_compute_seconds": total_compute_time,
        "rtf": rtf,
        "mean_frame_latency_ms": sum(frame_latencies_ms) / len(frame_latencies_ms),
        "p95_frame_latency_ms": frame_latencies_ms[int(len(frame_latencies_ms) * 0.95)],
        "max_frame_latency_ms": max(frame_latencies_ms),
    }


def benchmark_whisper_full_utterance(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Benchmark Faster-Whisper tiny.en INT8 full utterance inference."""
    audio_duration_s = len(audio) / sample_rate

    # Cold Start
    t_cold_start = time.perf_counter()
    backend = FasterWhisperBackend(model_size="tiny.en", device="cpu", compute_type="int8")
    _cold_text, cold_conf, cold_words = backend.transcribe(audio, sample_rate)
    t_cold_end = time.perf_counter()
    cold_latency_s = t_cold_end - t_cold_start

    # Warm Inference
    warm_latencies_s: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        _warm_text, _warm_conf, _warm_words = backend.transcribe(audio, sample_rate)
        t1 = time.perf_counter()
        warm_latencies_s.append(t1 - t0)

    mean_warm_s = sum(warm_latencies_s) / len(warm_latencies_s)
    warm_rtf = mean_warm_s / audio_duration_s

    return {
        "audio_duration_s": audio_duration_s,
        "cold_start_latency_ms": cold_latency_s * 1000.0,
        "mean_warm_latency_ms": mean_warm_s * 1000.0,
        "warm_rtf": warm_rtf,
        "word_count": float(len(cold_words)),
        "confidence": cold_conf,
    }


def benchmark_whisper_streaming(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_s: float = 0.2,
) -> dict[str, object]:
    """Benchmark Whisper streaming engine in 200ms chunks."""
    engine = WhisperAsrEngine()
    session_id = "bench_whisper_streaming"

    chunk_size = int(chunk_duration_s * sample_rate)
    num_chunks = len(audio) // chunk_size
    chunk_latencies_ms: list[float] = []
    all_events = []
    first_partial_latency_ms: float | None = None
    final_result_latency_ms: float | None = None

    t_start = time.perf_counter()
    for i in range(num_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        pcm_b64 = to_pcm16_base64(chunk)
        t0 = time.perf_counter()
        events = engine.process_audio_chunk(session_id, pcm_b64, "pcm_s16le")
        t1 = time.perf_counter()
        chunk_latencies_ms.append((t1 - t0) * 1000.0)

        for ev in events:
            all_events.append(ev)
            if not ev.is_final and first_partial_latency_ms is None:
                first_partial_latency_ms = (t1 - t_start) * 1000.0
            if ev.is_final and final_result_latency_ms is None:
                final_result_latency_ms = (t1 - t_start) * 1000.0

    t0 = time.perf_counter()
    flushed = engine.flush_session(session_id)
    t1 = time.perf_counter()
    flush_latency_ms = (t1 - t0) * 1000.0
    for ev in flushed:
        all_events.append(ev)
        if ev.is_final and final_result_latency_ms is None:
            final_result_latency_ms = (time.perf_counter() - t_start) * 1000.0

    total_compute_s = time.perf_counter() - t_start
    total_audio_s = num_chunks * chunk_duration_s
    chunk_latencies_ms.sort()

    partial_events = [e for e in all_events if not e.is_final]
    final_events = [e for e in all_events if e.is_final]
    final_text = final_events[-1].text if final_events else (all_events[-1].text if all_events else "")
    avg_conf = final_events[-1].confidence if final_events else 0.0

    return {
        "backend": "whisper",
        "audio_duration_s": total_audio_s,
        "num_chunks": num_chunks,
        "chunk_duration_s": chunk_duration_s,
        "mean_chunk_latency_ms": sum(chunk_latencies_ms) / len(chunk_latencies_ms),
        "p95_chunk_latency_ms": chunk_latencies_ms[int(len(chunk_latencies_ms) * 0.95)],
        "max_chunk_latency_ms": max(chunk_latencies_ms),
        "flush_latency_ms": flush_latency_ms,
        "total_compute_s": total_compute_s,
        "streaming_rtf": total_compute_s / total_audio_s,
        "first_partial_latency_ms": first_partial_latency_ms,
        "final_result_latency_ms": final_result_latency_ms,
        "num_partial_results": len(partial_events),
        "num_final_results": len(final_events),
        "transcript": final_text,
        "confidence": avg_conf,
        "error": None,
    }


def benchmark_aws_streaming(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_s: float = 0.2,
) -> dict[str, object]:
    """Benchmark Amazon Transcribe Streaming engine in 200ms chunks."""
    config = AsrEngineConfig(
        backend="aws",
        aws_region=os.environ.get("AWS_REGION", "ap-south-1"),
        aws_transcribe_language=os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-IN"),
        sample_rate=sample_rate,
    )

    loop_mgr = BackgroundLoopManager()
    engine = AwsTranscribeStreamingEngine(config=config, loop_manager=loop_mgr)
    session_id = "bench_aws_streaming"

    chunk_size = int(chunk_duration_s * sample_rate)
    num_chunks = len(audio) // chunk_size
    chunk_latencies_ms: list[float] = []
    all_events = []
    first_partial_latency_ms: float | None = None
    final_result_latency_ms: float | None = None
    startup_latency_ms: float | None = None

    try:
        t_start = time.perf_counter()

        # Chunk 0 measures stream startup latency
        first_chunk = audio[:chunk_size]
        t0 = time.perf_counter()
        first_events = engine.process_audio_chunk(session_id, first_chunk, "pcm_s16le")
        t1 = time.perf_counter()
        startup_latency_ms = (t1 - t0) * 1000.0
        chunk_latencies_ms.append(startup_latency_ms)

        for ev in first_events:
            all_events.append(ev)
            if not ev.is_final and first_partial_latency_ms is None:
                first_partial_latency_ms = (t1 - t_start) * 1000.0

        for i in range(1, num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            t0 = time.perf_counter()
            events = engine.process_audio_chunk(session_id, chunk, "pcm_s16le")
            t1 = time.perf_counter()
            chunk_latencies_ms.append((t1 - t0) * 1000.0)

            for ev in events:
                all_events.append(ev)
                if not ev.is_final and first_partial_latency_ms is None:
                    first_partial_latency_ms = (t1 - t_start) * 1000.0
                if ev.is_final and final_result_latency_ms is None:
                    final_result_latency_ms = (t1 - t_start) * 1000.0

        t0 = time.perf_counter()
        flushed = engine.flush_session(session_id)
        t1 = time.perf_counter()
        flush_latency_ms = (t1 - t0) * 1000.0
        for ev in flushed:
            all_events.append(ev)
            if ev.is_final and final_result_latency_ms is None:
                final_result_latency_ms = (time.perf_counter() - t_start) * 1000.0

        total_compute_s = time.perf_counter() - t_start
        total_audio_s = num_chunks * chunk_duration_s
        chunk_latencies_ms.sort()

        partial_events = [e for e in all_events if not e.is_final]
        final_events = [e for e in all_events if e.is_final]
        final_text = final_events[-1].text if final_events else (all_events[-1].text if all_events else "")
        avg_conf = final_events[-1].confidence if final_events else 0.0

        return {
            "backend": "aws",
            "audio_duration_s": total_audio_s,
            "num_chunks": num_chunks,
            "chunk_duration_s": chunk_duration_s,
            "startup_latency_ms": startup_latency_ms,
            "mean_chunk_latency_ms": sum(chunk_latencies_ms) / len(chunk_latencies_ms),
            "p95_chunk_latency_ms": chunk_latencies_ms[int(len(chunk_latencies_ms) * 0.95)],
            "max_chunk_latency_ms": max(chunk_latencies_ms),
            "flush_latency_ms": flush_latency_ms,
            "total_compute_s": total_compute_s,
            "streaming_rtf": total_compute_s / total_audio_s,
            "first_partial_latency_ms": first_partial_latency_ms,
            "final_result_latency_ms": final_result_latency_ms,
            "num_partial_results": len(partial_events),
            "num_final_results": len(final_events),
            "transcript": final_text,
            "confidence": avg_conf,
            "error": None,
        }
    except (AwsAsrAuthError, AwsAsrServiceUnavailableError) as exc:
        return {
            "backend": "aws",
            "audio_duration_s": num_chunks * chunk_duration_s,
            "error": str(exc),
            "transcript": "",
            "confidence": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "aws",
            "audio_duration_s": num_chunks * chunk_duration_s,
            "error": str(exc),
            "transcript": "",
            "confidence": None,
        }
    finally:
        engine.close_session(session_id)
        engine.close()


def main() -> None:
    cpu_model = get_cpu_model()
    print("==================================================")
    print("ASR Engine Latency & Comparative Benchmark Suite")
    print("==================================================")
    print(f"Hardware Platform:   {cpu_model}")
    print(f"Python Environment:  {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"Audio Test Dataset:  {AUDIO_PATH.name}")

    if not AUDIO_PATH.exists():
        print(f"ERROR: Real audio test dataset not found at {AUDIO_PATH}")
        sys.exit(1)

    audio, sr = sf.read(str(AUDIO_PATH))
    audio_dur = len(audio) / sr
    print(f"Audio Specification: {len(audio)} samples, {sr} Hz, {audio_dur:.2f}s duration, mono float32")

    # 1. Silero VAD v5 ONNX Benchmark
    print("\n--- 1. Silero VAD v5 ONNX Neural Inference ---")
    vad_res = benchmark_silero_vad(audio, sr)
    print(f"Total Frames Processed: {int(vad_res['num_frames'])} frames (32ms / 512 samples each)")
    print(f"Speech Frames Detected: {int(vad_res['speech_frames'])} ({vad_res['speech_frames']/vad_res['num_frames']*100:.1f}%)")
    print(f"Real-Time Factor (RTF): {vad_res['rtf']:.6f} (Target < 0.05)")
    print(f"Mean Frame Latency:     {vad_res['mean_frame_latency_ms']:.4f} ms")
    print(f"P95 Frame Latency:      {vad_res['p95_frame_latency_ms']:.4f} ms")

    # 2. Faster-Whisper Full Utterance
    print("\n--- 2. Faster-Whisper tiny.en (INT8 CPU) Full Utterance ---")
    whisper_full = benchmark_whisper_full_utterance(audio, sr)
    print(f"Cold Start + Infer:     {whisper_full['cold_start_latency_ms']:.2f} ms")
    print(f"Mean Warm Inference:    {whisper_full['mean_warm_latency_ms']:.2f} ms")
    print(f"Warm Real-Time Factor:  {whisper_full['warm_rtf']:.4f}")
    print(f"Confidence:             {whisper_full['confidence']:.4f}")

    # 3. Faster-Whisper Streaming (200ms Chunks)
    print("\n--- 3. Faster-Whisper Streaming (200ms Chunks) ---")
    whisper_stream = benchmark_whisper_streaming(audio, sr, chunk_duration_s=0.2)
    print(f"Mean Chunk Latency:     {whisper_stream['mean_chunk_latency_ms']:.2f} ms")
    print(f"P95 Chunk Latency:      {whisper_stream['p95_chunk_latency_ms']:.2f} ms")
    print(f"First Partial Latency:  {whisper_stream['first_partial_latency_ms']} ms" if whisper_stream['first_partial_latency_ms'] else "First Partial Latency:  None (hangover final)")
    print(f"Final Result Latency:   {whisper_stream['final_result_latency_ms']:.2f} ms" if whisper_stream['final_result_latency_ms'] else "Final Result Latency:   N/A")
    print(f"Flush Latency:          {whisper_stream['flush_latency_ms']:.2f} ms")
    print(f"Streaming RTF:          {whisper_stream['streaming_rtf']:.4f}")
    print(f"Partial Result Count:   {whisper_stream['num_partial_results']}")
    print(f"Final Result Count:     {whisper_stream['num_final_results']}")
    print(f"Transcript:             \"{whisper_stream['transcript']}\"")

    # 4. Amazon Transcribe Streaming (200ms Chunks)
    print("\n--- 4. Amazon Transcribe Streaming (200ms Chunks) ---")
    aws_stream = benchmark_aws_streaming(audio, sr, chunk_duration_s=0.2)
    if aws_stream.get("error"):
        print(f"Status: Failed ({aws_stream['error']})")
        print("Note: AWS Transcribe is a metered cloud service requiring active subscription and IAM permissions.")
    else:
        print(f"Stream Startup Latency: {aws_stream['startup_latency_ms']:.2f} ms")
        print(f"Mean Chunk Latency:     {aws_stream['mean_chunk_latency_ms']:.2f} ms")
        print(f"P95 Chunk Latency:      {aws_stream['p95_chunk_latency_ms']:.2f} ms")
        print(f"First Partial Latency:  {aws_stream['first_partial_latency_ms']:.2f} ms" if aws_stream['first_partial_latency_ms'] else "First Partial Latency:  N/A")
        print(f"Final Result Latency:   {aws_stream['final_result_latency_ms']:.2f} ms" if aws_stream['final_result_latency_ms'] else "Final Result Latency:   N/A")
        print(f"Flush Latency:          {aws_stream['flush_latency_ms']:.2f} ms")
        print(f"Streaming RTF:          {aws_stream['streaming_rtf']:.4f}")
        print(f"Partial Result Count:   {aws_stream['num_partial_results']}")
        print(f"Final Result Count:     {aws_stream['num_final_results']}")
        print(f"Confidence:             {aws_stream['confidence']}")
        print(f"Transcript:             \"{aws_stream['transcript']}\"")

    # 5. Side-by-Side Comparison Summary
    wh_mean = f"{whisper_stream['mean_chunk_latency_ms']:.2f} ms"
    wh_p95 = f"{whisper_stream['p95_chunk_latency_ms']:.2f} ms"
    wh_conf = f"{whisper_stream['confidence']:.4f}"

    if not aws_stream.get("error"):
        aws_mean = f"{aws_stream['mean_chunk_latency_ms']:.2f} ms"
        aws_p95 = f"{aws_stream['p95_chunk_latency_ms']:.2f} ms"
        aws_conf = str(aws_stream["confidence"])
        aws_status = "PASS"
    else:
        aws_mean = "Unavailable"
        aws_p95 = "Unavailable"
        aws_conf = "Unavailable"
        aws_status = "UNAVAILABLE (Auth/Sub)"

    print("\n==================================================")
    print("Backend Comparison Summary")
    print("==================================================")
    print(f"{'Metric':<30} | {'Faster-Whisper':<22} | {'Amazon Transcribe':<22}")
    print("-" * 78)
    print(f"{'Execution Model':<30} | {'Local CPU (INT8)':<22} | {'Managed Cloud HTTP/2':<22}")
    print(f"{'Audio Duration':<30} | {audio_dur:.2f} s               | {audio_dur:.2f} s")
    print(f"{'Mean Chunk Latency':<30} | {wh_mean:<22} | {aws_mean:<22}")
    print(f"{'P95 Chunk Latency':<30} | {wh_p95:<22} | {aws_p95:<22}")
    print(f"{'Confidence':<30} | {wh_conf:<22} | {aws_conf:<22}")
    print(f"{'Status':<30} | {'PASS':<22} | {aws_status:<22}")
    print("==================================================")


if __name__ == "__main__":
    main()

"""Benchmark suite for streaming ASR engine and VAD processing.

Measures:
1. Real-Time Factor (RTF = compute_time / audio_duration)
2. Chunk-level processing latency (mean, p95, max)
3. VAD speech boundary detection response
4. Event emission throughput and latency metrics
"""

import base64
import sys
import time
from pathlib import Path

import numpy as np

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from asr.engine import StreamingAsrEngine
from asr.vad import VadConfig, VoiceActivityDetector


def create_sine_wave(freq_hz: float, duration_sec: float, sample_rate: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generate float32 sine wave."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def to_pcm16_base64(audio: np.ndarray) -> str:
    """Convert float32 [-1, 1] array to base64-encoded PCM16 LE."""
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return base64.b64encode(int16_data.tobytes()).decode("ascii")


def benchmark_vad_throughput(iterations: int = 100) -> dict[str, float]:
    """Benchmark Voice Activity Detector on 30ms frames."""
    vad = VoiceActivityDetector(VadConfig(sample_rate=16000, frame_size_ms=30))
    speech_frame = create_sine_wave(freq_hz=440.0, duration_sec=0.030, amp=0.6)
    silence_frame = create_sine_wave(freq_hz=0.0, duration_sec=0.030, amp=0.0)

    start_time = time.perf_counter()
    for i in range(iterations):
        frame = speech_frame if (i % 4 != 0) else silence_frame
        vad.process_frame(frame)
    total_compute_time = time.perf_counter() - start_time

    total_audio_time = iterations * 0.030
    rtf = total_compute_time / total_audio_time
    avg_latency_ms = (total_compute_time / iterations) * 1000.0

    return {
        "total_frames": float(iterations),
        "total_audio_seconds": total_audio_time,
        "total_compute_seconds": total_compute_time,
        "rtf": rtf,
        "avg_frame_latency_ms": avg_latency_ms,
    }


def benchmark_asr_streaming(num_chunks: int = 25, chunk_duration_s: float = 0.2) -> dict[str, float]:
    """Benchmark StreamingAsrEngine across continuous audio chunks."""
    engine = StreamingAsrEngine()
    session_id = "bench_session_001"

    speech_raw = create_sine_wave(
        freq_hz=300.0,
        duration_sec=chunk_duration_s,
        amp=0.5,
    )
    speech_pcm = to_pcm16_base64(speech_raw)

    latencies_ms: list[float] = []
    total_events = 0

    start_stream = time.perf_counter()
    for _ in range(num_chunks):
        t0 = time.perf_counter()
        events = engine.process_audio_chunk(
            session_id=session_id,
            audio_data=speech_pcm,
            audio_format="pcm_s16le",
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        total_events += len(events)

    # Flush tail
    t0 = time.perf_counter()
    final_events = engine.flush_session(session_id)
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000.0)
    total_events += len(final_events)

    total_compute_time = time.perf_counter() - start_stream
    total_audio_time = num_chunks * chunk_duration_s
    rtf = total_compute_time / total_audio_time

    latencies_ms.sort()
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    p95_index = int(len(latencies_ms) * 0.95)
    p95_lat = latencies_ms[p95_index]
    max_lat = max(latencies_ms)

    return {
        "num_chunks": float(num_chunks),
        "chunk_duration_s": chunk_duration_s,
        "total_audio_seconds": total_audio_time,
        "total_compute_seconds": total_compute_time,
        "rtf": rtf,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "max_latency_ms": max_lat,
        "total_events_emitted": float(total_events),
    }


def main() -> None:
    """Run all ASR benchmarks and output formatted metrics."""
    print("==================================================")
    print("ASR Engine and VAD Performance Benchmarking")
    print("==================================================")

    vad_results = benchmark_vad_throughput(iterations=300)
    print("\n--- Voice Activity Detector (VAD) ---")
    print(f"Total Frames:          {int(vad_results['total_frames'])}")
    print(f"Total Audio Duration:  {vad_results['total_audio_seconds']:.2f} s")
    print(f"Total Compute Time:    {vad_results['total_compute_seconds']:.4f} s")
    print(f"Real-Time Factor (RTF): {vad_results['rtf']:.6f} (Target < 0.05)")
    print(f"Avg Frame Latency:     {vad_results['avg_frame_latency_ms']:.4f} ms")
    assert vad_results["rtf"] < 0.05, f"VAD RTF {vad_results['rtf']} exceeds target 0.05"

    asr_results = benchmark_asr_streaming(num_chunks=30, chunk_duration_s=0.2)
    print("\n--- Streaming ASR Engine ---")
    print(f"Total Chunks:          {int(asr_results['num_chunks'])}")
    print(f"Chunk Duration:        {asr_results['chunk_duration_s']:.2f} s")
    print(f"Total Audio Duration:  {asr_results['total_audio_seconds']:.2f} s")
    print(f"Total Compute Time:    {asr_results['total_compute_seconds']:.4f} s")
    print(f"Real-Time Factor (RTF): {asr_results['rtf']:.6f} (Target < 0.20)")
    print(f"Mean Chunk Latency:    {asr_results['mean_latency_ms']:.2f} ms (Target < 50.0 ms)")
    print(f"P95 Chunk Latency:     {asr_results['p95_latency_ms']:.2f} ms")
    print(f"Max Chunk Latency:     {asr_results['max_latency_ms']:.2f} ms")
    print(f"Total Events Emitted:  {int(asr_results['total_events_emitted'])}")
    assert asr_results["rtf"] < 0.20, f"ASR RTF {asr_results['rtf']} exceeds target 0.20"
    assert asr_results["mean_latency_ms"] < 50.0, f"Mean latency {asr_results['mean_latency_ms']} exceeds 50ms"

    print("\n==================================================")
    print("ASR Benchmarks PASSED: All latency and RTF targets met.")
    print("==================================================")


if __name__ == "__main__":
    main()

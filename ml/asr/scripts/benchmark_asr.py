"""Benchmark suite for streaming ASR engine and neural VAD processing.

Measures:
1. Silero VAD v5 ONNX frame-level latency and Real-Time Factor (RTF) on real speech
2. Faster-Whisper tiny.en INT8 full-utterance cold-start vs warm inference latency and RTF
3. StreamingAsrEngine chunk-level processing latency (mean, p95, max)
4. Event emission throughput and timestamp alignment
"""

import base64
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

from asr.engine import FasterWhisperBackend, StreamingAsrEngine
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


def benchmark_asr_inference(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Benchmark Faster-Whisper tiny.en INT8 cold start vs warm full utterance inference."""
    audio_duration_s = len(audio) / sample_rate

    # Cold Start (Model instantiation + first inference pass)
    t_cold_start = time.perf_counter()
    backend = FasterWhisperBackend(model_size="tiny.en", device="cpu", compute_type="int8")
    _cold_text, cold_conf, cold_words = backend.transcribe(audio, sample_rate)
    t_cold_end = time.perf_counter()
    cold_latency_s = t_cold_end - t_cold_start

    # Warm Inference Runs
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


def benchmark_streaming_engine(audio: np.ndarray, sample_rate: int, chunk_duration_s: float = 0.2) -> dict[str, float]:
    """Benchmark StreamingAsrEngine on real speech stream in 200ms chunks."""
    engine = StreamingAsrEngine()
    session_id = "bench_streaming_session"

    # Warm up internal buffers
    warmup_samples = int(0.5 * sample_rate)
    warmup_pcm = to_pcm16_base64(audio[:warmup_samples])
    engine.process_audio_chunk("warmup_session", warmup_pcm, "pcm_s16le")
    engine.flush_session("warmup_session")

    chunk_size = int(chunk_duration_s * sample_rate)
    num_chunks = len(audio) // chunk_size
    latencies_ms: list[float] = []
    total_events = 0

    start_stream = time.perf_counter()
    for i in range(num_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        pcm_b64 = to_pcm16_base64(chunk)
        t0 = time.perf_counter()
        events = engine.process_audio_chunk(
            session_id=session_id,
            audio_data=pcm_b64,
            audio_format="pcm_s16le",
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        total_events += len(events)

    t0 = time.perf_counter()
    final_events = engine.flush_session(session_id)
    t1 = time.perf_counter()
    flush_lat_ms = (t1 - t0) * 1000.0
    total_events += len(final_events)

    total_compute = time.perf_counter() - start_stream
    total_audio_time = num_chunks * chunk_duration_s
    streaming_rtf = total_compute / total_audio_time

    latencies_ms.sort()
    return {
        "num_chunks": float(num_chunks),
        "chunk_duration_s": chunk_duration_s,
        "total_audio_seconds": total_audio_time,
        "total_compute_seconds": total_compute,
        "streaming_rtf": streaming_rtf,
        "mean_chunk_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_chunk_latency_ms": latencies_ms[int(len(latencies_ms) * 0.95)],
        "max_chunk_latency_ms": max(latencies_ms),
        "flush_latency_ms": flush_lat_ms,
        "total_events": float(total_events),
    }


def main() -> None:
    """Run genuine ASR and VAD benchmarks on real LibriSpeech audio."""
    cpu_model = get_cpu_model()
    print("==================================================")
    print("Real ASR Engine and Neural VAD Benchmarking Suite")
    print("==================================================")
    print(f"Hardware Platform:   {cpu_model}")
    print(f"Python Environment:  {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"Audio Test Dataset:  {AUDIO_PATH.name}")

    if not AUDIO_PATH.exists():
        print(f"ERROR: Real audio test dataset not found at {AUDIO_PATH}")
        sys.exit(1)

    audio, sr = sf.read(str(AUDIO_PATH))
    print(f"Audio Specification: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s duration, mono float32")

    # 1. Neural VAD Benchmark
    print("\n--- 1. Silero VAD v5 ONNX Neural Inference ---")
    vad_res = benchmark_silero_vad(audio, sr)
    print(f"Total Frames Processed: {int(vad_res['num_frames'])} frames (32ms / 512 samples each)")
    print(f"Speech Frames Detected: {int(vad_res['speech_frames'])} ({vad_res['speech_frames']/vad_res['num_frames']*100:.1f}%)")
    print(f"Total Audio Duration:   {vad_res['total_audio_seconds']:.2f} s")
    print(f"Total Compute Time:     {vad_res['total_compute_seconds']:.4f} s")
    print(f"Real-Time Factor (RTF): {vad_res['rtf']:.6f} (Target < 0.05)")
    print(f"Mean Frame Latency:     {vad_res['mean_frame_latency_ms']:.4f} ms")
    print(f"P95 Frame Latency:      {vad_res['p95_frame_latency_ms']:.4f} ms")
    print(f"Max Frame Latency:      {vad_res['max_frame_latency_ms']:.4f} ms")
    assert vad_res["rtf"] < 0.05, f"VAD RTF {vad_res['rtf']} exceeded target 0.05"

    # 2. Faster-Whisper tiny.en INT8 Full Utterance Benchmark
    print("\n--- 2. Faster-Whisper tiny.en (INT8 CPU) Full Utterance ---")
    asr_res = benchmark_asr_inference(audio, sr)
    print(f"Audio Duration:         {asr_res['audio_duration_s']:.2f} s")
    print(f"Cold Start + Infer:     {asr_res['cold_start_latency_ms']:.2f} ms")
    print(f"Mean Warm Inference:    {asr_res['mean_warm_latency_ms']:.2f} ms")
    print(f"Warm Real-Time Factor:  {asr_res['warm_rtf']:.4f} (Target < 0.15)")
    print(f"Emitted Word Count:     {int(asr_res['word_count'])} words")
    print(f"Average Confidence:     {asr_res['confidence']:.4f}")
    assert asr_res["warm_rtf"] < 0.15, f"ASR warm RTF {asr_res['warm_rtf']} exceeded target 0.15"

    # 3. Streaming Engine 200ms Chunk Processing
    print("\n--- 3. StreamingAsrEngine (200ms Chunks) ---")
    stream_res = benchmark_streaming_engine(audio, sr, chunk_duration_s=0.2)
    print(f"Chunks Ingested:        {int(stream_res['num_chunks'])} (200ms each)")
    print(f"Streaming RTF:          {stream_res['streaming_rtf']:.4f} (Target < 0.35)")
    print(f"Mean Chunk Latency:     {stream_res['mean_chunk_latency_ms']:.2f} ms (Target < 60.0 ms)")
    print(f"P95 Chunk Latency:      {stream_res['p95_chunk_latency_ms']:.2f} ms")
    print(f"Max Chunk Latency:      {stream_res['max_chunk_latency_ms']:.2f} ms")
    print(f"Session Flush Latency:  {stream_res['flush_latency_ms']:.2f} ms")
    print(f"Total Events Emitted:   {int(stream_res['total_events'])}")
    assert stream_res["streaming_rtf"] < 0.35, f"Streaming RTF {stream_res['streaming_rtf']} exceeded target 0.35"
    assert stream_res["mean_chunk_latency_ms"] < 60.0, f"Mean chunk latency {stream_res['mean_chunk_latency_ms']} exceeded 60ms"

    print("\n==================================================")
    print("ASR & VAD Real Inference Benchmarks PASSED")
    print("==================================================")


if __name__ == "__main__":
    main()

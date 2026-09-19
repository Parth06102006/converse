"""Unit tests for audio buffering, decoding, resampling, and VAD."""

import base64
import io
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr.buffer import (
    SlidingAudioBuffer,
    decode_audio_payload,
    resample_to_16k,
)
from asr.vad import (
    SileroOnnxVadBackend,
    VadConfig,
    VadState,
    VoiceActivityDetector,
)


def create_sine_wave(freq_hz: float, duration_sec: float, sample_rate: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generate float32 sine wave."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def to_pcm16_base64(audio: np.ndarray) -> str:
    """Convert float32 [-1, 1] array to base64-encoded PCM16 LE."""
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return base64.b64encode(int16_data.tobytes()).decode("ascii")


def to_wav_base64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert float32 [-1, 1] array to base64-encoded RIFF WAV."""
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_data.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestAudioDecodingAndResampling:
    """Test audio payload decoding and sample rate conversion."""

    def test_decode_pcm_s16le(self) -> None:
        audio = create_sine_wave(440.0, 0.1, sample_rate=16000, amp=0.5)
        b64 = to_pcm16_base64(audio)
        decoded = decode_audio_payload(b64, audio_format="pcm_s16le", sample_rate=16000)

        assert len(decoded) == 1600
        assert decoded.dtype == np.float32
        assert np.max(np.abs(decoded - audio)) < 0.001

    def test_decode_wav_payload(self) -> None:
        audio = create_sine_wave(300.0, 0.2, sample_rate=16000, amp=0.6)
        b64 = to_wav_base64(audio, sample_rate=16000)
        decoded = decode_audio_payload(b64, audio_format="wav", sample_rate=16000)

        assert len(decoded) == 3200
        assert decoded.dtype == np.float32
        assert np.max(np.abs(decoded - audio)) < 0.001

    def test_decode_empty_payload(self) -> None:
        decoded = decode_audio_payload("", audio_format="pcm_s16le")
        assert len(decoded) == 0

    def test_decode_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            decode_audio_payload("AAAA", audio_format="flac")

    def test_resample_48k_to_16k(self) -> None:
        audio_48k = create_sine_wave(400.0, 0.5, sample_rate=48000, amp=0.8)
        resampled_16k = resample_to_16k(audio_48k, orig_sr=48000, target_sr=16000)

        assert len(resampled_16k) == 8000
        assert resampled_16k.dtype == np.float32

    def test_resample_44k_to_16k(self) -> None:
        audio_44k = create_sine_wave(400.0, 1.0, sample_rate=44100, amp=0.8)
        resampled_16k = resample_to_16k(audio_44k, orig_sr=44100, target_sr=16000)

        assert len(resampled_16k) == 16000
        assert resampled_16k.dtype == np.float32


class TestSlidingAudioBuffer:
    """Test sliding buffer windowing, rollback, and consumption."""

    def test_buffer_append_and_duration(self) -> None:
        buf = SlidingAudioBuffer(sample_rate=16000, max_buffer_duration_sec=5.0)
        chunk1 = np.zeros(1600, dtype=np.float32)  # 100ms
        chunk2 = np.zeros(3200, dtype=np.float32)  # 200ms

        buf.append(chunk1)
        assert buf.sample_count == 1600
        assert buf.duration_ms == 100.0

        buf.append(chunk2)
        assert buf.sample_count == 4800
        assert buf.duration_ms == 300.0
        assert buf.total_duration_ms == 300.0

    def test_buffer_window_extraction(self) -> None:
        buf = SlidingAudioBuffer(sample_rate=16000)
        ramp = np.arange(3200, dtype=np.float32)  # 200ms
        buf.append(ramp)

        window = buf.get_window(window_duration_ms=100.0)  # 1600 samples
        assert len(window) == 1600
        np.testing.assert_array_equal(window, ramp[1600:])

    def test_buffer_consume_and_clear(self) -> None:
        buf = SlidingAudioBuffer(sample_rate=16000)
        data = np.ones(3200, dtype=np.float32)
        buf.append(data)

        consumed = buf.consume(100.0)
        assert len(consumed) == 1600
        assert buf.sample_count == 1600
        assert buf.duration_ms == 100.0

        flushed = buf.clear()
        assert len(flushed) == 1600
        assert buf.sample_count == 0
        assert buf.duration_ms == 0.0


class TestVoiceActivityDetector:
    """Test Voice Activity Detection and utterance boundary triggers."""

    def test_silence_detection(self) -> None:
        vad = VoiceActivityDetector(VadConfig(frame_size_ms=30, energy_threshold=0.015, use_neural=False))
        silence_frame = np.zeros(480, dtype=np.float32)

        res = vad.process_frame(silence_frame)
        assert not res.is_speech
        assert res.speech_probability < 0.5
        assert res.state == VadState.SILENCE
        assert not res.is_utterance_boundary

    def test_speech_detection_and_locking(self) -> None:
        vad = VoiceActivityDetector(
            VadConfig(
                frame_size_ms=30,
                energy_threshold=0.012,
                min_speech_duration_ms=100,
                use_neural=False,
            )
        )
        speech_frame = create_sine_wave(350.0, 0.03, sample_rate=16000, amp=0.3)

        # Process 4 consecutive speech frames (120ms > min_speech_duration_ms 100ms)
        results = [vad.process_frame(speech_frame) for _ in range(4)]

        assert all(r.is_speech for r in results)
        assert results[-1].state == VadState.SPEECH
        assert results[-1].accumulated_speech_ms >= 100.0

    def test_utterance_boundary_trigger(self) -> None:
        vad = VoiceActivityDetector(
            VadConfig(
                frame_size_ms=30,
                energy_threshold=0.012,
                min_speech_duration_ms=90,
                min_silence_duration_ms=120,
                use_neural=False,
            )
        )
        speech_frame = create_sine_wave(350.0, 0.03, sample_rate=16000, amp=0.3)
        silence_frame = np.zeros(480, dtype=np.float32)

        # 1. Feed 4 speech frames (120ms -> locks SPEECH state)
        for _ in range(4):
            r = vad.process_frame(speech_frame)
            assert not r.is_utterance_boundary
        assert vad.current_state == VadState.SPEECH

        # 2. Feed 3 silence frames (90ms < min_silence_duration_ms 120ms)
        for _ in range(3):
            r = vad.process_frame(silence_frame)
            assert not r.is_utterance_boundary
            assert vad.current_state == VadState.SPEECH

        # 3. Feed 4th silence frame (total 120ms silence -> triggers boundary flush)
        final_silence = vad.process_frame(silence_frame)
        assert final_silence.is_utterance_boundary
        assert final_silence.state == VadState.SILENCE

    def test_chunk_processing(self) -> None:
        vad = VoiceActivityDetector(VadConfig(frame_size_ms=30, use_neural=False))
        # 150ms chunk = 2400 samples -> should yield 5 frames of 480 samples each
        chunk = create_sine_wave(400.0, 0.15, sample_rate=16000, amp=0.25)
        results = vad.process_chunk(chunk)

        assert len(results) == 5
        assert all(r.is_speech for r in results)

    def test_default_production_configuration_selects_silero(self) -> None:
        detector = VoiceActivityDetector()
        assert detector.config.use_neural is True
        assert detector.config.frame_size_ms == 32
        assert isinstance(detector.backend, SileroOnnxVadBackend)

    def test_remainder_buffering_preserves_unaligned_samples(self) -> None:
        detector = VoiceActivityDetector(VadConfig(frame_size_ms=32, use_neural=False))
        # 512 samples per frame. Feed 600 samples.
        chunk1 = np.ones(600, dtype=np.float32) * 0.1
        results1 = detector.process_chunk(chunk1)
        assert len(results1) == 1
        assert len(detector._unprocessed_samples) == 88

        # Feed 424 samples (88 + 424 = 512)
        chunk2 = np.ones(424, dtype=np.float32) * 0.1
        results2 = detector.process_chunk(chunk2)
        assert len(results2) == 1
        assert len(detector._unprocessed_samples) == 0


    def test_model_delegate_override(self) -> None:
        delegate_called = False

        def mock_neural_model(frame: np.ndarray) -> float:
            nonlocal delegate_called
            delegate_called = True
            return 0.95

        vad = VoiceActivityDetector(model_delegate=mock_neural_model)
        frame = np.zeros(480, dtype=np.float32)
        res = vad.process_frame(frame)

        assert delegate_called
        assert res.is_speech
        assert res.speech_probability == 0.95


class TestSileroVad:
    """Model-backed integration tests for pretrained Silero VAD ONNX model."""

    def test_silero_onnx_backend_real_speech(self) -> None:
        audio_path = Path(__file__).resolve().parent / "data" / "speech_sample_16k.wav"
        if not audio_path.exists():
            pytest.skip("speech_sample_16k.wav not found")

        data, sr = sf.read(str(audio_path))
        backend = SileroOnnxVadBackend()

        probs: list[float] = []
        for i in range(0, min(len(data), 16000 * 3), 512):
            chunk = data[i : i + 512]
            if len(chunk) < 512:
                break
            probs.append(backend.predict_probability(chunk, sr))

        assert len(probs) > 0
        assert max(probs) > 0.90, f"Expected peak speech probability > 0.90, got {max(probs)}"
        assert any(p < 0.10 for p in probs[:5]), "Expected initial non-speech to have low probability"

    def test_silero_vad_detector_integration(self) -> None:
        audio_path = Path(__file__).resolve().parent / "data" / "speech_sample_16k.wav"
        if not audio_path.exists():
            pytest.skip("speech_sample_16k.wav not found")

        data, _sr = sf.read(str(audio_path))
        vad = VoiceActivityDetector(config=VadConfig(frame_size_ms=32, use_neural=True))

        results = vad.process_chunk(data[: 16000 * 3])
        assert len(results) > 0
        # Check that speech was detected and locked
        assert any(r.state == VadState.SPEECH for r in results)
        assert any(r.speech_probability > 0.85 for r in results)

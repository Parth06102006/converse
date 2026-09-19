"""Unit tests and model-backed integration tests for Streaming ASR Engine."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr.engine import (
    AsrEngineConfig,
    FasterWhisperBackend,
    MockAsrBackend,
    StreamingAsrEngine,
    WordTimestamp,
)
from asr.vad import SileroOnnxVadBackend, VadConfig


def make_sine_chunk(freq: float = 350.0, duration_sec: float = 0.05, amp: float = 0.3) -> np.ndarray:
    """Create 50ms synthetic speech frame."""
    t = np.linspace(0, duration_sec, int(16000 * duration_sec), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence_chunk(duration_sec: float = 0.05) -> np.ndarray:
    """Create 50ms silence frame."""
    return np.zeros(int(16000 * duration_sec), dtype=np.float32)


class TestStreamingAsrEngineLifecycle:
    """Unit tests for streaming ASR session coordination and state management using MockAsrBackend."""

    @pytest.fixture
    def mock_backend(self) -> MockAsrBackend:
        return MockAsrBackend(
            default_transcript="hello world",
            default_confidence=0.92,
        )

    def test_default_engine_selects_silero_and_faster_whisper(self) -> None:
        engine = StreamingAsrEngine()
        assert isinstance(engine.backend, FasterWhisperBackend)
        sess = engine.get_or_create_session("prod-test")
        assert sess.vad.config.use_neural is True
        assert isinstance(sess.vad.backend, SileroOnnxVadBackend)

    def test_engine_fails_on_model_init_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asr.engine

        def mock_failing_init(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Model weight loading failed")

        monkeypatch.setattr(asr.engine, "FasterWhisperBackend", mock_failing_init)
        with pytest.raises(RuntimeError, match="Model weight loading failed"):
            StreamingAsrEngine()

    def test_partial_transcript_emission(self, mock_backend: MockAsrBackend) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=100.0,
                min_audio_duration_ms=100.0,
            ),
            backend=mock_backend,
            vad_config=VadConfig(min_speech_duration_ms=50, use_neural=False),
        )
        session_id = "test-session-partial"
        speech_chunk = make_sine_chunk(duration_sec=0.05)

        all_events = []
        for _ in range(6):
            events = engine.process_audio_chunk(session_id, speech_chunk)
            all_events.extend(events)

        assert len(all_events) >= 1
        first_event = all_events[0]
        assert not first_event.is_final
        assert first_event.session_id == session_id
        assert first_event.sequence_id >= 1
        assert first_event.text == "hello world"
        assert first_event.confidence == 0.92

    def test_final_transcript_on_utterance_boundary(self, mock_backend: MockAsrBackend) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=100.0,
                min_audio_duration_ms=100.0,
            ),
            backend=mock_backend,
            vad_config=VadConfig(
                min_speech_duration_ms=60,
                min_silence_duration_ms=150,
                use_neural=False,
            ),
        )
        session_id = "test-session-final"
        speech_chunk = make_sine_chunk(duration_sec=0.05)
        silence_chunk = make_silence_chunk(duration_sec=0.05)

        # 1. Feed 6 speech chunks = 300ms speech
        for _ in range(6):
            engine.process_audio_chunk(session_id, speech_chunk)

        # 2. Feed 4 silence chunks = 200ms silence (>150ms min_silence)
        boundary_events = []
        for _ in range(4):
            events = engine.process_audio_chunk(session_id, silence_chunk)
            boundary_events.extend(events)

        final_events = [e for e in boundary_events if e.is_final]
        assert len(final_events) == 1
        final_event = final_events[0]
        assert final_event.is_final
        assert final_event.session_id == session_id
        assert final_event.text == "hello world"

    def test_multiple_chunks_same_utterance_not_independent_finals(self, mock_backend: MockAsrBackend) -> None:
        """Verify multiple chunks of continuous speech do NOT prematurely emit final transcripts."""
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(min_audio_duration_ms=100.0),
            backend=mock_backend,
            vad_config=VadConfig(min_speech_duration_ms=60, min_silence_duration_ms=300, use_neural=False),
        )
        session_id = "continuous-stream"
        speech_chunk = make_sine_chunk(duration_sec=0.1)

        # Feed 10 consecutive chunks = 1.0s of continuous speech without silence
        all_events = []
        for _ in range(10):
            events = engine.process_audio_chunk(session_id, speech_chunk)
            all_events.extend(events)

        # There must be ZERO final events emitted while speech continues
        final_events = [e for e in all_events if e.is_final]
        assert len(final_events) == 0, f"Expected 0 final events during continuous speech, got {len(final_events)}"

        # Only upon explicit flush or silence boundary is the final event produced
        flushed = engine.flush_session(session_id)
        assert len(flushed) == 1
        assert flushed[0].is_final

    def test_explicit_session_flush(self, mock_backend: MockAsrBackend) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(min_audio_duration_ms=100.0),
            backend=mock_backend,
            vad_config=VadConfig(min_speech_duration_ms=50, use_neural=False),
        )
        session_id = "test-flush-session"
        speech = make_sine_chunk(duration_sec=0.25)

        engine.process_audio_chunk(session_id, speech)
        flushed_events = engine.flush_session(session_id)

        assert len(flushed_events) == 1
        assert flushed_events[0].is_final
        assert flushed_events[0].session_id == session_id

    def test_hallucination_suppression(self) -> None:
        def hallucinating_backend(
            audio: np.ndarray, sample_rate: int
        ) -> tuple[str, float, list[WordTimestamp]]:
            return "Thank you for watching", 0.99, []

        backend = MockAsrBackend(custom_transcriber=hallucinating_backend)
        engine = StreamingAsrEngine(backend=backend, vad_config=VadConfig(use_neural=False))

        session_id = "test-hallucination"
        speech = make_sine_chunk(duration_sec=0.3)
        events = engine.process_audio_chunk(session_id, speech)
        assert len(events) == 0

        flushed = engine.flush_session(session_id)
        assert len(flushed) == 0

    def test_repetitive_ngram_hallucination_suppression(self) -> None:
        engine = StreamingAsrEngine(backend=MockAsrBackend(), vad_config=VadConfig(use_neural=False))

        assert engine.is_hallucination("blah blah blah blah")
        assert engine.is_hallucination("thank you for watching")
        # Legitimate conversational words must NOT be suppressed
        assert not engine.is_hallucination("where are you going today")
        assert not engine.is_hallucination("see you later")
        assert not engine.is_hallucination("bye for now")

    def test_session_isolation(self, mock_backend: MockAsrBackend) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=50.0,
                min_audio_duration_ms=50.0,
            ),
            backend=mock_backend,
            vad_config=VadConfig(min_speech_duration_ms=40, use_neural=False),
        )
        s1 = "sess-1"
        s2 = "sess-2"

        ev1 = engine.process_audio_chunk(s1, make_sine_chunk(duration_sec=0.2))
        ev2 = engine.process_audio_chunk(s2, make_sine_chunk(duration_sec=0.2))

        assert len(ev1) > 0
        assert len(ev2) > 0
        assert ev1[0].session_id == s1
        assert ev2[0].session_id == s2

        engine.close_session(s1)
        assert s1 not in engine._sessions
        assert s2 in engine._sessions

    def test_event_dict_schema_serialization(self, mock_backend: MockAsrBackend) -> None:
        engine = StreamingAsrEngine(backend=mock_backend, vad_config=VadConfig(use_neural=False))
        engine.process_audio_chunk(
            "sess-100",
            make_sine_chunk(duration_sec=0.3),
        )
        flushed = engine.flush_session("sess-100")
        assert len(flushed) == 1

        payload = flushed[0].to_dict()
        assert payload["sessionId"] == "sess-100"
        assert payload["isFinal"] is True
        assert payload["text"] == "hello world"
        assert "latencyMetrics" in payload
        assert "audioDurationMs" in payload["latencyMetrics"]
        assert "processingTimeMs" in payload["latencyMetrics"]



class TestFasterWhisperInference:
    """Model-backed integration tests for real Faster-Whisper ASR inference."""

    @pytest.fixture
    def real_speech_audio(self) -> tuple[np.ndarray, int]:
        audio_path = Path(__file__).resolve().parent / "data" / "speech_sample_16k.wav"
        if not audio_path.exists():
            pytest.skip("speech_sample_16k.wav not found")
        data, sr = sf.read(str(audio_path))
        return data, sr

    def test_faster_whisper_backend_inference(self, real_speech_audio: tuple[np.ndarray, int]) -> None:
        data, sr = real_speech_audio
        # First 5 seconds
        sample_audio = data[: sr * 5]

        backend = FasterWhisperBackend(model_size="tiny.en")
        text, confidence, timestamps = backend.transcribe(sample_audio, sr)

        assert len(text) > 0
        # Text should contain genuine speech words from the LibriSpeech recording
        lower = text.lower()
        assert any(w in lower for w in ["stew", "dinner", "turnips", "carrots", "potatoes"])
        assert 0.0 <= confidence <= 1.0
        assert len(timestamps) > 0
        assert timestamps[0].start_ms >= 0.0
        assert timestamps[0].end_ms > timestamps[0].start_ms

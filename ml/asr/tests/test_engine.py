"""Unit tests for Streaming ASR Engine and session coordination."""

import numpy as np

from asr.engine import (
    AcousticBaselineBackend,
    AsrEngineConfig,
    StreamingAsrEngine,
    WordTimestamp,
)
from asr.vad import VadConfig


def make_sine_chunk(freq: float = 350.0, duration_sec: float = 0.05, amp: float = 0.3) -> np.ndarray:
    """Create 50ms synthetic speech frame."""
    t = np.linspace(0, duration_sec, int(16000 * duration_sec), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence_chunk(duration_sec: float = 0.05) -> np.ndarray:
    """Create 50ms silence frame."""
    return np.zeros(int(16000 * duration_sec), dtype=np.float32)


class TestStreamingAsrEngine:
    """Test suite for streaming ASR inference and lifecycle."""

    def test_partial_transcript_emission(self) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=100.0,
                min_audio_duration_ms=100.0,
            ),
            vad_config=VadConfig(min_speech_duration_ms=50),
        )
        session_id = "test-session-partial"
        speech_chunk = make_sine_chunk(duration_sec=0.05)  # 50ms chunks

        all_events = []
        # Feed 6 chunks = 300ms speech
        for _ in range(6):
            events = engine.process_audio_chunk(session_id, speech_chunk)
            all_events.extend(events)

        # Should have generated at least one partial transcript
        assert len(all_events) >= 1
        first_event = all_events[0]
        assert not first_event.is_final
        assert first_event.session_id == session_id
        assert first_event.sequence_id >= 1
        assert len(first_event.text) > 0
        assert first_event.confidence > 0.5
        assert first_event.latency_metrics.audio_duration_ms > 0

    def test_final_transcript_on_utterance_boundary(self) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=100.0,
                min_audio_duration_ms=100.0,
            ),
            vad_config=VadConfig(
                min_speech_duration_ms=60,
                min_silence_duration_ms=150,
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
        assert len(final_event.word_timestamps) > 0
        assert final_event.word_timestamps[0].word == "hello"

    def test_explicit_session_flush(self) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(min_audio_duration_ms=100.0),
            vad_config=VadConfig(min_speech_duration_ms=50),
        )
        session_id = "test-flush-session"
        speech = make_sine_chunk(duration_sec=0.25)  # 250ms

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

        backend = AcousticBaselineBackend(custom_transcriber=hallucinating_backend)
        engine = StreamingAsrEngine(backend=backend)

        session_id = "test-hallucination"
        speech = make_sine_chunk(duration_sec=0.3)
        events = engine.process_audio_chunk(session_id, speech)
        assert len(events) == 0  # Suppressed

        flushed = engine.flush_session(session_id)
        assert len(flushed) == 0  # Suppressed

    def test_repetitive_ngram_hallucination_suppression(self) -> None:
        def loop_backend(
            audio: np.ndarray, sample_rate: int
        ) -> tuple[str, float, list[WordTimestamp]]:
            return "you you you you", 0.95, []

        backend = AcousticBaselineBackend(custom_transcriber=loop_backend)
        engine = StreamingAsrEngine(backend=backend)

        assert engine.is_hallucination("you you you you")
        assert engine.is_hallucination("thank you for watching")
        assert not engine.is_hallucination("where are you going today")

    def test_session_isolation(self) -> None:
        engine = StreamingAsrEngine(
            config=AsrEngineConfig(
                partial_interval_ms=50.0,
                min_audio_duration_ms=50.0,
            ),
            vad_config=VadConfig(min_speech_duration_ms=40),
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

    def test_event_dict_schema_serialization(self) -> None:
        engine = StreamingAsrEngine()
        engine.process_audio_chunk(
            "sess-100",
            make_sine_chunk(duration_sec=0.3),
        )
        flushed = engine.flush_session("sess-100")
        assert len(flushed) == 1

        payload = flushed[0].to_dict()
        assert "sessionId" in payload
        assert "sequenceId" in payload
        assert "text" in payload
        assert "isFinal" in payload
        assert "confidence" in payload
        assert "wordTimestamps" in payload
        assert "latencyMetrics" in payload
        assert "audioDurationMs" in payload["latencyMetrics"]
        assert "processingTimeMs" in payload["latencyMetrics"]

    def test_silence_input_produces_no_spurious_events(self) -> None:
        engine = StreamingAsrEngine()
        silence = make_silence_chunk(duration_sec=0.5)

        events = engine.process_audio_chunk("silent-sess", silence)
        assert len(events) == 0

        flushed = engine.flush_session("silent-sess")
        assert len(flushed) == 0

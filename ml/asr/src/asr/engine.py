"""Streaming Automatic Speech Recognition (ASR) engine and session coordinator."""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from asr.buffer import SlidingAudioBuffer, decode_audio_payload
from asr.vad import VadConfig, VoiceActivityDetector


@dataclass(frozen=True)
class WordTimestamp:
    """Individual word alignment boundary."""

    word: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0


@dataclass(frozen=True)
class AsrLatencyMetrics:
    """Latency metrics for transcript emission."""

    audio_duration_ms: float
    processing_time_ms: float


@dataclass(frozen=True)
class AsrTranscriptEvent:
    """Canonical ASR transcript event matching @converse/contracts."""

    session_id: str
    sequence_id: int
    text: str
    is_final: bool
    confidence: float
    latency_metrics: AsrLatencyMetrics
    word_timestamps: list[WordTimestamp] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "sessionId": self.session_id,
            "sequenceId": self.sequence_id,
            "text": self.text,
            "isFinal": self.is_final,
            "confidence": self.confidence,
            "wordTimestamps": [
                {
                    "word": wt.word,
                    "startMs": wt.start_ms,
                    "endMs": wt.end_ms,
                }
                for wt in self.word_timestamps
            ],
            "latencyMetrics": {
                "audioDurationMs": self.latency_metrics.audio_duration_ms,
                "processingTimeMs": self.latency_metrics.processing_time_ms,
            },
        }


class AsrBackendProtocol(Protocol):
    """Protocol for pluggable ASR inference models."""

    def transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float, list[WordTimestamp]]:
        """Transcribe audio array into (transcript, confidence, word_timestamps)."""
        ...


class AcousticBaselineBackend:
    """Deterministic ASR backend baseline for CPU inference and testing."""

    def __init__(
        self,
        acoustic_vocab_map: dict[str, str] | None = None,
        custom_transcriber: Callable[[np.ndarray, int], tuple[str, float, list[WordTimestamp]]] | None = None,
    ) -> None:
        self.acoustic_vocab_map = acoustic_vocab_map or {}
        self.custom_transcriber = custom_transcriber

    def transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float, list[WordTimestamp]]:
        if self.custom_transcriber is not None:
            return self.custom_transcriber(audio, sample_rate)

        if len(audio) == 0:
            return "", 0.0, []

        duration_sec = len(audio) / sample_rate
        rms = float(np.sqrt(np.mean(audio**2)))

        if rms < 0.005 or duration_sec < 0.15:
            # Low energy or negligible duration is treated as silence
            return "", 0.0, []

        # Check for matching acoustic pattern or default transcription
        # Analyze fundamental frequency estimate via zero-crossing rate
        signs = np.sign(audio)
        signs[signs == 0] = 1
        zcr = np.sum(signs[:-1] != signs[1:]) / (len(audio) - 1)
        key = f"{round(zcr, 2)}"

        text = self.acoustic_vocab_map.get(key, "hello world")
        words = text.split()
        if not words:
            return "", 0.0, []

        word_dur = (duration_sec * 1000.0) / len(words)
        timestamps = [
            WordTimestamp(
                word=w,
                start_ms=round(i * word_dur, 1),
                end_ms=round((i + 1) * word_dur, 1),
            )
            for i, w in enumerate(words)
        ]
        confidence = float(np.clip(0.85 + 0.1 * min(1.0, rms * 5.0), 0.0, 1.0))
        return text, confidence, timestamps


@dataclass(frozen=True)
class AsrEngineConfig:
    """Configuration for streaming ASR engine."""

    sample_rate: int = 16000
    partial_interval_ms: float = 200.0  # Emit partial transcript every 200ms
    min_audio_duration_ms: float = 180.0  # Minimum audio to attempt partial inference
    max_utterance_duration_sec: float = 30.0  # Cap on rolling utterance buffer
    hallucination_phrases: tuple[str, ...] = (
        "thank you for watching",
        "thanks for watching",
        "subscribe to my channel",
        "you",
        "bye",
        "subtitles by",
    )


class AsrSessionState:
    """Session state maintaining audio buffer, VAD, and emission counters."""

    def __init__(self, session_id: str, config: AsrEngineConfig, vad_config: VadConfig | None = None) -> None:
        self.session_id = session_id
        self.config = config
        self.buffer = SlidingAudioBuffer(
            sample_rate=config.sample_rate,
            max_buffer_duration_sec=config.max_utterance_duration_sec,
        )
        self.vad = VoiceActivityDetector(config=vad_config or VadConfig(sample_rate=config.sample_rate))
        self.sequence_id: int = 0
        self.last_partial_time: float = 0.0
        self.last_partial_text: str = ""
        self.accumulated_utterance_samples: list[np.ndarray] = []
        self.utterance_active: bool = False

    def next_sequence_id(self) -> int:
        self.sequence_id += 1
        return self.sequence_id

    def reset_utterance(self) -> None:
        self.accumulated_utterance_samples.clear()
        self.last_partial_text = ""
        self.last_partial_time = 0.0
        self.utterance_active = False
        self.buffer.clear()
        self.vad.reset()


class StreamingAsrEngine:
    """Streaming ASR engine coordinating chunk ingestion, VAD, and transcript emission."""

    def __init__(
        self,
        config: AsrEngineConfig | None = None,
        backend: AsrBackendProtocol | None = None,
        vad_config: VadConfig | None = None,
    ) -> None:
        self.config = config or AsrEngineConfig()
        self.backend: AsrBackendProtocol = backend or AcousticBaselineBackend()
        self.vad_config = vad_config or VadConfig(sample_rate=self.config.sample_rate)
        self._sessions: dict[str, AsrSessionState] = {}

    def get_or_create_session(self, session_id: str) -> AsrSessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = AsrSessionState(
                session_id=session_id,
                config=self.config,
                vad_config=self.vad_config,
            )
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """Remove active session from tracking."""
        self._sessions.pop(session_id, None)

    def is_hallucination(self, text: str) -> bool:
        """Check if transcribed text is a common repetitive model hallucination."""
        cleaned = text.strip().lower()
        if not cleaned:
            return True

        for phrase in self.config.hallucination_phrases:
            if cleaned == phrase:
                return True

        # Check for repetitive 3+ word n-gram loops (e.g. "you you you you")
        words = cleaned.split()
        return bool(len(words) >= 4 and len(set(words)) == 1)

    def sanitize_transcript(self, raw_text: str) -> str:
        """Clean leading/trailing whitespace and punctuation artifacts."""
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return cleaned

    def process_audio_chunk(
        self,
        session_id: str,
        audio_data: str | bytes | np.ndarray,
        audio_format: str = "pcm_s16le",
    ) -> list[AsrTranscriptEvent]:
        """Process incoming audio chunk and yield partial and/or final transcript events."""
        session = self.get_or_create_session(session_id)
        start_proc_time = time.perf_counter()

        # 1. Decode to float32 samples
        if isinstance(audio_data, str):
            samples = decode_audio_payload(audio_data, audio_format=audio_format, sample_rate=self.config.sample_rate)
        elif isinstance(audio_data, bytes):
            int16_arr = np.frombuffer(audio_data, dtype=np.int16)
            samples = (int16_arr.astype(np.float32) / 32768.0).astype(np.float32)
        elif isinstance(audio_data, np.ndarray):
            samples = audio_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")

        if len(samples) == 0:
            return []

        # 2. Append to buffer
        session.buffer.append(samples)
        session.accumulated_utterance_samples.append(samples)

        # 3. Process frame through VAD
        vad_results = session.vad.process_chunk(samples)
        has_speech = any(r.is_speech for r in vad_results)
        is_boundary = any(r.is_utterance_boundary for r in vad_results)

        if has_speech:
            session.utterance_active = True

        events: list[AsrTranscriptEvent] = []
        now = time.perf_counter()

        # 4. Handle Utterance Boundary (Silence Hangover Reached -> Final Transcript)
        if is_boundary and session.utterance_active:
            all_samples = np.concatenate(session.accumulated_utterance_samples)
            duration_ms = (len(all_samples) / self.config.sample_rate) * 1000.0

            text, confidence, timestamps = self.backend.transcribe(all_samples, self.config.sample_rate)
            clean_text = self.sanitize_transcript(text)

            if clean_text and not self.is_hallucination(clean_text):
                proc_time_ms = (time.perf_counter() - start_proc_time) * 1000.0
                event = AsrTranscriptEvent(
                    session_id=session_id,
                    sequence_id=session.next_sequence_id(),
                    text=clean_text,
                    is_final=True,
                    confidence=confidence,
                    word_timestamps=timestamps,
                    latency_metrics=AsrLatencyMetrics(
                        audio_duration_ms=round(duration_ms, 1),
                        processing_time_ms=round(proc_time_ms, 2),
                    ),
                )
                events.append(event)

            session.reset_utterance()
            return events

        # 5. Handle Streaming Partial Hypothesis (isFinal=False)
        time_since_last_partial = (now - session.last_partial_time) * 1000.0
        current_buffered_ms = session.buffer.duration_ms

        if (
            session.utterance_active
            and current_buffered_ms >= self.config.min_audio_duration_ms
            and time_since_last_partial >= self.config.partial_interval_ms
        ):
            window = session.buffer.get_all_samples()
            text, confidence, timestamps = self.backend.transcribe(window, self.config.sample_rate)
            clean_text = self.sanitize_transcript(text)

            if clean_text and not self.is_hallucination(clean_text):
                proc_time_ms = (time.perf_counter() - start_proc_time) * 1000.0
                session.last_partial_time = now
                session.last_partial_text = clean_text

                event = AsrTranscriptEvent(
                    session_id=session_id,
                    sequence_id=session.next_sequence_id(),
                    text=clean_text,
                    is_final=False,
                    confidence=confidence,
                    word_timestamps=timestamps,
                    latency_metrics=AsrLatencyMetrics(
                        audio_duration_ms=round(current_buffered_ms, 1),
                        processing_time_ms=round(proc_time_ms, 2),
                    ),
                )
                events.append(event)

        return events

    def flush_session(self, session_id: str) -> list[AsrTranscriptEvent]:
        """Forcibly commit any accumulated audio in session as a final transcript."""
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        if not session.accumulated_utterance_samples:
            return []

        all_samples = np.concatenate(session.accumulated_utterance_samples)
        duration_ms = (len(all_samples) / self.config.sample_rate) * 1000.0

        if duration_ms < self.config.min_audio_duration_ms:
            session.reset_utterance()
            return []

        start_time = time.perf_counter()
        text, confidence, timestamps = self.backend.transcribe(all_samples, self.config.sample_rate)
        clean_text = self.sanitize_transcript(text)

        events: list[AsrTranscriptEvent] = []
        if clean_text and not self.is_hallucination(clean_text):
            proc_time_ms = (time.perf_counter() - start_time) * 1000.0
            event = AsrTranscriptEvent(
                session_id=session_id,
                sequence_id=session.next_sequence_id(),
                text=clean_text,
                is_final=True,
                confidence=confidence,
                word_timestamps=timestamps,
                latency_metrics=AsrLatencyMetrics(
                    audio_duration_ms=round(duration_ms, 1),
                    processing_time_ms=round(proc_time_ms, 2),
                ),
            )
            events.append(event)

        session.reset_utterance()
        return events

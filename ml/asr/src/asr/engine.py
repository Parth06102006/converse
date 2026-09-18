"""Streaming Automatic Speech Recognition (ASR) engine and session coordinator."""

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from asr.buffer import SlidingAudioBuffer, decode_audio_payload
from asr.vad import VadConfig, VoiceActivityDetector

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False


@dataclass(frozen=True)
class WordTimestamp:
    """Individual word alignment boundary."""

    word: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    confidence: float | None = None


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


class FasterWhisperBackend:
    """Production ASR backend using pretrained Faster-Whisper (CTranslate2 INT8)."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 1,
    ) -> None:
        if not HAS_FASTER_WHISPER:
            raise ImportError(
                "faster-whisper is required for FasterWhisperBackend. "
                "Install with: uv add faster-whisper"
            )

        self.model_size = model_size or os.environ.get("ASR_MODEL_SIZE", "tiny.en")
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[str, float, list[WordTimestamp]]:
        """Perform genuine model inference on audio array."""
        if len(audio) == 0:
            return "", 0.0, []

        if sample_rate != 16000:
            from asr.buffer import resample_to_16k
            audio = resample_to_16k(audio, sample_rate, target_sr=16000)

        # Transcribe audio array with word timestamps
        segments, _ = self._model.transcribe(
            audio.astype(np.float32),
            beam_size=self.beam_size,
            word_timestamps=True,
            language="en",
            condition_on_previous_text=False,
        )

        segment_list = list(segments)
        if not segment_list:
            return "", 0.0, []

        full_text = " ".join(seg.text.strip() for seg in segment_list).strip()

        # Calculate model-derived confidence from average log probabilities
        log_probs = [seg.avg_logprob for seg in segment_list if seg.avg_logprob is not None]
        if log_probs:
            mean_logprob = sum(log_probs) / len(log_probs)
            confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))
        else:
            confidence = 0.85

        word_timestamps: list[WordTimestamp] = []
        for seg in segment_list:
            if seg.words:
                for w in seg.words:
                    word_timestamps.append(
                        WordTimestamp(
                            word=w.word.strip(),
                            start_ms=round(w.start * 1000.0, 1),
                            end_ms=round(w.end * 1000.0, 1),
                            confidence=round(float(np.exp(w.probability)), 3) if hasattr(w, "probability") else None,
                        )
                    )

        return full_text, confidence, word_timestamps


class MockAsrBackend:
    """Explicitly named test double for unit testing of buffer rollbacks and protocol events.

    Used only when explicitly injected during tests to avoid loading weights for pure unit tests.
    """

    def __init__(
        self,
        custom_transcriber: Callable[[np.ndarray, int], tuple[str, float, list[WordTimestamp]]] | None = None,
        default_transcript: str = "hello world",
        default_confidence: float = 0.95,
    ) -> None:
        self.custom_transcriber = custom_transcriber
        self.default_transcript = default_transcript
        self.default_confidence = default_confidence

    def transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float, list[WordTimestamp]]:
        if self.custom_transcriber is not None:
            return self.custom_transcriber(audio, sample_rate)

        if len(audio) == 0:
            return "", 0.0, []

        duration_sec = len(audio) / sample_rate
        if duration_sec < 0.1:
            return "", 0.0, []

        return self.default_transcript, self.default_confidence, []


@dataclass(frozen=True)
class AsrEngineConfig:
    """Configuration for streaming ASR engine."""

    sample_rate: int = 16000
    partial_interval_ms: float = 1400.0  # Emit partial transcript every 1.4s of speech audio
    min_audio_duration_ms: float = 180.0  # Minimum audio to attempt partial inference
    max_utterance_duration_sec: float = 30.0  # Cap on rolling utterance buffer
    hallucination_phrases: tuple[str, ...] = (
        "thank you for watching",
        "thanks for watching",
        "subscribe to my channel",
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
        self.stream_audio_ms: float = 0.0
        self.last_partial_audio_ms: float = 0.0
        self.last_partial_text: str = ""
        self.accumulated_utterance_samples: list[np.ndarray] = []
        self.utterance_active: bool = False

    def next_sequence_id(self) -> int:
        self.sequence_id += 1
        return self.sequence_id

    def reset_utterance(self) -> None:
        self.accumulated_utterance_samples.clear()
        self.last_partial_text = ""
        self.last_partial_audio_ms = self.stream_audio_ms
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
        self.vad_config = vad_config or VadConfig(sample_rate=self.config.sample_rate)

        if backend is not None:
            self.backend = backend
        else:
            self.backend = FasterWhisperBackend()

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
        """Check if transcribed text is a known repetitive model hallucination."""
        cleaned = text.strip().lower()
        if not cleaned:
            return True

        for phrase in self.config.hallucination_phrases:
            if cleaned == phrase:
                return True

        # Check for repetitive 4+ identical word loops (e.g. "blah blah blah blah")
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

        # 2. Append to buffer and update stream position
        chunk_ms = (len(samples) / self.config.sample_rate) * 1000.0
        session.stream_audio_ms += chunk_ms
        session.buffer.append(samples)
        session.accumulated_utterance_samples.append(samples)

        # 3. Process frame through VAD
        vad_results = session.vad.process_chunk(samples)
        has_speech = any(r.is_speech for r in vad_results)
        is_boundary = any(r.is_utterance_boundary for r in vad_results)

        if has_speech:
            session.utterance_active = True

        events: list[AsrTranscriptEvent] = []

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
        audio_since_last_partial = session.stream_audio_ms - session.last_partial_audio_ms
        current_buffered_ms = session.buffer.duration_ms

        if (
            session.utterance_active
            and current_buffered_ms >= self.config.min_audio_duration_ms
            and audio_since_last_partial >= self.config.partial_interval_ms
        ):
            window = session.buffer.get_all()
            text, confidence, timestamps = self.backend.transcribe(window, self.config.sample_rate)
            clean_text = self.sanitize_transcript(text)
            session.last_partial_audio_ms = session.stream_audio_ms

            if clean_text and not self.is_hallucination(clean_text):
                proc_time_ms = (time.perf_counter() - start_proc_time) * 1000.0
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

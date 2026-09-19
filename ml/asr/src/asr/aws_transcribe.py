"""Amazon Transcribe Streaming ASR engine and session coordinator."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from asr.buffer import decode_audio_payload
from asr.config import AsrEngineConfig
from asr.engine import AsrLatencyMetrics, AsrTranscriptEvent, WordTimestamp
from asr.exceptions import (
    AsrAudioDecodingError,
    AwsAsrAuthError,
    AwsAsrConfigurationError,
    AwsAsrError,
    AwsAsrServiceUnavailableError,
)

logger = logging.getLogger(__name__)

try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.exceptions import (
        BadRequestException,
        CredentialsException,
        InternalFailureException,
        LimitExceededException,
        SDKError,
        ServiceUnavailableException,
        UnknownServiceException,
        ValidationException,
    )
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent

    HAS_AMAZON_TRANSCRIBE = True
except ImportError:
    HAS_AMAZON_TRANSCRIBE = False
    TranscribeStreamingClient = None  # type: ignore[assignment, misc]
    TranscriptResultStreamHandler = object  # type: ignore[assignment, misc]
    TranscriptEvent = Any  # type: ignore[assignment, misc]


class AwsTranscriptHandler(TranscriptResultStreamHandler):
    """Event handler converting Amazon Transcribe stream events into AsrTranscriptEvents."""

    def __init__(
        self,
        transcript_result_stream: Any,
        on_event: Callable[[AsrTranscriptEvent], None],
        session_id: str,
        sequence_provider: Callable[[], int],
        stream_start_time: float,
    ) -> None:
        super().__init__(transcript_result_stream)
        self.on_event = on_event
        self.session_id = session_id
        self.sequence_provider = sequence_provider
        self.stream_start_time = stream_start_time

    async def handle_transcript_event(self, transcript_event: TranscriptEvent) -> None:
        """Handle incoming transcript event from AWS stream."""
        transcript = getattr(transcript_event, "transcript", None)
        if not transcript or not transcript.results:
            return

        for result in transcript.results:
            is_final = not bool(getattr(result, "is_partial", False))
            alternatives = getattr(result, "alternatives", []) or []
            if not alternatives:
                continue

            for alt in alternatives:
                raw_text = getattr(alt, "transcript", "") or ""
                text = raw_text.strip()
                if not text:
                    continue

                word_timestamps: list[WordTimestamp] = []
                confidences: list[float] = []
                items = getattr(alt, "items", []) or []

                for item in items:
                    content = getattr(item, "content", "") or ""
                    start_sec = getattr(item, "start_time", None)
                    end_sec = getattr(item, "end_time", None)
                    conf = getattr(item, "confidence", None)

                    # Convert seconds to milliseconds where provided
                    start_ms = round(start_sec * 1000.0, 1) if start_sec is not None else 0.0
                    end_ms = round(end_sec * 1000.0, 1) if end_sec is not None else start_ms

                    item_conf = float(conf) if conf is not None else None
                    if item_conf is not None:
                        confidences.append(item_conf)

                    word_timestamps.append(
                        WordTimestamp(
                            word=content,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            confidence=item_conf,
                        )
                    )

                # Use genuine average confidence from items if available; do not fabricate
                confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

                # Duration calculation
                res_end_time = getattr(result, "end_time", None)
                if res_end_time is not None:
                    audio_dur_ms = round(res_end_time * 1000.0, 1)
                else:
                    audio_dur_ms = 0.0

                proc_time_ms = round((time.perf_counter() - self.stream_start_time) * 1000.0, 2)

                event = AsrTranscriptEvent(
                    session_id=self.session_id,
                    sequence_id=self.sequence_provider(),
                    text=text,
                    is_final=is_final,
                    confidence=confidence,
                    word_timestamps=word_timestamps,
                    latency_metrics=AsrLatencyMetrics(
                        audio_duration_ms=audio_dur_ms,
                        processing_time_ms=proc_time_ms,
                    ),
                )
                self.on_event(event)


class AwsTranscribeStreamSession:
    """Represents an active bi-directional streaming session with Amazon Transcribe."""

    def __init__(
        self,
        session_id: str,
        config: AsrEngineConfig,
        client: Any | None = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self._client = client
        self._stream: Any | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._sequence_id = 0
        self._received_events: list[AsrTranscriptEvent] = []
        self._stream_start_time = 0.0
        self._total_audio_ms = 0.0
        self._is_closed = False
        self._stream_ended = False
        self._stream_error: Exception | None = None

    def next_sequence_id(self) -> int:
        self._sequence_id += 1
        return self._sequence_id

    def _enqueue_event(self, event: AsrTranscriptEvent) -> None:
        self._received_events.append(event)

    async def _start_stream_if_needed(self) -> None:
        """Establish HTTP/2 connection with Amazon Transcribe Streaming."""
        if self._stream is not None:
            return

        if not HAS_AMAZON_TRANSCRIBE and self._client is None:
            raise AwsAsrConfigurationError(
                "amazon-transcribe package is not installed. Install with: uv add amazon-transcribe"
            )

        self._stream_start_time = time.perf_counter()
        if self._client is None:
            client = TranscribeStreamingClient(region=self.config.aws_region)
        else:
            client = self._client

        try:
            self._stream = await client.start_stream_transcription(
                language_code=self.config.aws_transcribe_language,
                media_sample_rate_hz=self.config.sample_rate,
                media_encoding=self.config.media_encoding,
            )
        except CredentialsException as e:
            raise AwsAsrAuthError(f"AWS credentials not found or invalid: {e}") from e
        except UnknownServiceException as e:
            msg = str(e)
            if "SubscriptionRequiredException" in msg or "403" in msg:
                raise AwsAsrAuthError(
                    f"AWS Transcribe subscription required or access denied: {e}"
                ) from e
            raise AwsAsrServiceUnavailableError(f"AWS Transcribe service error: {e}") from e
        except (ValidationException, BadRequestException) as e:
            raise AwsAsrConfigurationError(f"AWS Transcribe validation error: {e}") from e
        except (ServiceUnavailableException, InternalFailureException, LimitExceededException) as e:
            raise AwsAsrServiceUnavailableError(f"AWS Transcribe service unavailable: {e}") from e
        except SDKError as e:
            raise AwsAsrError(f"AWS SDK error during stream initiation: {e}") from e
        except Exception as e:
            raise AwsAsrServiceUnavailableError(f"Failed to connect to AWS Transcribe: {e}") from e

        # Start consumer task for output stream
        self._consumer_task = asyncio.create_task(self._consume_output())

    async def _consume_output(self) -> None:
        """Background coroutine consuming server transcript events from the output stream."""
        if self._stream is None:
            return

        try:
            handler = AwsTranscriptHandler(
                transcript_result_stream=self._stream.output_stream,
                on_event=self._enqueue_event,
                session_id=self.session_id,
                sequence_provider=self.next_sequence_id,
                stream_start_time=self._stream_start_time,
            )
            await handler.handle_events()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error in AWS Transcribe output consumer for session {self.session_id}: {e}")
            self._stream_error = e

    async def send_audio_chunk(self, audio_bytes: bytes) -> list[AsrTranscriptEvent]:
        """Send raw PCM16-LE audio chunk to active AWS Transcribe stream and drain available events."""
        if self._is_closed or self._stream_ended:
            raise AwsAsrError(f"Session '{self.session_id}' stream has already ended or closed.")

        await self._start_stream_if_needed()

        if self._stream_error is not None:
            err = self._stream_error
            self._stream_error = None
            raise AwsAsrServiceUnavailableError(f"AWS stream failed: {err}") from err

        chunk_ms = (len(audio_bytes) / 2 / self.config.sample_rate) * 1000.0
        self._total_audio_ms += chunk_ms

        try:
            if self._stream is not None and self._stream.input_stream is not None:
                await self._stream.input_stream.send_audio_event(audio_chunk=audio_bytes)
        except Exception as e:
            raise AwsAsrServiceUnavailableError(f"Failed to transmit audio chunk to AWS: {e}") from e

        # Brief yield to allow consumer task to process any immediate return frames
        await asyncio.sleep(0.01)

        events = list(self._received_events)
        self._received_events.clear()
        return events

    async def flush(self) -> list[AsrTranscriptEvent]:
        """End input audio stream and wait for final transcript events."""
        if self._stream is None or self._stream_ended:
            events = list(self._received_events)
            self._received_events.clear()
            return events

        self._stream_ended = True

        try:
            if self._stream.input_stream is not None:
                await self._stream.input_stream.end_stream()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error signaling end of audio stream for session {self.session_id}: {e}")

        # Await completion of consumer task with timeout
        if self._consumer_task is not None and not self._consumer_task.done():
            try:
                await asyncio.wait_for(self._consumer_task, timeout=self.config.stream_timeout_sec)
            except TimeoutError:
                logger.warning(f"Timed out awaiting AWS Transcribe stream completion for session {self.session_id}")
                self._consumer_task.cancel()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error during AWS stream flush for session {self.session_id}: {e}")

        events = list(self._received_events)
        self._received_events.clear()
        return events

    async def close(self) -> None:
        """Cancel background tasks and release streaming resources."""
        self._is_closed = True
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                pass
        self._stream = None
        self._received_events.clear()


class BackgroundLoopManager:
    """Thread-safe manager running an asyncio event loop in a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AwsAsrLoop")
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coroutine(self, coro: Any, timeout: float = 30.0) -> Any:
        """Submit coroutine to background loop and wait synchronously for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        """Stop background event loop cleanly."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3.0)


class AwsTranscribeStreamingEngine:
    """Production streaming ASR engine backed by Amazon Transcribe Streaming."""

    def __init__(
        self,
        config: AsrEngineConfig | None = None,
        client: Any | None = None,
        loop_manager: BackgroundLoopManager | None = None,
    ) -> None:
        self.config = config or AsrEngineConfig()
        self.config.validate()
        self._client = client
        self._loop_manager = loop_manager or BackgroundLoopManager()
        self._owns_loop_manager = loop_manager is None
        self._sessions: dict[str, AwsTranscribeStreamSession] = {}
        self._lock = threading.Lock()

    def get_or_create_session(self, session_id: str) -> AwsTranscribeStreamSession:
        """Retrieve existing streaming session or instantiate a new one."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = AwsTranscribeStreamSession(
                    session_id=session_id,
                    config=self.config,
                    client=self._client,
                )
            return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """Terminate and clean up active streaming session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            self._loop_manager.run_coroutine(session.close(), timeout=5.0)

    def _prepare_pcm_payload(
        self,
        audio_data: str | bytes | np.ndarray,
        audio_format: str = "pcm_s16le",
    ) -> bytes:
        """Decode input audio (WAV, WebM, Opus, PCM) and convert to raw signed 16-bit mono PCM."""
        if isinstance(audio_data, str):
            try:
                samples = decode_audio_payload(
                    audio_data,
                    audio_format=audio_format,
                    sample_rate=self.config.sample_rate,
                )
            except Exception as exc:
                raise AsrAudioDecodingError(f"Failed to decode base64 audio: {exc}") from exc
        elif isinstance(audio_data, bytes):
            # Check for WAV RIFF container header
            if audio_data.startswith(b"RIFF"):
                import base64
                b64 = base64.b64encode(audio_data).decode("ascii")
                samples = decode_audio_payload(b64, audio_format="wav", sample_rate=self.config.sample_rate)
            else:
                # Direct raw PCM16-LE bytes
                if len(audio_data) % 2 != 0:
                    raise AsrAudioDecodingError(
                        f"PCM16-LE byte payload must have even byte length, got {len(audio_data)}"
                    )
                int16_arr = np.frombuffer(audio_data, dtype="<i2")
                samples = (int16_arr.astype(np.float32) / 32768.0).astype(np.float32)
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                samples = (audio_data.astype(np.float32) / 32768.0).astype(np.float32)
            else:
                samples = audio_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")

        if len(samples) == 0:
            return b""

        # Convert normalized float32 mono samples to raw 16-bit signed little-endian PCM
        int16_arr = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        return int16_arr.tobytes()

    def process_audio_chunk(
        self,
        session_id: str,
        audio_data: str | bytes | np.ndarray,
        audio_format: str = "pcm_s16le",
    ) -> list[AsrTranscriptEvent]:
        """Ingest audio chunk, forward raw PCM to Amazon Transcribe, and return transcript events."""
        pcm_bytes = self._prepare_pcm_payload(audio_data, audio_format=audio_format)
        if len(pcm_bytes) == 0:
            return []

        session = self.get_or_create_session(session_id)
        return self._loop_manager.run_coroutine(
            session.send_audio_chunk(pcm_bytes),
            timeout=self.config.stream_timeout_sec,
        )

    def flush_session(self, session_id: str) -> list[AsrTranscriptEvent]:
        """Finalize the active streaming session and return all remaining final events."""
        with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            return []

        events = self._loop_manager.run_coroutine(
            session.flush(),
            timeout=self.config.stream_timeout_sec + 5.0,
        )
        # Ensure session resources are released
        self._loop_manager.run_coroutine(session.close(), timeout=5.0)
        return events

    def close(self) -> None:
        """Close all active sessions and stop background worker loop."""
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            self.close_session(sid)

        if self._owns_loop_manager:
            self._loop_manager.stop()

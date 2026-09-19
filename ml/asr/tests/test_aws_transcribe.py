"""Comprehensive unit test suite for Amazon Transcribe Streaming ASR engine."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from asr.aws_transcribe import (
    AwsTranscribeStreamingEngine,
    AwsTranscribeStreamSession,
    BackgroundLoopManager,
)
from asr.config import AsrEngineConfig
from asr.engine import (
    create_asr_engine,
)
from asr.exceptions import (
    AsrAudioDecodingError,
    AwsAsrAuthError,
    AwsAsrConfigurationError,
    AwsAsrServiceUnavailableError,
)


class MockAwsItem:
    def __init__(
        self,
        content: str,
        start_time: float | None = None,
        end_time: float | None = None,
        confidence: float | None = None,
    ) -> None:
        self.content = content
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence


class MockAwsAlternative:
    def __init__(self, transcript: str, items: list[MockAwsItem] | None = None) -> None:
        self.transcript = transcript
        self.items = items or []


class MockAwsResult:
    def __init__(
        self,
        alternatives: list[MockAwsAlternative],
        is_partial: bool = False,
        start_time: float | None = 0.0,
        end_time: float | None = 1.0,
    ) -> None:
        self.alternatives = alternatives
        self.is_partial = is_partial
        self.start_time = start_time
        self.end_time = end_time


class MockAwsTranscript:
    def __init__(self, results: list[MockAwsResult]) -> None:
        self.results = results


from amazon_transcribe.model import TranscriptEvent


class MockAwsTranscriptEvent(TranscriptEvent):
    def __init__(self, transcript: MockAwsTranscript) -> None:
        self.transcript = transcript


class MockAudioStream:
    """Mock for amazon_transcribe.model.AudioStream."""

    def __init__(self) -> None:
        self.sent_chunks: list[bytes] = []
        self.is_ended = False

    async def send_audio_event(self, audio_chunk: bytes | None) -> None:
        if audio_chunk is not None:
            self.sent_chunks.append(audio_chunk)

    async def end_stream(self) -> None:
        self.is_ended = True


class MockOutputStream:
    """Mock async generator for stream.output_stream."""

    def __init__(
        self,
        events: list[MockAwsTranscriptEvent] | None = None,
        keep_open: bool = False,
    ) -> None:
        self.events = events or []
        self.keep_open = keep_open

    async def __aiter__(self) -> Any:
        for ev in self.events:
            yield ev
        if self.keep_open:
            while True:
                await asyncio.sleep(0.05)


class MockStream:
    """Mock for StartStreamTranscriptionEventStream."""

    def __init__(
        self,
        output_events: list[MockAwsTranscriptEvent] | None = None,
        keep_open: bool = False,
    ) -> None:
        self.input_stream = MockAudioStream()
        self.output_stream = MockOutputStream(output_events, keep_open=keep_open)


class MockTranscribeClient:
    """Mock for TranscribeStreamingClient."""

    def __init__(
        self,
        output_events: list[MockAwsTranscriptEvent] | None = None,
        keep_open: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.output_events = output_events or []
        self.stream = MockStream(self.output_events, keep_open=keep_open)

    async def start_stream_transcription(
        self,
        *,
        language_code: str,
        media_sample_rate_hz: int,
        media_encoding: str,
        **kwargs: Any,
    ) -> MockStream:
        self.calls.append({
            "language_code": language_code,
            "media_sample_rate_hz": media_sample_rate_hz,
            "media_encoding": media_encoding,
            "kwargs": kwargs,
        })
        return self.stream


@pytest.fixture
def loop_manager() -> Any:
    mgr = BackgroundLoopManager()
    yield mgr
    mgr.stop()


class TestAwsTranscribeConfiguration:
    """Tests for configuration, region, language, and validation."""

    def test_default_configuration(self) -> None:
        config = AsrEngineConfig()
        assert config.backend == "aws"
        assert config.aws_region == "ap-south-1"
        assert config.aws_transcribe_language == "en-IN"
        assert config.sample_rate == 16000
        assert config.media_encoding == "pcm"

    def test_custom_configuration(self) -> None:
        config = AsrEngineConfig(
            aws_region="us-east-1",
            aws_transcribe_language="en-US",
            sample_rate=16000,
        )
        assert config.aws_region == "us-east-1"
        assert config.aws_transcribe_language == "en-US"

    def test_invalid_backend_raises(self) -> None:
        config = AsrEngineConfig(backend="invalid_backend")
        with pytest.raises(AwsAsrConfigurationError, match="Unsupported ASR_BACKEND"):
            config.validate()

    def test_invalid_sample_rate_raises(self) -> None:
        config = AsrEngineConfig(sample_rate=-1)
        with pytest.raises(AwsAsrConfigurationError, match="sample_rate must be positive"):
            config.validate()


class TestAwsTranscribeStreamingSession:
    """Unit tests for AwsTranscribeStreamSession lifecycle and stream handling."""

    def test_session_starts_stream_with_correct_parameters(self) -> None:
        async def _run() -> None:
            mock_client = MockTranscribeClient()
            config = AsrEngineConfig(
                aws_region="ap-south-1",
                aws_transcribe_language="en-IN",
                sample_rate=16000,
                media_encoding="pcm",
            )
            session = AwsTranscribeStreamSession("test-sess", config, client=mock_client)

            chunk = b"\x00\x00" * 160  # 10ms of 16kHz PCM16
            await session.send_audio_chunk(chunk)

            assert len(mock_client.calls) == 1
            call = mock_client.calls[0]
            assert call["language_code"] == "en-IN"
            assert call["media_sample_rate_hz"] == 16000
            assert call["media_encoding"] == "pcm"
            await session.close()

        asyncio.run(_run())

    def test_audio_chunk_forwarding(self) -> None:
        async def _run() -> None:
            mock_client = MockTranscribeClient()
            config = AsrEngineConfig()
            session = AwsTranscribeStreamSession("test-sess", config, client=mock_client)

            chunk1 = b"\x01\x00" * 80
            chunk2 = b"\x02\x00" * 80

            await session.send_audio_chunk(chunk1)
            await session.send_audio_chunk(chunk2)

            sent = mock_client.stream.input_stream.sent_chunks
            assert len(sent) == 2
            assert sent[0] == chunk1
            assert sent[1] == chunk2
            await session.close()

        asyncio.run(_run())

    def test_partial_transcript_conversion(self) -> None:
        async def _run() -> None:
            event = MockAwsTranscriptEvent(
                transcript=MockAwsTranscript(
                    results=[
                        MockAwsResult(
                            alternatives=[
                                MockAwsAlternative(
                                    transcript="hello wor",
                                    items=[
                                        MockAwsItem("hello", 0.1, 0.4, 0.95),
                                        MockAwsItem("wor", 0.5, 0.7, 0.70),
                                    ],
                                )
                            ],
                            is_partial=True,
                            end_time=0.7,
                        )
                    ]
                )
            )
            mock_client = MockTranscribeClient(output_events=[event])
            session = AwsTranscribeStreamSession("sess-part", AsrEngineConfig(), client=mock_client)

            events = await session.send_audio_chunk(b"\x00\x00" * 160)
            flushed = await session.flush()
            all_events = events + flushed

            assert len(all_events) >= 1
            ev = all_events[0]
            assert ev.session_id == "sess-part"
            assert ev.text == "hello wor"
            assert ev.is_final is False
            assert ev.confidence == round((0.95 + 0.70) / 2, 4)
            assert len(ev.word_timestamps) == 2
            assert ev.word_timestamps[0].word == "hello"
            assert ev.word_timestamps[0].start_ms == 100.0
            assert ev.word_timestamps[0].end_ms == 400.0
            await session.close()

        asyncio.run(_run())

    def test_final_transcript_conversion(self) -> None:
        async def _run() -> None:
            event = MockAwsTranscriptEvent(
                transcript=MockAwsTranscript(
                    results=[
                        MockAwsResult(
                            alternatives=[
                                MockAwsAlternative(
                                    transcript="hello world",
                                    items=[
                                        MockAwsItem("hello", 0.1, 0.4, 0.98),
                                        MockAwsItem("world", 0.5, 0.9, 0.92),
                                    ],
                                )
                            ],
                            is_partial=False,
                            end_time=0.9,
                        )
                    ]
                )
            )
            mock_client = MockTranscribeClient(output_events=[event])
            session = AwsTranscribeStreamSession("sess-final", AsrEngineConfig(), client=mock_client)

            events = await session.send_audio_chunk(b"\x00\x00" * 160)
            flushed = await session.flush()
            all_events = events + flushed

            assert len(all_events) == 1
            ev = all_events[0]
            assert ev.session_id == "sess-final"
            assert ev.text == "hello world"
            assert ev.is_final is True
            assert ev.confidence == round((0.98 + 0.92) / 2, 4)
            await session.close()

        asyncio.run(_run())

    def test_missing_optional_fields_no_fabrication(self) -> None:
        """Verify that when AWS omits confidence or timestamps, they are NOT fabricated."""
        async def _run() -> None:
            event = MockAwsTranscriptEvent(
                transcript=MockAwsTranscript(
                    results=[
                        MockAwsResult(
                            alternatives=[
                                MockAwsAlternative(
                                    transcript="namaste",
                                    items=[
                                        MockAwsItem("namaste", start_time=None, end_time=None, confidence=None),
                                    ],
                                )
                            ],
                            is_partial=False,
                            start_time=None,
                            end_time=None,
                        )
                    ]
                )
            )
            mock_client = MockTranscribeClient(output_events=[event])
            session = AwsTranscribeStreamSession("sess-opt", AsrEngineConfig(), client=mock_client)

            events = await session.send_audio_chunk(b"\x00\x00" * 160)
            flushed = await session.flush()
            all_events = events + flushed

            assert len(all_events) == 1
            ev = all_events[0]
            assert ev.text == "namaste"
            # Confidence must be 0.0 (unspecified) rather than fabricated high score
            assert ev.confidence == 0.0
            assert len(ev.word_timestamps) == 1
            assert ev.word_timestamps[0].confidence is None
            await session.close()

        asyncio.run(_run())

    def test_explicit_flush_ends_stream(self) -> None:
        async def _run() -> None:
            mock_client = MockTranscribeClient()
            session = AwsTranscribeStreamSession("sess-flush", AsrEngineConfig(), client=mock_client)

            await session.send_audio_chunk(b"\x00\x00" * 160)
            assert mock_client.stream.input_stream.is_ended is False

            await session.flush()
            assert mock_client.stream.input_stream.is_ended is True
            await session.close()

        asyncio.run(_run())

    def test_cancellation_and_cleanup(self) -> None:
        async def _run() -> None:
            mock_client = MockTranscribeClient(keep_open=True)
            session = AwsTranscribeStreamSession("sess-clean", AsrEngineConfig(), client=mock_client)

            await session.send_audio_chunk(b"\x00\x00" * 160)
            assert session._consumer_task is not None
            assert not session._consumer_task.done()

            await session.close()
            assert session._is_closed is True
            assert session._stream is None

        asyncio.run(_run())


class TestAwsTranscribeErrorsAndNoFallback:
    """Verify explicit typed error handling and ensure NO silent fallback."""

    def test_authentication_error_on_credentials_failure(self) -> None:
        async def _run() -> None:
            from amazon_transcribe.exceptions import CredentialsException

            client = MagicMock()
            client.start_stream_transcription = AsyncMock(
                side_effect=CredentialsException("Unable to locate credentials")
            )

            session = AwsTranscribeStreamSession("sess-auth-err", AsrEngineConfig(), client=client)
            with pytest.raises(AwsAsrAuthError, match="AWS credentials not found"):
                await session.send_audio_chunk(b"\x00\x00" * 160)
            await session.close()

        asyncio.run(_run())

    def test_subscription_required_maps_to_auth_error(self) -> None:
        async def _run() -> None:
            from amazon_transcribe.exceptions import UnknownServiceException

            client = MagicMock()
            client.start_stream_transcription = AsyncMock(
                side_effect=UnknownServiceException(403, "SubscriptionRequiredException", "Subscription required")
            )

            session = AwsTranscribeStreamSession("sess-sub-err", AsrEngineConfig(), client=client)
            with pytest.raises(AwsAsrAuthError, match="subscription required or access denied"):
                await session.send_audio_chunk(b"\x00\x00" * 160)
            await session.close()

        asyncio.run(_run())

    def test_service_unavailable_error(self) -> None:
        async def _run() -> None:
            from amazon_transcribe.exceptions import ServiceUnavailableException

            client = MagicMock()
            client.start_stream_transcription = AsyncMock(
                side_effect=ServiceUnavailableException("Service temporarily overloaded")
            )

            session = AwsTranscribeStreamSession("sess-503-err", AsrEngineConfig(), client=client)
            with pytest.raises(AwsAsrServiceUnavailableError, match="service unavailable"):
                await session.send_audio_chunk(b"\x00\x00" * 160)
            await session.close()

        asyncio.run(_run())

    def test_validation_error(self) -> None:
        async def _run() -> None:
            from amazon_transcribe.exceptions import ValidationException

            client = MagicMock()
            client.start_stream_transcription = AsyncMock(
                side_effect=ValidationException("Invalid language code")
            )

            session = AwsTranscribeStreamSession("sess-val-err", AsrEngineConfig(), client=client)
            with pytest.raises(AwsAsrConfigurationError, match="validation error"):
                await session.send_audio_chunk(b"\x00\x00" * 160)
            await session.close()

        asyncio.run(_run())

    def test_zero_silent_fallback_to_whisper_on_aws_failure(self, loop_manager: BackgroundLoopManager) -> None:
        """Verify that when AWS fails, the engine throws AwsAsrError and NEVER falls back to Whisper."""
        from amazon_transcribe.exceptions import CredentialsException

        mock_client = MagicMock()
        mock_client.start_stream_transcription = AsyncMock(
            side_effect=CredentialsException("No AWS credentials")
        )

        engine = AwsTranscribeStreamingEngine(
            config=AsrEngineConfig(backend="aws"),
            client=mock_client,
            loop_manager=loop_manager,
        )

        with pytest.raises(AwsAsrAuthError):
            engine.process_audio_chunk("sess-no-fallback", b"\x00\x00" * 160)

        # Confirm session was NOT replaced with mock or Whisper transcript
        events = engine.flush_session("sess-no-fallback")
        assert len(events) == 0


class TestAwsTranscribeAudioPreprocessing:
    """Verify input audio format conversions and validation."""

    def test_pcm16_le_encoding(self, loop_manager: BackgroundLoopManager) -> None:
        mock_client = MockTranscribeClient()
        engine = AwsTranscribeStreamingEngine(
            config=AsrEngineConfig(),
            client=mock_client,
            loop_manager=loop_manager,
        )

        # 3 samples: 0.0, 0.5, -0.5
        samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        raw_pcm = engine._prepare_pcm_payload(samples)

        assert len(raw_pcm) == 6
        int16_arr = np.frombuffer(raw_pcm, dtype="<i2")
        assert int16_arr[0] == 0
        assert abs(int16_arr[1] - 16383) <= 2
        assert abs(int16_arr[2] - (-16383)) <= 2

    def test_odd_pcm_byte_length_raises(self, loop_manager: BackgroundLoopManager) -> None:
        mock_client = MockTranscribeClient()
        engine = AwsTranscribeStreamingEngine(client=mock_client, loop_manager=loop_manager)

        with pytest.raises(AsrAudioDecodingError, match="even byte length"):
            engine.process_audio_chunk("sess-odd", b"\x00\x01\x02")

    def test_wav_container_payload_decoded(self, loop_manager: BackgroundLoopManager) -> None:
        import io
        import wave

        mock_client = MockTranscribeClient()
        engine = AwsTranscribeStreamingEngine(client=mock_client, loop_manager=loop_manager)

        # Generate RIFF WAV container in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((np.zeros(160, dtype=np.int16)).tobytes())
        wav_bytes = buf.getvalue()

        # Engine must strip WAV container and forward raw PCM
        raw_pcm = engine._prepare_pcm_payload(wav_bytes)
        assert not raw_pcm.startswith(b"RIFF")
        assert len(raw_pcm) == 320


class TestEngineFactory:
    """Verify create_asr_engine factory and backend selection."""

    def test_create_aws_engine(self) -> None:
        engine = create_asr_engine(backend="aws")
        assert isinstance(engine, AwsTranscribeStreamingEngine)
        engine.close()

    def test_create_whisper_engine(self) -> None:
        engine = create_asr_engine(backend="whisper")
        from asr.engine import WhisperAsrEngine
        assert isinstance(engine, WhisperAsrEngine)

    def test_create_mock_engine(self) -> None:
        engine = create_asr_engine(backend="mock")
        from asr.engine import WhisperAsrEngine
        assert isinstance(engine, WhisperAsrEngine)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(AwsAsrConfigurationError):
            create_asr_engine(backend="nonexistent_engine")

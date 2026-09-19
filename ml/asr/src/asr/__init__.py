"""Streaming ASR and Voice Activity Detection (VAD) module."""

from asr.aws_transcribe import (
    AwsTranscribeStreamingEngine,
    AwsTranscribeStreamSession,
    AwsTranscriptHandler,
    BackgroundLoopManager,
)
from asr.buffer import (
    AudioChunk,
    SlidingAudioBuffer,
    decode_audio_payload,
    resample_to_16k,
)
from asr.config import AsrEngineConfig
from asr.engine import (
    AsrBackendProtocol,
    AsrEngineProtocol,
    AsrLatencyMetrics,
    AsrSessionState,
    AsrTranscriptEvent,
    FasterWhisperBackend,
    MockAsrBackend,
    StreamingAsrEngine,
    WhisperAsrEngine,
    WordTimestamp,
    create_asr_engine,
)
from asr.exceptions import (
    AsrAudioDecodingError,
    AsrError,
    AsrSessionNotFoundError,
    AwsAsrAuthError,
    AwsAsrConfigurationError,
    AwsAsrError,
    AwsAsrServiceUnavailableError,
)
from asr.vad import (
    EnergyZcrBaselineBackend,
    SileroOnnxVadBackend,
    VadBackendProtocol,
    VadConfig,
    VadFrameResult,
    VadState,
    VoiceActivityDetector,
)

__all__ = [
    "AsrAudioDecodingError",
    "AsrBackendProtocol",
    "AsrEngineConfig",
    "AsrEngineProtocol",
    "AsrError",
    "AsrLatencyMetrics",
    "AsrSessionNotFoundError",
    "AsrSessionState",
    "AsrTranscriptEvent",
    "AudioChunk",
    "AwsAsrAuthError",
    "AwsAsrConfigurationError",
    "AwsAsrError",
    "AwsAsrServiceUnavailableError",
    "AwsTranscribeStreamSession",
    "AwsTranscribeStreamingEngine",
    "AwsTranscriptHandler",
    "BackgroundLoopManager",
    "EnergyZcrBaselineBackend",
    "FasterWhisperBackend",
    "MockAsrBackend",
    "SileroOnnxVadBackend",
    "SlidingAudioBuffer",
    "StreamingAsrEngine",
    "VadBackendProtocol",
    "VadConfig",
    "VadFrameResult",
    "VadState",
    "VoiceActivityDetector",
    "WhisperAsrEngine",
    "WordTimestamp",
    "create_asr_engine",
    "decode_audio_payload",
    "resample_to_16k",
]


def main() -> None:
    print("Converse ASR engine initialized")

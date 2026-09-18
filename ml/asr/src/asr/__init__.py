"""Streaming ASR and Voice Activity Detection (VAD) module."""

from asr.buffer import (
    AudioChunk,
    SlidingAudioBuffer,
    decode_audio_payload,
    resample_to_16k,
)
from asr.engine import (
    AcousticBaselineBackend,
    AsrBackendProtocol,
    AsrEngineConfig,
    AsrLatencyMetrics,
    AsrSessionState,
    AsrTranscriptEvent,
    StreamingAsrEngine,
    WordTimestamp,
)
from asr.vad import (
    VadConfig,
    VadFrameResult,
    VadState,
    VoiceActivityDetector,
)

__all__ = [
    "AcousticBaselineBackend",
    "AsrBackendProtocol",
    "AsrEngineConfig",
    "AsrLatencyMetrics",
    "AsrSessionState",
    "AsrTranscriptEvent",
    "AudioChunk",
    "SlidingAudioBuffer",
    "StreamingAsrEngine",
    "VadConfig",
    "VadFrameResult",
    "VadState",
    "VoiceActivityDetector",
    "WordTimestamp",
    "decode_audio_payload",
    "resample_to_16k",
]


def main() -> None:
    print("Converse ASR engine initialized")

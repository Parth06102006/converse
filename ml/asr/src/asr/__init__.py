"""Streaming ASR and Voice Activity Detection (VAD) module."""

from asr.buffer import (
    AudioChunk,
    SlidingAudioBuffer,
    decode_audio_payload,
    resample_to_16k,
)
from asr.engine import (
    AsrBackendProtocol,
    AsrEngineConfig,
    AsrLatencyMetrics,
    AsrSessionState,
    AsrTranscriptEvent,
    FasterWhisperBackend,
    MockAsrBackend,
    StreamingAsrEngine,
    WordTimestamp,
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
    "AsrBackendProtocol",
    "AsrEngineConfig",
    "AsrLatencyMetrics",
    "AsrSessionState",
    "AsrTranscriptEvent",
    "AudioChunk",
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
    "WordTimestamp",
    "decode_audio_payload",
    "resample_to_16k",
]


def main() -> None:
    print("Converse ASR engine initialized")

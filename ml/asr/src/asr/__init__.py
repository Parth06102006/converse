"""Streaming ASR and Voice Activity Detection (VAD) module."""

from asr.buffer import (
    AudioChunk,
    SlidingAudioBuffer,
    decode_audio_payload,
    resample_to_16k,
)
from asr.vad import (
    VadConfig,
    VadFrameResult,
    VadState,
    VoiceActivityDetector,
)

__all__ = [
    "AudioChunk",
    "SlidingAudioBuffer",
    "VadConfig",
    "VadFrameResult",
    "VadState",
    "VoiceActivityDetector",
    "decode_audio_payload",
    "resample_to_16k",
]


def main() -> None:
    print("Converse ASR engine initialized")

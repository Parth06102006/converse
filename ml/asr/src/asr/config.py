"""Configuration models and environment loading for ASR engines."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AsrEngineConfig:
    """Unified configuration for ASR engines."""

    backend: str = field(
        default_factory=lambda: os.environ.get("ASR_BACKEND", "aws").strip().lower()
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "ap-south-1").strip()
    )
    aws_transcribe_language: str = field(
        default_factory=lambda: os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-IN").strip()
    )
    sample_rate: int = 16000
    media_encoding: str = "pcm"
    partial_interval_ms: float = 1400.0
    min_audio_duration_ms: float = 180.0
    max_utterance_duration_sec: float = 30.0
    stream_timeout_sec: float = 15.0
    hallucination_phrases: tuple[str, ...] = (
        "thank you for watching",
        "thanks for watching",
        "subscribe to my channel",
        "subtitles by",
    )

    def validate(self) -> None:
        """Validate configuration invariants."""
        if self.backend not in ("aws", "whisper", "mock"):
            from asr.exceptions import AwsAsrConfigurationError
            raise AwsAsrConfigurationError(
                f"Unsupported ASR_BACKEND '{self.backend}'. Must be 'aws', 'whisper', or 'mock'."
            )
        if self.sample_rate <= 0:
            from asr.exceptions import AwsAsrConfigurationError
            raise AwsAsrConfigurationError(f"sample_rate must be positive, got {self.sample_rate}")
        if not self.aws_region:
            from asr.exceptions import AwsAsrConfigurationError
            raise AwsAsrConfigurationError("AWS_REGION must not be empty.")
        if not self.aws_transcribe_language:
            from asr.exceptions import AwsAsrConfigurationError
            raise AwsAsrConfigurationError("AWS_TRANSCRIBE_LANGUAGE must not be empty.")

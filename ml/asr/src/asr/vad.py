"""Voice Activity Detection (VAD) engine for streaming audio segmentation."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class VadState(str, Enum):
    """Internal state of the speech detector."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True)
class VadConfig:
    """Configuration parameters for Voice Activity Detection."""

    sample_rate: int = 16000
    frame_size_ms: int = 30  # Standard 30ms frame (480 samples @ 16kHz)
    energy_threshold: float = 0.012  # RMS threshold for acoustic speech
    zcr_threshold: float = 0.35  # Max Zero Crossing Rate to filter high-freq noise
    min_speech_duration_ms: int = 250  # Minimum active speech to lock utterance
    min_silence_duration_ms: int = 300  # Silence hangover threshold to trigger boundary
    speech_pad_ms: int = 60  # Leading/trailing audio padding


@dataclass(frozen=True)
class VadFrameResult:
    """Per-frame classification output."""

    is_speech: bool
    speech_probability: float
    energy_rms: float
    is_utterance_boundary: bool
    accumulated_speech_ms: float
    state: VadState


class VoiceActivityDetector:
    """Stateful streaming Voice Activity Detector.

    Analyzes continuous 16kHz audio chunks, maintains speech/silence state machines,
    and detects utterance boundaries based on configurable hangover durations.
    Supports an optional neural model delegate (such as Silero VAD ONNX).
    """

    def __init__(
        self,
        config: VadConfig | None = None,
        model_delegate: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config or VadConfig()
        self.model_delegate = model_delegate

        self.frame_samples = int((self.config.frame_size_ms / 1000.0) * self.config.sample_rate)

        self._state: VadState = VadState.SILENCE
        self._consecutive_speech_ms: float = 0.0
        self._consecutive_silence_ms: float = 0.0
        self._accumulated_speech_ms: float = 0.0

    @property
    def current_state(self) -> VadState:
        return self._state

    @property
    def accumulated_speech_ms(self) -> float:
        return self._accumulated_speech_ms

    def reset(self) -> None:
        """Reset internal temporal counters and state machine."""
        self._state = VadState.SILENCE
        self._consecutive_speech_ms = 0.0
        self._consecutive_silence_ms = 0.0
        self._accumulated_speech_ms = 0.0

    def compute_energy_rms(self, frame: np.ndarray) -> float:
        """Compute root-mean-square (RMS) energy of audio frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame**2)))

    def compute_zero_crossing_rate(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate of audio frame."""
        if len(frame) <= 1:
            return 0.0
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zero_crossings = np.sum(signs[:-1] != signs[1:])
        return float(zero_crossings / (len(frame) - 1))

    def evaluate_frame_probability(self, frame: np.ndarray) -> float:
        """Compute speech probability in range [0.0, 1.0]."""
        if self.model_delegate is not None:
            raw_prob = self.model_delegate(frame)
            return float(np.clip(raw_prob, 0.0, 1.0))

        rms = self.compute_energy_rms(frame)
        zcr = self.compute_zero_crossing_rate(frame)

        if rms < self.config.energy_threshold:
            # Low energy is overwhelmingly silence
            return float(np.clip(rms / (self.config.energy_threshold * 2.0), 0.0, 0.45))

        # Reasonable energy above threshold
        # Filter high ZCR (often white noise or hiss)
        noise_penalty = 0.0
        if zcr > self.config.zcr_threshold:
            noise_penalty = min(0.3, (zcr - self.config.zcr_threshold) * 0.8)

        energy_factor = min(1.0, (rms - self.config.energy_threshold) / (self.config.energy_threshold * 3.0))
        prob = 0.5 + 0.5 * energy_factor - noise_penalty
        return float(np.clip(prob, 0.0, 1.0))

    def process_frame(self, frame: np.ndarray) -> VadFrameResult:
        """Process a single frame of audio (typically 20ms-100ms) and update state machine."""
        duration_ms = (len(frame) / self.config.sample_rate) * 1000.0
        prob = self.evaluate_frame_probability(frame)
        rms = self.compute_energy_rms(frame)
        is_speech = prob >= 0.5

        is_boundary = False

        if is_speech:
            self._consecutive_speech_ms += duration_ms
            self._consecutive_silence_ms = 0.0

            if self._state == VadState.SILENCE:
                if self._consecutive_speech_ms >= self.config.min_speech_duration_ms:
                    self._state = VadState.SPEECH
                    self._accumulated_speech_ms = self._consecutive_speech_ms
            else:
                self._accumulated_speech_ms += duration_ms
        else:
            self._consecutive_silence_ms += duration_ms
            self._consecutive_speech_ms = 0.0

            if self._state == VadState.SPEECH:
                self._accumulated_speech_ms += duration_ms
                if self._consecutive_silence_ms >= self.config.min_silence_duration_ms:
                    is_boundary = True
                    self._state = VadState.SILENCE
                    self._accumulated_speech_ms = 0.0

        return VadFrameResult(
            is_speech=is_speech,
            speech_probability=prob,
            energy_rms=rms,
            is_utterance_boundary=is_boundary,
            accumulated_speech_ms=self._accumulated_speech_ms,
            state=self._state,
        )

    def process_chunk(self, chunk: np.ndarray) -> list[VadFrameResult]:
        """Process an arbitrarily sized audio chunk by slicing into frame_size_ms windows."""
        if len(chunk) == 0:
            return []

        results: list[VadFrameResult] = []
        frame_len = self.frame_samples

        for start in range(0, len(chunk), frame_len):
            frame = chunk[start : start + frame_len]
            if len(frame) < frame_len // 2:
                # Discard negligible trailing stub
                continue
            res = self.process_frame(frame)
            results.append(res)

        return results

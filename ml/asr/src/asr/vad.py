"""Voice Activity Detection (VAD) engine with pretrained Silero ONNX model and state machine."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class VadState(str, Enum):
    """Internal state of the temporal speech detector."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True)
class VadConfig:
    """Configuration parameters for Voice Activity Detection."""

    sample_rate: int = 16000
    frame_size_ms: int = 32  # Standard 32ms frame (512 samples @ 16kHz for Silero ONNX)
    speech_threshold: float = 0.5  # Model probability threshold to declare speech
    energy_threshold: float = 0.012  # RMS threshold for acoustic heuristic fallback
    zcr_threshold: float = 0.35  # Max Zero Crossing Rate for heuristic fallback
    min_speech_duration_ms: int = 250  # Minimum active speech to lock utterance
    min_silence_duration_ms: int = 300  # Silence hangover threshold to trigger boundary
    speech_pad_ms: int = 60  # Pre/post padding
    use_neural: bool = True  # Production default using pretrained Silero ONNX model


@dataclass(frozen=True)
class VadFrameResult:
    """Per-frame classification output."""

    is_speech: bool
    speech_probability: float
    energy_rms: float
    is_utterance_boundary: bool
    accumulated_speech_ms: float
    state: VadState


class VadBackendProtocol(Protocol):
    """Protocol interface for VAD probability estimation backends."""

    def predict_probability(self, frame: np.ndarray, sample_rate: int) -> float:
        """Compute speech probability in range [0.0, 1.0]."""
        ...

    def reset(self) -> None:
        """Reset internal recurrent states."""
        ...


class SileroOnnxVadBackend:
    """Production neural Voice Activity Detection using pretrained Silero VAD v5 ONNX."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required for SileroOnnxVadBackend but is not installed.")

        default_path = Path(__file__).resolve().parent.parent.parent / "models" / "silero_vad.onnx"
        self.model_path = Path(model_path or os.environ.get("VAD_MODEL_PATH", default_path))

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model file not found at '{self.model_path}'. "
                "Download it with: curl -L -o ml/asr/models/silero_vad.onnx "
                "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._sr_arr = np.array(16000, dtype=np.int64)

    def reset(self) -> None:
        """Reset recurrent neural state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def predict_probability(self, frame: np.ndarray, sample_rate: int = 16000) -> float:
        """Run Silero ONNX model inference on audio chunk.

        Expects 512 samples at 16kHz (or 256 samples at 8kHz). Prepends 64-sample context.
        """
        flat = frame.flatten().astype(np.float32)
        target_len = 512 if sample_rate == 16000 else 256
        if len(flat) < target_len:
            flat = np.pad(flat, (0, target_len - len(flat)))
        elif len(flat) > target_len:
            flat = flat[:target_len]

        chunk = flat.reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)

        sr_tensor = np.array(sample_rate, dtype=np.int64)
        outputs = self._session.run(None, {
            "input": x,
            "state": self._state,
            "sr": sr_tensor,
        })

        prob, self._state = outputs[0], outputs[1]
        self._context = chunk[:, -64:]
        return float(np.clip(prob[0][0], 0.0, 1.0))


class EnergyZcrBaselineBackend:
    """Heuristic baseline computing probability via RMS energy and Zero-Crossing Rate.

    Used strictly as an explicitly named fallback when pretrained neural models
    cannot be initialized.
    """

    def __init__(self, energy_threshold: float = 0.012, zcr_threshold: float = 0.35) -> None:
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold

    def reset(self) -> None:
        pass

    def predict_probability(self, frame: np.ndarray, sample_rate: int = 16000) -> float:
        if len(frame) == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(frame**2)))
        if len(frame) > 1:
            signs = np.sign(frame)
            signs[signs == 0] = 1
            zcr = float(np.sum(signs[:-1] != signs[1:]) / (len(frame) - 1))
        else:
            zcr = 0.0

        if rms < (self.energy_threshold * 0.5):
            return 0.0

        energy_score = min(1.0, rms / (self.energy_threshold * 2.5))
        zcr_penalty = max(0.0, 1.0 - (zcr / self.zcr_threshold)) if zcr > self.zcr_threshold else 1.0
        return float(np.clip(energy_score * zcr_penalty, 0.0, 1.0))


class VoiceActivityDetector:
    """Temporal Voice Activity Detector with temporal hangover state machine.

    Driven primarily by pretrained Silero VAD neural probabilities, falling back
    to EnergyZcrBaselineBackend if neural models are unavailable.
    """

    def __init__(
        self,
        config: VadConfig | None = None,
        backend: VadBackendProtocol | None = None,
        model_delegate: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config or VadConfig()
        self.frame_samples = int((self.config.frame_size_ms / 1000.0) * self.config.sample_rate)

        # Configure probability estimator
        if backend is not None:
            self.backend = backend
        elif model_delegate is not None:
            # Custom delegate wrapper
            class DelegateWrapper:
                def __init__(self, fn: Callable[[np.ndarray], float]) -> None:
                    self.fn = fn

                def predict_probability(self, frame: np.ndarray, sample_rate: int) -> float:
                    return self.fn(frame)

                def reset(self) -> None:
                    pass

            self.backend = DelegateWrapper(model_delegate)
        elif self.config.use_neural:
            # Pretrained neural Silero ONNX backend
            self.backend = SileroOnnxVadBackend()
        else:
            # Deterministic heuristic energy baseline
            self.backend = EnergyZcrBaselineBackend(
                energy_threshold=self.config.energy_threshold,
                zcr_threshold=self.config.zcr_threshold,
            )

        self._state: VadState = VadState.SILENCE
        self._consecutive_speech_ms: float = 0.0
        self._consecutive_silence_ms: float = 0.0
        self._accumulated_speech_ms: float = 0.0
        self._unprocessed_samples: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def current_state(self) -> VadState:
        return self._state

    @property
    def accumulated_speech_ms(self) -> float:
        return self._accumulated_speech_ms

    def reset(self) -> None:
        """Reset internal temporal counters, remainder buffer, and backend states."""
        self._state = VadState.SILENCE
        self._consecutive_speech_ms = 0.0
        self._consecutive_silence_ms = 0.0
        self._accumulated_speech_ms = 0.0
        self._unprocessed_samples = np.empty(0, dtype=np.float32)
        self.backend.reset()

    def compute_energy_rms(self, frame: np.ndarray) -> float:
        """Compute root-mean-square (RMS) energy of audio frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame**2)))

    def evaluate_frame_probability(self, frame: np.ndarray) -> float:
        """Compute speech probability using active backend."""
        return self.backend.predict_probability(frame, self.config.sample_rate)

    def process_frame(self, frame: np.ndarray) -> VadFrameResult:
        """Process a single audio frame and update the temporal state machine."""
        rms = self.compute_energy_rms(frame)
        prob = self.evaluate_frame_probability(frame)
        frame_ms = (len(frame) / self.config.sample_rate) * 1000.0

        is_prob_speech = prob >= self.config.speech_threshold
        is_utterance_boundary = False

        if is_prob_speech:
            self._consecutive_speech_ms += frame_ms
            self._consecutive_silence_ms = 0.0

            # Lock into SPEECH state if sustained speech threshold reached
            if self._state == VadState.SILENCE:
                if self._consecutive_speech_ms >= self.config.min_speech_duration_ms:
                    self._state = VadState.SPEECH
                    self._accumulated_speech_ms += self._consecutive_speech_ms
            else:
                self._accumulated_speech_ms += frame_ms
        else:
            self._consecutive_silence_ms += frame_ms
            self._consecutive_speech_ms = 0.0

            # Trigger boundary flush if in SPEECH state and silence hangover threshold exceeded
            if (
                self._state == VadState.SPEECH
                and self._consecutive_silence_ms >= self.config.min_silence_duration_ms
            ):
                self._state = VadState.SILENCE
                is_utterance_boundary = True
                self._consecutive_silence_ms = 0.0

        return VadFrameResult(
            is_speech=(self._state == VadState.SPEECH or is_prob_speech),
            speech_probability=prob,
            energy_rms=rms,
            is_utterance_boundary=is_utterance_boundary,
            accumulated_speech_ms=self._accumulated_speech_ms,
            state=self._state,
        )

    def process_chunk(self, chunk: np.ndarray) -> list[VadFrameResult]:
        """Process an arbitrary length audio chunk, preserving unaligned frame remainders."""
        results: list[VadFrameResult] = []
        if len(self._unprocessed_samples) > 0:
            flat = np.concatenate((self._unprocessed_samples, chunk.flatten().astype(np.float32)))
        else:
            flat = chunk.flatten().astype(np.float32)

        num_frames = len(flat) // self.frame_samples
        for i in range(num_frames):
            frame = flat[i * self.frame_samples : (i + 1) * self.frame_samples]
            results.append(self.process_frame(frame))

        self._unprocessed_samples = flat[num_frames * self.frame_samples :]
        return results

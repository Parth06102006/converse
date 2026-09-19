"""Real-time ASL vision perception engine integrating landmarks, normalization, buffer, and ST-GCN."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from asl_vision.landmarks import ExtractedLandmarks, LandmarkExtractor
from asl_vision.models.stgcn import STGCN
from asl_vision.normalization import LandmarkNormalizer, NormalizedFrame
from asl_vision.sliding_window import SlidingWindowBuffer, SlidingWindowOutput

# Top 100 conversational vocabulary items from WLASL benchmark
DEFAULT_WLASL_100_GLOSSES: tuple[str, ...] = (
    "book", "drink", "computer", "before", "chair", "go", "clothes", "dance",
    "all", "bad", "black", "hot", "thank you", "hello", "yes", "no", "please",
    "help", "school", "family", "friend", "love", "walk", "sleep", "eat",
    "water", "house", "car", "work", "play", "see", "think", "know", "want",
    "like", "good", "happy", "sad", "angry", "fine", "sorry", "name", "sign",
    "language", "learn", "study", "read", "write", "talk", "hear", "listen",
    "look", "find", "give", "take", "make", "buy", "pay", "money", "time",
    "day", "night", "week", "month", "year", "today", "tomorrow", "yesterday",
    "now", "later", "again", "stop", "finish", "start", "open", "close",
    "meet", "leave", "stay", "come", "wait", "tell", "ask", "answer",
    "remember", "forget", "feel", "cold", "warm", "tired", "hungry", "thirsty",
    "sick", "doctor", "hospital", "police", "help me", "understand", "different", "same",
)


@dataclass(frozen=True)
class SignDetection:
    """Represents a discrete candidate ASL sign gloss detection event.

    Matches packages/contracts/src/vision.ts SignDetection schema.
    """

    gloss: str
    confidence: float
    start_time_ms: float
    end_time_ms: float
    id: str | None = None
    duration_ms: float | None = None
    is_fingerspelled: bool = False
    hand_dominance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to serializable dictionary matching @converse/contracts."""
        d: dict[str, Any] = {
            "gloss": self.gloss,
            "confidence": float(self.confidence),
            "startTimeMs": float(self.start_time_ms),
            "endTimeMs": float(self.end_time_ms),
        }
        if self.id is not None:
            d["id"] = self.id
        if self.duration_ms is not None:
            d["durationMs"] = float(self.duration_ms)
        if self.is_fingerspelled:
            d["isFingerspelled"] = True
        if self.hand_dominance is not None:
            d["handDominance"] = self.hand_dominance
        return d



@dataclass
class EngineConfig:
    """Configuration options for ASLVisionEngine."""

    window_size: int = 30
    stride: int = 5
    min_confidence: float = 0.60
    min_window_landmark_confidence: float = 0.25
    min_detection_interval_ms: float = 500.0
    suppress_repeated_gloss: bool = True
    num_classes: int = 100
    num_nodes: int = 75
    device: str = "cpu"
    vocabulary: Sequence[str] | None = None
    use_onnx: bool = False
    onnx_path: str | Path | None = None


class ASLVisionEngine:
    """Unified real-time perception engine running landmark extraction, normalization, and ST-GCN inference."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        model: STGCN | torch.nn.Module | None = None,
        extractor: LandmarkExtractor | None = None,
    ) -> None:
        """Initialize the real-time ASL vision perception engine.

        Args:
            config: Engine configuration parameters.
            model: Optional pre-instantiated ST-GCN PyTorch model.
            extractor: Optional MediaPipe landmark extractor.
        """
        self.config = config or EngineConfig()

        # 1. Vocabulary resolution
        if self.config.vocabulary is not None:
            self.vocabulary = list(self.config.vocabulary)
        elif self.config.num_classes == 100:
            self.vocabulary = list(DEFAULT_WLASL_100_GLOSSES)
        else:
            self.vocabulary = [f"GLOSS_{i}" for i in range(self.config.num_classes)]

        # 2. Perception components
        self.normalizer = LandmarkNormalizer()
        include_face = self.config.num_nodes > 75
        self.buffer = SlidingWindowBuffer(
            window_size=self.config.window_size,
            stride=self.config.stride,
            confidence_threshold=self.config.min_window_landmark_confidence,
            include_pose=True,
            include_hands=True,
            include_face=include_face,
            emit_only_valid=False,
        )
        self.extractor = extractor

        # 3. Model setup (PyTorch or ONNX Runtime)
        self.device = torch.device(self.config.device)
        self.onnx_session: Any = None
        self.pytorch_model: torch.nn.Module | None = None
        self._onnx_input_name: str = ""
        self._onnx_output_name: str = ""

        if self.config.use_onnx and self.config.onnx_path is not None:
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(
                str(self.config.onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_name = self.onnx_session.get_inputs()[0].name
            self._onnx_output_name = self.onnx_session.get_outputs()[0].name
        elif model is not None:
            self.pytorch_model = model.to(self.device)
            self.pytorch_model.eval()
        else:
            self.pytorch_model = STGCN(
                in_channels=3,
                num_classes=len(self.vocabulary),
                num_nodes=self.config.num_nodes,
                temporal_window_size=self.config.window_size,
            ).to(self.device)
            self.pytorch_model.eval()

        # 4. State tracking
        self.last_detection: SignDetection | None = None
        self.last_detection_time_ms: float = -1.0
        self.frame_counter: int = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
    ) -> list[SignDetection]:
        """Extract landmarks from raw RGB camera frame and process through pipeline.

        Args:
            frame: RGB video frame as uint8 ndarray of shape (H, W, 3).
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            List of discrete SignDetection events (0 or 1 per step).
        """
        if self.extractor is None:
            raise RuntimeError(
                "LandmarkExtractor was not provided to ASLVisionEngine. "
                "Initialize with an extractor or call process_landmarks / process_normalized_frame directly."
            )

        extracted = self.extractor.extract(frame, timestamp_ms=timestamp_ms)
        return self.process_landmarks(extracted, timestamp_ms=timestamp_ms)

    def process_landmarks(
        self,
        extracted: ExtractedLandmarks,
        timestamp_ms: float,
    ) -> list[SignDetection]:
        """Normalize extracted landmarks and push into sliding window inference buffer.

        Args:
            extracted: Pre-extracted landmarks from MediaPipe.
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            List of discrete SignDetection events (0 or 1 per step).
        """
        if extracted.pose is None:
            return []

        norm_frame = self.normalizer.normalize(extracted, timestamp_ms=timestamp_ms)
        return self.process_normalized_frame(norm_frame, timestamp_ms=timestamp_ms)

    def process_normalized_frame(
        self,
        norm_frame: NormalizedFrame,
        timestamp_ms: float,
    ) -> list[SignDetection]:
        """Push normalized coordinate frame into sliding window buffer and run model when window triggers.

        Args:
            norm_frame: Mathematically normalized frame.
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            List of discrete SignDetection events (0 or 1 per step).
        """
        self.frame_counter += 1
        window_out = self.buffer.add_frame(norm_frame)

        if window_out is None:
            return []

        return self._infer_window(window_out)

    def _infer_window(
        self,
        window_out: SlidingWindowOutput,
    ) -> list[SignDetection]:
        """Execute neural model on sliding window tensor and apply confidence gating and debouncing.

        Args:
            window_out: Prepared (1, 3, 30, V) sliding window tensor with metadata.

        Returns:
            List containing candidate SignDetection if confidence and cooldown thresholds are satisfied.
        """
        # Gating 1: Landmark visibility confidence in window
        if window_out.confidence < self.config.min_window_landmark_confidence:
            return []

        # Neural forward pass
        if self.onnx_session is not None:
            tensor_np = window_out.tensor.cpu().numpy().astype(np.float32)
            raw_logits = self.onnx_session.run(
                [self._onnx_output_name],
                {self._onnx_input_name: tensor_np},
            )[0]
            # Softmax calculation
            shift_logits = raw_logits - np.max(raw_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shift_logits)
            probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).squeeze(0)
        elif self.pytorch_model is not None:
            with torch.no_grad():
                tensor_torch = window_out.tensor.to(self.device)
                logits = self.pytorch_model(tensor_torch)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        else:
            return []

        # Find top predicted class
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        # Gating 2: Minimum model confidence threshold
        if confidence < self.config.min_confidence:
            return []

        # Gating 3: Debounce temporal interval
        time_since_last = window_out.end_timestamp_ms - self.last_detection_time_ms
        if self.last_detection_time_ms >= 0 and time_since_last < self.config.min_detection_interval_ms:
            return []

        # Resolve gloss label
        gloss = (
            self.vocabulary[top_idx]
            if top_idx < len(self.vocabulary)
            else f"GLOSS_{top_idx}"
        )

        # Gating 4: Suppress immediate identical gloss repetition
        if (
            self.config.suppress_repeated_gloss
            and self.last_detection is not None
            and self.last_detection.gloss == gloss
            and time_since_last < (2.0 * self.config.min_detection_interval_ms)
        ):
            return []

        detection = SignDetection(
            gloss=gloss,
            confidence=confidence,
            start_time_ms=window_out.start_timestamp_ms,
            end_time_ms=window_out.end_timestamp_ms,
        )

        self.last_detection = detection
        self.last_detection_time_ms = window_out.end_timestamp_ms
        return [detection]

    def reset(self) -> None:
        """Reset internal buffer, normalizer filters, and detection history."""
        self.buffer.clear()
        self.normalizer.reset()
        self.last_detection = None
        self.last_detection_time_ms = -1.0
        self.frame_counter = 0

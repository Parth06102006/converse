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
from asl_vision.models.tgcn_wlasl import (
    WLASL_100_GLOSSES,
    TGCNModel,
)
from asl_vision.normalization import (
    POSE_LEFT_HIP,
    POSE_LEFT_WRIST,
    POSE_RIGHT_HIP,
    POSE_RIGHT_WRIST,
    LandmarkNormalizer,
    NormalizedFrame,
)
from asl_vision.sliding_window import (
    SlidingWindowBuffer,
    SlidingWindowOutput,
    compute_temporal_variance,
)

# Canonical 100 conversational vocabulary items from WLASL benchmark in order
DEFAULT_WLASL_100_GLOSSES: tuple[str, ...] = tuple(g.lower() for g in WLASL_100_GLOSSES)

DEFAULT_CHECKPOINT_PATHS: tuple[Path, ...] = (
    Path("ml/asl-vision/models/tgcn_asl100.bin"),
    Path("models/checkpoints/tgcn_asl100.bin"),
)


def resolve_checkpoint_path(path: str | Path | None = None) -> Path | None:
    """Resolve model checkpoint path with fallback resolution.

    Checks:
    1. Explicit path passed by caller (if provided)
    2. ml/asl-vision/models/tgcn_asl100.bin
    3. models/checkpoints/tgcn_asl100.bin
    4. Relative to workspace directory
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p.resolve()
        for base in [Path.cwd(), Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
            cand = base / path
            if cand.is_file():
                return cand.resolve()
        # Explicit path was requested but not found. If it is just the
        # canonical default filename, allow fallback to known locations
        # (handles CWD differences). Otherwise return None so callers
        # can fall back to the randomly-initialized STGCN.
        if p.name != "tgcn_asl100.bin":
            return None

    candidates = [
        Path("ml/asl-vision/models/tgcn_asl100.bin"),
        Path("models/checkpoints/tgcn_asl100.bin"),
        Path(__file__).resolve().parent.parent.parent / "models" / "tgcn_asl100.bin",
        Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints" / "tgcn_asl100.bin",
        Path("models/tgcn_asl100.bin"),
        Path("checkpoints/tgcn_asl100.bin"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


class RestingPoseDetector:
    """Detects when wrists or hands remain stationary or resting for sustained duration (> 300ms).

    Suppresses spurious sign detections during resting periods (e.g. hands on table,
    wrists stationary, or hands held still at chin).
    """

    def __init__(
        self,
        stationary_duration_ms: float = 300.0,
        wrist_movement_threshold: float = 0.012,
        chest_level_margin: float = 0.6,
    ) -> None:
        self.stationary_duration_ms = stationary_duration_ms
        self.wrist_movement_threshold = wrist_movement_threshold
        self.chest_level_margin = chest_level_margin
        self.last_movement_timestamp_ms: float = -1.0
        self._prev_wrist_coords: np.ndarray | None = None
        self._prev_timestamp_ms: float = -1.0

    def reset(self) -> None:
        """Reset internal tracking state."""
        self.last_movement_timestamp_ms = -1.0
        self._prev_wrist_coords = None
        self._prev_timestamp_ms = -1.0

    def is_hands_below_chest(self, frame: NormalizedFrame) -> bool:
        """Check whether hands/wrists are resting below chest level or inactive in signing space."""
        if not frame.is_left_hand_visible and not frame.is_right_hand_visible:
            return True

        if frame.pose is None or len(frame.pose) < 17:
            return False

        lw_y = float(frame.pose[POSE_LEFT_WRIST, 1])
        rw_y = float(frame.pose[POSE_RIGHT_WRIST, 1])

        if len(frame.pose) > 24:
            lh_y = float(frame.pose[POSE_LEFT_HIP, 1])
            rh_y = float(frame.pose[POSE_RIGHT_HIP, 1])
            hip_y = (lh_y + rh_y) / 2.0
            chest_y = hip_y * self.chest_level_margin if hip_y > 0.5 else 0.8
        else:
            chest_y = 0.8

        return (lw_y > chest_y) and (rw_y > chest_y)

    def update(self, frame: NormalizedFrame, timestamp_ms: float) -> bool:
        """Process a frame and determine if signer is in resting / stationary pose.

        Returns:
            True if resting pose is detected (emission should be suppressed), False otherwise.
        """
        if self.is_hands_below_chest(frame):
            if self.last_movement_timestamp_ms < 0:
                return True
            elapsed = timestamp_ms - self.last_movement_timestamp_ms
            return elapsed > self.stationary_duration_ms

        if frame.pose is not None and len(frame.pose) >= 17:
            curr_wrists = frame.pose[[POSE_LEFT_WRIST, POSE_RIGHT_WRIST], :2].copy()
            if self._prev_wrist_coords is not None and self._prev_timestamp_ms >= 0:
                disp = np.linalg.norm(curr_wrists - self._prev_wrist_coords, axis=-1)
                max_disp = float(np.max(disp))
                if max_disp >= self.wrist_movement_threshold:
                    self.last_movement_timestamp_ms = timestamp_ms
            else:
                self.last_movement_timestamp_ms = timestamp_ms

            self._prev_wrist_coords = curr_wrists
            self._prev_timestamp_ms = timestamp_ms
        else:
            self.last_movement_timestamp_ms = timestamp_ms

        if self.last_movement_timestamp_ms < 0:
            return True

        elapsed = timestamp_ms - self.last_movement_timestamp_ms
        return elapsed > self.stationary_duration_ms


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
    checkpoint_path: str | Path | None = None
    motion_threshold: float = 0.015
    idle_suppression_ms: float = 300.0


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
        self.resting_detector = RestingPoseDetector(
            stationary_duration_ms=self.config.idle_suppression_ms,
        )

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
            ckpt_path = resolve_checkpoint_path(self.config.checkpoint_path)
            if ckpt_path is not None and ckpt_path.is_file():
                state_dict = torch.load(str(ckpt_path), map_location=self.device)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                is_tgcn = any(k.startswith(("gc1.", "gcbs.")) for k in state_dict)
                num_classes_in_ckpt = (
                    state_dict["fc_out.bias"].shape[0]
                    if "fc_out.bias" in state_dict
                    else (state_dict["fc.bias"].shape[0] if "fc.bias" in state_dict else None)
                )
                classes_match = num_classes_in_ckpt is None or num_classes_in_ckpt == len(self.vocabulary)

                if is_tgcn and classes_match:
                    self.pytorch_model = TGCNModel(
                        input_feature=100,
                        hidden_feature=64,
                        num_class=len(self.vocabulary),
                        p_dropout=0.3,
                        num_stage=20,
                    ).to(self.device)
                    self.pytorch_model.load_state_dict(state_dict, strict=True)
                    self.pytorch_model.eval()
                elif not is_tgcn and classes_match:
                    self.pytorch_model = STGCN(
                        in_channels=3,
                        num_classes=len(self.vocabulary),
                        num_nodes=self.config.num_nodes,
                        temporal_window_size=self.config.window_size,
                    ).to(self.device)
                    self.pytorch_model.load_state_dict(state_dict, strict=False)
                    self.pytorch_model.eval()
                else:
                    self.pytorch_model = STGCN(
                        in_channels=3,
                        num_classes=len(self.vocabulary),
                        num_nodes=self.config.num_nodes,
                        temporal_window_size=self.config.window_size,
                    ).to(self.device)
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

        # Gating 2: Temporal motion energy gating (variance >= motion_threshold)
        # Suppress static false positives (e.g. idle 'orange' predictions when hands rest)
        motion_var = (
            window_out.variance
            if hasattr(window_out, "variance") and window_out.variance > 0
            else compute_temporal_variance(tensor=window_out.tensor)
        )
        if self.config.motion_threshold > 0.0 and motion_var < self.config.motion_threshold:
            return []

        # Gating 3: Resting pose detector (wrists stationary > 300ms or below chest)
        if (
            self.config.idle_suppression_ms > 0.0
            and len(window_out.frames) > 0
            and self.resting_detector.update(window_out.frames[-1], window_out.end_timestamp_ms)
        ):
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

        # Gating 4: Minimum model confidence threshold
        if confidence < self.config.min_confidence:
            return []

        # Gating 5: Debounce temporal interval
        time_since_last = window_out.end_timestamp_ms - self.last_detection_time_ms
        if self.last_detection_time_ms >= 0 and time_since_last < self.config.min_detection_interval_ms:
            return []

        # Resolve gloss label
        gloss = (
            self.vocabulary[top_idx]
            if top_idx < len(self.vocabulary)
            else f"GLOSS_{top_idx}"
        )

        # Gating 6: Suppress immediate identical gloss repetition
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
        """Reset internal buffer, normalizer filters, resting detector, and detection history."""
        self.buffer.clear()
        self.normalizer.reset()
        self.resting_detector.reset()
        self.last_detection = None
        self.last_detection_time_ms = -1.0
        self.frame_counter = 0

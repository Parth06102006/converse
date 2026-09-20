"""MediaPipe Tasks Vision / Holistic 3D landmark extractor wrapper.

Ingests webcam video frames, extracts 21 hand landmarks per hand, 33 body pose
landmarks, and facial contour landmarks for Non-Manual Markers (NMM), and integrates
with LandmarkNormalizer for scale and translation invariant representations.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np

from asl_vision.normalization import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    LandmarkNormalizer,
    NormalizedFrame,
)

# Non-Manual Marker (NMM) facial contour keypoint indices (MediaPipe Face Mesh topology)
# Eyebrow contours (used for Wh-questions furrowed brows, Yes/No questions raised brows)
NMM_LEFT_EYEBROW: tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
NMM_RIGHT_EYEBROW: tuple[int, ...] = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
NMM_EYEBROW_INDICES: tuple[int, ...] = tuple(
    sorted(set(NMM_LEFT_EYEBROW + NMM_RIGHT_EYEBROW))
)

# Lips and mouth contours (used for mouth morphemes e.g. "th", "mm", "cha", "puff")
NMM_LIPS_OUTER: tuple[int, ...] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
)
NMM_LIPS_INNER: tuple[int, ...] = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
)
NMM_LIP_INDICES: tuple[int, ...] = tuple(
    sorted(set(NMM_LIPS_OUTER + NMM_LIPS_INNER))
)

# Midline and nose bridge anchors for head pose orientation
NMM_MIDLINE_INDICES: tuple[int, ...] = (1, 4, 6, 168)

# Combined subset of facial contour landmarks for Non-Manual Markers
FACIAL_CONTOUR_INDICES: tuple[int, ...] = tuple(
    sorted(set(NMM_EYEBROW_INDICES + NMM_LIP_INDICES + NMM_MIDLINE_INDICES))
)


@dataclass(frozen=True)
class ExtractedLandmarks:
    """Raw 3D skeletal and facial landmarks extracted from a video frame."""

    pose: np.ndarray | None  # Shape (33, 4) [x, y, z, visibility] or (33, 3)
    left_hand: np.ndarray | None  # Shape (21, 3) [x, y, z]
    right_hand: np.ndarray | None  # Shape (21, 3) [x, y, z]
    face: np.ndarray | None  # Full face mesh shape (N_face, 3)
    facial_contours: np.ndarray | None  # NMM contour points shape (N_contours, 3)
    face_blendshapes: dict[str, float] | None  # Blendshape weights (ARKit compatible)
    pose_world: np.ndarray | None  # World coordinates in meters shape (33, 3)
    timestamp_ms: float
    confidence: float
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0
    frame_width: int | None = None
    frame_height: int | None = None


def _generate_synthetic_extracted_landmarks(
    timestamp_ms: float = 0.0,
    has_left_hand: bool = True,
    has_right_hand: bool = True,
    has_face: bool = True,
) -> ExtractedLandmarks:
    """Generate realistic synthetic landmark structures for headless testing."""
    # Synthetic pose (33 keypoints with shoulder anchors)
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[POSE_LEFT_SHOULDER] = np.array([-0.2, 0.4, 0.0, 0.95], dtype=np.float32)
    pose[POSE_RIGHT_SHOULDER] = np.array([0.2, 0.4, 0.0, 0.95], dtype=np.float32)
    for i in range(33):
        if i not in (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER):
            pose[i] = np.array([i * 0.01, i * 0.02, 0.0, 0.9], dtype=np.float32)

    # Synthetic left hand (21 keypoints with wrist and middle MCP)
    left_hand: np.ndarray | None = None
    if has_left_hand:
        left_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand[HAND_WRIST] = np.array([-0.2, 0.1, 0.0], dtype=np.float32)
        left_hand[HAND_MIDDLE_MCP] = np.array([-0.2, 0.2, 0.0], dtype=np.float32)
        for i in range(21):
            if i not in (HAND_WRIST, HAND_MIDDLE_MCP):
                left_hand[i] = np.array([-0.2 + i * 0.005, 0.1 + i * 0.01, 0.0], dtype=np.float32)

    # Synthetic right hand (21 keypoints with wrist and middle MCP)
    right_hand: np.ndarray | None = None
    if has_right_hand:
        right_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand[HAND_WRIST] = np.array([0.2, 0.1, 0.0], dtype=np.float32)
        right_hand[HAND_MIDDLE_MCP] = np.array([0.2, 0.2, 0.0], dtype=np.float32)
        for i in range(21):
            if i not in (HAND_WRIST, HAND_MIDDLE_MCP):
                right_hand[i] = np.array([0.2 + i * 0.005, 0.1 + i * 0.01, 0.0], dtype=np.float32)

    # Synthetic facial contours
    facial_contours: np.ndarray | None = None
    face_mesh: np.ndarray | None = None
    blendshapes: dict[str, float] | None = None
    if has_face:
        n_contours = len(FACIAL_CONTOUR_INDICES)
        facial_contours = np.zeros((n_contours, 3), dtype=np.float32)
        for idx in range(n_contours):
            facial_contours[idx] = np.array([0.0, 0.5 + idx * 0.002, 0.0], dtype=np.float32)
        face_mesh = np.zeros((468, 3), dtype=np.float32)
        blendshapes = {
            "browDownLeft": 0.0,
            "browDownRight": 0.0,
            "browInnerUp": 0.0,
            "jawOpen": 0.0,
            "mouthPucker": 0.0,
        }

    return ExtractedLandmarks(
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face_mesh,
        facial_contours=facial_contours,
        face_blendshapes=blendshapes,
        pose_world=pose[:, :3].copy(),
        timestamp_ms=timestamp_ms,
        confidence=0.9,
        left_hand_confidence=0.88 if has_left_hand else 0.0,
        right_hand_confidence=0.88 if has_right_hand else 0.0,
    )


def assign_hands_by_geometry(
    candidates: list[tuple[np.ndarray, float]],
    pose: np.ndarray | None = None,
    existing_left: np.ndarray | None = None,
    existing_right: np.ndarray | None = None,
    existing_left_conf: float = 0.0,
    existing_right_conf: float = 0.0,
    min_separation: float = 0.08,
) -> tuple[np.ndarray | None, float, np.ndarray | None, float]:
    """Assign candidate hand detections to left and right slots using anatomical geometry.

    Uses pose wrist landmarks (MediaPipe pose[15] LWrist and pose[16] RWrist) or spatial
    partitioning to prevent single-hand duplication and eliminate handedness inversion artifacts.

    Returns:
        (left_hand, left_conf, right_hand, right_conf)
    """
    left_hand = existing_left
    left_conf = existing_left_conf
    right_hand = existing_right
    right_conf = existing_right_conf

    # Filter out candidate hands that are physically overlapping with already established hands
    valid_candidates: list[tuple[np.ndarray, float]] = []
    for arr, conf in candidates:
        if arr is None or len(arr) < 21:
            continue
        wrist = arr[0, :2]
        if left_hand is not None and float(np.linalg.norm(wrist - left_hand[0, :2])) < min_separation:
            continue
        if right_hand is not None and float(np.linalg.norm(wrist - right_hand[0, :2])) < min_separation:
            continue
        valid_candidates.append((arr, conf))

    if not valid_candidates:
        return left_hand, left_conf, right_hand, right_conf

    # Determine signer's midline (center x) in unmirrored camera coordinates
    mid_x = 0.5
    if pose is not None and len(pose) > 12:
        has_s11 = not (float(pose[11, 0]) == 0.0 and float(pose[11, 1]) == 0.0) and not np.isnan(pose[11, 0])
        has_s12 = not (float(pose[12, 0]) == 0.0 and float(pose[12, 1]) == 0.0) and not np.isnan(pose[12, 0])
        if has_s11 and has_s12:
            mid_x = float((pose[11, 0] + pose[12, 0]) / 2.0)
        elif float(pose[0, 0]) != 0.0 and not np.isnan(pose[0, 0]):
            mid_x = float(pose[0, 0])
    elif pose is not None and len(pose) > 0 and float(pose[0, 0]) != 0.0 and not np.isnan(pose[0, 0]):
        mid_x = float(pose[0, 0])

    has_lwrist = (
        pose is not None
        and len(pose) > 15
        and not (float(pose[15, 0]) == 0.0 and float(pose[15, 1]) == 0.0)
        and not np.isnan(pose[15, 0])
    )
    has_rwrist = (
        pose is not None
        and len(pose) > 16
        and not (float(pose[16, 0]) == 0.0 and float(pose[16, 1]) == 0.0)
        and not np.isnan(pose[16, 0])
    )
    has_pose_wrists = has_lwrist and has_rwrist

    # Case 1: Neither hand established yet
    if left_hand is None and right_hand is None:
        if len(valid_candidates) >= 2:
            if has_pose_wrists:
                c0, c1 = valid_candidates[0], valid_candidates[1]
                cost_a = float(np.linalg.norm(c0[0][0, :2] - pose[16, :2])) + float(np.linalg.norm(c1[0][0, :2] - pose[15, :2]))
                cost_b = float(np.linalg.norm(c0[0][0, :2] - pose[15, :2])) + float(np.linalg.norm(c1[0][0, :2] - pose[16, :2]))
                if cost_a <= cost_b:
                    right_hand, right_conf = c0
                    left_hand, left_conf = c1
                else:
                    left_hand, left_conf = c0
                    right_hand, right_conf = c1
            else:
                sorted_c = sorted(valid_candidates, key=lambda c: float(c[0][0, 0]))
                right_hand, right_conf = sorted_c[0]
                left_hand, left_conf = sorted_c[-1]
        elif len(valid_candidates) == 1:
            c0 = valid_candidates[0]
            wrist = c0[0][0, :2]
            if has_pose_wrists:
                d_r = float(np.linalg.norm(wrist - pose[16, :2]))
                d_l = float(np.linalg.norm(wrist - pose[15, :2]))
                if d_r <= d_l:
                    right_hand, right_conf = c0
                else:
                    left_hand, left_conf = c0
            elif has_rwrist:
                d_r = float(np.linalg.norm(wrist - pose[16, :2]))
                if d_r < 0.25 or float(wrist[0]) < mid_x:
                    right_hand, right_conf = c0
                else:
                    left_hand, left_conf = c0
            elif has_lwrist:
                d_l = float(np.linalg.norm(wrist - pose[15, :2]))
                if d_l < 0.25 or float(wrist[0]) >= mid_x:
                    left_hand, left_conf = c0
                else:
                    right_hand, right_conf = c0
            else:
                if float(wrist[0]) < mid_x:
                    right_hand, right_conf = c0
                else:
                    left_hand, left_conf = c0

    # Case 2: Only right hand established; fill left hand only if candidate genuinely belongs to left side
    elif left_hand is None and right_hand is not None:
        left_candidates: list[tuple[tuple[np.ndarray, float], float]] = []
        for c in valid_candidates:
            wrist = c[0][0, :2]
            if has_pose_wrists:
                d_l = float(np.linalg.norm(wrist - pose[15, :2]))
                d_r = float(np.linalg.norm(wrist - pose[16, :2]))
                if d_l < d_r:
                    left_candidates.append((c, d_l))
            elif has_lwrist:
                d_l = float(np.linalg.norm(wrist - pose[15, :2]))
                if d_l < 0.35 or float(wrist[0]) >= mid_x:
                    left_candidates.append((c, d_l))
            else:
                if float(wrist[0]) >= mid_x:
                    left_candidates.append((c, abs(float(wrist[0]) - mid_x)))
        if left_candidates:
            left_candidates.sort(key=lambda x: x[1])
            left_hand, left_conf = left_candidates[0][0]

    # Case 3: Only left hand established; fill right hand only if candidate genuinely belongs to right side
    elif right_hand is None and left_hand is not None:
        right_candidates: list[tuple[tuple[np.ndarray, float], float]] = []
        for c in valid_candidates:
            wrist = c[0][0, :2]
            if has_pose_wrists:
                d_r = float(np.linalg.norm(wrist - pose[16, :2]))
                d_l = float(np.linalg.norm(wrist - pose[15, :2]))
                if d_r < d_l:
                    right_candidates.append((c, d_r))
            elif has_rwrist:
                d_r = float(np.linalg.norm(wrist - pose[16, :2]))
                if d_r < 0.35 or float(wrist[0]) < mid_x:
                    right_candidates.append((c, d_r))
            else:
                if float(wrist[0]) < mid_x:
                    right_candidates.append((c, abs(float(wrist[0]) - mid_x)))
        if right_candidates:
            right_candidates.sort(key=lambda x: x[1])
            right_hand, right_conf = right_candidates[0][0]

    return left_hand, left_conf, right_hand, right_conf


class LandmarkExtractor:
    """MediaPipe Tasks Vision Holistic 3D landmark extractor wrapper.

    Extracts bilateral hand landmarks (21 per hand), body pose (33 landmarks),
    and facial contour keypoints for Non-Manual Markers (NMMs).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_buffer: bytes | None = None,
        normalizer: LandmarkNormalizer | None = None,
        min_detection_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
        min_pose_confidence: float = 0.5,
        min_face_confidence: float = 0.5,
        running_mode: str = "IMAGE",
        output_face_blendshapes: bool = True,
        use_mock: bool = False,
        backend: Callable[[np.ndarray, float], ExtractedLandmarks] | None = None,
        hand_model_path: str | Path | None = None,
    ) -> None:
        """Initialize LandmarkExtractor.

        Args:
            model_path: Path to MediaPipe holistic_landmarker.task model file.
            model_buffer: Binary content of holistic_landmarker.task model.
            normalizer: Optional LandmarkNormalizer instance. If None, instantiates a default.
            min_detection_confidence: Minimum detection confidence score.
            min_hand_confidence: Minimum hand landmarks confidence score.
            min_pose_confidence: Minimum pose landmarks confidence score.
            min_face_confidence: Minimum face landmarks confidence score.
            running_mode: 'IMAGE' or 'VIDEO'.
            output_face_blendshapes: Whether to output facial blendshape weights for NMMs.
            use_mock: If True, uses synthetic landmark generator for testing without model files.
            backend: Optional custom extraction callback taking (frame, timestamp_ms).
            hand_model_path: Optional path to MediaPipe gesture_recognizer.task for robust hand fallback.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_hand_confidence = min_hand_confidence
        self.min_pose_confidence = min_pose_confidence
        self.min_face_confidence = min_face_confidence

        running_mode_upper = running_mode.upper()
        if running_mode_upper not in ("IMAGE", "VIDEO", "LIVE_STREAM"):
            raise ValueError(
                f"Invalid running_mode '{running_mode}'. Must be 'IMAGE', 'VIDEO', or 'LIVE_STREAM'."
            )
        self.running_mode = running_mode_upper
        self.output_face_blendshapes = output_face_blendshapes
        self.normalizer = normalizer if normalizer is not None else LandmarkNormalizer()
        self.use_mock = use_mock
        self._custom_backend = backend

        self._landmarker: Any = None
        self._hand_recognizer: Any = None
        self._mock_result_override: ExtractedLandmarks | None = None
        self._prev_video_timestamp_ms: int | None = None

        if self._custom_backend is not None or self.use_mock:
            return

        # Attempt to initialize MediaPipe HolisticLandmarker if model is supplied
        if model_path is not None or model_buffer is not None:
            if model_path is not None:
                path_obj = Path(model_path)
                if not path_obj.is_file():
                    raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")
            self._init_mediapipe(model_path, model_buffer)
            self._init_hand_fallback(hand_model_path)
        else:
            raise ValueError(
                "No model_path or model_buffer provided and use_mock=False. "
                "Specify a model file or enable use_mock=True for synthetic inference."
            )

    def _init_mediapipe(
        self, model_path: str | Path | None, model_buffer: bytes | None
    ) -> None:
        """Initialize MediaPipe Tasks Vision HolisticLandmarker."""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path) if model_path is not None else None,
            model_asset_buffer=model_buffer,
        )

        task_running_mode = (
            mp_vision.RunningMode.VIDEO
            if self.running_mode == "VIDEO"
            else mp_vision.RunningMode.LIVE_STREAM
            if self.running_mode == "LIVE_STREAM"
            else mp_vision.RunningMode.IMAGE
        )

        options = mp_vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=task_running_mode,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_landmarks_confidence=self.min_face_confidence,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_landmarks_confidence=self.min_pose_confidence,
            min_hand_landmarks_confidence=self.min_hand_confidence,
            output_face_blendshapes=self.output_face_blendshapes,
        )

        self._landmarker = mp_vision.HolisticLandmarker.create_from_options(options)

    def _init_hand_fallback(self, hand_model_path: str | Path | None) -> None:
        """Initialize direct hand landmarking fallback model if available."""
        if hand_model_path is None:
            return

        target_path = Path(hand_model_path)
        if not target_path.is_file():
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        with contextlib.suppress(Exception):
            base_options = mp_python.BaseOptions(model_asset_path=str(target_path))
            options = mp_vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=self.min_hand_confidence,
                min_hand_presence_confidence=self.min_hand_confidence,
                min_tracking_confidence=self.min_hand_confidence,
            )
            self._hand_recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def set_mock_result(self, result: ExtractedLandmarks | None) -> None:
        """Set an explicit ExtractedLandmarks result to be returned in mock mode."""
        self._mock_result_override = result

    def validate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Validate input webcam frame dimensions, dtype, and shape.

        Args:
            frame: Input video frame array.

        Returns:
            Validated uint8 RGB frame array.

        Raises:
            ValueError: If frame is None, empty, or has incorrect channels.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Input frame must be a numpy.ndarray, got {type(frame)}.")

        if frame.ndim != 3:
            raise ValueError(
                f"Input frame must have 3 dimensions (height, width, channels), got shape {frame.shape}."
            )

        height, width, channels = frame.shape
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid frame dimensions: height={height}, width={width}.")

        if channels != 3:
            raise ValueError(
                f"Input frame must have 3 color channels (RGB), got {channels} channels."
            )

        # Normalize dtype to uint8
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        if np.issubdtype(frame.dtype, np.floating):
            # Scale float values to [0, 255] if values are in [0, 1] range
            if frame.max() <= 1.0:
                return np.ascontiguousarray(
                    np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                )
            return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))

        return np.ascontiguousarray(frame.astype(np.uint8))

    def extract(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
        is_bgr: bool = False,
    ) -> ExtractedLandmarks:
        """Extract 3D landmarks for pose, bilateral hands, and facial contours.

        Args:
            frame: Input image array of shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.
            is_bgr: True if the input frame is in BGR format (e.g., from OpenCV).

        Returns:
            ExtractedLandmarks containing raw coordinate arrays.
        """
        valid_frame = self.validate_frame(frame)

        if is_bgr:
            valid_frame = np.ascontiguousarray(
                cv2.cvtColor(valid_frame, cv2.COLOR_BGR2RGB)
            )

        # Custom backend delegation
        if self._custom_backend is not None:
            return self._custom_backend(valid_frame, timestamp_ms)

        # Mock generator execution
        if self.use_mock or self._landmarker is None:
            if self._mock_result_override is not None:
                return self._mock_result_override
            return _generate_synthetic_extracted_landmarks(timestamp_ms=timestamp_ms)

        # MediaPipe Tasks Vision inference
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=valid_frame)

        if self.running_mode == "VIDEO":
            ts_int = int(timestamp_ms)
            if (
                self._prev_video_timestamp_ms is not None
                and ts_int <= self._prev_video_timestamp_ms
            ):
                raise ValueError(
                    f"Timestamp must be strictly monotonically increasing in VIDEO mode. "
                    f"Got {ts_int}ms <= previous {self._prev_video_timestamp_ms}ms."
                )
            self._prev_video_timestamp_ms = ts_int
            result = self._landmarker.detect_for_video(mp_image, ts_int)
        elif self.running_mode == "LIVE_STREAM":
            result = self._landmarker.detect_async(mp_image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(mp_image)

        return self._parse_mediapipe_result(result, timestamp_ms, mp_image=mp_image)

    def _parse_mediapipe_result(
        self, result: Any, timestamp_ms: float, mp_image: Any = None
    ) -> ExtractedLandmarks:
        """Parse MediaPipe HolisticLandmarkerResult into ExtractedLandmarks."""
        # 1. Pose landmarks (33, 4)
        pose_arr: np.ndarray | None = None
        if result.pose_landmarks and len(result.pose_landmarks) >= 33:
            pose_arr = np.array(
                [
                    [
                        lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                        lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                        lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                        (
                            lm.visibility
                            if lm.visibility is not None and not np.isnan(lm.visibility)
                            else 1.0
                        ),
                    ]
                    for lm in result.pose_landmarks[:33]
                ],
                dtype=np.float32,
            )

        # 2. Pose world landmarks (33, 3)
        pose_world_arr: np.ndarray | None = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) >= 33:
            pose_world_arr = np.array(
                [
                    [
                        lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                        lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                        lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                    ]
                    for lm in result.pose_world_landmarks[:33]
                ],
                dtype=np.float32,
            )

        # 3. Left hand landmarks (21, 3)
        left_hand_arr: np.ndarray | None = None
        left_hand_conf = 0.0
        if result.left_hand_landmarks and len(result.left_hand_landmarks) >= 21:
            left_hand_arr = np.array(
                [
                    [
                        lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                        lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                        lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                    ]
                    for lm in result.left_hand_landmarks[:21]
                ],
                dtype=np.float32,
            )
            scores = [
                lm.presence
                if hasattr(lm, "presence") and lm.presence is not None
                else (
                    lm.visibility
                    if hasattr(lm, "visibility") and lm.visibility is not None
                    else 1.0
                )
                for lm in result.left_hand_landmarks[:21]
            ]
            left_hand_conf = float(np.mean(scores)) if scores else 1.0

        # 4. Right hand landmarks (21, 3)
        right_hand_arr: np.ndarray | None = None
        right_hand_conf = 0.0
        if result.right_hand_landmarks and len(result.right_hand_landmarks) >= 21:
            right_hand_arr = np.array(
                [
                    [
                        lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                        lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                        lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                    ]
                    for lm in result.right_hand_landmarks[:21]
                ],
                dtype=np.float32,
            )
            scores = [
                lm.presence
                if hasattr(lm, "presence") and lm.presence is not None
                else (
                    lm.visibility
                    if hasattr(lm, "visibility") and lm.visibility is not None
                    else 1.0
                )
                for lm in result.right_hand_landmarks[:21]
            ]
            right_hand_conf = float(np.mean(scores)) if scores else 1.0

        # Fallback to direct hand tracking if Holistic missed hands
        if (left_hand_arr is None or right_hand_arr is None) and self._hand_recognizer is not None and mp_image is not None:
            with contextlib.suppress(Exception):
                hand_res = self._hand_recognizer.recognize(mp_image)
                if hand_res and hand_res.hand_landmarks:
                    candidates: list[tuple[np.ndarray, float]] = []
                    for idx, hand_lms in enumerate(hand_res.hand_landmarks):
                        arr = np.array(
                            [
                                [
                                    lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                                    lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                                    lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                                ]
                                for lm in hand_lms[:21]
                            ],
                            dtype=np.float32,
                        )
                        score = 0.8
                        if idx < len(hand_res.handedness) and hand_res.handedness[idx]:
                            score = float(hand_res.handedness[idx][0].score)
                        candidates.append((arr, score))

                    left_hand_arr, left_hand_conf, right_hand_arr, right_hand_conf = (
                        assign_hands_by_geometry(
                            candidates=candidates,
                            pose=pose_arr,
                            existing_left=left_hand_arr,
                            existing_right=right_hand_arr,
                            existing_left_conf=left_hand_conf,
                            existing_right_conf=right_hand_conf,
                        )
                    )

        # 5. Face mesh and NMM contours
        face_arr: np.ndarray | None = None
        facial_contours_arr: np.ndarray | None = None
        raw_face_lms = result.face_landmarks
        if raw_face_lms and len(raw_face_lms) > 0 and isinstance(raw_face_lms[0], list):
            raw_face_lms = raw_face_lms[0]

        if raw_face_lms and len(raw_face_lms) > 0:
            face_arr = np.array(
                [
                    [
                        lm.x if lm.x is not None and not np.isnan(lm.x) else 0.0,
                        lm.y if lm.y is not None and not np.isnan(lm.y) else 0.0,
                        lm.z if lm.z is not None and not np.isnan(lm.z) else 0.0,
                    ]
                    for lm in raw_face_lms
                ],
                dtype=np.float32,
            )
            valid_indices = [idx for idx in FACIAL_CONTOUR_INDICES if idx < len(face_arr)]
            if valid_indices:
                facial_contours_arr = face_arr[valid_indices]

        # 6. Face blendshapes for NMMs
        blendshapes: dict[str, float] | None = None
        if result.face_blendshapes:
            raw_blendshapes = result.face_blendshapes
            if len(raw_blendshapes) > 0 and isinstance(raw_blendshapes[0], list):
                raw_blendshapes = raw_blendshapes[0]
            blendshapes = {c.category_name: float(c.score) for c in raw_blendshapes}

        confidence = (
            float(np.mean(pose_arr[:, 3]))
            if pose_arr is not None and pose_arr.shape[1] > 3
            else (1.0 if pose_arr is not None else 0.0)
        )

        return ExtractedLandmarks(
            pose=pose_arr,
            left_hand=left_hand_arr,
            right_hand=right_hand_arr,
            face=face_arr,
            facial_contours=facial_contours_arr,
            face_blendshapes=blendshapes,
            pose_world=pose_world_arr,
            timestamp_ms=timestamp_ms,
            confidence=confidence,
            left_hand_confidence=left_hand_conf,
            right_hand_confidence=right_hand_conf,
            frame_width=getattr(mp_image, "width", None),
            frame_height=getattr(mp_image, "height", None),
        )

    def extract_and_normalize(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
        max_extrapolation_frames: int = 2,
        is_bgr: bool = False,
    ) -> NormalizedFrame:
        """Extract landmarks and normalize coordinates via LandmarkNormalizer.

        Args:
            frame: Video frame array.
            timestamp_ms: Frame timestamp in milliseconds.
            max_extrapolation_frames: Consecutive frames to extrapolate during occlusion.
            is_bgr: True if frame is BGR color format.

        Returns:
            NormalizedFrame containing scale- and translation-invariant coordinates.

        Raises:
            ValueError: If frame is invalid or pose landmarks cannot be detected.
        """
        extracted = self.extract(frame, timestamp_ms=timestamp_ms, is_bgr=is_bgr)

        if extracted.pose is None:
            raise ValueError(
                f"No pose landmarks detected in frame at timestamp {timestamp_ms}ms. "
                "Normalization requires valid upper body pose coordinates."
            )

        face_lm = (
            extracted.facial_contours
            if extracted.facial_contours is not None
            else extracted.face
        )

        return self.normalizer.process_frame(
            pose_landmarks=extracted.pose,
            left_hand_landmarks=extracted.left_hand,
            right_hand_landmarks=extracted.right_hand,
            face_landmarks=face_lm,
            timestamp_ms=timestamp_ms,
            max_extrapolation_frames=max_extrapolation_frames,
        )

    def to_contracts_dict(
        self,
        extracted: ExtractedLandmarks | NormalizedFrame,
        frame_id: int,
    ) -> dict[str, Any]:
        """Convert extracted or normalized landmarks to @converse/contracts FrameLandmarks schema.

        Matches packages/contracts/src/vision.ts FrameLandmarks specification.
        """
        if int(frame_id) < 0:
            raise ValueError(f"frame_id must be non-negative, got {frame_id}.")

        if isinstance(extracted, NormalizedFrame):
            data: dict[str, Any] = {
                "frameId": int(frame_id),
                "timestampMs": float(extracted.timestamp_ms),
            }

            if extracted.left_hand is not None:
                data["leftHand"] = {
                    "landmarks": [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                        for p in extracted.left_hand
                    ],
                    "handedness": "left",
                    "confidence": 1.0 if extracted.is_left_hand_visible else 0.0,
                }

            if extracted.right_hand is not None:
                data["rightHand"] = {
                    "landmarks": [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                        for p in extracted.right_hand
                    ],
                    "handedness": "right",
                    "confidence": 1.0 if extracted.is_right_hand_visible else 0.0,
                }

            if extracted.pose is not None:
                data["pose"] = {
                    "landmarks": [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                        for p in extracted.pose
                    ]
                }

            return data

        data = {
            "frameId": int(frame_id),
            "timestampMs": float(extracted.timestamp_ms),
        }

        if extracted.left_hand is not None:
            data["leftHand"] = {
                "landmarks": [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in extracted.left_hand
                ],
                "handedness": "left",
                "confidence": float(extracted.left_hand_confidence),
            }

        if extracted.right_hand is not None:
            data["rightHand"] = {
                "landmarks": [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in extracted.right_hand
                ],
                "handedness": "right",
                "confidence": float(extracted.right_hand_confidence),
            }

        if extracted.pose is not None:
            pose_list: list[dict[str, float]] = []
            for p in extracted.pose:
                pt: dict[str, float] = {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                }
                if len(p) > 3:
                    pt["visibility"] = float(p[3])
                pose_list.append(pt)
            data["pose"] = {"landmarks": pose_list}

        return data

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        if self._landmarker is not None and hasattr(self._landmarker, "close"):
            with contextlib.suppress(Exception):
                self._landmarker.close()
            self._landmarker = None
        if self._hand_recognizer is not None and hasattr(self._hand_recognizer, "close"):
            with contextlib.suppress(Exception):
                self._hand_recognizer.close()
            self._hand_recognizer = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit releasing resources."""
        self.close()

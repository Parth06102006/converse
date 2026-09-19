"""Coordinate normalization and temporal filtering for ASL 3D skeletal landmarks.

Implements translation invariance (root joint centering), scale invariance (Euclidean
distance normalization), One-Euro jitter filtering, and velocity extrapolation for occlusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asl_vision.filters import OneEuroFilter

__all__ = ["OneEuroFilter"]

# Standard landmark indices for MediaPipe Pose
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Standard landmark indices for MediaPipe Hands
HAND_WRIST = 0
HAND_THUMB_CMC = 1
HAND_THUMB_TIP = 4
HAND_INDEX_MCP = 5
HAND_INDEX_TIP = 8
HAND_MIDDLE_MCP = 9
HAND_MIDDLE_TIP = 12
HAND_RING_MCP = 13
HAND_RING_TIP = 16
HAND_PINKY_MCP = 17
HAND_PINKY_TIP = 20

# Upper body landmark subset (13 keypoints)
UPPER_BODY_LANDMARKS: tuple[int, ...] = (
    0,  # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    17,  # left pinky
    18,  # right pinky
    19,  # left index
    20,  # right index
    23,  # left hip
    24,  # right hip
)


# OneEuroFilter is canonically defined in asl_vision.filters and re-exported
# here for backward compatibility (tests import it from this module).


@dataclass(frozen=True)
class NormalizedFrame:
    """Immutable representation of a mathematically normalized landmark frame."""

    pose: (
        np.ndarray
    )  # Shape (N_pose, 3), centered at mid-shoulder, scaled by shoulder width
    left_hand: (
        np.ndarray | None
    )  # Shape (21, 3), centered at wrist, scaled by palm width
    right_hand: (
        np.ndarray | None
    )  # Shape (21, 3), centered at wrist, scaled by palm width
    face: np.ndarray | None  # Optional facial mesh points for NMMs
    shoulder_distance: float  # Absolute scale reference length in original units
    root_joint: np.ndarray  # Mid-shoulder coordinates (x, y, z) in camera space
    timestamp_ms: float
    is_left_hand_visible: bool
    is_right_hand_visible: bool


class LandmarkNormalizer:
    """Normalizes 3D skeletal landmarks for camera distance and translation invariance."""

    def __init__(
        self,
        apply_temporal_filtering: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the normalizer.

        Args:
            apply_temporal_filtering: Whether to apply One-Euro filtering.
            min_cutoff: Minimum cutoff frequency for One-Euro filter.
            beta: Speed coefficient for One-Euro filter.
            eps: Epsilon to avoid division by zero in scale normalization.
        """
        self.eps = eps
        self.apply_temporal_filtering = apply_temporal_filtering

        self._pose_filter = (
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            if apply_temporal_filtering
            else None
        )
        self._left_hand_filter = (
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            if apply_temporal_filtering
            else None
        )
        self._right_hand_filter = (
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            if apply_temporal_filtering
            else None
        )
        self._face_filter = (
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            if apply_temporal_filtering
            else None
        )

        # Occlusion tracking for linear velocity extrapolation
        self._prev_left_hand: np.ndarray | None = None
        self._left_hand_velocity: np.ndarray | None = None
        self._left_hand_occluded_frames = 0

        self._prev_right_hand: np.ndarray | None = None
        self._right_hand_velocity: np.ndarray | None = None
        self._right_hand_occluded_frames = 0

    def reset(self) -> None:
        """Reset temporal filters and occlusion velocity memory."""
        if self._pose_filter:
            self._pose_filter.reset()
        if self._left_hand_filter:
            self._left_hand_filter.reset()
        if self._right_hand_filter:
            self._right_hand_filter.reset()
        if self._face_filter:
            self._face_filter.reset()

        self._prev_left_hand = None
        self._left_hand_velocity = None
        self._left_hand_occluded_frames = 0

        self._prev_right_hand = None
        self._right_hand_velocity = None
        self._right_hand_occluded_frames = 0

    def normalize_pose(
        self,
        pose_landmarks: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Normalize pose landmarks relative to mid-shoulder root and shoulder distance.

        Args:
            pose_landmarks: Array of shape (33, 3) or (33, 4).

        Returns:
            Tuple of:
                - Normalized pose array of shape (N, 3)
                - Shoulder Euclidean distance scale factor
                - Mid-shoulder root joint coordinates (3,)
        """
        if pose_landmarks is None:
            raise ValueError("pose_landmarks cannot be None.")
        pose_arr = np.asarray(pose_landmarks, dtype=np.float32)
        if pose_arr.ndim != 2 or pose_arr.shape[0] < 13 or pose_arr.shape[1] < 3:
            raise ValueError(
                f"pose_landmarks must have shape (N, >=3) with N >= 13, got {pose_arr.shape}."
            )
        pose_xyz = np.nan_to_num(pose_arr[:, :3], nan=0.0)

        # Mid-shoulder root anchor
        left_shoulder = pose_xyz[POSE_LEFT_SHOULDER]
        right_shoulder = pose_xyz[POSE_RIGHT_SHOULDER]
        root_joint = (left_shoulder + right_shoulder) / 2.0

        # Translation invariance: center coordinates at root joint
        centered_pose = pose_xyz - root_joint

        # Scale invariance: divide by Euclidean distance between shoulders
        shoulder_dist = float(np.linalg.norm(left_shoulder - right_shoulder))
        scale = max(shoulder_dist, self.eps)
        normalized_pose = centered_pose / scale

        return normalized_pose, scale, root_joint

    def normalize_hand(
        self,
        hand_landmarks: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Normalize hand landmarks relative to wrist joint and palm length.

        Args:
            hand_landmarks: Array of shape (21, 3) or (21, 4).

        Returns:
            Tuple of:
                - Normalized hand array of shape (21, 3)
                - Palm Euclidean distance scale factor
        """
        if hand_landmarks is None:
            raise ValueError("hand_landmarks cannot be None.")
        hand_arr = np.asarray(hand_landmarks, dtype=np.float32)
        if hand_arr.ndim != 2 or hand_arr.shape[0] < 21 or hand_arr.shape[1] < 3:
            raise ValueError(
                f"hand_landmarks must have shape (N, >=3) with N >= 21, got {hand_arr.shape}."
            )
        hand_xyz = np.nan_to_num(hand_arr[:, :3], nan=0.0)

        # Wrist root anchor (joint 0)
        wrist = hand_xyz[HAND_WRIST]
        centered_hand = hand_xyz - wrist

        # Scale invariance: divide by distance from wrist to middle MCP joint
        middle_mcp = hand_xyz[HAND_MIDDLE_MCP]
        palm_dist = float(np.linalg.norm(wrist - middle_mcp))
        scale = max(palm_dist, self.eps)
        normalized_hand = centered_hand / scale

        return normalized_hand, scale

    def process_frame(
        self,
        pose_landmarks: np.ndarray,
        left_hand_landmarks: np.ndarray | None = None,
        right_hand_landmarks: np.ndarray | None = None,
        face_landmarks: np.ndarray | None = None,
        timestamp_ms: float = 0.0,
        max_extrapolation_frames: int = 2,
    ) -> NormalizedFrame:
        """Process and normalize a complete multi-modal landmark frame.

        Args:
            pose_landmarks: Array of shape (33, 3) or (33, 4).
            left_hand_landmarks: Optional array of shape (21, 3) or (21, 4).
            right_hand_landmarks: Optional array of shape (21, 3) or (21, 4).
            face_landmarks: Optional array of shape (K, 3).
            timestamp_ms: Frame timestamp in milliseconds.
            max_extrapolation_frames: Maximum consecutive frames to extrapolate during occlusion.

        Returns:
            NormalizedFrame containing standardized coordinate matrices.
        """
        timestamp_sec = timestamp_ms / 1000.0

        # 1. Normalize pose
        norm_pose, shoulder_dist, root_joint = self.normalize_pose(pose_landmarks)
        if self._pose_filter:
            norm_pose = self._pose_filter.filter(norm_pose, timestamp=timestamp_sec)

        # 2. Process Left Hand with occlusion handling
        norm_left: np.ndarray | None = None
        is_left_visible = False
        if left_hand_landmarks is not None and len(left_hand_landmarks) == 21:
            raw_norm_left, _ = self.normalize_hand(left_hand_landmarks)
            if self._left_hand_filter:
                norm_left = self._left_hand_filter.filter(
                    raw_norm_left, timestamp=timestamp_sec
                )
            else:
                norm_left = raw_norm_left

            # Update velocity only when continuous from immediately prior visible frame
            if self._prev_left_hand is not None and self._left_hand_occluded_frames == 0:
                self._left_hand_velocity = norm_left - self._prev_left_hand
            else:
                self._left_hand_velocity = None
            self._prev_left_hand = norm_left.copy()
            self._left_hand_occluded_frames = 0
            is_left_visible = True
        elif (
            self._prev_left_hand is not None
            and self._left_hand_velocity is not None
            and self._left_hand_occluded_frames < max_extrapolation_frames
        ):
            # Extrapolate position using previous velocity
            self._left_hand_occluded_frames += 1
            norm_left = self._prev_left_hand + self._left_hand_velocity
            self._prev_left_hand = norm_left.copy()
            is_left_visible = False
        else:
            self._left_hand_occluded_frames += 1
            self._prev_left_hand = None
            self._left_hand_velocity = None
            if self._left_hand_filter:
                self._left_hand_filter.reset()
            norm_left = None
            is_left_visible = False

        # 3. Process Right Hand with occlusion handling
        norm_right: np.ndarray | None = None
        is_right_visible = False
        if right_hand_landmarks is not None and len(right_hand_landmarks) == 21:
            raw_norm_right, _ = self.normalize_hand(right_hand_landmarks)
            if self._right_hand_filter:
                norm_right = self._right_hand_filter.filter(
                    raw_norm_right, timestamp=timestamp_sec
                )
            else:
                norm_right = raw_norm_right

            # Update velocity only when continuous from immediately prior visible frame
            if self._prev_right_hand is not None and self._right_hand_occluded_frames == 0:
                self._right_hand_velocity = norm_right - self._prev_right_hand
            else:
                self._right_hand_velocity = None
            self._prev_right_hand = norm_right.copy()
            self._right_hand_occluded_frames = 0
            is_right_visible = True
        elif (
            self._prev_right_hand is not None
            and self._right_hand_velocity is not None
            and self._right_hand_occluded_frames < max_extrapolation_frames
        ):
            # Extrapolate position using previous velocity
            self._right_hand_occluded_frames += 1
            norm_right = self._prev_right_hand + self._right_hand_velocity
            self._prev_right_hand = norm_right.copy()
            is_right_visible = False
        else:
            self._right_hand_occluded_frames += 1
            self._prev_right_hand = None
            self._right_hand_velocity = None
            if self._right_hand_filter:
                self._right_hand_filter.reset()
            norm_right = None
            is_right_visible = False

        # 4. Process Face Mesh if present
        norm_face: np.ndarray | None = None
        if face_landmarks is not None and len(face_landmarks) > 0:
            face_arr = np.asarray(face_landmarks, dtype=np.float32)
            face_xyz = np.nan_to_num(face_arr[:, :3], nan=0.0)
            # Center face at root joint and scale by shoulder distance for body consistency
            raw_norm_face = (face_xyz - root_joint) / max(shoulder_dist, self.eps)
            if self._face_filter:
                norm_face = self._face_filter.filter(
                    raw_norm_face, timestamp=timestamp_sec
                )
            else:
                norm_face = raw_norm_face

        return NormalizedFrame(
            pose=norm_pose,
            left_hand=norm_left,
            right_hand=norm_right,
            face=norm_face,
            shoulder_distance=shoulder_dist,
            root_joint=root_joint,
            timestamp_ms=timestamp_ms,
            is_left_hand_visible=is_left_visible,
            is_right_hand_visible=is_right_visible,
        )

    def normalize(
        self,
        landmarks_or_pose: Any,
        left_hand_landmarks: np.ndarray | None = None,
        right_hand_landmarks: np.ndarray | None = None,
        face_landmarks: np.ndarray | None = None,
        timestamp_ms: float = 0.0,
        max_extrapolation_frames: int = 2,
    ) -> NormalizedFrame:
        """Standardize coordinates across coordinate space, supporting both ExtractedLandmarks objects and raw arrays.

        Args:
            landmarks_or_pose: ExtractedLandmarks object or pose array of shape (33, 3).
            left_hand_landmarks: Optional left hand keypoints (21, 3).
            right_hand_landmarks: Optional right hand keypoints (21, 3).
            face_landmarks: Optional facial mesh contour points.
            timestamp_ms: Frame capture timestamp.
            max_extrapolation_frames: Consecutive occlusion extrapolation threshold.

        Returns:
            NormalizedFrame containing scale and translation invariant coordinates.
        """
        if hasattr(landmarks_or_pose, "pose"):
            extracted = landmarks_or_pose
            if extracted.pose is None:
                raise ValueError("Cannot normalize landmarks: pose landmarks are None.")
            face_lm = (
                extracted.facial_contours
                if getattr(extracted, "facial_contours", None) is not None
                else getattr(extracted, "face", None)
            )
            return self.process_frame(
                pose_landmarks=extracted.pose,
                left_hand_landmarks=extracted.left_hand,
                right_hand_landmarks=extracted.right_hand,
                face_landmarks=face_lm,
                timestamp_ms=timestamp_ms,
                max_extrapolation_frames=max_extrapolation_frames,
            )

        return self.process_frame(
            pose_landmarks=landmarks_or_pose,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
            timestamp_ms=timestamp_ms,
            max_extrapolation_frames=max_extrapolation_frames,
        )


def pack_frame_tensor(
    frame: NormalizedFrame,
    include_pose: bool = True,
    include_hands: bool = True,
    include_face: bool = False,
) -> np.ndarray:
    """Pack normalized frame landmarks into a contiguous 1D or 2D feature tensor.

    Args:
        frame: NormalizedFrame instance.
        include_pose: Whether to include 33 pose landmarks.
        include_hands: Whether to include bilateral 21 hand landmarks (left + right = 42).
        include_face: Whether to include facial contour landmarks for NMMs.

    Returns:
        Flattened float32 array suitable for neural sequence model ingestion.
    """
    chunks: list[np.ndarray] = []

    if include_pose:
        chunks.append(frame.pose.reshape(-1))

    if include_hands:
        if frame.left_hand is not None:
            chunks.append(frame.left_hand.reshape(-1))
        else:
            chunks.append(np.zeros(21 * 3, dtype=np.float32))

        if frame.right_hand is not None:
            chunks.append(frame.right_hand.reshape(-1))
        else:
            chunks.append(np.zeros(21 * 3, dtype=np.float32))

    if include_face and frame.face is not None:
        chunks.append(frame.face.reshape(-1))

    if not chunks:
        raise ValueError(
            "At least one landmark stream must be included in packed frame tensor "
            "(include_pose, include_hands, or include_face)."
        )

    return np.concatenate(chunks).astype(np.float32)

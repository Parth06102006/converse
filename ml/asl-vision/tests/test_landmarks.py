"""Unit tests verifying landmark extraction, frame validation, NMM contours, and normalization integration."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from asl_vision.landmarks import (
    FACIAL_CONTOUR_INDICES,
    NMM_EYEBROW_INDICES,
    NMM_LIP_INDICES,
    ExtractedLandmarks,
    LandmarkExtractor,
)
from asl_vision.normalization import LandmarkNormalizer, NormalizedFrame


def test_frame_validation_valid_rgb() -> None:
    """Verifies that a standard 640x480 RGB uint8 frame passes validation."""
    extractor = LandmarkExtractor(use_mock=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    validated = extractor.validate_frame(frame)
    assert validated.shape == (480, 640, 3)
    assert validated.dtype == np.uint8


def test_frame_validation_float_scaling() -> None:
    """Verifies that normalized float32 frames in [0, 1] are converted to uint8 [0, 255]."""
    extractor = LandmarkExtractor(use_mock=True)
    frame_float = np.ones((480, 640, 3), dtype=np.float32)

    validated = extractor.validate_frame(frame_float)
    assert validated.dtype == np.uint8
    assert np.all(validated == 255)


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "not_an_array",
        np.zeros((480, 640), dtype=np.uint8),  # 2D grayscale
        np.zeros((480, 640, 4), dtype=np.uint8),  # 4-channel RGBA
        np.zeros((0, 640, 3), dtype=np.uint8),  # Empty height
        np.zeros((480, 0, 3), dtype=np.uint8),  # Empty width
    ],
)
def test_frame_validation_invalid_shapes_raise(invalid_input: object) -> None:
    """Verifies that non-standard, malformed, or empty frames raise ValueError."""
    extractor = LandmarkExtractor(use_mock=True)
    with pytest.raises((ValueError, TypeError)):
        extractor.validate_frame(invalid_input)  # type: ignore[arg-type]


def test_nmm_facial_contour_indices_topology() -> None:
    """Verifies eyebrow and lip contour index subsets are properly included in facial contours."""
    assert len(NMM_EYEBROW_INDICES) > 0
    assert len(NMM_LIP_INDICES) > 0
    assert set(NMM_EYEBROW_INDICES).issubset(set(FACIAL_CONTOUR_INDICES))
    assert set(NMM_LIP_INDICES).issubset(set(FACIAL_CONTOUR_INDICES))


def test_landmark_extraction_shapes_and_topology() -> None:
    """Verifies extracted landmarks adhere to ASL topological specifications:

    - 33 pose landmarks
    - 21 left hand landmarks
    - 21 right hand landmarks
    - Facial contour landmarks for NMMs (eyebrows + lips + midline)
    """
    extractor = LandmarkExtractor(use_mock=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    extracted = extractor.extract(frame, timestamp_ms=33.3)

    assert extracted.pose is not None
    assert extracted.pose.shape == (33, 4)  # (x, y, z, visibility)

    assert extracted.left_hand is not None
    assert extracted.left_hand.shape == (21, 3)

    assert extracted.right_hand is not None
    assert extracted.right_hand.shape == (21, 3)

    assert extracted.facial_contours is not None
    assert extracted.facial_contours.shape == (len(FACIAL_CONTOUR_INDICES), 3)

    assert extracted.face_blendshapes is not None
    assert "browDownLeft" in extracted.face_blendshapes
    assert "browInnerUp" in extracted.face_blendshapes

    assert extracted.timestamp_ms == 33.3
    assert extracted.confidence > 0.0


def test_bgr_frame_conversion() -> None:
    """Verifies that BGR frames are converted to RGB without errors."""
    extractor = LandmarkExtractor(use_mock=True)
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 255  # Blue channel in BGR

    extracted = extractor.extract(bgr_frame, is_bgr=True)
    assert extracted.pose is not None


def test_custom_backend_injection() -> None:
    """Verifies that an external backend callable can be injected for custom inference."""
    mock_extracted = ExtractedLandmarks(
        pose=np.ones((33, 4), dtype=np.float32),
        left_hand=np.ones((21, 3), dtype=np.float32),
        right_hand=np.ones((21, 3), dtype=np.float32),
        face=None,
        facial_contours=None,
        face_blendshapes=None,
        pose_world=None,
        timestamp_ms=123.0,
        confidence=0.99,
    )

    backend_mock = MagicMock(return_value=mock_extracted)
    extractor = LandmarkExtractor(backend=backend_mock)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = extractor.extract(frame, timestamp_ms=123.0)

    assert backend_mock.call_count == 1
    assert result.timestamp_ms == 123.0
    assert result.confidence == 0.99


def test_extract_and_normalize_integration() -> None:
    """Verifies seamless integration with LandmarkNormalizer to produce NormalizedFrame."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)
    extractor = LandmarkExtractor(normalizer=normalizer, use_mock=True)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    normalized = extractor.extract_and_normalize(frame, timestamp_ms=50.0)

    assert isinstance(normalized, NormalizedFrame)
    assert normalized.timestamp_ms == 50.0
    assert normalized.pose.shape == (33, 3)
    assert normalized.left_hand is not None
    assert normalized.left_hand.shape == (21, 3)
    assert normalized.right_hand is not None
    assert normalized.right_hand.shape == (21, 3)
    assert normalized.face is not None
    assert normalized.shoulder_distance > 0.0
    assert normalized.is_left_hand_visible is True
    assert normalized.is_right_hand_visible is True


def test_extract_and_normalize_missing_pose_raises() -> None:
    """Verifies ValueError when pose cannot be extracted from the frame."""
    extractor = LandmarkExtractor(use_mock=True)
    # Set mock result with None pose
    extractor.set_mock_result(
        ExtractedLandmarks(
            pose=None,
            left_hand=None,
            right_hand=None,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=0.0,
            confidence=0.0,
        )
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No pose landmarks detected"):
        extractor.extract_and_normalize(frame, timestamp_ms=0.0)


def test_to_contracts_dict_schema() -> None:
    """Verifies serialization conforming to @converse/contracts FrameLandmarks schema."""
    extractor = LandmarkExtractor(use_mock=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    extracted = extractor.extract(frame, timestamp_ms=100.0)

    contract_dict = extractor.to_contracts_dict(extracted, frame_id=42)

    assert contract_dict["frameId"] == 42
    assert contract_dict["timestampMs"] == 100.0

    # Left Hand schema
    assert "leftHand" in contract_dict
    assert len(contract_dict["leftHand"]["landmarks"]) == 21
    assert contract_dict["leftHand"]["handedness"] == "left"
    assert "confidence" in contract_dict["leftHand"]
    assert "x" in contract_dict["leftHand"]["landmarks"][0]
    assert "y" in contract_dict["leftHand"]["landmarks"][0]
    assert "z" in contract_dict["leftHand"]["landmarks"][0]

    # Right Hand schema
    assert "rightHand" in contract_dict
    assert len(contract_dict["rightHand"]["landmarks"]) == 21
    assert contract_dict["rightHand"]["handedness"] == "right"

    # Pose schema
    assert "pose" in contract_dict
    assert len(contract_dict["pose"]["landmarks"]) == 33
    assert "visibility" in contract_dict["pose"]["landmarks"][0]


def test_context_manager_lifecycle() -> None:
    """Verifies that LandmarkExtractor operates correctly inside a context manager."""
    with LandmarkExtractor(use_mock=True) as extractor:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(frame)
        assert result.pose is not None
    # Confirm landmarker reference cleared
    assert extractor._landmarker is None


def test_parse_mediapipe_result_mock_objects() -> None:
    """Verifies parsing logic from MediaPipe result objects to ExtractedLandmarks."""
    extractor = LandmarkExtractor(use_mock=True)

    # Mock landmark point
    class MockPoint:
        def __init__(self, x: float, y: float, z: float, visibility: float = 0.9):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility

    class MockCategory:
        def __init__(self, name: str, score: float):
            self.category_name = name
            self.score = score

    mock_mp_result = MagicMock()
    mock_mp_result.pose_landmarks = [MockPoint(0.1 * i, 0.2 * i, 0.0) for i in range(33)]
    mock_mp_result.pose_world_landmarks = [MockPoint(0.05 * i, 0.1 * i, 0.0) for i in range(33)]
    mock_mp_result.left_hand_landmarks = [MockPoint(0.01 * i, 0.02 * i, 0.0) for i in range(21)]
    mock_mp_result.right_hand_landmarks = [MockPoint(0.02 * i, 0.03 * i, 0.0) for i in range(21)]
    mock_mp_result.face_landmarks = [MockPoint(0.001 * i, 0.002 * i, 0.0) for i in range(468)]
    mock_mp_result.face_blendshapes = [
        MockCategory("browDownLeft", 0.45),
        MockCategory("jawOpen", 0.72),
    ]

    parsed = extractor._parse_mediapipe_result(mock_mp_result, timestamp_ms=250.0)

    assert parsed.pose is not None
    assert parsed.pose.shape == (33, 4)
    assert parsed.pose_world is not None
    assert parsed.pose_world.shape == (33, 3)
    assert parsed.left_hand is not None
    assert parsed.left_hand.shape == (21, 3)
    assert parsed.right_hand is not None
    assert parsed.right_hand.shape == (21, 3)
    assert parsed.face is not None
    assert parsed.face.shape == (468, 3)
    assert parsed.facial_contours is not None
    assert parsed.facial_contours.shape == (len(FACIAL_CONTOUR_INDICES), 3)
    assert parsed.face_blendshapes == {"browDownLeft": 0.45, "jawOpen": 0.72}
    assert parsed.timestamp_ms == 250.0


def test_landmark_extractor_invalid_running_mode_raises() -> None:
    """Verifies that invalid running modes raise ValueError."""
    with pytest.raises(ValueError, match="Invalid running_mode"):
        LandmarkExtractor(running_mode="BATCH", use_mock=True)


def test_landmark_extractor_missing_model_raises() -> None:
    """Verifies that nonexistent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="MediaPipe model file not found"):
        LandmarkExtractor(model_path="/nonexistent/model.task")


def test_landmark_extractor_no_model_and_use_mock_false_raises() -> None:
    """Verifies that setting use_mock=False without a model raises ValueError."""
    with pytest.raises(ValueError, match="No model_path or model_buffer provided and use_mock=False"):
        LandmarkExtractor(use_mock=False)


def test_landmark_extractor_monotonic_timestamps_video_mode() -> None:
    """Verifies that video mode detects non-monotonic timestamps."""
    extractor = LandmarkExtractor(running_mode="VIDEO", use_mock=True)
    extractor.use_mock = False
    # Inject a mock landmarker to test VIDEO mode timestamp validation
    extractor._landmarker = MagicMock()
    extractor._landmarker.detect_for_video.return_value = MagicMock(
        pose_landmarks=[],
        pose_world_landmarks=[],
        left_hand_landmarks=[],
        right_hand_landmarks=[],
        face_landmarks=[],
        face_blendshapes=None,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    extractor.extract(frame, timestamp_ms=100.0)

    # Re-sending same or earlier timestamp raises ValueError
    with pytest.raises(ValueError, match="Timestamp must be strictly monotonically increasing"):
        extractor.extract(frame, timestamp_ms=100.0)

    with pytest.raises(ValueError, match="Timestamp must be strictly monotonically increasing"):
        extractor.extract(frame, timestamp_ms=90.0)


def test_to_contracts_dict_normalized_frame_and_negative_id() -> None:
    """Verifies to_contracts_dict with NormalizedFrame and validation for negative frame_id."""
    extractor = LandmarkExtractor(use_mock=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    normalized = extractor.extract_and_normalize(frame, timestamp_ms=50.0)

    # Valid conversion with NormalizedFrame
    res = extractor.to_contracts_dict(normalized, frame_id=10)
    assert res["frameId"] == 10
    assert res["timestampMs"] == 50.0
    assert "leftHand" in res
    assert "rightHand" in res
    assert "pose" in res

    # Negative frame_id raises ValueError
    with pytest.raises(ValueError, match="frame_id must be non-negative"):
        extractor.to_contracts_dict(normalized, frame_id=-1)


def test_validate_frame_non_contiguous() -> None:
    """Verifies that non-contiguous frames (e.g. webcam selfie horizontal flips) are made contiguous."""
    extractor = LandmarkExtractor(use_mock=True)
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    # Horizontal flip creates negative stride (non-contiguous)
    flipped = raw[:, ::-1, :]
    assert not flipped.flags["C_CONTIGUOUS"]

    validated = extractor.validate_frame(flipped)
    assert validated.flags["C_CONTIGUOUS"]


def test_parse_mediapipe_result_nan_and_none_sanitization() -> None:
    """Verifies that None and NaN in MediaPipe landmarks do not crash or corrupt output."""
    extractor = LandmarkExtractor(use_mock=True)

    class PointWithNone:
        x = None
        y = np.nan
        z = 0.5
        visibility = None

    mock_res = MagicMock()
    mock_res.pose_landmarks = [PointWithNone() for _ in range(33)]
    mock_res.pose_world_landmarks = [PointWithNone() for _ in range(33)]
    mock_res.left_hand_landmarks = [PointWithNone() for _ in range(21)]
    mock_res.right_hand_landmarks = [PointWithNone() for _ in range(21)]
    mock_res.face_landmarks = [PointWithNone() for _ in range(468)]
    mock_res.face_blendshapes = None

    parsed = extractor._parse_mediapipe_result(mock_res, timestamp_ms=0.0)
    assert parsed.pose is not None
    assert not np.isnan(parsed.pose).any()
    assert parsed.left_hand is not None
    assert not np.isnan(parsed.left_hand).any()
    assert parsed.right_hand is not None
    assert not np.isnan(parsed.right_hand).any()


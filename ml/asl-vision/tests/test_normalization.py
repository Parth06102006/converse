"""Unit tests verifying mathematical normalization, One-Euro filtering, and occlusion extrapolation."""

import numpy as np
import pytest

from asl_vision.normalization import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    LandmarkNormalizer,
    OneEuroFilter,
    pack_frame_tensor,
)


def _generate_synthetic_pose(
    scale: float = 1.0, offset: np.ndarray | None = None
) -> np.ndarray:
    """Generate a realistic synthetic upper body pose array of shape (33, 3)."""
    pose = np.zeros((33, 3), dtype=np.float32)
    # Left shoulder at (-0.2, 0.4, 0.0)
    pose[POSE_LEFT_SHOULDER] = np.array([-0.2, 0.4, 0.0], dtype=np.float32) * scale
    # Right shoulder at (0.2, 0.4, 0.0)
    pose[POSE_RIGHT_SHOULDER] = np.array([0.2, 0.4, 0.0], dtype=np.float32) * scale

    # Populate other joints with arbitrary values
    for i in range(33):
        if i not in (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER):
            pose[i] = np.array([i * 0.01, i * 0.02, 0.0], dtype=np.float32) * scale

    if offset is not None:
        pose += offset

    return pose


def _generate_synthetic_hand(
    scale: float = 1.0, offset: np.ndarray | None = None
) -> np.ndarray:
    """Generate a synthetic hand array of shape (21, 3)."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[HAND_WRIST] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hand[HAND_MIDDLE_MCP] = np.array([0.0, 0.1, 0.0], dtype=np.float32) * scale

    for i in range(21):
        if i not in (HAND_WRIST, HAND_MIDDLE_MCP):
            hand[i] = np.array([i * 0.005, i * 0.01, 0.01], dtype=np.float32) * scale

    if offset is not None:
        hand += offset

    return hand


def test_pose_translation_invariance() -> None:
    """Verifies that translating the signer anywhere in 3D space produces identical normalized coordinates."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)

    base_pose = _generate_synthetic_pose(scale=1.0, offset=np.array([0.0, 0.0, 0.0]))
    offset_pose = _generate_synthetic_pose(scale=1.0, offset=np.array([1.5, -2.3, 4.7]))

    norm_base, scale_base, root_base = normalizer.normalize_pose(base_pose)
    norm_offset, scale_offset, root_offset = normalizer.normalize_pose(offset_pose)

    assert np.allclose(scale_base, scale_offset, atol=1e-5)
    assert np.allclose(norm_base, norm_offset, atol=1e-5)
    assert np.allclose(root_offset - root_base, np.array([1.5, -2.3, 4.7]), atol=1e-5)


def test_pose_scale_invariance() -> None:
    """Verifies that signers at different camera distances produce identical normalized coordinates."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)

    near_pose = _generate_synthetic_pose(scale=2.0)  # Signer close to camera
    far_pose = _generate_synthetic_pose(scale=0.5)  # Signer far from camera

    norm_near, _, _ = normalizer.normalize_pose(near_pose)
    norm_far, _, _ = normalizer.normalize_pose(far_pose)

    assert np.allclose(norm_near, norm_far, atol=1e-5)


def test_hand_translation_and_scale_invariance() -> None:
    """Verifies hand centering at wrist and scale invariance relative to palm length."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)

    hand1 = _generate_synthetic_hand(scale=1.0, offset=np.array([0.3, 0.5, -0.2]))
    hand2 = _generate_synthetic_hand(scale=2.5, offset=np.array([-1.2, 0.8, 1.4]))

    norm_hand1, _ = normalizer.normalize_hand(hand1)
    norm_hand2, _ = normalizer.normalize_hand(hand2)

    # Root joint (wrist) must always be at origin (0, 0, 0)
    assert np.allclose(norm_hand1[HAND_WRIST], np.zeros(3), atol=1e-6)
    assert np.allclose(norm_hand2[HAND_WRIST], np.zeros(3), atol=1e-6)

    # Normalized relative joint geometries must match
    assert np.allclose(norm_hand1, norm_hand2, atol=1e-5)


def test_one_euro_filter_jitter_damping() -> None:
    """Verifies that high-frequency noise on static signal is substantially attenuated."""
    filter_inst = OneEuroFilter(min_cutoff=1.0, beta=0.007)

    # Static signal with Gaussian noise
    np.random.seed(42)
    truth = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    noisy_inputs = [
        truth + np.random.normal(0, 0.05, 3).astype(np.float32) for _ in range(30)
    ]

    filtered_outputs = [
        filter_inst.filter(val, timestamp=i / 30.0)
        for i, val in enumerate(noisy_inputs)
    ]

    raw_variance = np.var([x[0] for x in noisy_inputs[10:]])
    filtered_variance = np.var([x[0] for x in filtered_outputs[10:]])

    # Filtered variance must be reduced by at least 60%
    assert filtered_variance < raw_variance * 0.4


def test_occlusion_velocity_extrapolation() -> None:
    """Verifies linear velocity extrapolation during a 1-frame occlusion."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)

    pose = _generate_synthetic_pose()
    hand_t0 = _generate_synthetic_hand(offset=np.array([0.0, 0.0, 0.0]))
    hand_t1 = _generate_synthetic_hand(offset=np.array([0.02, 0.02, 0.0]))

    # Frame 0: hand at pos 0
    f0 = normalizer.process_frame(pose, right_hand_landmarks=hand_t0, timestamp_ms=0)
    assert f0.is_right_hand_visible is True

    # Frame 1: hand moved by (+0.02, +0.02, 0.0)
    f1 = normalizer.process_frame(pose, right_hand_landmarks=hand_t1, timestamp_ms=33)
    assert f1.is_right_hand_visible is True

    # Frame 2: hand occluded (None passed)
    f2 = normalizer.process_frame(pose, right_hand_landmarks=None, timestamp_ms=66)
    assert f2.is_right_hand_visible is False
    assert f2.right_hand is not None  # Must be extrapolated

    # Extrapolated position must continue the trajectory
    expected_pos = f1.right_hand[HAND_MIDDLE_MCP] + (
        f1.right_hand[HAND_MIDDLE_MCP] - f0.right_hand[HAND_MIDDLE_MCP]
    )
    assert np.allclose(f2.right_hand[HAND_MIDDLE_MCP], expected_pos, atol=1e-5)


def test_pack_frame_tensor_dimensions() -> None:
    """Verifies consistent tensor packing shape for neural model ingestion."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)
    pose = _generate_synthetic_pose()
    hand = _generate_synthetic_hand()

    frame = normalizer.process_frame(
        pose, left_hand_landmarks=hand, right_hand_landmarks=hand
    )
    tensor = pack_frame_tensor(frame, include_pose=True, include_hands=True)

    # 33 pose joints * 3 + 21 left hand * 3 + 21 right hand * 3 = 99 + 63 + 63 = 225 floats
    assert tensor.shape == (225,)
    assert tensor.dtype == np.float32


def test_occlusion_memory_cleanup_after_expiration() -> None:
    """Verifies that exceeding max_extrapolation_frames completely purges memory so reappearance starts fresh."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=True)
    pose = _generate_synthetic_pose()
    hand0 = _generate_synthetic_hand(offset=np.array([0.0, 0.0, 0.0]))
    hand1 = _generate_synthetic_hand(offset=np.array([0.01, 0.01, 0.0]))

    # Frame 0 and 1: visible
    normalizer.process_frame(pose, right_hand_landmarks=hand0, timestamp_ms=0)
    normalizer.process_frame(pose, right_hand_landmarks=hand1, timestamp_ms=33)

    # Frame 2: occluded frame 1 (extrapolated)
    f2 = normalizer.process_frame(pose, right_hand_landmarks=None, timestamp_ms=66)
    assert f2.right_hand is not None

    # Frame 3: occluded frame 2 (extrapolated)
    f3 = normalizer.process_frame(pose, right_hand_landmarks=None, timestamp_ms=99)
    assert f3.right_hand is not None

    # Frame 4: occluded frame 3 (> max_extrapolation_frames=2) -> memory purged
    f4 = normalizer.process_frame(pose, right_hand_landmarks=None, timestamp_ms=132)
    assert f4.right_hand is None
    assert normalizer._prev_right_hand is None
    assert normalizer._right_hand_velocity is None

    # Frame 10: hand reappears at a distant location
    hand_reappear = _generate_synthetic_hand(offset=np.array([0.5, 0.5, 0.0]))
    f10 = normalizer.process_frame(pose, right_hand_landmarks=hand_reappear, timestamp_ms=330)
    assert f10.right_hand is not None
    # Velocity must NOT be calculated across the 6-frame gap
    assert normalizer._right_hand_velocity is None


def test_normalize_pose_invalid_and_nan() -> None:
    """Verifies input validation and NaN sanitization in normalize_pose."""
    normalizer = LandmarkNormalizer()
    with pytest.raises(ValueError, match="pose_landmarks cannot be None"):
        normalizer.normalize_pose(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pose_landmarks must have shape"):
        normalizer.normalize_pose(np.zeros((10, 3)))

    # Pose with NaNs should be sanitized without crash
    pose_nan = _generate_synthetic_pose()
    pose_nan[0, 0] = np.nan
    norm_pose, scale, root = normalizer.normalize_pose(pose_nan)
    assert not np.isnan(norm_pose).any()
    assert not np.isnan(scale)
    assert not np.isnan(root).any()


def test_normalize_hand_invalid_and_nan() -> None:
    """Verifies input validation and NaN sanitization in normalize_hand."""
    normalizer = LandmarkNormalizer()
    with pytest.raises(ValueError, match="hand_landmarks cannot be None"):
        normalizer.normalize_hand(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hand_landmarks must have shape"):
        normalizer.normalize_hand(np.zeros((15, 3)))

    # Hand with NaNs should be sanitized without crash
    hand_nan = _generate_synthetic_hand()
    hand_nan[4, 1] = np.nan
    norm_hand, scale = normalizer.normalize_hand(hand_nan)
    assert not np.isnan(norm_hand).any()
    assert not np.isnan(scale)


def test_one_euro_filter_validation_and_nan_resilience() -> None:
    """Verifies parameter validation and NaN resilience in OneEuroFilter."""
    with pytest.raises(ValueError, match="min_cutoff must be positive"):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError, match="d_cutoff must be positive"):
        OneEuroFilter(d_cutoff=-1.0)
    with pytest.raises(ValueError, match="beta must be non-negative"):
        OneEuroFilter(beta=-0.1)

    f = OneEuroFilter()
    x0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    f.filter(x0, timestamp=0.0)

    # Duplicate timestamp returns previous state safely
    out_dup = f.filter(x0 + 10.0, timestamp=0.0)
    assert np.allclose(out_dup, x0)

    # NaN in input does not poison the filter
    nan_in = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    out_nan = f.filter(nan_in, timestamp=0.033)
    assert not np.isnan(out_nan).any()


def test_pack_frame_tensor_with_face_and_validation() -> None:
    """Verifies packing facial landmarks and validating non-empty streams."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=False)
    pose = _generate_synthetic_pose()
    hand = _generate_synthetic_hand()
    face = np.ones((40, 3), dtype=np.float32)

    frame = normalizer.process_frame(
        pose, left_hand_landmarks=hand, right_hand_landmarks=hand, face_landmarks=face
    )

    # With face: 33*3 + 21*3 + 21*3 + 40*3 = 99 + 63 + 63 + 120 = 345
    tensor_with_face = pack_frame_tensor(frame, include_pose=True, include_hands=True, include_face=True)
    assert tensor_with_face.shape == (345,)

    # Empty selection raises ValueError
    with pytest.raises(ValueError, match="At least one landmark stream must be included"):
        pack_frame_tensor(frame, include_pose=False, include_hands=False, include_face=False)


def test_face_temporal_filtering() -> None:
    """Verifies that facial landmarks are smoothed by OneEuroFilter."""
    normalizer = LandmarkNormalizer(apply_temporal_filtering=True)
    pose = _generate_synthetic_pose()

    # Create face with jitter
    np.random.seed(42)
    base_face = np.ones((40, 3), dtype=np.float32)
    frames = []
    for i in range(20):
        jittered_face = base_face + np.random.normal(0, 0.05, (40, 3)).astype(np.float32)
        f = normalizer.process_frame(pose, face_landmarks=jittered_face, timestamp_ms=i * 33.3)
        frames.append(f.face)

    # Confirm face is not None and filter smoothed variance
    assert frames[-1] is not None
    assert frames[-1].shape == (40, 3)


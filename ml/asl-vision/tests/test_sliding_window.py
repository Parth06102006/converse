"""Unit tests verifying rolling FIFO temporal buffer, tensor packing, stride mechanics, and confidence thresholding."""

import numpy as np
import pytest
import torch

from asl_vision.normalization import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    NormalizedFrame,
)
from asl_vision.sliding_window import (
    SlidingWindowBuffer,
    SlidingWindowOutput,
)


def _create_synthetic_frame(
    timestamp_ms: float = 0.0,
    has_hands: bool = True,
) -> NormalizedFrame:
    """Helper creating a valid NormalizedFrame for testing."""
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[POSE_LEFT_SHOULDER] = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
    pose[POSE_RIGHT_SHOULDER] = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    left_hand = None
    right_hand = None
    if has_hands:
        left_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand[HAND_WRIST] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        left_hand[HAND_MIDDLE_MCP] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        right_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand[HAND_WRIST] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        right_hand[HAND_MIDDLE_MCP] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return NormalizedFrame(
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=None,
        shoulder_distance=1.0,
        root_joint=np.zeros(3, dtype=np.float32),
        timestamp_ms=timestamp_ms,
        is_left_hand_visible=has_hands,
        is_right_hand_visible=has_hands,
    )


def test_buffer_initialization_defaults() -> None:
    """Verifies default parameters and initial state of SlidingWindowBuffer."""
    buf = SlidingWindowBuffer()
    assert buf.window_size == 30
    assert buf.stride == 5
    assert buf.confidence_threshold == 0.5
    assert buf.frame_count == 0
    assert buf.window_index == 0
    assert len(buf) == 0
    assert not buf.is_full


@pytest.mark.parametrize(
    ("w", "s", "c"),
    [
        (0, 5, 0.5),  # Zero window size
        (-10, 5, 0.5),  # Negative window size
        (30, 0, 0.5),  # Zero stride
        (30, -1, 0.5),  # Negative stride
        (30, 5, -0.1),  # Negative confidence threshold
        (30, 5, 1.5),  # Confidence threshold > 1.0
    ],
)
def test_invalid_initialization_parameters_raise(
    w: int, s: int, c: float
) -> None:
    """Verifies that illegal buffer arguments raise ValueError."""
    with pytest.raises(ValueError):
        SlidingWindowBuffer(window_size=w, stride=s, confidence_threshold=c)


def test_stride_and_accumulation_cadence() -> None:
    """Verifies window emission cadence:

    - Frames 1..29 return None
    - Frame 30 returns Window 0
    - Frames 31..34 return None
    - Frame 35 returns Window 1
    - Frame 40 returns Window 2
    """
    buf = SlidingWindowBuffer(window_size=30, stride=5)

    emitted_windows: list[SlidingWindowOutput] = []

    for i in range(1, 46):
        frame = _create_synthetic_frame(timestamp_ms=i * 33.3)
        res = buf.add_frame(frame)
        if res is not None:
            emitted_windows.append(res)

    assert len(emitted_windows) == 4
    assert emitted_windows[0].window_index == 0
    assert emitted_windows[1].window_index == 1
    assert emitted_windows[2].window_index == 2
    assert emitted_windows[3].window_index == 3

    assert buf.frame_count == 45
    assert buf.window_index == 4


def test_packed_tensor_shape_and_layout() -> None:
    """Verifies tensor shape (1, 3, 30, 75) for standard upper body + bilateral hands."""
    buf = SlidingWindowBuffer(window_size=30, stride=5)

    out: SlidingWindowOutput | None = None
    for i in range(30):
        frame = _create_synthetic_frame(timestamp_ms=i * 33.3)
        out = buf.add_frame(frame)

    assert out is not None
    assert isinstance(out.tensor, torch.Tensor)
    # Shape: (Batch=1, Channels=3, Time=30, Joints=75)
    assert out.tensor.shape == (1, 3, 30, 75)
    assert out.tensor.dtype == torch.float32

    # Numpy parity
    arr = out.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 3, 30, 75)
    assert arr.dtype == np.float32


def test_coordinate_channel_integrity() -> None:
    """Verifies channel 0 is X, channel 1 is Y, channel 2 is Z."""
    buf = SlidingWindowBuffer(window_size=30, stride=5)

    for i in range(30):
        # Create frame where shoulder has known coordinates
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[POSE_LEFT_SHOULDER] = np.array([1.23, 4.56, 7.89], dtype=np.float32)
        frame = NormalizedFrame(
            pose=pose,
            left_hand=np.zeros((21, 3), dtype=np.float32),
            right_hand=np.zeros((21, 3), dtype=np.float32),
            face=None,
            shoulder_distance=1.0,
            root_joint=np.zeros(3, dtype=np.float32),
            timestamp_ms=float(i),
            is_left_hand_visible=True,
            is_right_hand_visible=True,
        )
        out = buf.add_frame(frame)

    assert out is not None
    # Index POSE_LEFT_SHOULDER is joint 11
    # Channel 0: X
    assert np.isclose(out.tensor[0, 0, 0, POSE_LEFT_SHOULDER].item(), 1.23, atol=1e-5)
    # Channel 1: Y
    assert np.isclose(out.tensor[0, 1, 0, POSE_LEFT_SHOULDER].item(), 4.56, atol=1e-5)
    # Channel 2: Z
    assert np.isclose(out.tensor[0, 2, 0, POSE_LEFT_SHOULDER].item(), 7.89, atol=1e-5)


def test_rolling_fifo_window_timestamps() -> None:
    """Verifies that sliding window drops oldest frames as new ones arrive."""
    buf = SlidingWindowBuffer(window_size=10, stride=5)

    windows: list[SlidingWindowOutput] = []
    for i in range(15):
        frame = _create_synthetic_frame(timestamp_ms=float(i * 100))
        res = buf.add_frame(frame)
        if res is not None:
            windows.append(res)

    assert len(windows) == 2
    # Window 0 covers frames 0..9 -> timestamp 0..900ms
    assert windows[0].start_timestamp_ms == 0.0
    assert windows[0].end_timestamp_ms == 900.0

    # Window 1 covers frames 5..14 -> timestamp 500..1400ms
    assert windows[1].start_timestamp_ms == 500.0
    assert windows[1].end_timestamp_ms == 1400.0


def test_buffer_reset_and_clearing() -> None:
    """Verifies that reset() and clear() reinitialize all internal state."""
    buf = SlidingWindowBuffer(window_size=30, stride=5)

    for i in range(25):
        buf.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))

    assert len(buf) == 25
    assert buf.frame_count == 25

    buf.reset()

    assert len(buf) == 0
    assert buf.frame_count == 0
    assert buf.window_index == 0
    assert buf.get_current_window() is None

    # Adding new frames restarts accumulation
    for i in range(30):
        res = buf.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))
    assert res is not None
    assert res.window_index == 0


def test_confidence_thresholding_valid_and_invalid() -> None:
    """Verifies confidence calculation based on hand presence ratio."""
    buf = SlidingWindowBuffer(window_size=10, stride=5, confidence_threshold=0.5)

    # 1. High hand presence: all frames have hands
    for i in range(10):
        out = buf.add_frame(_create_synthetic_frame(has_hands=True, timestamp_ms=float(i)))

    assert out is not None
    assert out.confidence == 1.0
    assert out.is_valid is True

    buf.reset()

    # 2. Low hand presence: only 2 of 10 frames have hands (confidence = 0.2 < 0.5)
    for i in range(10):
        has_hands = i < 2
        out = buf.add_frame(_create_synthetic_frame(has_hands=has_hands, timestamp_ms=float(i)))

    assert out is not None
    assert np.isclose(out.confidence, 0.2)
    assert out.is_valid is False


def test_emit_only_valid_mode() -> None:
    """Verifies emit_only_valid=True suppresses invalid windows."""
    buf = SlidingWindowBuffer(
        window_size=10,
        stride=5,
        confidence_threshold=0.5,
        emit_only_valid=True,
    )

    # Window with zero hand presence
    out: SlidingWindowOutput | None = None
    for i in range(10):
        out = buf.add_frame(_create_synthetic_frame(has_hands=False, timestamp_ms=float(i)))

    # Must be suppressed
    assert out is None


def test_get_current_window_peek() -> None:
    """Verifies get_current_window allows peeking without advancing counters."""
    buf = SlidingWindowBuffer(window_size=10, stride=5)

    assert buf.get_current_window() is None

    for i in range(10):
        buf.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))

    peek1 = buf.get_current_window()
    assert peek1 is not None
    assert peek1.window_index == 0

    peek2 = buf.get_current_window()
    assert peek2 is not None
    # Window index has not advanced
    assert peek2.window_index == 0


def test_add_invalid_frame_raises() -> None:
    """Verifies that non-NormalizedFrame input raises ValueError."""
    buf = SlidingWindowBuffer()
    with pytest.raises(ValueError, match="Expected NormalizedFrame instance"):
        buf.add_frame(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected NormalizedFrame instance"):
        buf.add_frame("not_a_frame")  # type: ignore[arg-type]


def test_custom_joint_selection_tensor_shapes() -> None:
    """Verifies tensor shape adjusts when pose or hands are excluded."""
    # 1. Pose only: V = 33
    buf_pose = SlidingWindowBuffer(window_size=10, include_pose=True, include_hands=False)
    for i in range(10):
        out_pose = buf_pose.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))
    assert out_pose is not None
    assert out_pose.tensor.shape == (1, 3, 10, 33)

    # 2. Hands only: V = 42
    buf_hands = SlidingWindowBuffer(window_size=10, include_pose=False, include_hands=True)
    for i in range(10):
        out_hands = buf_hands.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))
    assert out_hands is not None
    assert out_hands.tensor.shape == (1, 3, 10, 42)


def test_empty_joint_selection_raises() -> None:
    """Verifies that selecting no joint sets raises ValueError."""
    with pytest.raises(ValueError, match="At least one of include_pose, include_hands, or include_face must be True"):
        SlidingWindowBuffer(include_pose=False, include_hands=False, include_face=False)


def test_hands_false_confidence_and_emit_only_valid() -> None:
    """Verifies that include_hands=False yields confidence=1.0 and is not dropped by emit_only_valid."""
    buf = SlidingWindowBuffer(
        window_size=10,
        stride=5,
        include_pose=True,
        include_hands=False,
        emit_only_valid=True,
    )
    out: SlidingWindowOutput | None = None
    for i in range(10):
        out = buf.add_frame(_create_synthetic_frame(has_hands=False, timestamp_ms=float(i)))

    assert out is not None
    assert out.confidence == 1.0
    assert out.is_valid is True
    assert buf.emitted_window_count == 1


def test_emitted_window_count_with_dropped_windows() -> None:
    """Verifies emitted_window_count increments only for windows actually emitted."""
    buf = SlidingWindowBuffer(
        window_size=10,
        stride=5,
        confidence_threshold=0.5,
        emit_only_valid=True,
    )
    # 10 invalid frames -> window 0 evaluated, dropped
    for i in range(10):
        res = buf.add_frame(_create_synthetic_frame(has_hands=False, timestamp_ms=float(i)))
        assert res is None

    assert buf.window_index == 1
    assert buf.emitted_window_count == 0

    # 5 valid frames -> window 1 evaluated, emitted
    out = None
    for i in range(10, 15):
        out = buf.add_frame(_create_synthetic_frame(has_hands=True, timestamp_ms=float(i)))

    assert out is not None
    assert out.window_index == 1
    assert buf.window_index == 2
    assert buf.emitted_window_count == 1


def test_peek_sequence_index_coherence() -> None:
    """Verifies that peeking right after emission returns emitted index, and mid-stride returns next index."""
    buf = SlidingWindowBuffer(window_size=10, stride=5)
    for i in range(10):
        w0 = buf.add_frame(_create_synthetic_frame(timestamp_ms=float(i)))
    assert w0 is not None
    assert w0.window_index == 0

    # Immediately after emission, peek returns the index of the window currently in buffer (0)
    peek0 = buf.get_current_window()
    assert peek0 is not None
    assert peek0.window_index == 0

    # Push 2 frames: buffer is now in-flight towards window 1
    buf.add_frame(_create_synthetic_frame(timestamp_ms=10.0))
    buf.add_frame(_create_synthetic_frame(timestamp_ms=11.0))
    peek_inflight = buf.get_current_window()
    assert peek_inflight is not None
    assert peek_inflight.window_index == 1


def test_sliding_window_with_face() -> None:
    """Verifies tensor shape with include_face=True."""
    buf = SlidingWindowBuffer(window_size=10, stride=5, include_face=True)

    face = np.ones((40, 3), dtype=np.float32)
    out = None
    for i in range(10):
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[POSE_LEFT_SHOULDER] = [-0.5, 0.0, 0.0]
        pose[POSE_RIGHT_SHOULDER] = [0.5, 0.0, 0.0]
        hand = np.zeros((21, 3), dtype=np.float32)
        frame = NormalizedFrame(
            pose=pose,
            left_hand=hand,
            right_hand=hand,
            face=face,
            shoulder_distance=1.0,
            root_joint=np.zeros(3, dtype=np.float32),
            timestamp_ms=float(i),
            is_left_hand_visible=True,
            is_right_hand_visible=True,
        )
        out = buf.add_frame(frame)

    assert out is not None
    # 33 + 21 + 21 + 40 = 115 joints
    assert out.tensor.shape == (1, 3, 10, 115)


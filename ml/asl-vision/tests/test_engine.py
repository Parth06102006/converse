"""Unit tests verifying ASLVisionEngine real-time inference, confidence gating, debouncing, and ONNX parity."""

from pathlib import Path

import numpy as np
import pytest
import torch

from asl_vision.engine import (
    DEFAULT_WLASL_100_GLOSSES,
    ASLVisionEngine,
    EngineConfig,
    RestingPoseDetector,
    SignDetection,
    resolve_checkpoint_path,
)
from asl_vision.models.stgcn import STGCN
from asl_vision.models.tgcn_wlasl import TGCNModel
from asl_vision.normalization import NormalizedFrame
from scripts.export_onnx import export_stgcn_to_onnx


def make_dummy_normalized_frame(
    timestamp_ms: float = 100.0,
    moving: bool = True,
) -> NormalizedFrame:
    """Create a synthetic NormalizedFrame for testing with moving or static hands."""
    t_sec = timestamp_ms / 1000.0
    if moving:
        motion_x = float(0.25 * np.sin(2.0 * np.pi * t_sec))
        motion_y = float(-0.2 + 0.25 * np.cos(2.0 * np.pi * t_sec))
    else:
        motion_x = 0.0
        motion_y = 0.0

    pose = np.random.randn(33, 3).astype(np.float32) * 0.01
    pose[11] = np.array([-0.3, 0.0, 0.0], dtype=np.float32)  # L Shoulder
    pose[12] = np.array([0.3, 0.0, 0.0], dtype=np.float32)   # R Shoulder
    pose[23] = np.array([-0.2, 1.2, 0.0], dtype=np.float32)  # L Hip
    pose[24] = np.array([0.2, 1.2, 0.0], dtype=np.float32)   # R Hip
    pose[15, 0] = -0.2 + motion_x
    pose[15, 1] = motion_y
    pose[16, 0] = 0.2 + motion_x
    pose[16, 1] = motion_y

    left_hand = np.random.randn(21, 3).astype(np.float32) * 0.01
    left_hand[:, 0] += motion_x
    left_hand[:, 1] += motion_y
    right_hand = np.random.randn(21, 3).astype(np.float32) * 0.01
    right_hand[:, 0] += motion_x
    right_hand[:, 1] += motion_y
    face = np.random.randn(20, 3).astype(np.float32) * 0.01
    return NormalizedFrame(
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face,
        shoulder_distance=0.4,
        root_joint=np.zeros(3, dtype=np.float32),
        timestamp_ms=timestamp_ms,
        is_left_hand_visible=True,
        is_right_hand_visible=True,
    )


def test_sign_detection_contract_serialization() -> None:
    """Verifies that SignDetection serializes to @converse/contracts SignDetection schema."""
    detection = SignDetection(
        gloss="hello",
        confidence=0.954,
        start_time_ms=1000.0,
        end_time_ms=2000.0,
    )
    serialized = detection.to_dict()

    assert serialized["gloss"] == "hello"
    assert serialized["confidence"] == pytest.approx(0.954)
    assert serialized["startTimeMs"] == 1000.0
    assert serialized["endTimeMs"] == 2000.0

    # Ensure required contract keys are present
    assert set(serialized.keys()) == {"gloss", "confidence", "startTimeMs", "endTimeMs"}


def test_engine_initialization_defaults() -> None:
    """Verifies engine default configuration and vocabulary setup."""
    engine = ASLVisionEngine()
    assert len(engine.vocabulary) == 100
    assert engine.vocabulary[0] == DEFAULT_WLASL_100_GLOSSES[0]
    assert engine.last_detection is None
    assert engine.last_detection_time_ms == -1.0


def test_engine_streaming_window_accumulation() -> None:
    """Verifies that sliding window requires full window size before running inference."""
    config = EngineConfig(window_size=30, stride=5, min_confidence=0.5)
    engine = ASLVisionEngine(config=config)

    # Push 29 frames: no detection should be emitted
    for i in range(29):
        norm_frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3)
        detections = engine.process_normalized_frame(norm_frame, timestamp_ms=i * 33.3)
        assert detections == []

    # Frame 30 completes the first window
    norm_frame = make_dummy_normalized_frame(timestamp_ms=29 * 33.3)
    detections = engine.process_normalized_frame(norm_frame, timestamp_ms=29 * 33.3)
    # Output is either [] (if confidence below threshold) or list of 1 detection
    assert isinstance(detections, list)
    assert len(detections) in (0, 1)


def test_engine_confidence_gating() -> None:
    """Verifies that detections below min_confidence threshold are gated."""
    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.9999,  # Unattainably high threshold
        num_classes=5,
        vocabulary=["book", "drink", "chair", "go", "hello"],
    )
    engine = ASLVisionEngine(config=config)

    for i in range(35):
        norm_frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3)
        detections = engine.process_normalized_frame(norm_frame, timestamp_ms=i * 33.3)
        assert detections == []


def test_engine_successful_detection_with_mock_model() -> None:
    """Verifies successful candidate sign emission when model predicts high confidence."""
    num_classes = 4
    vocab = ["water", "food", "help", "thank you"]

    class HighConfidenceMock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Emit huge logit for class 2 ("help")
            logits = torch.zeros(x.shape[0], num_classes)
            logits[:, 2] = 20.0
            return logits

    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.80,
        num_classes=num_classes,
        vocabulary=vocab,
    )
    mock_model = HighConfidenceMock()
    engine = ASLVisionEngine(config=config, model=mock_model)

    detected: list[SignDetection] = []
    for i in range(30):
        norm_frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3)
        res = engine.process_normalized_frame(norm_frame, timestamp_ms=i * 33.3)
        detected.extend(res)

    assert len(detected) == 1
    d = detected[0]
    assert d.gloss == "help"
    assert d.confidence > 0.99
    assert d.start_time_ms == pytest.approx(0.0)
    assert d.end_time_ms == pytest.approx(29 * 33.3)


def test_engine_temporal_debouncing() -> None:
    """Verifies that debounce cooldown prevents rapid duplicate detections."""
    class ConstantMock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 2)
            logits[:, 0] = 10.0
            return logits

    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.5,
        min_detection_interval_ms=1000.0,  # 1 second cooldown
        num_classes=2,
        vocabulary=["yes", "no"],
    )
    engine = ASLVisionEngine(config=config, model=ConstantMock())

    all_detections: list[SignDetection] = []
    # 40 frames @ 33.3ms = 1333ms total
    for i in range(40):
        t_ms = i * 33.3
        norm_frame = make_dummy_normalized_frame(timestamp_ms=t_ms)
        res = engine.process_normalized_frame(norm_frame, timestamp_ms=t_ms)
        all_detections.extend(res)

    # First window finishes at frame 30 (999ms). Second window at frame 35 (1165ms) is only 166ms later,
    # so it should be suppressed by the 1000ms cooldown.
    assert len(all_detections) == 1


def test_engine_reset_clears_state() -> None:
    """Verifies that engine.reset() completely clears buffers and detection history."""
    engine = ASLVisionEngine()
    for i in range(15):
        norm_frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3)
        engine.process_normalized_frame(norm_frame, timestamp_ms=i * 33.3)

    assert len(engine.buffer) == 15
    assert engine.frame_counter == 15

    engine.reset()
    assert len(engine.buffer) == 0
    assert engine.frame_counter == 0
    assert engine.last_detection is None
    assert engine.last_detection_time_ms == -1.0


def test_engine_with_onnx_backend(tmp_path: Path) -> None:
    """Verifies ASLVisionEngine running inference against an exported ONNX model."""
    onnx_file = tmp_path / "model.onnx"
    pytorch_model = STGCN(
        in_channels=3,
        num_classes=5,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1), (32, 2)],
    )

    export_stgcn_to_onnx(
        model=pytorch_model,
        output_path=onnx_file,
        num_nodes=75,
        temporal_window_size=30,
        verify=True,
    )

    vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.01,  # low threshold so whatever ONNX predicts passes
        num_classes=5,
        vocabulary=vocab,
        use_onnx=True,
        onnx_path=onnx_file,
    )
    engine = ASLVisionEngine(config=config)
    assert engine.onnx_session is not None

    detections: list[SignDetection] = []
    for i in range(30):
        norm_frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3)
        res = engine.process_normalized_frame(norm_frame, timestamp_ms=i * 33.3)
        detections.extend(res)

    assert len(detections) == 1
    assert detections[0].gloss in vocab
    assert 0.0 <= detections[0].confidence <= 1.0


def test_engine_motion_energy_gating_rejects_static_hands() -> None:
    """Verifies that temporal motion energy gating (< 0.015) suppresses static frames."""
    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.10,
        motion_threshold=0.015,
    )
    engine = ASLVisionEngine(config=config)

    detections: list[SignDetection] = []
    # 35 static frames with variance < 0.015
    for i in range(35):
        frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3, moving=False)
        res = engine.process_normalized_frame(frame, timestamp_ms=i * 33.3)
        detections.extend(res)

    assert detections == [], f"Expected static hands to be gated, got: {detections}"


def test_engine_motion_energy_gating_rejects_static_orange_at_chin() -> None:
    """Verifies hand held static at chin (classic ORANGE bias) is rejected by motion gating."""
    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.10,
        motion_threshold=0.015,
    )
    engine = ASLVisionEngine(config=config)

    detections: list[SignDetection] = []
    # Hand held static at chin level across 35 frames
    for i in range(35):
        frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3, moving=False)
        # Position hand at chin (y around -0.3)
        if frame.right_hand is not None:
            frame.right_hand[:, 0] = 0.0
            frame.right_hand[:, 1] = -0.3
        res = engine.process_normalized_frame(frame, timestamp_ms=i * 33.3)
        detections.extend(res)

    assert detections == [], f"Static hand at chin must not trigger detection, got: {detections}"


def test_engine_hands_below_chest_suppressed() -> None:
    """Verifies that hands resting below chest level are suppressed as idle."""
    config = EngineConfig(
        window_size=30,
        stride=5,
        min_confidence=0.10,
    )
    engine = ASLVisionEngine(config=config)

    detections: list[SignDetection] = []
    # Hands placed at lap / desk level (y = 1.3, below hips at 1.2)
    for i in range(35):
        frame = make_dummy_normalized_frame(timestamp_ms=i * 33.3, moving=False)
        frame.pose[15, 1] = 1.3
        frame.pose[16, 1] = 1.3
        res = engine.process_normalized_frame(frame, timestamp_ms=i * 33.3)
        detections.extend(res)

    assert detections == []


def test_engine_resting_pose_detector_stationary_suppression() -> None:
    """Verifies that wrists stationary for > 300ms suppress sign emission."""
    detector = RestingPoseDetector(stationary_duration_ms=300.0, wrist_movement_threshold=0.012)
    assert detector.last_movement_timestamp_ms == -1.0

    # Frame 0: active moving frame
    f0 = make_dummy_normalized_frame(timestamp_ms=0.0, moving=True)
    is_resting_0 = detector.update(f0, timestamp_ms=0.0)
    assert not is_resting_0

    # 12 frames of stationary wrists (12 * 33.3 = 400ms > 300ms)
    # Note: first static frame transitions from moving pose (disp ~0.05),
    # so last_movement resets at t=33.3ms; resting begins at t>333.3ms.
    for i in range(1, 13):
        t = i * 33.3
        f_static = make_dummy_normalized_frame(timestamp_ms=t, moving=False)
        is_resting = detector.update(f_static, timestamp_ms=t)
        if t > 350.0:
            assert is_resting, f"Expected resting at t={t}ms"

    # Moving frame at 500ms restores active state
    f_active = make_dummy_normalized_frame(timestamp_ms=500.0, moving=True)
    is_resting_after = detector.update(f_active, timestamp_ms=500.0)
    assert not is_resting_after


def test_engine_checkpoint_loading_tgcn_model() -> None:
    """Verifies default engine loads tgcn_asl100.bin into TGCNModel in eval mode."""
    engine = ASLVisionEngine()
    assert isinstance(engine.pytorch_model, TGCNModel)
    assert engine.pytorch_model.training is False
    assert len(engine.vocabulary) == 100
    assert engine.vocabulary[0] == "book"
    assert engine.vocabulary[27] == "orange"


def test_engine_checkpoint_fallback_when_missing() -> None:
    """Verifies engine gracefully falls back to STGCN when checkpoint does not exist."""
    config = EngineConfig(checkpoint_path="non_existent/path/model.bin")
    engine = ASLVisionEngine(config=config)
    assert isinstance(engine.pytorch_model, STGCN)
    assert engine.pytorch_model.training is False


def test_resolve_checkpoint_path() -> None:
    """Verifies resolve_checkpoint_path finds valid models and handles None gracefully."""
    resolved = resolve_checkpoint_path()
    assert resolved is not None
    assert resolved.is_file()
    assert resolved.name == "tgcn_asl100.bin"

    missing = resolve_checkpoint_path("definitely_not_a_valid_checkpoint.bin")
    assert missing is None


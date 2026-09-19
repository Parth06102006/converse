"""Unit tests verifying ASLVisionEngine real-time inference, confidence gating, debouncing, and ONNX parity."""

from pathlib import Path

import numpy as np
import pytest
import torch

from asl_vision.engine import (
    DEFAULT_WLASL_100_GLOSSES,
    ASLVisionEngine,
    EngineConfig,
    SignDetection,
)
from asl_vision.models.stgcn import STGCN
from asl_vision.normalization import NormalizedFrame
from scripts.export_onnx import export_stgcn_to_onnx


def make_dummy_normalized_frame(timestamp_ms: float = 100.0) -> NormalizedFrame:
    """Create a synthetic NormalizedFrame for testing."""
    pose = np.random.randn(33, 3).astype(np.float32) * 0.1
    left_hand = np.random.randn(21, 3).astype(np.float32) * 0.05
    right_hand = np.random.randn(21, 3).astype(np.float32) * 0.05
    face = np.random.randn(20, 3).astype(np.float32) * 0.02
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

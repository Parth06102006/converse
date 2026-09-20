"""Unit tests for the pretrained WLASL-100 Temporal Graph Convolutional Network (TGCN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from asl_vision.landmarks import ExtractedLandmarks
from asl_vision.models.tgcn_wlasl import (
    UPPER_BODY_POSE_INDICES,
    WLASL_100_GLOSSES,
    GC_Block,
    GCN_muti_att,
    GraphConvolution_att,
    TGCNWLASLClassifier,
    _extract_upper_body_pose,
)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "tgcn_asl100.bin"


def test_wlasl_100_glosses_invariants() -> None:
    """Verify canonical 100 gloss vocabulary list integrity."""
    assert len(WLASL_100_GLOSSES) == 100
    assert len(set(WLASL_100_GLOSSES)) == 100, "Gloss list contains duplicates"
    assert "BOOK" in WLASL_100_GLOSSES
    assert "DRINK" in WLASL_100_GLOSSES
    assert "COMPUTER" in WLASL_100_GLOSSES
    assert "DOCTOR" in WLASL_100_GLOSSES
    assert "FAMILY" in WLASL_100_GLOSSES
    assert "WORK" in WLASL_100_GLOSSES
    assert "HELP" in WLASL_100_GLOSSES
    assert "YES" in WLASL_100_GLOSSES
    assert "NO" in WLASL_100_GLOSSES


def test_upper_body_pose_indices() -> None:
    """Verify 13 upper-body pose keypoint indices."""
    assert len(UPPER_BODY_POSE_INDICES) == 13
    assert 0 in UPPER_BODY_POSE_INDICES  # Nose
    assert 11 in UPPER_BODY_POSE_INDICES  # Left shoulder
    assert 12 in UPPER_BODY_POSE_INDICES  # Right shoulder
    assert 15 in UPPER_BODY_POSE_INDICES  # Left wrist
    assert 16 in UPPER_BODY_POSE_INDICES  # Right wrist


def test_graph_convolution_layer_forward() -> None:
    """Verify GraphConvolution_att layer dimensions and gradient flow."""
    layer = GraphConvolution_att(in_features=100, out_features=64)
    x = torch.randn(4, 55, 100)
    out = layer(x)
    assert out.shape == (4, 55, 64)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
    assert layer.att.grad is not None
    assert not torch.isnan(layer.att.grad).any()


def test_gc_block_residual_forward() -> None:
    """Verify GC_Block residual connection and normalization."""
    block = GC_Block(in_features=64, p_dropout=0.1, is_resi=True)
    block.eval()
    x = torch.randn(2, 55, 64)
    out = block(x)
    assert out.shape == (2, 55, 64)


def test_gcn_muti_att_forward() -> None:
    """Verify GCN_muti_att end-to-end forward pass with 55 joints and 100 features."""
    model = GCN_muti_att(
        input_feature=100,
        hidden_feature=32,
        num_class=100,
        num_stage=2,
    )
    model.eval()
    x = torch.randn(3, 55, 100)
    out = model(x)
    assert out.shape == (3, 100)


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_loading() -> None:
    """Verify TGCNWLASLClassifier loads checkpoint weights with strict parameter matching."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.45)
    assert len(classifier.vocab) == 100
    assert classifier.model.training is False


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_predict_insufficient_frames() -> None:
    """Verify classifier requires minimum frames before returning predictions."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.45)

    dummy_extracted = ExtractedLandmarks(
        pose=np.zeros((33, 3), dtype=np.float32),
        left_hand=np.zeros((21, 3), dtype=np.float32),
        right_hand=np.zeros((21, 3), dtype=np.float32),
        face=None,
        facial_contours=None,
        face_blendshapes=None,
        pose_world=None,
        timestamp_ms=0.0,
        confidence=1.0,
    )

    # Add only 5 frames (< 15 threshold)
    for _ in range(5):
        classifier.add_frame(dummy_extracted)

    gloss, score, top_preds = classifier.predict()
    assert gloss is None
    assert score == 0.0
    assert top_preds == []


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_full_inference() -> None:
    """Verify full inference pass with synthetic skeletal motion."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.01)

    rng = np.random.default_rng(42)

    # Simulate 50 frames of skeletal motion
    for i in range(50):
        pose = rng.standard_normal((33, 3)).astype(np.float32)
        left = rng.standard_normal((21, 3)).astype(np.float32)
        right = rng.standard_normal((21, 3)).astype(np.float32)
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=left,
            right_hand=right,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    gloss, score, top_preds = classifier.predict()

    # Top-3 predictions should be returned
    assert len(top_preds) == 3
    for pred_gloss, pred_prob in top_preds:
        assert isinstance(pred_gloss, str)
        assert pred_gloss in WLASL_100_GLOSSES
        assert 0.0 <= pred_prob <= 1.0

    # Probabilities should be sorted descending
    assert top_preds[0][1] >= top_preds[1][1] >= top_preds[2][1]

    # Best gloss matches top prediction
    if gloss is not None:
        assert gloss == top_preds[0][0]
        assert score == top_preds[0][1]

    # Reset clears the buffer
    classifier.reset()
    assert len(classifier.frame_buffer) == 0


def test_extract_upper_body_pose_bounds_and_missing() -> None:
    """Verify pose extraction reproduces BODY_25 training slot order with -1.0 for missing landmarks."""
    # Case 1: None pose
    out_none = _extract_upper_body_pose(None)
    assert out_none.shape == (13, 2)
    assert np.all(out_none == -1.0)

    # Case 2: Short/truncated pose (< 11 points)
    short_pose = np.zeros((10, 3), dtype=np.float32)
    short_pose[0] = [0.5, 0.5, 0.0]  # Nose at center -> maps to 0.0
    out_short = _extract_upper_body_pose(short_pose)
    assert out_short.shape == (13, 2)
    assert out_short[0, 0] == pytest.approx(0.0)
    assert out_short[0, 1] == pytest.approx(0.0)
    # All remaining BODY_25 slots need landmarks beyond length 10 (or are exact-zero)
    for i in range(1, 13):
        assert out_short[i, 0] == -1.0
        assert out_short[i, 1] == -1.0

    # Case 3: Standard 33-point MediaPipe pose, BODY_25 slot order:
    # [Nose, Neck, RSh, RElb, RWri, LSh, LElb, LWri, MidHip, REye, LEye, REar, LEar]
    full_pose = np.zeros((33, 4), dtype=np.float32)
    full_pose[0] = [0.5, 0.25, 0.0, 1.0]   # Nose
    full_pose[11] = [0.6, 0.40, 0.0, 1.0]  # LShoulder
    full_pose[12] = [0.4, 0.40, 0.0, 1.0]  # RShoulder
    full_pose[13] = [0.65, 0.60, 0.0, 1.0] # LElbow
    full_pose[14] = [0.35, 0.60, 0.0, 1.0] # RElbow
    full_pose[15] = [0.65, 0.75, 0.0, 1.0] # LWrist
    full_pose[16] = [0.35, 0.75, 0.0, 1.0] # RWrist
    out_full = _extract_upper_body_pose(full_pose)
    assert out_full.shape == (13, 2)
    # 0: Nose (0.5, 0.25) -> (0.0, -0.5)
    assert out_full[0, 0] == pytest.approx(0.0)
    assert out_full[0, 1] == pytest.approx(-0.5)
    # 1: Neck = midpoint of shoulders ((0.6+0.4)/2, 0.40) -> (0.0, -0.2)
    assert out_full[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert out_full[1, 1] == pytest.approx(-0.2)
    # 2: RShoulder (MediaPipe 12) at 0.4 -> maps to -0.2
    assert out_full[2, 0] == pytest.approx(-0.2)
    # 4: RWrist (MediaPipe 16) at 0.35 -> maps to -0.3
    assert out_full[4, 0] == pytest.approx(-0.3)
    # 5: LShoulder (MediaPipe 11) at 0.6 -> maps to 0.2
    assert out_full[5, 0] == pytest.approx(0.2)
    # 7: LWrist (MediaPipe 15) at 0.65 -> maps to 0.3
    assert out_full[7, 0] == pytest.approx(0.3)
    # 8: MidHip needs MediaPipe 23/24 (absent) -> stays -1.0
    assert out_full[8, 0] == -1.0
    assert out_full[8, 1] == -1.0


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_rejects_static_hands_motion_gate() -> None:
    """Verify classifier rejects static/frozen hands to suppress spurious false-positive detections."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.45, min_motion=0.015)

    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = [0.5, 0.25, 0.0, 1.0]
    pose[11] = [0.6, 0.40, 0.0, 1.0]
    pose[12] = [0.4, 0.40, 0.0, 1.0]

    # Hand held completely static in active signing space for 30 frames
    static_hand = np.full((21, 3), [0.4, 0.45, 0.0], dtype=np.float32)
    for i in range(30):
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=static_hand,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    gloss, score, top_preds = classifier.predict()
    assert gloss is None, f"Static hand should not trigger detection, got: {gloss}"
    assert score == 0.0
    assert top_preds == []


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_rejects_spread_static_hand_at_chin_for_orange() -> None:
    """Verify hand with finger spread held static at chin (where model has ORANGE bias) is rejected."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.40, min_motion=0.020)

    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = [0.50, 0.25, 0.0, 1.0]   # Nose
    pose[11] = [0.60, 0.40, 0.0, 1.0]  # LShoulder
    pose[12] = [0.40, 0.40, 0.0, 1.0]  # RShoulder

    # Hand with natural finger spread (spatial std > 0.03) held still at chin across 30 frames
    spread_hand = np.zeros((21, 3), dtype=np.float32)
    for j in range(21):
        spread_hand[j] = [0.40 + 0.02 * (j % 5), 0.35 + 0.02 * (j // 5), 0.0]

    for i in range(30):
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=spread_hand,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    gloss, score, top_preds = classifier.predict()
    assert gloss is None, f"Static hand at chin must not trigger false positive, got: {gloss}"
    assert score == 0.0
    assert top_preds == []


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_accepts_moving_hands() -> None:
    """Verify classifier accepts active hands with dynamic kinematic motion above motion gate."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.01, min_motion=0.015)

    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = [0.5, 0.25, 0.0, 1.0]
    pose[11] = [0.6, 0.40, 0.0, 1.0]
    pose[12] = [0.4, 0.40, 0.0, 1.0]

    # Hand performing dynamic sign trajectory across 30 frames
    for i in range(30):
        hand = np.zeros((21, 3), dtype=np.float32)
        hx = 0.35 + 0.10 * np.cos(i * 0.25)
        hy = 0.45 + 0.10 * np.sin(i * 0.25)
        hand[:] = [hx, hy, 0.0]
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=hand,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    _gloss, _score, top_preds = classifier.predict()
    assert len(top_preds) == 3
    assert top_preds[0][1] > 0.01


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_resting_hands_suppression() -> None:
    """Verify hands resting at or below desk/table level (y >= 0.76) are rejected as idle."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.45)

    pose = np.zeros((33, 4), dtype=np.float32)
    # Resting hands at bottom of camera frame (y = 0.82)
    resting_hand = np.full((21, 3), [0.35, 0.82, 0.0], dtype=np.float32)

    for i in range(20):
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=resting_hand,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    # Frame buffer should remain empty because resting hands are treated as idle
    assert len(classifier.frame_buffer) == 0
    assert classifier.active_hand_frames == 0
    gloss, score, top_preds = classifier.predict()
    assert gloss is None
    assert score == 0.0
    assert top_preds == []


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Pretrained TGCN checkpoint not downloaded")
def test_tgcn_classifier_hand_occlusion_continuity() -> None:
    """Verify brief hand occlusion during an active stroke preserves buffer and pose continuity."""
    classifier = TGCNWLASLClassifier(CHECKPOINT_PATH, num_samples=50, min_confidence=0.45)

    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = [0.5, 0.25, 0.0, 1.0]

    # 15 active frames
    for i in range(15):
        hand = np.full((21, 3), [0.35 + 0.05 * np.cos(i * 0.2), 0.45, 0.0], dtype=np.float32)
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=hand,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    assert len(classifier.frame_buffer) == 15
    assert classifier.active_hand_frames == 15

    # 3 frames of brief occlusion (hands drop or disappear momentarily)
    for i in range(15, 18):
        extracted = ExtractedLandmarks(
            pose=pose,
            left_hand=None,
            right_hand=None,
            face=None,
            facial_contours=None,
            face_blendshapes=None,
            pose_world=None,
            timestamp_ms=float(i * 33.3),
            confidence=1.0,
        )
        classifier.add_frame(extracted)

    # Buffer should not be wiped; should preserve continuity
    assert len(classifier.frame_buffer) == 18
    assert classifier.idle_frames == 3

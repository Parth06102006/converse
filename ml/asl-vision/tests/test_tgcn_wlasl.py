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

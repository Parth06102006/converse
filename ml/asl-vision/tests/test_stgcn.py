"""Unit tests verifying ST-GCN architecture, anatomical graph adjacency, and gradient flow."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from asl_vision.models.stgcn import (
    POSE_LEFT_ELBOW,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_RIGHT_ELBOW,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
    STGCN,
    SkeletalGraph,
    SpatialGraphConv,
    STGCNBlock,
    TemporalConv,
)
from asl_vision.normalization import NormalizedFrame
from asl_vision.sliding_window import SlidingWindowBuffer


def test_skeletal_graph_wlasl_75_dimensions() -> None:
    """Verifies that the default WLASL-75 graph produces expected node count and partition dimensions."""
    graph = SkeletalGraph(layout="wlasl_75", strategy="spatial")
    assert graph.num_nodes == 75
    # Strategy 'spatial' produces 3 partitions: root/self, centripetal, centrifugal
    assert graph.A.shape == (3, 75, 75)
    assert graph.A.dtype == np.float32


def test_skeletal_graph_normalization_properties() -> None:
    """Verifies normalization bounds and absence of NaNs/Infs in adjacency matrices."""
    for strategy in ("spatial", "uniform", "distance"):
        graph = SkeletalGraph(layout="wlasl_75", strategy=strategy)
        A = graph.A

        # Must not contain NaN or Inf
        assert not np.isnan(A).any(), f"NaN found in adjacency matrix with strategy {strategy}"
        assert not np.isinf(A).any(), f"Inf found in adjacency matrix with strategy {strategy}"

        # All values should be non-negative
        assert (A >= 0.0).all(), f"Negative values found in adjacency matrix with strategy {strategy}"

        # Check eigenvalue bounds for each partition matrix: max absolute eigenvalue <= 1.0 + eps
        for k in range(A.shape[0]):
            Ak = A[k]
            if np.count_nonzero(Ak) > 0:
                eigvals = np.linalg.eigvals(Ak)
                max_eig = float(np.max(np.abs(eigvals)))
                assert max_eig <= 1.0 + 1e-4, (
                    f"Eigenvalue magnitude {max_eig} exceeds 1.0 for partition {k} under {strategy}"
                )


def test_skeletal_graph_spatial_edge_preservation() -> None:
    """Verifies that directed spatial partitioning preserves all anatomical edges without zeroing out nodes."""
    graph = SkeletalGraph(layout="wlasl_75", strategy="spatial")
    A = graph.A
    assert A.shape == (3, 75, 75)

    # Self-loops partition: exactly 75 ones on diagonal
    assert np.count_nonzero(A[0]) == 75
    assert np.allclose(np.diag(A[0]), 1.0)

    # Centripetal partition (A[1]): exactly 73 directed edges towards root
    assert np.count_nonzero(A[1]) == 73
    # Centrifugal partition (A[2]): exactly 97 directed edges away from / equidistant to root
    assert np.count_nonzero(A[2]) == 97

    # In-degree normalization: For every target node receiving messages, incoming weights must sum to 1.0
    for k in (1, 2):
        in_sums = np.sum(A[k], axis=0)  # column sums
        active_nodes = in_sums > 0
        assert np.allclose(in_sums[active_nodes], 1.0, atol=1e-5)


def test_skeletal_graph_anatomical_connectivity() -> None:
    """Verifies bone connectivity for upper-body pose, bilateral hands, and wrist bridge edges."""
    graph = SkeletalGraph(layout="wlasl_75", strategy="spatial")
    edges_set = set(graph.edges)

    # 1. Upper-body pose connections
    assert (
        (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER) in edges_set
        or (POSE_RIGHT_SHOULDER, POSE_LEFT_SHOULDER) in edges_set
    )
    assert (
        (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW) in edges_set
        or (POSE_LEFT_ELBOW, POSE_LEFT_SHOULDER) in edges_set
    )
    assert (
        (POSE_LEFT_ELBOW, POSE_LEFT_WRIST) in edges_set
        or (POSE_LEFT_WRIST, POSE_LEFT_ELBOW) in edges_set
    )
    assert (
        (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW) in edges_set
        or (POSE_RIGHT_ELBOW, POSE_RIGHT_SHOULDER) in edges_set
    )
    assert (
        (POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST) in edges_set
        or (POSE_RIGHT_WRIST, POSE_RIGHT_ELBOW) in edges_set
    )

    # 2. Left hand internal chains (offset 33)
    left_offset = 33
    # Left wrist to thumb CMC
    assert (left_offset + 0, left_offset + 1) in edges_set
    # Left thumb chain: 1-2, 2-3, 3-4
    assert (left_offset + 1, left_offset + 2) in edges_set
    assert (left_offset + 2, left_offset + 3) in edges_set
    assert (left_offset + 3, left_offset + 4) in edges_set
    # Left index chain: 0-5, 5-6, 6-7, 7-8
    assert (left_offset + 0, left_offset + 5) in edges_set
    assert (left_offset + 5, left_offset + 6) in edges_set

    # 3. Right hand internal chains (offset 54)
    right_offset = 54
    assert (right_offset + 0, right_offset + 1) in edges_set
    assert (right_offset + 0, right_offset + 5) in edges_set

    # 4. Bridge connections: Pose Wrists -> Hand Wrists
    assert (POSE_LEFT_WRIST, left_offset + 0) in edges_set
    assert (POSE_RIGHT_WRIST, right_offset + 0) in edges_set


def test_skeletal_graph_custom_layout() -> None:
    """Verifies that custom skeletal topologies can be constructed cleanly."""
    custom_edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
    graph = SkeletalGraph(
        layout="custom",
        strategy="spatial",
        custom_edges=custom_edges,
        num_nodes=5,
    )
    assert graph.num_nodes == 5
    assert graph.A.shape == (3, 5, 5)
    assert not np.isnan(graph.A).any()


def test_spatial_graph_conv_layer() -> None:
    """Verifies forward output dimensions of SpatialGraphConv layer."""
    graph = SkeletalGraph(layout="wlasl_75", strategy="spatial")
    A_tensor = torch.from_numpy(graph.A).float()

    layer = SpatialGraphConv(in_channels=3, out_channels=64, adjacency_matrix=A_tensor)
    x = torch.randn(2, 3, 30, 75)
    out = layer(x)

    assert out.shape == (2, 64, 30, 75)
    assert out.dtype == torch.float32


def test_temporal_conv_layer() -> None:
    """Verifies TemporalConv output dimensions with and without temporal downsampling."""
    # Stride 1: preserves T
    tconv1 = TemporalConv(channels=64, kernel_size=9, stride=1)
    x = torch.randn(2, 64, 30, 75)
    out1 = tconv1(x)
    assert out1.shape == (2, 64, 30, 75)

    # Stride 2: downsamples T from 30 to 15
    tconv2 = TemporalConv(channels=64, kernel_size=9, stride=2)
    out2 = tconv2(x)
    assert out2.shape == (2, 64, 15, 75)


def test_stgcn_block_residual() -> None:
    """Verifies STGCNBlock residual identity and projection downsampling."""
    graph = SkeletalGraph(layout="wlasl_75", strategy="spatial")
    A_tensor = torch.from_numpy(graph.A).float()

    # Case 1: In == Out, stride 1 (Identity residual)
    block1 = STGCNBlock(in_channels=64, out_channels=64, adjacency_matrix=A_tensor, stride=1)
    x1 = torch.randn(2, 64, 30, 75)
    out1 = block1(x1)
    assert out1.shape == (2, 64, 30, 75)

    # Case 2: In != Out, stride 2 (1x1 Conv projection residual)
    block2 = STGCNBlock(in_channels=64, out_channels=128, adjacency_matrix=A_tensor, stride=2)
    out2 = block2(x1)
    assert out2.shape == (2, 128, 15, 75)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_classes", [10, 100])
def test_stgcn_model_output_shapes(batch_size: int, num_classes: int) -> None:
    """Verifies model output shape (B, K) for single and batched inputs."""
    model = STGCN(
        in_channels=3,
        num_classes=num_classes,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(32, 1), (64, 2)],  # Lightweight configuration for fast testing
    )
    model.eval()

    # Batched input: (B, 3, 30, 75)
    x = torch.randn(batch_size, 3, 30, 75)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, num_classes)
    assert not torch.isnan(out).any()


def test_stgcn_single_sample_3d_input() -> None:
    """Verifies unbatched 3D tensor input (3, 30, 75) is processed to shape (1, K)."""
    model = STGCN(
        in_channels=3,
        num_classes=100,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(32, 1), (64, 2)],
    )
    model.eval()

    x = torch.randn(3, 30, 75)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 100)


def test_stgcn_feature_extraction_mode() -> None:
    """Verifies extract_features=True returns global average pooled embedding (B, C_final)."""
    model = STGCN(
        in_channels=3,
        num_classes=100,
        num_nodes=75,
        block_channels=[(32, 1), (64, 2)],
    )
    model.eval()

    x = torch.randn(2, 3, 30, 75)
    with torch.no_grad():
        features = model(x, extract_features=True)
    assert features.shape == (2, 64)
    assert not torch.isnan(features).any()


def test_stgcn_backward_gradient_flow() -> None:
    """Verifies backward pass gradient flow and confirms zero NaN gradients across all parameters."""
    model = STGCN(
        in_channels=3,
        num_classes=10,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(32, 1), (64, 2)],
    )
    model.train()

    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 30, 75, requires_grad=True)
    targets = torch.tensor([0, 3, 7, 2], dtype=torch.long)

    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()

    assert not torch.isnan(loss).item()

    # Verify every trainable parameter has a valid, non-NaN gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."


def test_stgcn_learnable_edge_importance_updated() -> None:
    """Verifies that learnable edge importance weights receive gradients during backprop."""
    model = STGCN(
        in_channels=3,
        num_classes=5,
        num_nodes=75,
        block_channels=[(32, 1)],
    )
    model.train()

    x = torch.randn(2, 3, 30, 75)
    targets = torch.tensor([1, 4], dtype=torch.long)

    logits = model(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Find edge_importance parameters and assert non-zero, non-NaN gradients
    found_edge_param = False
    for name, param in model.named_parameters():
        if "edge_importance" in name:
            found_edge_param = True
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert torch.count_nonzero(param.grad) > 0

    assert found_edge_param, "No edge_importance parameters were located in model."


def test_stgcn_end_to_end_with_sliding_window_buffer() -> None:
    """Verifies that sliding window output tensor feeds directly into ST-GCN without transformation."""
    buf = SlidingWindowBuffer(window_size=30, stride=5)
    model = STGCN(
        in_channels=3,
        num_classes=100,
        num_nodes=75,
        block_channels=[(32, 1), (64, 2)],
    )
    model.eval()

    # Feed 30 synthetic frames into buffer
    out = None
    for i in range(30):
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[POSE_LEFT_SHOULDER] = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
        pose[POSE_RIGHT_SHOULDER] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        frame = NormalizedFrame(
            pose=pose,
            left_hand=np.zeros((21, 3), dtype=np.float32),
            right_hand=np.zeros((21, 3), dtype=np.float32),
            face=None,
            shoulder_distance=1.0,
            root_joint=np.zeros(3, dtype=np.float32),
            timestamp_ms=float(i * 33.3),
            is_left_hand_visible=True,
            is_right_hand_visible=True,
        )
        out = buf.add_frame(frame)

    assert out is not None
    assert out.tensor.shape == (1, 3, 30, 75)

    # Feed directly into model
    with torch.no_grad():
        logits = model(out.tensor)

    assert logits.shape == (1, 100)
    assert not torch.isnan(logits).any()

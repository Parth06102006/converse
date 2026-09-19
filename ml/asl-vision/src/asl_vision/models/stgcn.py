"""Spatial-Temporal Graph Convolutional Network (ST-GCN) for ASL sign recognition.

Implements skeletal graph definition with anatomical upper-body and bilateral hand
joint connectivity, partitioned and normalized graph convolution with learnable edge
importance weights, temporal 1D convolutions over sliding temporal windows, and linear
classification mapping to target gloss classes (e.g. WLASL-100/1000).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

# Standard upper-body keypoint indices (MediaPipe Pose subset)
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Hand wrist joint indices (local index 0 within 21-joint hand coordinate streams)
HAND_WRIST_LOCAL = 0


class SkeletalGraph:
    """Anatomical skeletal graph topology and normalized adjacency construction.

    Models human joints as graph nodes and biological bones/inter-joint connections
    as undirected graph edges. Supports standard 75-keypoint layout (33 MediaPipe Pose +
    21 left hand + 21 right hand joints) as well as custom joint topologies.
    """

    def __init__(
        self,
        layout: str = "wlasl_75",
        strategy: str = "spatial",
        max_hop: int = 1,
        root_node: int | None = None,
        custom_edges: Sequence[tuple[int, int]] | None = None,
        num_nodes: int | None = None,
    ) -> None:
        """Initialize the skeletal graph.

        Args:
            layout: Graph preset name ("wlasl_75", "upper_body_27", or "custom").
            strategy: Adjacency partitioning strategy ("spatial", "uniform", "distance").
            max_hop: Maximum hop distance for neighbor aggregation.
            root_node: Anatomical root joint index for spatial partitioning (e.g. mid-shoulder).
            custom_edges: Explicit edge pairs (u, v) when layout="custom".
            num_nodes: Number of nodes V when layout="custom".
        """
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop

        if layout == "wlasl_75":
            self.num_nodes = 75
            self.edges = self._build_wlasl_75_edges()
            self.root_node = POSE_LEFT_SHOULDER if root_node is None else root_node
        elif layout == "upper_body_27":
            # 6 pose arm joints (shoulders, elbows, wrists) + 21 right hand
            self.num_nodes = 27
            self.edges = self._build_upper_body_27_edges()
            self.root_node = 0 if root_node is None else root_node
        elif layout == "custom":
            if custom_edges is None or num_nodes is None:
                raise ValueError(
                    "custom_edges and num_nodes must be provided when layout='custom'."
                )
            self.num_nodes = num_nodes
            self.edges = list(custom_edges)
            self.root_node = 0 if root_node is None else root_node
        else:
            raise ValueError(f"Unsupported skeletal graph layout: {layout}")

        self.hop_matrix = self._compute_hop_distance_matrix()
        self.A = self._build_adjacency_matrices()

    @staticmethod
    def _build_wlasl_75_edges() -> list[tuple[int, int]]:
        """Construct anatomical bone edges for 75-node skeleton."""
        edges: list[tuple[int, int]] = []

        # 1. Pose Upper Body (0..32)
        # Head and facial contour anchors
        edges.extend(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 7),
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 8),
                (9, 10),
            ]
        )
        # Neck / Collarbone / Torso
        edges.extend(
            [
                (0, 11),  # Nose to left shoulder
                (0, 12),  # Nose to right shoulder
                (11, 12),  # Mid-shoulder / collarbone
                (11, 23),  # Left torso (shoulder to hip)
                (12, 24),  # Right torso (shoulder to hip)
                (23, 24),  # Pelvis / hips
            ]
        )
        # Arms: Shoulders -> Elbows -> Wrists
        edges.extend(
            [
                (11, 13),  # Left upper arm
                (13, 15),  # Left forearm
                (12, 14),  # Right upper arm
                (14, 16),  # Right forearm
            ]
        )
        # Pose hands (secondary wrist anchors in MediaPipe Pose)
        edges.extend(
            [
                (15, 17),
                (15, 19),
                (15, 21),
                (17, 19),
                (16, 18),
                (16, 20),
                (16, 22),
                (18, 20),
            ]
        )
        # Lower body legs/feet (for complete 33 pose topology)
        edges.extend(
            [
                (23, 25),
                (25, 27),
                (27, 29),
                (29, 31),
                (27, 31),
                (24, 26),
                (26, 28),
                (28, 30),
                (30, 32),
                (28, 32),
            ]
        )

        # 2. Left Hand (joints 33..53, offset 33)
        left_offset = 33
        # Wrist to finger roots
        edges.extend(
            [
                (left_offset + 0, left_offset + 1),  # Wrist to thumb CMC
                (left_offset + 0, left_offset + 5),  # Wrist to index MCP
                (left_offset + 0, left_offset + 9),  # Wrist to middle MCP
                (left_offset + 0, left_offset + 13),  # Wrist to ring MCP
                (left_offset + 0, left_offset + 17),  # Wrist to pinky MCP
            ]
        )
        # Finger bone chains
        edges.extend(
            [
                # Thumb
                (left_offset + 1, left_offset + 2),
                (left_offset + 2, left_offset + 3),
                (left_offset + 3, left_offset + 4),
                # Index
                (left_offset + 5, left_offset + 6),
                (left_offset + 6, left_offset + 7),
                (left_offset + 7, left_offset + 8),
                # Middle
                (left_offset + 9, left_offset + 10),
                (left_offset + 10, left_offset + 11),
                (left_offset + 11, left_offset + 12),
                # Ring
                (left_offset + 13, left_offset + 14),
                (left_offset + 14, left_offset + 15),
                (left_offset + 15, left_offset + 16),
                # Pinky
                (left_offset + 17, left_offset + 18),
                (left_offset + 18, left_offset + 19),
                (left_offset + 19, left_offset + 20),
            ]
        )
        # Palm MCP transverse arch
        edges.extend(
            [
                (left_offset + 5, left_offset + 9),
                (left_offset + 9, left_offset + 13),
                (left_offset + 13, left_offset + 17),
            ]
        )

        # 3. Right Hand (joints 54..74, offset 54)
        right_offset = 54
        # Wrist to finger roots
        edges.extend(
            [
                (right_offset + 0, right_offset + 1),  # Wrist to thumb CMC
                (right_offset + 0, right_offset + 5),  # Wrist to index MCP
                (right_offset + 0, right_offset + 9),  # Wrist to middle MCP
                (right_offset + 0, right_offset + 13),  # Wrist to ring MCP
                (right_offset + 0, right_offset + 17),  # Wrist to pinky MCP
            ]
        )
        # Finger bone chains
        edges.extend(
            [
                # Thumb
                (right_offset + 1, right_offset + 2),
                (right_offset + 2, right_offset + 3),
                (right_offset + 3, right_offset + 4),
                # Index
                (right_offset + 5, right_offset + 6),
                (right_offset + 6, right_offset + 7),
                (right_offset + 7, right_offset + 8),
                # Middle
                (right_offset + 9, right_offset + 10),
                (right_offset + 10, right_offset + 11),
                (right_offset + 11, right_offset + 12),
                # Ring
                (right_offset + 13, right_offset + 14),
                (right_offset + 14, right_offset + 15),
                (right_offset + 15, right_offset + 16),
                # Pinky
                (right_offset + 17, right_offset + 18),
                (right_offset + 18, right_offset + 19),
                (right_offset + 19, right_offset + 20),
            ]
        )
        # Palm MCP transverse arch
        edges.extend(
            [
                (right_offset + 5, right_offset + 9),
                (right_offset + 9, right_offset + 13),
                (right_offset + 13, right_offset + 17),
            ]
        )

        # 4. Anatomical Bridges: Pose Wrists <-> Detailed Hand Wrists
        edges.extend(
            [
                (POSE_LEFT_WRIST, left_offset + 0),  # Left arm to left hand wrist
                (POSE_RIGHT_WRIST, right_offset + 0),  # Right arm to right hand wrist
            ]
        )

        return edges

    @staticmethod
    def _build_upper_body_27_edges() -> list[tuple[int, int]]:
        """Construct bone edges for a compact 27-node skeleton."""
        # Nodes 0..5: shoulders (0, 1), elbows (2, 3), wrists (4, 5)
        # Nodes 6..26: 21 hand joints attached to right wrist (5)
        edges: list[tuple[int, int]] = [
            (0, 1),  # Shoulders
            (0, 2),  # Left shoulder to left elbow
            (2, 4),  # Left elbow to left wrist
            (1, 3),  # Right shoulder to right elbow
            (3, 5),  # Right elbow to right wrist
            (5, 6),  # Right wrist to hand root
        ]
        # Hand internal edges (offset 6)
        hand_off = 6
        edges.extend(
            [
                (hand_off + 0, hand_off + 1),
                (hand_off + 1, hand_off + 2),
                (hand_off + 2, hand_off + 3),
                (hand_off + 3, hand_off + 4),
                (hand_off + 0, hand_off + 5),
                (hand_off + 5, hand_off + 6),
                (hand_off + 6, hand_off + 7),
                (hand_off + 7, hand_off + 8),
                (hand_off + 0, hand_off + 9),
                (hand_off + 9, hand_off + 10),
                (hand_off + 10, hand_off + 11),
                (hand_off + 11, hand_off + 12),
                (hand_off + 0, hand_off + 13),
                (hand_off + 13, hand_off + 14),
                (hand_off + 14, hand_off + 15),
                (hand_off + 15, hand_off + 16),
                (hand_off + 0, hand_off + 17),
                (hand_off + 17, hand_off + 18),
                (hand_off + 18, hand_off + 19),
                (hand_off + 19, hand_off + 20),
            ]
        )
        return edges

    def _compute_hop_distance_matrix(self) -> np.ndarray:
        """Compute shortest path hop distance matrix using BFS across all joint pairs."""
        adj: dict[int, list[int]] = {i: [] for i in range(self.num_nodes)}
        for u, v in self.edges:
            if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
                adj[u].append(v)
                adj[v].append(u)

        dist = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=np.float32)
        for i in range(self.num_nodes):
            dist[i, i] = 0.0
            queue: deque[int] = deque([i])
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if dist[i, neighbor] == np.inf:
                        dist[i, neighbor] = dist[i, curr] + 1.0
                        queue.append(neighbor)

        return dist

    @staticmethod
    def _normalize_digraph(A: np.ndarray) -> np.ndarray:
        """Compute column (incoming-degree) normalized adjacency matrix for directed graph partitions.

        For einsum contraction 'nkctv,kvw->nctw', w is the target node receiving
        messages from source node v. Normalizing columns by their in-degree ensures
        that incoming feature weights for each target node sum to 1.0 without erasing
        edges connected to zero-outdegree nodes.
        """
        in_degree = np.sum(A, axis=0)
        d_inv = np.zeros_like(in_degree, dtype=np.float32)
        nonzero_mask = in_degree > 0
        d_inv[nonzero_mask] = 1.0 / in_degree[nonzero_mask]
        return (A * d_inv[np.newaxis, :]).astype(np.float32)

    @staticmethod
    def _normalize_undigraph(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute symmetric normalized adjacency matrix D^{-1/2} A D^{-1/2} for undirected graphs."""
        degree = np.sum(A, axis=1)
        d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nonzero_mask = degree > 0
        d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask] + eps)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)

    def _build_adjacency_matrices(self) -> np.ndarray:
        """Construct normalized adjacency tensor according to configured partitioning strategy."""
        V = self.num_nodes
        hop = self.hop_matrix

        if self.strategy == "uniform":
            # Single partition: D^{-1/2} (A + I) D^{-1/2}
            adj = np.zeros((V, V), dtype=np.float32)
            for u, v in self.edges:
                if 0 <= u < V and 0 <= v < V:
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
            adj_with_self = adj + np.eye(V, dtype=np.float32)
            norm_adj = self._normalize_undigraph(adj_with_self)
            # Shape (1, V, V)
            return norm_adj[np.newaxis, :, :]

        elif self.strategy == "distance":
            # Two partitions: hop 0 (self-loop) and hop 1 (1-hop neighbors)
            A = np.zeros((2, V, V), dtype=np.float32)
            # Hop 0
            A[0] = np.eye(V, dtype=np.float32)
            # Hop 1
            adj_1 = (hop == 1).astype(np.float32)
            A[1] = self._normalize_undigraph(adj_1)
            return A

        elif self.strategy == "spatial":
            # Standard ST-GCN spatial configuration partitioning (Yan et al., 2018)
            # K_v = 3 partitions:
            # 0: Root node (self-loop)
            # 1: Centripetal group (neighbor closer to skeletal root node than current node)
            # 2: Centrifugal group (neighbor farther from or equal distance to root)
            A = np.zeros((3, V, V), dtype=np.float32)
            root = self.root_node
            root_dists = hop[root]

            for i in range(V):
                # Partition 0: Self-loops
                A[0, i, i] = 1.0
                for j in range(V):
                    if hop[i, j] == 1.0:
                        if root_dists[j] < root_dists[i]:
                            # Centripetal: closer to root
                            A[1, i, j] = 1.0
                        else:
                            # Centrifugal: farther from root
                            A[2, i, j] = 1.0

            # Partition 0 is self-loops (identity matrix)
            A[1] = self._normalize_digraph(A[1])
            A[2] = self._normalize_digraph(A[2])

            return A

        else:
            raise ValueError(f"Unknown graph adjacency strategy: {self.strategy}")


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolutional layer with learnable edge importance weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency_matrix: torch.Tensor,
    ) -> None:
        """Initialize the SpatialGraphConv layer.

        Args:
            in_channels: Dimensionality of input joint features C_in.
            out_channels: Dimensionality of output joint features C_out.
            adjacency_matrix: Tensor of shape (K_v, V, V).
        """
        super().__init__()
        if adjacency_matrix.ndim != 3:
            raise ValueError(
                f"Expected 3D adjacency matrix (K, V, V), got shape {adjacency_matrix.shape}."
            )

        self.num_partitions = adjacency_matrix.shape[0]
        self.num_nodes = adjacency_matrix.shape[1]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Register normalized adjacency matrix as persistent non-trainable buffer
        self.register_buffer("A", adjacency_matrix.clone().detach().float())

        # Learnable edge importance weights initialized to ones: shape (K_v, V, V)
        self.edge_importance = nn.Parameter(
            torch.ones(self.num_partitions, self.num_nodes, self.num_nodes)
        )

        # 1x1 Convolution projecting input channels to (K_v * out_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_partitions,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial graph convolution.

        Args:
            x: Input tensor of shape (B, C_in, T, V).

        Returns:
            Output tensor of shape (B, C_out, T, V).
        """
        B, _C, T, V = x.shape
        if V != self.num_nodes:
            raise ValueError(
                f"Input joint count {V} does not match graph node count {self.num_nodes}."
            )

        # Apply 1x1 conv across channels: (B, K * C_out, T, V)
        x_proj = self.conv(x)
        # Reshape to (B, K, C_out, T, V)
        x_proj = x_proj.view(B, self.num_partitions, self.out_channels, T, V)

        # Effective adjacency with learnable edge importance weights: (K, V, V)
        A_eff = self.A * self.edge_importance

        # Contract over partition K and node V: (B, C_out, T, V)
        # n: batch, k: partition, c: channels, t: time, v: source node, w: target node
        out = torch.einsum("nkctv,kvw->nctw", x_proj, A_eff)

        out = self.bn(out)
        out = self.relu(out)
        return out


class TemporalConv(nn.Module):
    """1D Temporal Convolution layer operating over temporal frame dimension T."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the TemporalConv layer.

        Args:
            channels: Feature channel depth C.
            kernel_size: Temporal filter kernel size along time dimension (default 9).
            stride: Temporal stride step for downsampling (default 1).
            dropout: Dropout probability applied to temporal representations.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for temporal convolution.

        Args:
            x: Input tensor of shape (B, C, T, V).

        Returns:
            Output tensor of shape (B, C, T', V).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.drop(x)
        return x


class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolutional Block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency_matrix: torch.Tensor,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the STGCNBlock.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            adjacency_matrix: Adjacency tensor of shape (K, V, V).
            temporal_kernel_size: 1D temporal convolution kernel size.
            stride: Temporal downsampling stride.
            dropout: Temporal dropout rate.
        """
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency_matrix)
        self.tcn = TemporalConv(
            out_channels,
            kernel_size=temporal_kernel_size,
            stride=stride,
            dropout=dropout,
        )

        # Residual connection
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for STGCNBlock.

        Args:
            x: Input tensor of shape (B, C_in, T, V).

        Returns:
            Output tensor of shape (B, C_out, T', V).
        """
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for ASL Sign Recognition.

    Accepts spatiotemporal landmark sequence tensors directly matching the output
    from SlidingWindowBuffer: (B, 3, T, V), processes through stacked ST-GCN units,
    performs global spatiotemporal average pooling, and predicts target ASL glosses.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        num_nodes: int = 75,
        temporal_window_size: int = 30,
        layout: str = "wlasl_75",
        graph_strategy: str = "spatial",
        temporal_kernel_size: int = 9,
        dropout: float = 0.2,
        block_channels: Sequence[tuple[int, int]] | None = None,
        custom_edges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize the full ST-GCN model.

        Args:
            in_channels: Input coordinate dimensions (default 3: x, y, z).
            num_classes: Number of target gloss classes (e.g. 100 for WLASL-100).
            num_nodes: Number of skeletal joints V (default 75).
            temporal_window_size: Fixed temporal window length T (default 30 frames).
            layout: Graph preset ("wlasl_75", "upper_body_27", or "custom").
            graph_strategy: Adjacency partitioning strategy ("spatial", "uniform", "distance").
            temporal_kernel_size: Temporal convolution kernel length (default 9).
            dropout: Dropout probability.
            block_channels: Optional custom sequence of (out_channels, stride) tuples.
            custom_edges: Explicit edge pairs when layout="custom".
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.temporal_window_size = temporal_window_size

        # 1. Anatomical Skeletal Graph Adjacency
        if layout == "custom":
            self.graph = SkeletalGraph(
                layout="custom",
                strategy=graph_strategy,
                custom_edges=custom_edges,
                num_nodes=num_nodes,
            )
        else:
            self.graph = SkeletalGraph(
                layout=layout,
                strategy=graph_strategy,
            )
            self.num_nodes = self.graph.num_nodes

        # Convert adjacency numpy array to PyTorch float tensor
        A_tensor = torch.from_numpy(self.graph.A).float()

        # 2. Dynamic Joint Data Batch Normalization across coordinates and joints
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

        # 3. Stack of Spatial-Temporal Convolutional Blocks
        # Default architecture: 9 STGCN blocks with stride-2 temporal downsampling
        if block_channels is None:
            # (out_channels, temporal_stride)
            cfg: list[tuple[int, int]] = [
                (64, 1),
                (64, 1),
                (64, 1),
                (128, 2),
                (128, 1),
                (128, 1),
                (256, 2),
                (256, 1),
                (256, 1),
            ]
        else:
            cfg = list(block_channels)

        blocks: list[nn.Module] = []
        curr_in = in_channels
        for out_c, stride in cfg:
            blocks.append(
                STGCNBlock(
                    in_channels=curr_in,
                    out_channels=out_c,
                    adjacency_matrix=A_tensor,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=stride,
                    dropout=dropout,
                )
            )
            curr_in = out_c

        self.st_blocks = nn.ModuleList(blocks)
        self.final_channels = curr_in

        # 4. Global Average Pooling & Linear Classification Head
        self.head_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(self.final_channels, num_classes)

        # Initialize network weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """Return the base graph adjacency tensor of shape (K, V, V)."""
        return torch.from_numpy(self.graph.A).float()

    def forward(
        self,
        x: torch.Tensor,
        extract_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass for ST-GCN.

        Args:
            x: Input landmark tensor of shape (B, C, T, V) or (C, T, V).
            extract_features: If True, return pooled feature embedding (B, C_final).

        Returns:
            Logits tensor of shape (B, num_classes), or feature tensor (B, C_final).
        """
        if x.ndim == 3:
            # (C, T, V) -> (1, C, T, V)
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(
                f"Expected 3D or 4D tensor (B, C, T, V), got shape {x.shape}."
            )

        B, C, T, V = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Input channel count {C} does not match expected {self.in_channels}."
            )
        if V != self.num_nodes:
            raise ValueError(
                f"Input joint count {V} does not match model joint count {self.num_nodes}."
            )

        # 1. Apply data batch normalization across (V * C)
        # Transpose: (B, C, T, V) -> (B, V, C, T) -> (B, V * C, T)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x = self.data_bn(x)
        # Restore: (B, V * C, T) -> (B, V, C, T) -> (B, C, T, V)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()

        # 2. ST-GCN block pipeline
        for block in self.st_blocks:
            x = block(x)

        # 3. Global Spatiotemporal Average Pooling over (T', V): shape (B, C_final)
        features = x.mean(dim=(2, 3))

        if extract_features:
            return features

        # 4. Classification Head
        out = self.head_drop(features)
        logits = self.fc(out)

        return logits

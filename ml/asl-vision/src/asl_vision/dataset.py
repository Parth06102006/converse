"""WLASL landmark sequence dataset and batch collator for ASL sign recognition.

Handles variable-length landmark frame sequences, temporal resampling/padding/truncation
to fixed length T=30 frames, spatial and temporal data augmentations (temporal scaling,
coordinate jitter, spatial scaling, planar rotation, translation), and DataLoader collation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class WLASLBatch:
    """Collated batch for WLASL landmark sequences.

    Provides tuple unpacking (tensors, labels = batch) and attribute access.
    """

    tensors: torch.Tensor  # Shape (B, C, T, V)
    labels: torch.Tensor  # Shape (B,)
    video_ids: list[str]
    glosses: list[str]

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Allow tuple unpacking: tensors, labels = batch."""
        yield self.tensors
        yield self.labels

    def __getitem__(self, idx: int | str) -> Any:
        """Allow index access [0], [1] or key access ['tensor'], ['label']."""
        if idx in (0, "tensor", "tensors"):
            return self.tensors
        if idx in (1, "label", "labels"):
            return self.labels
        if idx in (2, "video_id", "video_ids"):
            return self.video_ids
        if idx in (3, "gloss", "glosses"):
            return self.glosses
        raise KeyError(f"Invalid batch key or index: {idx}")

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self.tensors.shape[0]


class WLASLDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for WLASL landmark sequences with temporal resampling and augmentation."""

    def __init__(
        self,
        data: Sequence[dict[str, Any]] | str | Path | None = None,
        annotations_file: str | Path | None = None,
        landmarks_dir: str | Path | None = None,
        split: str | None = None,
        target_length: int = 30,
        num_nodes: int = 75,
        temporal_mode: str = "interpolate",
        augment: bool = False,
        jitter_std: float = 0.005,
        scale_range: tuple[float, float] = (0.9, 1.1),
        temporal_scale_range: tuple[float, float] = (0.8, 1.2),
        max_rotation_deg: float = 10.0,
        max_translation: float = 0.02,
        seed: int | None = None,
    ) -> None:
        """Initialize the WLASLDataset.

        Args:
            data: In-memory sequence of sample dicts or directory/path.
            annotations_file: Path to WLASL JSON annotations (e.g. WLASL_v0.3.json).
            landmarks_dir: Directory containing per-video landmark files (.npy or .json).
            split: Filter subset ("train", "val", "test", or None for all).
            target_length: Fixed temporal length T (default 30 frames).
            num_nodes: Expected number of joints V (default 75).
            temporal_mode: Temporal resizing strategy ("interpolate", "pad_truncate", "uniform_sample").
            augment: Whether to apply spatial and temporal data augmentation.
            jitter_std: Standard deviation of coordinate jitter noise.
            scale_range: Min and max factors for random spatial coordinate scaling.
            temporal_scale_range: Min and max factors for temporal duration scaling.
            max_rotation_deg: Maximum random XY planar rotation in degrees.
            max_translation: Maximum random coordinate translation offset.
            seed: Optional random seed for reproducible augmentation.
        """
        self.target_length = target_length
        self.num_nodes = num_nodes
        self.temporal_mode = temporal_mode
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.temporal_scale_range = temporal_scale_range
        self.max_rotation_deg = max_rotation_deg
        self.max_translation = max_translation
        self.rng = np.random.default_rng(seed)

        self.samples: list[dict[str, Any]] = []

        if data is not None:
            if isinstance(data, (str, Path)):
                self._load_from_path(Path(data), split)
            elif isinstance(data, Sequence):
                self.samples = list(data)
            else:
                raise ValueError(f"Unsupported data source type: {type(data)}")
        elif annotations_file is not None:
            self._load_from_wlasl_annotations(
                Path(annotations_file),
                Path(landmarks_dir) if landmarks_dir else None,
                split,
            )

    def _load_from_path(self, path: Path, split: str | None) -> None:
        """Load samples from directory containing .npy files or JSON index."""
        if path.is_file() and path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list):
                for rec in records:
                    if split is None or rec.get("split") == split:
                        self.samples.append(rec)
        elif path.is_dir():
            for f in sorted(path.glob("*.npy")):
                video_id = f.stem
                self.samples.append(
                    {
                        "video_id": video_id,
                        "landmarks_file": str(f),
                        "label": 0,
                        "gloss": video_id,
                    }
                )

    def _load_from_wlasl_annotations(
        self,
        annotations_file: Path,
        landmarks_dir: Path | None,
        split: str | None,
    ) -> None:
        """Parse canonical WLASL_v0.3.json index file."""
        with open(annotations_file, encoding="utf-8") as f:
            wlasl_data = json.load(f)

        gloss_to_label = {entry["gloss"]: idx for idx, entry in enumerate(wlasl_data)}

        for entry in wlasl_data:
            gloss = entry["gloss"]
            label = gloss_to_label[gloss]
            for inst in entry.get("instances", []):
                inst_split = inst.get("split")
                if split is not None and inst_split != split:
                    continue

                video_id = inst.get("video_id")
                landmarks_path: str | None = None
                if landmarks_dir is not None and video_id is not None:
                    candidate_npy = landmarks_dir / f"{video_id}.npy"
                    candidate_json = landmarks_dir / f"{video_id}.json"
                    if candidate_npy.exists():
                        landmarks_path = str(candidate_npy)
                    elif candidate_json.exists():
                        landmarks_path = str(candidate_json)

                self.samples.append(
                    {
                        "video_id": str(video_id),
                        "gloss": gloss,
                        "label": label,
                        "split": inst_split,
                        "landmarks_file": landmarks_path,
                    }
                )

    @classmethod
    def create_synthetic(
        cls,
        num_samples: int = 20,
        num_classes: int = 10,
        num_nodes: int = 75,
        min_length: int = 15,
        max_length: int = 60,
        augment: bool = False,
        seed: int = 42,
    ) -> WLASLDataset:
        """Construct synthetic in-memory dataset for testing and verification."""
        rng = np.random.default_rng(seed)
        samples: list[dict[str, Any]] = []

        for i in range(num_samples):
            length = int(rng.integers(min_length, max_length + 1))
            landmarks = rng.normal(0.0, 0.5, size=(length, num_nodes, 3)).astype(
                np.float32
            )
            label = int(rng.integers(0, num_classes))
            samples.append(
                {
                    "video_id": f"syn_{i:04d}",
                    "gloss": f"gloss_{label}",
                    "label": label,
                    "landmarks": landmarks,
                }
            )

        return cls(
            data=samples,
            num_nodes=num_nodes,
            augment=augment,
            seed=seed,
        )

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)

    def _load_landmarks_tensor(self, sample: dict[str, Any]) -> torch.Tensor:
        """Extract and format raw landmarks as a torch FloatTensor (C, T, V)."""
        raw: Any = sample.get("landmarks")

        if raw is None and "landmarks_file" in sample and sample["landmarks_file"]:
            filepath = Path(sample["landmarks_file"])
            if filepath.suffix == ".npy":
                raw = np.load(filepath)
            elif filepath.suffix == ".json":
                with open(filepath, encoding="utf-8") as f:
                    raw = np.array(json.load(f), dtype=np.float32)

        if raw is None:
            # Fallback zero sequence of target length
            return torch.zeros((3, self.target_length, self.num_nodes), dtype=torch.float32)

        if not isinstance(raw, torch.Tensor):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            arr = raw.detach().cpu().numpy().astype(np.float32)

        # Standardize array layout to (C, T, V)
        # Case 1: (T, V, 3)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Transpose: (T, V, 3) -> (3, T, V)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        # Case 2: (3, T, V)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            tensor = torch.from_numpy(arr.copy())
        # Case 3: (T, 3 * V)
        elif arr.ndim == 2 and arr.shape[1] % 3 == 0:
            T = arr.shape[0]
            V = arr.shape[1] // 3
            arr_3d = arr.reshape(T, V, 3).transpose(2, 0, 1)
            tensor = torch.from_numpy(arr_3d.copy())
        else:
            raise ValueError(
                f"Unsupported landmark array shape: {arr.shape}. "
                "Expected (T, V, 3), (3, T, V), or (T, 3*V)."
            )

        # If joint count V differs from self.num_nodes, adjust appropriately
        C, T, V = tensor.shape
        if V < self.num_nodes:
            # Zero-pad extra joints up to expected num_nodes
            pad_joints = torch.zeros((C, T, self.num_nodes - V), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_joints], dim=2)
        elif V > self.num_nodes:
            # Truncate to expected num_nodes
            tensor = tensor[:, :, : self.num_nodes]

        return tensor

    def _apply_temporal_resampling(
        self,
        tensor: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Resample, pad, or truncate sequence along time dimension to target_len.

        Input shape: (C, T, V). Output shape: (C, target_len, V).
        """
        C, T, V = tensor.shape

        if T == target_len:
            return tensor

        if self.temporal_mode == "interpolate":
            if T == 0:
                return torch.zeros((C, target_len, V), dtype=tensor.dtype)
            if T == 1:
                return tensor.repeat(1, target_len, 1)

            # Flatten (C, V) into channels: (1, C * V, T)
            flat = tensor.reshape(C * V, T).unsqueeze(0)
            # Linear 1D interpolation along time axis
            resampled = F.interpolate(
                flat,
                size=target_len,
                mode="linear",
                align_corners=True,
            )
            # Reshape back to (C, target_len, V)
            return resampled.squeeze(0).reshape(C, target_len, V)

        elif self.temporal_mode == "pad_truncate":
            if T < target_len:
                # Replicate last frame to pad to target_len
                diff = target_len - T
                last_frame = tensor[:, -1:, :]
                pad = last_frame.repeat(1, diff, 1)
                return torch.cat([tensor, pad], dim=1)
            else:
                # Center crop to target_len
                start = (T - target_len) // 2
                return tensor[:, start : start + target_len, :]

        elif self.temporal_mode == "uniform_sample":
            if T <= 1:
                return tensor.repeat(1, target_len, 1)
            indices = np.linspace(0, T - 1, target_len, dtype=int)
            idx_tensor = torch.from_numpy(indices)
            return tensor.index_select(1, idx_tensor)

        else:
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")

    def _apply_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial and temporal augmentations to landmark sequence (C, T, V)."""
        _, T, _ = tensor.shape

        # 1. Temporal Duration Scaling (speed perturbation)
        if self.temporal_scale_range != (1.0, 1.0) and T > 2:
            min_scale, max_scale = self.temporal_scale_range
            scale = float(self.rng.uniform(min_scale, max_scale))
            new_len = max(2, round(T * scale))
            if new_len < T:
                # Random subsegment crop
                max_start = T - new_len
                start_t = int(self.rng.integers(0, max_start + 1))
                tensor = tensor[:, start_t : start_t + new_len, :]
            elif new_len > T:
                # Temporal stretch
                tensor = self._apply_temporal_resampling(tensor, new_len)

        # Resample to fixed target length T=30
        tensor = self._apply_temporal_resampling(tensor, self.target_length)

        # 2. Random 2D/3D Coordinate Jitter (Gaussian noise)
        if self.jitter_std > 0.0:
            noise = torch.randn_like(tensor) * self.jitter_std
            tensor = tensor + noise

        # 3. Random Spatial Scaling
        if self.scale_range != (1.0, 1.0):
            min_s, max_s = self.scale_range
            s = float(self.rng.uniform(min_s, max_s))
            tensor = tensor * s

        # 4. Random XY Planar Rotation around origin
        if self.max_rotation_deg > 0.0:
            angle_rad = float(
                self.rng.uniform(
                    -math.radians(self.max_rotation_deg),
                    math.radians(self.max_rotation_deg),
                )
            )
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            x = tensor[0].clone()
            y = tensor[1].clone()
            tensor[0] = cos_a * x - sin_a * y
            tensor[1] = sin_a * x + cos_a * y

        # 5. Random Coordinate Translation / Offset
        if self.max_translation > 0.0:
            offsets = torch.from_numpy(
                self.rng.uniform(-self.max_translation, self.max_translation, size=(3, 1, 1)).astype(
                    np.float32
                )
            )
            tensor = tensor + offsets

        return tensor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch and process sample at index idx."""
        sample = self.samples[idx]
        tensor = self._load_landmarks_tensor(sample)

        if self.augment:
            tensor = self._apply_augmentations(tensor)
        else:
            tensor = self._apply_temporal_resampling(tensor, self.target_length)

        label = int(sample.get("label", 0))
        video_id = str(sample.get("video_id", f"sample_{idx}"))
        gloss = str(sample.get("gloss", ""))

        return {
            "tensor": tensor.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "gloss": gloss,
        }


def wlasl_collate_fn(batch: Sequence[dict[str, Any]]) -> WLASLBatch:
    """Collate sequence of sample dicts into a structured WLASLBatch.

    Args:
        batch: List of dictionaries emitted by WLASLDataset.__getitem__.

    Returns:
        WLASLBatch instance containing batched (B, C, T, V) tensor and (B,) labels.
    """
    tensors = torch.stack([item["tensor"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    video_ids = [item.get("video_id", "") for item in batch]
    glosses = [item.get("gloss", "") for item in batch]

    return WLASLBatch(
        tensors=tensors,
        labels=labels,
        video_ids=video_ids,
        glosses=glosses,
    )

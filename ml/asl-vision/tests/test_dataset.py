"""Unit tests verifying WLASL landmark dataset, temporal resampling, augmentation, and collation."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from asl_vision.dataset import WLASLBatch, WLASLDataset, wlasl_collate_fn
from asl_vision.models.stgcn import STGCN


def test_synthetic_dataset_creation_and_item_shapes() -> None:
    """Verifies synthetic dataset creation and individual item tensor shapes."""
    dataset = WLASLDataset.create_synthetic(
        num_samples=15,
        num_classes=5,
        num_nodes=75,
        min_length=20,
        max_length=40,
        augment=False,
    )
    assert len(dataset) == 15

    item = dataset[0]
    assert isinstance(item, dict)
    assert "tensor" in item
    assert "label" in item
    assert "video_id" in item
    assert "gloss" in item

    tensor = item["tensor"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 30, 75)
    assert tensor.dtype == torch.float32

    label = item["label"]
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert 0 <= label.item() < 5


@pytest.mark.parametrize("input_len", [1, 10, 30, 65])
@pytest.mark.parametrize("temporal_mode", ["interpolate", "pad_truncate", "uniform_sample"])
def test_variable_length_resampling_modes(input_len: int, temporal_mode: str) -> None:
    """Verifies that sequences of any length are normalized to exactly T=30."""
    raw_landmarks = np.ones((input_len, 75, 3), dtype=np.float32) * 1.5
    data = [{"video_id": "test", "label": 0, "landmarks": raw_landmarks}]

    dataset = WLASLDataset(
        data=data,
        target_length=30,
        num_nodes=75,
        temporal_mode=temporal_mode,
        augment=False,
    )
    item = dataset[0]
    assert item["tensor"].shape == (3, 30, 75)
    assert not torch.isnan(item["tensor"]).any()


def test_zero_length_sequence_handled_gracefully() -> None:
    """Verifies that empty landmark sequences return zero-filled target tensor."""
    data = [{"video_id": "empty", "label": 0, "landmarks": None}]
    dataset = WLASLDataset(data=data, target_length=30, num_nodes=75)
    item = dataset[0]
    assert item["tensor"].shape == (3, 30, 75)
    assert (item["tensor"] == 0.0).all()


def test_coordinate_jitter_augmentation() -> None:
    """Verifies coordinate jitter introduces bounded Gaussian noise."""
    raw = np.zeros((30, 75, 3), dtype=np.float32)
    sample = {"video_id": "test", "label": 0, "landmarks": raw}

    # Deterministic dataset: zero perturbation
    ds_clean = WLASLDataset(
        data=[sample],
        augment=False,
        jitter_std=0.0,
    )
    tensor_clean = ds_clean[0]["tensor"]
    assert (tensor_clean == 0.0).all()

    # Augmented dataset with jitter only
    ds_jitter = WLASLDataset(
        data=[sample],
        augment=True,
        jitter_std=0.05,
        scale_range=(1.0, 1.0),
        temporal_scale_range=(1.0, 1.0),
        max_rotation_deg=0.0,
        max_translation=0.0,
        seed=42,
    )
    tensor_jitter = ds_jitter[0]["tensor"]
    assert not (tensor_jitter == 0.0).all()
    # Jitter standard deviation approximately matches target std
    assert abs(tensor_jitter.std().item() - 0.05) < 0.02


def test_spatial_scaling_augmentation() -> None:
    """Verifies random spatial scaling modifies coordinate magnitude within bounds."""
    raw = np.ones((30, 75, 3), dtype=np.float32)
    sample = {"video_id": "test", "label": 0, "landmarks": raw}

    ds_scale = WLASLDataset(
        data=[sample],
        augment=True,
        jitter_std=0.0,
        scale_range=(1.5, 2.0),
        temporal_scale_range=(1.0, 1.0),
        max_rotation_deg=0.0,
        max_translation=0.0,
        seed=123,
    )
    tensor_scaled = ds_scale[0]["tensor"]
    val = tensor_scaled[0, 0, 0].item()
    assert 1.5 <= val <= 2.0


def test_planar_rotation_augmentation() -> None:
    """Verifies XY planar rotation transforms coordinates while preserving Z channel."""
    # Place unit vector along X axis
    raw = np.zeros((30, 75, 3), dtype=np.float32)
    raw[:, :, 0] = 1.0  # X = 1.0
    raw[:, :, 2] = 0.42  # Z = 0.42
    sample = {"video_id": "test", "label": 0, "landmarks": raw}

    ds_rot = WLASLDataset(
        data=[sample],
        augment=True,
        jitter_std=0.0,
        scale_range=(1.0, 1.0),
        temporal_scale_range=(1.0, 1.0),
        max_rotation_deg=45.0,
        max_translation=0.0,
        seed=456,
    )
    tensor_rot = ds_rot[0]["tensor"]
    # Z coordinate should remain exactly 0.42
    assert torch.allclose(tensor_rot[2], torch.tensor(0.42), atol=1e-5)
    # Norm of (X, Y) should remain 1.0 under rotation
    xy_norm = torch.sqrt(tensor_rot[0] ** 2 + tensor_rot[1] ** 2)
    assert torch.allclose(xy_norm, torch.tensor(1.0), atol=1e-4)


def test_dataloader_batch_collation() -> None:
    """Verifies DataLoader batch collation using wlasl_collate_fn."""
    dataset = WLASLDataset.create_synthetic(num_samples=12, num_classes=4, num_nodes=75)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=wlasl_collate_fn)

    batches: list[WLASLBatch] = list(loader)
    assert len(batches) == 3

    batch = batches[0]
    assert isinstance(batch, WLASLBatch)
    assert batch.batch_size == 4
    assert batch.tensors.shape == (4, 3, 30, 75)
    assert batch.labels.shape == (4,)
    assert len(batch.video_ids) == 4
    assert len(batch.glosses) == 4

    # Test tuple unpacking: tensors, labels = batch
    tensors, labels = batch
    assert tensors.shape == (4, 3, 30, 75)
    assert labels.shape == (4,)

    # Test index and key access
    assert torch.equal(batch[0], batch.tensors)
    assert torch.equal(batch[1], batch.labels)
    assert torch.equal(batch["tensor"], batch.tensors)
    assert torch.equal(batch["label"], batch.labels)


def test_dataset_loading_from_disk(tmp_path: Path) -> None:
    """Verifies dataset loading from .npy landmark files on disk."""
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir()

    # Save two synthetic landmark numpy files
    np.save(landmarks_dir / "clip_001.npy", np.random.randn(25, 75, 3).astype(np.float32))
    np.save(landmarks_dir / "clip_002.npy", np.random.randn(45, 75, 3).astype(np.float32))

    dataset = WLASLDataset(data=landmarks_dir, target_length=30, num_nodes=75)
    assert len(dataset) == 2

    item1 = dataset[0]
    assert item1["tensor"].shape == (3, 30, 75)
    assert item1["video_id"] in ("clip_001", "clip_002")


def test_end_to_end_dataloader_to_stgcn_pipeline() -> None:
    """Verifies full pipeline: Synthetic dataset -> DataLoader collation -> ST-GCN forward & backward."""
    dataset = WLASLDataset.create_synthetic(
        num_samples=8,
        num_classes=5,
        num_nodes=75,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=wlasl_collate_fn)

    model = STGCN(
        in_channels=3,
        num_classes=5,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(32, 1), (64, 2)],
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for tensors, labels in loader:
        optimizer.zero_grad()
        logits = model(tensors)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss).item()
        assert logits.shape == (4, 5)

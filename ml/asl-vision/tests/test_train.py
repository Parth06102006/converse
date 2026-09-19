"""Unit tests verifying ST-GCN training loop, accuracy computation, evaluation, and checkpointing."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from asl_vision.dataset import WLASLDataset, wlasl_collate_fn
from asl_vision.models.stgcn import STGCN
from scripts.train import (
    compute_accuracy,
    evaluate,
    parse_args,
    run_training,
    train_one_epoch,
)


def test_compute_accuracy_exact() -> None:
    """Verifies top-1 and top-5 accuracy calculation logic."""
    # Batch of 4 samples, 5 classes
    logits = torch.tensor([
        [10.0, 1.0, 0.0, 0.0, 0.0],  # Pred 0
        [1.0, 10.0, 0.0, 0.0, 0.0],  # Pred 1
        [0.0, 0.0, 10.0, 0.0, 0.0],  # Pred 2
        [0.0, 0.0, 0.0, 10.0, 0.0],  # Pred 3
    ])
    # 3 correct, 1 wrong
    targets = torch.tensor([0, 1, 2, 4])

    top1, top5 = compute_accuracy(logits, targets, top_k=(1, 5))
    assert top1 == pytest.approx(75.0)
    assert top5 == pytest.approx(100.0)


def test_train_one_epoch_synthetic() -> None:
    """Verifies single training epoch executes forward, backward, and weight updates without NaNs."""
    dataset = WLASLDataset.create_synthetic(num_samples=8, num_classes=4, num_nodes=75)
    loader = DataLoader(dataset, batch_size=4, collate_fn=wlasl_collate_fn)

    model = STGCN(
        in_channels=3,
        num_classes=4,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1)],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss, top1, top5 = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert not torch.isnan(torch.tensor(loss))
    assert loss > 0.0
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top5 <= 100.0


def test_evaluate_synthetic() -> None:
    """Verifies evaluation produces valid loss and accuracy metrics."""
    dataset = WLASLDataset.create_synthetic(num_samples=6, num_classes=3, num_nodes=75)
    loader = DataLoader(dataset, batch_size=3, collate_fn=wlasl_collate_fn)

    model = STGCN(
        in_channels=3,
        num_classes=3,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1)],
    )
    criterion = torch.nn.CrossEntropyLoss()

    val_loss, top1, top5 = evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert not torch.isnan(torch.tensor(val_loss))
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top5 <= 100.0


def test_run_training_lifecycle_and_checkpoints(tmp_path: Path) -> None:
    """Verifies complete multi-epoch training lifecycle and checkpoint persistence."""
    train_ds = WLASLDataset.create_synthetic(num_samples=8, num_classes=3, num_nodes=75, augment=True)
    val_ds = WLASLDataset.create_synthetic(num_samples=4, num_classes=3, num_nodes=75, augment=False)

    ckpt_dir = tmp_path / "checkpoints"
    results = run_training(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=3,
        num_nodes=75,
        temporal_window_size=30,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        device_name="cpu",
        output_dir=ckpt_dir,
    )

    assert Path(results["best_checkpoint_path"]).is_file()
    assert Path(results["last_checkpoint_path"]).is_file()
    assert len(results["history"]) == 2

    # Verify checkpoint contents
    loaded = torch.load(results["best_checkpoint_path"], map_location="cpu")
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert loaded["num_classes"] == 3
    assert loaded["num_nodes"] == 75


def test_train_parse_args_defaults() -> None:
    """Verifies default argument parsing for train script."""
    args = parse_args([])
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.lr == 1e-3
    assert args.num_classes == 100
    assert args.num_nodes == 75
    assert args.device == "auto"

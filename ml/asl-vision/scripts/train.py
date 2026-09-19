"""Supervised training and validation pipeline for ST-GCN sign language recognition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from asl_vision.dataset import WLASLDataset, wlasl_collate_fn
from asl_vision.models.stgcn import STGCN


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: tuple[int, ...] = (1, 5)) -> list[float]:
    """Compute top-k accuracy percentages for model predictions.

    Args:
        logits: Model raw output tensor of shape (B, K).
        targets: Ground truth class index tensor of shape (B,).
        top_k: Tuple of k values to compute.

    Returns:
        List of accuracy percentages matching top_k order.
    """
    with torch.no_grad():
        max_k = min(max(top_k), logits.size(1))
        batch_size = targets.size(0)

        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res: list[float] = []
        for k in top_k:
            if k > logits.size(1):
                res.append(0.0)
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return res


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> tuple[float, float, float]:
    """Execute a single training epoch over the dataset.

    Args:
        model: ST-GCN model.
        dataloader: PyTorch DataLoader yielding WLASLBatch instances.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: Target compute device.
        clip_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Tuple of (average_loss, top1_accuracy, top5_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    for batch in dataloader:
        tensors = batch.tensors.to(device)
        labels = batch.labels.to(device)

        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)

        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        top1, top5 = compute_accuracy(logits, labels, top_k=(1, 5))
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / num_batches,
        total_top1 / num_batches,
        total_top5 / num_batches,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model performance on a validation or test dataset.

    Args:
        model: ST-GCN model.
        dataloader: PyTorch DataLoader yielding WLASLBatch instances.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Tuple of (average_loss, top1_accuracy, top5_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tensors = batch.tensors.to(device)
            labels = batch.labels.to(device)

            logits = model(tensors)
            loss = criterion(logits, labels)

            top1, top5 = compute_accuracy(logits, labels, top_k=(1, 5))
            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / num_batches,
        total_top1 / num_batches,
        total_top5 / num_batches,
    )


def run_training(
    train_dataset: WLASLDataset,
    val_dataset: WLASLDataset,
    num_classes: int = 100,
    num_nodes: int = 75,
    temporal_window_size: int = 30,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    output_dir: str | Path = "checkpoints",
) -> dict[str, Any]:
    """Run full training lifecycle with checkpointing and validation tracking.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        num_classes: Gloss vocabulary size.
        num_nodes: Graph node count V.
        temporal_window_size: Window sequence length T.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Base learning rate.
        weight_decay: L2 regularization factor.
        device_name: 'auto', 'cuda', or 'cpu'.
        output_dir: Directory to save model checkpoints.

    Returns:
        Dictionary containing final metrics and saved checkpoint paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=wlasl_collate_fn,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=wlasl_collate_fn,
    )

    model = STGCN(
        in_channels=3,
        num_classes=num_classes,
        num_nodes=num_nodes,
        temporal_window_size=temporal_window_size,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = -1.0
    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)

        checkpoint_state = {
            "epoch": epoch,
            "num_classes": num_classes,
            "num_nodes": num_nodes,
            "temporal_window_size": temporal_window_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
        }

        torch.save(checkpoint_state, str(last_checkpoint_path))

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(checkpoint_state, str(best_checkpoint_path))

    return {
        "best_val_top1": best_val_acc,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history": history,
    }


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ST-GCN training."""
    parser = argparse.ArgumentParser(
        description="Train ST-GCN model on ASL landmark sequences."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing WLASL landmark files.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic landmark sequences for test/benchmarking.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of sign classes.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=75,
        help="Number of skeletal joint nodes.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Temporal sequence frame length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for model checkpoints.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """CLI entrypoint for ST-GCN training."""
    parsed = parse_args(args)

    if parsed.synthetic or parsed.data_dir is None:
        print("Generating synthetic WLASL landmark datasets for training...")
        train_ds = WLASLDataset.create_synthetic(
            num_samples=64,
            num_classes=parsed.num_classes,
            num_nodes=parsed.num_nodes,
            augment=True,
            seed=42,
        )
        val_ds = WLASLDataset.create_synthetic(
            num_samples=16,
            num_classes=parsed.num_classes,
            num_nodes=parsed.num_nodes,
            augment=False,
            seed=100,
        )
    else:
        data_path = Path(parsed.data_dir)
        train_ds = WLASLDataset(
            data=data_path / "train",
            target_length=parsed.window_size,
            num_nodes=parsed.num_nodes,
            augment=True,
        )
        val_ds = WLASLDataset(
            data=data_path / "val",
            target_length=parsed.window_size,
            num_nodes=parsed.num_nodes,
            augment=False,
        )

    results = run_training(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=parsed.num_classes,
        num_nodes=parsed.num_nodes,
        temporal_window_size=parsed.window_size,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        device_name=parsed.device,
        output_dir=parsed.output_dir,
    )

    print(
        f"Training complete. Best Val Top-1: {results['best_val_top1']:.2f}%. "
        f"Saved to: {results['best_checkpoint_path']}"
    )


if __name__ == "__main__":
    main()

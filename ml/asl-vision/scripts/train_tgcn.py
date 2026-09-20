"""Train GCN_muti_att (TGCN) on MediaPipe-harvested WLASL features.

Trains on the exact feature distribution the live pipeline produces, closing
the train/serve mismatch that made the third-party checkpoint unusable.

Data layout (see harvest_wlasl_features.py):
    <data_dir>/manifest.jsonl + <video_id>.npz {features: (T,55,2), active: (T,)}

Usage:
    uv run python scripts/train_tgcn.py /tmp/wlasl-train --epochs 60 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from asl_vision.models.tgcn_wlasl import WLASL_100_GLOSSES, GCN_muti_att

GLOSS_TO_IDX = {g: i for i, g in enumerate(WLASL_100_GLOSSES)}
NUM_SAMPLES = 50


class HarvestedTGCNDataset(Dataset):
    """WLASL clips as fixed 50-frame (55,100) TGCN inputs with integer labels."""

    def __init__(
        self,
        data_dir: Path,
        rows: list[dict],
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.data_dir = data_dir
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_frames(self, feats: np.ndarray, active: np.ndarray) -> np.ndarray:
        t = feats.shape[0]
        if t >= NUM_SAMPLES:
            if self.augment:
                start = int(self.rng.integers(0, t - NUM_SAMPLES + 1))
            else:
                start = (t - NUM_SAMPLES) // 2
            return feats[start:start + NUM_SAMPLES]
        idx = np.linspace(0, t - 1, NUM_SAMPLES).astype(np.int32)
        return feats[idx]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows[i]
        z = np.load(self.data_dir / f"{row['video_id']}.npz")
        feats = z["features"].astype(np.float32)
        if self.augment:
            feats = feats + self.rng.normal(0, 0.01, feats.shape).astype(np.float32)
        sampled = self._sample_frames(feats, z["active"])
        flat = sampled.transpose(1, 0, 2).reshape(55, NUM_SAMPLES * 2)
        return torch.from_numpy(flat), GLOSS_TO_IDX[row["gloss"]]


def topk_acc(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        _, pred = logits.topk(5, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        n = targets.size(0)
        t1 = correct[:, :1].reshape(-1).float().sum().item() * 100.0 / n
        t5 = correct.reshape(-1).float().sum().item() * 100.0 / n
        return t1, t5


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_t1 = tot_t5 = nb = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            t1, t5 = topk_acc(logits, y)
            tot_loss += loss.item()
            tot_t1 += t1
            tot_t5 += t5
            nb += 1
    return tot_loss / max(nb, 1), tot_t1 / max(nb, 1), tot_t5 / max(nb, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models/tgcn_retrained.bin"))
    args = ap.parse_args(argv)

    with open(args.data_dir / "manifest.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["gloss"] in GLOSS_TO_IDX]
    train_rows = [r for r in rows if r.get("split", "train") == "train"]
    val_rows = [r for r in rows if r.get("split", "train") != "train"]
    if not val_rows:  # fall back to 80/20 split by video
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_rows))
        cut = int(0.8 * len(train_rows))
        val_rows = [train_rows[i] for i in idx[cut:]]
        train_rows = [train_rows[i] for i in idx[:cut]]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    device = torch.device("cpu")
    train_loader = DataLoader(
        HarvestedTGCNDataset(args.data_dir, train_rows, augment=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        HarvestedTGCNDataset(args.data_dir, val_rows),
        batch_size=args.batch_size, num_workers=0,
    )

    model = GCN_muti_att(input_feature=100, hidden_feature=64, num_class=100,
                         p_dropout=0.3, num_stage=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-3)

    best_t1 = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_t1, tr_t5 = run_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_t1, va_t5 = run_epoch(model, val_loader, None, criterion, device)
        print(f"epoch {epoch:3d} train loss={tr_loss:.3f} t1={tr_t1:.1f} t5={tr_t5:.1f} "
              f"| val loss={va_loss:.3f} t1={va_t1:.1f} t5={va_t5:.1f}", flush=True)
        if va_t1 > best_t1:
            best_t1 = va_t1
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (val t1={best_t1:.1f})", flush=True)
    print(f"Best val Top-1: {best_t1:.1f}% (criterion: >=70%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

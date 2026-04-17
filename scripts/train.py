"""
Training script for DnCNN coronary angiography denoising.

Trains a DnCNN model (Zhang et al. 2017) on the CADICA dataset using either
single-frame or temporal (3-frame) input, with on-the-fly Poisson-Gaussian
noise synthesis.

Typical usage::

    # Single-frame, low25 dose, default hyperparameters
    python scripts/train.py --data-dir data/cadica --mode single --dose low25

    # Temporal, low10 dose, resume from checkpoint
    python scripts/train.py --mode temporal --dose low10 \\
        --resume checkpoints/last_temporal_low10.pt

Checkpoints are saved to ``--save-dir`` (default: ``checkpoints/``):

  - ``best_{mode}_{dose}.pt``  — lowest validation loss seen so far.
  - ``last_{mode}_{dose}.pt``  — end-of-epoch snapshot (safe resume point).

Each checkpoint stores::

    {
        "epoch":                int,
        "model_state_dict":     ...,
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ...,
        "best_val_loss":        float,
        "args":                 Namespace,
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as  python scripts/train.py  from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import SingleFrameDataset, TemporalDataset
from models.dncnn import build_dncnn
from utils.metrics import psnr, ssim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train DnCNN on the CADICA coronary angiography dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",           default="data/cadica",   help="Root CADICA directory")
    p.add_argument("--split-json",         default="data/split.json", help="Patient-level split file")
    p.add_argument("--mode",               choices=["single", "temporal"], default="single",
                   help="'single' → SingleFrameDataset; 'temporal' → TemporalDataset")
    p.add_argument("--dose",               choices=["low25", "low10", "low5"], default="low25",
                   help="Synthetic noise dose level")
    p.add_argument("--depth",              type=int,   default=17,   help="DnCNN depth (conv layers)")
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--batch-size",         type=int,   default=32)
    p.add_argument("--lr",                 type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--patch-size",         type=int,   default=128,  help="Random crop side length (px)")
    p.add_argument("--patches-per-frame",  type=int,   default=4,    help="Random crops per frame")
    p.add_argument("--num-workers",        type=int,   default=4,    help="DataLoader worker count")
    p.add_argument("--save-dir",           default="checkpoints",    help="Directory for saved checkpoints")
    p.add_argument("--resume",             default=None,             help="Path to .pt checkpoint to resume from")
    return p


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_split(split_json: Path, data_dir: Path) -> dict[str, list[Path]]:
    """Load split.json and reconstruct full Path objects.

    split.json is expected to map ``"train" / "val" / "test"`` to lists of
    patient folder names (e.g. ``["patient01", "patient02", …]``).

    Args:
        split_json: Path to the JSON split file.
        data_dir:   Root CADICA directory; patient names are joined onto it.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to lists of
        absolute ``Path`` objects.
    """
    with split_json.open() as f:
        raw: dict[str, list[str]] = json.load(f)
    return {split: [data_dir / name for name in names] for split, names in raw.items()}


def _make_dataset(
    patient_dirs: list[Path],
    mode: str,
    dose: str,
    patch_size: int,
    patches_per_frame: int,
) -> SingleFrameDataset | TemporalDataset:
    """Instantiate the correct dataset class for the given mode."""
    kwargs = dict(
        patient_dirs=patient_dirs,
        dose_level=dose,
        patch_size=patch_size,
        patches_per_frame=patches_per_frame,
    )
    if mode == "single":
        return SingleFrameDataset(**kwargs)
    else:
        return TemporalDataset(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "args":                 args,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Restore model/optimizer/scheduler from *path*.

    Returns:
        Tuple of ``(start_epoch, best_val_loss)`` so the training loop can
        resume from the correct epoch.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed from '{path}'  (epoch {ckpt['epoch']}  best_val_loss={best_val_loss:.6f})")
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [train]", leave=False)

    for noisy, clean in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(noisy)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return running_loss / len(loader)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch.

    Returns:
        Tuple of ``(mean_val_loss, mean_psnr_db, mean_ssim)``.
    """
    model.eval()
    total_loss  = 0.0
    total_psnr  = 0.0
    total_ssim  = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        pred = model(noisy)
        total_loss += criterion(pred, clean).item()
        total_psnr += psnr(pred.clamp(0, 1), clean).item()
        total_ssim += ssim(pred.clamp(0, 1), clean).item()

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    # ---- Paths ----
    data_dir   = Path(args.data_dir)
    split_json = Path(args.split_json)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best = save_dir / f"best_{args.mode}_{args.dose}.pt"
    ckpt_last = save_dir / f"last_{args.mode}_{args.dose}.pt"

    # ---- Data ----
    split = _load_split(split_json, data_dir)

    train_ds = _make_dataset(split["train"], args.mode, args.dose, args.patch_size, args.patches_per_frame)
    val_ds   = _make_dataset(split["val"],   args.mode, args.dose, args.patch_size, args.patches_per_frame)

    print(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = build_dncnn(mode=args.mode, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DnCNN ({args.mode}, depth={args.depth})  parameters: {n_params:,}")

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # ---- Resume ----
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        start_epoch, best_val_loss = _load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_psnr, val_ssim_score = _val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            _save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler, best_val_loss, args)

        _save_checkpoint(ckpt_last, epoch, model, optimizer, scheduler, best_val_loss, args)

        best_tag = "  [best]" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"PSNR={val_psnr:.1f}dB  "
            f"SSIM={val_ssim_score:.3f}"
            f"{best_tag}"
        )

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint : {ckpt_best}")
    print(f"Last checkpoint : {ckpt_last}")


if __name__ == "__main__":
    main()

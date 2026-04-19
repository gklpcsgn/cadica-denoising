"""
Visual comparison grid for trained DnCNN denoising models on CADICA test frames.

Saves two figures per dose level:
  results/figures/comparison_{dose}.png  — 4-column grid (noisy / single / temporal / clean)
  results/figures/residuals_{dose}.png   — absolute residual maps with shared colorbar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import SingleFrameDataset
from data.noise import add_poisson_gaussian_noise, DOSE_LEVELS
from models.dncnn import build_dncnn
from utils.metrics import psnr

DATA_DIR_DEFAULT = (
    "data/archive/CADICA a new dataset for coronary artery disease"
    "/CADICA/CADICA/selectedVideos"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize DnCNN denoising results on CADICA test frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dose",        choices=["low25", "low10", "low5"], default="low25")
    p.add_argument("--n-samples",   type=int,  default=6)
    p.add_argument("--depth",       type=int,  default=17)
    p.add_argument("--out-dir",     default="results/figures")
    p.add_argument("--data-dir",    default=DATA_DIR_DEFAULT)
    p.add_argument("--split-json",  default="data/split.json")
    p.add_argument("--patch-size",  type=int,  default=128)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_split(split_json: Path, data_dir: Path) -> list[Path]:
    with split_json.open() as f:
        raw: dict[str, list[str]] = json.load(f)
    return [data_dir / name for name in raw["test"]]


def _load_model(ckpt_path: Path, mode: str, depth: int, device: torch.device) -> torch.nn.Module:
    model = build_dncnn(mode=mode, depth=depth).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _to_np(t: torch.Tensor) -> np.ndarray:
    """(1, H, W) tensor → (H, W) numpy array."""
    return t.squeeze(0).cpu().numpy()


def _compute_psnr(pred: torch.Tensor, clean: torch.Tensor) -> float:
    return psnr(pred.clamp(0, 1), clean).item()


def _make_temporal_input(noisy_1chw: torch.Tensor, clean_1chw: torch.Tensor, dose: str) -> torch.Tensor:
    """Build a 3-channel temporal input from a single frame.

    The temporal model expects (prev, curr, next) each independently noisy.
    We re-noise the same clean patch twice more to simulate the neighbouring
    frames — acceptable for visualization purposes.
    """
    params = DOSE_LEVELS[dose]
    clean_np = _to_np(clean_1chw)

    prev_n = torch.from_numpy(add_poisson_gaussian_noise(clean_np, **params)).unsqueeze(0)
    next_n = torch.from_numpy(add_poisson_gaussian_noise(clean_np, **params)).unsqueeze(0)
    # channel order: prev | curr (use the same noisy patch) | next
    return torch.cat([prev_n, noisy_1chw, next_n], dim=0).unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    test_dirs = _load_split(Path(args.split_json), Path(args.data_dir))
    ds = SingleFrameDataset(
        patient_dirs=test_dirs,
        dose_level=args.dose,
        patch_size=args.patch_size,
        patches_per_frame=1,
    )

    n = min(args.n_samples, len(ds))
    pool_size = min(100, len(ds))
    candidate_indices = [int(round(i * (len(ds) - 1) / max(pool_size - 1, 1))) for i in range(pool_size)]
    candidates = []
    for idx in candidate_indices:
        _, clean = ds[idx]
        candidates.append((idx, clean.std().item()))
    candidates.sort(key=lambda x: x[1], reverse=True)
    indices = [c[0] for c in candidates[:n]]

    # ── Models ───────────────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints")
    single_model   = _load_model(ckpt_dir / f"best_single_{args.dose}.pt",   "single",   args.depth, device)
    temporal_model = _load_model(ckpt_dir / f"best_temporal_{args.dose}.pt", "temporal", args.depth, device)

    # ── Collect samples ───────────────────────────────────────────────────────
    samples: list[dict[str, np.ndarray | float]] = []
    with torch.no_grad():
        for idx in indices:
            noisy, clean = ds[idx]                           # both (1, H, W)
            noisy_dev = noisy.unsqueeze(0).to(device)        # (1, 1, H, W)
            clean_dev = clean.unsqueeze(0).to(device)        # (1, 1, H, W)

            pred_single = single_model(noisy_dev).clamp(0, 1)

            temp_in = _make_temporal_input(noisy, clean, args.dose).to(device)
            pred_temporal = temporal_model(temp_in).clamp(0, 1)

            samples.append({
                "noisy":    _to_np(noisy),
                "single":   _to_np(pred_single.squeeze(0)),
                "temporal": _to_np(pred_temporal.squeeze(0)),
                "clean":    _to_np(clean),
                "psnr_single":   _compute_psnr(pred_single,   clean_dev),
                "psnr_temporal": _compute_psnr(pred_temporal, clean_dev),
                "psnr_noisy":    _compute_psnr(noisy_dev,     clean_dev),
            })

    # ── Figure 1: comparison grid ─────────────────────────────────────────────
    col_titles = ["Noisy Input", "Single DnCNN", "Temporal DnCNN", "Clean (GT)"]
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    imshow_kw = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    for row, s in enumerate(samples):
        imgs = [s["noisy"], s["single"], s["temporal"], s["clean"]]
        psnrs = [s["psnr_noisy"], s["psnr_single"], s["psnr_temporal"], None]

        for col, (img, psnr_val) in enumerate(zip(imgs, psnrs)):
            ax = axes[row, col]
            ax.imshow(img, **imshow_kw)
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold", pad=4)

            if psnr_val is not None:
                ax.set_xlabel(f"PSNR {psnr_val:.2f} dB", fontsize=8, labelpad=2)

    fig.suptitle(f"DnCNN denoising comparison — dose: {args.dose}", fontsize=12, y=1.01)
    fig.tight_layout()

    comp_path = out_dir / f"comparison_{args.dose}.png"
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: residual maps ────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(n, 2, figsize=(4 * 2, 3 * n))
    if n == 1:
        axes2 = axes2[np.newaxis, :]

    res_kw = dict(cmap="RdBu_r", vmin=0, vmax=0.2, interpolation="nearest")
    res_titles = ["Residual — Single DnCNN", "Residual — Temporal DnCNN"]

    for row, s in enumerate(samples):
        residuals = [
            np.abs(s["single"]   - s["clean"]),
            np.abs(s["temporal"] - s["clean"]),
        ]
        for col, (res, title) in enumerate(zip(residuals, res_titles)):
            ax = axes2[row, col]
            im = ax.imshow(res, **res_kw)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    # Shared colorbar on the right
    fig2.subplots_adjust(right=0.88)
    cbar_ax = fig2.add_axes([0.91, 0.15, 0.02, 0.7])
    fig2.colorbar(im, cax=cbar_ax, label="|pred − clean|")

    fig2.suptitle(f"Residual maps — dose: {args.dose}", fontsize=12, y=1.01)
    fig2.tight_layout(rect=[0, 0, 0.9, 1])

    res_path = out_dir / f"residuals_{args.dose}.png"
    fig2.savefig(res_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {comp_path}")
    print(f"Saved: {res_path}")


if __name__ == "__main__":
    main()

"""
Visualization utilities for denoising evaluation results.

All functions write PNG files and do not display any interactive window
(``Agg`` backend is set at import time).

Functions
---------

save_comparison_grid
    Side-by-side image grid: rows = samples, columns = conditions.
    Annotates each cell with PSNR relative to the clean reference.

save_residual_maps
    Absolute difference maps ``|condition − clean|`` per sample.
    Shared colour scale makes per-condition errors directly comparable.

save_psnr_ssim_bars
    Grouped bar chart of PSNR and SSIM across dose levels and conditions,
    with standard-deviation error bars.  Loads ``eval_summary.csv``.

save_inference_time_plot
    Bar chart of inference time (ms/frame) per condition.
    Loads ``eval_summary.csv`` and reads the ``low25`` rows.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_CONDITIONS    = ["noisy", "nlm", "single", "temporal"]
_COL_HEADERS   = {
    "noisy":    "Noisy Input",
    "nlm":      "NLM",
    "single":   "Single DnCNN",
    "temporal": "Temporal DnCNN",
    "clean":    "Clean (GT)",
}
_CONDITION_COLORS = {
    "noisy":    "#9ecae1",
    "nlm":      "#fdae6b",
    "single":   "#74c476",
    "temporal": "#9e9ac8",
}
_DOSE_ORDER = ["low25", "low10", "low5"]
_DOSE_LABELS = {"low25": "Low 25%", "low10": "Low 10%", "low5": "Low 5%"}


def _psnr_np(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Per-image PSNR (dB) between two float32 arrays."""
    mse = np.mean((pred.astype("float32") - target.astype("float32")) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ---------------------------------------------------------------------------
# 1. Comparison grid
# ---------------------------------------------------------------------------

def save_comparison_grid(
    samples: list[dict],
    out_path: str | Path,
    title: str = "",
) -> None:
    """Save a side-by-side comparison grid of denoising conditions.

    Each row corresponds to one sample; columns show the five conditions
    (noisy, nlm, single, temporal, clean).  A PSNR annotation is placed
    below every image except the clean ground-truth column.

    Args:
        samples:  List of dicts, one per sample.  Each dict must contain
                  keys ``"noisy"``, ``"nlm"``, ``"single"``, ``"temporal"``,
                  and ``"clean"`` — all float32 NumPy arrays of shape ``(H, W)``.
        out_path: Destination PNG path.
        title:    Optional figure-level super-title.
    """
    cols     = _CONDITIONS + ["clean"]
    n_rows   = len(samples)
    n_cols   = len(cols)
    cell_in  = 2.2                              # inches per cell
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_in * n_cols, cell_in * n_rows + (0.6 if title else 0.2)),
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.0)

    for row_idx, sample in enumerate(samples):
        clean = sample["clean"]
        for col_idx, key in enumerate(cols):
            ax  = axes[row_idx][col_idx]
            img = sample[key]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Column header on first row only
            if row_idx == 0:
                ax.set_title(_COL_HEADERS[key], fontsize=8, fontweight="bold", pad=3)

            # PSNR subtitle (skip clean column)
            if key != "clean":
                p = _psnr_np(img, clean)
                p_str = f"{p:.1f} dB" if np.isfinite(p) else "∞ dB"
                ax.set_xlabel(p_str, fontsize=7, labelpad=2)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Residual maps
# ---------------------------------------------------------------------------

def save_residual_maps(
    samples: list[dict],
    out_path: str | Path,
) -> None:
    """Save absolute difference maps ``|condition − clean|`` for each sample.

    A shared colour scale (``vmin=0``, ``vmax=0.2``) makes per-condition
    residuals directly comparable.  A single shared colorbar is appended to
    the right of each row.

    Args:
        samples:  List of dicts with the same schema as
                  :func:`save_comparison_grid`.
        out_path: Destination PNG path.
    """
    n_rows  = len(samples)
    n_cols  = len(_CONDITIONS)
    cell_in = 2.2
    vmax    = 0.2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_in * n_cols + 0.5, cell_in * n_rows),
        squeeze=False,
    )
    fig.suptitle("Absolute residual maps  |prediction − clean|",
                 fontsize=10, fontweight="bold")

    im_ref = None
    for row_idx, sample in enumerate(samples):
        clean = sample["clean"]
        for col_idx, key in enumerate(_CONDITIONS):
            ax       = axes[row_idx][col_idx]
            residual = np.abs(sample[key].astype("float32") - clean.astype("float32"))
            im = ax.imshow(
                residual,
                cmap="RdBu_r",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(_COL_HEADERS[key], fontsize=8, fontweight="bold", pad=3)
            im_ref = im                         # all share the same scale

    # One colorbar on the right
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Absolute error", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_summary_csv(
    csv_path: str | Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load ``eval_summary.csv`` into a nested dict.

    Returns:
        ``results[dose][condition]`` → dict with keys
        ``psnr_mean``, ``psnr_std``, ``ssim_mean``, ``ssim_std``, ``time_ms``.
    """
    results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            dose      = row["dose"]
            condition = row["condition"]
            results[dose][condition] = {
                "psnr_mean": float(row["psnr_mean"]),
                "psnr_std":  float(row["psnr_std"]),
                "ssim_mean": float(row["ssim_mean"]),
                "ssim_std":  float(row["ssim_std"]),
                "time_ms":   float(row["time_ms"]) if row["time_ms"] else 0.0,
            }
    return results


# ---------------------------------------------------------------------------
# 3. PSNR / SSIM grouped bar chart
# ---------------------------------------------------------------------------

def save_psnr_ssim_bars(
    results_csv: str | Path,
    out_path: str | Path,
) -> None:
    """Save a grouped bar chart of PSNR and SSIM across dose levels.

    Creates a two-row figure: the top row shows PSNR (dB) and the bottom row
    shows SSIM, both grouped by dose level with one bar per condition and
    standard-deviation error bars.

    Args:
        results_csv: Path to ``eval_summary.csv`` produced by ``evaluate.py``.
        out_path:    Destination PNG path.
    """
    data = _load_summary_csv(results_csv)

    # Only include dose levels actually present in the CSV, preserving order.
    doses      = [d for d in _DOSE_ORDER if d in data]
    conditions = [c for c in _CONDITIONS if any(c in data[d] for d in doses)]

    n_groups = len(doses)
    n_bars   = len(conditions)
    bar_w    = 0.8 / n_bars
    x        = np.arange(n_groups)

    fig, (ax_psnr, ax_ssim) = plt.subplots(2, 1, figsize=(max(6, n_groups * 2.5), 7))
    fig.suptitle("DnCNN denoising performance by dose level",
                 fontsize=11, fontweight="bold")

    for metric, ax, ylabel in [
        ("psnr", ax_psnr, "PSNR (dB)"),
        ("ssim", ax_ssim, "SSIM"),
    ]:
        for bar_idx, cond in enumerate(conditions):
            means = []
            stds  = []
            for dose in doses:
                entry = data[dose].get(cond, {})
                means.append(entry.get(f"{metric}_mean", 0.0))
                stds.append(entry.get(f"{metric}_std",  0.0))

            offsets = (bar_idx - (n_bars - 1) / 2.0) * bar_w
            ax.bar(
                x + offsets,
                means,
                width=bar_w * 0.9,
                label=_COL_HEADERS.get(cond, cond),
                color=_CONDITION_COLORS.get(cond, "#aaaaaa"),
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_DOSE_LABELS.get(d, d) for d in doses], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Inference time bar chart
# ---------------------------------------------------------------------------

def save_inference_time_plot(
    results_csv: str | Path,
    out_path: str | Path,
) -> None:
    """Save a bar chart of inference time (ms/frame) per condition.

    Uses the ``low25`` rows from ``eval_summary.csv`` (inference time does not
    meaningfully vary by dose level).  Each bar is annotated with its value.

    Args:
        results_csv: Path to ``eval_summary.csv`` produced by ``evaluate.py``.
        out_path:    Destination PNG path.
    """
    data = _load_summary_csv(results_csv)

    # Fall back to first available dose if low25 is absent.
    dose = "low25" if "low25" in data else next(iter(data))
    dose_data = data[dose]

    conditions = [c for c in _CONDITIONS if c in dose_data]
    times      = [dose_data[c]["time_ms"] for c in conditions]
    labels     = [_COL_HEADERS.get(c, c) for c in conditions]
    colors     = [_CONDITION_COLORS.get(c, "#aaaaaa") for c in conditions]

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 1.4), 4))
    bars = ax.bar(labels, times, color=colors, width=0.55, zorder=2)

    # Annotate each bar with its numeric value
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(times) * 0.02,
            f"{t:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Inference time (ms / frame)", fontsize=9)
    ax.set_title("Inference time per condition  (low25 dose)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.2)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=1)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quick self-test (run with:  python -m utils.visualization)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)

    def _fake_sample(rng: np.random.Generator) -> dict:
        clean = rng.random((128, 128), dtype=np.float32)
        return {
            "clean":    clean,
            "noisy":    np.clip(clean + 0.15 * rng.standard_normal((128, 128)).astype("float32"), 0, 1),
            "nlm":      np.clip(clean + 0.08 * rng.standard_normal((128, 128)).astype("float32"), 0, 1),
            "single":   np.clip(clean + 0.04 * rng.standard_normal((128, 128)).astype("float32"), 0, 1),
            "temporal": np.clip(clean + 0.03 * rng.standard_normal((128, 128)).astype("float32"), 0, 1),
        }

    samples = [_fake_sample(rng) for _ in range(3)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        grid_path = tmp / "comparison_grid.png"
        save_comparison_grid(samples, grid_path, title="Test comparison grid")
        assert grid_path.is_file()
        print(f"save_comparison_grid  → {grid_path.name}  [OK]")

        res_path = tmp / "residual_maps.png"
        save_residual_maps(samples, res_path)
        assert res_path.is_file()
        print(f"save_residual_maps    → {res_path.name}   [OK]")

        # Write a fake eval_summary.csv
        csv_path = tmp / "eval_summary.csv"
        rows = [
            ("low25", "noisy",    24.3, 1.1, 0.712, 0.020, 0.0),
            ("low25", "nlm",      28.1, 0.9, 0.801, 0.015, 12.4),
            ("low25", "single",   33.2, 0.8, 0.901, 0.010,  2.1),
            ("low25", "temporal", 34.5, 0.7, 0.918, 0.009,  3.8),
            ("low10", "noisy",    21.1, 1.2, 0.680, 0.022, 0.0),
            ("low10", "nlm",      25.6, 1.0, 0.772, 0.018, 12.4),
            ("low10", "single",   30.1, 0.9, 0.875, 0.012,  2.1),
            ("low10", "temporal", 31.8, 0.8, 0.891, 0.011,  3.8),
            ("low5",  "noisy",    18.4, 1.3, 0.640, 0.025, 0.0),
            ("low5",  "nlm",      22.9, 1.1, 0.740, 0.021, 12.4),
            ("low5",  "single",   27.3, 1.0, 0.845, 0.015,  2.1),
            ("low5",  "temporal", 29.0, 0.9, 0.862, 0.013,  3.8),
        ]
        with csv_path.open("w", newline="") as f:
            w = __import__("csv").writer(f)
            w.writerow(["dose", "condition", "psnr_mean", "psnr_std",
                        "ssim_mean", "ssim_std", "time_ms"])
            w.writerows(rows)

        bars_path = tmp / "psnr_ssim_bars.png"
        save_psnr_ssim_bars(csv_path, bars_path)
        assert bars_path.is_file()
        print(f"save_psnr_ssim_bars   → {bars_path.name}  [OK]")

        time_path = tmp / "inference_time.png"
        save_inference_time_plot(csv_path, time_path)
        assert time_path.is_file()
        print(f"save_inference_time_plot → {time_path.name} [OK]")

    print("All checks passed.")

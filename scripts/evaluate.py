"""
Evaluation script for DnCNN coronary angiography denoising.

Benchmarks four denoising conditions against the CADICA test split and
reports PSNR, SSIM, and inference time per frame:

  1. noisy    — raw noisy input (no processing); establishes a lower bound.
  2. nlm      — Non-Local Means (cv2.fastNlMeansDenoising); classical baseline.
  3. single   — DnCNN trained on single-frame input.
  4. temporal — DnCNN trained on 3-frame temporal input.

Typical usage::

    # Evaluate all conditions for low25 dose
    python scripts/evaluate.py --dose low25

    # Sweep all three dose levels and write a summary CSV
    python scripts/evaluate.py --all-doses

Results are written to ``--results-dir`` (default: ``results/``):

  - ``eval_{dose}.json``   — full detail: per-sample arrays + summary stats.
  - ``eval_summary.csv``   — (only with ``--all-doses``) one row per condition
                             per dose, columns: dose, condition, psnr_mean,
                             psnr_std, ssim_mean, ssim_std, time_ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import SingleFrameDataset, TemporalDataset
from models.dncnn import build_dncnn
from utils.metrics import psnr_batch, ssim_batch, time_inference


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate DnCNN denoising on the CADICA test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",           default="data/cadica",    help="Root CADICA directory")
    p.add_argument("--split-json",         default="data/split.json", help="Patient-level split file")
    p.add_argument("--dose",               choices=["low25", "low10", "low5"], default="low25")
    p.add_argument("--depth",              type=int, default=17,      help="DnCNN depth (conv layers)")
    p.add_argument("--batch-size",         type=int, default=16)
    p.add_argument("--patch-size",         type=int, default=128)
    p.add_argument("--patches-per-frame",  type=int, default=1)
    p.add_argument("--num-workers",        type=int, default=4)
    p.add_argument("--checkpoint-dir",     default="checkpoints")
    p.add_argument("--results-dir",        default="results")
    p.add_argument("--all-doses",          action="store_true",
                   help="Run all three dose levels and write eval_summary.csv")
    return p


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_split(split_json: Path, data_dir: Path) -> dict[str, list[Path]]:
    """Load split.json and reconstruct full Path objects for each patient.

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


def _make_loader(
    patient_dirs: list[Path],
    mode: str,
    dose: str,
    patch_size: int,
    patches_per_frame: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a non-shuffled test DataLoader for *mode*."""
    kwargs = dict(
        patient_dirs=patient_dirs,
        dose_level=dose,
        patch_size=patch_size,
        patches_per_frame=patches_per_frame,
    )
    dataset = SingleFrameDataset(**kwargs) if mode == "single" else TemporalDataset(**kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# NLM baseline
# ---------------------------------------------------------------------------

def _nlm_denoise_batch(noisy: torch.Tensor) -> torch.Tensor:
    """Apply cv2.fastNlMeansDenoising to every frame in a batch.

    Converts each float32 frame in ``[0, 1]`` to uint8, denoises, then
    converts back to float32.

    Args:
        noisy: Tensor ``(B, 1, H, W)`` with values in ``[0, 1]``.

    Returns:
        Denoised tensor ``(B, 1, H, W)`` in ``[0, 1]``.
    """
    import cv2
    import numpy as np

    results = []
    np_batch = (noisy.squeeze(1).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    for frame in np_batch:
        denoised = cv2.fastNlMeansDenoising(
            frame,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        results.append(denoised.astype("float32") / 255.0)
    out = torch.from_numpy(
        __import__("numpy").stack(results, axis=0)
    ).unsqueeze(1)                          # (B, 1, H, W)
    return out


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_predictions(
    loader: DataLoader,
    predict_fn,                             # callable(noisy_tensor) -> pred_tensor (CPU)
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run *predict_fn* over the full loader and collect all predictions.

    Args:
        loader:     DataLoader yielding ``(noisy, clean)`` batches.
        predict_fn: Callable that takes a noisy tensor on *device* and returns
                    a clean prediction (may be on CPU or GPU — will be moved to CPU).
        device:     Torch device for model inference.

    Returns:
        Tuple ``(all_preds, all_clean)`` — concatenated CPU tensors
        of shape ``(N, 1, H, W)``.
    """
    preds_list: list[torch.Tensor] = []
    clean_list: list[torch.Tensor] = []

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        pred  = predict_fn(noisy).cpu()
        preds_list.append(pred)
        clean_list.append(clean)

    return torch.cat(preds_list, dim=0), torch.cat(clean_list, dim=0)


def _summarise(values: torch.Tensor) -> dict[str, Any]:
    """Return mean, std, min, max as plain Python floats."""
    return {
        "mean": values.mean().item(),
        "std":  values.std().item(),
        "min":  values.min().item(),
        "max":  values.max().item(),
    }


def _evaluate_condition(
    condition: str,
    loader: DataLoader,
    predict_fn,
    device: torch.device,
    timing_input: torch.Tensor,
    timing_model,
) -> dict[str, Any]:
    """Evaluate a single denoising condition.

    Args:
        condition:    Human-readable label (e.g. ``"single"``).
        loader:       Test DataLoader.
        predict_fn:   Callable ``(noisy: Tensor) -> pred: Tensor``.
        device:       Inference device.
        timing_input: A single representative input tensor for timing.
        timing_model: Object with a ``__call__`` method — passed to
                      ``time_inference``.

    Returns:
        Dict containing per-sample metric arrays and summary statistics.
    """
    preds, cleans = _collect_predictions(loader, predict_fn, device)
    preds  = preds.clamp(0.0, 1.0)
    cleans = cleans.clamp(0.0, 1.0)

    psnr_vals = psnr_batch(preds, cleans)   # (N,)
    ssim_vals = ssim_batch(preds, cleans)   # (N,)

    timing = time_inference(timing_model, timing_input, n_warmup=5, n_runs=50)

    return {
        "condition":    condition,
        "psnr":         _summarise(psnr_vals),
        "ssim":         _summarise(ssim_vals),
        "time_ms":      timing["mean_ms"],
        "time_std_ms":  timing["std_ms"],
        "psnr_per_sample": psnr_vals.tolist(),
        "ssim_per_sample": ssim_vals.tolist(),
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_model(
    ckpt_path: Path,
    mode: str,
    depth: int,
    device: torch.device,
) -> nn.Module | None:
    """Load a DnCNN from *ckpt_path*, returning ``None`` if the file is absent.

    Args:
        ckpt_path: Path to the ``.pt`` checkpoint.
        mode:      ``"single"`` or ``"temporal"``.
        depth:     Number of conv layers.
        device:    Target device.

    Returns:
        Loaded model in eval mode, or ``None`` with a warning if the
        checkpoint does not exist.
    """
    if not ckpt_path.is_file():
        warnings.warn(
            f"Checkpoint not found, skipping condition '{mode}': {ckpt_path}",
            stacklevel=2,
        )
        return None
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_dncnn(mode=mode, depth=depth).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_CONDITION_LABELS = {
    "noisy":    "Noisy",
    "nlm":      "NLM",
    "single":   "Single DnCNN",
    "temporal": "Temporal DnCNN",
}

def _print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    col_w = [20, 12, 8, 16]
    header = (
        f"{'Condition':<{col_w[0]}}"
        f"{'PSNR (dB)':>{col_w[1]}}"
        f"{'SSIM':>{col_w[2]}}"
        f"{'Time (ms/frame)':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    print(header)
    print(sep)
    for r in results:
        label    = _CONDITION_LABELS.get(r["condition"], r["condition"])
        psnr_str = f"{r['psnr']['mean']:.1f}"
        ssim_str = f"{r['ssim']['mean']:.3f}"
        time_str = f"{r['time_ms']:.1f}" if r["time_ms"] is not None else "-"
        print(
            f"{label:<{col_w[0]}}"
            f"{psnr_str:>{col_w[1]}}"
            f"{ssim_str:>{col_w[2]}}"
            f"{time_str:>{col_w[3]}}"
        )


# ---------------------------------------------------------------------------
# Single-dose evaluation
# ---------------------------------------------------------------------------

def evaluate_dose(
    dose: str,
    args: argparse.Namespace,
    device: torch.device,
    results_dir: Path,
) -> list[dict[str, Any]]:
    """Run all four conditions for a single dose level.

    Args:
        dose:        Dose level key (``"low25"``, ``"low10"``, or ``"low5"``).
        args:        Parsed CLI namespace.
        device:      Torch device.
        results_dir: Directory where ``eval_{dose}.json`` will be written.

    Returns:
        List of per-condition result dicts (for CSV aggregation).
    """
    print(f"\n{'='*60}")
    print(f"Dose: {dose}")
    print(f"{'='*60}")

    data_dir   = Path(args.data_dir)
    split_json = Path(args.split_json)
    ckpt_dir   = Path(args.checkpoint_dir)

    split       = _load_split(split_json, data_dir)
    test_dirs   = split["test"]

    # Build a shared single-frame loader (used for noisy and NLM baselines
    # and the single-model condition).
    single_loader = _make_loader(
        test_dirs, "single", dose,
        args.patch_size, args.patches_per_frame,
        args.batch_size, args.num_workers,
    )

    # Representative timing input: one batch on the evaluation device.
    timing_noisy, _ = next(iter(single_loader))
    timing_noisy = timing_noisy.to(device)

    all_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Noisy baseline
    # ------------------------------------------------------------------
    print("  Evaluating: noisy …")

    class _Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1]             # single channel regardless of input

    noisy_result = _evaluate_condition(
        condition="noisy",
        loader=single_loader,
        predict_fn=lambda x: x[:, :1].cpu(),
        device=device,
        timing_input=timing_noisy,
        timing_model=_Identity().to(device),
    )
    all_results.append(noisy_result)

    # ------------------------------------------------------------------
    # 2. NLM baseline
    # ------------------------------------------------------------------
    print("  Evaluating: nlm …")

    class _NLMWrapper(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _nlm_denoise_batch(x.cpu()).to(x.device)

    nlm_result = _evaluate_condition(
        condition="nlm",
        loader=single_loader,
        predict_fn=lambda x: _nlm_denoise_batch(x.cpu()),
        device=device,
        timing_input=timing_noisy.cpu(),    # NLM runs on CPU
        timing_model=_NLMWrapper(),
    )
    all_results.append(nlm_result)

    # ------------------------------------------------------------------
    # 3. Single DnCNN
    # ------------------------------------------------------------------
    print("  Evaluating: single …")

    single_ckpt = ckpt_dir / f"best_single_{dose}.pt"
    single_model = _load_model(single_ckpt, "single", args.depth, device)

    if single_model is not None:
        single_result = _evaluate_condition(
            condition="single",
            loader=single_loader,
            predict_fn=lambda x: single_model(x).cpu(),
            device=device,
            timing_input=timing_noisy,
            timing_model=single_model,
        )
        all_results.append(single_result)
    else:
        print("    [skipped — checkpoint missing]")

    # ------------------------------------------------------------------
    # 4. Temporal DnCNN
    # ------------------------------------------------------------------
    print("  Evaluating: temporal …")

    temporal_ckpt = ckpt_dir / f"best_temporal_{dose}.pt"
    temporal_model = _load_model(temporal_ckpt, "temporal", args.depth, device)

    if temporal_model is not None:
        temporal_loader = _make_loader(
            test_dirs, "temporal", dose,
            args.patch_size, args.patches_per_frame,
            args.batch_size, args.num_workers,
        )
        timing_temporal, _ = next(iter(temporal_loader))
        timing_temporal = timing_temporal.to(device)

        temporal_result = _evaluate_condition(
            condition="temporal",
            loader=temporal_loader,
            predict_fn=lambda x: temporal_model(x).cpu(),
            device=device,
            timing_input=timing_temporal,
            timing_model=temporal_model,
        )
        all_results.append(temporal_result)
    else:
        print("    [skipped — checkpoint missing]")

    # ------------------------------------------------------------------
    # Print table & save JSON
    # ------------------------------------------------------------------
    print()
    _print_table(all_results)

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"eval_{dose}.json"
    with json_path.open("w") as f:
        json.dump({"dose": dose, "conditions": all_results}, f, indent=2)
    print(f"\nSaved: {json_path}")

    return all_results


# ---------------------------------------------------------------------------
# CSV summary (--all-doses)
# ---------------------------------------------------------------------------

def _write_summary_csv(
    all_dose_results: dict[str, list[dict[str, Any]]],
    results_dir: Path,
) -> None:
    """Write eval_summary.csv aggregating results across all dose levels.

    Args:
        all_dose_results: Mapping of dose level → list of condition result dicts.
        results_dir:      Directory where the CSV will be written.
    """
    csv_path = results_dir / "eval_summary.csv"
    fieldnames = [
        "dose", "condition",
        "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std",
        "time_ms",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dose, results in all_dose_results.items():
            for r in results:
                writer.writerow({
                    "dose":       dose,
                    "condition":  r["condition"],
                    "psnr_mean":  f"{r['psnr']['mean']:.4f}",
                    "psnr_std":   f"{r['psnr']['std']:.4f}",
                    "ssim_mean":  f"{r['ssim']['mean']:.4f}",
                    "ssim_std":   f"{r['ssim']['std']:.4f}",
                    "time_ms":    f"{r['time_ms']:.3f}" if r["time_ms"] is not None else "",
                })
    print(f"\nSummary CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    results_dir = Path(args.results_dir)
    doses = ["low25", "low10", "low5"] if args.all_doses else [args.dose]

    all_dose_results: dict[str, list[dict[str, Any]]] = {}
    for dose in doses:
        all_dose_results[dose] = evaluate_dose(dose, args, device, results_dir)

    if args.all_doses:
        _write_summary_csv(all_dose_results, results_dir)


if __name__ == "__main__":
    main()

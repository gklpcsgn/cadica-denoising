"""
Image quality metrics for denoising evaluation.

All functions accept torch tensors and operate in batch.

PSNR  – Peak Signal-to-Noise Ratio (higher is better, dB)
SSIM  – Structural Similarity Index Measure (higher is better, [0, 1])
        Computed via a Gaussian-weighted sliding window following
        Wang et al. 2004 (same defaults as skimage.metrics.structural_similarity).

Also provides ``time_inference`` to benchmark model throughput (ms / frame).
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute PSNR between predicted and target tensors.

    Args:
        pred: Denoised/predicted image tensor, any shape (…, H, W).
        target: Ground-truth clean image tensor, same shape as pred.
        data_range: Maximum possible pixel value (1.0 for float images in [0,1]).

    Returns:
        Scalar tensor with the mean PSNR across the batch (dB).
        Returns ``inf`` if pred == target (zero MSE).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mse = F.mse_loss(pred.float(), target.float(), reduction="mean")
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10.0 * torch.log10(torch.tensor(data_range**2) / mse)


def psnr_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per-sample PSNR for a batch (B, C, H, W).

    Returns:
        1-D tensor of shape (B,) with per-sample PSNR values (dB).
    """
    b = pred.shape[0]
    pred_f = pred.float().view(b, -1)
    target_f = target.float().view(b, -1)
    mse_per_sample = ((pred_f - target_f) ** 2).mean(dim=1)
    # Avoid log(0) for identical images
    mse_per_sample = mse_per_sample.clamp(min=1e-10)
    return 10.0 * torch.log10(torch.tensor(data_range**2) / mse_per_sample)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian kernel, normalised to sum to 1."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _gaussian_window_2d(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """2-D Gaussian window as a depthwise conv kernel (channels, 1, W, W)."""
    k1d = _gaussian_kernel(window_size, sigma, device)
    k2d = k1d.outer(k1d)                           # (W, W)
    k2d = k2d.expand(channels, 1, window_size, window_size)
    return k2d


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Structural Similarity Index Measure (mean over spatial locations & batch).

    Follows Wang et al. 2004 with the same constants used in scikit-image.

    Args:
        pred: Predicted tensor (B, C, H, W) or (C, H, W).
        target: Ground-truth tensor, same shape as pred.
        data_range: Pixel value range (1.0 for [0, 1] images).
        window_size: Gaussian window size (11 in the reference paper).
        sigma: Gaussian standard deviation (1.5 in the reference paper).
        k1, k2: Stability constants (0.01, 0.03 in the reference paper).

    Returns:
        Scalar tensor: mean SSIM across the batch.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    pred = pred.float()
    target = target.float()

    B, C, H, W = pred.shape
    pad = window_size // 2

    kernel = _gaussian_window_2d(window_size, sigma, C, pred.device)

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_x = _conv(pred)
    mu_y = _conv(target)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x_sq = _conv(pred * pred)   - mu_x_sq
    sigma_y_sq = _conv(target * target) - mu_y_sq
    sigma_xy   = _conv(pred * target)  - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator   = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator.clamp(min=1e-8)
    return ssim_map.mean()


def ssim_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Per-sample SSIM for a batch (B, C, H, W).

    Returns:
        1-D tensor of shape (B,) with per-sample SSIM values.
    """
    scores = [ssim(pred[i], target[i], **kwargs) for i in range(pred.shape[0])]
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

def time_inference(
    model: Callable,
    input_tensor: torch.Tensor,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict[str, float]:
    """Benchmark model inference time per frame.

    Runs the model ``n_warmup`` times (discarded) then ``n_runs`` times to
    collect timing statistics. Uses CUDA events when available for accurate
    GPU timing; falls back to ``time.perf_counter`` on CPU.

    Args:
        model: Callable (e.g. a nn.Module) that takes ``input_tensor``.
        input_tensor: A single input tensor (will be used as-is for every run).
        n_warmup: Number of warm-up iterations before timing starts.
        n_runs: Number of timed iterations.

    Returns:
        Dict with keys:
          - "mean_ms": mean time per frame in milliseconds
          - "std_ms": standard deviation
          - "min_ms": minimum observed time
          - "max_ms": maximum observed time
    """
    device = input_tensor.device
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            _ = model(input_tensor)
        if use_cuda:
            torch.cuda.synchronize()

        # Timed runs
        times: list[float] = []
        for _ in range(n_runs):
            if use_cuda:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                _ = model(input_tensor)
                end_ev.record()
                torch.cuda.synchronize()
                times.append(start_ev.elapsed_time(end_ev))   # ms
            else:
                t0 = time.perf_counter()
                _ = model(input_tensor)
                times.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "std_ms":  t.std().item(),
        "min_ms":  t.min().item(),
        "max_ms":  t.max().item(),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.rand(4, 1, 128, 128)
    noisy = clean + 0.05 * torch.randn_like(clean)
    noisy = noisy.clamp(0, 1)

    p = psnr(noisy, clean)
    s = ssim(noisy, clean)
    print(f"PSNR: {p:.2f} dB   SSIM: {s:.4f}")

    pb = psnr_batch(noisy, clean)
    sb = ssim_batch(noisy, clean)
    print(f"Per-sample PSNR: {pb.tolist()}")
    print(f"Per-sample SSIM: {[f'{v:.4f}' for v in sb.tolist()]}")

    # Timing with a trivial identity model
    class Identity(torch.nn.Module):
        def forward(self, x): return x

    timing = time_inference(Identity(), clean, n_warmup=2, n_runs=10)
    print(f"Inference timing: {timing['mean_ms']:.3f} ± {timing['std_ms']:.3f} ms/frame")

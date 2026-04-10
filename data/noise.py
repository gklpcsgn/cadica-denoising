"""
Physically-motivated low-dose X-ray noise simulation.

Model
-----
X-ray detectors produce two dominant noise sources:

1. **Poisson (quantum / photon shot noise)**
   The number of photons hitting each pixel follows a Poisson distribution
   whose mean equals the expected photon count.  Reducing the radiation dose
   means fewer photons → variance scales with the signal → lower SNR.

2. **Gaussian (electronic / thermal noise)**
   Read-out electronics add approximately Gaussian noise with a fixed
   standard deviation that is *independent* of the signal level.

Combined model for a pixel with normalised intensity x ∈ [0, 1]:

    peak_count  = 1000 * dose_fraction          # expected photons at full white
    n_photons   ~ Poisson(λ = x * peak_count)   # signal-dependent shot noise
    y           = n_photons / peak_count         # back to [0, 1]
    y_out       = clip(y + N(0, σ²), 0, 1)      # add electronic noise

Usage
-----
    from data.noise import add_poisson_gaussian_noise, DOSE_LEVELS

    rng = np.random.default_rng(0)
    noisy = add_poisson_gaussian_noise(clean, dose_fraction=0.25, sigma=0.02, rng=rng)

    # or via named preset
    params = DOSE_LEVELS["low25"]
    noisy  = add_poisson_gaussian_noise(clean, **params, rng=rng)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Preset dose levels
# ---------------------------------------------------------------------------

DOSE_LEVELS: dict[str, dict[str, float]] = {
    "low25": {"dose_fraction": 0.25, "sigma": 0.02},   # 25 % of full dose
    "low10": {"dose_fraction": 0.10, "sigma": 0.03},   # 10 % of full dose
    "low5":  {"dose_fraction": 0.05, "sigma": 0.04},   #  5 % of full dose
}

_PEAK_COUNT_FULL: int = 1000  # photons per pixel at normalised value 1.0, full dose


# ---------------------------------------------------------------------------
# Core noise function
# ---------------------------------------------------------------------------

def add_poisson_gaussian_noise(
    image: np.ndarray,
    dose_fraction: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply Poisson-Gaussian mixture noise to a normalised X-ray image.

    Args:
        image:         Float32 numpy array with pixel values in [0, 1].
                       Any shape is accepted (H, W), (1, H, W), (C, H, W), …
        dose_fraction: Fraction of full dose to simulate (0 < dose_fraction ≤ 1).
                       Lower values → fewer photons → stronger Poisson noise.
        sigma:         Standard deviation of additive Gaussian (electronic) noise
                       in the [0, 1] normalised scale.
        rng:           Optional ``numpy.random.Generator`` for reproducibility.
                       If *None* a new default generator is created each call.

    Returns:
        Noisy float32 array with the same shape as *image*, clipped to [0, 1].
    """
    if not (0 < dose_fraction <= 1.0):
        raise ValueError(f"dose_fraction must be in (0, 1], got {dose_fraction!r}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")

    if rng is None:
        rng = np.random.default_rng()

    image = np.asarray(image, dtype=np.float32)

    # --- Poisson noise (signal-dependent) ---
    peak_count = _PEAK_COUNT_FULL * dose_fraction           # expected photons at pixel value 1
    photon_counts = rng.poisson(
        np.maximum(image * peak_count, 0.0)                 # λ must be ≥ 0
    ).astype(np.float32)
    noisy = photon_counts / peak_count                       # back to [0, 1] scale

    # --- Gaussian noise (signal-independent) ---
    if sigma > 0:
        noisy = noisy + rng.standard_normal(image.shape).astype(np.float32) * sigma

    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# __main__: generate a side-by-side preview image
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try to load a real grayscale image; fall back to a synthetic phantom.
    _test_img_path = Path("data/cadica").glob("patient*/video*/*.png")
    _first = next(_test_img_path, None)

    if _first is not None:
        from PIL import Image
        clean = np.asarray(Image.open(_first).convert("L"), dtype=np.float32) / 255.0
        print(f"Using real frame: {_first}")
    else:
        # Synthetic disk-and-vessels phantom
        print("No CADICA frames found — using synthetic phantom.")
        h, w = 256, 256
        y_grid, x_grid = np.mgrid[:h, :w]
        clean = np.zeros((h, w), dtype=np.float32)
        # Background gradient
        clean += (x_grid / w) * 0.3
        # Disk vessel
        dist = np.sqrt((y_grid - 128) ** 2 + (x_grid - 128) ** 2)
        clean += 0.5 * np.exp(-dist / 40)
        # Thin vessels
        for cx, cy, r in [(80, 90, 6), (160, 160, 5), (100, 180, 4)]:
            d = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
            clean += 0.4 * np.exp(-(d / r) ** 2)
        clean = np.clip(clean, 0, 1)

    rng = np.random.default_rng(42)

    levels = [("Clean (full dose)", None)] + [(name, params) for name, params in DOSE_LEVELS.items()]
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4))

    for ax, (label, params) in zip(axes, levels):
        if params is None:
            img = clean
        else:
            img = add_poisson_gaussian_noise(clean, rng=rng, **params)
            diff = img - clean
            noise_std = diff.std()
            psnr = 10 * np.log10(1.0 / np.mean(diff ** 2)) if noise_std > 0 else float("inf")
            label = f"{label}\nPSNR={psnr:.1f} dB"

        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle("Poisson-Gaussian noise at different dose levels", fontsize=11)
    fig.tight_layout()

    out_path = Path("noise_preview.png")
    fig.savefig(out_path, dpi=120)
    print(f"Preview saved → {out_path.resolve()}")

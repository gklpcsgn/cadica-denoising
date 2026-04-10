"""
PyTorch Dataset classes for CADICA coronary angiography denoising.

CADICA folder layout::

    <root>/
        patientXX/
            videoXX/
                frameXXXX.png

Two Dataset variants
--------------------

SingleFrameDataset
    Returns ``(noisy, clean)`` — both shape ``(1, H, W)``.
    For each frame, ``patches_per_frame`` random crops are drawn, so
    ``len(dataset) == n_frames * patches_per_frame``.

TemporalDataset
    Returns ``(noisy_stack, clean_center)`` where ``noisy_stack`` is
    ``(3, H, W)`` (independent noise on frames t-1, t, t+1) and
    ``clean_center`` is ``(1, H, W)`` (clean frame t).
    Boundary frames (first and last of each video) are skipped.
    The same random crop coordinates are used for all three frames.

Both datasets
    - Load grayscale PNGs as float32 in [0, 1].
    - Generate noise on-the-fly, so each epoch sees different noise
      realisations (true when DataLoader worker seeds differ per epoch).
    - Accept a list of patient directory ``Path`` objects (as returned by
      ``download.patient_level_split``).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.noise import add_poisson_gaussian_noise, DOSE_LEVELS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_videos(patient_dirs: list[Path]) -> list[Path]:
    """Return sorted list of all video directories across the given patients."""
    videos: list[Path] = []
    for p in patient_dirs:
        for v in sorted(p.iterdir()):
            if v.is_dir():
                videos.append(v)
    return videos


def _collect_frames(video_dir: Path) -> list[Path]:
    """Return sorted PNG frame paths inside a single video directory."""
    return sorted(video_dir.glob("*.png"))


def _load_gray(path: Path) -> np.ndarray:
    """Load a PNG as a float32 array in [0, 1], shape (H, W)."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _random_crop(
    arrays: list[np.ndarray],
    patch_size: int,
    rng: random.Random,
) -> list[np.ndarray]:
    """Apply the same random square crop to every array in *arrays*.

    All arrays must have the same (H, W) spatial dimensions.
    """
    h, w = arrays[0].shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"patch_size={patch_size} exceeds image dimensions ({h}×{w}). "
            "Use a smaller patch_size or larger images."
        )
    top  = rng.randint(0, h - patch_size)
    left = rng.randint(0, w - patch_size)
    return [a[top : top + patch_size, left : left + patch_size] for a in arrays]


# ---------------------------------------------------------------------------
# SingleFrameDataset
# ---------------------------------------------------------------------------

class SingleFrameDataset(Dataset):
    """Returns (noisy, clean) pairs — both shape (1, H, W).

    Args:
        patient_dirs:     List of patient directory Paths to include.
        dose_level:       Key in ``DOSE_LEVELS`` — "low25", "low10", or "low5".
        patch_size:       Side length of the random square crop (pixels).
        patches_per_frame: Number of random crops drawn from each frame.
                           ``len(dataset) == n_frames * patches_per_frame``.
    """

    def __init__(
        self,
        patient_dirs: list[Path],
        dose_level: str = "low25",
        patch_size: int = 128,
        patches_per_frame: int = 4,
    ) -> None:
        super().__init__()
        if dose_level not in DOSE_LEVELS:
            raise ValueError(f"Unknown dose_level '{dose_level}'. Choose from {list(DOSE_LEVELS)}")

        self.dose_params      = DOSE_LEVELS[dose_level]
        self.patch_size       = patch_size
        self.patches_per_frame = patches_per_frame

        # Build flat index: each entry is (frame_path, crop_seed_offset)
        self._index: list[tuple[Path, int]] = []
        for video_dir in _collect_videos(patient_dirs):
            for frame_path in _collect_frames(video_dir):
                for crop_idx in range(patches_per_frame):
                    self._index.append((frame_path, crop_idx))

        if not self._index:
            raise RuntimeError(
                "No PNG frames found under the given patient directories. "
                "Check the CADICA folder structure."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_path, crop_idx = self._index[idx]

        clean = _load_gray(frame_path)                          # (H, W)

        # Deterministic crop position: reproducible given idx but unique per
        # crop_idx so multiple patches from the same frame differ.
        crop_rng = random.Random(idx * 97 + crop_idx)
        (clean_crop,) = _random_crop([clean], self.patch_size, crop_rng)

        # On-the-fly noise — no fixed seed so each DataLoader epoch varies.
        noisy_crop = add_poisson_gaussian_noise(clean_crop, **self.dose_params)

        clean_t = torch.from_numpy(clean_crop).unsqueeze(0)     # (1, H, W)
        noisy_t = torch.from_numpy(noisy_crop).unsqueeze(0)     # (1, H, W)
        return noisy_t, clean_t


# ---------------------------------------------------------------------------
# TemporalDataset
# ---------------------------------------------------------------------------

class TemporalDataset(Dataset):
    """Returns (noisy_stack, clean_center) triplets — (3, H, W) and (1, H, W).

    Loads frames at positions (t-1, t, t+1) from the same video, applies
    **independent** noise to each frame, and stacks the noisy versions along
    the channel axis.  The same random crop is used for all three frames so
    spatial alignment is preserved.

    Boundary frames (index 0 and last) in each video are skipped so that
    t-1 and t+1 are always valid.

    Args:
        patient_dirs:     List of patient directory Paths to include.
        dose_level:       Key in ``DOSE_LEVELS``.
        patch_size:       Side length of the random square crop.
        patches_per_frame: Random crops per valid centre frame.
    """

    def __init__(
        self,
        patient_dirs: list[Path],
        dose_level: str = "low25",
        patch_size: int = 128,
        patches_per_frame: int = 4,
    ) -> None:
        super().__init__()
        if dose_level not in DOSE_LEVELS:
            raise ValueError(f"Unknown dose_level '{dose_level}'. Choose from {list(DOSE_LEVELS)}")

        self.dose_params       = DOSE_LEVELS[dose_level]
        self.patch_size        = patch_size
        self.patches_per_frame = patches_per_frame

        # Build flat index: each entry is (prev, curr, next, crop_offset)
        self._index: list[tuple[Path, Path, Path, int]] = []
        for video_dir in _collect_videos(patient_dirs):
            frames = _collect_frames(video_dir)
            if len(frames) < 3:
                continue                                        # skip too-short videos
            for t in range(1, len(frames) - 1):                # skip boundary frames
                for crop_idx in range(patches_per_frame):
                    self._index.append((frames[t - 1], frames[t], frames[t + 1], crop_idx))

        if not self._index:
            raise RuntimeError(
                "No valid triplets found. Each video must have ≥ 3 frames. "
                "Check the CADICA folder structure."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prev_path, curr_path, next_path, crop_idx = self._index[idx]

        prev_clean = _load_gray(prev_path)
        curr_clean = _load_gray(curr_path)
        next_clean = _load_gray(next_path)

        # Same crop coordinates for all three frames
        crop_rng = random.Random(idx * 97 + crop_idx)
        prev_c, curr_c, next_c = _random_crop(
            [prev_clean, curr_clean, next_clean], self.patch_size, crop_rng
        )

        # Independent noise per frame (physically correct — each frame is an
        # independent X-ray exposure)
        prev_n = add_poisson_gaussian_noise(prev_c, **self.dose_params)
        curr_n = add_poisson_gaussian_noise(curr_c, **self.dose_params)
        next_n = add_poisson_gaussian_noise(next_c, **self.dose_params)

        noisy_stack = torch.stack([
            torch.from_numpy(prev_n),
            torch.from_numpy(curr_n),
            torch.from_numpy(next_n),
        ], dim=0)                                               # (3, H, W)
        clean_center = torch.from_numpy(curr_c).unsqueeze(0)   # (1, H, W)

        return noisy_stack, clean_center


# ---------------------------------------------------------------------------
# Quick self-test (run with:  python -m data.dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from PIL import Image as PILImage

    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Fake CADICA layout: 3 patients, 1 video each, 6 frames each
        for pid in ["patient01", "patient02", "patient03"]:
            video_dir = root / pid / "video01"
            video_dir.mkdir(parents=True)
            for i in range(6):
                arr = rng.integers(0, 256, (64, 64), dtype=np.uint8)
                PILImage.fromarray(arr, mode="L").save(
                    video_dir / f"frame{i:04d}.png"
                )

        patients = [root / p for p in ["patient01", "patient02", "patient03"]]

        # SingleFrameDataset
        ds = SingleFrameDataset(patients, dose_level="low25", patch_size=32, patches_per_frame=4)
        noisy, clean = ds[0]
        assert noisy.shape == (1, 32, 32) and clean.shape == (1, 32, 32)
        assert noisy.dtype == torch.float32
        print(f"SingleFrameDataset  len={len(ds)}  noisy={tuple(noisy.shape)}  clean={tuple(clean.shape)}")

        # TemporalDataset
        ds_t = TemporalDataset(patients, dose_level="low10", patch_size=32, patches_per_frame=4)
        stack, center = ds_t[0]
        assert stack.shape == (3, 32, 32) and center.shape == (1, 32, 32)
        print(f"TemporalDataset     len={len(ds_t)}  stack={tuple(stack.shape)}  center={tuple(center.shape)}")

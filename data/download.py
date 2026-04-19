"""
Download the CADICA dataset from Kaggle and split patients into train/val/test.

Usage
-----
    # Download + split
    python data/download.py --dest data/cadica

    # Skip download if already present, just (re)print the split
    python data/download.py --dest data/cadica --skip-download

Requirements
------------
    pip install kaggle
    Place ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY env vars).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

DATASET_SLUG = "arejimenezpartinen/cadica"

CADICA_ROOT = Path("data/archive/CADICA a new dataset for coronary artery disease/CADICA/CADICA")


def download_cadica(dest: str | Path = "data/cadica") -> Path:
    """Download and unzip the CADICA dataset using the Kaggle CLI.

    Args:
        dest: Local directory where the dataset will be extracted.

    Returns:
        Path to the extracted dataset root.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_SLUG} → {dest.resolve()} ...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(dest),
            "--unzip",
        ],
        check=True,
    )
    print("Download complete.")
    return dest


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------

def patient_level_split(
    data_dir: str | Path = CADICA_ROOT / "selectedVideos",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split patient folders into train / val / test by patient ID.

    Splitting at the patient level (not the frame level) prevents data
    leakage: frames from the same patient are highly correlated, so mixing
    them across splits inflates val/test metrics.

    CADICA folder layout (selectedVideos only)::

        <data_dir>/
            p1/
                v7/
                    input/
                        p1_v7_00001.png
            p2/
                ...

    Args:
        data_dir:   Root of selectedVideos (p1, p2, ... folders live here).
        val_ratio:  Fraction of patients reserved for validation.
        test_ratio: Fraction of patients reserved for testing.
        seed:       Random seed for reproducible shuffling.

    Returns:
        Dict ``{"train": [...], "val": [...], "test": [...]}`` where each
        value is a sorted list of **Path objects** pointing to patient dirs.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    data_dir = Path(data_dir)
    patient_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not patient_dirs:
        raise FileNotFoundError(
            f"No patient directories found under {data_dir}. "
            "Run download_cadica() first, or check the path."
        )

    rng = random.Random(seed)
    shuffled = patient_dirs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val  = round(n * val_ratio)
    n_test = round(n * test_ratio)
    n_train = n - n_val - n_test

    split = {
        "train": sorted(shuffled[:n_train]),
        "val":   sorted(shuffled[n_train : n_train + n_val]),
        "test":  sorted(shuffled[n_train + n_val :]),
    }

    print(
        f"Patient split (seed={seed}, total={n}): "
        f"train={len(split['train'])}, "
        f"val={len(split['val'])}, "
        f"test={len(split['test'])}"
    )
    return split


def save_split(split: dict[str, list[Path]], out_file: str | Path) -> None:
    """Save the split to JSON (stores directory names, not full paths)."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: [p.name for p in v] for k, v in split.items()}
    with open(out_file, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Split saved → {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CADICA and create patient-level split.")
    parser.add_argument("--dest",          default="data/archive/CADICA a new dataset for coronary artery disease/CADICA/CADICA/selectedVideos",    help="selectedVideos root directory")
    parser.add_argument("--split-out",     default="data/split.json", help="Where to save the split JSON")
    parser.add_argument("--val-ratio",     type=float, default=0.15)
    parser.add_argument("--test-ratio",    type=float, default=0.15)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; just compute and print the split")
    args = parser.parse_args()

    if not args.skip_download:
        download_cadica(args.dest)

    split = patient_level_split(
        args.dest,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_split(split, args.split_out)


if __name__ == "__main__":
    main()

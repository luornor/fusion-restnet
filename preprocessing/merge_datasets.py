#!/usr/bin/env python3
"""
merge_datasets.py - Merge individual dataset .npy files into data/combined/.

Run this once after cloning (or whenever a new dataset is added):
    python preprocessing/merge_datasets.py

Sources merged (in order):
    data/plaid/X_real.npy     + y_real.npy     (PLAID,  60 Hz US)
    data/hifda/X_hifda.npy    + y_hifda.npy    (HIFDA,  50 Hz ES)
    data/whited/X_whited.npy  + y_whited.npy   (WHITED, 50 Hz AT)
    data/cooll/X_cooll.npy    + y_cooll.npy    (COOLL,  50 Hz FR)

Output:
    data/combined/X_combined.npy   shape (N, 400)  float32
    data/combined/y_combined.npy   shape (N,)      int64
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOURCES = [
    ("data/plaid/X_real.npy",   "data/plaid/y_real.npy"),
    ("data/hifda/X_hifda.npy",  "data/hifda/y_hifda.npy"),
    ("data/whited/X_whited.npy", "data/whited/y_whited.npy"),
    ("data/cooll/X_cooll.npy",  "data/cooll/y_cooll.npy"),
]

OUT_DIR = Path("data/combined")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_x = root / OUT_DIR / "X_combined.npy"
    out_y = root / OUT_DIR / "y_combined.npy"

    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)

    X_parts, y_parts = [], []
    for x_rel, y_rel in SOURCES:
        x_path = root / x_rel
        y_path = root / y_rel
        if not x_path.exists():
            print(f"[SKIP] {x_rel} not found")
            continue
        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        assert X.shape[1] == 400, f"Expected 400-sample windows, got {X.shape}"
        X_parts.append(X)
        y_parts.append(y)
        print(f"[OK]   {x_rel:35s}  {X.shape[0]:>6,} windows")

    if not X_parts:
        print("[ERROR] No source files found.")
        sys.exit(1)

    X_combined = np.concatenate(X_parts, axis=0)
    y_combined = np.concatenate(y_parts, axis=0)

    np.save(out_x, X_combined)
    np.save(out_y, y_combined)

    print(f"\n[DONE] X_combined: {X_combined.shape}  ({out_x.stat().st_size / 1e6:.1f} MB)")
    print(f"       y_combined: {y_combined.shape}")
    print(f"       Saved to  : {OUT_DIR}/")


if __name__ == "__main__":
    main()

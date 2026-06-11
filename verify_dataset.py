#!/usr/bin/env python3
"""
verify_dataset.py ? Verify that a dataset directory is compatible with train_fusion_resnet.py.

Usage:
  python verify_dataset.py --data-dir data
  python verify_dataset.py --data-dir data/hifda
  python verify_dataset.py --data-dir data --reference-dir data   # compare two dirs
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EXPECTED_WINDOW  = 400
EXPECTED_DTYPE   = np.float32
EXPECTED_Y_DTYPE = np.int64
MIN_SAMPLES_PER_CLASS = 10

PLAID_CLASSES = [
    "Air Conditioner",           # 0
    "Blender",                   # 1
    "Coffee maker",              # 2
    "Compact Fluorescent Lamp",  # 3
    "Fan",                       # 4
    "Fridge",                    # 5
    "Hair Iron",                 # 6
    "Hairdryer",                 # 7
    "Heater",                    # 8
    "Incandescent Light Bulb",   # 9
    "Laptop",                    # 10
    "Microwave",                 # 11
    "Soldering Iron",            # 12
    "Vacuum",                    # 13
    "Washing Machine",           # 14
    "Water kettle",              # 15
]


def class_name(label_id: int) -> str:
    if 0 <= label_id < len(PLAID_CLASSES):
        return PLAID_CLASSES[label_id]
    return f"unknown_{label_id}"


def check(data_dir: str, reference_dir: str | None = None, verbose: bool = True) -> bool:
    data_dir = Path(data_dir)
    ok       = True
    passed   = []
    failed   = []

    def pass_(msg):
        passed.append(msg)
        if verbose:
            print(f"  [OK]  {msg}")

    def fail(msg):
        failed.append(msg)
        nonlocal ok
        ok = False
        if verbose:
            print(f"  [FAIL]  {msg}")

    def warn(msg):
        if verbose:
            print(f"  [WARN]   {msg}")

    # -- 1. File existence -----------------------------------------------------
    x_path = data_dir / "X_real.npy"
    y_path = data_dir / "y_real.npy"

    x_exists = x_path.exists()
    y_exists = y_path.exists()

    # Also accept X_hifda / X_combined if X_real not present
    alt_names = ["X_hifda.npy", "X_combined.npy"]
    alt_y     = ["y_hifda.npy", "y_combined.npy"]
    if not x_exists:
        for alt, alty in zip(alt_names, alt_y):
            if (data_dir / alt).exists():
                x_path = data_dir / alt
                y_path = data_dir / alty
                x_exists = x_path.exists()
                y_exists = y_path.exists()
                if verbose:
                    print(f"  [info] Using {alt} / {alty} (X_real.npy not found)")
                break

    if not x_exists:
        fail(f"X array not found in {data_dir} "
             f"(tried X_real.npy, X_hifda.npy, X_combined.npy)")
        return False
    if not y_exists:
        fail(f"y array not found (expected {y_path})")
        return False

    pass_(f"Array files found: {x_path.name}, {y_path.name}")

    # -- 2. Load ---------------------------------------------------------------
    try:
        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
    except Exception as e:
        fail(f"Failed to load arrays: {e}")
        return False

    # -- 3. Shape --------------------------------------------------------------
    if X.ndim != 2:
        fail(f"X must be 2-D, got {X.ndim}-D  shape={X.shape}")
        return False
    pass_(f"X shape: {X.shape}  (N={X.shape[0]:,}, window={X.shape[1]})")

    if X.shape[1] != EXPECTED_WINDOW:
        fail(f"X window length {X.shape[1]} != expected {EXPECTED_WINDOW}")
    else:
        pass_(f"Window length {X.shape[1]} matches expected {EXPECTED_WINDOW}")

    if y.ndim != 1:
        fail(f"y must be 1-D, got {y.ndim}-D  shape={y.shape}")
    else:
        pass_(f"y shape: {y.shape}")

    if len(X) != len(y):
        fail(f"len(X)={len(X)} != len(y)={len(y)}")
    else:
        pass_(f"X and y lengths match ({len(X):,})")

    # -- 4. Dtype --------------------------------------------------------------
    if X.dtype != EXPECTED_DTYPE:
        fail(f"X dtype {X.dtype} != expected {EXPECTED_DTYPE}")
    else:
        pass_(f"X dtype: {X.dtype}")

    if y.dtype != EXPECTED_Y_DTYPE:
        warn(f"y dtype {y.dtype} (expected {EXPECTED_Y_DTYPE}) ? will cast in training")
    else:
        pass_(f"y dtype: {y.dtype}")

    # -- 5. NaN / Inf ----------------------------------------------------------
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0:
        fail(f"X contains {n_nan:,} NaN values")
    else:
        pass_("No NaN values in X")
    if n_inf > 0:
        fail(f"X contains {n_inf:,} Inf values")
    else:
        pass_("No Inf values in X")

    # -- 6. Label space --------------------------------------------------------
    unique_labels = sorted(np.unique(y).tolist())
    max_label     = max(unique_labels)
    if max_label >= len(PLAID_CLASSES):
        fail(f"y contains label {max_label} outside PLAID class range "
             f"[0, {len(PLAID_CLASSES)-1}]")
    else:
        pass_(f"All {len(unique_labels)} label IDs within valid range [0, {len(PLAID_CLASSES)-1}]")

    # -- 7. Per-class counts ---------------------------------------------------
    print()
    print(f"  {'Label':>5}  {'Class Name':<28}  {'Count':>7}  {'Status'}")
    print(f"  {'-'*60}")
    total = 0
    classes_above_min = 0
    for uid in unique_labels:
        cnt  = int((y == uid).sum())
        name = class_name(uid)
        total += cnt
        if cnt >= MIN_SAMPLES_PER_CLASS:
            status = "[OK]"
            classes_above_min += 1
        else:
            status = f"[WARN]  < {MIN_SAMPLES_PER_CLASS} -> will be dropped by train script"
        print(f"  {uid:>5}  {name:<28}  {cnt:>7,}  {status}")
    print(f"  {'-'*60}")
    print(f"  {'':>5}  {'TOTAL':<28}  {total:>7,}")
    print()
    pass_(f"{classes_above_min} classes with ? {MIN_SAMPLES_PER_CLASS} samples "
          f"(will survive training filter)")

    # -- 8. Signal statistics --------------------------------------------------
    per_sample_absmax = np.max(np.abs(X), axis=1)
    per_sample_rms    = np.sqrt(np.mean(X**2, axis=1))
    global_mean       = float(np.mean(X))

    print(f"  Signal statistics:")
    print(f"    global min       : {X.min():.4f}")
    print(f"    global max       : {X.max():.4f}")
    print(f"    global mean      : {global_mean:.6f}  "
          f"{'(near zero [OK])' if abs(global_mean) < 0.1 else '(non-zero [WARN])'}")
    print(f"    global std       : {X.std():.4f}")
    print(f"    per-sample absmax  min={per_sample_absmax.min():.4f}  "
          f"max={per_sample_absmax.max():.4f}  "
          f"median={float(np.median(per_sample_absmax)):.4f}")
    print(f"    per-sample RMS   : mean={per_sample_rms.mean():.4f}  "
          f"std={per_sample_rms.std():.4f}")

    if abs(global_mean) > 1.0:
        fail(f"Global mean {global_mean:.4f} is large ? possible unremoved DC offset")
    else:
        pass_("Global mean near zero (DC offset appears removed)")

    # -- 9. Compare to reference -----------------------------------------------
    if reference_dir and Path(reference_dir) != data_dir:
        ref_x = Path(reference_dir) / "X_real.npy"
        ref_y = Path(reference_dir) / "y_real.npy"
        if ref_x.exists():
            print()
            print(f"  [Comparison with reference: {reference_dir}]")
            X_ref = np.load(ref_x, allow_pickle=True)
            y_ref = np.load(ref_y, allow_pickle=True)
            if X.shape[1:] == X_ref.shape[1:]:
                pass_(f"Shape[1:] matches reference  {X.shape[1:]} == {X_ref.shape[1:]}")
            else:
                fail(f"Shape[1:] mismatch: {X.shape[1:]} vs reference {X_ref.shape[1:]}")
            if X.dtype == X_ref.dtype:
                pass_(f"dtype matches reference  ({X.dtype})")
            else:
                fail(f"dtype mismatch: {X.dtype} vs reference {X_ref.dtype}")

    # -- 10. Training-script CLI check -----------------------------------------
    train_script = Path("train_fusion_resnet.py")
    supported_flags = []
    if train_script.exists():
        with open(train_script) as f:
            src = f.read()
        for flag in ["--mains-freq", "--mains-volt", "--fp64",
                     "--early-stopping-patience", "--variant", "--model-version"]:
            if flag.replace("-", "_").replace("--", "") in src or flag in src:
                supported_flags.append(flag)
        print()
        print(f"  Training script flags detected: {', '.join(supported_flags)}")

    return ok


def main():
    p = argparse.ArgumentParser(
        description="Verify dataset compatibility with train_fusion_resnet.py")
    p.add_argument("--data-dir",      default="data",
                   help="Directory containing X_real.npy / y_real.npy (default: data)")
    p.add_argument("--reference-dir", default=None,
                   help="Optional reference dataset directory to compare against")
    p.add_argument("--quiet",         action="store_true",
                   help="Suppress per-check output, show only final result")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  Dataset Verification: {args.data_dir}")
    print(f"{'='*60}")

    result = check(args.data_dir, args.reference_dir, verbose=not args.quiet)

    print()
    if result:
        print(f"  [OK]  COMPATIBLE ? safe to use with train_fusion_resnet.py")
    else:
        print(f"  [FAIL]  INCOMPATIBLE ? fix the failures above before training")
        sys.exit(1)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

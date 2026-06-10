#!/usr/bin/env python3
"""
Preprocess HIFDA dataset for Fusion-ResNet NILM training.

HIFDA (High-resolution Individual Appliance Current Dataset):
  DOI: 10.1038/s41597-025-04859-3
  Source: https://zenodo.org/records/14886758
  Sampling rate: 100 kHz  |  Grid: 50 Hz / 230 V (Spain)

Outputs (saved to --out-dir, default: data/):
  X_hifda.npy          -- (N, 400) float32 normalized current windows
  y_hifda.npy          -- (N,)     int   class labels (same index space as PLAID)
  X_combined.npy       -- X_real + X_hifda merged (if --merge and X_real.npy exists)
  y_combined.npy       -- matching labels

Usage (HIFDA only):
  python prepare_hifda.py --hifda-dir /path/to/hifda

Usage (merge with existing PLAID):
  python prepare_hifda.py --hifda-dir /path/to/hifda --merge --data-dir data

Usage (train after this script):
  python train_fusion_resnet.py --data-dir data --mains-freq 50 --mains-volt 230 \
      --model-version hifda-v1 [--variant lite] [--device cuda]

  For a PLAID+HIFDA combined run use --data-dir pointing at X_combined.npy / y_combined.npy.
  NOTE: Mixed-frequency training (60 Hz PLAID + 50 Hz HIFDA) requires --mains-freq 50
  because the Fryze branch uses a single grid reference. Training at 50 Hz on the combined
  set slightly degrades PLAID-derived features, but the 50 Hz representation is what Ghana
  hardware will use at inference time — so this is the correct trade-off.

HIFDA directory structure expected (from Zenodo):
  <hifda-dir>/
    air_conditioner/   measurement_001.csv  ...
    coffee_maker/      ...
    hair_dryer/        ...
    ...

Each CSV has at least a 'current' column (or 'i', 'I', 'current_A') at 100 kHz.
Optionally a 'voltage' column — not used here (Fryze is recomputed from synthetic ref
during training or from real ZMPT101B on-device).

Class mapping (HIFDA name → model index):
  Included (8 of 14 HIFDA classes overlap with the 15-class model vocabulary):
    air_conditioner    → 0  Air Conditioner
    coffee_maker       → 1  Coffee maker
    hair_dryer         → 6  Hairdryer
    heater             → 7  Heater
    laptop             → 9  Laptop
    microwave          → 10 Microwave
    vacuum_cleaner     → 12 Vacuum
    washing_machine    → 13 Washing Machine
  Uncertain / flagged:
    iron               → 5  Hair Iron  (HIFDA 'iron' is a clothes iron, not a styling tool;
                                        waveform may differ — inspect before using)
    lamp               → 8  Incandescent Light Bulb  (verify bulb type in HIFDA metadata)
  Excluded:
    charger, desktop_computer, griddle, monitor  (no equivalent in model vocabulary)
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
from pathlib import Path

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    print("ERROR: scipy is required. Run: pip install scipy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_RATE   = 30_000   # Hz — must match PLAID / model expectation
SOURCE_RATE   = 100_000  # Hz — HIFDA native rate
WINDOW_SIZE   = 400      # samples at TARGET_RATE
SKIP_FRACTION = 0.10     # skip first+last 10% of each recording (transient region)

_g = gcd(TARGET_RATE, SOURCE_RATE)
RESAMPLE_UP   = TARGET_RATE  // _g   # 3
RESAMPLE_DOWN = SOURCE_RATE  // _g   # 10

# Model appliance index (sorted alphabetically — matches training checkpoint order)
MODEL_CLASSES = [
    'Air Conditioner',         # 0
    'Coffee maker',            # 1
    'Compact Fluorescent Lamp',# 2
    'Fan',                     # 3
    'Fridge',                  # 4
    'Hair Iron',               # 5
    'Hairdryer',               # 6
    'Heater',                  # 7
    'Incandescent Light Bulb', # 8
    'Laptop',                  # 9
    'Microwave',               # 10
    'Soldering Iron',          # 11
    'Vacuum',                  # 12
    'Washing Machine',         # 13
    'Water kettle',            # 14
]
CLASS_INDEX = {name: i for i, name in enumerate(MODEL_CLASSES)}

# HIFDA folder-name → model class name
# Keys must match the actual subdirectory names inside the HIFDA archive.
HIFDA_MAPPING = {
    'air_conditioner':  'Air Conditioner',
    'coffee_maker':     'Coffee maker',
    'hair_dryer':       'Hairdryer',
    'heater':           'Heater',
    'laptop':           'Laptop',
    'microwave':        'Microwave',
    'vacuum_cleaner':   'Vacuum',
    'washing_machine':  'Washing Machine',
    # Uncertain mappings — included by default but flagged in output
    'iron':             'Hair Iron',
    'lamp':             'Incandescent Light Bulb',
}

UNCERTAIN_CLASSES = {'iron', 'lamp'}

EXCLUDED_CLASSES = {'charger', 'desktop_computer', 'griddle', 'monitor'}

# Common column name variants across HIFDA CSV files
CURRENT_COL_VARIANTS = ['current', 'i', 'I', 'current_a', 'current_A', 'I_A', 'i_a']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_current_column(df) -> str:
    for name in CURRENT_COL_VARIANTS:
        if name in df.columns:
            return name
    cols = list(df.columns)
    raise ValueError(
        f"Cannot find current column. Available columns: {cols}. "
        "Edit CURRENT_COL_VARIANTS in this script to add your column name."
    )


def load_csv_current(path: str) -> np.ndarray:
    """Load a HIFDA CSV and return the raw current signal as float32."""
    import pandas as pd
    df = pd.read_csv(path)
    col = find_current_column(df)
    return df[col].to_numpy(dtype=np.float32)


def resample_signal(signal: np.ndarray) -> np.ndarray:
    """Resample from SOURCE_RATE to TARGET_RATE using polyphase filter."""
    return resample_poly(signal, RESAMPLE_UP, RESAMPLE_DOWN).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Unit-magnitude normalization — same as PLAID preprocessing."""
    peak = np.max(np.abs(signal))
    if peak < 1e-9:
        return signal
    return signal / peak


def extract_windows(signal: np.ndarray, window_size: int = WINDOW_SIZE,
                    skip_fraction: float = SKIP_FRACTION) -> np.ndarray:
    """
    Extract non-overlapping windows from the steady-state region of a signal.
    Skips the first and last `skip_fraction` of samples to avoid transients.
    Returns array of shape (n_windows, window_size) or empty array if signal too short.
    """
    n = len(signal)
    start = int(n * skip_fraction)
    end   = int(n * (1 - skip_fraction))
    steady = signal[start:end]

    n_windows = len(steady) // window_size
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)

    trimmed = steady[:n_windows * window_size]
    return trimmed.reshape(n_windows, window_size)


def process_appliance_dir(appl_dir: str, model_class: str, uncertain: bool,
                          verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Process all CSVs in one HIFDA appliance directory."""
    csv_files = sorted(glob.glob(os.path.join(appl_dir, '**', '*.csv'), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(appl_dir, '*.csv')))

    if not csv_files:
        warnings.warn(f"No CSV files found in {appl_dir}")
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), np.empty(0, dtype=np.int64)

    class_idx = CLASS_INDEX[model_class]
    all_windows = []
    skipped = 0

    for csv_path in csv_files:
        try:
            raw = load_csv_current(csv_path)
        except Exception as e:
            warnings.warn(f"Skipping {csv_path}: {e}")
            skipped += 1
            continue

        resampled  = resample_signal(raw)
        normalized = normalize_signal(resampled)
        windows    = extract_windows(normalized)

        if len(windows) == 0:
            skipped += 1
            continue

        all_windows.append(windows)

    if not all_windows:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.full(len(X), class_idx, dtype=np.int64)

    flag = " [UNCERTAIN MAPPING — verify before training]" if uncertain else ""
    if verbose:
        print(f"  {model_class:<28} {len(X):>5} windows from {len(csv_files)} files"
              f"  (skipped {skipped}){flag}")

    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess HIFDA dataset for Fusion-ResNet NILM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--hifda-dir',  required=True,
                   help='Root directory of the downloaded HIFDA dataset')
    p.add_argument('--out-dir',    default='data',
                   help='Output directory for .npy files (default: data/)')
    p.add_argument('--data-dir',   default='data',
                   help='Directory containing existing X_real.npy / y_real.npy for merging')
    p.add_argument('--merge',      action='store_true',
                   help='Merge HIFDA output with existing PLAID data into X_combined.npy')
    p.add_argument('--exclude-uncertain', action='store_true',
                   help='Exclude iron and lamp (uncertain class mappings) from output')
    p.add_argument('--window-size', type=int, default=WINDOW_SIZE,
                   help=f'Window length in samples at {TARGET_RATE} Hz (default: {WINDOW_SIZE})')
    p.add_argument('--skip-fraction', type=float, default=SKIP_FRACTION,
                   help=f'Fraction of each recording to skip at start/end (default: {SKIP_FRACTION})')
    p.add_argument('--dry-run',    action='store_true',
                   help='Scan files and report counts without saving anything')
    return p.parse_args()


def main():
    args = parse_args()

    hifda_root = Path(args.hifda_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hifda_root.exists():
        print(f"ERROR: HIFDA directory not found: {hifda_root}")
        sys.exit(1)

    print(f"\nHIFDA preprocessing")
    print(f"  Source dir  : {hifda_root}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Resample    : {SOURCE_RATE} Hz → {TARGET_RATE} Hz  "
          f"(poly {RESAMPLE_UP}/{RESAMPLE_DOWN})")
    print(f"  Window size : {args.window_size} samples")
    print(f"  Skip frac   : {args.skip_fraction:.0%} at each end\n")

    # Scan available subdirectories
    available = {d.name.lower(): d for d in hifda_root.iterdir() if d.is_dir()}

    print("Appliance directories found:")
    for name in sorted(available):
        tag = ''
        if name in HIFDA_MAPPING:
            tag = f"→ {HIFDA_MAPPING[name]}"
            if name in UNCERTAIN_CLASSES:
                tag += " [UNCERTAIN]"
        elif name in EXCLUDED_CLASSES:
            tag = "EXCLUDED"
        else:
            tag = "UNRECOGNISED — will be skipped"
        print(f"  {name:<30} {tag}")

    print()

    all_X, all_y = [], []

    for folder_name, folder_path in sorted(available.items()):
        if folder_name in EXCLUDED_CLASSES:
            print(f"  Skipping excluded class: {folder_name}")
            continue

        if folder_name not in HIFDA_MAPPING:
            print(f"  Skipping unrecognised class: {folder_name}")
            continue

        uncertain = folder_name in UNCERTAIN_CLASSES
        if uncertain and args.exclude_uncertain:
            print(f"  Skipping uncertain class: {folder_name}")
            continue

        model_class = HIFDA_MAPPING[folder_name]

        if args.dry_run:
            csvs = list(folder_path.rglob('*.csv'))
            print(f"  {model_class:<28} {len(csvs)} CSV files (dry run)")
            continue

        X, y = process_appliance_dir(
            str(folder_path), model_class, uncertain=uncertain
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    if not all_X:
        print("ERROR: No windows extracted. Check --hifda-dir and CSV column names.")
        sys.exit(1)

    X_hifda = np.concatenate(all_X, axis=0)
    y_hifda = np.concatenate(all_y, axis=0)

    print(f"\nHIFDA dataset summary:")
    print(f"  Total windows : {len(X_hifda)}")
    print(f"  Shape         : {X_hifda.shape}")
    unique, counts = np.unique(y_hifda, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"    {MODEL_CLASSES[idx]:<28} {cnt:>5} windows")

    # Save HIFDA-only arrays
    x_path = out_dir / 'X_hifda.npy'
    y_path = out_dir / 'y_hifda.npy'
    np.save(x_path, X_hifda)
    np.save(y_path, y_hifda)
    print(f"\nSaved: {x_path}  ({X_hifda.nbytes / 1e6:.1f} MB)")
    print(f"Saved: {y_path}")

    # Optionally merge with PLAID data
    if args.merge:
        plaid_x = Path(args.data_dir) / 'X_real.npy'
        plaid_y = Path(args.data_dir) / 'y_real.npy'

        if not plaid_x.exists() or not plaid_y.exists():
            print(f"\nWARNING: --merge requested but PLAID data not found at {args.data_dir}/. "
                  "Skipping merge.")
        else:
            X_plaid = np.load(plaid_x, allow_pickle=True).astype(np.float32)
            y_plaid = np.load(plaid_y, allow_pickle=True).astype(np.int64)

            X_combined = np.concatenate([X_plaid, X_hifda], axis=0)
            y_combined = np.concatenate([y_plaid, y_hifda], axis=0)

            xc_path = out_dir / 'X_combined.npy'
            yc_path = out_dir / 'y_combined.npy'
            np.save(xc_path, X_combined)
            np.save(yc_path, y_combined)

            print(f"\nMerged dataset:")
            print(f"  PLAID windows : {len(X_plaid)}")
            print(f"  HIFDA windows : {len(X_hifda)}")
            print(f"  Combined      : {len(X_combined)}")
            print(f"  Saved: {xc_path}  ({X_combined.nbytes / 1e6:.1f} MB)")
            print(f"  Saved: {yc_path}")

            print("\nTo train on combined data (50 Hz/230 V for Ghana deployment):")
            print("  python train_fusion_resnet.py \\")
            print(f"    --data-dir {out_dir} \\")
            print("    --mains-freq 50 --mains-volt 230 \\")
            print("    --model-version combined-50hz-v1 \\")
            print("    --epochs 300 --early-stopping-patience 40 [--variant lite] [--device cuda]")

    print("\nTo train on HIFDA only (50 Hz / 230 V):")
    print("  python train_fusion_resnet.py \\")
    print(f"    --data-dir {out_dir} \\")
    print("    --mains-freq 50 --mains-volt 230 \\")
    print("    --model-version hifda-v1 \\")
    print("    --epochs 300 --early-stopping-patience 40 [--variant lite] [--device cuda]")
    print()
    print("NOTE: Run with --data-dir pointing at X_hifda.npy / y_hifda.npy by renaming")
    print("      those files to X_real.npy / y_real.npy, or pass --data-dir to a separate")
    print("      directory containing the HIFDA files under those names.")


if __name__ == '__main__':
    main()

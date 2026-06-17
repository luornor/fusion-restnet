#!/usr/bin/env python3
"""
prepare_whited.py - Preprocess WHITED dataset for Fusion-ResNet NILM training.

WHITED: Worldwide Household and Industry Transient Energy Data
  - 44,100 Hz, stereo FLAC (channel 0 = current, channel 1 = voltage)
  - ~5.1 seconds per recording (startup transient + steady-state)
  - All recordings confirmed 50 Hz mains (MK1 and MK2 measurement kits)
  - 1,339 single-appliance recordings across 56 appliance types

Pipeline per file:
  1. Read current channel (ch0) from FLAC
  2. Skip first 1 second (startup transient)
  3. Remove DC offset (subtract per-recording mean)
  4. Resample 44,100 Hz -> 30,000 Hz via resample_poly(up=100, down=147)
  5. Extract non-overlapping 400-sample windows
  6. Reject near-zero windows (RMS < 0.001)
  7. Map appliance type to PLAID class ID

Output:
  data/whited/X_whited.npy   -- (N, 400) float32
  data/whited/y_whited.npy   -- (N,) int64
  data/whited/whited_class_map.json
  data/whited/whited_preprocess_report.json / .md

Usage:
  python preprocessing/prepare_whited.py --whited-dir ../WhiteD/DATEN
  python preprocessing/prepare_whited.py --whited-dir ../WhiteD/DATEN --include-uncertain
  python preprocessing/prepare_whited.py --whited-dir ../WhiteD/DATEN --merge --reference-data-dir data/combined
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_RATE   = 44_100   # Hz -- WHITED native
TARGET_RATE   = 30_000   # Hz -- must match PLAID
WINDOW_SIZE   = 400      # samples
SKIP_SECONDS  = 1.0      # skip first N seconds (startup transient)
SKIP_SAMPLES  = int(SOURCE_RATE * SKIP_SECONDS)
MIN_RMS       = 0.001    # reject near-zero / dead-signal windows

# Resampling ratio: 44100 -> 30000
# GCD(44100, 30000) = 300  =>  44100/300=147, 30000/300=100
RESAMPLE_UP   = 100
RESAMPLE_DOWN = 147

# PLAID class vocabulary (16 classes, alphabetically sorted, same as training)
PLAID_CLASSES = [
    "Air Conditioner",          # 0
    "Blender",                  # 1
    "Coffee maker",             # 2
    "Compact Fluorescent Lamp", # 3
    "Fan",                      # 4
    "Fridge",                   # 5
    "Hair Iron",                # 6
    "Hairdryer",                # 7
    "Heater",                   # 8
    "Incandescent Light Bulb",  # 9
    "Laptop",                   # 10
    "Microwave",                # 11
    "Soldering Iron",           # 12
    "Vacuum",                   # 13
    "Washing Machine",          # 14
    "Water kettle",             # 15
]

PLAID_CLASS_ID = {name: i for i, name in enumerate(PLAID_CLASSES)}

# Confident appliance-type -> PLAID class mappings
SAFE_MAPPING = {
    "AC":           "Air Conditioner",
    "CFL":          "Compact Fluorescent Lamp",
    "CoffeeMachine":"Coffee maker",
    "Fan":          "Fan",
    "Fridge":       "Fridge",
    "HairDryer":    "Hairdryer",
    "Hairdryer":    "Hairdryer",
    "Heater":       "Heater",
    "Kettle":       "Water kettle",
    "Laptop":       "Laptop",
    "Microwave":    "Microwave",
    "SolderingIron":"Soldering Iron",
    "VacuumCleaner":"Vacuum",
    "WashingMachine":"Washing Machine",
}

# Uncertain mappings (appliance type differs from PLAID's definition)
UNCERTAIN_MAPPING = {
    "FlatIron":   "Hair Iron",            # clothes iron != hair iron
    "Iron":       "Hair Iron",            # clothes iron != hair iron
    "LightBulb":  "Incandescent Light Bulb",  # may include LED/halogen
    "FanHeater":  "Heater",              # combination device
    "WaterHeater":"Heater",              # different thermal load profile
    "Mixer":      "Blender",             # food mixer similar load to blender
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def appliance_from_filename(fname: str) -> str:
    return fname.split("_")[0]


def process_file(path: Path) -> np.ndarray | None:
    """Return (N, 400) float32 windows for one FLAC file, or None on error."""
    try:
        data, sr = sf.read(str(path))
    except Exception as e:
        print(f"    [WARN] Cannot read {path.name}: {e}")
        return None

    if data.ndim == 1:
        current = data.astype(np.float32)
    else:
        current = data[:, 0].astype(np.float32)  # channel 0 = current

    # Skip transient
    current = current[SKIP_SAMPLES:]
    if len(current) < WINDOW_SIZE:
        return None

    # Remove DC
    current = current - current.mean()

    # Resample
    resampled = resample_poly(current, RESAMPLE_UP, RESAMPLE_DOWN).astype(np.float32)

    # Windows
    n = len(resampled) // WINDOW_SIZE
    if n == 0:
        return None
    windows = resampled[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)

    # Reject near-zero
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    good = windows[rms >= MIN_RMS]
    return good if len(good) > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess WHITED dataset for Fusion-ResNet NILM training")
    p.add_argument("--whited-dir", default="../WhiteD/DATEN",
                   help="Path to WHITED DATEN/ folder (default: ../WhiteD/DATEN)")
    p.add_argument("--out-dir", default="data/whited",
                   help="Output directory (default: data/whited)")
    p.add_argument("--reference-data-dir", default=None,
                   help="If set, verify shape/dtype compatibility with X_real.npy in this dir")
    p.add_argument("--merge", action="store_true",
                   help="Merge with data in --reference-data-dir and save to data/combined/")
    p.add_argument("--include-uncertain", action="store_true", default=False,
                   help="Include uncertain class mappings (FlatIron->Hair Iron, etc.)")
    p.add_argument("--max-windows-per-class", type=int, default=2000,
                   help="Max windows to keep per PLAID class (default: 2000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be processed without writing files")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    whited_dir = Path(args.whited_dir)
    out_dir    = Path(args.out_dir)

    if not whited_dir.exists():
        print(f"[ERROR] WHITED directory not found: {whited_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    x_out = out_dir / "X_whited.npy"
    y_out = out_dir / "y_whited.npy"
    if x_out.exists() and not args.force and not args.dry_run:
        print(f"[INFO] Output already exists. Use --force to overwrite.")
        sys.exit(0)

    # Build active mapping
    active_mapping = dict(SAFE_MAPPING)
    if args.include_uncertain:
        active_mapping.update(UNCERTAIN_MAPPING)
        print("[INFO] Including UNCERTAIN class mappings.")
    else:
        print("[INFO] Using SAFE mappings only. Use --include-uncertain to add more.")

    # Discover FLAC files
    flac_files = [f for f in whited_dir.iterdir() if f.suffix == ".flac"]
    print(f"\n[INFO] Found {len(flac_files)} FLAC files in {whited_dir}")

    # Group by appliance type and filter
    grouped: dict[str, list[Path]] = defaultdict(list)
    skipped_types: set[str] = set()
    for fpath in sorted(flac_files):
        app_type = appliance_from_filename(fpath.name)
        if app_type in active_mapping:
            grouped[app_type].append(fpath)
        else:
            skipped_types.add(app_type)

    print(f"[INFO] Appliance types to process: {sorted(grouped.keys())}")
    print(f"[INFO] Appliance types skipped:    {sorted(skipped_types)}")

    # Process
    class_windows: dict[int, list[np.ndarray]] = defaultdict(list)
    file_report: list[dict] = []
    total_files = 0
    total_windows = 0

    for app_type, files in sorted(grouped.items()):
        plaid_name = active_mapping[app_type]
        class_id   = PLAID_CLASS_ID[plaid_name]
        mapping_type = "SAFE" if app_type in SAFE_MAPPING else "UNCERTAIN"

        print(f"\n  [{app_type}] -> class {class_id} ({plaid_name})  [{mapping_type}]  {len(files)} files")

        for fpath in files:
            if args.dry_run:
                print(f"    [DRY] Would process: {fpath.name}")
                continue

            windows = process_file(fpath)
            if windows is None:
                print(f"    [SKIP] {fpath.name}: no valid windows")
                continue

            class_windows[class_id].extend(windows)
            total_files += 1
            print(f"    {fpath.name}: {len(windows)} windows -> class {class_id}")

        if not args.dry_run:
            n = sum(1 for _ in grouped[app_type])
            file_report.append({
                "whited_type": app_type,
                "plaid_class": plaid_name,
                "class_id": class_id,
                "mapping_type": mapping_type,
                "files_processed": total_files,
            })

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Cap per class
    print(f"\n[INFO] Capping at {args.max_windows_per_class} windows per class...")
    X_parts, y_parts = [], []
    class_map = {}

    for class_id in sorted(class_windows.keys()):
        wins = class_windows[class_id]
        rng.shuffle(wins)
        wins = wins[:args.max_windows_per_class]
        arr = np.stack(wins, axis=0)
        X_parts.append(arr)
        y_parts.append(np.full(len(arr), class_id, dtype=np.int64))
        plaid_name = PLAID_CLASSES[class_id]
        class_map[class_id] = plaid_name
        total_windows += len(arr)
        print(f"  class {class_id:>2} ({plaid_name:<28}): {len(arr):>5} windows")

    if not X_parts:
        print("[ERROR] No windows produced. Check --whited-dir path.")
        sys.exit(1)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)

    print(f"\n[INFO] Final dataset: X={X.shape}  y={y.shape}  dtype={X.dtype}")

    # Validate
    assert X.ndim == 2 and X.shape[1] == WINDOW_SIZE, f"Bad shape: {X.shape}"
    assert len(X) == len(y)
    assert not np.isnan(X).any(), "NaN in X"
    assert not np.isinf(X).any(), "Inf in X"
    assert abs(X.mean()) < 1.0, f"Large global mean {X.mean():.4f} -- DC offset issue?"

    # Save
    np.save(x_out, X)
    np.save(y_out, y)
    print(f"[OK] Saved: {x_out}  ({x_out.stat().st_size/1e6:.1f} MB)")
    print(f"[OK] Saved: {y_out}")

    # Class map
    class_map_path = out_dir / "whited_class_map.json"
    with open(class_map_path, "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
    print(f"[OK] Saved: {class_map_path}")

    # Report
    report = {
        "source":         "WHITED",
        "source_rate_hz": SOURCE_RATE,
        "target_rate_hz": TARGET_RATE,
        "window_size":    WINDOW_SIZE,
        "skip_seconds":   SKIP_SECONDS,
        "resample_up":    RESAMPLE_UP,
        "resample_down":  RESAMPLE_DOWN,
        "uncertain_included": args.include_uncertain,
        "max_windows_per_class": args.max_windows_per_class,
        "total_files_processed": total_files,
        "total_windows": int(total_windows),
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "global_mean": float(X.mean()),
        "global_std":  float(X.std()),
        "classes": {
            PLAID_CLASSES[cid]: int((y == cid).sum())
            for cid in sorted(class_windows.keys())
        },
    }

    report_json = out_dir / "whited_preprocess_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    report_md = out_dir / "whited_preprocess_report.md"
    with open(report_md, "w") as f:
        f.write("# WHITED Preprocessing Report\n\n")
        f.write(f"- Source rate: {SOURCE_RATE:,} Hz\n")
        f.write(f"- Target rate: {TARGET_RATE:,} Hz\n")
        f.write(f"- Resample: {RESAMPLE_UP}/{RESAMPLE_DOWN} (up/down)\n")
        f.write(f"- Skip transient: {SKIP_SECONDS}s\n")
        f.write(f"- Window size: {WINDOW_SIZE} samples\n")
        f.write(f"- Uncertain mappings: {'YES' if args.include_uncertain else 'NO'}\n")
        f.write(f"- Total files processed: {total_files}\n")
        f.write(f"- Total windows: {total_windows:,}\n")
        f.write(f"- X shape: {X.shape}  dtype: {X.dtype}\n")
        f.write(f"- Global mean: {X.mean():.6f}  std: {X.std():.4f}\n\n")
        f.write("## Per-Class Counts\n\n")
        f.write("| Class ID | Appliance | Windows |\n")
        f.write("|---|---|---|\n")
        for cid in sorted(class_windows.keys()):
            n = int((y == cid).sum())
            f.write(f"| {cid} | {PLAID_CLASSES[cid]} | {n:,} |\n")

    print(f"[OK] Saved: {report_json}")
    print(f"[OK] Saved: {report_md}")

    # Optional: verify against reference data
    if args.reference_data_dir:
        ref_dir = Path(args.reference_data_dir)
        for ref_name in ["X_real.npy", "X_combined.npy"]:
            ref_path = ref_dir / ref_name
            if ref_path.exists():
                X_ref = np.load(ref_path, allow_pickle=True)
                ok = True
                if X_ref.shape[1] != WINDOW_SIZE:
                    print(f"[FAIL] Window size mismatch: {X_ref.shape[1]} vs {WINDOW_SIZE}")
                    ok = False
                if X_ref.dtype != X.dtype:
                    print(f"[WARN] dtype mismatch: {X_ref.dtype} vs {X.dtype}")
                if ok:
                    print(f"[OK] Compatible with {ref_name} (shape[1]={WINDOW_SIZE}, dtype={X.dtype})")
                break

    # Optional: merge with reference dataset
    if args.merge and args.reference_data_dir:
        ref_dir = Path(args.reference_data_dir)
        combined_dir = Path("data/combined")
        combined_dir.mkdir(parents=True, exist_ok=True)

        for ref_x_name, ref_y_name in [
            ("X_real.npy", "y_real.npy"),
            ("X_combined.npy", "y_combined.npy"),
        ]:
            rx = ref_dir / ref_x_name
            ry = ref_dir / ref_y_name
            if rx.exists() and ry.exists():
                X_ref = np.load(rx, allow_pickle=True).astype(np.float32)
                y_ref = np.load(ry, allow_pickle=True).astype(np.int64)
                X_merged = np.concatenate([X_ref, X], axis=0)
                y_merged = np.concatenate([y_ref, y], axis=0)
                np.save(combined_dir / "X_combined.npy", X_merged)
                np.save(combined_dir / "y_combined.npy", y_merged)
                print(f"\n[MERGE] {ref_x_name} ({X_ref.shape[0]:,}) + WHITED ({X.shape[0]:,})"
                      f" -> combined ({X_merged.shape[0]:,})")
                break

    print(f"\n[DONE] WHITED preprocessing complete.")
    print(f"  SAFE_TO_TRAIN: {not args.include_uncertain or 'review uncertain mappings'}")


if __name__ == "__main__":
    main()

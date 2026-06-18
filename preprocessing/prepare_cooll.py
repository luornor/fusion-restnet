#!/usr/bin/env python3
"""
prepare_cooll.py - Preprocess COOLL dataset for Fusion-ResNet NILM training.

COOLL: Controlled On/Off Loads Library
  - 100,000 Hz mono FLAC (current only; voltage in separate zips)
  - 6 seconds per recording (includes controlled on + steady-state + off)
  - 50 Hz mains (France)
  - 12 appliance types x 70 recordings = 840 total scenarios

Scenario numbering (alphabetical, 70 recordings per appliance):
  1-70   : Drill               -> EXCLUDE
  71-140 : Fan                 -> Fan (class 4)          [SAFE]
  141-210: Grinder             -> EXCLUDE
  211-280: Hair dryer          -> Hairdryer (class 7)    [SAFE]
  281-350: Hedge trimmer       -> EXCLUDE
  351-420: Lamp (fluorescent)  -> CFL (class 3)          [UNCERTAIN - tube != compact]
  421-490: Paint stripper      -> EXCLUDE
  491-560: Planer              -> EXCLUDE
  561-630: Router              -> EXCLUDE
  631-700: Sander              -> EXCLUDE
  701-770: Saw                 -> EXCLUDE
  771-840: Vacuum cleaner      -> Vacuum (class 13)      [SAFE]

Zip layout (password: prismecooll):
  Current_1to210.zip   -> scenarios 1-210
  Current_211to420.zip -> scenarios 211-420
  Current_421to630.zip -> scenarios 421-630
  Current_631to840.zip -> scenarios 631-840

Pipeline per file:
  1. Extract FLAC from zip (password protected)
  2. Decode to raw PCM via ffmpeg (soundfile cannot handle 100 kHz FLAC)
  3. Skip first 1 second (on-transient)
  4. Remove DC offset (subtract mean)
  5. Resample 100,000 Hz -> 30,000 Hz via resample_poly(up=3, down=10)
  6. Extract non-overlapping 400-sample windows
  7. Reject near-zero windows (RMS < 0.001)

Usage:
  python preprocessing/prepare_cooll.py --cooll-dir "../COOLL Data/COOLL Data"
  python preprocessing/prepare_cooll.py --cooll-dir "../COOLL Data/COOLL Data" --include-uncertain
  python preprocessing/prepare_cooll.py --cooll-dir "../COOLL Data/COOLL Data" --merge --reference-data-dir data/combined
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_RATE    = 100_000  # Hz
TARGET_RATE    = 30_000   # Hz
WINDOW_SIZE    = 400
SKIP_SECONDS   = 1.0
SKIP_SAMPLES   = int(SOURCE_RATE * SKIP_SECONDS)
MIN_RMS        = 0.001
RESAMPLE_UP    = 3
RESAMPLE_DOWN  = 10
ZIP_PASSWORD   = b'prismecooll'

# Scenario ranges (1-indexed, inclusive)
SCENARIO_MAP = {
    # (start, end): (plaid_class_id, plaid_name, mapping_type)
    (71,  140): (4,  "Fan",                      "SAFE"),
    (211, 280): (7,  "Hairdryer",                "SAFE"),
    (771, 840): (13, "Vacuum",                   "SAFE"),
    # Uncertain — only used with --include-uncertain
    (351, 420): (3,  "Compact Fluorescent Lamp", "UNCERTAIN"),
}

# Which zip covers which scenario range
ZIP_RANGES = [
    (1,   210, "Current_1to210.zip"),
    (211, 420, "Current_211to420.zip"),
    (421, 630, "Current_421to630.zip"),
    (631, 840, "Current_631to840.zip"),
]

PLAID_CLASSES = [
    "Air Conditioner", "Blender", "Coffee maker", "Compact Fluorescent Lamp",
    "Fan", "Fridge", "Hair Iron", "Hairdryer", "Heater",
    "Incandescent Light Bulb", "Laptop", "Microwave", "Soldering Iron",
    "Vacuum", "Washing Machine", "Water kettle",
]


def zip_for_scenario(scenario: int, cooll_dir: Path) -> Path:
    for start, end, fname in ZIP_RANGES:
        if start <= scenario <= end:
            return cooll_dir / fname
    raise ValueError(f"No zip for scenario {scenario}")


def decode_flac_ffmpeg(flac_path: Path, tmp_dir: Path) -> np.ndarray | None:
    raw_path = tmp_dir / "out.raw"
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(flac_path),
        "-f", "s16le", "-ar", str(SOURCE_RATE), "-ac", "1", str(raw_path),
    ], capture_output=True)
    if result.returncode != 0:
        return None
    raw = np.frombuffer(raw_path.read_bytes(), dtype=np.int16).astype(np.float32)
    raw /= 32768.0
    return raw


def process_signal(raw: np.ndarray) -> np.ndarray | None:
    signal = raw[SKIP_SAMPLES:]
    if len(signal) < WINDOW_SIZE:
        return None
    signal = signal - signal.mean()
    resampled = resample_poly(signal, RESAMPLE_UP, RESAMPLE_DOWN).astype(np.float32)
    n = len(resampled) // WINDOW_SIZE
    if n == 0:
        return None
    windows = resampled[: n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    good = windows[rms >= MIN_RMS]
    return good if len(good) > 0 else None


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess COOLL for Fusion-ResNet NILM")
    p.add_argument("--cooll-dir", default="../COOLL Data/COOLL Data",
                   help="Path to folder containing the Current_*.zip files")
    p.add_argument("--out-dir", default="data/cooll",
                   help="Output directory (default: data/cooll)")
    p.add_argument("--reference-data-dir", default=None,
                   help="Verify compatibility with X_real.npy / X_combined.npy in this dir")
    p.add_argument("--merge", action="store_true",
                   help="Merge with data/combined/ after processing")
    p.add_argument("--include-uncertain", action="store_true", default=False,
                   help="Include uncertain Lamp->CFL mapping")
    p.add_argument("--max-windows-per-class", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cooll_dir = Path(args.cooll_dir)
    out_dir   = Path(args.out_dir)

    if not cooll_dir.exists():
        print(f"[ERROR] COOLL directory not found: {cooll_dir}")
        sys.exit(1)

    zips = [cooll_dir / fname for _, _, fname in ZIP_RANGES]
    missing = [z for z in zips if not z.exists()]
    if missing:
        print(f"[ERROR] Missing zip files: {missing}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    x_out = out_dir / "X_cooll.npy"
    if x_out.exists() and not args.force and not args.dry_run:
        print("[INFO] Output already exists. Use --force to overwrite.")
        sys.exit(0)

    # Build active scenario map
    active = {
        k: v for k, v in SCENARIO_MAP.items()
        if v[2] == "SAFE" or args.include_uncertain
    }
    print(f"\n[INFO] Active class ranges:")
    for (s, e), (cid, name, mtype) in sorted(active.items()):
        print(f"  Scenarios {s:>3}-{e}: class {cid} ({name})  [{mtype}]")

    # Group scenarios by zip file to minimise re-opening
    zip_work: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    for (s, e), (cid, name, mtype) in active.items():
        for scenario in range(s, e + 1):
            zpath = zip_for_scenario(scenario, cooll_dir)
            zip_work[str(zpath)].append((scenario, cid, name, mtype))

    class_windows: dict[int, list[np.ndarray]] = defaultdict(list)
    total_files = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for zpath_str, items in sorted(zip_work.items()):
            zpath = Path(zpath_str)
            print(f"\n[ZIP] {zpath.name}  ({len(items)} scenarios to extract)")

            if args.dry_run:
                for scenario, cid, name, _ in items[:3]:
                    print(f"  [DRY] Would process scenarioC1_{scenario}.flac -> class {cid} ({name})")
                print(f"  [DRY] ... and {max(0, len(items)-3)} more")
                continue

            with zipfile.ZipFile(zpath) as z:
                for scenario, cid, name, mtype in items:
                    fname = f"scenarioC1_{scenario}.flac"
                    try:
                        z.extract(fname, tmp, pwd=ZIP_PASSWORD)
                    except Exception as e:
                        print(f"  [WARN] Cannot extract {fname}: {e}")
                        continue

                    flac_path = tmp / fname
                    raw = decode_flac_ffmpeg(flac_path, tmp)
                    if raw is None:
                        print(f"  [WARN] ffmpeg failed for {fname}")
                        flac_path.unlink(missing_ok=True)
                        continue

                    windows = process_signal(raw)
                    flac_path.unlink(missing_ok=True)

                    if windows is None:
                        print(f"  [SKIP] {fname}: no valid windows")
                        continue

                    class_windows[cid].extend(windows)
                    total_files += 1
                    if total_files % 20 == 0:
                        print(f"  ... {total_files} files processed so far")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Cap per class and assemble
    print(f"\n[INFO] Capping at {args.max_windows_per_class} windows per class...")
    X_parts, y_parts = [], []
    class_map = {}
    total_windows = 0

    for cid in sorted(class_windows.keys()):
        wins = class_windows[cid]
        random.shuffle(wins)
        wins = wins[: args.max_windows_per_class]
        arr = np.stack(wins, axis=0)
        X_parts.append(arr)
        y_parts.append(np.full(len(arr), cid, dtype=np.int64))
        class_map[cid] = PLAID_CLASSES[cid]
        total_windows += len(arr)
        print(f"  class {cid:>2} ({PLAID_CLASSES[cid]:<28}): {len(arr):>5} windows")

    if not X_parts:
        print("[ERROR] No windows produced.")
        sys.exit(1)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)

    print(f"\n[INFO] Final: X={X.shape}  y={y.shape}  dtype={X.dtype}")
    assert not np.isnan(X).any() and not np.isinf(X).any()
    assert abs(X.mean()) < 1.0, f"Large global mean {X.mean():.4f}"

    np.save(out_dir / "X_cooll.npy", X)
    np.save(out_dir / "y_cooll.npy", y)
    print(f"[OK] Saved: {out_dir}/X_cooll.npy  ({(out_dir/'X_cooll.npy').stat().st_size/1e6:.1f} MB)")

    with open(out_dir / "cooll_class_map.json", "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)

    report = {
        "source": "COOLL",
        "source_rate_hz": SOURCE_RATE,
        "target_rate_hz": TARGET_RATE,
        "window_size": WINDOW_SIZE,
        "skip_seconds": SKIP_SECONDS,
        "resample_up": RESAMPLE_UP,
        "resample_down": RESAMPLE_DOWN,
        "uncertain_included": args.include_uncertain,
        "max_windows_per_class": args.max_windows_per_class,
        "total_files_processed": total_files,
        "total_windows": total_windows,
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "global_mean": float(X.mean()),
        "global_std": float(X.std()),
        "classes": {PLAID_CLASSES[cid]: int((y == cid).sum()) for cid in sorted(class_windows)},
    }
    with open(out_dir / "cooll_preprocess_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "cooll_preprocess_report.md", "w") as f:
        f.write("# COOLL Preprocessing Report\n\n")
        f.write(f"- Source: COOLL (France, 50 Hz)\n")
        f.write(f"- Source rate: {SOURCE_RATE:,} Hz -> {TARGET_RATE:,} Hz\n")
        f.write(f"- Resample: {RESAMPLE_UP}/{RESAMPLE_DOWN} (up/down)\n")
        f.write(f"- Skip transient: {SKIP_SECONDS}s\n")
        f.write(f"- Uncertain mappings: {'YES' if args.include_uncertain else 'NO'}\n")
        f.write(f"- Total windows: {total_windows:,}\n")
        f.write(f"- X shape: {X.shape}  dtype: {X.dtype}\n\n")
        f.write("## Per-Class Counts\n\n| Class | Appliance | Windows |\n|---|---|---|\n")
        for cid in sorted(class_windows):
            f.write(f"| {cid} | {PLAID_CLASSES[cid]} | {int((y==cid).sum()):,} |\n")

    print(f"[OK] Reports saved to {out_dir}/")

    # Merge into combined
    if args.merge:
        combined_dir = Path("data/combined")
        combined_dir.mkdir(parents=True, exist_ok=True)
        for rx_name, ry_name in [("X_combined.npy", "y_combined.npy"), ("X_real.npy", "y_real.npy")]:
            rx = combined_dir / rx_name
            ry = combined_dir / ry_name
            if rx.exists() and ry.exists():
                X_ref = np.load(rx).astype(np.float32)
                y_ref = np.load(ry).astype(np.int64)
                X_m = np.concatenate([X_ref, X], axis=0)
                y_m = np.concatenate([y_ref, y], axis=0)
                np.save(combined_dir / "X_combined.npy", X_m)
                np.save(combined_dir / "y_combined.npy", y_m)
                print(f"\n[MERGE] {rx_name}({X_ref.shape[0]:,}) + COOLL({X.shape[0]:,}) -> combined({X_m.shape[0]:,})")
                break

    print("\n[DONE] COOLL preprocessing complete.")


if __name__ == "__main__":
    main()

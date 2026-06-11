#!/usr/bin/env python3
"""
prepare_hifda.py ? Preprocess HIFDA for Fusion-ResNet NILM training.

Produces arrays that exactly match the schema of the existing PLAID X_real.npy / y_real.npy.

Usage examples:
  # Dry-run (inspect only, no files written):
  python prepare_hifda.py --hifda-dir C:/path/to/HIFDA_HF_electrical_signals_dataset --dry-run

  # Generate HIFDA-only arrays (safe classes only):
  python prepare_hifda.py --hifda-dir C:/path/to/HIFDA_HF_electrical_signals_dataset \
      --out-dir data/hifda --reference-data-dir data --exclude-uncertain

  # Include uncertain classes (iron -> Hair Iron, light -> ILB):
  python prepare_hifda.py --hifda-dir C:/path/to/HIFDA_HF_electrical_signals_dataset \
      --out-dir data/hifda --reference-data-dir data --include-uncertain

  # Merge with PLAID:
  python prepare_hifda.py --hifda-dir C:/path/to/HIFDA_HF_electrical_signals_dataset \
      --out-dir data --reference-data-dir data --merge --exclude-uncertain

  # Custom class map:
  python prepare_hifda.py --hifda-dir C:/path/to/HIFDA --class-map custom.json ...

Outputs (--out-dir data/hifda):
  X_hifda.npy, y_hifda.npy, hifda_class_map.json
  hifda_preprocess_report.json, hifda_preprocess_report.md

Outputs (--merge, --out-dir data):
  X_combined.npy, y_combined.npy
  combined_preprocess_report.json, combined_preprocess_report.md

HIFDA dataset info (DOI: 10.1038/s41597-025-04859-3):
  Sampling rate : 100 kHz
  Grid          : 50 Hz / 230 V (Spain)
  Window splits : 10.24ms, 163.84ms, 1310.72ms, Full_time
  Format        : Current/<appliance>/*.txt  (one float per line, no header)
  DC offset     : ~1.647 constant sensor bias ? removed by subtracting per-window mean

Selected split: 163.84ms_window_dataset
  Rationale: 16384 samples/file at 100kHz -> resample to 30kHz -> 4915 samples
  -> 12 non-overlapping 400-sample sub-windows per file (matches PLAID window length).
  The 10.24ms split (1024 samples -> 307 after resample) is shorter than the
  required 400-sample window and would need zero-padding, distorting the waveform.
"""

import os
import sys
import json
import math
import time
import random
import argparse
import warnings
import textwrap
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    print("ERROR: scipy is required.  pip install scipy")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SOURCE_RATE   = 100_000   # Hz ? HIFDA native sampling rate
TARGET_RATE   = 30_000    # Hz ? must match PLAID / model expectation
WINDOW_SIZE   = 400       # samples at TARGET_RATE (matches PLAID X_real shape[1])
HIFDA_SPLIT   = "163.84ms_window_dataset"

_g            = gcd(TARGET_RATE, SOURCE_RATE)
RESAMPLE_UP   = TARGET_RATE  // _g   # 3
RESAMPLE_DOWN = SOURCE_RATE  // _g   # 10

# After resampling 163.84ms (16384 samples @ 100kHz) -> 4915.2 -> 4915 samples @ 30kHz
RESAMPLED_LEN = int(16384 * RESAMPLE_UP / RESAMPLE_DOWN)
WINDOWS_PER_FILE = RESAMPLED_LEN // WINDOW_SIZE   # 12

# -----------------------------------------------------------------------------
# Label encoder ? PLAID raw class IDs (must match real_label_encoder.npy order)
# -----------------------------------------------------------------------------
# These 16 names and their integer indices match the sklearn LabelEncoder
# serialised in data/real_label_encoder.npy (alphabetically sorted).
PLAID_CLASSES = [
    "Air Conditioner",          # 0
    "Blender",                  # 1  (< 10 samples ? dropped by train script)
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
CLASS_INDEX = {name: i for i, name in enumerate(PLAID_CLASSES)}

# HIFDA folder name -> PLAID class name
# Keys must be lowercase versions of actual HIFDA sub-directory names.
SAFE_MAPPING = {
    "air_conditioner": "Air Conditioner",
    "coffeemaker":     "Coffee maker",
    "hairdryer":       "Hairdryer",
    "heater":          "Heater",
    "laptop":          "Laptop",
    "microwave":       "Microwave",
    "vacuum":          "Vacuum",
    "washing_machine": "Washing Machine",
}

UNCERTAIN_MAPPING = {
    # HIFDA 'Iron' is a clothes iron; PLAID 'Hair Iron' is a styling tool.
    # Current waveforms may differ significantly ? verify before including.
    "iron":  "Hair Iron",
    # HIFDA 'Light' type unknown (could be CFL, LED, or ILB).
    # Only include if README or metadata confirms it is an incandescent bulb.
    "light": "Incandescent Light Bulb",
}

EXCLUDED_FOLDERS = {
    "charger", "computer", "emptygrid", "griddle", "monitor"
}


# -----------------------------------------------------------------------------
# Signal processing helpers
# -----------------------------------------------------------------------------

def load_txt_signal(path: str) -> np.ndarray | None:
    """Load a single-column HIFDA .txt file -> 1-D float32 array, or None on error."""
    try:
        data = np.loadtxt(path, dtype=np.float64)
        if data.ndim != 1:
            data = data.ravel()
        if len(data) == 0:
            return None
        return data.astype(np.float32)
    except Exception:
        return None


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract per-window mean (removes ~1.647 ADC bias present in all HIFDA files)."""
    return signal - signal.mean()


def resample_signal(signal: np.ndarray) -> np.ndarray:
    """Polyphase resample from SOURCE_RATE (100kHz) to TARGET_RATE (30kHz)."""
    resampled = resample_poly(signal.astype(np.float64), RESAMPLE_UP, RESAMPLE_DOWN)
    return resampled.astype(np.float32)


def extract_windows(signal: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Extract non-overlapping windows. Returns (n_windows, window_size) or empty."""
    n_windows = len(signal) // window_size
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)
    trimmed = signal[:n_windows * window_size]
    return trimmed.reshape(n_windows, window_size).copy()


def process_file(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Full pipeline for one HIFDA .txt file.
    Returns (windows array shape (n,400), list of rejection reasons).
    """
    reasons = []

    raw = load_txt_signal(path)
    if raw is None:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), ["unreadable"]

    if np.isnan(raw).any() or np.isinf(raw).any():
        raw = raw[np.isfinite(raw)]
        if len(raw) < WINDOW_SIZE:
            return np.empty((0, WINDOW_SIZE), dtype=np.float32), ["nan_inf_too_short"]

    detrended  = remove_dc(raw)
    resampled  = resample_signal(detrended)
    windows    = extract_windows(resampled)

    if len(windows) == 0:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), ["too_short_after_resample"]

    # Sanity: reject windows that are all zeros (silent channel)
    valid_mask = np.max(np.abs(windows), axis=1) > 1e-9
    windows    = windows[valid_mask]
    n_rejected_zero = (~valid_mask).sum()
    if n_rejected_zero > 0:
        reasons.append(f"zero_windows:{n_rejected_zero}")

    if len(windows) == 0:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32), reasons + ["all_zero"]

    return windows, reasons


# -----------------------------------------------------------------------------
# Directory helpers
# -----------------------------------------------------------------------------

def discover_hifda_root(hifda_dir: str) -> Path:
    """
    HIFDA archives may be nested (e.g. archive/HIFDA_HF.../HIFDA_HF.../163.84ms...).
    Walk up to 2 levels to find the directory that contains the expected split folder.
    """
    root = Path(hifda_dir)
    if (root / HIFDA_SPLIT).exists():
        return root
    for child in root.iterdir():
        if child.is_dir() and (child / HIFDA_SPLIT).exists():
            return child
        for grandchild in child.iterdir() if child.is_dir() else []:
            if grandchild.is_dir() and (grandchild / HIFDA_SPLIT).exists():
                return grandchild
    return root  # fall through ? caller will handle missing split


def build_class_map(hifda_current_dir: Path,
                    include_uncertain: bool,
                    custom_map: dict | None) -> dict[str, dict]:
    """
    Return {hifda_folder_name: {model_class, label_id, uncertain, excluded}} for
    every sub-directory found under hifda_current_dir.
    """
    mapping = {}
    effective_safe      = SAFE_MAPPING.copy()
    effective_uncertain = UNCERTAIN_MAPPING.copy()

    if custom_map:
        for k, v in custom_map.items():
            kl = k.lower()
            if v is None:
                effective_safe.pop(kl, None)
                effective_uncertain.pop(kl, None)
                EXCLUDED_FOLDERS.add(kl)
            elif v in CLASS_INDEX:
                effective_safe[kl]      = v
                effective_uncertain.pop(kl, None)
            else:
                warnings.warn(f"Custom map: '{v}' is not a known PLAID class ? ignoring {k}")

    for folder in sorted(hifda_current_dir.iterdir()):
        if not folder.is_dir():
            continue
        fname = folder.name
        fl    = fname.lower()
        if fl in EXCLUDED_FOLDERS:
            mapping[fname] = {"status": "excluded", "reason": "no_model_equivalent"}
            continue
        if fl in effective_safe:
            model_class = effective_safe[fl]
            mapping[fname] = {
                "status":      "included",
                "model_class": model_class,
                "label_id":    CLASS_INDEX[model_class],
                "uncertain":   False,
            }
        elif fl in effective_uncertain:
            model_class = effective_uncertain[fl]
            mapping[fname] = {
                "status":      "uncertain",
                "model_class": model_class,
                "label_id":    CLASS_INDEX[model_class],
                "uncertain":   True,
                "included":    include_uncertain,
            }
        else:
            mapping[fname] = {"status": "unrecognised"}
    return mapping


# -----------------------------------------------------------------------------
# Per-class processing
# -----------------------------------------------------------------------------

def process_class(folder: Path, label_id: int, model_class: str,
                  max_files: int | None, max_windows_per_class: int | None,
                  seed: int, verbose: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process all .txt files in one HIFDA appliance folder."""
    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        txt_files = sorted(folder.rglob("*.txt"))

    if not txt_files:
        return (np.empty((0, WINDOW_SIZE), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                {"files_found": 0, "files_accepted": 0, "files_rejected": 0,
                 "windows": 0, "reject_reasons": {}})

    if max_files is not None and len(txt_files) > max_files:
        rng = random.Random(seed)
        txt_files = rng.sample(txt_files, max_files)

    all_windows   = []
    accepted      = 0
    rejected      = 0
    reject_counts = defaultdict(int)

    for fpath in txt_files:
        windows, reasons = process_file(str(fpath))
        if len(windows) == 0:
            rejected += 1
            for r in reasons:
                reject_counts[r] += 1
        else:
            accepted += 1
            all_windows.append(windows)

    if not all_windows:
        return (np.empty((0, WINDOW_SIZE), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                {"files_found": len(txt_files), "files_accepted": 0,
                 "files_rejected": rejected, "windows": 0,
                 "reject_reasons": dict(reject_counts)})

    X = np.concatenate(all_windows, axis=0)

    # Cap windows per class
    if max_windows_per_class is not None and len(X) > max_windows_per_class:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_windows_per_class, replace=False)
        X   = X[idx]

    y = np.full(len(X), label_id, dtype=np.int64)

    stats = {
        "files_found":    len(txt_files),
        "files_accepted": accepted,
        "files_rejected": rejected,
        "windows":        len(X),
        "reject_reasons": dict(reject_counts),
    }

    if verbose:
        uncertain_tag = " [UNCERTAIN]" if model_class in {
            v for v in UNCERTAIN_MAPPING.values()} else ""
        print(f"    {model_class:<28} label={label_id:2d}  "
              f"files={accepted}/{len(txt_files)}  windows={len(X)}{uncertain_tag}")

    return X, y, stats


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_arrays(X: np.ndarray, y: np.ndarray,
                    X_ref: np.ndarray, y_ref: np.ndarray,
                    label: str = "HIFDA") -> list[str]:
    """Run all validation checks. Returns list of FAILED check descriptions."""
    failures = []

    if X.dtype != X_ref.dtype:
        failures.append(f"{label} X dtype {X.dtype} != reference {X_ref.dtype}")
    if X.shape[1:] != X_ref.shape[1:]:
        failures.append(f"{label} X shape[1:] {X.shape[1:]} != reference {X_ref.shape[1:]}")
    if y.shape[1:] != y_ref.shape[1:]:
        failures.append(f"{label} y shape[1:] {y.shape[1:]} != reference {y_ref.shape[1:]}")
    if y.dtype != y_ref.dtype:
        failures.append(f"{label} y dtype {y.dtype} != reference {y_ref.dtype}")
    if np.isnan(X).any():
        failures.append(f"{label} X contains NaNs")
    if np.isinf(X).any():
        failures.append(f"{label} X contains Infs")
    if len(X) == 0:
        failures.append(f"{label} X is empty")
    if len(X) != len(y):
        failures.append(f"{label} len(X)={len(X)} != len(y)={len(y)}")

    # Check every label exists in PLAID class space
    bad_labels = set(np.unique(y)) - set(range(len(PLAID_CLASSES)))
    if bad_labels:
        failures.append(f"{label} y has out-of-range labels: {bad_labels}")

    # Check no empty classes
    unique, counts = np.unique(y, return_counts=True)
    for uid, cnt in zip(unique, counts):
        if cnt == 0:
            failures.append(f"{label} class {uid} is empty")

    return failures


def signal_stats(X: np.ndarray) -> dict:
    per_rms = np.sqrt(np.mean(X**2, axis=1))
    return {
        "n_windows":   int(len(X)),
        "global_min":  float(np.nanmin(X)),
        "global_max":  float(np.nanmax(X)),
        "global_mean": float(np.nanmean(X)),
        "global_std":  float(np.nanstd(X)),
        "rms_mean":    float(per_rms.mean()),
        "rms_std":     float(per_rms.std()),
        "rms_min":     float(per_rms.min()),
        "rms_max":     float(per_rms.max()),
        "rms_p25":     float(np.percentile(per_rms, 25)),
        "rms_p75":     float(np.percentile(per_rms, 75)),
    }


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def class_counts(y: np.ndarray, plaid_classes: list[str]) -> dict:
    counts = {}
    unique, c = np.unique(y, return_counts=True)
    for uid, cnt in zip(unique, c):
        name = plaid_classes[uid] if uid < len(plaid_classes) else f"unknown_{uid}"
        counts[name] = int(cnt)
    return counts


def write_md_report(path: Path, report: dict) -> None:
    lines = [
        f"# HIFDA Preprocessing Report",
        f"",
        f"Generated: {report['generated_at']}",
        f"",
        f"## Summary",
        f"",
        f"| Item | Value |",
        f"|---|---|",
        f"| HIFDA root | `{report['hifda_root']}` |",
        f"| Split used | `{report['split_used']}` |",
        f"| Uncertain classes | {'included' if report['uncertain_included'] else 'excluded'} |",
        f"| Total HIFDA windows | {report['hifda_total_windows']} |",
        f"| X_hifda shape | {report['X_hifda_shape']} |",
        f"| X_hifda dtype | {report['X_hifda_dtype']} |",
        f"| y_hifda shape | {report['y_hifda_shape']} |",
        f"| **SAFE_TO_TRAIN** | **{'[OK] YES' if report['safe_to_train'] else 'FAIL NO'}** |",
        f"",
        f"## Class Mapping",
        f"",
        f"| HIFDA Folder | Model Class | Label ID | Status |",
        f"|---|---|---|---|",
    ]
    for folder, info in report["class_map"].items():
        status = info.get("status", "?")
        mc     = info.get("model_class", "?")
        lid    = info.get("label_id", "?")
        lines.append(f"| {folder} | {mc} | {lid} | {status} |")

    lines += [
        f"",
        f"## Per-Class Statistics",
        f"",
        f"| Class | PLAID Count | HIFDA Count | Combined |",
        f"|---|---|---|---|",
    ]
    plaid_counts    = report.get("plaid_class_counts", {})
    hifda_counts    = report.get("hifda_class_counts", {})
    combined_counts = report.get("combined_class_counts", {})
    all_classes     = sorted(set(list(plaid_counts) + list(hifda_counts)))
    for cls in all_classes:
        pc = plaid_counts.get(cls, 0)
        hc = hifda_counts.get(cls, 0)
        cc = combined_counts.get(cls, pc + hc)
        lines.append(f"| {cls} | {pc} | {hc} | {cc} |")

    lines += [
        f"",
        f"## Signal Statistics",
        f"",
        f"| Metric | PLAID | HIFDA |",
        f"|---|---|---|",
    ]
    ps = report.get("plaid_signal_stats", {})
    hs = report.get("hifda_signal_stats", {})
    for k in ["global_min", "global_max", "global_mean", "global_std",
              "rms_mean", "rms_std", "rms_min", "rms_max"]:
        pv = f"{ps.get(k, 'N/A'):.4f}" if isinstance(ps.get(k), float) else str(ps.get(k, "N/A"))
        hv = f"{hs.get(k, 'N/A'):.4f}" if isinstance(hs.get(k), float) else str(hs.get(k, "N/A"))
        lines.append(f"| {k} | {pv} | {hv} |")

    if report.get("validation_failures"):
        lines += [f"", f"## FAIL Validation Failures", f""]
        for fail in report["validation_failures"]:
            lines.append(f"- {fail}")
    else:
        lines += [f"", f"## [OK] All Validation Checks Passed", f""]

    lines += [
        f"",
        f"## Preprocessing Steps Applied",
        f"",
        f"1. Load HIFDA Current .txt file (one float per line, no header)",
        f"2. Remove DC offset: subtract per-window mean (~1.647 ADC bias)",
        f"3. Resample: 100 kHz -> 30 kHz via polyphase filter (up={RESAMPLE_UP}, down={RESAMPLE_DOWN})",
        f"4. Extract {WINDOWS_PER_FILE} non-overlapping 400-sample sub-windows per file",
        f"5. Reject all-zero windows (silent channel artefacts)",
        f"6. Cast to float32",
        f"",
        f"## Reproduction Commands",
        f"",
        f"```bash",
    ]
    for cmd in report.get("reproduction_commands", []):
        lines.append(cmd)
    lines += ["```", ""]

    if report.get("recommended_train_command"):
        lines += [
            f"## Recommended Training Command",
            f"",
            f"```bash",
            report["recommended_train_command"],
            f"```",
            f"",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess HIFDA dataset for Fusion-ResNet NILM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              Dry-run:
                python prepare_hifda.py --hifda-dir /path/to/HIFDA --dry-run

              Generate HIFDA arrays (exclude uncertain):
                python prepare_hifda.py --hifda-dir /path/to/HIFDA \\
                    --out-dir data/hifda --reference-data-dir data --exclude-uncertain

              Merge with PLAID:
                python prepare_hifda.py --hifda-dir /path/to/HIFDA \\
                    --out-dir data --reference-data-dir data --merge --exclude-uncertain
        """),
    )
    p.add_argument("--hifda-dir",           required=True,
                   help="Root of downloaded HIFDA dataset")
    p.add_argument("--out-dir",             default="data/hifda",
                   help="Output directory (default: data/hifda)")
    p.add_argument("--reference-data-dir",  default="data",
                   help="Directory with existing PLAID X_real.npy / y_real.npy (default: data)")

    unc = p.add_mutually_exclusive_group()
    unc.add_argument("--include-uncertain", dest="include_uncertain",
                     action="store_true",  default=False,
                     help="Include iron->Hair Iron and light->ILB mappings (off by default)")
    unc.add_argument("--exclude-uncertain", dest="include_uncertain",
                     action="store_false",
                     help="Exclude uncertain mappings (default ? safe)")

    p.add_argument("--class-map",           default=None,
                   help="Path to JSON file overriding class mapping "
                        '(e.g. {"iron": null} to exclude, or {"iron": "Hair Iron"} to include)')
    p.add_argument("--merge",               action="store_true",
                   help="Combine HIFDA with existing PLAID arrays into X_combined / y_combined")
    p.add_argument("--max-files-per-class", type=int, default=None,
                   help="Debug: limit source .txt files per class")
    p.add_argument("--max-hifda-per-class", type=int, default=2000,
                   help="Cap HIFDA windows per class (default: 2000 to avoid "
                        "dominating PLAID; set 0 to disable)")
    p.add_argument("--force",               action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--dry-run",             action="store_true",
                   help="Inspect and report only ? do not write any files")
    p.add_argument("--seed",                type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args       = parse_args()
    start_time = time.time()

    hifda_dir  = discover_hifda_root(args.hifda_dir)
    out_dir    = Path(args.out_dir)
    ref_dir    = Path(args.reference_data_dir)
    max_cap    = args.max_hifda_per_class if args.max_hifda_per_class > 0 else None

    print(f"\n{'='*70}")
    print(f"  HIFDA Preprocessing Pipeline")
    print(f"{'='*70}")
    print(f"  HIFDA root     : {hifda_dir}")
    print(f"  Split          : {HIFDA_SPLIT}")
    print(f"  Resample       : {SOURCE_RATE} Hz -> {TARGET_RATE} Hz  "
          f"(poly {RESAMPLE_UP}/{RESAMPLE_DOWN})")
    print(f"  Window size    : {WINDOW_SIZE} samples  "
          f"({WINDOW_SIZE/TARGET_RATE*1000:.2f} ms at {TARGET_RATE} Hz)")
    print(f"  Windows/file   : {WINDOWS_PER_FILE}")
    print(f"  Cap/class      : {max_cap if max_cap else 'disabled'}")
    print(f"  Uncertain      : {'included' if args.include_uncertain else 'excluded (default)'}")
    print(f"  Dry run        : {'YES ? no files will be written' if args.dry_run else 'no'}")
    print()

    # -- 1. Locate split ------------------------------------------------------
    split_dir   = hifda_dir / HIFDA_SPLIT
    current_dir = split_dir / "Current"
    if not current_dir.exists():
        print(f"ERROR: Expected directory not found: {current_dir}")
        print(f"  Available under {hifda_dir}:")
        for item in sorted(hifda_dir.iterdir()):
            print(f"    {item.name}")
        sys.exit(1)

    # -- 2. Load reference PLAID data -----------------------------------------
    plaid_x_path = ref_dir / "X_real.npy"
    plaid_y_path = ref_dir / "y_real.npy"
    X_plaid, y_plaid = None, None

    if plaid_x_path.exists() and plaid_y_path.exists():
        print(f"[Reference] Loading PLAID arrays from {ref_dir} ...")
        X_plaid = np.load(plaid_x_path, allow_pickle=True)
        y_plaid = np.load(plaid_y_path, allow_pickle=True)
        plaid_counts = class_counts(y_plaid, PLAID_CLASSES)
        print(f"  X_plaid : {X_plaid.shape}  dtype={X_plaid.dtype}")
        print(f"  y_plaid : {y_plaid.shape}  dtype={y_plaid.dtype}")
        print(f"  Classes : {len(np.unique(y_plaid))} unique raw IDs")
    else:
        print(f"WARNING: PLAID reference arrays not found at {ref_dir}. "
              f"Validation against PLAID schema will be skipped.")
        plaid_counts = {}
        X_plaid = np.empty((0, WINDOW_SIZE), dtype=np.float32)
        y_plaid = np.empty(0, dtype=np.int64)

    print()

    # -- 3. Load custom class map ----------------------------------------------
    custom_map = None
    if args.class_map:
        with open(args.class_map) as f:
            custom_map = json.load(f)
        print(f"[Custom map] loaded from {args.class_map}")

    # -- 4. Build class mapping ------------------------------------------------
    class_map = build_class_map(current_dir, args.include_uncertain, custom_map)
    print("[Class mapping]")
    for folder, info in class_map.items():
        status = info["status"]
        if status == "included":
            print(f"  [OK] {folder:<22} -> {info['model_class']} (label {info['label_id']})")
        elif status == "uncertain":
            act = "[OK] INCLUDED [UNCERTAIN]" if info["included"] else "[WARN]  EXCLUDED [UNCERTAIN]"
            print(f"  {act}  {folder:<22} -> {info['model_class']} (label {info['label_id']})")
        elif status == "excluded":
            print(f"  - {folder:<22} excluded ({info.get('reason', '')})")
        else:
            print(f"  ? {folder:<22} unrecognised ? skipped")
    print()

    if args.dry_run:
        print("[Dry-run] Scanning file counts per class ...")
        total_files = 0
        for folder, info in class_map.items():
            if info["status"] not in ("included",) and not (
                    info["status"] == "uncertain" and info.get("included")):
                continue
            d = current_dir / folder
            n = len(list(d.glob("*.txt")))
            est_windows = n * WINDOWS_PER_FILE
            capped = min(est_windows, max_cap) if max_cap else est_windows
            total_files += n
            print(f"  {info['model_class']:<28} {n} files "
                  f"-> est. {est_windows} windows -> capped to {capped}")
        print(f"\n  Total source files : {total_files}")
        print(f"\n[Dry-run complete] No files written.")
        return

    # -- 5. Process each active class -----------------------------------------
    print("[Processing HIFDA classes]")
    all_X, all_y = [], []
    per_class_stats = {}
    class_map_out   = {}

    for folder, info in class_map.items():
        is_active = (info["status"] == "included") or (
            info["status"] == "uncertain" and info.get("included"))

        class_map_out[folder] = info

        if not is_active:
            continue

        folder_path = current_dir / folder
        if not folder_path.exists():
            print(f"  WARNING: folder not found: {folder_path}")
            continue

        X_cls, y_cls, stats = process_class(
            folder         = folder_path,
            label_id       = info["label_id"],
            model_class    = info["model_class"],
            max_files      = args.max_files_per_class,
            max_windows_per_class = max_cap,
            seed           = args.seed,
            verbose        = True,
        )

        per_class_stats[folder] = stats
        if len(X_cls) > 0:
            all_X.append(X_cls)
            all_y.append(y_cls)

    if not all_X:
        print("\nERROR: No windows extracted. "
              "Check --hifda-dir and ensure .txt files exist.")
        sys.exit(1)

    X_hifda = np.concatenate(all_X, axis=0).astype(np.float32)
    y_hifda = np.concatenate(all_y, axis=0).astype(np.int64)

    # Shuffle
    rng  = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_hifda))
    X_hifda, y_hifda = X_hifda[perm], y_hifda[perm]

    print(f"\n[HIFDA summary]  {len(X_hifda)} windows  shape={X_hifda.shape}  "
          f"dtype={X_hifda.dtype}")
    hifda_counts = class_counts(y_hifda, PLAID_CLASSES)
    for cls, cnt in sorted(hifda_counts.items()):
        pc = plaid_counts.get(cls, 0)
        print(f"  {cls:<28} HIFDA={cnt:5d}  PLAID={pc:5d}  combined={cnt+pc}")

    # -- 6. Validate -----------------------------------------------------------
    print("\n[Validation]")
    failures = validate_arrays(X_hifda, y_hifda, X_plaid, y_plaid, label="HIFDA")
    if failures:
        print("  FAIL VALIDATION FAILED:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  [OK] All validation checks passed")

    # -- 7. Save HIFDA-only arrays ---------------------------------------------
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

        x_out = out_dir / "X_hifda.npy"
        y_out = out_dir / "y_hifda.npy"

        if x_out.exists() and not args.force:
            print(f"\nWARNING: {x_out} already exists. Use --force to overwrite.")
        else:
            np.save(x_out, X_hifda)
            np.save(y_out, y_hifda)
            print(f"\n  Saved: {x_out}  ({X_hifda.nbytes/1e6:.1f} MB)")
            print(f"  Saved: {y_out}")

        # -- 8. Class map JSON -------------------------------------------------
        cm_out = out_dir / "hifda_class_map.json"
        with open(cm_out, "w") as f:
            json.dump(class_map_out, f, indent=2)
        print(f"  Saved: {cm_out}")

    # -- 9. Merge with PLAID ---------------------------------------------------
    combined_counts = {}
    X_combined, y_combined = None, None

    if args.merge:
        if len(X_plaid) == 0:
            print("\nWARNING: --merge requested but PLAID arrays are empty/missing. "
                  "Skipping merge.")
        else:
            X_combined = np.concatenate([X_plaid.astype(np.float32),
                                          X_hifda], axis=0)
            y_combined = np.concatenate([y_plaid.astype(np.int64),
                                          y_hifda], axis=0)

            # Shuffle combined
            perm2 = np.random.default_rng(args.seed).permutation(len(X_combined))
            X_combined, y_combined = X_combined[perm2], y_combined[perm2]

            combined_counts = class_counts(y_combined, PLAID_CLASSES)

            print(f"\n[Combined dataset]  {len(X_combined)} windows  "
                  f"(PLAID={len(X_plaid)} + HIFDA={len(X_hifda)})")

            merge_failures = validate_arrays(
                X_combined, y_combined, X_plaid, y_plaid, label="Combined")
            if merge_failures:
                print("  FAIL COMBINED VALIDATION FAILED:")
                for f in merge_failures:
                    print(f"    - {f}")
                failures += merge_failures
            else:
                print("  [OK] Combined validation passed")

            if not args.dry_run:
                merge_out_dir = Path(args.out_dir)
                merge_out_dir.mkdir(parents=True, exist_ok=True)

                xc = merge_out_dir / "X_combined.npy"
                yc = merge_out_dir / "y_combined.npy"

                if xc.exists() and not args.force:
                    print(f"\n  WARNING: {xc} exists. Use --force to overwrite.")
                else:
                    np.save(xc, X_combined)
                    np.save(yc, y_combined)
                    print(f"  Saved: {xc}  ({X_combined.nbytes/1e6:.1f} MB)")
                    print(f"  Saved: {yc}")

    # -- 10. Reports -----------------------------------------------------------
    safe_to_train = len(failures) == 0
    elapsed       = time.time() - start_time

    repro_cmds = [
        f"# HIFDA only (exclude uncertain):",
        f"python prepare_hifda.py --hifda-dir \"{args.hifda_dir}\" "
        f"--out-dir \"{args.out_dir}\" "
        f"--reference-data-dir \"{args.reference_data_dir}\" "
        f"{'--include-uncertain' if args.include_uncertain else '--exclude-uncertain'} "
        f"--seed {args.seed}",
    ]
    if args.merge:
        repro_cmds += [
            "",
            "# Merge with PLAID:",
            f"python prepare_hifda.py --hifda-dir \"{args.hifda_dir}\" "
            f"--out-dir \"{args.out_dir}\" "
            f"--reference-data-dir \"{args.reference_data_dir}\" "
            f"--merge "
            f"{'--include-uncertain' if args.include_uncertain else '--exclude-uncertain'} "
            f"--seed {args.seed}",
        ]

    train_cmd = (
        "python train_fusion_resnet.py \\\n"
        "  --data-dir data \\\n"
        "  --mains-freq 50 --mains-volt 230 \\\n"
        "  --model-version combined-50hz-v1 \\\n"
        "  --epochs 300 --early-stopping-patience 40 \\\n"
        "  --variant lite --device cuda"
        if safe_to_train else None
    )

    report = {
        "generated_at":          datetime.now().isoformat(),
        "elapsed_seconds":       round(elapsed, 1),
        "hifda_root":            str(hifda_dir),
        "split_used":            HIFDA_SPLIT,
        "source_rate_hz":        SOURCE_RATE,
        "target_rate_hz":        TARGET_RATE,
        "window_size":           WINDOW_SIZE,
        "windows_per_file":      WINDOWS_PER_FILE,
        "max_hifda_per_class":   max_cap,
        "uncertain_included":    args.include_uncertain,
        "class_map":             class_map_out,
        "per_class_stats":       per_class_stats,
        "X_hifda_shape":         list(X_hifda.shape),
        "X_hifda_dtype":         str(X_hifda.dtype),
        "y_hifda_shape":         list(y_hifda.shape),
        "y_hifda_dtype":         str(y_hifda.dtype),
        "X_plaid_shape":         list(X_plaid.shape) if X_plaid is not None else None,
        "X_plaid_dtype":         str(X_plaid.dtype)  if X_plaid is not None else None,
        "hifda_total_windows":   int(len(X_hifda)),
        "plaid_class_counts":    plaid_counts,
        "hifda_class_counts":    hifda_counts,
        "combined_class_counts": combined_counts,
        "plaid_signal_stats":    signal_stats(X_plaid) if len(X_plaid) > 0 else {},
        "hifda_signal_stats":    signal_stats(X_hifda),
        "validation_failures":   failures,
        "safe_to_train":         safe_to_train,
        "reproduction_commands": repro_cmds,
        "recommended_train_command": train_cmd,
    }

    if X_combined is not None:
        report["X_combined_shape"] = list(X_combined.shape)
        report["X_combined_dtype"] = str(X_combined.dtype)

    if not args.dry_run:
        rj = out_dir / "hifda_preprocess_report.json"
        rm = out_dir / "hifda_preprocess_report.md"
        with open(rj, "w") as f:
            json.dump(report, f, indent=2)
        write_md_report(rm, report)
        print(f"\n  Saved: {rj}")
        print(f"  Saved: {rm}")

        if args.merge and X_combined is not None:
            merge_dir = Path(args.out_dir)
            crj = merge_dir / "combined_preprocess_report.json"
            crm = merge_dir / "combined_preprocess_report.md"
            with open(crj, "w") as f:
                json.dump(report, f, indent=2)
            write_md_report(crm, report)
            print(f"  Saved: {crj}")
            print(f"  Saved: {crm}")

    # -- 11. Final status ------------------------------------------------------
    print(f"\n{'='*70}")
    if safe_to_train:
        print(f"  SAFE_TO_TRAIN = true  [OK]")
        print(f"\n  Next steps:")
        print(f"  1. Verify dataset:  python verify_dataset.py --data-dir {args.out_dir}")
        if args.merge:
            print(f"  2. Copy to data/:   copy data\\X_combined.npy data\\X_real.npy")
            print(f"                      copy data\\y_combined.npy data\\y_real.npy")
        print(f"  3. Train (50 Hz / 230 V for Ghana):")
        print(f"       python train_fusion_resnet.py \\")
        print(f"         --mains-freq 50 --mains-volt 230 \\")
        print(f"         --model-version combined-50hz-v1 \\")
        print(f"         --epochs 300 --early-stopping-patience 40 \\")
        print(f"         --variant lite --device cuda")
    else:
        print(f"  SAFE_TO_TRAIN = false  FAIL")
        print(f"  Failures:")
        for f in failures:
            print(f"    - {f}")
        print(f"  Fix the above issues before retraining.")
    print(f"{'='*70}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()

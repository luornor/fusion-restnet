#!/usr/bin/env python3
"""
prepare_natural.py - Prepare PLAID Natural recordings for Fusion-ResNet inference.

The Natural dataset contains whole-home (aggregate) current recordings with multiple
appliances running simultaneously. Each .mat file is ~900 seconds at 15,384 Hz.

This script extracts the current waveform (iHall) from each .mat file and saves it
as a plain .npy array, along with a metadata JSON describing the appliances and
any annotated on/off events.

The inference pipeline handles per-window resampling and normalization, so this
script outputs raw current in Amperes — no normalization applied here.

Usage:
    python preprocessing/prepare_natural.py \\
        --mat-dir "../Matlab_Data/Natural/7/7H2A2B2C2K2F2D2" \\
        --out-dir data/natural

Then run inference on a single recording:
    python inference/inference_pipeline.py \\
        --checkpoint model_registry/<name>_best.pt \\
        --input data/natural/waveform_70000.npy \\
        --sample-rate 15384 --mains-freq 60 \\
        --output inference_results/natural_70000

Or batch over all recordings:
    for f in data/natural/waveform_7000*.npy; do
        python inference/inference_pipeline.py \\
            --checkpoint model_registry/<name>_best.pt \\
            --input $f --sample-rate 15384 --mains-freq 60 \\
            --output inference_results/$(basename $f .npy)
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


SOURCE_RATE = 15_384   # Hz — as reported in the .mat sps field
MAINS_FREQ  = 60       # Hz — US mains (confirmed from mat['mains_freq'])

# Map of known appliance keywords to our 16-class PLAID label scheme.
# Natural recordings contain appliances that may not all be in our training set.
APPLIANCE_MAP = {
    "fan":              "Fan",
    "pedestal fan":     "Fan",
    "hair dryer":       "Hairdryer",
    "hairdryer":        "Hairdryer",
    "vacuum":           "Vacuum",
    "air conditioner":  "Air Conditioner",
    "fridge":           "Fridge",
    "refrigerator":     "Fridge",
    "microwave":        "Microwave",
    "laptop":           "Laptop",
    "heater":           "Heater",
    "hot-air":          "Heater",
    "washing machine":  "Washing Machine",
    "coffee":           "Coffee maker",
    "kettle":           "Water kettle",
    "soldering":        "Soldering Iron",
    "incandescent":     "Incandescent Light Bulb",
    "cfl":              "Compact Fluorescent Lamp",
    "fluorescent":      "Compact Fluorescent Lamp",
    "blender":          "Blender",
}


def map_appliances(load_descr: str) -> dict:
    """Return dict with raw description and any matched training-set classes."""
    raw_appliances = [a.strip() for a in load_descr.split("  ") if a.strip()]
    if len(raw_appliances) == 1:
        raw_appliances = [a.strip() for a in load_descr.split(",") if a.strip()]

    matched, unmatched = [], []
    desc_lower = load_descr.lower()
    seen = set()
    for kw, label in APPLIANCE_MAP.items():
        if kw in desc_lower and label not in seen:
            matched.append(label)
            seen.add(label)

    unmatched = [a for a in load_descr.split() if a not in seen]

    return {
        "raw_description": load_descr.strip(),
        "matched_training_classes": matched,
        "note": "Unmatched appliances produce no output from the model (out-of-domain).",
    }


def extract_events(events_r: np.ndarray, sps: int) -> list[dict]:
    """Convert the events_r array into a list of timestamped on/off events."""
    ev = events_r.ravel().astype(np.int16)
    event_indices = np.where(ev != 0)[0]
    events = []
    for idx in event_indices:
        events.append({
            "time_s": round(float(idx) / sps, 4),
            "sample": int(idx),
            "type": "ON" if ev[idx] > 0 else "OFF",
        })
    return events


def process_mat(mat_path: Path, out_dir: Path) -> dict:
    """Load one .mat file, save iHall as .npy, return metadata dict."""
    stem = mat_path.stem          # e.g. "Waveform70000"
    waveform_id = stem.lower()    # e.g. "waveform70000"

    mat = sio.loadmat(str(mat_path))

    ihall     = mat["iHall"].ravel().astype(np.float32)
    events_r  = mat["events_r"].ravel()
    sps       = int(mat["sps"].ravel()[0])
    mains_freq = int(mat["mains_freq"].ravel()[0])
    load_descr = str(mat["load_descr"].ravel()[0]).strip()
    load_short = str(mat["load_descr_short"].ravel()[0]).strip()
    duration_s = len(ihall) / sps

    # Save raw current waveform
    out_npy = out_dir / f"{waveform_id}.npy"
    np.save(out_npy, ihall)

    rms = float(np.sqrt(np.mean(ihall ** 2)))
    peak = float(np.abs(ihall).max())
    events = extract_events(events_r, sps)
    appliance_info = map_appliances(load_descr)

    meta = {
        "waveform_id":   stem,
        "source_file":   mat_path.name,
        "npy_file":      out_npy.name,
        "sample_rate_hz": sps,
        "mains_freq_hz": mains_freq,
        "duration_s":    round(duration_s, 2),
        "n_samples":     len(ihall),
        "current_rms_A": round(rms, 4),
        "current_peak_A": round(peak, 4),
        "load_descr_short": load_short,
        "appliances":    appliance_info,
        "events":        events,
        "inference_command": (
            f"python inference/inference_pipeline.py "
            f"--checkpoint model_registry/<model_version>_best.pt "
            f"--input data/natural/{out_npy.name} "
            f"--sample-rate {sps} --mains-freq {mains_freq} "
            f"--output inference_results/{waveform_id}"
        ),
    }

    print(f"  [{stem}]  {duration_s:.0f}s  {rms:.3f} A rms  "
          f"{len(events)} events  -> {out_npy.name}")
    print(f"    Appliances: {load_descr[:80]}{'...' if len(load_descr)>80 else ''}")
    if appliance_info["matched_training_classes"]:
        print(f"    Model classes: {appliance_info['matched_training_classes']}")
    else:
        print(f"    Model classes: (none matched — all out-of-domain)")

    return meta


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare PLAID Natural recordings for Fusion-ResNet inference")
    p.add_argument("--mat-dir",
                   default="../Matlab_Data/Natural/7/7H2A2B2C2K2F2D2",
                   help="Directory containing Waveform7000X.mat files")
    p.add_argument("--out-dir", default="data/natural",
                   help="Output directory for .npy files and metadata")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files")
    return p.parse_args()


def main():
    args = parse_args()
    mat_dir = Path(args.mat_dir)
    out_dir  = Path(args.out_dir)

    if not mat_dir.exists():
        print(f"[ERROR] --mat-dir not found: {mat_dir}")
        sys.exit(1)

    mat_files = sorted(mat_dir.glob("Waveform7000*.mat"))
    if not mat_files:
        print(f"[ERROR] No Waveform7000*.mat files in {mat_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "natural_meta.json"
    if meta_path.exists() and not args.force:
        print(f"[INFO] {meta_path} already exists. Use --force to overwrite.")
        sys.exit(0)

    print(f"\n[INFO] Processing {len(mat_files)} Natural recordings from:\n  {mat_dir}")
    print(f"[INFO] Output directory: {out_dir}\n")

    all_meta = []
    for mat_path in mat_files:
        meta = process_mat(mat_path, out_dir)
        all_meta.append(meta)
        print()

    # Write combined metadata
    summary = {
        "dataset":        "PLAID Natural Recordings",
        "source_dir":     str(mat_dir),
        "n_recordings":   len(all_meta),
        "sample_rate_hz": SOURCE_RATE,
        "mains_freq_hz":  MAINS_FREQ,
        "total_duration_s": sum(m["duration_s"] for m in all_meta),
        "note": (
            "Raw current waveforms in Amperes. "
            "Inference pipeline handles per-window resampling and normalization. "
            "Pass --sample-rate 15384 --mains-freq 60 to inference_pipeline.py."
        ),
        "recordings": all_meta,
    }

    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_h = summary["total_duration_s"] / 3600
    print(f"[DONE] {len(all_meta)} recordings, "
          f"{summary['total_duration_s']:.0f}s ({total_h:.1f}h) total")
    print(f"[DONE] Metadata saved to: {meta_path}")
    print()
    print("Next - run inference on all recordings:")
    print("  for f in data/natural/waveform_7000*.npy; do")
    print("    python inference/inference_pipeline.py \\")
    print("      --checkpoint model_registry/<checkpoint>.pt \\")
    print("      --input $f --sample-rate 15384 --mains-freq 60 \\")
    print("      --output inference_results/$(basename $f .npy)")
    print("  done")


if __name__ == "__main__":
    main()

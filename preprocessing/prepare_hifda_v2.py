#!/usr/bin/env python3
"""
prepare_hifda_v2.py - HIFDA full-length recordings -> cycles-time-base windows, voltage-aligned.

Input: Full_time_window_dataset/ from HIFDA on Zenodo, record 14886758
(CC BY 4.0, 2025-02 version; the 2024 versions lack Air_conditioner and
Washing_machine). Only that folder is needed; the 10 ms / 100 ms / 1 s
folders are the same signals cut shorter.
    Current/<Appliance>/<Name><k>Current.txt   one float per line
    Voltage/<Appliance>/<Name><k>Voltage.txt
    100 kHz, about 5.4 s each, 50 records per appliance, Spain 230 V / 50 Hz.

Why a v2: prepare_hifda.py uses the 163.84 ms split without voltage and cuts
raw30k windows (13 ms). This script cuts 8-cycle windows from the voltage
rising zero crossing, like every other cycles source.

Units are the acquisition board's sensor volts with a DC bias of about 1.6 V
(removed per window). They are not calibrated to amps, so mix with
--current-prior.

Channel skew: on every resistive load the recorded current lags the voltage
by about 37 deg (about 2 ms), where it should be in phase. That is an
acquisition offset between the channels, so the script measures it on the
Heater records (power factor 1) and moves the window start by that amount
(--channel-skew-deg overrides). Without it every HIFDA window would be
rotated against PLAID, WHITED and the rest.

Some current files contain NaN samples; windows touching them are dropped.

Groups: one physical unit per appliance, 50 records. Records are grouped in
blocks of --records-per-group (default 10): hifda:<Appliance>:b<k//10>. As
with KAIST, group split then controls recording leakage only.

Usage:
  python preprocessing/prepare_hifda_v2.py --hifda-dir ../datasets_raw/hifda_zenodo_14886758
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, load_registry, rms,  # noqa: E402
                             save_outputs, zc_windows)

SOURCE = "hifda_v2"
SOURCE_RATE = 100_000
MAINS_HZ = 50.0

# lower-case HIFDA folder -> (model class, status, note)
MAPPING = {
    # Added in the 2025 release. After the Heater skew correction its current
    # leads by ~80 deg (PLAID and WHITED ACs: in phase to 10 deg lagging), so
    # it was likely recorded with a different channel skew that cannot be
    # calibrated without a resistive load from the same session.
    "air_conditioner": ("Air Conditioner", "UNCERTAIN",  "phase reference unverifiable (2025 addition)"),
    "coffeemaker":     ("Coffee maker",    "SAFE",       ""),
    "hairdryer":       ("Hairdryer",       "SAFE",       ""),
    "heater":          ("Heater",          "SAFE",       ""),
    "laptop":          ("Laptop",          "SAFE",       ""),
    "microwave":       ("Microwave",       "SAFE",       ""),
    "vacuum":          ("Vacuum",          "SAFE",       ""),
    "washing_machine": ("Washing Machine", "UNCERTAIN",  "phase +99 deg after correction; as air_conditioner"),
    "griddle":         ("Heater",          "EQUIVALENT", "resistive"),
    "iron":            ("Heater",          "EQUIVALENT", "clothes iron, resistive"),
    "light":           ("Incandescent Light Bulb", "UNCERTAIN", "lamp type not stated"),
    "charger":         ("Laptop",          "UNCERTAIN",  "SMPS"),
    "computer":        ("Laptop",          "UNCERTAIN",  "SMPS"),
    "monitor":         ("Laptop",          "UNCERTAIN",  "SMPS"),
}
EXCLUDED = {"emptynetwork", "emptygrid"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--hifda-dir", default="../datasets_raw/hifda_zenodo_14886758")
    p.add_argument("--out-dir", default="data/hifda_v2")
    p.add_argument("--include-uncertain", action="store_true")
    p.add_argument("--max-windows-per-record", type=int, default=12,
                   help="Evenly spaced over the 5.4 s record (33 available). Default 12 gives "
                        "600 windows per appliance, so one unit does not swamp a class")
    p.add_argument("--records-per-group", type=int, default=10)
    p.add_argument("--min-on-fraction", type=float, default=0.5)
    p.add_argument("--channel-skew-deg", type=float, default=None,
                   help="Current-vs-voltage acquisition skew. Default: measured on the Heater records")
    return p.parse_args()


def load_signal(path: Path) -> np.ndarray:
    return np.fromfile(path, sep=" ", dtype=np.float64)


def measure_skew(root: Path, n_records: int = 10) -> float:
    """Median lag of current behind voltage at 50 Hz on the resistive Heater (should be 0)."""
    lags = []
    for cpath in sorted((root / "Current" / "Heater").glob("*Current.txt"))[:n_records]:
        i = load_signal(cpath)
        v = load_signal(root / "Voltage" / "Heater" / cpath.name.replace("Current.txt", "Voltage.txt"))
        n = 25 * SOURCE_RATE // int(MAINS_HZ)  # exactly 25 cycles: the fundamental falls on bin 25
        i, v = i[:n], v[:n]
        if not (np.isfinite(i).all() and np.isfinite(v).all()):
            continue
        ratio = np.fft.rfft(i - i.mean())[25] / np.fft.rfft(v - v.mean())[25]
        lags.append(-np.degrees(np.angle(ratio)))
    if not lags:
        sys.exit("[ERROR] could not measure the channel skew; pass --channel-skew-deg")
    return float(np.median(lags))


def main():
    args = parse_args()
    reg = load_registry()
    root = next(Path(args.hifda_dir).rglob("Full_time_window_dataset"), None)
    if root is None:
        sys.exit(f"[ERROR] no Full_time_window_dataset under {args.hifda_dir}")
    skew = measure_skew(root) if args.channel_skew_deg is None else args.channel_skew_deg
    print(f"[INFO] channel skew {skew:.1f} deg (current behind voltage)")

    X, y, g = [], [], []
    skipped = Counter()
    used = defaultdict(int)
    for folder in sorted((root / "Current").iterdir()):
        key = folder.name.lower()
        if key in EXCLUDED:
            continue
        if key not in MAPPING:
            skipped[f"unmapped:{folder.name}"] += 1
            continue
        cls, status, _ = MAPPING[key]
        if status == "UNCERTAIN" and not args.include_uncertain:
            skipped[f"uncertain:{folder.name}"] += 1
            continue
        for cpath in sorted(folder.glob("*Current.txt")):
            m = re.search(r"(\d+)Current\.txt$", cpath.name)
            vpath = root / "Voltage" / folder.name / cpath.name.replace("Current.txt", "Voltage.txt")
            if not m or not vpath.exists():
                skipped["missing_voltage"] += 1
                continue
            i, v = load_signal(cpath), load_signal(vpath)
            n = min(len(i), len(v))
            if n < SOURCE_RATE:
                skipped["short_record"] += 1
                continue
            if not np.isfinite(v[:n]).all():
                skipped["nan_in_voltage"] += 1
                continue
            w = zc_windows(i[:n], SOURCE_RATE, MAINS_HZ, voltage=v[:n], channel_skew_deg=skew)
            finite = np.isfinite(w).all(axis=1)
            skipped["nan_windows"] += int((~finite).sum())
            w = w[finite]
            if not len(w):
                skipped["no_zero_crossings"] += 1
                continue
            r = rms(w)
            keep = np.where(r >= args.min_on_fraction * float(np.median(r)))[0]
            if len(keep) > args.max_windows_per_record:
                keep = keep[np.linspace(0, len(keep) - 1, args.max_windows_per_record).round().astype(int)]
            X.append(w[keep])
            y += [class_id(reg, cls)] * len(keep)
            g += [f"hifda:{folder.name}:b{int(m.group(1)) // args.records_per_group}"] * len(keep)
            used[folder.name] += 1

    report = {
        "input": str(root),
        "timebase": "cycles",
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": MAINS_HZ,
        "retimed_to_50hz": False,
        "bandwidth_hz": bandwidth_hz("cycles", SOURCE_RATE, MAINS_HZ),
        "alignment": f"voltage rising zero crossing, moved {skew:.1f} deg later for the channel skew",
        "channel_skew_deg": round(skew, 2),
        "units": "sensor volts (uncalibrated); use --current-prior",
        "include_uncertain": args.include_uncertain,
        "records_used_per_appliance": dict(used),
        "skipped": dict(skipped),
        "caveats": [
            "One physical unit per appliance type.",
            "Spain 230 V / 50 Hz, the same supply as Ghana.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), np.array(g),
                 {k: {"model_class": c, "class_id": class_id(reg, c), "status": s, "note": note}
                  for k, (c, s, note) in MAPPING.items()}, report)
    print("     skipped:", dict(skipped))


if __name__ == "__main__":
    main()

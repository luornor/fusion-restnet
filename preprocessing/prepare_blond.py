#!/usr/bin/env python3
"""
prepare_blond.py - BLOND office appliances -> cycles-time-base windows in amps, unit groups.

Input: MEDAL files fetched by preprocessing/fetch_blond.py (BLOND, TUM,
Kriechbaumer & Jacobsen 2018, CC BY 4.0) plus appliance_log.json.
    BLOND-50/<day>/medal-<k>/*.hdf5    6.4 kHz, 15 min per file
    BLOND-250/<day>/medal-<k>/*.hdf5   50 kHz, 2 min per file
Each file: current1..current6 (one per socket) and voltage, int16, with
calibration_factor attributes (offset already removed). 230 V / 50 Hz.

Per file and socket
-------------------
1. Look up the appliance in appliance_log.json. A file is split where the
   MEDAL's socket layout changes inside it (the logged test sessions).
2. Scale to amps and volts, windows from the voltage rising zero crossing,
   shifted by --channel-skew-deg (signature_utils.zc_windows).
3. Steady state only: RMS >= --min-rms-a and within --steady-tol of both
   neighbouring windows. Appliances switch on and off inside a file, so
   inrush and switching windows are dropped this way.
4. Keep up to --max-windows-per-file, evenly spaced.

Groups are appliance models: blond:<normalised appliance name>. The log
cannot tell a device that moved sockets from a second device of the same
model, so this undercounts units rather than leak one across splits.

Channel skew: resistive loads read -99.8 deg (kettle, BLOND-50) and -99.0 deg
(space heater, BLOND-250) against about -90 for the other sources, with third
harmonic under 1%. The MEDAL current channel lags by about 9.4 deg on both
rates; the default compensates that. The report gives the corrected resistive
phase so a regression shows up.

Usage:
  python preprocessing/prepare_blond.py --blond-dir ../datasets_raw/blond_mediatum_1375836
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, load_registry, rms,  # noqa: E402
                             save_outputs, zc_windows)

SOURCE = "blond"
MAINS_HZ = 50.0

# BLOND class_name -> (model class, status, note)
MAPPING = {
    "Laptop":              ("Laptop",                       "SAFE",       ""),
    "Kettle":              ("Water kettle",                 "SAFE",       ""),
    "Fan":                 ("Fan",                          "SAFE",       "desk fans"),
    "Space Heater":        ("Heater",                       "SAFE",       ""),
    "Multi-Tool":          ("Universal-motor tool",         "SAFE",       "Mannesmann M92577 rotary multi-tool, 135 W"),
    "Screen Motor":        ("Single-phase induction motor", "EQUIVALENT", "projector screen tubular motor"),
    "Monitor":             ("Laptop",                       "UNCERTAIN",  "SMPS, as WHITED Monitor"),
    "PC":                  ("Laptop",                       "UNCERTAIN",  "SMPS with PFC"),
    "USB Charger":         ("Laptop",                       "UNCERTAIN",  "SMPS, 5-15 W: near the current resolution"),
    "Battery Charger":     ("Laptop",                       "UNCERTAIN",  "SMPS"),
    "Daylight":            ("Incandescent Light Bulb",      "UNCERTAIN",  "Philips wake-up light: dimmed halogen"),
    "Paper Shredder":      ("Universal-motor tool",         "UNCERTAIN",  "motor type not verified"),
}
# Not mapped: Printer (fuser plus SMPS), Projector (lamp ballast), Dev Board, Electric Toothbrush.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--blond-dir", default="../datasets_raw/blond_mediatum_1375836")
    p.add_argument("--out-dir", default="data/blond")
    p.add_argument("--include-uncertain", action="store_true")
    p.add_argument("--channel-skew-deg", type=float, default=9.4)
    p.add_argument("--min-rms-a", type=float, default=0.1,
                   help="Default 0.1 A (23 W): the MEDAL LSB is 5.4 mA on sockets 2-6, 15 mA on socket 1")
    p.add_argument("--steady-tol", type=float, default=0.1)
    p.add_argument("--max-windows-per-file", type=int, default=10)
    return p.parse_args()


def file_start(f: h5py.File) -> datetime:
    a = f.attrs
    return datetime(int(a["year"]), int(a["month"]), int(a["day"]),
                    int(a["hours"]), int(a["minutes"]), int(a["seconds"]))


def segments(log: dict, medal: str, start: datetime, n: int, fs: int) -> list[tuple[int, int, dict]]:
    """(first sample, last sample, layout) for each part of a file with one socket layout.
    Layouts sometimes change mid-file (the logged test sessions)."""
    entries = sorted(log[medal.upper()]["entries"], key=lambda e: e["timestamp"])
    times = [datetime.strptime(e["timestamp"], "%Y-%m-%dT%H-%M-%S") for e in entries]
    out = []
    for k, e in enumerate(entries):
        lo = int((times[k] - start).total_seconds() * fs)
        hi = int((times[k + 1] - start).total_seconds() * fs) if k + 1 < len(entries) else n
        lo, hi = max(lo, 0), min(hi, n)
        if hi > lo:
            out.append((lo, hi, e))
    return out


def unit_key(name: str) -> str:
    """Group id from the appliance name. Devices move between sockets (the
    multi-tool's two test sessions are on MEDAL-1 and MEDAL-2) and names vary
    in spelling, so one model name is one group: conservative, never leaks."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def scaled(d: h5py.Dataset) -> np.ndarray:
    x = d[:].astype(np.float64) * float(d.attrs["calibration_factor"])
    return x - x.mean()


def main():
    args = parse_args()
    reg = load_registry()
    root = Path(args.blond_dir)
    log = json.loads((root / "appliance_log.json").read_text())
    files = sorted(root.glob("BLOND-*/*/medal-*/*.hdf5"))

    X, y, g = [], [], []
    skipped = Counter()
    used = Counter()
    resistive_phase = []
    rates = Counter()
    for path in files:
        medal = path.parent.name
        try:
            f = h5py.File(path, "r")
        except OSError:
            skipped["unreadable_file"] += 1
            continue
        with f:
            fs = int(f.attrs["frequency"])
            rates[fs] += 1
            v_all = scaled(f["voltage"])
            parts = segments(log, medal, file_start(f), len(v_all), fs)
            if not parts:
                skipped["no_layout"] += 1
                continue
            currents = {}
            for (lo, hi, layout), k in ((p, k) for p in parts for k in range(1, 7)):
                socket = f"socket_{k}"
                app = layout.get(socket) or {}
                name, typ = app.get("appliance_name"), app.get("class_name")
                if not name:
                    continue
                if typ not in MAPPING:
                    skipped[f"unmapped:{typ}"] += 1
                    continue
                cls, status, _ = MAPPING[typ]
                if status == "UNCERTAIN" and not args.include_uncertain:
                    skipped[f"uncertain:{typ}"] += 1
                    continue
                if k not in currents:
                    currents[k] = scaled(f[f"current{k}"])
                i, v = currents[k][lo:hi], v_all[lo:hi]
                w = zc_windows(i, fs, MAINS_HZ, voltage=v, channel_skew_deg=args.channel_skew_deg)
                if len(w) < 3:
                    skipped["too_short"] += 1
                    continue
                r = rms(w)
                steady = np.zeros(len(r), dtype=bool)
                steady[1:-1] = ((r[1:-1] >= args.min_rms_a)
                                & (np.abs(r[1:-1] - r[:-2]) <= args.steady_tol * r[1:-1])
                                & (np.abs(r[1:-1] - r[2:]) <= args.steady_tol * r[1:-1]))
                keep = np.where(steady)[0]
                if not len(keep):
                    skipped["off_or_switching"] += 1
                    continue
                if len(keep) > args.max_windows_per_file:
                    keep = keep[np.linspace(0, len(keep) - 1, args.max_windows_per_file).round().astype(int)]
                X.append(w[keep])
                y += [class_id(reg, cls)] * len(keep)
                g += [f"blond:{unit_key(name)}"] * len(keep)
                used[typ] += 1
                if cls in ("Water kettle", "Heater"):
                    F = np.fft.rfft(w[keep], axis=1)
                    resistive_phase.append(float(np.degrees(np.angle(np.median(F[:, 8])))))

    units = defaultdict(set)
    for gg, cid in zip(g, y):
        units[reg["classes"][cid]["name"]].add(gg)
    report = {
        "input": str(root),
        "timebase": "cycles",
        "source_rate_hz": "6400 (BLOND-50), 50000 (BLOND-250)",
        "files_per_rate": {str(k): n for k, n in rates.items()},
        "source_mains_hz": MAINS_HZ,
        "retimed_to_50hz": False,
        "bandwidth_hz": bandwidth_hz("cycles", 6400, MAINS_HZ),
        "alignment": "voltage rising zero crossing, current channel skew compensated",
        "channel_skew_deg": args.channel_skew_deg,
        "resistive_phase_deg_after_correction": [round(p, 1) for p in resistive_phase],
        "units": "A, via the per-dataset calibration_factor",
        "include_uncertain": args.include_uncertain,
        "socket_files_used_per_type": dict(used),
        "models_per_class": {k: sorted(v) for k, v in units.items()},
        "skipped": dict(skipped),
        "caveats": [
            "Office building in Germany, 230 V / 50 Hz, CC BY 4.0.",
            "Sampled download (fetch_blond.py): each socket-appliance instance on up to 3 weekdays.",
            "Groups are appliance models, so several identical laptops count as one unit (no split leakage).",
            "The Clatronic kettle on MEDAL-13 was not caught in the sampled office-hour files; "
            "Water kettle and Heater have one unit each, from the logged test sessions.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), np.array(g),
                 {t: {"model_class": c, "class_id": class_id(reg, c), "status": s, "note": note}
                  for t, (c, s, note) in MAPPING.items()}, report)
    print("     skipped:", dict(skipped))
    print("     resistive phase after correction:", report["resistive_phase_deg_after_correction"])


if __name__ == "__main__":
    main()

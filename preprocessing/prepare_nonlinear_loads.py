#!/usr/bin/env python3
"""
prepare_nonlinear_loads.py - "Dataset of non-linear household loads" -> training windows.

Input: the .xlsx files of Mendeley dataset nkrhgfp9xy (CC BY 4.0; Santo
Domingo, Dominican Republic; Metrel MI 2883 analyser). Each workbook's
"Voltage & Current Table" sheet holds a ~0.2 s snapshot (~12 cycles) of
voltage and current at ~7.66 kHz on a 60 Hz grid.

This adds the only public waveforms found for an inverter air conditioner and
a UPS. Each is ONE snapshot of ONE unit, so it seeds the class but cannot train
it on its own. Treat these as placeholders until field data arrives.

Per workbook
------------
1. Read I1 (line current, A) and the timestamps; derive the sampling rate
   from the timestamps rather than trusting the nominal 7,680 Hz.
2. Retime 60 -> 50 Hz and resample to 30 kHz (see signature_utils), unless
   --no-retime.
3. Cut 400-sample windows with --hop overlap. The snapshot is short, so
   overlapping windows are the only way to get more than a handful.

Usage:
  python preprocessing/prepare_nonlinear_loads.py --xlsx-dir ../datasets_raw/nonlinear_loads_mendeley_nkrhgfp9xy
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, estimate_mains_hz,  # noqa: E402
                             load_registry, resample, rms, save_outputs, timebase_rate, windows,
                             zc_windows)

SOURCE = "nonlinear_loads"
SOURCE_MAINS = 60.0

# workbook stem -> (registry class, status). TV and phone charger have no class.
MAPPING = {
    "AC_Inverter":           ("Inverter air conditioner", "SAFE"),
    "UPS":                   ("UPS",                      "SAFE"),
    "Fridge":                ("Fridge",                   "SAFE"),
    "Microwave":             ("Microwave",                "SAFE"),
    "Laptop_Charger":        ("Laptop",                   "SAFE"),
    "TV_SONY":               (None,                       "NO_CLASS"),
    "Phone_Charger_Samsung": (None,                       "NO_CLASS"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--xlsx-dir", default="../datasets_raw/nonlinear_loads_mendeley_nkrhgfp9xy")
    p.add_argument("--out-dir", default="data/nonlinear_loads")
    p.add_argument("--timebase", choices=["cycles", "raw30k"], default="cycles",
                   help="cycles (default): 8 mains cycles x 50 samples, the model's design. "
                        "raw30k: legacy 400 raw samples at 30 kHz. See signature_utils.py")
    p.add_argument("--hop", type=int, default=50,
                   help="Window hop in samples (default 50: one cycle in the cycles time base)")
    p.add_argument("--no-retime", action="store_true")
    return p.parse_args()


def parse_time(s: str) -> float:
    date, clock = s.strip().split(" ")
    hms, frac = clock.split(".")
    base = datetime.strptime(f"{date} {hms}", "%m/%d/%Y %H:%M:%S").timestamp()
    return base + float("0." + frac)


def read_sheet(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Voltage & Current Table"]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    ti, vi = head.index("Time [UTC]"), head.index("U1 [V]")
    # Low-current loads (laptop and phone chargers) are logged in mA.
    if "I1 [A]" in head:
        ii, to_amps = head.index("I1 [A]"), 1.0
    else:
        ii, to_amps = head.index("I1 [mA]"), 1e-3
    data = [r for r in rows[1:] if r[ii] is not None and r[ti] is not None]
    t = np.array([parse_time(str(r[ti])) for r in data])
    i = np.array([float(r[ii]) * to_amps for r in data])
    v = np.array([float(r[vi]) for r in data])
    rate = 1.0 / float(np.median(np.diff(t)))
    return i, v, rate


def main():
    args = parse_args()
    reg = load_registry()
    d = Path(args.xlsx_dir)
    dst = timebase_rate(args.timebase, SOURCE_MAINS, retime_to_50hz=not args.no_retime)

    X, y, g, rates, mains, used = [], [], [], [], [], {}
    for stem, (name, status) in MAPPING.items():
        path = d / f"{stem}.xlsx"
        if name is None or not path.exists():
            continue
        i, v, rate = read_sheet(path)
        rates.append(rate)
        mains.append(estimate_mains_hz(v, rate))
        if args.timebase == "cycles":
            # ~12 cycles per snapshot: step one cycle so it yields ~4 windows, not 1.
            w = zc_windows(i, rate, SOURCE_MAINS, voltage=v, step_cycles=1)
        else:
            w = windows(resample(i - i.mean(), rate, dst), hop=args.hop)
        w = w[rms(w) > 0.02]
        X.append(w)
        y += [class_id(reg, name)] * len(w)
        g += [f"nonlinear_loads:{stem}"] * len(w)
        used[stem] = {"samples": int(len(i)), "rate_hz": round(rate, 1),
                      "seconds": round(len(i) / rate, 3), "windows": int(len(w))}
        print(f"     {stem:<22} {len(i):>5} samples @ {rate:7.1f} Hz -> {len(w)} windows ({name})")

    class_map = {stem: {"model_class": name, "class_id": class_id(reg, name) if name else None,
                        "status": status, "included": name is not None}
                 for stem, (name, status) in MAPPING.items()}
    report = {
        "input": str(d),
        "source_rate_hz": round(float(np.median(rates)), 1),
        "source_mains_hz": round(float(np.median(mains)), 2),
        "timebase": args.timebase,
        "retimed_to_50hz": args.timebase == "cycles" or not args.no_retime,
        "bandwidth_hz": bandwidth_hz(args.timebase, float(np.median(rates)), SOURCE_MAINS),
        "window_hop": args.hop,
        "workbooks": used,
        "caveats": [
            "Each class here is ONE ~0.2 s snapshot of ONE unit. Overlapping windows are near-duplicates.",
            "Inverter air conditioner and UPS are seeds only. Do not report accuracy on them until field data exists.",
            "~3.8 kHz source bandwidth: irrelevant in the cycles time base; with raw30k, train with --aug-bandlimit-prob.",
            "Recorded on a 120/240 V 60 Hz grid; retimed to 50 Hz unless --no-retime.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), g, class_map, report)


if __name__ == "__main__":
    main()

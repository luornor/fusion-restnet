#!/usr/bin/env python3
"""
prepare_kaist_hvac.py - KAIST HVAC testbed fan motor -> "Three-phase induction motor" windows.

Input: Current_D0..D4.tdms extracted from HVAC.zip, Mendeley dataset
5tt3zb2pns (CC BY 4.0; Jung et al., Data in Brief 2025).

Source: one three-phase induction motor driving an HVAC fan through a belt,
direct-on-line on 60 Hz mains (verified: fundamental 60.00 Hz, harmonics about
45 dB down, no drive switching content). Three phase currents in amps at
100 kHz, 300 s per file. Files D0-D4 are belt looseness levels; electrically
they are all the same running motor, so they share one class.

Per file and phase
------------------
Take --windows-per-phase positions evenly spread over the 300 s recording.
At each, read a short stretch of source samples, resample it to the time base
(see signature_utils.py), and keep the middle 400 samples so resampling filter
edges never reach the window. One window per position keeps windows far apart
in time instead of back to back.

In the cycles time base, 60 Hz becomes 50 samples per cycle automatically. In
raw30k the 60 Hz cycle is retimed to 50 Hz unless --no-retime.

Group ids are (belt level, 60 s block): kaist_hvac:D<k>:b<j>. Everything comes
from one motor, so group split controls recording leakage only.

Usage:
  python preprocessing/prepare_kaist_hvac.py --tdms-dir ../datasets_raw/kaist_hvac_belt_mendeley_5tt3zb2pns/extracted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from nptdms import TdmsFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (WINDOW_SIZE, bandwidth_hz, class_id, estimate_mains_hz,  # noqa: E402
                             load_registry, resample, rms, save_outputs, timebase_rate,
                             zc_windows)

SOURCE = "kaist_hvac"
SOURCE_RATE = 100_000
SOURCE_MAINS = 60.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--tdms-dir", default="../datasets_raw/kaist_hvac_belt_mendeley_5tt3zb2pns/extracted")
    p.add_argument("--out-dir", default="data/kaist_hvac")
    p.add_argument("--timebase", choices=["cycles", "raw30k"], default="cycles",
                   help="cycles (default): 8 mains cycles x 50 samples, the model's design. "
                        "raw30k: legacy 400 raw samples at 30 kHz. See signature_utils.py")
    p.add_argument("--windows-per-phase", type=int, default=180,
                   help="Evenly spaced windows per file and phase. Default 180 gives 2,700 "
                        "windows, about the size of LILAC's three motors, so one machine does "
                        "not dominate the class")
    p.add_argument("--block-s", type=float, default=60.0, help="Group block length (s)")
    p.add_argument("--no-retime", action="store_true", help="raw30k only: keep 60 Hz timing")
    p.add_argument("--assumed-lag-deg", type=float, default=37.0,
                   help="cycles only. No voltage was recorded, so windows are aligned on the "
                        "current, shifted by this lag (37 deg = power factor 0.8, typical for a "
                        "loaded induction motor). An assumption, recorded in the report")
    return p.parse_args()


def main():
    args = parse_args()
    reg = load_registry()
    cid = class_id(reg, "Three-phase induction motor")
    files = sorted(Path(args.tdms_dir).glob("Current_D*.tdms"))
    if not files:
        sys.exit(f"[ERROR] no Current_D*.tdms in {args.tdms_dir}")
    dst = timebase_rate(args.timebase, SOURCE_MAINS, retime_to_50hz=not args.no_retime)
    # Source samples needed for three windows: the middle one is kept.
    span = int(np.ceil(3 * WINDOW_SIZE * SOURCE_RATE / dst))
    if args.timebase == "cycles":
        span = int(np.ceil(12 * SOURCE_RATE / SOURCE_MAINS))  # 12 cycles: room for one 8-cycle window

    X, y, g, mains = [], [], [], []
    for f in files:
        level = f.stem.split("_")[-1]
        with TdmsFile.open(f) as td:
            chans = list(td["Log"].channels())
            n = len(chans[0])
            starts = np.linspace(SOURCE_RATE, n - span - SOURCE_RATE, args.windows_per_phase).astype(int)
            for c in chans:
                if len(mains) < 3:
                    probe = c[SOURCE_RATE:SOURCE_RATE * 3].astype(np.float64)
                    mains.append(estimate_mains_hz(probe - probe.mean(), SOURCE_RATE))
                for s in starts:
                    x = c[int(s):int(s) + span].astype(np.float64)
                    if args.timebase == "cycles":
                        ws = zc_windows(x, SOURCE_RATE, SOURCE_MAINS, voltage=None,
                                        assumed_lag_deg=args.assumed_lag_deg, max_windows=1)
                        w = ws[0] if len(ws) else np.empty(0)
                    else:
                        r = resample(x - x.mean(), SOURCE_RATE, dst)
                        mid = (len(r) - WINDOW_SIZE) // 2
                        w = r[mid:mid + WINDOW_SIZE]
                    if len(w) == WINDOW_SIZE and rms(w) > 0.05:
                        X.append(w)
                        y.append(cid)
                        g.append(f"kaist_hvac:{level}:b{int(s / SOURCE_RATE // args.block_s)}")

    report = {
        "input": str(args.tdms_dir),
        "files": [f.name for f in files],
        "timebase": args.timebase,
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": round(float(np.median(mains)), 2),
        "retimed_to_50hz": args.timebase == "cycles" or not args.no_retime,
        "bandwidth_hz": bandwidth_hz(args.timebase, SOURCE_RATE, SOURCE_MAINS),
        "alignment": (f"current zero crossing shifted by an ASSUMED {args.assumed_lag_deg} deg lag "
                      "(no voltage recorded)") if args.timebase == "cycles" else "none",
        "caveats": [
            "One physical motor (HVAC fan via belt). Adds diversity to the Three-phase induction motor class but is not a fleet.",
            "Direct-on-line 60 Hz, 380 V class motor.",
            "Belt levels D0-D4 are kept as groups, not classes. They are useful later for motor-health work (see docs/MCSA_PLAN.md).",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.stack(X), np.array(y), g,
                 {"HVAC fan motor (D0-D4)": {"model_class": "Three-phase induction motor",
                                             "class_id": cid, "status": "SAFE"}}, report)
    print(f"     mains estimate {report['source_mains_hz']} Hz, time base {args.timebase}")


if __name__ == "__main__":
    main()

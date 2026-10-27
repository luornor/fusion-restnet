#!/usr/bin/env python3
"""
prepare_lilac.py - LILAC (TUM, lab-measured industrial appliances) -> training windows.

Input: the preprocessed LILAC arrays published with AWRGNILM
(github.com/sambaiga/AWRGNILM, data/lilac/{submetered,aggregated}/). The
official 35 GiB LILAC download link is dead (404) as of 2026-10-27.

What those arrays are
---------------------
Each record is ONE mains cycle (1000 samples at 50 kHz, 50 Hz, ~232 V, in amps)
on all three phases, computed as current-after-event minus current-before-event
around a switching event. The subtraction isolates the appliance that switched,
so even the "aggregated" records (1-3 appliances running) are single-appliance
signatures. Records alternate on/off events of the same appliance.

Per record
----------
1. Phases: three-phase loads contribute all three phases (each is what a
   single-phase clamp on that phase sees). Single-phase loads contribute only
   the phase carrying the current.
2. FFT-resample the exact cycle 1000 -> 600 samples (50 Hz at 30 kHz).
3. Emit --rotations windows of 400 samples from the cycle tiled twice, at evenly
   spaced start offsets. Device windows start at arbitrary mains phase; the
   source cycles all start at a voltage zero crossing.

Group ids pair each appliance's k-th on and off events: lilac:<subset>:<label>:<k//2>.
LILAC has one physical unit per appliance type, so group split controls
recording leakage, not unit-to-unit generalisation.

Usage:
  python preprocessing/prepare_lilac.py --lilac-dir ../datasets_raw/lilac_awrgnilm_preprocessed
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, cycle_windows, load_registry,  # noqa: E402
                             rms, save_outputs)

SOURCE = "lilac"
SOURCE_RATE = 50_000

# LILAC label -> (registry class name, mapping status, three_phase)
# status: SAFE = same appliance type; EQUIVALENT = electrically the same kind of
# load (included by default); UNCERTAIN = excluded unless --include-uncertain.
MAPPING = {
    "Bulb":                    ("Incandescent Light Bulb",      "SAFE",       False),
    "Coffee-machine":          ("Coffee maker",                 "SAFE",       False),
    "Hair-dryer":              ("Hairdryer",                    "SAFE",       False),
    "Kettle":                  ("Water kettle",                 "SAFE",       False),
    "Refrigerator":            ("Fridge",                       "SAFE",       False),
    "Vacuum-cleaner":          ("Vacuum",                       "SAFE",       False),
    "1-phase-async-motor":     ("Single-phase induction motor", "SAFE",       False),
    "3-phase-async-motor":     ("Three-phase induction motor",  "SAFE",       True),
    "Squirrel-3-async":        ("Three-phase induction motor",  "SAFE",       True),
    "Squirrel-3-async-2x":     ("Three-phase induction motor",  "SAFE",       True),
    "Freq-conv-squirrel-3-2x": ("VSD-driven motor",             "SAFE",       True),
    "Fluorescent-lamp":        ("Fluorescent tube lighting",    "SAFE",       False),
    "Drilling-machine":        ("Universal-motor tool",         "SAFE",       False),
    "Resistor":                ("Heater",                       "EQUIVALENT", True),
    "Raclette":                ("Heater",                       "EQUIVALENT", False),
    "Dumper-machine":          ("Three-phase induction motor",  "UNCERTAIN",  True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--lilac-dir", default="../datasets_raw/lilac_awrgnilm_preprocessed")
    p.add_argument("--out-dir", default="data/lilac")
    p.add_argument("--subsets", nargs="+", default=["submetered", "aggregated"])
    p.add_argument("--timebase", choices=["cycles", "raw30k"], default="cycles",
                   help="cycles (default): 8 mains cycles x 50 samples, the model's design. "
                        "raw30k: legacy 400 raw samples at 30 kHz. See signature_utils.py")
    p.add_argument("--rotations", type=int, default=None,
                   help="Windows per cycle at evenly spaced start phases. Default: 1 for cycles "
                        "(keeps the voltage zero-crossing alignment the model expects), 3 for raw30k")
    p.add_argument("--min-rms-a", type=float, default=0.05,
                   help="Drop phases whose event current is below this RMS (default 0.05 A)")
    p.add_argument("--include-uncertain", action="store_true",
                   help="Also map Dumper-machine (unidentified 3-phase machine)")
    p.add_argument("--exclude-equivalent", action="store_true",
                   help="Do not add Resistor/Raclette to the Heater class")
    return p.parse_args()


def main():
    args = parse_args()
    reg = load_registry()
    root = Path(args.lilac_dir)
    if args.rotations is None:
        args.rotations = 1 if args.timebase == "cycles" else 3

    X, y, g = [], [], []
    skipped = defaultdict(int)
    for subset in args.subsets:
        d = root / subset
        I = np.load(d / "current.npy", allow_pickle=False)
        L = np.load(d / "labels.npy", allow_pickle=False)
        occurrence = defaultdict(int)
        for i in range(len(L)):
            label = str(L[i])
            k = occurrence[label]
            occurrence[label] += 1
            if label not in MAPPING:
                skipped[f"unmapped:{label}"] += 1
                continue
            name, status, three_phase = MAPPING[label]
            if status == "UNCERTAIN" and not args.include_uncertain:
                skipped[f"uncertain:{label}"] += 1
                continue
            if status == "EQUIVALENT" and args.exclude_equivalent:
                skipped[f"equivalent:{label}"] += 1
                continue
            phase_rms = rms(I[i], axis=1)
            phases = [p for p in range(3) if phase_rms[p] >= args.min_rms_a] if three_phase \
                else [int(np.argmax(phase_rms))]
            phases = [p for p in phases if phase_rms[p] >= args.min_rms_a]
            if not phases:
                skipped[f"below_min_rms:{label}"] += 1
                continue
            cid = class_id(reg, name)
            group = f"lilac:{subset}:{label}:{k // 2}"
            for p in phases:
                for w in cycle_windows(I[i, p], args.timebase, args.rotations):
                    X.append(w)
                    y.append(cid)
                    g.append(group)

    class_map = {label: {"model_class": name, "class_id": class_id(reg, name), "status": status,
                         "three_phase": tp,
                         "included": not ((status == "UNCERTAIN" and not args.include_uncertain)
                                          or (status == "EQUIVALENT" and args.exclude_equivalent))}
                 for label, (name, status, tp) in MAPPING.items()}
    report = {
        "input": str(root),
        "subsets": args.subsets,
        "source_rate_hz": SOURCE_RATE,
        "timebase": args.timebase,
        "source_mains_hz": 50,
        "retimed_to_50hz": False,
        "bandwidth_hz": bandwidth_hz(args.timebase, SOURCE_RATE, 50),
        "rotations_per_cycle": args.rotations,
        "skipped": dict(skipped),
        "caveats": [
            "One physical unit per LILAC appliance type: new classes seeded here will not generalise to other makes until field data is added.",
            "Windows from one record are rotations of the same cycle, not independent observations. In the cycles time base each window is that cycle repeated 8 times.",
            "Licence: none stated by LILAC. Do not ship a model trained on this without the authors' permission.",
            "Resistor and Raclette are mapped to Heater as electrically equivalent resistive loads.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.stack(X), np.array(y), g, class_map, report)
    if skipped:
        print("     skipped:", dict(skipped))


if __name__ == "__main__":
    main()

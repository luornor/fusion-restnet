#!/usr/bin/env python3
"""
prepare_caffeine.py - CAFFEINE espresso machine dataset -> "Espresso machine" windows.

Input: CAFFEINE_Dataset.zip from Zenodo record 8105706 (CC BY 4.0). Read in
place; nothing is extracted.

Source: one espresso machine, 45 trials (130 coffees), mains current I1 in amps
at 6,250 Hz, with per-sample activation intervals for each actuator (heater,
pump, infuser motor, grinder, electronics) in each trial's parquet metadata.

Per trial
---------
1. Read I1 (and V1 to confirm the mains frequency) from all parquet parts.
2. Take every "clustering" interval where at least one actuator besides the
   electronics is on, trimmed by --edge-s at both ends to skip switching
   transients.
3. Resample 6,250 Hz -> 30 kHz and cut non-overlapping 400-sample windows.
4. Cap windows per (trial, operating state) so long heater-only stretches do
   not swamp short grinder and pump phases.

Every window is labelled "Espresso machine": an aggregate meter sees the whole
machine, not its parts. The operating state of each window (e.g.
"heater+pump") is saved alongside as states_caffeine.npy for future
component-level classes.

Bandwidth caveat: 6,250 Hz sampling carries nothing above 3,125 Hz. Upsampled
windows have an empty high band, which no real 30 kHz capture has. Train with
--aug-bandlimit-prob so the model cannot use "no high-frequency content" as a
shortcut to this class.

Usage:
  python preprocessing/prepare_caffeine.py --zip ../datasets_raw/caffeine_zenodo_8105706/CAFFEINE_Dataset.zip
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, estimate_mains_hz, load_registry,  # noqa: E402
                             resample, rms, save_outputs, timebase_rate, windows, zc_windows)

SOURCE = "caffeine"
SOURCE_RATE = 6_250


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--zip", default="../datasets_raw/caffeine_zenodo_8105706/CAFFEINE_Dataset.zip")
    p.add_argument("--out-dir", default="data/caffeine")
    p.add_argument("--timebase", choices=["cycles", "raw30k"], default="cycles",
                   help="cycles (default): 8 mains cycles x 50 samples, the model's design. "
                        "raw30k: legacy 400 raw samples at 30 kHz. See signature_utils.py")
    p.add_argument("--edge-s", type=float, default=0.1, help="Trim each interval end by this (s)")
    p.add_argument("--min-interval-s", type=float, default=0.2)
    p.add_argument("--max-windows-per-state", type=int, default=25,
                   help="Cap per (trial, operating state), spread evenly over the state's time")
    p.add_argument("--min-rms-a", type=float, default=0.1)
    return p.parse_args()


def trial_parts(z: zipfile.ZipFile) -> dict[int, list[str]]:
    parts = defaultdict(list)
    for n in z.namelist():
        m = re.search(r"/dataset/data/trial(\d+)/part\.(\d+)\.parquet$", n)
        if m:
            parts[int(m.group(1))].append((int(m.group(2)), n))
    return {t: [n for _, n in sorted(v)] for t, v in sorted(parts.items())}


def main():
    args = parse_args()
    reg = load_registry()
    cid = class_id(reg, "Espresso machine")
    z = zipfile.ZipFile(args.zip)
    trials = trial_parts(z)
    edge = int(args.edge_s * SOURCE_RATE)
    dst_rate = timebase_rate(args.timebase, 50.0)
    min_len = int(args.min_interval_s * SOURCE_RATE)

    X, y, g, states = [], [], [], []
    mains_seen, per_state = [], defaultdict(int)
    for t, parts in trials.items():
        meta_name = parts[0].rsplit("/", 1)[0] + "/_common_metadata"
        meta = json.loads(pq.read_schema(io.BytesIO(z.read(meta_name))).metadata[b"meta"])
        assert float(meta["fs"]) == SOURCE_RATE, (t, meta["fs"])
        cols = [pq.read_table(io.BytesIO(z.read(n)), columns=["I1", "V1"]) for n in parts]
        current = np.concatenate([c.column("I1").to_numpy() for c in cols]).astype(np.float64)
        voltage = np.concatenate([c.column("V1").to_numpy() for c in cols]).astype(np.float64)
        mains_seen.append(estimate_mains_hz(voltage[: SOURCE_RATE * 4], SOURCE_RATE))

        for state, intervals in meta["labels"]["clustering"].items():
            if state == "electronics":
                continue  # standby: no actuator running
            segs = []
            for a, b in intervals:
                a, b = int(a) + edge, int(b) - edge
                if b - a < min_len:
                    continue
                if args.timebase == "cycles":
                    w = zc_windows(current[a:b], SOURCE_RATE, 50.0, voltage=voltage[a:b])
                else:
                    w = windows(resample(current[a:b] - np.mean(current[a:b]), SOURCE_RATE, dst_rate))
                w = w[rms(w) >= args.min_rms_a]
                if len(w):
                    segs.append(w)
            if not segs:
                continue
            w = np.concatenate(segs)
            if len(w) > args.max_windows_per_state:
                w = w[np.linspace(0, len(w) - 1, args.max_windows_per_state).round().astype(int)]
            X.append(w)
            y += [cid] * len(w)
            g += [f"caffeine:trial{t}"] * len(w)
            states += [state.replace("+electronics", "")] * len(w)
            per_state[state.replace("+electronics", "")] += len(w)

    report = {
        "input": str(args.zip),
        "trials": len(trials),
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": round(float(np.median(mains_seen)), 2),
        "timebase": args.timebase,
        "retimed_to_50hz": False,
        "bandwidth_hz": bandwidth_hz(args.timebase, SOURCE_RATE, 50),
        "windows_per_operating_state": dict(sorted(per_state.items())),
        "caveats": [
            "One physical machine: the Espresso machine class will not generalise to other makes until field data is added.",
            "6.25 kHz source: nothing above 3.125 kHz. Irrelevant in the cycles time base (1.25 kHz cap); with raw30k, train with --aug-bandlimit-prob.",
            "Labels are whole-machine. Component states are saved in states_caffeine.npy.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), g,
                 {"espresso machine (all actuator states)": {"model_class": "Espresso machine",
                                                             "class_id": cid, "status": "SAFE"}},
                 report, extra={"states": np.array(states)})
    print("     per operating state:", dict(sorted(per_state.items())))
    print(f"     mains frequency from V1: {report['source_mains_hz']} Hz")


if __name__ == "__main__":
    main()

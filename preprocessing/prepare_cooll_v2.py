#!/usr/bin/env python3
"""
prepare_cooll_v2.py - COOLL -> cycles-time-base windows in amps, voltage-aligned, unit groups.

Input: the COOLL release (Univ. Orleans, CC BY-NC 4.0: NOT for commercial use),
read in place from its zips:
    Current_<a>to<b>.zip, Voltage_<a>to<b>.zip   scenarioC1_<id>.flac / scenarioV1_<id>.flac
                                                (password-protected, password in the release)
    scaleFactors.txt                  line <id>: current factor, voltage factor -> A, V
    appliances_and_action_delays.txt  <id>: <Type>_<unit>_<delay>ms
    100 kHz, 6 s per record: 1 s off, 4 s on, 1 s off. 42 units x 20 records.

Why a v2: prepare_cooll.py keeps 3 of 12 types, ignores the voltage and scale
factors, and cuts raw30k windows (13 ms). COOLL's value here is 24 power-tool
units (drills, saws, sanders, grinders, hedge trimmers, planer, router), the
widest sample of universal motors in any public set.

Per record: cycle windows from the voltage rising zero crossing; keep windows
at or above --min-on-fraction of the on-level (90th percentile RMS), skip
--skip-after-on-s of motor run-up, keep up to --max-windows-per-record.

Groups are physical units: cooll:<Type>_<unit>.

Usage:
  python preprocessing/prepare_cooll_v2.py --cooll-dir ../datasets_raw/cooll_gdrive
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (CYCLES_PER_WINDOW, bandwidth_hz, class_id,  # noqa: E402
                             load_registry, rms, save_outputs, zc_windows)

SOURCE = "cooll_v2"
SOURCE_RATE = 100_000
MAINS_HZ = 50.0
ZIP_PASSWORD = b"prismecooll"

# COOLL type -> (model class, status, note); unit details from Headers.zip
MAPPING = {
    "Drill":          ("Universal-motor tool",    "SAFE",       "6 drills, 500-850 W"),
    "Saw":            ("Universal-motor tool",    "SAFE",       "8 saws: jigsaws, sabre, circular"),
    "Sander":         ("Universal-motor tool",    "SAFE",       "3 orbital sanders"),
    "Grinder":        ("Universal-motor tool",    "SAFE",       "2 angle grinders"),
    "Hedge_trimmer":  ("Universal-motor tool",    "SAFE",       "3 hedge trimmers"),
    "Planer":         ("Universal-motor tool",    "SAFE",       ""),
    "Router":         ("Universal-motor tool",    "SAFE",       ""),
    "Fan":            ("Fan",                     "SAFE",       "table fan, tower fan"),
    "Hair_drayer":    ("Hairdryer",               "SAFE",       "4 units (sic: 'drayer' in the release)"),
    "Vacuum_cleaner": ("Vacuum",                  "SAFE",       "7 units"),
    "Paint_stripper": ("Hairdryer",               "EQUIVALENT", "heat gun: heater plus fan"),
    "Lamp_2":         ("Compact Fluorescent Lamp", "SAFE",      "15 W CFL"),
    "Lamp_3":         ("Incandescent Light Bulb", "EQUIVALENT", "105 W halogen"),
    "Lamp_4":         ("Incandescent Light Bulb", "EQUIVALENT", "100 W halogen"),
    "Lamp_1":         ("Laptop",                  "UNCERTAIN",  "1.6 W LED lamp: no LED class"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cooll-dir", default="../datasets_raw/cooll_gdrive")
    p.add_argument("--out-dir", default="data/cooll_v2")
    p.add_argument("--include-uncertain", action="store_true")
    p.add_argument("--min-on-fraction", type=float, default=0.5)
    p.add_argument("--skip-after-on-s", type=float, default=1.0,
                   help="Seconds after switch-on to skip: universal motors take up to ~1 s to reach speed")
    p.add_argument("--max-windows-per-record", type=int, default=8,
                   help="Default 8 gives 160 windows per unit")
    return p.parse_args()


def read_flac(z: zipfile.ZipFile, name: str) -> np.ndarray:
    """Decode with PyAV (bundled ffmpeg): libsndfile loses sync on 100 kHz FLAC."""
    with av.open(io.BytesIO(z.read(name, pwd=ZIP_PASSWORD))) as f:
        stream = f.streams.audio[0]
        if stream.rate != SOURCE_RATE:
            raise ValueError(f"{name}: rate {stream.rate}")
        frames = [fr.to_ndarray() for fr in f.decode(stream)]
    data = np.concatenate(frames, axis=-1)
    data = data[0] if data.ndim == 2 else data
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max + 1)  # back to the [-1, 1] range scaleFactors expects
    return data.astype(np.float64)


def main():
    args = parse_args()
    reg = load_registry()
    root = Path(args.cooll_dir)
    factors = np.loadtxt(root / "scaleFactors.txt")
    labels = {}
    for line in (root / "appliances_and_action_delays.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+):\s*(.+?)_(\d+)_(\d+)ms", line)
        if m:
            labels[int(m.group(1))] = (m.group(2), int(m.group(3)))
    members = {}
    for zp in sorted(root.glob("*_*to*.zip")):
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            m = re.search(r"scenario([CV])1_(\d+)\.flac$", n)
            if m:
                members[(m.group(1), int(m.group(2)))] = (z, n)

    win_s = CYCLES_PER_WINDOW / MAINS_HZ
    X, y, g = [], [], []
    skipped = Counter()
    units = defaultdict(set)
    for rid in sorted(labels):
        typ, unit = labels[rid]
        key = f"{typ}_{unit}" if typ == "Lamp" else typ
        if key not in MAPPING:
            skipped[f"unmapped:{key}"] += 1
            continue
        cls, status, _ = MAPPING[key]
        if status == "UNCERTAIN" and not args.include_uncertain:
            skipped[f"uncertain:{key}"] += 1
            continue
        if ("C", rid) not in members or ("V", rid) not in members:
            skipped["missing_flac"] += 1
            continue
        i = read_flac(*members[("C", rid)]) * factors[rid - 1, 0]
        v = read_flac(*members[("V", rid)]) * factors[rid - 1, 1]
        n = min(len(i), len(v))
        w = zc_windows(i[:n], SOURCE_RATE, MAINS_HZ, voltage=v[:n])
        if not len(w):
            skipped["no_zero_crossings"] += 1
            continue
        r = rms(w)
        on = np.where(r >= args.min_on_fraction * float(np.percentile(r, 90)))[0]
        if not len(on):
            skipped["never_on"] += 1
            continue
        keep = on[on >= on[0] + int(np.ceil(args.skip_after_on_s / win_s))]
        if not len(keep):
            skipped["no_steady_state"] += 1
            continue
        if len(keep) > args.max_windows_per_record:
            keep = keep[np.linspace(0, len(keep) - 1, args.max_windows_per_record).round().astype(int)]
        X.append(w[keep])
        y += [class_id(reg, cls)] * len(keep)
        g += [f"cooll:{typ}_{unit}"] * len(keep)
        units[cls].add(f"{typ}_{unit}")

    report = {
        "input": str(root),
        "timebase": "cycles",
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": MAINS_HZ,
        "retimed_to_50hz": False,
        "bandwidth_hz": bandwidth_hz("cycles", SOURCE_RATE, MAINS_HZ),
        "alignment": "voltage rising zero crossing",
        "units": "A, via scaleFactors.txt",
        "include_uncertain": args.include_uncertain,
        "units_per_class": {k: sorted(v) for k, v in units.items()},
        "skipped": dict(skipped),
        "caveats": [
            "CC BY-NC 4.0: excluded by --commercial-only. Development and evaluation only.",
            "Tools run with no load, so their current is below working level.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), np.array(g),
                 {k: {"model_class": c, "class_id": class_id(reg, c), "status": s, "note": note}
                  for k, (c, s, note) in MAPPING.items()}, report)
    print("     skipped:", dict(skipped))


if __name__ == "__main__":
    main()

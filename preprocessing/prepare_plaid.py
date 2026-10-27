#!/usr/bin/env python3
"""
prepare_plaid.py - Rebuild PLAID from the raw 30 kHz recordings, with recording groups.

Input: PLAID 2018 from figshare article 10084619 (CC BY 4.0):
    submetered.zip            1,876 CSVs (current A, voltage V) at 30 kHz, 60 Hz US
    metadata_submetered.json  type, house, brand, model, operating state per record

Why rebuild: data/plaid/X_real.npy has no record ids, so --group-split cannot
work on it, and nothing in this repo documents how it was made. Its windows do
match the model's design (8 voltage-aligned cycles x 50 samples), and this
script reproduces that convention from source with provenance.

Per record
----------
1. Cut cycle-synchronous windows: voltage rising zero crossing to the 8th
   crossing after it, resampled to 400 samples (signature_utils.zc_windows).
2. Steady state only. Drop windows below --min-on-fraction of the record's
   final current (appliance still off), and the --skip-after-on windows that
   follow switch-on (inrush). Records in a steady operating state (e.g.
   "high", "lowcool") keep every window.

Groups are appliance INSTANCES, not recordings: plaid:<house>:<type>:<brand>:<model>.
PLAID records the same physical appliance several times; splitting by
recording would still put one appliance on both sides of the split.

Output: data/plaid_v2/ (X_plaid_v2.npy etc.). data/plaid/X_real.npy is left untouched.

Usage:
  python preprocessing/prepare_plaid.py --plaid-dir ../datasets_raw/plaid2018_figshare_10084619
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, load_registry, rms,  # noqa: E402
                             save_outputs, zc_windows)

SOURCE = "plaid_v2"
SOURCE_RATE = 30_000
MAINS_HZ = 60.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--plaid-dir", default="../datasets_raw/plaid2018_figshare_10084619")
    p.add_argument("--out-dir", default="data/plaid_v2")
    p.add_argument("--min-on-fraction", type=float, default=0.5,
                   help="Keep windows whose RMS is at least this fraction of the record's final RMS")
    p.add_argument("--skip-after-on", type=int, default=1,
                   help="Windows to drop right after switch-on, to skip inrush (default 1 = 133 ms)")
    p.add_argument("--min-rms-a", type=float, default=0.02)
    return p.parse_args()


def instance_key(meta: dict) -> str:
    a = meta["appliance"]
    parts = [meta.get("location", "?"), a.get("type", "?"), a.get("brand", "") or "-",
             a.get("model_number", "") or "-"]
    return "plaid:" + ":".join(p.strip().replace(":", "_") for p in parts)


def main():
    args = parse_args()
    reg = load_registry()
    root = Path(args.plaid_dir)
    meta = json.loads((root / "metadata_submetered.json").read_text(encoding="utf-8"))
    z = zipfile.ZipFile(root / "submetered.zip")
    csv_by_id = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}

    X, y, g = [], [], []
    skipped = Counter()
    per_type_records = defaultdict(int)
    for rid in sorted(meta, key=int):
        m = meta[rid]
        typ = m["appliance"]["type"]
        try:
            cid = class_id(reg, typ)
        except KeyError:
            skipped[f"unmapped:{typ}"] += 1
            continue
        if rid not in csv_by_id:
            skipped["missing_csv"] += 1
            continue
        data = np.loadtxt(io.BytesIO(z.read(csv_by_id[rid])), delimiter=",", dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 2 or len(data) < SOURCE_RATE // 4:
            skipped["bad_csv"] += 1
            continue
        w = zc_windows(data[:, 0], SOURCE_RATE, MAINS_HZ, voltage=data[:, 1])
        if not len(w):
            skipped["no_zero_crossings"] += 1
            continue
        r = rms(w)
        final = float(np.median(r[-3:]))
        on = np.where(r >= args.min_on_fraction * final)[0]
        if final < args.min_rms_a or not len(on):
            skipped["below_min_rms"] += 1
            continue
        if m["appliance"].get("status") == "off-on":
            first = on[0] + args.skip_after_on
            keep = [i for i in on if i >= first]
        else:
            keep = list(on)
        if not keep:
            skipped["no_steady_state"] += 1
            continue
        X.append(w[keep])
        y += [cid] * len(keep)
        g += [instance_key(m)] * len(keep)
        per_type_records[typ] += 1

    g_arr = np.array(g)
    instances = {t: len({k for k, yy in zip(g, y) if yy == class_id(reg, t)}) for t in per_type_records}
    report = {
        "input": str(root),
        "records_in_metadata": len(meta),
        "timebase": "cycles",
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": MAINS_HZ,
        "retimed_to_50hz": True,
        "bandwidth_hz": bandwidth_hz("cycles", SOURCE_RATE, MAINS_HZ),
        "alignment": "voltage rising zero crossing",
        "records_used_per_type": dict(per_type_records),
        "appliance_instances_per_type": instances,
        "skipped": dict(skipped),
        "caveats": [
            "US 120 V / 60 Hz appliances. Cycle-synchronous sampling removes the frequency difference; the current level reflects 120 V (use --current-prior when mixing).",
            "Groups are appliance instances, so --group-split tests generalisation to unseen appliances, not just unseen recordings.",
            "Blender (2 records) and Hair Iron, Coffee maker, Water kettle (10 each) remain thin.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), g_arr,
                 {t: {"model_class": t, "class_id": class_id(reg, t), "status": "SAFE"}
                  for t in per_type_records}, report)
    print("     skipped:", dict(skipped))
    print("     appliance instances per type:", instances)


if __name__ == "__main__":
    main()

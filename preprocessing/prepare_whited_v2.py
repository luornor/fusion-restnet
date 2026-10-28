#!/usr/bin/env python3
"""
prepare_whited_v2.py - WHITED -> cycles-time-base windows in amps, with appliance-instance groups.

Input: WhiteD.zip from the WHITED page (TUM, Kahl et al. 2017), read in place.
    DATEN/<Type>_<Brand>_<region>_<kit>_<timestamp>.flac
    44.1 kHz stereo, about 5 s from switch-on. No licence stated ("for research").

Why a v2: prepare_whited.py reads channel 0 as current. Channel 0 is VOLTAGE
(about 222 V after the kit factor, 116 V on the Canadian r8 recordings, THD
under 1% for every appliance). Channel 1 is current (kettle 7.8 A, laptop THD
0.8). data/whited therefore holds mains voltage, not appliance current. It is
also raw30k (13 ms windows) and normalised per recording.

Per record
----------
1. Scale both channels by the measurement kit's factors (DATEN/_readme.txt):
   volts and amps.
2. Mains from the voltage: r8 (Canada) is 60 Hz, the rest 50 Hz.
3. Cycle-synchronous windows from the voltage rising zero crossing
   (signature_utils.zc_windows).
4. Steady state only, as prepare_plaid.py: drop windows below
   --min-on-fraction of the record's final RMS, and the --skip-after-on
   windows after switch-on (inrush).
5. Keep up to --max-windows-per-record, evenly spaced.

Groups are appliance instances: whited:<Type>:<Brand>. Each instance has
about 10 recordings, sometimes made in several regions.

Mapping statuses follow prepare_lilac.py: SAFE and EQUIVALENT are included,
UNCERTAIN only with --include-uncertain. The motor and resistive calls were
checked against measured THD and power factor (see the report).

Usage:
  python preprocessing/prepare_whited_v2.py --zip ../datasets_raw/whited_tum_gdrive/WhiteD.zip
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import (bandwidth_hz, class_id, estimate_mains_hz,  # noqa: E402
                             load_registry, rms, save_outputs, zc_windows)

SOURCE = "whited_v2"
SOURCE_RATE = 44_100
# DATEN/_readme.txt: sample value x factor = volts / amps, per measurement kit
KIT_FACTORS = {"MK1": (1033.64, 61.4835), "MK2": (861.15, 60.200), "MK3": (988.926, 60.9562)}

# WHITED type -> (model class, status, note)
MAPPING = {
    "AC":              ("Air Conditioner",              "SAFE",       ""),
    "CFL":             ("Compact Fluorescent Lamp",     "SAFE",       ""),
    "CoffeeMachine":   ("Coffee maker",                 "SAFE",       ""),
    "Fan":             ("Fan",                          "SAFE",       ""),
    "Fridge":          ("Fridge",                       "SAFE",       "Canadian 115 V unit"),
    "FlatIron":        ("Hair Iron",                    "SAFE",       "flat iron = hair straightener"),
    "HairDryer":       ("Hairdryer",                    "SAFE",       ""),
    "Hairdryer":       ("Hairdryer",                    "SAFE",       ""),
    "Heater":          ("Heater",                       "SAFE",       ""),
    "Kettle":          ("Water kettle",                 "SAFE",       ""),
    "Laptop":          ("Laptop",                       "SAFE",       ""),
    "LightBulb":       ("Incandescent Light Bulb",      "SAFE",       ""),
    "Microwave":       ("Microwave",                    "SAFE",       ""),
    "SolderingIron":   ("Soldering Iron",               "SAFE",       ""),
    "VacuumCleaner":   ("Vacuum",                       "SAFE",       ""),
    "DrillingMachine": ("Universal-motor tool",         "SAFE",       ""),
    "JigSaw":          ("Universal-motor tool",         "SAFE",       ""),
    "MultiTool":       ("Universal-motor tool",         "SAFE",       "rotary tool"),
    "BenchGrinder":    ("Single-phase induction motor", "EQUIVALENT", "THD 0.03, PF 0.92: induction, not universal"),
    "Halogen":         ("Incandescent Light Bulb",      "EQUIVALENT", "halogen is incandescent"),
    "HalogenFluter":   ("Incandescent Light Bulb",      "EQUIVALENT", "300 W halogen"),
    "Mixer":           ("Blender",                      "EQUIVALENT", "universal-motor kitchen appliance"),
    "JuiceMaker":      ("Blender",                      "EQUIVALENT", "universal-motor kitchen appliance"),
    "SewingMachine":   ("Universal-motor tool",         "EQUIVALENT", "universal motor"),
    "Iron":            ("Heater",                       "EQUIVALENT", "clothes iron, resistive (PF 1.00)"),
    "WaterHeater":     ("Heater",                       "EQUIVALENT", "resistive (PF 1.00)"),
    "FanHeater":       ("Heater",                       "EQUIVALENT", "resistive plus small fan (PF 1.00)"),
    "DeepFryer":       ("Heater",                       "EQUIVALENT", "resistive (PF 1.00)"),
    "Toaster":         ("Heater",                       "EQUIVALENT", "resistive (PF 1.00)"),
    "SandwichMaker":   ("Heater",                       "EQUIVALENT", "resistive (PF 1.00)"),
    "RiceCooker":      ("Heater",                       "EQUIVALENT", "resistive (PF 1.00)"),
    "Stove":           ("Heater",                       "EQUIVALENT", "resistive hotplate (PF 1.00)"),
    "WaterPump":       ("Single-phase induction motor", "UNCERTAIN",  "Grundfos circulator, PF 0.97: may be an electronic (ECM) drive"),
    "KitchenHood":     ("Fan",                          "UNCERTAIN",  "fan motor, PF 0.56"),
    "WashingMachine":  ("Washing Machine",              "UNCERTAIN",  "5 s from switch-on is only the inlet valve and electronics (0.11 A, PF 0.27)"),
    "Treadmill":       ("VSD-driven motor",             "UNCERTAIN",  "rectifier-fed DC motor drive"),
    "DesktopPC":       ("Laptop",                       "UNCERTAIN",  "SMPS with PFC"),
    "Monitor":         ("Laptop",                       "UNCERTAIN",  "SMPS"),
    "Charger":         ("Laptop",                       "UNCERTAIN",  "SMPS"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--zip", default="../datasets_raw/whited_tum_gdrive/WhiteD.zip")
    p.add_argument("--out-dir", default="data/whited_v2")
    p.add_argument("--include-uncertain", action="store_true")
    p.add_argument("--min-on-fraction", type=float, default=0.5)
    p.add_argument("--skip-after-on", type=int, default=1,
                   help="Windows to drop after switch-on (default 1 = 160 ms at 50 Hz)")
    p.add_argument("--max-windows-per-record", type=int, default=12,
                   help="Default 12 keeps every steady window of a 5 s record at 50 Hz")
    p.add_argument("--min-rms-a", type=float, default=0.02)
    return p.parse_args()


def parse_name(name: str) -> tuple[str, str, str, str] | None:
    parts = Path(name).stem.split("_")
    if len(parts) < 5 or parts[3] not in KIT_FACTORS:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def main():
    args = parse_args()
    reg = load_registry()
    z = zipfile.ZipFile(args.zip)
    records = sorted(n for n in z.namelist() if n.startswith("DATEN/") and n.count("/") == 1
                     and n.endswith(".flac"))

    X, y, g = [], [], []
    skipped = Counter()
    used = defaultdict(int)
    mains_seen = Counter()
    for n in records:
        meta = parse_name(n)
        if meta is None:
            skipped["unparsed_name"] += 1
            continue
        typ, brand, region, kit = meta
        if typ not in MAPPING:
            skipped[f"unmapped:{typ}"] += 1
            continue
        cls, status, _ = MAPPING[typ]
        if status == "UNCERTAIN" and not args.include_uncertain:
            skipped[f"uncertain:{typ}"] += 1
            continue
        data, sr = sf.read(io.BytesIO(z.read(n)), dtype="float64")
        if sr != SOURCE_RATE or data.ndim != 2 or data.shape[1] < 2:
            skipped["bad_audio"] += 1
            continue
        vf, cf = KIT_FACTORS[kit]
        v = (data[:, 0] - data[:, 0].mean()) * vf
        i = (data[:, 1] - data[:, 1].mean()) * cf
        mains = 60.0 if abs(estimate_mains_hz(v, sr) - 60.0) < 2.0 else 50.0
        mains_seen[(region, mains)] += 1
        w = zc_windows(i, sr, mains, voltage=v)
        if not len(w):
            skipped["no_zero_crossings"] += 1
            continue
        r = rms(w)
        final = float(np.median(r[-3:]))
        on = np.where(r >= args.min_on_fraction * final)[0]
        if final < args.min_rms_a or not len(on):
            skipped["below_min_rms"] += 1
            continue
        keep = on[on >= on[0] + args.skip_after_on]
        if not len(keep):
            skipped["no_steady_state"] += 1
            continue
        if len(keep) > args.max_windows_per_record:
            keep = keep[np.linspace(0, len(keep) - 1, args.max_windows_per_record).round().astype(int)]
        X.append(w[keep])
        y += [class_id(reg, cls)] * len(keep)
        g += [f"whited:{typ}:{brand}"] * len(keep)
        used[typ] += 1

    instances = defaultdict(set)
    for gg in g:
        instances[gg.split(":")[1]].add(gg)
    report = {
        "input": str(args.zip),
        "timebase": "cycles",
        "source_rate_hz": SOURCE_RATE,
        "source_mains_hz": "50 (r1-r7), 60 (r8 Canada), from the voltage channel",
        "retimed_to_50hz": True,
        "bandwidth_hz": bandwidth_hz("cycles", SOURCE_RATE, 50.0),
        "alignment": "voltage rising zero crossing (channel 0 = voltage, channel 1 = current)",
        "units": "A, via the kit factors in DATEN/_readme.txt",
        "include_uncertain": args.include_uncertain,
        "records_used_per_type": dict(used),
        "instances_per_type": {k: len(v) for k, v in instances.items()},
        "regions_and_mains": {f"{r}@{int(m)}Hz": c for (r, m), c in sorted(mains_seen.items())},
        "skipped": dict(skipped),
        "caveats": [
            "No licence stated. Development use only until the authors confirm terms.",
            "Most types are one or two physical units; Universal-motor tool gains 4 units (drill x2, jigsaw, rotary tool).",
            "Records are about 5 s from switch-on, so slow-starting loads (fridge, washing machine) show their start-up state.",
        ],
    }
    save_outputs(Path(args.out_dir), SOURCE, np.concatenate(X), np.array(y), np.array(g),
                 {t: {"model_class": c, "class_id": class_id(reg, c), "status": s, "note": note}
                  for t, (c, s, note) in MAPPING.items()}, report)
    print("     skipped:", dict(skipped))
    print("     regions/mains:", report["regions_and_mains"])


if __name__ == "__main__":
    main()

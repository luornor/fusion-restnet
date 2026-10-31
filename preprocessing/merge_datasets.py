#!/usr/bin/env python3
"""
merge_datasets.py - Merge per-source .npy arrays into one training set.

Run after the prepare_*.py scripts:
    python preprocessing/merge_datasets.py --out-dir data/combined_v2            # every cycles-time-base source
    python preprocessing/merge_datasets.py --commercial-only --out-dir data/combined_commercial
    python preprocessing/merge_datasets.py --sources plaid hifda whited cooll --allow-mixed-timebase
                                                                     # legacy mixed set (see TIME BASE)

Sources (in merge order; ids and licences come from preprocessing/class_registry.json):
    plaid           data/plaid/X_real.npy            PLAID, 60 Hz US       (legacy, no record ids)
    plaid_v2        data/plaid_v2/X_plaid_v2.npy     PLAID rebuilt from raw by prepare_plaid.py
    hifda           data/hifda/X_hifda.npy           HIFDA, 50 Hz ES       (legacy raw30k)
    hifda_v2        data/hifda_v2/X_hifda_v2.npy     HIFDA full-length records, voltage-aligned
    whited          data/whited/X_whited.npy         WHITED, 50 Hz AT      (legacy raw30k, VOLTAGE channel)
    whited_v2       data/whited_v2/X_whited_v2.npy   WHITED current channel, amps (licence unknown)
    cooll           data/cooll/X_cooll.npy           COOLL, 50 Hz FR       (legacy raw30k; CC BY-NC: not commercial)
    cooll_v2        data/cooll_v2/X_cooll_v2.npy     COOLL all 12 types, voltage-aligned, amps (CC BY-NC)
    lilac           data/lilac/X_lilac.npy           LILAC, 50 Hz DE       (licence unknown)
    caffeine        data/caffeine/X_caffeine.npy     CAFFEINE espresso, 50 Hz
    kaist_hvac      data/kaist_hvac/X_kaist_hvac.npy KAIST HVAC motor, 60->50 Hz
    nonlinear_loads data/nonlinear_loads/X_...npy    inverter AC, UPS etc., 60->50 Hz
    blond           data/blond/X_blond.npy           BLOND office appliances, 50 Hz DE (CC BY 4.0)

Output (in --out-dir, default data/combined):
    X_combined.npy       (N, 400)  float32
    y_combined.npy       (N,)      int64
    groups_combined.npy  (N,)      str      (see GROUP IDS)
    sources_combined.npy (N,)      str      source name per window
    groups_report.json, merge_report.json

GROUP IDS
---------
Each source may ship groups_<name>.npy naming the recording every window was
cut from. train_fusion_resnet.py --group-split keeps a recording wholly inside
train, val or test. groups_combined.npy is written only when EVERY merged
source has aligned ids. With --allow-missing-groups, sources without them get
one id per window: the split still runs, but those sources leak as before, and
groups_report.json lists them.

TIME BASE
---------
Windows are either 8 voltage-aligned cycles x 50 samples ("cycles", what the
model is built for) or 400 raw samples at 30 kHz ("raw30k", legacy HIFDA,
WHITED, COOLL). Adding a 133 ms window to a 13 ms one makes no physical
mixture, so merging both is refused unless --allow-mixed-timebase. See
preprocessing/signature_utils.py.

LICENCES
--------
--commercial-only keeps only sources whose registry entry has
commercial_use: true. Unknown (null) counts as not cleared. Use it to build
the training set for any model that will be sold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_utils import load_registry  # noqa: E402

# name -> (X, y, groups); paths relative to the repo root
SOURCES = {
    "plaid":           ("data/plaid/X_real.npy", "data/plaid/y_real.npy", "data/plaid/groups_real.npy"),
    "plaid_v2":        ("data/plaid_v2/X_plaid_v2.npy", "data/plaid_v2/y_plaid_v2.npy", "data/plaid_v2/groups_plaid_v2.npy"),
    "hifda":           ("data/hifda/X_hifda.npy", "data/hifda/y_hifda.npy", "data/hifda/groups_hifda.npy"),
    "hifda_v2":        ("data/hifda_v2/X_hifda_v2.npy", "data/hifda_v2/y_hifda_v2.npy", "data/hifda_v2/groups_hifda_v2.npy"),
    "whited":          ("data/whited/X_whited.npy", "data/whited/y_whited.npy", "data/whited/groups_whited.npy"),
    "whited_v2":       ("data/whited_v2/X_whited_v2.npy", "data/whited_v2/y_whited_v2.npy", "data/whited_v2/groups_whited_v2.npy"),
    "cooll":           ("data/cooll/X_cooll.npy", "data/cooll/y_cooll.npy", "data/cooll/groups_cooll.npy"),
    "cooll_v2":        ("data/cooll_v2/X_cooll_v2.npy", "data/cooll_v2/y_cooll_v2.npy", "data/cooll_v2/groups_cooll_v2.npy"),
    "lilac":           ("data/lilac/X_lilac.npy", "data/lilac/y_lilac.npy", "data/lilac/groups_lilac.npy"),
    "caffeine":        ("data/caffeine/X_caffeine.npy", "data/caffeine/y_caffeine.npy", "data/caffeine/groups_caffeine.npy"),
    "kaist_hvac":      ("data/kaist_hvac/X_kaist_hvac.npy", "data/kaist_hvac/y_kaist_hvac.npy", "data/kaist_hvac/groups_kaist_hvac.npy"),
    "nonlinear_loads": ("data/nonlinear_loads/X_nonlinear_loads.npy", "data/nonlinear_loads/y_nonlinear_loads.npy", "data/nonlinear_loads/groups_nonlinear_loads.npy"),
    "blond":           ("data/blond/X_blond.npy", "data/blond/y_blond.npy", "data/blond/groups_blond.npy"),
}
ORIGINAL = ["plaid", "hifda", "whited", "cooll"]
# A source listed here is dropped from "all" when its replacement exists.
SUPERSEDED_BY = {"plaid": "plaid_v2", "hifda": "hifda_v2", "whited": "whited_v2", "cooll": "cooll_v2"}


def source_timebase(name: str, root: Path, reg: dict) -> str | None:
    rep = root / "data" / name / f"{name}_preprocess_report.json"
    if rep.exists():
        tb = json.loads(rep.read_text(encoding="utf-8")).get("timebase")
        if tb:
            return tb
    return reg["sources"].get(name, {}).get("timebase")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-source arrays into one training set")
    p.add_argument("--sources", nargs="+", default=["all"],
                   help="Source names, or 'all' (default). 'all' uses plaid_v2 instead of legacy plaid "
                        "when it exists")
    p.add_argument("--timebase", choices=["cycles", "raw30k"], default="cycles",
                   help="Keep only sources in this time base (default cycles)")
    p.add_argument("--allow-mixed-timebase", action="store_true",
                   help="Reproduce the legacy set that mixes time bases. Not physically valid")
    p.add_argument("--out-dir", default="data/combined")
    p.add_argument("--commercial-only", action="store_true",
                   help="Keep only sources whose licence clearly allows commercial use")
    p.add_argument("--allow-missing-groups", action="store_true",
                   help="Write group ids even if some sources lack them, using one id "
                        "per window for those sources. They still leak across splits.")
    return p.parse_args()


def load_source_groups(name: str, g_path: Path, n: int) -> np.ndarray | None:
    if not g_path.exists():
        return None
    g = np.load(g_path, allow_pickle=True)
    if len(g) != n:
        print(f"[WARN] {g_path.name} has {len(g):,} ids for {n:,} windows - ignoring")
        return None
    # Prefix with the source so ids can never collide across datasets.
    return np.array([str(x) if str(x).startswith(f"{name}:") else f"{name}:{x}" for x in g])


def main() -> None:
    args = parse_args()
    reg = load_registry()
    n_classes = len(reg["classes"])
    root = Path(__file__).resolve().parent.parent
    explicit = args.sources != ["all"]
    names = args.sources if explicit else list(SOURCES)
    unknown = [s for s in names if s not in SOURCES]
    if unknown:
        sys.exit(f"[ERROR] unknown sources {unknown}; choose from {list(SOURCES)}")
    if not explicit:
        for old, new in SUPERSEDED_BY.items():
            if old in names and (root / SOURCES[new][0]).exists():
                names.remove(old)
                print(f"[INFO] using {new} instead of legacy {old}")

    timebases = {s: source_timebase(s, root, reg) for s in names}
    if not args.allow_mixed_timebase:
        off = [s for s in names if timebases[s] != args.timebase]
        if off and explicit:
            sys.exit(f"[ERROR] time base mismatch: {', '.join(f'{s}={timebases[s]}' for s in off)} "
                     f"vs --timebase {args.timebase}. Re-run their prepare scripts in this time base, "
                     "or pass --allow-mixed-timebase to reproduce the legacy (invalid) mix.")
        for s in off:
            print(f"[TIMEBASE] skipping {s}: {timebases[s]} (merging --timebase {args.timebase})")
        names = [s for s in names if s not in off]

    licence_dropped = []
    if args.commercial_only:
        kept = [s for s in names if reg["sources"].get(s, {}).get("commercial_use") is True]
        licence_dropped = [s for s in names if s not in kept]
        names = kept
        for s in licence_dropped:
            src = reg["sources"].get(s, {})
            print(f"[LICENCE] dropping {s}: {src.get('license')} (commercial_use={src.get('commercial_use')})")

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_g = out_dir / "groups_combined.npy"

    X_parts, y_parts, g_parts, s_parts = [], [], [], []
    with_groups, without_groups, missing_files = [], [], []
    for name in names:
        x_rel, y_rel, g_rel = SOURCES[name]
        x_path, y_path = root / x_rel, root / y_rel
        if not x_path.exists():
            print(f"[SKIP] {x_rel} not found (run preprocessing/prepare_{name}.py)")
            missing_files.append(name)
            continue
        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        assert X.shape[1] == 400, f"Expected 400-sample windows, got {X.shape}"
        assert len(X) == len(y), f"{x_rel}: {len(X)} windows but {len(y)} labels"
        assert y.min() >= 0 and y.max() < n_classes, f"{y_rel}: label outside registry 0..{n_classes - 1}"

        g = load_source_groups(name, root / g_rel, len(X))
        if g is None:
            without_groups.append(name)
            g = np.array([f"{name}:window:{i}" for i in range(len(X))])
            tag = "no group ids"
        else:
            with_groups.append(name)
            tag = f"{len(set(g)):,} recordings"

        X_parts.append(X)
        y_parts.append(y)
        g_parts.append(g)
        s_parts.append(np.full(len(X), name))
        print(f"[OK]   {name:<16} {X.shape[0]:>7,} windows  ({tag})")

    if not X_parts:
        sys.exit("[ERROR] No source files found.")

    X_c = np.concatenate(X_parts)
    y_c = np.concatenate(y_parts)
    g_c = np.concatenate(g_parts).astype(str)
    s_c = np.concatenate(s_parts).astype(str)

    np.save(out_dir / "X_combined.npy", X_c)
    np.save(out_dir / "y_combined.npy", y_c)
    np.save(out_dir / "sources_combined.npy", s_c)

    write_groups = not without_groups or args.allow_missing_groups
    if write_groups:
        np.save(out_g, g_c)
    elif out_g.exists():
        out_g.unlink()  # an old groups file would no longer line up with the new X

    class_names = [c["name"] for c in reg["classes"]]
    per_class = {}
    for cid in np.unique(y_c):
        m = y_c == cid
        per_class[class_names[cid]] = {
            "windows": int(m.sum()),
            "recordings": int(len(set(g_c[m]))),
            "sources": sorted(set(s_c[m].tolist())),
        }
    groups_report = {
        "groups_written": write_groups,
        "sources_with_recording_ids": with_groups,
        "sources_with_per_window_ids": without_groups if write_groups else [],
        "sources_missing_ids": without_groups,
        "n_windows": int(len(X_c)),
        "n_groups": int(len(set(g_c))) if write_groups else None,
        "clean_group_split": write_groups and not without_groups,
    }
    (out_dir / "groups_report.json").write_text(json.dumps(groups_report, indent=2), encoding="utf-8")
    merge_report = {
        "sources": names,
        "timebase": args.timebase if not args.allow_mixed_timebase else "MIXED",
        "source_timebases": {s: timebases.get(s) for s in names},
        "missing_source_files": missing_files,
        "dropped_for_licence": licence_dropped,
        "licences": {s: reg["sources"].get(s, {}).get("license") for s in names},
        "per_class": per_class,
        **groups_report,
    }
    (out_dir / "merge_report.json").write_text(json.dumps(merge_report, indent=2), encoding="utf-8")

    print(f"\n[DONE] {len(X_c):,} windows, {len(per_class)} classes -> {out_dir.relative_to(root)}/")
    for cname, v in per_class.items():
        print(f"       {cname:<30} {v['windows']:>7,} windows  {v['recordings']:>6} rec  {','.join(v['sources'])}")
    if without_groups:
        print(f"\n[WARN] No recording ids for: {', '.join(without_groups)}.")
        if write_groups:
            print("       They got one id per window, so --group-split does NOT stop leakage for them.")
        else:
            print("       groups_combined.npy was not written. Re-run those sources' prepare_*.py "
                  "scripts, or pass --allow-missing-groups.")


if __name__ == "__main__":
    main()

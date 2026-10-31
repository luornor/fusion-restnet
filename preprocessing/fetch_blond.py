"""Sampled download of BLOND per-appliance (MEDAL) files over the public FTP.

One ordinary weekday per socket-layout change in appliance_log.json, plus
evenly spaced extra weekdays, two office-hour files per MEDAL per day, plus the
logged test sessions. Usage: python preprocessing/fetch_blond.py ../datasets_raw/blond_mediatum_1375836 [--dry-run]
"""
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

FTP = "ftp://m1375836:m1375836@dataserv.ub.tum.de/BLOND"
MEDALS = [f"medal-{k}" for k in range(1, 16)]
HOLIDAYS = {"2016-10-03", "2016-11-01", "2016-12-26", "2017-01-06", "2017-04-14",
            "2017-04-17", "2017-05-01", "2017-05-25", "2017-06-05", "2017-06-15"}
XMAS = ("2016-12-23", "2017-01-08")
TIMES = {"BLOND-50": ["10:30", "14:30"], "BLOND-250": ["10:00", "11:30", "14:00", "15:30"]}
# (subset, date, medal, from, to): logged test sessions, fetched completely
SESSIONS = [("BLOND-50", "2016-11-21", "medal-2", "10:50", "11:30"),   # kettle, multi-tool
            ("BLOND-50", "2016-12-06", "medal-1", "08:45", "12:00"),   # paper shredder
            ("BLOND-50", "2016-12-07", "medal-1", "08:45", "12:00"),   # multi-tool
            ("BLOND-250", "2017-06-12", "medal-1", "10:50", "12:00")]  # space heater
EXTRA_PER_SUBSET = {"BLOND-50": 12, "BLOND-250": 4}


def ftp_list(path, names_only=False):
    cmd = ["curl", "-s", "-m", "120", "--retry", "3", f"{FTP}/{path}"]
    if names_only:
        cmd.insert(1, "--list-only")
    return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()


def usable(day):
    d = dt.date.fromisoformat(day)
    return d.weekday() < 5 and day not in HOLIDAYS and not (XMAS[0] <= day <= XMAS[1])


def file_time(name):
    m = re.search(r"T(\d\d)-(\d\d)-(\d\d)", name)
    return int(m[1]) * 60 + int(m[2]) if m else None


def minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def pick_days(subset, days, log_stamps):
    chosen = set()
    for stamp in log_stamps:
        d = dt.date.fromisoformat(stamp[:10]) + dt.timedelta(days=1)
        for _ in range(10):
            if d.isoformat() in days and usable(d.isoformat()):
                chosen.add(d.isoformat())
                break
            d += dt.timedelta(days=1)
    pool = [d for d in days if usable(d)]
    n = EXTRA_PER_SUBSET[subset]
    chosen.update(pool[int(k * (len(pool) - 1) / max(n - 1, 1))] for k in range(n))
    return sorted(chosen)


def main():
    out, dry = Path(sys.argv[1]), "--dry-run" in sys.argv
    log = json.loads((out / "appliance_log.json").read_text())
    stamps = sorted({e["timestamp"] for v in log.values() for e in v["entries"]})
    jobs = []
    for subset in ("BLOND-50", "BLOND-250"):
        days = sorted(n for n in ftp_list(f"{subset}/", names_only=True) if re.match(r"\d{4}-\d\d-\d\d$", n))
        for day in pick_days(subset, set(days), stamps):
            for medal in MEDALS:
                jobs.append((subset, day, medal, [minutes(t) for t in TIMES[subset]], None))
        print(subset, len(days), "days on server")
    for subset, day, medal, a, b in SESSIONS:
        jobs.append((subset, day, medal, None, (minutes(a), minutes(b))))

    plan = []
    for subset, day, medal, targets, window in jobs:
        listing = []
        for line in ftp_list(f"{subset}/{day}/{medal}/"):
            parts = line.split()
            if parts and parts[-1].endswith(".hdf5") and file_time(parts[-1]) is not None:
                listing.append((parts[-1], int(parts[4])))
        if not listing:
            continue
        if window:
            sel = [x for x in listing if window[0] <= file_time(x[0]) <= window[1]]
        else:
            sel = {min(listing, key=lambda x: abs(file_time(x[0]) - t)) for t in targets}
        plan += [(subset, day, medal, n, s) for n, s in sel]
    plan = sorted(set(plan))
    total = sum(p[4] for p in plan)
    print(f"{len(plan)} files, {total / 1e9:.1f} GB", flush=True)
    (out / "fetch_plan.json").write_text(json.dumps(plan, indent=0))
    if dry:
        return

    done = 0
    for subset, day, medal, name, size in plan:
        dest = out / subset / day / medal / name
        if not (dest.exists() and dest.stat().st_size == size):
            dest.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(3):
                subprocess.run(["curl", "-s", "-m", "900", f"{FTP}/{subset}/{day}/{medal}/{name}", "-o", str(dest)])
                if dest.exists() and dest.stat().st_size == size:
                    break
            else:
                print("FAILED", subset, day, medal, name, flush=True)
        done += 1
        if done % 25 == 0:
            print(f"{done}/{len(plan)}", flush=True)
    print("finished", done, flush=True)


if __name__ == "__main__":
    main()

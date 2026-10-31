"""Sampled download of BLOND per-appliance (MEDAL) files over the public FTP.

Every socket-appliance instance in appliance_log.json is covered on up to
DAYS_PER_UNIT ordinary weekdays spread over the time it was plugged in. Only
the MEDAL holding it is fetched, at the office-hour TIMES. The logged test
sessions are fetched completely. Usage: python preprocessing/fetch_blond.py ../datasets_raw/blond_mediatum_1375836 [--dry-run] [--replan]

The file list is cached in fetch_plan.json; --replan rebuilds it. Rerunning
skips files that are already complete.
"""
import datetime as dt
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
DAYS_PER_UNIT = 3
LIST_WORKERS, FETCH_WORKERS = 8, 4


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


def unit_days(log, day_subset):
    """(subset, day, medal) jobs covering every socket-appliance instance on
    up to DAYS_PER_UNIT usable days spread across the time it was plugged in.
    A layout entry applies from the day after its timestamp until the next one."""
    jobs, units = set(), 0
    for medal_key, v in log.items():
        medal = medal_key.lower()
        entries = sorted(v["entries"], key=lambda e: e["timestamp"])
        dates = {}
        for k, e in enumerate(entries):
            start = (dt.date.fromisoformat(e["timestamp"][:10]) + dt.timedelta(days=1)).isoformat()
            stop = entries[k + 1]["timestamp"][:10] if k + 1 < len(entries) else "9999"
            span = [d for d in day_subset if start <= d < stop and usable(d)]
            for socket, a in e.items():
                if socket.startswith("socket") and a.get("appliance_name"):
                    dates.setdefault((socket, a["appliance_name"]), []).extend(span)
        for days in dates.values():
            days = sorted(set(days))
            if not days:
                continue
            units += 1
            n = min(DAYS_PER_UNIT, len(days))
            for k in range(n):
                d = days[int(k * (len(days) - 1) / max(n - 1, 1))]
                jobs.add((day_subset[d], d, medal))
    return sorted(jobs), units


def main():
    out, dry = Path(sys.argv[1]), "--dry-run" in sys.argv
    log = json.loads((out / "appliance_log.json").read_text())
    day_subset = {}
    for subset in ("BLOND-50", "BLOND-250"):
        days = [n for n in ftp_list(f"{subset}/", names_only=True) if re.match(r"\d{4}-\d\d-\d\d$", n)]
        day_subset.update((d, subset) for d in days)
        print(subset, len(days), "days on server")
    picked, units = unit_days(log, day_subset)
    print(units, "socket-appliance instances,", len(picked), "medal-days")
    jobs = [(s, d, m, [minutes(t) for t in TIMES[s]], None) for s, d, m in picked]
    for subset, day, medal, a, b in SESSIONS:
        jobs.append((subset, day, medal, None, (minutes(a), minutes(b))))

    def select(job):
        subset, day, medal, targets, window = job
        listing = []
        for line in ftp_list(f"{subset}/{day}/{medal}/"):
            parts = line.split()
            if parts and parts[-1].endswith(".hdf5") and file_time(parts[-1]) is not None:
                listing.append((parts[-1], int(parts[4])))
        if not listing:
            return []
        if window:
            sel = [x for x in listing if window[0] <= file_time(x[0]) <= window[1]]
        else:
            sel = {min(listing, key=lambda x: abs(file_time(x[0]) - t)) for t in targets}
        return [(subset, day, medal, n, s) for n, s in sel]

    plan_file = out / "fetch_plan.json"
    if plan_file.exists() and "--replan" not in sys.argv:
        plan = [tuple(p) for p in json.loads(plan_file.read_text())]
    else:
        print(len(jobs), "directories to list", flush=True)
        with ThreadPoolExecutor(LIST_WORKERS) as pool:
            plan = sorted({p for sel in pool.map(select, jobs) for p in sel})
        plan_file.write_text(json.dumps(plan, indent=0))
    total = sum(p[4] for p in plan)
    print(f"{len(plan)} files, {total / 1e9:.1f} GB", flush=True)
    if dry:
        return

    def fetch(item):
        subset, day, medal, name, size = item
        dest = out / subset / day / medal / name
        if dest.exists() and dest.stat().st_size == size:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            subprocess.run(["curl", "-s", "-m", "900", f"{FTP}/{subset}/{day}/{medal}/{name}", "-o", str(dest)])
            if dest.exists() and dest.stat().st_size == size:
                return True
        print("FAILED", subset, day, medal, name, flush=True)
        return False

    done = ok = 0
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        for good in pool.map(fetch, plan):
            done, ok = done + 1, ok + good
            if done % 25 == 0:
                print(f"{done}/{len(plan)}", flush=True)
    print(f"finished {ok}/{len(plan)} complete", flush=True)


if __name__ == "__main__":
    main()

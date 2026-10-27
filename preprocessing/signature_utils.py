"""Shared helpers for turning appliance recordings into training signatures.

Every prepare_<source>.py script produces the same artefacts in data/<source>/:

    X_<source>.npy        (N, 400) float32   current windows at 30 kHz
    y_<source>.npy        (N,)     int64     class ids from class_registry.json
    groups_<source>.npy   (N,)     str       recording id per window, for --group-split
    <source>_class_map.json                  source label -> class id, with mapping status
    <source>_preprocess_report.json / .md    counts, rates, licence, caveats

Group ids are saved as a fixed-width string array, never an object array, so
they load without pickle.

TIME BASE
---------
A window is 400 samples, but two conventions exist in this repo:

  cycles  (default)  8 mains cycles, resampled to exactly 50 samples per cycle
                     (2.5 kHz at 50 Hz, 3 kHz at 60 Hz). This is what the model
                     was designed for: FryzeDecomposition reshapes the window to
                     (8, 50) and treats each 50-sample row as one mains cycle
                     against a one-cycle synthetic voltage. PLAID X_real is in
                     this form. Content up to 25x mains (1.25 kHz at 50 Hz).
  raw30k             400 consecutive raw samples at 30 kHz (13.3 ms, 2/3 of a
                     50 Hz cycle). The legacy HIFDA/WHITED/COOLL scripts produce
                     this. The Fryze branch is meaningless on it.

Never mix the two in one training set: merge_datasets.py refuses to.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 30_000
WINDOW_SIZE = 400
TARGET_MAINS_HZ = 50.0
SAMPLES_PER_CYCLE = 50
CYCLES_PER_WINDOW = WINDOW_SIZE // SAMPLES_PER_CYCLE  # 8
TIMEBASES = ("cycles", "raw30k")

REGISTRY_PATH = Path(__file__).resolve().parent / "class_registry.json"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry(path: Path | str | None = None) -> dict:
    with open(path or REGISTRY_PATH, encoding="utf-8") as f:
        reg = json.load(f)
    ids = [c["id"] for c in reg["classes"]]
    if ids != list(range(len(ids))):
        raise ValueError(f"class_registry ids must be 0..N-1 in order, got {ids}")
    return reg


def class_id(reg: dict, name: str) -> int:
    for c in reg["classes"]:
        if c["name"] == name:
            return int(c["id"])
    raise KeyError(f"class {name!r} is not in class_registry.json")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def effective_target_rate(source_mains_hz: float, retime_to_50hz: bool) -> float:
    """Rate to resample to so that, read as 30 kHz, the signal is at 50 Hz.

    A 60 Hz recording resampled to 36 kHz and then labelled 30 kHz plays one
    mains cycle in exactly 600 samples: the same as a 50 Hz cycle at 30 kHz.
    Harmonics move with it (60 Hz 5th = 300 Hz becomes 250 Hz), which is what
    the same load does on a 50 Hz grid to first order.
    """
    if retime_to_50hz and source_mains_hz and abs(source_mains_hz - TARGET_MAINS_HZ) > 1:
        return TARGET_RATE * source_mains_hz / TARGET_MAINS_HZ
    return float(TARGET_RATE)


def timebase_rate(timebase: str, source_mains_hz: float, retime_to_50hz: bool = True) -> float:
    """Sampling rate a source must be resampled to for the given time base.

    cycles: exactly SAMPLES_PER_CYCLE per nominal mains cycle, so 50 Hz and
            60 Hz sources both give 50 samples per cycle with no retiming.
    raw30k: 30 kHz, optionally retimed so 60 Hz plays as 50 Hz.
    """
    if timebase == "cycles":
        return float(SAMPLES_PER_CYCLE * source_mains_hz)
    if timebase == "raw30k":
        return effective_target_rate(source_mains_hz, retime_to_50hz)
    raise ValueError(f"unknown timebase {timebase!r}; choose from {TIMEBASES}")


def bandwidth_hz(timebase: str, source_rate: float, source_mains_hz: float) -> int:
    """Highest frequency, in real Hz, that the windows can carry."""
    cap = SAMPLES_PER_CYCLE * source_mains_hz / 2 if timebase == "cycles" else TARGET_RATE / 2
    return int(min(source_rate / 2, cap))


def cycle_windows(cycle: np.ndarray, timebase: str, rotations: int) -> list[np.ndarray]:
    """Windows from ONE exact steady-state mains cycle, at evenly spaced phases.

    Device windows start at arbitrary mains phase; a recorded cycle starts at
    one phase. Rotating the periodic cycle gives windows at other phases.
    """
    n = SAMPLES_PER_CYCLE if timebase == "cycles" else int(round(TARGET_RATE / TARGET_MAINS_HZ))
    cyc = resample_cycle(cycle, n)
    reps = int(np.ceil(WINDOW_SIZE / n)) + 1
    tiled = np.tile(cyc, reps)
    offsets = [int(round(k * n / rotations)) for k in range(rotations)]
    return [tiled[o:o + WINDOW_SIZE] for o in offsets]


def rising_zero_crossings(v: np.ndarray, rate: float, mains_hz: float) -> np.ndarray:
    """Fractional sample indices of rising zero crossings of the mains fundamental.

    The trace is band-passed around the fundamental first, so harmonics and
    noise cannot add spurious crossings.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(v, dtype=np.float64)
    x = x - x.mean()
    sos = butter(2, [0.6 * mains_hz, 1.6 * mains_hz], btype="bandpass", fs=rate, output="sos")
    f = sosfiltfilt(sos, x)
    i = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    return i + f[i] / (f[i] - f[i + 1])


def zc_windows(current: np.ndarray, rate: float, mains_hz: float,
               voltage: np.ndarray | None = None, assumed_lag_deg: float = 0.0,
               max_windows: int | None = None, step_cycles: int = CYCLES_PER_WINDOW) -> np.ndarray:
    """Cycle-synchronous 400-sample windows aligned to the voltage rising zero crossing.

    Each window is the source current from one rising voltage zero crossing to
    the 8th after it (8 real cycles, so grid frequency drift is absorbed),
    FFT-resampled to exactly 400 samples. This matches PLAID X_real, where a
    resistive load's fundamental sits at about -90 degrees: a sine from zero.

    voltage=None: align on the current's own crossings, shifted earlier by
    assumed_lag_deg (current lags voltage by acos(PF) in an inductive load).
    That is an assumption; prefer real voltage whenever the source has it.

    step_cycles < 8 gives overlapping windows (for very short recordings).
    """
    from scipy.signal import resample as fft_resample
    current = np.asarray(current, dtype=np.float64)
    if voltage is not None:
        # A consuming load draws positive average power. Negative means one
        # sensor is wired with reversed polarity (seen in CAFFEINE and the
        # nonlinear_loads analyser): flip the current so its phase against
        # voltage is physical.
        n = min(len(current), len(voltage))
        if float(np.mean(current[:n] * np.asarray(voltage[:n], dtype=np.float64))) < 0:
            current = -current
    ref = voltage if voltage is not None else current
    zc = rising_zero_crossings(ref, rate, mains_hz)
    if voltage is None and assumed_lag_deg:
        zc = zc - (assumed_lag_deg / 360.0) * (rate / mains_hz)
        zc = zc[zc >= 0]
    out = []
    for k in range(0, len(zc) - CYCLES_PER_WINDOW, step_cycles):
        a, b = int(round(zc[k])), int(round(zc[k + CYCLES_PER_WINDOW]))
        seg = np.asarray(current[a:b], dtype=np.float64)
        expected = CYCLES_PER_WINDOW * rate / mains_hz
        if len(seg) < 0.9 * expected or len(seg) > 1.1 * expected:
            continue  # a missed or doubled crossing: skip rather than distort
        out.append(fft_resample(seg - seg.mean(), WINDOW_SIZE).astype(np.float32))
        if max_windows and len(out) >= max_windows:
            break
    return np.stack(out) if out else np.empty((0, WINDOW_SIZE), dtype=np.float32)


def resample(x: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    """Polyphase resample with a rational ratio close to dst/src."""
    ratio = Fraction(dst_rate / src_rate).limit_denominator(1000)
    if ratio == 1:
        return np.asarray(x, dtype=np.float32)
    y = resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)
    return y.astype(np.float32)


def resample_cycle(cycle: np.ndarray, n_out: int) -> np.ndarray:
    """FFT-resample one exact period to n_out samples (treats it as periodic)."""
    from scipy.signal import resample as fft_resample
    return fft_resample(np.asarray(cycle, dtype=np.float64), n_out).astype(np.float32)


def windows(x: np.ndarray, size: int = WINDOW_SIZE, hop: int | None = None) -> np.ndarray:
    hop = hop or size
    n = (len(x) - size) // hop + 1 if len(x) >= size else 0
    if n <= 0:
        return np.empty((0, size), dtype=np.float32)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return np.asarray(x, dtype=np.float32)[idx]


def rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=axis))


def estimate_mains_hz(v: np.ndarray, rate: float) -> float:
    """Mains frequency from a voltage (or near-sinusoidal current) trace."""
    x = np.asarray(v, dtype=np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    band = (freqs > 40) & (freqs < 70)
    return float(freqs[band][np.argmax(spec[band])])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(out_dir: Path, source: str, X: np.ndarray, y: np.ndarray,
                 groups: list[str] | np.ndarray, class_map: dict, report: dict,
                 extra: dict[str, np.ndarray] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(groups).astype(str)
    if not (len(X) == len(y) == len(g)):
        raise ValueError(f"misaligned outputs: X {len(X)}, y {len(y)}, groups {len(g)}")
    if X.ndim != 2 or X.shape[1] != WINDOW_SIZE:
        raise ValueError(f"X must be (N, {WINDOW_SIZE}), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf")

    np.save(out_dir / f"X_{source}.npy", X)
    np.save(out_dir / f"y_{source}.npy", y)
    np.save(out_dir / f"groups_{source}.npy", g)
    for name, arr in (extra or {}).items():
        np.save(out_dir / f"{name}_{source}.npy", np.asarray(arr).astype(str)
                if np.asarray(arr).dtype == object else np.asarray(arr))

    with open(out_dir / f"{source}_class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    reg = load_registry()
    names = [c["name"] for c in reg["classes"]]
    counts = {names[int(c)]: int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    groups_per_class = {names[int(c)]: int(len(set(g[y == c]))) for c in np.unique(y)}
    report = {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timebase": report.get("timebase"),
        "window_size": WINDOW_SIZE,
        "n_windows": int(len(X)),
        "n_groups": int(len(set(g))),
        "class_counts": counts,
        "groups_per_class": groups_per_class,
        "rms_a_median_per_class": {
            names[int(c)]: round(float(np.median(rms(X[y == c]))), 4) for c in np.unique(y)},
        "licence": reg["sources"].get(source, {}).get("license"),
        "commercial_use": reg["sources"].get(source, {}).get("commercial_use"),
        **report,
    }
    with open(out_dir / f"{source}_preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [f"# {source} preprocessing report", "",
             f"- Generated: {report['generated_at']}",
             f"- Windows: {report['n_windows']:,} from {report['n_groups']} recordings",
             f"- Licence: {report['licence']} (commercial use: {report['commercial_use']})",
             f"- Time base: {report.get('timebase')}"
             f" ({'8 cycles x 50 samples' if report.get('timebase') == 'cycles' else '400 raw samples at 30 kHz'})",
             f"- Source rate: {report.get('source_rate_hz')} Hz, mains {report.get('source_mains_hz')} Hz,"
             f" retimed to 50 Hz: {report.get('retimed_to_50hz')}",
             f"- Usable bandwidth: up to {report.get('bandwidth_hz')} Hz", "",
             "| Class | Windows | Recordings | Median RMS (A) |", "|---|---:|---:|---:|"]
    for name, n in counts.items():
        lines.append(f"| {name} | {n:,} | {groups_per_class[name]} | "
                     f"{report['rms_a_median_per_class'][name]} |")
    if report.get("caveats"):
        lines += ["", "## Caveats", ""] + [f"- {c}" for c in report["caveats"]]
    (out_dir / f"{source}_preprocess_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] {source}: {len(X):,} windows, {len(set(g))} groups -> {out_dir}")
    for name, n in counts.items():
        print(f"     {name:<30} {n:>6,} windows  {groups_per_class[name]:>4} recordings")

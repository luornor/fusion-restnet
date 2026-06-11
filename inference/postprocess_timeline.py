"""
Post-Process Fusion-ResNet Inference: Timeline, Duration & Power
================================================================

Generic postprocessor that chains after inference_pipeline.py.
Reads the inference output directory, smooths predictions over time,
detects appliance ON/OFF intervals, and estimates power consumption.

Works with ANY inference output — no dependency on original source
format (.mat, .csv, .npy). Power is estimated from the saved
signal windows (current RMS * assumed voltage).

Usage:
    # Basic — reads from inference output dir
    python postprocess_timeline.py --input inference_results/natural

    # Custom smoothing & filtering
    python postprocess_timeline.py --input inference_results/natural \
        --smooth-window 10 --min-duration 5

    # Provide voltage for true power (otherwise uses current-only proxy)
    python postprocess_timeline.py --input inference_results/natural \
        --voltage 120

    # Custom output location
    python postprocess_timeline.py --input inference_results/natural \
        --output my_timeline
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-process Fusion-ResNet inference into appliance timelines')
    parser.add_argument('--input', type=str, required=True,
                        help='Inference output directory (contains predictions.npy, '
                             'probabilities.npy, predictions.json, and optionally '
                             'windows.npy, timestamps.npy, metadata.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: <input>/timeline)')
    parser.add_argument('--smooth-window', type=float, default=5.0,
                        help='Temporal smoothing window in seconds (default: 5)')
    parser.add_argument('--min-duration', type=float, default=2.0,
                        help='Minimum ON duration in seconds to report (default: 2)')
    parser.add_argument('--voltage', type=float, default=120.0,
                        help='Assumed RMS mains voltage for power estimation '
                             '(default: 120 V for US)')
    parser.add_argument('--window-duration', type=float, default=None,
                        help='Override window duration in seconds '
                             '(default: auto from metadata or timestamps)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


# ==============================================================================
# Data Loading
# ==============================================================================

def load_inference_outputs(input_dir: str) -> dict:
    """Load all outputs from the inference pipeline.

    Returns a dict with all available data.
    """
    data = {}

    # Required
    data['predictions'] = np.load(os.path.join(input_dir, 'predictions.npy'))
    data['probabilities'] = np.load(os.path.join(input_dir, 'probabilities.npy'))

    with open(os.path.join(input_dir, 'predictions.json')) as f:
        pjson = json.load(f)
    data['appliance_names'] = pjson['appliance_names']

    # Optional: metadata
    meta_path = os.path.join(input_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data['metadata'] = json.load(f)
    else:
        data['metadata'] = {}

    # Optional: timestamps
    ts_path = os.path.join(input_dir, 'timestamps.npy')
    if os.path.exists(ts_path):
        data['timestamps'] = np.load(ts_path)
    else:
        data['timestamps'] = None

    # Optional: signal windows (for power estimation)
    win_path = os.path.join(input_dir, 'windows.npy')
    if os.path.exists(win_path):
        data['windows'] = np.load(win_path)
    else:
        data['windows'] = None

    return data


def resolve_window_duration(data: dict, override: float = None) -> float:
    """Determine the duration of each window in seconds."""
    if override is not None:
        return override

    # From metadata
    meta = data.get('metadata', {})
    if meta.get('window_duration_s', 0) > 0:
        return meta['window_duration_s']

    # From timestamps
    ts = data.get('timestamps')
    if ts is not None and len(ts) > 1:
        return float(ts[1] - ts[0])

    # Fallback: 10 cycles at 60 Hz
    return 10.0 / 60.0


def build_timestamps(data: dict, window_dur: float) -> np.ndarray:
    """Get or synthesize timestamps for each window."""
    if data.get('timestamps') is not None:
        return data['timestamps']

    n = data['predictions'].shape[0]
    return np.arange(n) * window_dur


# ==============================================================================
# Power Estimation
# ==============================================================================

def estimate_window_power(windows: np.ndarray | None,
                          voltage: float = 120.0) -> np.ndarray | None:
    """Estimate apparent power per window from normalized current waveforms.

    Since the model input windows are normalized to unit magnitude,
    the RMS of each window reflects the *shape* of the current waveform
    (peaky = higher RMS ratio, sinusoidal ~ 0.707).

    Without the original un-normalized amplitude we can't get absolute watts,
    but we CAN compute a relative power profile that shows when power
    changes occur. We scale by voltage to give a ballpark figure.

    If windows were NOT saved during inference, returns None and power
    features are skipped.
    """
    if windows is None:
        return None

    # RMS current per window (on normalized signal, so this is 0..1)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))

    # Apparent power proxy: V_rms * I_rms_normalized
    # This gives relative power — useful for detecting changes
    power = voltage * rms

    return power


# ==============================================================================
# Temporal Smoothing
# ==============================================================================

def smooth_predictions(preds: np.ndarray, kernel_size: int) -> np.ndarray:
    """Majority-vote smoothing over a sliding window.

    For each position, if more than half the neighbors are ON, keep it ON.
    This removes flickering while preserving real state changes.
    """
    n_windows, n_classes = preds.shape
    smoothed = np.zeros_like(preds)
    half = kernel_size // 2

    for c in range(n_classes):
        col = preds[:, c].astype(np.float64)
        cumsum = np.concatenate([[0], np.cumsum(col)])

        for i in range(n_windows):
            lo = max(0, i - half)
            hi = min(n_windows, i + half + 1)
            mean_val = (cumsum[hi] - cumsum[lo]) / (hi - lo)
            smoothed[i, c] = 1 if mean_val >= 0.5 else 0

    return smoothed


# ==============================================================================
# Interval Detection
# ==============================================================================

def detect_intervals(smoothed: np.ndarray, timestamps: np.ndarray,
                     window_dur: float, appliance_names: list[str],
                     power: np.ndarray | None = None,
                     min_duration: float = 2.0) -> list[dict]:
    """Detect contiguous ON intervals for each appliance.

    Returns list of interval dicts sorted by start time.
    """
    n_windows, n_classes = smoothed.shape
    intervals = []

    for c in range(n_classes):
        states = smoothed[:, c]
        diffs = np.diff(states.astype(int))

        on_starts = np.where(diffs == 1)[0] + 1
        off_ends = np.where(diffs == -1)[0] + 1

        if states[0] == 1:
            on_starts = np.concatenate([[0], on_starts])
        if states[-1] == 1:
            off_ends = np.concatenate([off_ends, [len(states)]])

        for start_i, end_i in zip(on_starts, off_ends):
            t_start = timestamps[start_i]
            t_end = timestamps[min(end_i, len(timestamps) - 1)] + window_dur
            duration = t_end - t_start

            if duration < min_duration:
                continue

            entry = {
                'appliance': appliance_names[c],
                'class_idx': c,
                'start_s': round(float(t_start), 2),
                'end_s': round(float(t_end), 2),
                'duration_s': round(float(duration), 2),
                'n_windows': int(end_i - start_i),
            }

            # Power estimation
            if power is not None:
                on_mask = np.zeros(n_windows, dtype=bool)
                on_mask[start_i:end_i] = True

                P_on = float(power[on_mask].mean())

                # Baseline: nearby OFF windows (+/-30s)
                t_lo = t_start - 30
                t_hi = t_end + 30
                off_mask = (smoothed[:, c] == 0)
                near_mask = (timestamps >= t_lo) & (timestamps < t_hi)
                baseline_mask = off_mask & near_mask

                if baseline_mask.sum() > 5:
                    P_base = float(power[baseline_mask].mean())
                elif off_mask.sum() > 0:
                    P_base = float(power[off_mask].mean())
                else:
                    P_base = 0.0

                P_diff = abs(P_on - P_base)

                entry['power_on_W'] = round(P_on, 2)
                entry['power_baseline_W'] = round(P_base, 2)
                entry['power_delta_W'] = round(P_diff, 2)
                entry['energy_Wh'] = round(P_diff * duration / 3600, 4)

            intervals.append(entry)

    intervals.sort(key=lambda x: x['start_s'])
    return intervals


# ==============================================================================
# Reporting
# ==============================================================================

def print_timeline(intervals: list[dict], appliance_names: list[str],
                   has_power: bool):
    """Print timeline and per-appliance summary."""

    print(f"\n{'='*80}")
    print(f"  APPLIANCE ACTIVITY TIMELINE  ({len(intervals)} events)")
    print(f"{'='*80}")

    if has_power:
        print(f"\n  {'Appliance':<28s} {'ON':>8s} {'OFF':>8s} "
              f"{'Duration':>10s} {'~Power':>8s} {'Energy':>10s}")
        print(f"  {'-'*76}")
    else:
        print(f"\n  {'Appliance':<28s} {'ON':>8s} {'OFF':>8s} {'Duration':>10s}")
        print(f"  {'-'*58}")

    for iv in intervals:
        dur = iv['duration_s']
        dur_str = f"{dur/60:.1f} min" if dur >= 60 else f"{dur:.1f} s"

        line = (f"  {iv['appliance']:<28s} "
                f"{iv['start_s']:>7.1f}s {iv['end_s']:>7.1f}s "
                f"{dur_str:>10s}")

        if has_power:
            pwr = iv.get('power_delta_W', 0)
            energy = iv.get('energy_Wh', 0)
            line += f" {pwr:>7.1f}W {energy:>9.4f}Wh"

        print(line)

    # Per-appliance summary
    print(f"\n{'='*80}")
    print(f"  PER-APPLIANCE SUMMARY")
    print(f"{'='*80}")

    if has_power:
        print(f"  {'Appliance':<28s} {'#Events':>7s} {'Total ON':>10s} "
              f"{'Avg Power':>10s} {'Total Energy':>12s}")
        print(f"  {'-'*70}")
    else:
        print(f"  {'Appliance':<28s} {'#Events':>7s} {'Total ON':>10s}")
        print(f"  {'-'*48}")

    for name in appliance_names:
        app_ivs = [iv for iv in intervals if iv['appliance'] == name]
        if not app_ivs:
            continue

        n_events = len(app_ivs)
        total_on = sum(iv['duration_s'] for iv in app_ivs)
        dur_str = f"{total_on/60:.1f} min" if total_on >= 60 else f"{total_on:.1f} s"

        line = f"  {name:<28s} {n_events:>7d} {dur_str:>10s}"

        if has_power:
            total_energy = sum(iv.get('energy_Wh', 0) for iv in app_ivs)
            avg_power = np.mean([iv.get('power_delta_W', 0) for iv in app_ivs])
            line += f" {avg_power:>9.1f}W {total_energy:>11.4f}Wh"

        print(line)

    if has_power:
        grand_total = sum(iv.get('energy_Wh', 0) for iv in intervals)
        print(f"\n  Total estimated energy: {grand_total:.4f} Wh")


# ==============================================================================
# Saving
# ==============================================================================

def save_results(intervals: list[dict], output_dir: str):
    """Save intervals to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, 'timeline.json')
    with open(json_path, 'w') as f:
        json.dump({'n_events': len(intervals), 'intervals': intervals}, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # CSV
    import csv
    csv_path = os.path.join(output_dir, 'timeline.csv')
    if intervals:
        keys = intervals[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(intervals)
    print(f"  Saved: {csv_path}")


# ==============================================================================
# Plotting
# ==============================================================================

def plot_timeline(smoothed: np.ndarray, timestamps: np.ndarray,
                  intervals: list[dict], power: np.ndarray | None,
                  appliance_names: list[str], output_dir: str):
    """Generate a timeline plot with power curve + per-appliance ON/OFF bars."""

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 9,
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # Only plot appliances that were ever detected
    active_classes = [c for c in range(len(appliance_names))
                      if smoothed[:, c].sum() > 0]

    if not active_classes:
        print("  No active appliances to plot.")
        return

    has_power = power is not None
    n_panels = len(active_classes) + (1 if has_power else 0)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(16, 2.0 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle('Appliance Activity Timeline', fontsize=13, fontweight='bold')
    colors = plt.cm.tab20(np.linspace(0, 1, len(appliance_names)))

    panel = 0

    # Power panel (if available)
    if has_power:
        ax = axes[panel]
        ax.plot(timestamps, power, color='black', linewidth=0.5, alpha=0.8)
        ax.set_ylabel('Power\nProxy (W)')
        ax.set_title('Estimated Apparent Power', fontsize=10)
        panel += 1

    # Per-appliance panels
    for c in active_classes:
        ax = axes[panel]
        ax.fill_between(timestamps, 0, smoothed[:, c],
                        alpha=0.6, color=colors[c], step='mid')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OFF', 'ON'])
        ax.set_ylabel(appliance_names[c], fontsize=8)

        # Annotate intervals
        for iv in intervals:
            if iv['class_idx'] != c:
                continue
            mid = (iv['start_s'] + iv['end_s']) / 2
            dur = iv['duration_s']
            if has_power:
                pwr = iv.get('power_delta_W', 0)
                label = f"{dur:.0f}s | ~{pwr:.0f}W"
            else:
                label = f"{dur:.0f}s"
            ax.annotate(label, xy=(mid, 0.5), fontsize=7,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='white', ec='gray', alpha=0.8))
        panel += 1

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()

    path = os.path.join(output_dir, 'timeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_energy_summary(intervals: list[dict], appliance_names: list[str],
                        output_dir: str):
    """Bar chart of total energy and ON-time per appliance."""
    os.makedirs(output_dir, exist_ok=True)

    has_power = any('energy_Wh' in iv for iv in intervals)

    energy_by_app = {}
    duration_by_app = {}
    for iv in intervals:
        name = iv['appliance']
        energy_by_app[name] = energy_by_app.get(name, 0) + iv.get('energy_Wh', 0)
        duration_by_app[name] = duration_by_app.get(name, 0) + iv['duration_s']

    if not duration_by_app:
        return

    names = sorted(duration_by_app.keys(), key=lambda n: -duration_by_app[n])
    durations = [duration_by_app[n] / 60 for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    if has_power:
        energies = [energy_by_app.get(n, 0) for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Energy & Usage Summary', fontsize=13, fontweight='bold')

        ax1.barh(names, energies, color=colors)
        ax1.set_xlabel('Estimated Energy (Wh)')
        ax1.set_title('Total Energy per Appliance')
        ax1.invert_yaxis()
        for i, v in enumerate(energies):
            ax1.text(v + max(energies) * 0.01, i, f'{v:.3f}',
                     va='center', fontsize=8)

        ax2.barh(names, durations, color=colors)
        ax2.set_xlabel('Total ON Time (minutes)')
        ax2.set_title('Total ON Duration per Appliance')
        ax2.invert_yaxis()
        for i, v in enumerate(durations):
            ax2.text(v + max(durations) * 0.01, i, f'{v:.1f}',
                     va='center', fontsize=8)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle('Usage Summary', fontsize=13, fontweight='bold')
        ax.barh(names, durations, color=colors)
        ax.set_xlabel('Total ON Time (minutes)')
        ax.set_title('Total ON Duration per Appliance')
        ax.invert_yaxis()
        for i, v in enumerate(durations):
            ax.text(v + max(durations) * 0.01, i, f'{v:.1f}',
                    va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'energy_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, 'timeline')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Post-Processing: Timeline & Power Estimation")
    print(f"{'='*65}")
    print(f"  Input:          {input_dir}")
    print(f"  Output:         {output_dir}")
    print(f"  Smooth window:  {args.smooth_window}s")
    print(f"  Min duration:   {args.min_duration}s")
    print(f"  Voltage:        {args.voltage}V")

    # ---- Load inference outputs ----
    data = load_inference_outputs(input_dir)
    preds = data['predictions']
    probs = data['probabilities']
    appliance_names = data['appliance_names']
    n_windows, n_classes = preds.shape

    print(f"\n  Loaded: {n_windows} windows, {n_classes} classes")

    # ---- Resolve timing ----
    window_dur = resolve_window_duration(data, args.window_duration)
    timestamps = build_timestamps(data, window_dur)
    total_time = timestamps[-1] + window_dur if len(timestamps) > 0 else 0
    windows_per_sec = 1.0 / window_dur if window_dur > 0 else 6.0

    print(f"  Window duration: {window_dur*1000:.1f}ms ({windows_per_sec:.1f}/sec)")
    print(f"  Total duration:  {total_time:.1f}s ({total_time/60:.1f} min)")

    # ---- Compute smoothing kernel ----
    kernel = max(3, int(args.smooth_window * windows_per_sec))
    if kernel % 2 == 0:
        kernel += 1
    print(f"  Smoothing kernel: {kernel} windows ({kernel * window_dur:.1f}s)")

    # ---- Step 1: Smooth ----
    print("\n[1/4] Smoothing predictions...")
    smoothed = smooth_predictions(preds, kernel)
    raw_on = int(preds.sum())
    smooth_on = int(smoothed.sum())
    print(f"  Raw ON: {raw_on:,} -> Smoothed: {smooth_on:,} "
          f"({(smooth_on - raw_on) / max(raw_on, 1) * 100:+.1f}%)")

    # ---- Step 2: Estimate power ----
    print("\n[2/4] Estimating power...")
    power = estimate_window_power(data.get('windows'), args.voltage)
    if power is not None:
        print(f"  Power range: {power.min():.1f}W to {power.max():.1f}W, "
              f"mean={power.mean():.1f}W")
    else:
        print("  No windows.npy found — skipping power estimation.")
        print("  (Re-run inference to generate windows.npy)")

    # ---- Step 3: Detect intervals ----
    print("\n[3/4] Detecting ON/OFF intervals...")
    intervals = detect_intervals(
        smoothed, timestamps, window_dur, appliance_names,
        power=power, min_duration=args.min_duration)
    print(f"  Found {len(intervals)} events (>= {args.min_duration}s)")

    # ---- Step 4: Report & save ----
    print("\n[4/4] Generating reports...")
    has_power = power is not None
    print_timeline(intervals, appliance_names, has_power)
    save_results(intervals, output_dir)

    if not args.no_plots:
        plot_timeline(smoothed, timestamps, intervals, power,
                      appliance_names, output_dir)
        plot_energy_summary(intervals, appliance_names, output_dir)

    print(f"\nDone! Results in {output_dir}/")


if __name__ == '__main__':
    main()

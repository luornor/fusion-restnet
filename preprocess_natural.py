"""
Preprocess PLAID Natural .mat recordings for Fusion-ResNet inference.
=====================================================================

Reads all Waveform*.mat files from the Natural data folder, extracts
the iHall (current) signal, segments it into 10-cycle windows resampled
to 400 samples, normalizes to unit magnitude, and saves everything as
.npy files ready for the inference pipeline.

Usage:
    python preprocess_natural.py --mat-dir "path/to/Natural/7/7H2A2B2C2K2F2D2"
    python preprocess_natural.py --mat-dir "../Natural/7/7H2A2B2C2K2F2D2" --out-dir natural_processed
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess PLAID Natural .mat files for inference')
    parser.add_argument('--mat-dir', type=str, required=True,
                        help='Directory containing Waveform*.mat files')
    parser.add_argument('--out-dir', type=str, default='natural_processed',
                        help='Output directory for processed .npy files (default: natural_processed)')
    parser.add_argument('--window-size', type=int, default=400,
                        help='Target window size in samples (default: 400)')
    parser.add_argument('--n-cycles', type=int, default=10,
                        help='Number of mains cycles per window (default: 10)')
    parser.add_argument('--stride-cycles', type=int, default=None,
                        help='Stride in mains cycles (default: n_cycles, i.e. no overlap)')
    parser.add_argument('--signal-key', type=str, default='iHall',
                        choices=['iHall', 'iShunt'],
                        help='Which current signal to use (default: iHall)')
    return parser.parse_args()


def extract_and_segment(mat_path: str, signal_key: str = 'iHall',
                        window_size: int = 400, n_cycles: int = 10,
                        stride_cycles: int = None) -> dict:
    """Extract current from a .mat file and segment into windows.

    Returns dict with metadata and the (N, window_size) array.
    """
    data = sio.loadmat(mat_path)

    # --- Extract metadata ---
    sps = int(data['sps'].ravel()[0])
    mains_freq = int(data['mains_freq'].ravel()[0])
    duration = float(data.get('duration_t', [[0]])[0][0])
    desc = str(data.get('load_descr', [['unknown']])[0][0]).strip() if 'load_descr' in data else 'unknown'
    short = str(data.get('load_descr_short', [['unknown']])[0][0]).strip() if 'load_descr_short' in data else 'unknown'

    # --- Extract current signal ---
    current = data[signal_key].ravel().astype(np.float64)

    # --- Segment into windows ---
    samples_per_cycle = sps // mains_freq  # 15384 / 60 = 256.4 -> 256
    raw_window = samples_per_cycle * n_cycles  # e.g., 256 * 10 = 2560

    if stride_cycles is None:
        stride_cycles = n_cycles  # no overlap
    raw_stride = samples_per_cycle * stride_cycles

    n_windows = max(1, (len(current) - raw_window) // raw_stride + 1)

    windows = []
    timestamps = []

    for i in range(n_windows):
        start = i * raw_stride
        end = start + raw_window

        if end > len(current):
            break

        chunk = current[start:end]

        # Resample to target window_size via linear interpolation
        x_old = np.linspace(0, 1, len(chunk))
        x_new = np.linspace(0, 1, window_size)
        resampled = np.interp(x_new, x_old, chunk)

        windows.append(resampled)
        timestamps.append(start / sps)

    windows = np.stack(windows).astype(np.float64)
    timestamps = np.array(timestamps)

    # --- Normalize each window to unit magnitude ---
    max_vals = np.abs(windows).max(axis=1, keepdims=True)
    max_vals = np.where(max_vals == 0, 1.0, max_vals)
    windows = windows / max_vals

    return {
        'windows': windows,
        'timestamps': timestamps,
        'sps': sps,
        'mains_freq': mains_freq,
        'duration_s': len(current) / sps,
        'n_samples_raw': len(current),
        'load_descr': desc,
        'load_descr_short': short,
        'source_file': os.path.basename(mat_path),
    }


def main():
    args = parse_args()

    mat_dir = args.mat_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find all Waveform*.mat files
    mat_files = sorted(glob.glob(os.path.join(mat_dir, 'Waveform*.mat')))
    if not mat_files:
        print(f"ERROR: No Waveform*.mat files found in {mat_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  Preprocessing Natural Data for Fusion-ResNet")
    print(f"{'='*60}")
    print(f"  Source:   {mat_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  Files:    {len(mat_files)}")
    print(f"  Signal:   {args.signal_key}")
    print(f"  Window:   {args.n_cycles} cycles -> {args.window_size} samples")
    stride = args.stride_cycles or args.n_cycles
    print(f"  Stride:   {stride} cycles ({'no overlap' if stride == args.n_cycles else f'{(1 - stride/args.n_cycles)*100:.0f}% overlap'})")

    all_windows = []
    all_timestamps = []
    all_sources = []
    total_duration = 0

    for mat_path in mat_files:
        fname = os.path.basename(mat_path)
        print(f"\n  Processing: {fname}")

        result = extract_and_segment(
            mat_path,
            signal_key=args.signal_key,
            window_size=args.window_size,
            n_cycles=args.n_cycles,
            stride_cycles=args.stride_cycles,
        )

        windows = result['windows']
        timestamps = result['timestamps']

        print(f"    Appliances: {result['load_descr']}")
        print(f"    SPS={result['sps']}, mains={result['mains_freq']}Hz, "
              f"duration={result['duration_s']:.1f}s")
        print(f"    -> {windows.shape[0]} windows of shape {windows.shape[1]}")

        # Save per-file output
        stem = Path(fname).stem
        np.save(os.path.join(out_dir, f'{stem}_windows.npy'), windows)
        np.save(os.path.join(out_dir, f'{stem}_timestamps.npy'), timestamps)

        all_windows.append(windows)
        all_timestamps.append(timestamps)
        all_sources.extend([fname] * len(windows))
        total_duration += result['duration_s']

    # Concatenate everything into one big array
    all_windows = np.concatenate(all_windows, axis=0)
    all_timestamps = np.concatenate(all_timestamps, axis=0)

    # Save combined output
    np.save(os.path.join(out_dir, 'all_windows.npy'), all_windows)
    np.save(os.path.join(out_dir, 'all_timestamps.npy'), all_timestamps)

    # Save source index (which file each window came from)
    np.save(os.path.join(out_dir, 'all_sources.npy'), np.array(all_sources))

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Total recordings: {len(mat_files)}")
    print(f"  Total duration:   {total_duration:.0f}s ({total_duration/60:.1f} min)")
    print(f"  Total windows:    {all_windows.shape[0]}")
    print(f"  Combined shape:   {all_windows.shape}")
    print(f"  Output files:")
    print(f"    {out_dir}/all_windows.npy       <- use with --pre-segmented")
    print(f"    {out_dir}/all_timestamps.npy")
    print(f"    {out_dir}/all_sources.npy")
    for mat_path in mat_files:
        stem = Path(mat_path).stem
        print(f"    {out_dir}/{stem}_windows.npy")
    print(f"\n  Next step — run inference:")
    print(f"    python inference_pipeline.py --checkpoint checkpoints/best.pt \\")
    print(f"        --input {out_dir}/all_windows.npy --pre-segmented --device cuda")


if __name__ == '__main__':
    main()

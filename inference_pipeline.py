"""
Fusion-ResNet Inference Pipeline for NILM Deployment
=====================================================

Sliding-window inference on continuous household electrical signals.
Takes raw current waveform data and predicts which appliances are active
in each time window.

Usage:
    # On a CSV file with 'Current' column (sampled at 30kHz)
    python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
        --input data/household_recording.csv --sample-rate 30000

    # On a NumPy file (already at 30kHz)
    python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
        --input data/recording.npy --sample-rate 30000

    # On pre-segmented windows (400 samples each, already normalized)
    python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
        --input data/windows.npy --pre-segmented

    # With custom settings
    python inference_pipeline.py --checkpoint best.pt --input recording.csv \
        --sample-rate 30000 --window-size 400 --stride 200 --device cpu --fp32
"""

from __future__ import annotations

import os
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import FastICA

from fusion_resnet import FusionResNet, FusionResNetLite

warnings.filterwarnings('ignore')

# Default PLAID appliance names
DEFAULT_APPLIANCE_NAMES = [
    'Air Conditioner', 'Blender', 'Coffee maker',
    'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Laptop',
    'Microwave', 'Soldering Iron', 'Vacuum', 'Washing Machine',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fusion-ResNet Inference Pipeline for NILM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file: .csv (with Current column), .npy, or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                        help='Output directory for results')
    parser.add_argument('--sample-rate', type=int, default=30000,
                        help='Sampling rate of raw input data in Hz (default: 30000)')
    parser.add_argument('--window-size', type=int, default=400,
                        help='Model input window size in samples (default: 400)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride between windows (default: window_size // 2)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: use checkpoint value)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'lite'], help='Model variant')
    parser.add_argument('--fp32', action='store_true',
                        help='Use float32 instead of float64')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Inference batch size')
    parser.add_argument('--pre-segmented', action='store_true',
                        help='Input is already segmented into windows (N, 400)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory with training data (for ICA fitting)')
    parser.add_argument('--ica-path', type=str, default=None,
                        help='Path to saved ICA parameters (.npz). '
                             'If not provided, ICA is re-fitted from training data.')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Only show top-k most confident appliances per window')
    return parser.parse_args()


# ==============================================================================
# Signal Preprocessing
# ==============================================================================

def resample_signal(signal: np.ndarray, orig_rate: int,
                    target_length: int, mains_freq: int = 60) -> np.ndarray:
    """Resample a raw signal to target_length samples per window.

    For PLAID-compatible processing:
    - 1 electrical cycle at 60Hz = orig_rate/60 samples
    - We take ~10 cycles worth of data and resample to target_length

    Args:
        signal: 1D raw current waveform
        orig_rate: Original sampling rate (e.g., 30000 Hz)
        target_length: Target samples per window (e.g., 400)
        mains_freq: Mains frequency (60 Hz for US, 50 Hz for EU)

    Returns:
        Resampled signal of length target_length
    """
    samples_per_cycle = orig_rate // mains_freq  # e.g., 500 at 30kHz/60Hz
    n_cycles = 10  # Match PLAID preprocessing
    chunk_size = samples_per_cycle * n_cycles  # e.g., 5000

    if len(signal) < chunk_size:
        # Pad with zeros if signal is too short
        signal = np.pad(signal, (0, chunk_size - len(signal)))
    elif len(signal) > chunk_size:
        # Take the first chunk_size samples
        signal = signal[:chunk_size]

    # Resample to target_length using linear interpolation
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    resampled = np.interp(x_new, x_old, signal)

    return resampled


def segment_continuous_signal(signal: np.ndarray, orig_rate: int,
                              window_size: int = 400, stride: int = None,
                              mains_freq: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Segment a continuous raw signal into overlapping windows.

    Each window covers ~10 electrical cycles, resampled to window_size samples.

    Args:
        signal: 1D continuous current waveform at original sample rate
        orig_rate: Sampling rate in Hz
        window_size: Output window size (model input size)
        stride: Stride in terms of raw samples (default: half a window's raw equivalent)
        mains_freq: Mains frequency (60 Hz US, 50 Hz EU)

    Returns:
        windows: (N, window_size) array of normalized signal windows
        timestamps: (N,) array of start times in seconds for each window
    """
    samples_per_cycle = orig_rate // mains_freq  # e.g., 500
    raw_window = samples_per_cycle * 10  # 10 cycles of raw data per window

    if stride is None:
        stride = raw_window // 2  # 50% overlap by default

    n_windows = max(1, (len(signal) - raw_window) // stride + 1)

    windows = []
    timestamps = []

    for i in range(n_windows):
        start = i * stride
        end = start + raw_window

        if end > len(signal):
            break

        chunk = signal[start:end]

        # Resample to model input size
        resampled = resample_signal(chunk, orig_rate, window_size, mains_freq)
        windows.append(resampled)
        timestamps.append(start / orig_rate)  # Convert to seconds

    windows = np.stack(windows)
    timestamps = np.array(timestamps)

    return windows, timestamps


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Normalize each window to unit magnitude (matching training preprocessing)."""
    max_vals = np.abs(windows).max(axis=1, keepdims=True)
    max_vals = np.where(max_vals == 0, 1.0, max_vals)  # Avoid division by zero
    return windows / max_vals


# ==============================================================================
# ICA Utilities
# ==============================================================================

def fit_ica_from_training_data(data_dir: str, n_components: int) -> dict:
    """Fit ICA on training data (reproduces training pipeline).

    Args:
        data_dir: Path to directory containing X_real.npy and y_real.npy.
        n_components: Number of ICA components (must match checkpoint).
    """
    print("  Fitting ICA from training data...")
    X_real = np.load(f'{data_dir}/X_real.npy', allow_pickle=True)
    y_real = np.load(f'{data_dir}/y_real.npy', allow_pickle=True)

    # Drop rare classes (same as training)
    _, counts = np.unique(y_real, return_counts=True)
    to_drop = np.argwhere(counts < 10).ravel()
    for idx in to_drop:
        mask = y_real != idx
        y_real = y_real[mask]
        X_real = X_real[mask]

    # Normalize
    X_real = X_real / np.abs(X_real).max(axis=1, keepdims=True)

    # Fit ICA
    fica = FastICA(n_components, whiten='unit-variance', random_state=42)
    fica.fit(X_real)

    X_ica = fica.transform(X_real)
    X_exp = np.exp(X_ica)
    m = X_exp.mean(axis=0, keepdims=True)
    s = X_exp.std(axis=0, keepdims=True, ddof=1)

    return {
        'U': fica.components_.astype(np.float64),
        'M': fica.mean_.astype(np.float64),
        'm': m,
        's': s,
    }


def save_ica_params(ica_params: dict, path: str):
    """Save ICA parameters to a .npz file."""
    np.savez(path, **ica_params)
    print(f"  Saved ICA parameters to {path}")


def load_ica_params(path: str) -> dict:
    """Load ICA parameters from a .npz file."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(checkpoint_path: str, variant: str, n_classes: int,
               signal_length: int, ica_params: dict, device: str,
               dtype: torch.dtype) -> tuple:
    """Load a trained Fusion-ResNet model from checkpoint.

    Returns:
        model: Loaded model in eval mode
        threshold: Classification threshold from training
    """
    ModelClass = FusionResNet if variant == 'full' else FusionResNetLite

    model = ModelClass(
        n_classes=n_classes,
        signal_length=signal_length,
        U=ica_params['U'],
        M=ica_params['M'],
        m=ica_params['m'],
        s=ica_params['s'],
    )

    if dtype == torch.float64:
        model = model.double()
    else:
        model = model.float()

    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    threshold = ckpt.get('threshold', 0.5)
    epoch = ckpt.get('epoch', '?')
    best_f1 = ckpt.get('best_val_f1', 'N/A')

    print(f"  Loaded checkpoint: epoch {epoch}, val F1 = {best_f1}")

    model.eval()
    return model, threshold


# ==============================================================================
# Inference Engine
# ==============================================================================

@torch.no_grad()
def run_inference(model, windows: np.ndarray, threshold: float,
                  device: str, dtype: torch.dtype,
                  batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference on pre-processed windows.

    Args:
        model: Trained model in eval mode
        windows: (N, signal_length) normalized signal windows
        threshold: Classification threshold
        device: 'cuda' or 'cpu'
        dtype: torch.float32 or torch.float64
        batch_size: Batch size for inference

    Returns:
        predictions: (N, n_classes) binary predictions
        probabilities: (N, n_classes) sigmoid probabilities
    """
    all_probs = []

    for i in range(0, len(windows), batch_size):
        batch = torch.tensor(
            windows[i:i + batch_size], dtype=dtype, device=device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


# ==============================================================================
# Input Loading
# ==============================================================================

def load_input(input_path: str, pre_segmented: bool = False,
               sample_rate: int = 30000, window_size: int = 400,
               stride: int = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Load input data from various formats.

    Supports:
        - .csv: Expects 'Current' column (continuous raw signal)
        - .npy: Either (N, 400) pre-segmented or 1D continuous signal
        - directory of .csv/.npy files (processes each)

    Returns:
        windows: (N, window_size) array
        timestamps: (N,) array of seconds, or None if pre-segmented
    """
    path = Path(input_path)

    if path.is_dir():
        all_windows = []
        all_timestamps = []
        for f in sorted(path.iterdir()):
            if f.suffix in ('.csv', '.npy'):
                w, t = load_input(str(f), pre_segmented, sample_rate,
                                  window_size, stride)
                all_windows.append(w)
                all_timestamps.append(t)
        windows = np.concatenate(all_windows)
        timestamps = np.concatenate(all_timestamps) if all_timestamps[0] is not None else None
        return windows, timestamps

    print(f"  Loading: {input_path}")

    if path.suffix == '.csv':
        df = pd.read_csv(input_path)
        # Try common column names
        for col in ['Current', 'current', 'I', 'i', 'current_rms']:
            if col in df.columns:
                signal = df[col].values.astype(np.float64)
                break
        else:
            # Assume first numeric column is current
            signal = df.iloc[:, 0].values.astype(np.float64)
            print(f"  Warning: No 'Current' column found, using first column")

    elif path.suffix == '.npy':
        data = np.load(input_path, allow_pickle=True)
        if pre_segmented or (data.ndim == 2 and data.shape[1] == window_size):
            print(f"  Pre-segmented: {data.shape[0]} windows of {data.shape[1]} samples")
            windows = normalize_windows(data.astype(np.float64))
            return windows, None
        else:
            signal = data.ravel().astype(np.float64)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"  Raw signal: {len(signal)} samples at {sample_rate} Hz "
          f"({len(signal) / sample_rate:.2f}s)")

    # Segment into windows
    windows, timestamps = segment_continuous_signal(
        signal, sample_rate, window_size, stride)
    windows = normalize_windows(windows)

    print(f"  Segmented into {len(windows)} windows")
    return windows, timestamps


# ==============================================================================
# Result Formatting & Output
# ==============================================================================

def format_results(predictions: np.ndarray, probabilities: np.ndarray,
                   timestamps: np.ndarray | None,
                   appliance_names: list[str],
                   top_k: int = None) -> list[dict]:
    """Format predictions into human-readable results.

    Returns:
        List of dicts, one per window, with active appliances and confidences.
    """
    results = []

    for i in range(len(predictions)):
        active_indices = np.where(predictions[i] == 1)[0]
        active_appliances = []

        for idx in active_indices:
            active_appliances.append({
                'name': appliance_names[idx],
                'confidence': float(probabilities[i, idx]),
            })

        # Sort by confidence (highest first)
        active_appliances.sort(key=lambda x: x['confidence'], reverse=True)

        if top_k is not None:
            active_appliances = active_appliances[:top_k]

        entry = {
            'window': i,
            'active_appliances': active_appliances,
            'n_active': len(active_appliances),
        }

        if timestamps is not None:
            entry['time_start_s'] = float(timestamps[i])

        results.append(entry)

    return results


def print_results_summary(results: list[dict], appliance_names: list[str],
                          predictions: np.ndarray):
    """Print a concise summary of inference results."""
    n_windows = len(results)

    print(f"\n{'='*60}")
    print(f"  INFERENCE RESULTS ({n_windows} windows)")
    print(f"{'='*60}")

    # Overall appliance detection counts
    detection_counts = predictions.sum(axis=0)
    detection_pct = detection_counts / n_windows * 100

    print(f"\n  Appliance Detection Summary:")
    print(f"  {'Appliance':<30s} {'Detections':>10s} {'% Windows':>10s}")
    print(f"  {'-'*50}")

    for i, name in enumerate(appliance_names):
        if detection_counts[i] > 0:
            print(f"  {name:<30s} {int(detection_counts[i]):>10d} "
                  f"{detection_pct[i]:>9.1f}%")

    # Average active appliances per window
    avg_active = predictions.sum(axis=1).mean()
    print(f"\n  Average active appliances per window: {avg_active:.1f}")

    # Show a few sample windows
    n_show = min(10, n_windows)
    print(f"\n  Sample predictions (first {n_show} windows):")
    print(f"  {'Window':>6s} {'Time (s)':>9s}  Active Appliances")
    print(f"  {'-'*60}")

    for result in results[:n_show]:
        time_str = f"{result['time_start_s']:.3f}" if 'time_start_s' in result else "N/A"
        appliances = ', '.join(
            f"{a['name']} ({a['confidence']:.2f})"
            for a in result['active_appliances']
        )
        if not appliances:
            appliances = "(none detected)"
        print(f"  {result['window']:>6d} {time_str:>9s}  {appliances}")

    print(f"{'='*60}")


def save_results(results: list[dict], predictions: np.ndarray,
                 probabilities: np.ndarray, timestamps: np.ndarray | None,
                 windows: np.ndarray, appliance_names: list[str],
                 output_dir: str, sample_rate: int = 0,
                 window_duration_s: float = 0.0):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump({
            'n_windows': len(results),
            'appliance_names': appliance_names,
            'results': results,
        }, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save CSV summary (one row per window)
    rows = []
    for i, result in enumerate(results):
        row = {'window': i}
        if 'time_start_s' in result:
            row['time_s'] = result['time_start_s']
        for j, name in enumerate(appliance_names):
            row[f'{name}_pred'] = int(predictions[i, j])
            row[f'{name}_prob'] = float(probabilities[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save raw arrays
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(output_dir, 'windows.npy'), windows)
    if timestamps is not None:
        np.save(os.path.join(output_dir, 'timestamps.npy'), timestamps)

    # Save metadata for postprocessor
    meta = {
        'n_windows': int(predictions.shape[0]),
        'n_classes': int(predictions.shape[1]),
        'window_size': int(windows.shape[1]),
        'sample_rate': int(sample_rate),
        'window_duration_s': float(window_duration_s),
        'appliance_names': appliance_names,
        'has_timestamps': timestamps is not None,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved numpy arrays + metadata to {output_dir}/")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    dtype = torch.float32 if args.fp32 else torch.float64
    stride = args.stride

    print(f"\n{'='*60}")
    print(f"  Fusion-ResNet NILM Inference Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {args.device} | Dtype: {dtype}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input: {args.input}")

    # ------------------------------------------------------------------
    # 1. Detect n_classes from checkpoint
    # ------------------------------------------------------------------
    print("\n[1/5] Inspecting checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # The last classifier layer's output size == n_classes
    classifier_weight_key = [k for k in ckpt['model_state_dict']
                             if 'classifier' in k and 'weight' in k][-1]
    n_classes = ckpt['model_state_dict'][classifier_weight_key].shape[0]
    # ICA U shape[0] == n_ica_components
    ica_U_key = [k for k in ckpt['model_state_dict'] if 'ica' in k and '.U' in k][0]
    n_ica_components = ckpt['model_state_dict'][ica_U_key].shape[0]
    print(f"  Checkpoint has {n_classes} output classes, {n_ica_components} ICA components")

    # ------------------------------------------------------------------
    # 2. Load appliance names (filter to match checkpoint n_classes)
    # ------------------------------------------------------------------
    appliance_names = DEFAULT_APPLIANCE_NAMES
    lenc_path = os.path.join(args.data_dir, 'real_label_encoder.npy')
    if os.path.exists(lenc_path):
        try:
            lenc = np.load(lenc_path, allow_pickle=True).item()
            all_names = list(lenc.classes_)
            if len(all_names) != n_classes:
                # The training script drops classes with <10 samples.
                # Reproduce the same filtering to get the correct name list.
                y_real = np.load(os.path.join(args.data_dir, 'y_real.npy'),
                                 allow_pickle=True)
                _, counts = np.unique(y_real, return_counts=True)
                kept = [i for i, c in enumerate(counts) if c >= 10]
                appliance_names = [all_names[i] for i in kept]
                print(f"  Filtered to {len(appliance_names)} appliance names "
                      f"(dropped {len(all_names) - len(appliance_names)} rare classes)")
            else:
                appliance_names = all_names
        except Exception:
            pass

    # Final safety check
    if len(appliance_names) != n_classes:
        print(f"  Warning: name list ({len(appliance_names)}) != checkpoint classes "
              f"({n_classes}). Using generic names.")
        appliance_names = [f'Appliance_{i}' for i in range(n_classes)]

    # ------------------------------------------------------------------
    # 3. Load/Fit ICA parameters (with correct n_components from checkpoint)
    # ------------------------------------------------------------------
    print("\n[2/5] Preparing ICA parameters...")
    if args.ica_path and os.path.exists(args.ica_path):
        ica_params = load_ica_params(args.ica_path)
        # Validate dimensions match checkpoint
        if ica_params['U'].shape[0] != n_ica_components:
            print(f"  Warning: saved ICA has {ica_params['U'].shape[0]} components "
                  f"but checkpoint expects {n_ica_components}. Re-fitting...")
            ica_params = fit_ica_from_training_data(args.data_dir, n_ica_components)
            ica_save_path = os.path.join(args.data_dir, 'ica_params.npz')
            save_ica_params(ica_params, ica_save_path)
        else:
            print(f"  Loaded ICA from: {args.ica_path}")
    else:
        ica_params = fit_ica_from_training_data(args.data_dir, n_ica_components)
        # Auto-save for future use
        ica_save_path = os.path.join(args.data_dir, 'ica_params.npz')
        save_ica_params(ica_params, ica_save_path)

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    print("\n[3/5] Loading model...")
    model, ckpt_threshold = load_model(
        args.checkpoint, args.variant, n_classes, args.window_size,
        ica_params, args.device, dtype)

    threshold = args.threshold if args.threshold is not None else ckpt_threshold
    print(f"  Using threshold: {threshold:.2f}")

    # ------------------------------------------------------------------
    # 5. Load and preprocess input
    # ------------------------------------------------------------------
    print("\n[4/5] Loading and preprocessing input...")
    windows, timestamps = load_input(
        args.input, args.pre_segmented, args.sample_rate,
        args.window_size, stride)

    print(f"  Windows ready: {windows.shape}")

    # ------------------------------------------------------------------
    # 6. Run inference
    # ------------------------------------------------------------------
    print("\n[5/5] Running inference...")
    t0 = time.time()
    predictions, probabilities = run_inference(
        model, windows, threshold, args.device, dtype, args.batch_size)
    elapsed = time.time() - t0

    print(f"  Inference complete: {len(windows)} windows in {elapsed:.2f}s "
          f"({len(windows) / elapsed:.0f} windows/sec)")

    # ------------------------------------------------------------------
    # 6. Format and save results
    # ------------------------------------------------------------------
    results = format_results(
        predictions, probabilities, timestamps, appliance_names, args.top_k)

    print_results_summary(results, appliance_names, predictions)

    # Compute window duration
    if timestamps is not None and len(timestamps) > 1:
        window_duration_s = float(timestamps[1] - timestamps[0])
    else:
        window_duration_s = 10.0 / 60.0  # default: 10 cycles at 60Hz

    os.makedirs(args.output, exist_ok=True)
    save_results(results, predictions, probabilities, timestamps,
                 windows, appliance_names, args.output,
                 sample_rate=args.sample_rate,
                 window_duration_s=window_duration_s)

    print(f"\nDone! Results saved to {args.output}/")
    print(f"  Run postprocessing:")
    print(f"    python postprocess_timeline.py --input {args.output}")


if __name__ == '__main__':
    main()

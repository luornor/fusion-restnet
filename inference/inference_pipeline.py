"""
Fusion-ResNet Inference Pipeline for NILM
=========================================

Inference on pre-processed features from the ESP32 hardware module.
The ESP32 handles all preprocessing (FITPS, windowing, normalization, ICA, FFT, Fryze).
This pipeline performs model inference on the pre-processed features.

Usage:
    # On pre-processed features from ESP32 (NPZ file)
    python inference_pipeline.py --checkpoint model_registry/<version>_best.pt \
        --input preprocessed_features.npz --preprocessed

    # On pre-processed features directory (batch processing)
    python inference_pipeline.py --checkpoint model_registry/<version>_best.pt \
        --input /path/to/features/dir --preprocessed --device cpu --fp32

    # With anomaly detection and measured current
    python inference_pipeline.py --checkpoint model_registry/<version>_best.pt --input features.npz \
        --preprocessed --enable-anomaly-detection --measured-current 5.2 --device cpu

Legacy usage (for evaluation on raw signals - requires preprocessing):
    # On a CSV file with 'Current' column
    python inference_pipeline.py --checkpoint model_registry/<version>_best.pt \
        --input recording.csv --sample-rate 30000

    # On pre-segmented raw windows
    python inference_pipeline.py --checkpoint model_registry/<version>_best.pt \
        --input windows.npy --pre-segmented
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

# Allow running from project root or from inference/ subdirectory
_TRAINING = Path(__file__).resolve().parent.parent / "training"
if _TRAINING.exists() and str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import numpy as np
import pandas as pd
import torch

from fusion_resnet import (
    FusionResNet,
    FusionResNetLite,
    build_split_model_from_full_checkpoint,
)
from anomaly_detector import AnomalyDetector
from temporal_smoother import build_smoother_from_args

warnings.filterwarnings('ignore')

# Default PLAID appliance names
DEFAULT_APPLIANCE_NAMES = [
    'Air Conditioner', 'Blender', 'Coffee maker',
    'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Laptop',
    'Microwave', 'Soldering Iron', 'Vacuum', 'Washing Machine',
    'Water kettle',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fusion-ResNet Inference Pipeline for NILM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file: .npz (preprocessed), .csv (raw), .npy (raw), or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                        help='Output directory for results')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Input is pre-processed features from ESP32 (.npz with '
                             'raw_window, fft_magnitude, fryze_active, fryze_reactive, ica_features)')
    parser.add_argument('--sample-rate', type=int, default=30000,
                        help='Sampling rate of raw input data in Hz (default: 30000). '
                             'Ignored if --preprocessed is used.')
    parser.add_argument('--mains-freq', type=int, default=50,
                        help='Mains grid frequency in Hz (default: 50 for Ghana/Europe; '
                             'use 60 for US/Canada). Controls how many raw samples '
                             'make up one window (10 cycles).')
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
                        help='(Legacy) Input is already segmented into windows (N, 400). '
                             'Use --preprocessed for ESP32 features.')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory with training metadata (for label names fallback)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Only show top-k most confident appliances per window')
    # -- Temporal smoothing ------------------------------------------------
    parser.add_argument('--enable-temporal-smoothing', action='store_true',
                        help='Apply EMA + hysteresis temporal postprocessing to raw probabilities')
    parser.add_argument('--temporal-alpha', type=float, default=0.3,
                        help='EMA decay coefficient (0=max smoothing, 1=no smoothing; default 0.3)')
    parser.add_argument('--temporal-on-threshold', type=float, default=0.45,
                        help='Smoothed probability to switch appliance ON (default 0.45)')
    parser.add_argument('--temporal-off-threshold', type=float, default=0.30,
                        help='Smoothed probability to switch appliance OFF (default 0.30)')
    parser.add_argument('--temporal-min-on-windows', type=int, default=3,
                        help='Minimum consecutive ON windows to keep an event (default 3)')
    parser.add_argument('--temporal-min-off-windows', type=int, default=2,
                        help='Minimum consecutive OFF windows; shorter gaps are bridged (default 2)')
    parser.add_argument('--temporal-config', type=str, default=None,
                        help='Path to JSON with class-specific temporal config overrides')
    # -- Anomaly detection -------------------------------------------------
    parser.add_argument('--enable-anomaly-detection', action='store_true',
                        help='Enable fault detection and anomaly monitoring')
    parser.add_argument('--anomaly-history', type=str, default=None,
                        help='Path to save/load anomaly detection history')
    parser.add_argument('--measured-current', type=float, default=None,
                        help='Measured total RMS current (from PZEM or sensor) for anomaly detection')
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


def load_preprocessed_features(input_path: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """
    Load pre-processed features from ESP32.

    Expected .npz structure:
        - raw_window: (N, 400) - normalized current waveform
        - fft_magnitude: (N, 200) - FFT magnitude spectrum
        - fryze_active: (N, 50) - active power component
        - fryze_reactive: (N, 50) - reactive power component
        - ica_features: (N, 16) - ICA-decomposed components
        - timestamps (optional): (N,) - Unix timestamps
        - window_ids (optional): (N,) - window identifiers

    Returns:
        features: Dict with keys [raw_window, fft_magnitude, fryze_active, fryze_reactive, ica_features]
                  Each value is (N, size) numpy array
        timestamps: (N,) array of timestamps, or None if not in file
    """
    path = Path(input_path)

    if path.is_dir():
        # If input_path is a directory, concatenate all .npz files
        all_features = {
            'raw_window': [],
            'fft_magnitude': [],
            'fryze_active': [],
            'fryze_reactive': [],
            'ica_features': [],
        }
        all_timestamps = []

        for f in sorted(path.glob('*.npz')):
            feats, ts = load_preprocessed_features(str(f))
            for k in all_features:
                all_features[k].append(feats[k])
            if ts is not None:
                all_timestamps.extend(ts)

        # Concatenate
        for k in all_features:
            all_features[k] = np.concatenate(all_features[k])

        timestamps = np.array(all_timestamps) if all_timestamps else None
        print(f"  Loaded {len(all_features['raw_window'])} preprocessed windows from {path}")
        return all_features, timestamps

    print(f"  Loading preprocessed features: {input_path}")

    # Load .npz file
    data = np.load(input_path)

    features = {
        'raw_window': data['raw_window'].astype(np.float64),
        'fft_magnitude': data['fft_magnitude'].astype(np.float64),
        'fryze_active': data['fryze_active'].astype(np.float64),
        'fryze_reactive': data['fryze_reactive'].astype(np.float64),
        'ica_features': data['ica_features'].astype(np.float64),
    }

    # Load timestamps if available
    timestamps = data['timestamps'] if 'timestamps' in data.files else None

    # Validate shapes
    n = features['raw_window'].shape[0]
    assert features['raw_window'].shape == (n, 400), \
        f"raw_window shape {features['raw_window'].shape}, expected (N, 400)"
    assert features['fft_magnitude'].shape == (n, 200), \
        f"fft_magnitude shape {features['fft_magnitude'].shape}, expected (N, 200)"
    assert features['fryze_active'].shape == (n, 50), \
        f"fryze_active shape {features['fryze_active'].shape}, expected (N, 50)"
    assert features['fryze_reactive'].shape == (n, 50), \
        f"fryze_reactive shape {features['fryze_reactive'].shape}, expected (N, 50)"
    assert features['ica_features'].shape == (n, 16), \
        f"ica_features shape {features['ica_features'].shape}, expected (N, 16)"

    print(f"  Loaded {n} preprocessed windows")
    return features, timestamps



# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(checkpoint_path: str, variant: str, device: str,
               dtype: torch.dtype) -> tuple:
    """Load a trained Fusion-ResNet model from checkpoint.

    Returns:
        model: Loaded model in eval mode
        threshold: Classification threshold from training
        checkpoint: Loaded checkpoint dict
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = ckpt['model_state_dict']

    classifier_weight_key = [k for k in state_dict
                             if 'classifier' in k and 'weight' in k][-1]
    n_classes = state_dict[classifier_weight_key].shape[0]
    signal_length = int(ckpt.get('signal_length', 400))

    has_ica = 'ica_branch.ica.U' in state_dict
    model_kwargs = {
        'n_classes': n_classes,
        'signal_length': signal_length,
    }
    if has_ica:
        model_kwargs.update({
            'U': state_dict['ica_branch.ica.U'].cpu().numpy(),
            'M': state_dict['ica_branch.ica.M'].cpu().numpy(),
            'm': state_dict['ica_branch.norm.m'].cpu().numpy(),
            's': state_dict['ica_branch.norm.s'].cpu().numpy(),
        })

    ModelClass = FusionResNet if variant == 'full' else FusionResNetLite

    model = ModelClass(**model_kwargs)

    if dtype == torch.float64:
        model = model.double()
    else:
        model = model.float()

    model = model.to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    threshold = ckpt.get('threshold', 0.5)
    epoch = ckpt.get('epoch', '?')
    best_f1 = ckpt.get('best_val_f1', 'N/A')

    print(f"  Loaded checkpoint: epoch {epoch}, val F1 = {best_f1}")

    model.eval()
    return model, threshold, ckpt


def load_split_model(checkpoint_path: str, device: str,
                     dtype: torch.dtype) -> tuple:
    """Load the hardware-aligned split-head model from a full checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model, ica_params = build_split_model_from_full_checkpoint(
        ckpt, dtype=dtype, device=device)
    threshold = ckpt.get('threshold', 0.5)
    epoch = ckpt.get('epoch', '?')
    best_f1 = ckpt.get('best_val_f1', 'N/A')

    print(f"  Loaded split checkpoint: epoch {epoch}, val F1 = {best_f1}")
    return model, threshold, ckpt, ica_params


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


def run_inference_preprocessed(model, feature_tensors: tuple,
                               threshold: float, device: str, dtype: torch.dtype,
                               batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on pre-processed features from ESP32.

    Args:
        model: Trained model in eval mode (expects raw signal input)
        feature_tensors: Tuple of 5 tensors
            (raw_window, fft_magnitude, fryze_active, fryze_reactive, ica_features)
            Each of shape (N, feature_size)
        threshold: Classification threshold
        device: 'cuda' or 'cpu'
        dtype: torch.float32 or torch.float64
        batch_size: Batch size for inference

    Returns:
        predictions: (N, n_classes) binary predictions
        probabilities: (N, n_classes) sigmoid probabilities
    """
    raw_window, fft_magnitude, fryze_active, fryze_reactive, ica_features = feature_tensors
    n_samples = raw_window.shape[0]
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            raw_batch = raw_window[i:end_idx]
            fft_batch = fft_magnitude[i:end_idx]
            fryze_a_batch = fryze_active[i:end_idx]
            fryze_r_batch = fryze_reactive[i:end_idx]
            ica_batch = ica_features[i:end_idx]
            logits = model(
                raw_batch, fft_batch, fryze_a_batch, fryze_r_batch, ica_batch)
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
               stride: int = None,
               mains_freq: int = 50) -> tuple[np.ndarray, np.ndarray | None]:
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
        signal, sample_rate, window_size, stride, mains_freq=mains_freq)
    windows = normalize_windows(windows)

    print(f"  Segmented into {len(windows)} windows")
    return windows, timestamps


# ==============================================================================
# Result Formatting & Output
# ==============================================================================

def format_results(predictions: np.ndarray, probabilities: np.ndarray,
                   timestamps: np.ndarray | None,
                   appliance_names: list[str],
                   anomaly_detector: AnomalyDetector | None = None,
                   measured_current: float | None = None,
                   top_k: int = None,
                   smoothed_probs: np.ndarray | None = None,
                   temporal_active: np.ndarray | None = None) -> list[dict]:
    """Format predictions into human-readable results.

    Returns:
        List of dicts, one per window, with active appliances, confidences, and anomalies.
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

        if temporal_active is not None:
            temporal_indices = np.where(temporal_active[i] == 1)[0]
            entry['temporal_active_appliances'] = [
                appliance_names[j] for j in temporal_indices
            ]
            entry['n_temporal_active'] = len(temporal_indices)

        if timestamps is not None:
            entry['time_start_s'] = float(timestamps[i])
            window_time = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
            entry['timestamp'] = window_time.isoformat()
        else:
            window_time = None

        # Check for anomalies
        if anomaly_detector is not None:
            anomalies = anomaly_detector.check_window(
                predictions=predictions[i],
                probabilities=probabilities[i],
                measured_current=measured_current,
                timestamp=window_time,
            )
            if anomalies:
                entry['anomalies'] = [a.to_dict() for a in anomalies]

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
        
        # Show anomalies if present
        if 'anomalies' in result and result['anomalies']:
            for anomaly in result['anomalies']:
                print(f"    ⚠️  [{anomaly['severity'].upper()}] {anomaly['anomaly_type']}: {anomaly['message']}")

    # Summary of anomalies
    total_anomalies = sum(len(r.get('anomalies', [])) for r in results)
    if total_anomalies > 0:
        print(f"\n  Anomalies Detected: {total_anomalies} total")
        anomaly_counts = {}
        for result in results:
            for anomaly in result.get('anomalies', []):
                a_type = anomaly['anomaly_type']
                anomaly_counts[a_type] = anomaly_counts.get(a_type, 0) + 1
        for a_type, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            print(f"    - {a_type}: {count}")

    print(f"{'='*60}")


def create_mobile_payload(results: list[dict], appliance_names: list[str],
                         predictions: np.ndarray, probabilities: np.ndarray,
                         model_version: str = "1.0",
                         inference_time_ms: float = 0.0) -> dict:
    """
    Create a standardized payload for mobile client (app/dashboard).

    This formats the inference results into a clean JSON structure
    suitable for real-time visualization and alerts on mobile devices.

    Args:
        results: List of per-window results from format_results()
        appliance_names: List of appliance names
        predictions: (N, n_classes) binary predictions
        probabilities: (N, n_classes) confidence scores
        model_version: Model version string
        inference_time_ms: Inference time for this batch in milliseconds

    Returns:
        Dict with aggregated predictions, anomalies, and health info
    """
    n_windows = len(results)

    # Aggregate appliance activity
    appliance_stats = {}
    for i, name in enumerate(appliance_names):
        detections = predictions[:, i].sum()
        detection_rate = detections / n_windows * 100 if n_windows > 0 else 0
        avg_confidence = probabilities[predictions[:, i] == 1, i].mean() \
            if (predictions[:, i] == 1).any() else 0.0

        appliance_stats[name] = {
            'detected': bool(detections > 0),
            'detection_rate': float(detection_rate),
            'detections': int(detections),
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(probabilities[:, i].max()),
            'min_confidence': float(probabilities[:, i].min()),
        }

    # Aggregate anomalies
    all_anomalies = []
    anomaly_counts = {}
    for result in results:
        if 'anomalies' in result:
            for anomaly in result['anomalies']:
                all_anomalies.append({
                    'window': result['window'],
                    'type': anomaly['anomaly_type'],
                    'severity': anomaly['severity'],
                    'message': anomaly['message'],
                    'timestamp': result.get('timestamp', None),
                })
                a_type = anomaly['anomaly_type']
                anomaly_counts[a_type] = anomaly_counts.get(a_type, 0) + 1

    # Create mobile payload
    payload = {
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': model_version,
            'variant': 'preprocessed',  # Input is pre-processed features
            'n_windows': n_windows,
            'inference_time_ms': float(inference_time_ms),
        },
        'summary': {
            'most_active_appliance': max(
                appliance_stats.items(),
                key=lambda x: x[1]['detection_rate']
            )[0] if appliance_stats else None,
            'avg_active_per_window': float(predictions.sum(axis=1).mean()),
            'total_anomalies': len(all_anomalies),
            'anomaly_types': anomaly_counts,
        },
        'appliances': appliance_stats,
        'anomalies': all_anomalies,
        'recent_windows': results[:min(10, n_windows)],  # Last 10 for debugging
    }

    return payload


def save_results(results: list[dict], predictions: np.ndarray,
                 probabilities: np.ndarray, timestamps: np.ndarray | None,
                 windows: np.ndarray, appliance_names: list[str],
                 output_dir: str, sample_rate: int = 0,
                 window_duration_s: float = 0.0,
                 smoothed_probs: np.ndarray | None = None,
                 temporal_active: np.ndarray | None = None):
    """Save inference results to files.

    When temporal smoothing was applied, also saves:
      smoothed_probabilities.npy  (N, C) EMA-smoothed probs
      temporal_active.npy         (N, C) binary after full postprocessing pipeline
    The CSV gains {name}_smoothed_prob and {name}_temporal_active columns.
    Raw predictions are always preserved for comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump({
            'n_windows': len(results),
            'appliance_names': appliance_names,
            'temporal_smoothing_applied': temporal_active is not None,
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
            row[f'{name}_raw_prob'] = float(probabilities[i, j])
            row[f'{name}_raw_active'] = int(predictions[i, j])
            if smoothed_probs is not None:
                row[f'{name}_smoothed_prob'] = float(smoothed_probs[i, j])
            if temporal_active is not None:
                row[f'{name}_temporal_active'] = int(temporal_active[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save raw arrays (always)
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(output_dir, 'windows.npy'), windows)
    if timestamps is not None:
        np.save(os.path.join(output_dir, 'timestamps.npy'), timestamps)

    # Save temporal arrays when smoothing was applied
    if smoothed_probs is not None:
        np.save(os.path.join(output_dir, 'smoothed_probabilities.npy'), smoothed_probs)
    if temporal_active is not None:
        np.save(os.path.join(output_dir, 'temporal_active.npy'), temporal_active)

    # Save metadata for postprocessor
    meta = {
        'n_windows': int(predictions.shape[0]),
        'n_classes': int(predictions.shape[1]),
        'window_size': int(windows.shape[1]),
        'sample_rate': int(sample_rate),
        'window_duration_s': float(window_duration_s),
        'appliance_names': appliance_names,
        'has_timestamps': timestamps is not None,
        'temporal_smoothing_applied': temporal_active is not None,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    suffix = " + temporal smoothing" if temporal_active is not None else ""
    print(f"  Saved numpy arrays{suffix} + metadata to {output_dir}/")


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
    print(f"  Input Mode: {'PRE-PROCESSED (ESP32)' if args.preprocessed else 'RAW SIGNAL'}")
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
    appliance_names = [str(name) for name in ckpt.get('appliance_names', [])]
    if not appliance_names:
        lenc_path = os.path.join(args.data_dir, 'real_label_encoder.npy')
        if os.path.exists(lenc_path):
            try:
                lenc = np.load(lenc_path, allow_pickle=True).item()
                all_names = list(lenc.classes_)
                kept_class_ids = ckpt.get('kept_class_ids')
                if kept_class_ids:
                    appliance_names = [str(all_names[idx]) for idx in kept_class_ids]
                elif len(all_names) != n_classes:
                    y_real = np.load(os.path.join(args.data_dir, 'y_real.npy'),
                                     allow_pickle=True)
                    class_ids, counts = np.unique(y_real, return_counts=True)
                    kept = [int(cls) for cls, count in zip(class_ids, counts) if count >= 10]
                    appliance_names = [str(all_names[idx]) for idx in kept]
                else:
                    appliance_names = [str(name) for name in all_names]
            except Exception:
                pass

    # Final safety check
    if len(appliance_names) != n_classes:
        print(f"  Warning: name list ({len(appliance_names)}) != checkpoint classes "
              f"({n_classes}). Using generic names.")
        appliance_names = [f'Appliance_{i}' for i in range(n_classes)]

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print("\n[2/5] Loading model...")
    if args.preprocessed:
        model, ckpt_threshold, ckpt, _ = load_split_model(
            args.checkpoint, args.device, dtype)
    else:
        model, ckpt_threshold, ckpt = load_model(
            args.checkpoint, args.variant, args.device, dtype)

    threshold = args.threshold if args.threshold is not None else ckpt_threshold
    print(f"  Using threshold: {threshold:.2f}")

    # ------------------------------------------------------------------
    # 4. Load and preprocess input
    # ------------------------------------------------------------------
    print("\n[3/5] Loading input...")
    
    if args.preprocessed:
        # Load pre-processed features from ESP32
        features, timestamps = load_preprocessed_features(args.input)
        
        # Convert features to model input tensors
        raw_window = torch.tensor(features['raw_window'], dtype=dtype, device=args.device)
        fft_magnitude = torch.tensor(features['fft_magnitude'], dtype=dtype, device=args.device)
        fryze_active = torch.tensor(features['fryze_active'], dtype=dtype, device=args.device)
        fryze_reactive = torch.tensor(features['fryze_reactive'], dtype=dtype, device=args.device)
        ica_features = torch.tensor(features['ica_features'], dtype=dtype, device=args.device)
        
        feature_tensors = (raw_window, fft_magnitude, fryze_active, fryze_reactive, ica_features)
        n_windows = raw_window.shape[0]
        print(f"  Pre-processed features ready: {n_windows} windows")
    else:
        # Load raw signal and preprocess
        print(f"  Mains frequency: {args.mains_freq} Hz")
        windows, timestamps = load_input(
            args.input, args.pre_segmented, args.sample_rate,
            args.window_size, stride, mains_freq=args.mains_freq)

        print(f"  Windows ready: {windows.shape}")
        feature_tensors = None

    # ------------------------------------------------------------------
    # 5. Initialize anomaly detection (optional)
    # ------------------------------------------------------------------
    anomaly_detector = None
    if args.enable_anomaly_detection:
        print("\n[4/5] Initializing anomaly detection...")
        history_path = args.anomaly_history or os.path.join(args.output, '.anomaly_history.json')
        anomaly_detector = AnomalyDetector(appliance_names, history_file=history_path)
        print(f"  Anomaly detector ready")
        if args.measured_current:
            print(f"  Using measured current for validation: {args.measured_current:.2f}A")

    print(f"\n[5/5] Running inference...")
    t0 = time.time()
    
    if args.preprocessed:
        # Inference on pre-processed features
        predictions, probabilities = run_inference_preprocessed(
            model, feature_tensors, threshold, args.device, dtype, args.batch_size)
        windows = feature_tensors[0].cpu().numpy()  # For saving
    else:
        # Inference on raw windows
        predictions, probabilities = run_inference(
            model, windows, threshold, args.device, dtype, args.batch_size)
    
    elapsed = time.time() - t0

    print(f"  Inference complete: {len(predictions)} windows in {elapsed:.2f}s "
          f"({len(predictions) / elapsed:.0f} windows/sec)")

    # ------------------------------------------------------------------
    # 6. Temporal smoothing (optional)
    # ------------------------------------------------------------------
    smoothed_probs = None
    temporal_active = None

    if args.enable_temporal_smoothing:
        print("\n[TS] Applying temporal smoothing...")
        smoother = build_smoother_from_args(
            appliance_names=appliance_names,
            config_path=args.temporal_config,
            alpha=args.temporal_alpha,
            on_threshold=args.temporal_on_threshold,
            off_threshold=args.temporal_off_threshold,
            min_on_windows=args.temporal_min_on_windows,
            min_off_windows=args.temporal_min_off_windows,
        )
        print(smoother.describe())
        smoothed_probs, temporal_active = smoother.smooth(probabilities)

        raw_on = int(predictions.sum())
        temp_on = int(temporal_active.sum())
        print(f"  Raw active windows: {raw_on:,}  ->  Temporal: {temp_on:,} "
              f"({(temp_on - raw_on) / max(raw_on, 1) * 100:+.1f}%)")

    # ------------------------------------------------------------------
    # 7. Format and save results
    # ------------------------------------------------------------------
    results = format_results(
        predictions, probabilities, timestamps, appliance_names,
        anomaly_detector=anomaly_detector,
        measured_current=args.measured_current,
        top_k=args.top_k,
        smoothed_probs=smoothed_probs,
        temporal_active=temporal_active)

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
                 window_duration_s=window_duration_s,
                 smoothed_probs=smoothed_probs,
                 temporal_active=temporal_active)

    # Save mobile payload
    mobile_payload = create_mobile_payload(
        results, appliance_names, predictions, probabilities,
        model_version="1.0",
        inference_time_ms=elapsed * 1000)
    
    mobile_payload_path = os.path.join(args.output, 'mobile_payload.json')
    with open(mobile_payload_path, 'w') as f:
        json.dump(mobile_payload, f, indent=2)
    print(f"  Saved: {mobile_payload_path}")

    print(f"\nDone! Results saved to {args.output}/")
    print(f"  Run postprocessing:")
    print(f"    python postprocess_timeline.py --input {args.output}")


if __name__ == '__main__':
    main()

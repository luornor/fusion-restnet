"""
Fusion-ResNet NILM — Diagnostic Script
=======================================

Extracts training history from checkpoints, computes class/signature statistics,
generates calibration plots, and saves all diagnostic outputs to
reports/ml_diagnostics/.

Usage:
    # Basic: extract curves + class counts
    python generate_diagnostics.py

    # With calibration analysis (requires checkpoint with full model weights)
    python generate_diagnostics.py --calibrate

    # Custom paths
    python generate_diagnostics.py \
        --checkpoints-dir checkpoints \
        --data-dir data \
        --output-dir reports/ml_diagnostics \
        --metrics-dir figures
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

# Allow running from project root or from diagnostics/ subdirectory
_TRAINING = Path(__file__).resolve().parent.parent / "training"
if _TRAINING.exists() and str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints-dir', default='model_registry')
    p.add_argument('--data-dir', default='data/plaid')
    p.add_argument('--output-dir', default='reports/ml_diagnostics')
    p.add_argument('--metrics-dir', default='figures',
                   help='Directory containing test_metrics.json')
    p.add_argument('--metrics-dir-es', default='figures_earlystop-v1',
                   help='Directory for early-stop variant metrics')
    p.add_argument('--calibrate', action='store_true',
                   help='Run calibration analysis (needs checkpoint + data)')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


# ==============================================================================
# 1. Load checkpoint history
# ==============================================================================

def load_checkpoint_history(ckpt_path: str, device: str = 'cpu') -> dict | None:
    """Load history dict from a checkpoint file."""
    if not os.path.exists(ckpt_path):
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        history = ckpt.get('history')
        if history is None:
            print(f"  [warn] No 'history' key in {ckpt_path}")
            return None
        meta = {
            'epoch': ckpt.get('epoch', '?'),
            'best_val_f1': ckpt.get('best_val_f1', '?'),
            'threshold': ckpt.get('threshold', '?'),
            'early_stopped': ckpt.get('early_stopped', False),
            'model_version': ckpt.get('model_version', 'unknown'),
            'n_classes': ckpt.get('n_classes', '?'),
            'appliance_names': ckpt.get('appliance_names', []),
        }
        print(f"  Loaded: {os.path.basename(ckpt_path)} | "
              f"epoch={meta['epoch']} | best_val_f1={meta['best_val_f1']:.4f} "
              f"| threshold={meta['threshold']:.4f}")
        return {'history': history, 'meta': meta}
    except Exception as e:
        print(f"  [error] Could not load {ckpt_path}: {e}")
        return None


# ==============================================================================
# 2. Training Curve Plots
# ==============================================================================

def plot_training_comparison(runs: dict[str, dict], output_dir: str):
    """Plot training curves for one or multiple runs on the same axes."""
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (label, run) in enumerate(runs.items()):
        if run is None:
            continue
        h = run['history']
        color = colors[idx % len(colors)]
        n_epochs = len(h['train']['loss'])
        epochs = range(1, n_epochs + 1)

        axes[0, 0].plot(epochs, h['train']['loss'], color=color, lw=2,
                        alpha=0.85, label=f'{label} Train')
        axes[0, 0].plot(epochs, h['val']['loss'], color=color, lw=2,
                        alpha=0.55, linestyle='--', label=f'{label} Val')

        axes[0, 1].plot(epochs, h['train']['score'], color=color, lw=2,
                        alpha=0.85, label=f'{label} Train')
        axes[0, 1].plot(epochs, h['val']['score'], color=color, lw=2,
                        alpha=0.55, linestyle='--', label=f'{label} Val')

        axes[1, 0].plot(epochs, h['lr'], color=color, lw=2, label=label)

        if 'threshold' in h:
            axes[1, 1].plot(epochs, h['threshold'], color=color, lw=2, label=label)

    axes[0, 0].set_title('Loss (Train solid, Val dashed)')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('BCE Loss')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title('F1 (samples) — Train solid, Val dashed')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('F1')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title('Optimized Val Threshold per Epoch')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Threshold')
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Fusion-ResNet — Training Diagnostics (All Runs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_training_history_csv(run_label: str, run_data: dict, output_dir: str):
    """Save per-epoch training history as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    h = run_data['history']
    n = len(h['train']['loss'])
    rows = {
        'epoch': list(range(1, n + 1)),
        'train_loss': h['train']['loss'],
        'val_loss': h['val']['loss'],
        'train_f1': h['train']['score'],
        'val_f1': h['val']['score'],
        'lr': h.get('lr', [None] * n),
        'threshold': h.get('threshold', [None] * n),
    }
    df = pd.DataFrame(rows)
    fname = f'training_history_{run_label}.csv'
    path = os.path.join(output_dir, fname)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ==============================================================================
# 3. Class / Signature Statistics
# ==============================================================================

def compute_class_statistics(data_dir: str, output_dir: str) -> pd.DataFrame | None:
    """Count raw PLAID signatures per appliance class."""
    x_path = os.path.join(data_dir, 'X_real.npy')
    y_path = os.path.join(data_dir, 'y_real.npy')
    enc_path = os.path.join(data_dir, 'real_label_encoder.npy')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"  [skip] Data files not found in {data_dir}")
        return None

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    class_ids, counts = np.unique(y, return_counts=True)
    total = len(y)

    # Load appliance names if encoder available
    names = {}
    if os.path.exists(enc_path):
        try:
            enc = np.load(enc_path, allow_pickle=True).item()
            names = {i: enc.classes_[i] for i in range(len(enc.classes_))}
        except Exception:
            pass

    rows = []
    for cid, cnt in zip(class_ids, counts):
        rows.append({
            'class_id': int(cid),
            'appliance': names.get(int(cid), f'class_{cid}'),
            'n_signatures': int(cnt),
            'pct_of_total': round(100 * cnt / total, 2),
            'kept': int(cnt) >= 10,
            'diversity_risk': 'HIGH' if cnt < 50 else ('MEDIUM' if cnt < 150 else 'LOW'),
        })

    df = pd.DataFrame(rows).sort_values('n_signatures')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'class_counts.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#F44336' if r < 50 else ('#FF9800' if r < 150 else '#4CAF50')
              for r in df['n_signatures']]
    ax.barh(df['appliance'], df['n_signatures'], color=colors, edgecolor='white', alpha=0.85)
    ax.axvline(50, color='red', linestyle='--', lw=1.5, label='Diversity risk threshold (50)')
    ax.axvline(150, color='orange', linestyle='--', lw=1.5, label='Medium risk (150)')
    ax.set_xlabel('Number of Raw PLAID Signatures')
    ax.set_title('Fusion-ResNet — Appliance Signature Counts in PLAID Dataset', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path_png = os.path.join(output_dir, 'class_counts.png')
    plt.savefig(path_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_png}")

    print(f"\n  Class count summary:")
    print(f"  Total signatures: {total}")
    print(f"  Classes: {len(class_ids)} (before filter), {df['kept'].sum()} after ≥10 filter")
    print(f"  Min signatures: {df['n_signatures'].min()} ({df.iloc[0]['appliance']})")
    print(f"  Max signatures: {df['n_signatures'].max()} ({df.iloc[-1]['appliance']})")
    print(f"  High diversity risk (< 50): {(df['n_signatures'] < 50).sum()} classes")
    print(f"  Medium diversity risk (50–149): {((df['n_signatures'] >= 50) & (df['n_signatures'] < 150)).sum()} classes")

    return df


# ==============================================================================
# 4. Test Metrics Summary
# ==============================================================================

def parse_and_plot_test_metrics(metrics_dirs: dict[str, str], output_dir: str):
    """Load test_metrics.json files and generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_data = {}
    for label, mdir in metrics_dirs.items():
        # Try both test_metrics.json and test_metrics_*.json
        candidates = [
            os.path.join(mdir, 'test_metrics.json'),
        ]
        for f in os.listdir(mdir) if os.path.isdir(mdir) else []:
            if f.startswith('test_metrics_') and f.endswith('.json'):
                candidates.append(os.path.join(mdir, f))

        for cand in candidates:
            if os.path.exists(cand):
                with open(cand) as fh:
                    metrics_data[label] = json.load(fh)
                print(f"  Loaded metrics: {cand}")
                break
        else:
            print(f"  [skip] No test_metrics.json in {mdir}")

    if not metrics_data:
        return

    # Per-class F1 comparison
    run_labels = list(metrics_data.keys())
    all_appliances = list(next(iter(metrics_data.values()))['per_class'].keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_appliances))
    bar_w = 0.8 / max(len(run_labels), 1)
    colors = ['#2196F3', '#F44336', '#4CAF50']

    for i, label in enumerate(run_labels):
        f1s = [metrics_data[label]['per_class'].get(a, {}).get('f1', 0) for a in all_appliances]
        offset = (i - len(run_labels) / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w * 0.9, label=label,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(all_appliances, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Appliance F1 — Run Comparison', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_f1_comparison.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Mixture complexity comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(run_labels):
        comp = metrics_data[label].get('per_n_components', {})
        nc = sorted([int(k) for k in comp.keys()])
        f1s = [comp[str(k)]['f1_samples'] for k in nc]
        ax.plot(nc, f1s, 'o-', color=colors[i % len(colors)], lw=2, ms=5, label=label)

    ax.set_xlabel('# Active Appliances')
    ax.set_ylabel('F1 (samples)')
    ax.set_title('F1 vs. Mixture Complexity — Run Comparison', fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'mixture_complexity_comparison.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Global metrics table
    global_keys = ['f1_samples', 'f1_macro', 'f1_micro', 'f1_weighted',
                   'precision_samples', 'recall_samples', 'accuracy', 'hamming_loss', 'threshold']
    rows = []
    for label in run_labels:
        row = {'run': label}
        for k in global_keys:
            row[k] = metrics_data[label].get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'global_metrics_comparison.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ==============================================================================
# 5. Calibration Analysis
# ==============================================================================

def run_calibration_analysis(
    ckpt_path: str,
    data_dir: str,
    output_dir: str,
    device: str = 'cpu',
    seed: int = 42,
    n_samples: int = 2000,
):
    """Run inference on val/test split and compute calibration diagnostics."""
    try:
        from temperature_scaling import TemperatureScaler
        from sklearn.model_selection import train_test_split
        from sklearn.decomposition import FastICA
        from sklearn.preprocessing import MultiLabelBinarizer
        from fusion_resnet import FusionResNet
        import math, sys, warnings
        from functools import reduce
    except ImportError as e:
        print(f"  [skip] Calibration requires project modules: {e}")
        return

    if not os.path.exists(ckpt_path):
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Re-build dataset (same as training script)
    X_real = np.load(os.path.join(data_dir, 'X_real.npy'), allow_pickle=True)
    y_real = np.load(os.path.join(data_dir, 'y_real.npy'), allow_pickle=True)

    class_ids, counts = np.unique(y_real, return_counts=True)
    kept = [int(c) for c, cnt in zip(class_ids, counts) if cnt >= 10]
    mask = np.isin(y_real, kept)
    X_real, y_real = X_real[mask], y_real[mask]
    n_classes = len(np.unique(y_real))

    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=seed, stratify=y_real)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.7, random_state=seed, stratify=y_test)

    # Minimal compose helper
    def compose_simple(X, y, n_cls, n_samp, seed_c):
        from train_fusion_resnet import compose
        return compose(X, y, n_cls, n_samp)

    X_val_m, Y_val_m = compose_simple(X_val, y_val, n_classes, n_samples // 4, seed)
    X_val_m /= np.where(np.abs(X_val_m).max(axis=1, keepdims=True) == 0, 1.0,
                        np.abs(X_val_m).max(axis=1, keepdims=True))

    # Load ICA params from checkpoint
    state = ckpt.get('model_state_dict', ckpt)
    U = state['ica_branch.ica.U'].cpu().numpy()
    M = state['ica_branch.ica.M'].cpu().numpy()
    m = state['ica_branch.norm.m'].cpu().numpy()
    s = state['ica_branch.norm.s'].cpu().numpy()

    dtype = torch.float32
    model = FusionResNet(n_classes=n_classes, signal_length=400,
                         U=U, M=M, m=m, s=s).float().to(device)
    model.load_state_dict(state)
    model.eval()

    # Get val logits
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X_val_m), 256):
            batch = torch.tensor(X_val_m[i:i+256], dtype=dtype, device=device)
            all_logits.append(model(batch).cpu().numpy())
    val_logits = np.concatenate(all_logits, axis=0)
    val_labels = Y_val_m.astype(np.float32)

    # Split val into cal1 (fit T) / cal2 (report)
    n_half = len(val_logits) // 2
    cal_logits, rep_logits = val_logits[:n_half], val_logits[n_half:]
    cal_labels, rep_labels = val_labels[:n_half], val_labels[n_half:]

    scaler = TemperatureScaler()
    T = scaler.fit(cal_logits, cal_labels)
    metrics = scaler.calibration_metrics(rep_logits, rep_labels)
    best_thresh = scaler.find_best_threshold(cal_logits, cal_labels)

    print(f"  Temperature T = {T:.4f}")
    print(f"  ECE before: {metrics['ece_before']:.4f} → after: {metrics['ece_after']:.4f}")
    print(f"  Best threshold (calibrated): {best_thresh:.4f}")

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'calibration_metrics.json'), 'w') as fh:
        json.dump({**metrics, 'best_threshold': best_thresh}, fh, indent=2)

    # Reliability curve
    probs_raw = 1 / (1 + np.exp(-rep_logits))
    cal_rep_logits = rep_logits / T
    probs_cal = 1 / (1 + np.exp(-cal_rep_logits))

    def reliability_curve(probs_flat, labels_flat, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, mean_acc, mean_conf, counts_b = [], [], [], []
        for i in range(n_bins):
            mask = (probs_flat >= bins[i]) & (probs_flat < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            mean_acc.append(labels_flat[mask].mean())
            mean_conf.append(probs_flat[mask].mean())
            counts_b.append(mask.sum())
        return np.array(mean_conf), np.array(mean_acc), np.array(counts_b)

    pf = probs_raw.ravel()
    lf = rep_labels.ravel()
    pf_cal = probs_cal.ravel()

    conf_r, acc_r, _ = reliability_curve(pf, lf)
    conf_c, acc_c, _ = reliability_curve(pf_cal, lf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    axes[0].plot(conf_r, acc_r, 'o-', color='#F44336', lw=2, ms=5, label='Before calibration')
    axes[0].plot(conf_c, acc_c, 's-', color='#2196F3', lw=2, ms=5, label=f'After (T={T:.3f})')
    axes[0].set_xlabel('Mean Predicted Probability'); axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Reliability Diagram', fontweight='bold'); axes[0].legend()

    axes[1].hist(pf[lf == 0], bins=30, density=True, alpha=0.6,
                 color='#F44336', label='Negatives (before)')
    axes[1].hist(pf[lf == 1], bins=30, density=True, alpha=0.6,
                 color='#2196F3', label='Positives (before)')
    axes[1].axvline(0.5, color='black', lw=1, linestyle='--')
    axes[1].axvline(float(ckpt.get('threshold', 0.38)), color='purple',
                    lw=1.5, linestyle=':', label='Used threshold')
    axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Density')
    axes[1].set_title('Probability Distribution', fontweight='bold'); axes[1].legend()

    plt.suptitle('Fusion-ResNet — Calibration Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'calibration_analysis.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Fusion-ResNet NILM — Diagnostic Report Generator")
    print("=" * 60)

    # --- 1. Load checkpoint histories ---
    print("\n[1/4] Loading checkpoint histories...")
    ckpt_map = {
        'v0.0.1-dev (300 ep)': os.path.join(args.checkpoints_dir, 'v0.0.1-dev_best.pt'),
        'earlystop-v1': os.path.join(args.checkpoints_dir, 'earlystop-v1_best.pt'),
    }
    runs = {}
    best_ckpt_path = None
    for label, path in ckpt_map.items():
        result = load_checkpoint_history(path, args.device)
        runs[label] = result
        if result is not None and best_ckpt_path is None:
            best_ckpt_path = path  # use first available as default

    active_runs = {k: v for k, v in runs.items() if v is not None}
    if not active_runs:
        print("  [warn] No valid checkpoints found. Skipping history plots.")
    else:
        print(f"\n  Plotting training curves ({len(active_runs)} run(s))...")
        plot_training_comparison(active_runs, args.output_dir)
        for label, run in active_runs.items():
            save_training_history_csv(
                label.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_'),
                run, args.output_dir)

    # --- 2. Class / signature statistics ---
    print("\n[2/4] Computing class/signature statistics...")
    compute_class_statistics(args.data_dir, args.output_dir)

    # --- 3. Test metrics plots ---
    print("\n[3/4] Generating test metrics comparison...")
    parse_and_plot_test_metrics(
        metrics_dirs={
            'v0.0.1-dev': args.metrics_dir,
            'earlystop-v1': args.metrics_dir_es,
        },
        output_dir=args.output_dir,
    )

    # --- 4. Calibration analysis ---
    if args.calibrate:
        print("\n[4/4] Running calibration analysis...")
        ckpt_for_cal = os.path.join(args.checkpoints_dir, 'latest-v0.0.1-dev.pt')
        run_calibration_analysis(
            ckpt_path=ckpt_for_cal,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        print("\n[4/4] Calibration analysis skipped (pass --calibrate to enable).")

    print(f"\nAll diagnostics saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()

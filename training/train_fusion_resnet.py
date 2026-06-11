"""
Fusion-ResNet Training Script for NILM Energy Disaggregation
=============================================================

This script trains the Fusion-ResNet model on the PLAID dataset.
Compatible with both local GPU and Google Colab.

Checkpoints are saved with semantic versioning: latest_v{version}.pt

Usage:
    # Local (with CUDA GPU) - default 300 epochs
    python train_fusion_resnet.py --device cuda --variant full --model-version 0.0.1-dev

    # With custom epochs
    python train_fusion_resnet.py --device cuda --variant full --epochs 300 --model-version 1.0.0

    # Local with limited GPU (RTX 2050 / 4GB VRAM)
    python train_fusion_resnet.py --device cuda --variant lite --batch-size 64 --fp32 --model-version 0.0.1-dev

    # CPU fallback
    python train_fusion_resnet.py --device cpu --variant lite --epochs 100 --model-version 0.0.1-dev
"""

from __future__ import annotations

import os
import sys
import math
import time
import random
import argparse
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import FastICA
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, hamming_loss, jaccard_score,
    multilabel_confusion_matrix,
)

from timm.utils import AverageMeter

from fusion_resnet import FusionResNet, FusionResNetLite, model_summary

warnings.filterwarnings('ignore')

# Global plot style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Default appliance names for PLAID (will be overridden if label_encoder exists)
DEFAULT_APPLIANCE_NAMES = [
    'Air Conditioner', 'Blender', 'Coffee maker', 'Compact Fluorescent Lamp',
    'Fan', 'Fridge', 'Hair Iron', 'Hairdryer', 'Heater',
    'Incandescent Light Bulb', 'Laptop', 'Microwave', 'Soldering Iron',
    'Vacuum', 'Washing Machine', 'Water kettle',
]

# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fusion-ResNet for NILM')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'lite'], help='Model variant')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--model-version', type=str, default='0.0.1-dev',
                        help='Model version tag (e.g., 0.0.1-dev, 1.0.0, 1.0.0-rc1)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fp32', action='store_true',
                        help='(Deprecated — float32 is now the default. Has no effect.)')
    parser.add_argument('--fp64', action='store_true',
                        help='Use float64 instead of float32 (higher precision, more VRAM)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/plaid', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='model_registry',
                        help='Checkpoint save directory')
    parser.add_argument('--figures-dir', type=str, default='reports/figures',
                        help='Directory to save plots and test metrics')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Samples per component for mixture generation')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from a checkpoint path')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save last.pt every N epochs (default: 1)')
    parser.add_argument('--snapshot-every', type=int, default=25,
                        help='Also save epoch snapshots every N epochs (default: 25, 0 disables)')
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Stop if validation F1 does not improve for N epochs (0 disables, recommended: 40)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Minimum validation F1 improvement to reset early stopping')
    # --- Optimizer / scheduler ---
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 1e-4; try 5e-4 or 1e-3 with synthetic data)')
    parser.add_argument('--scheduler-factor', type=float, default=0.8,
                        help='ReduceLROnPlateau LR reduction factor (default: 0.8; try 0.5)')
    parser.add_argument('--scheduler-patience', type=int, default=15,
                        help='ReduceLROnPlateau patience in epochs (default: 15; try 10)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Linear LR warmup epochs from --warmup-start-lr to --lr (0=disabled; recommended: 5)')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-5,
                        help='Starting LR for linear warmup (default: 1e-5)')
    # --- Loss function ---
    parser.add_argument('--use-pos-weight', action='store_true',
                        help='Weight BCEWithLogitsLoss by inverse class frequency in training labels')
    # --- Source-level augmentation (applied before mixture composition) ---
    parser.add_argument('--aug-noise-sigma', type=float, default=0.0,
                        help='Gaussian noise sigma as fraction of signal RMS (0=off; try 0.02)')
    parser.add_argument('--aug-amplitude-scale', type=float, default=0.0,
                        help='Max amplitude perturbation fraction (0=off; try 0.15)')
    parser.add_argument('--aug-phase-shift', action='store_true',
                        help='Random cyclic phase shift of source signatures')
    parser.add_argument('--aug-time-warp', action='store_true',
                        help='Time-warping ±5%% on source signatures (optional; adds compute)')
    # --- Calibration ---
    parser.add_argument('--calibrate', action='store_true',
                        help='Post-training temperature scaling calibration on validation set')
    # --- Grid parameters ---
    parser.add_argument('--mains-freq', type=float, default=60.0,
                        help='Mains grid frequency in Hz (default: 60 for PLAID/US; use 50 for Ghana/Europe)')
    parser.add_argument('--mains-volt', type=float, default=120.0,
                        help='Mains RMS voltage in V (default: 120 for PLAID/US; use 230 for Ghana/Europe)')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Data Pipeline (from original project)
# ==============================================================================

class SignatureAugmentor:
    """Applies configurable augmentations to individual source signatures.

    Called per-signature BEFORE mixture composition, so each training mixture
    contains uniquely perturbed source waveforms.  Labels are never modified.
    Validation and test data must NOT use augmentation.

    All augmentations are independently configurable and off by default.
    """

    def __init__(
        self,
        noise_sigma: float = 0.0,
        amplitude_scale: float = 0.0,
        phase_shift: bool = False,
        time_warp: bool = False,
        seed: int = 42,
    ):
        self.noise_sigma = noise_sigma
        self.amplitude_scale = amplitude_scale
        self.phase_shift = phase_shift
        self.time_warp = time_warp
        self._rng = np.random.RandomState(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        rms = float(np.sqrt(np.mean(x ** 2))) + 1e-8

        if self.noise_sigma > 0:
            x = x + self._rng.normal(0, self.noise_sigma * rms, size=x.shape)

        if self.amplitude_scale > 0:
            scale = 1.0 + self._rng.uniform(-self.amplitude_scale, self.amplitude_scale)
            x = x * scale

        if self.phase_shift:
            shift = self._rng.randint(0, len(x))
            x = np.roll(x, shift)

        if self.time_warp:
            factor = 1.0 + self._rng.uniform(-0.05, 0.05)
            n = len(x)
            new_n = max(1, int(round(n * factor)))
            x_warped = np.interp(
                np.linspace(0, n - 1, new_n),
                np.arange(n), x,
            )
            x = x_warped[:n] if new_n >= n else np.pad(x_warped, (0, n - new_n), mode='edge')

        return x

    def is_active(self) -> bool:
        return (self.noise_sigma > 0 or self.amplitude_scale > 0
                or self.phase_shift or self.time_warp)

    def summary(self) -> str:
        parts = []
        if self.noise_sigma > 0:
            parts.append(f'noise_sigma={self.noise_sigma}')
        if self.amplitude_scale > 0:
            parts.append(f'amplitude_scale={self.amplitude_scale}')
        if self.phase_shift:
            parts.append('phase_shift=True')
        if self.time_warp:
            parts.append('time_warp=True')
        return ', '.join(parts) if parts else 'none'


class Composer:
    """Generates mixture signals from individual appliance signatures."""

    def __init__(self, X, y, random_state=None, augmentor=None):
        self._X = X
        self._y = y
        self._classes = np.unique(y)
        self._domains = {}
        for l in self._classes:
            domain = np.argwhere(y == l).ravel().tolist()
            self._domains[l] = domain
        if random_state is not None:
            seed_shift = round(np.sum(np.std(X, axis=1)))
            modified_seed = random_state + seed_shift
        else:
            modified_seed = random_state
        self._rng = np.random.RandomState(modified_seed)
        self._augmentor = augmentor

    @property
    def classes(self):
        return self._classes

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def domains(self):
        return self._domains

    def make_index_set(self, n_samples=100, n_classes=2,
                       min_freqs=None, max_freqs=None):
        n_combs_max = math.comb(self.n_classes, n_classes)
        n_combs = min(n_samples, n_combs_max)
        Y = set()
        while len(Y) < n_combs:
            comb = self._rng.choice(self.classes, n_classes, replace=False)
            comb = tuple(sorted(comb))
            if comb not in Y:
                Y.add(comb)
        Y = list(map(list, Y))

        if min_freqs is None:
            min_freqs = np.ones(n_classes)
        if max_freqs is None:
            max_freqs = np.ones(n_classes)

        F = self._rng.randint(min_freqs, max_freqs + 1, size=(n_samples, n_classes))

        n_reps = n_samples // len(Y)
        n_rem = n_samples % len(Y)
        rep_distr = np.asarray([n_reps] * len(Y))
        to_rep = self._rng.choice(range(len(rep_distr)), size=n_rem, replace=False)
        rep_distr[to_rep] += 1

        I = set()
        for i, n_rep in enumerate(rep_distr):
            y = Y[i]
            f = F[i]
            D = [self.domains[l] for l in y]
            n_max = reduce(
                lambda x, y: x * y,
                [math.comb(len(D[j]) + f[j] - 1, f[j]) for j in range(len(D))])
            Ii = set()
            while len(Ii) < min(n_rep, min(n_max, sys.maxsize)):
                sample = []
                for domain, freq in zip(D, f):
                    sample += self._rng.choice(domain, size=freq, replace=True).tolist()
                sample = tuple(sorted(sample))
                if sample not in Ii:
                    Ii.add(sample)
            I = I.union(Ii)

        dn = n_samples - len(I)
        if dn > 0:
            warnings.warn(f'{dn} samples were not obtained due to combinatorial limit.')

        return list(map(list, I))

    def compose_single(self, Ii):
        Ii = np.asarray(Ii)
        x = self._X[Ii].copy()
        y = self._y[np.asarray(Ii)]
        if self._augmentor is not None and self._augmentor.is_active():
            x = np.stack([self._augmentor(xi) for xi in x])
        x = np.sum(x, axis=0)
        y = np.unique(y)
        return x, y

    def make_samples(self, n_samples=100, n_classes=2,
                     min_freqs=None, max_freqs=None):
        I = self.make_index_set(n_samples=n_samples, n_classes=n_classes,
                                min_freqs=min_freqs, max_freqs=max_freqs)
        X, Y = [], []
        for Ii in I:
            x, y = self.compose_single(Ii)
            X.append(x)
            Y.append(y)
        return X, Y


def compose(X, y, n_classes, n_samples_per_component, n_min=1, n_max=None,
            min_freqs=1, max_freqs=10, share=1.0, augmentor=None):
    """Generate mixture signals from individual signatures.

    Args:
        augmentor: Optional SignatureAugmentor applied to each source signature
                   before composition. Must be None for val/test data.
    """
    if n_max is None:
        n_max = n_classes
    c = Composer(X, y, random_state=42, augmentor=augmentor)
    X_out = np.empty((0, X.shape[1]))
    Y_out = np.empty((0, n_classes))
    lenc = MultiLabelBinarizer(classes=np.unique(y))
    n_samples = round(share * n_samples_per_component)

    for n_comp in range(n_min, n_max + 1):
        X_i, Y_i = c.make_samples(n_samples, n_comp,
                                   min_freqs=min_freqs, max_freqs=max_freqs)
        X_i = np.stack(X_i)
        Y_i = np.stack(Y_i)
        Y_i = lenc.fit_transform(Y_i)
        X_out = np.concatenate((X_out, X_i))
        Y_out = np.concatenate((Y_out, Y_i))

    return X_out, Y_out


def get_stats(X):
    X = np.exp(X)
    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True, ddof=1)
    return m, s


# ==============================================================================
# Dataset & Metrics
# ==============================================================================

class NILMDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@torch.no_grad()
def f1_with_logits(y_pred, y_test, threshold=0.5, average='samples'):
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.where(y_pred >= threshold, 1, 0)
    y_pred, y_test = y_pred.cpu().numpy(), y_test.cpu().numpy()
    return f1_score(y_test, y_pred, average=average, zero_division=0)


# ==============================================================================
# Training Engine
# ==============================================================================

def compute_pos_weight(Y_train: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    """BCE pos_weight = neg_count / pos_count per class.

    Passed to BCEWithLogitsLoss to upweight underrepresented positive labels.
    Only use this if per-class label counts are confirmed to be imbalanced.
    """
    n = len(Y_train)
    n_pos = Y_train.sum(axis=0).clip(min=1)
    n_neg = n - n_pos
    weights = n_neg / n_pos
    return torch.tensor(weights, dtype=dtype, device=device)


def train_epoch(model, loader, loss_fn, optimizer, device, dtype, threshold=0.5):
    model.train()
    loss_m = AverageMeter()
    score_m = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=dtype)
        targets = targets.to(device, dtype=dtype)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        score = f1_with_logits(outputs, targets, threshold=threshold)
        score_m.update(score, len(inputs))
        loss_m.update(loss.item(), len(inputs))

    return {'train/loss': loss_m.avg, 'train/score': score_m.avg}


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, dtype):
    model.eval()
    loss_m = AverageMeter()
    all_pred, all_true = [], []

    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=dtype)
        targets = targets.to(device, dtype=dtype)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_m.update(loss.item(), len(inputs))

        all_pred.append(outputs.cpu())
        all_true.append(targets.cpu())

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)

    # Find optimal threshold
    scores = []
    thresholds = torch.linspace(0.1, 1.0, 20)
    for thresh in thresholds:
        scores.append(f1_with_logits(pred, true, threshold=thresh.item()))

    opt_idx = np.argmax(scores)
    return {
        'val/loss': loss_m.avg,
        'val/score': scores[opt_idx],
        'threshold': thresholds[opt_idx].item(),
    }


def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, dtype, save_dir='', checkpoint_meta=None,
                start_epoch=0, best_val=0.0, history=None, save_every=1,
                snapshot_every=25, early_stopping_patience=0,
                early_stopping_min_delta=0.0,
                warmup_epochs=0, warmup_start_lr=1e-5):
    if history is None:
        history = {
            "train": {"loss": [], "score": []},
            "val": {"loss": [], "score": []},
            "threshold": [], "lr": [],
        }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    val_stats = {}
    epochs_without_improvement = 0
    stop_epoch = None
    model_version = checkpoint_meta.get('model_version', '0.0.1-dev') if checkpoint_meta else '0.0.1-dev'
    best_versioned_path = os.path.join(save_dir, f"latest_v{model_version}.pt")
    last_versioned_path = os.path.join(save_dir, f"last_v{model_version}.pt")
    base_lr = optimizer.param_groups[0]['lr']

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Linear LR warmup: only applies to the first `warmup_epochs` steps of
        # this run.  The plateau scheduler is held off during warmup so its
        # internal patience counter doesn't start decaying before training is
        # fully warmed up.
        effective_epoch = epoch - start_epoch
        in_warmup = warmup_epochs > 0 and effective_epoch < warmup_epochs
        if in_warmup:
            progress = (effective_epoch + 1) / warmup_epochs
            warmup_lr = warmup_start_lr + (base_lr - warmup_start_lr) * progress
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, device, dtype,
            threshold=val_stats.get('threshold', 0.5))

        history["train"]["loss"].append(train_stats['train/loss'])
        history["train"]["score"].append(train_stats['train/score'])

        val_stats = val_epoch(model, val_loader, loss_fn, device, dtype)
        history["val"]["loss"].append(val_stats['val/loss'])
        history["val"]["score"].append(val_stats['val/score'])

        if scheduler and not in_warmup:
            scheduler.step(val_stats['val/loss'])

        history['threshold'].append(val_stats.get('threshold', 0.5))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:>3d}/{num_epochs} | "
                  f"Train Loss: {train_stats['train/loss']:.4f} F1: {train_stats['train/score']:.4f} | "
                  f"Val Loss: {val_stats['val/loss']:.4f} F1: {val_stats['val/score']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_f1": best_val,
            "threshold": val_stats.get('threshold', 0.5),
            "history": history,
        }
        if checkpoint_meta:
            ckpt.update(checkpoint_meta)

        improved = val_stats['val/score'] > (best_val + early_stopping_min_delta)
        if improved:
            best_val = val_stats['val/score']
            ckpt["best_val_f1"] = best_val
            epochs_without_improvement = 0
            if save_dir:
                torch.save(ckpt, f"{save_dir}/best.pt")
                torch.save(ckpt, best_versioned_path)
        else:
            epochs_without_improvement += 1

        if save_dir and save_every > 0 and ((epoch + 1) % save_every == 0):
            torch.save(ckpt, f"{save_dir}/last.pt")
            torch.save(ckpt, last_versioned_path)

        if save_dir and snapshot_every > 0 and ((epoch + 1) % snapshot_every == 0):
            torch.save(ckpt, f"{save_dir}/epoch_{epoch + 1:03d}.pt")

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stop_epoch = epoch
            print(f"Early stopping at epoch {epoch} | "
                  f"Best val F1: {best_val:.4f} | "
                  f"No improvement for {epochs_without_improvement} epochs")
            break

    if save_dir:
        ckpt = {
            "epoch": stop_epoch if stop_epoch is not None else epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_f1": best_val,
            "threshold": val_stats.get('threshold', 0.5),
            "history": history,
            "early_stopped": stop_epoch is not None,
        }
        if checkpoint_meta:
            ckpt.update(checkpoint_meta)
        torch.save(ckpt, f"{save_dir}/last.pt")
        torch.save(ckpt, last_versioned_path)

    print(f"\nBest validation F1: {best_val:.4f}")
    return history


# ==============================================================================
# Evaluation — Full Metrics Suite
# ==============================================================================

def compute_all_metrics(Y_true, Y_pred, threshold, appliance_names=None):
    """Compute comprehensive multi-label classification metrics."""
    n_classes = Y_true.shape[1]
    if appliance_names is None:
        appliance_names = [f'Class {i}' for i in range(n_classes)]

    results = {}

    # --- Global metrics ---
    results['f1_samples'] = f1_score(Y_true, Y_pred, average='samples', zero_division=0)
    results['f1_macro'] = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    results['f1_weighted'] = f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    results['precision_samples'] = precision_score(Y_true, Y_pred, average='samples', zero_division=0)
    results['precision_macro'] = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    results['recall_samples'] = recall_score(Y_true, Y_pred, average='samples', zero_division=0)
    results['recall_macro'] = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
    results['accuracy'] = accuracy_score(Y_true, Y_pred)
    results['hamming_loss'] = hamming_loss(Y_true, Y_pred)
    results['jaccard_samples'] = jaccard_score(Y_true, Y_pred, average='samples', zero_division=0)
    results['jaccard_macro'] = jaccard_score(Y_true, Y_pred, average='macro', zero_division=0)
    results['threshold'] = threshold

    # --- Per-class metrics ---
    per_class_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(Y_true, Y_pred, average=None, zero_division=0)

    results['per_class'] = {}
    for i in range(n_classes):
        name = appliance_names[i] if i < len(appliance_names) else f'Class {i}'
        results['per_class'][name] = {
            'f1': per_class_f1[i],
            'precision': per_class_prec[i],
            'recall': per_class_rec[i],
            'support': int(Y_true[:, i].sum()),
        }

    # --- Per-component-count metrics ---
    n_components = Y_true.sum(axis=1).astype(int)
    unique_counts = np.unique(n_components)
    results['per_n_components'] = {}
    for nc in unique_counts:
        mask = n_components == nc
        if mask.sum() == 0:
            continue
        results['per_n_components'][int(nc)] = {
            'f1_samples': f1_score(Y_true[mask], Y_pred[mask], average='samples', zero_division=0),
            'f1_macro': f1_score(Y_true[mask], Y_pred[mask], average='macro', zero_division=0),
            'precision': precision_score(Y_true[mask], Y_pred[mask], average='samples', zero_division=0),
            'recall': recall_score(Y_true[mask], Y_pred[mask], average='samples', zero_division=0),
            'accuracy': accuracy_score(Y_true[mask], Y_pred[mask]),
            'n_samples': int(mask.sum()),
        }

    return results


def evaluate(model, X_test, Y_test, threshold, device, dtype, n_classes,
             appliance_names=None, batch_size=256):
    """Run full evaluation on test set with batched inference."""
    model.eval()

    # Batched inference to avoid OOM
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.tensor(X_test[i:i+batch_size], dtype=dtype, device=device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    Y_prob = np.concatenate(all_preds, axis=0)
    Y_pred = np.where(Y_prob >= threshold, 1, 0)

    # Compute all metrics
    metrics = compute_all_metrics(Y_test, Y_pred, threshold, appliance_names)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*40}")
    print(f"  {'F1 Score (samples)':<30} {metrics['f1_samples']:>10.4f}")
    print(f"  {'F1 Score (macro)':<30} {metrics['f1_macro']:>10.4f}")
    print(f"  {'F1 Score (micro)':<30} {metrics['f1_micro']:>10.4f}")
    print(f"  {'F1 Score (weighted)':<30} {metrics['f1_weighted']:>10.4f}")
    print(f"  {'Precision (samples)':<30} {metrics['precision_samples']:>10.4f}")
    print(f"  {'Precision (macro)':<30} {metrics['precision_macro']:>10.4f}")
    print(f"  {'Recall (samples)':<30} {metrics['recall_samples']:>10.4f}")
    print(f"  {'Recall (macro)':<30} {metrics['recall_macro']:>10.4f}")
    print(f"  {'Exact Match Accuracy':<30} {metrics['accuracy']:>10.4f}")
    print(f"  {'Hamming Loss':<30} {metrics['hamming_loss']:>10.4f}")
    print(f"  {'Jaccard (samples)':<30} {metrics['jaccard_samples']:>10.4f}")
    print(f"  {'Jaccard (macro)':<30} {metrics['jaccard_macro']:>10.4f}")
    print(f"{'='*60}")

    print(f"\n  Per-Appliance Breakdown:")
    print(f"  {'Appliance':<22} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Support':>8}")
    print(f"  {'-'*52}")
    for name, vals in metrics['per_class'].items():
        print(f"  {name:<22} {vals['f1']:>7.4f} {vals['precision']:>7.4f} "
              f"{vals['recall']:>7.4f} {vals['support']:>8d}")

    print(f"\n  Performance by Number of Active Appliances:")
    print(f"  {'#Appl':<7} {'F1(s)':>7} {'F1(m)':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7} {'N':>6}")
    print(f"  {'-'*52}")
    for nc, vals in sorted(metrics['per_n_components'].items()):
        print(f"  {nc:<7} {vals['f1_samples']:>7.4f} {vals['f1_macro']:>7.4f} "
              f"{vals['precision']:>7.4f} {vals['recall']:>7.4f} "
              f"{vals['accuracy']:>7.4f} {vals['n_samples']:>6d}")
    print(f"{'='*60}")

    return Y_pred, Y_prob, metrics


# ==============================================================================
# Paper-Style Plots
# ==============================================================================

def plot_training_curves(history, save_dir='figures'):
    """Plot 1: Training & validation loss and F1 curves (smooth, publication style)."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train']['loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train']['loss'], color='#2196F3', linewidth=2,
                 label='Train Loss', alpha=0.85)
    axes[0].plot(epochs, history['val']['loss'], color='#F44336', linewidth=2,
                 label='Val Loss', alpha=0.85)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(framealpha=0.9)

    # F1 curves
    axes[1].plot(epochs, history['train']['score'], color='#2196F3', linewidth=2,
                 label='Train F1', alpha=0.85)
    axes[1].plot(epochs, history['val']['score'], color='#F44336', linewidth=2,
                 label='Val F1', alpha=0.85)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score (samples)')
    axes[1].set_title('Training & Validation F1 Score')
    axes[1].legend(framealpha=0.9)

    plt.suptitle('Fusion-ResNet — Training Progress', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/training_curves.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_per_class_f1_bars(metrics, save_dir='figures'):
    """Plot 2: Per-appliance F1 score bar chart (paper Figure: f1 bars)."""
    os.makedirs(save_dir, exist_ok=True)

    names = list(metrics['per_class'].keys())
    f1s = [v['f1'] for v in metrics['per_class'].values()]
    precs = [v['precision'] for v in metrics['per_class'].values()]
    recs = [v['recall'] for v in metrics['per_class'].values()]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, f1s, width, label='F1 Score', color='#2196F3', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x, precs, width, label='Precision', color='#4CAF50', alpha=0.85, edgecolor='white')
    bars3 = ax.bar(x + width, recs, width, label='Recall', color='#FF9800', alpha=0.85, edgecolor='white')

    ax.set_xlabel('Appliance')
    ax.set_ylabel('Score')
    ax.set_title('Fusion-ResNet — Per-Appliance Performance on Test Set', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)

    # Add value labels on F1 bars
    for bar, val in zip(bars1, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    path = f'{save_dir}/per_appliance_f1.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_f1_by_components(metrics, save_dir='figures'):
    """Plot 3: F1 score vs number of simultaneously active appliances."""
    os.makedirs(save_dir, exist_ok=True)

    comp_data = metrics['per_n_components']
    if not comp_data:
        return

    nc_list = sorted(comp_data.keys())
    f1_samples = [comp_data[nc]['f1_samples'] for nc in nc_list]
    f1_macro = [comp_data[nc]['f1_macro'] for nc in nc_list]
    accuracies = [comp_data[nc]['accuracy'] for nc in nc_list]
    n_samples = [comp_data[nc]['n_samples'] for nc in nc_list]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1, color2, color3 = '#2196F3', '#F44336', '#9E9E9E'

    ax1.plot(nc_list, f1_samples, 'o-', color=color1, linewidth=2.5,
             markersize=7, label='F1 (samples)', zorder=3)
    ax1.plot(nc_list, f1_macro, 's--', color=color2, linewidth=2,
             markersize=6, label='F1 (macro)', zorder=3)
    ax1.plot(nc_list, accuracies, 'D:', color='#4CAF50', linewidth=1.5,
             markersize=5, label='Exact Match Acc', alpha=0.7, zorder=3)

    ax1.set_xlabel('Number of Simultaneously Active Appliances')
    ax1.set_ylabel('Score')
    ax1.set_title('Fusion-ResNet — Performance vs Mixture Complexity', fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(nc_list)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Add sample counts as bar background
    ax2 = ax1.twinx()
    ax2.bar(nc_list, n_samples, alpha=0.08, color=color3, width=0.6, zorder=1)
    ax2.set_ylabel('Number of Test Samples', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)

    plt.tight_layout()
    path = f'{save_dir}/f1_by_components.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_per_class_heatmap(metrics, save_dir='figures'):
    """Plot 4: Per-class precision/recall/F1 heatmap (confusion-matrix style)."""
    os.makedirs(save_dir, exist_ok=True)

    names = list(metrics['per_class'].keys())
    data = np.array([
        [v['f1'] for v in metrics['per_class'].values()],
        [v['precision'] for v in metrics['per_class'].values()],
        [v['recall'] for v in metrics['per_class'].values()],
    ])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                     xticklabels=names, yticklabels=['F1 Score', 'Precision', 'Recall'],
                     vmin=0, vmax=1, linewidths=0.5, ax=ax,
                     cbar_kws={'label': 'Score'})

    ax.set_title('Fusion-ResNet — Per-Appliance Metrics Heatmap', fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    path = f'{save_dir}/metrics_heatmap.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_lr_schedule(history, save_dir='figures'):
    """Plot 5: Learning rate schedule over training."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history['lr']) + 1)
    ax.plot(epochs, history['lr'], color='#9C27B0', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_yscale('log')

    plt.tight_layout()
    path = f'{save_dir}/lr_schedule.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_summary_dashboard(history, metrics, save_dir='figures'):
    """Plot 6: Combined dashboard — all key results on one figure."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    epochs = range(1, len(history['train']['loss']) + 1)

    # --- (0,0) Loss curves ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, history['train']['loss'], color='#2196F3', lw=2, label='Train', alpha=0.85)
    ax0.plot(epochs, history['val']['loss'], color='#F44336', lw=2, label='Val', alpha=0.85)
    ax0.set_xlabel('Epoch'); ax0.set_ylabel('Loss')
    ax0.set_title('Loss Curves'); ax0.legend()

    # --- (0,1) F1 curves ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(epochs, history['train']['score'], color='#2196F3', lw=2, label='Train', alpha=0.85)
    ax1.plot(epochs, history['val']['score'], color='#F44336', lw=2, label='Val', alpha=0.85)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('F1')
    ax1.set_title('F1 Score (samples)'); ax1.legend()

    # --- (0,2) Global metrics summary ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = (
        f"FUSION-RESNET TEST RESULTS\n"
        f"{'─'*32}\n"
        f"F1 (samples):    {metrics['f1_samples']:.4f}\n"
        f"F1 (macro):      {metrics['f1_macro']:.4f}\n"
        f"F1 (weighted):   {metrics['f1_weighted']:.4f}\n"
        f"Precision (s):   {metrics['precision_samples']:.4f}\n"
        f"Recall (s):      {metrics['recall_samples']:.4f}\n"
        f"Exact Match:     {metrics['accuracy']:.4f}\n"
        f"Hamming Loss:    {metrics['hamming_loss']:.4f}\n"
        f"Jaccard (s):     {metrics['jaccard_samples']:.4f}\n"
        f"{'─'*32}\n"
        f"Threshold:       {metrics['threshold']:.2f}\n"
    )
    ax2.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#E3F2FD', alpha=0.9))

    # --- (1,0:2) Per-appliance bars ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    names = list(metrics['per_class'].keys())
    f1s = [v['f1'] for v in metrics['per_class'].values()]
    precs = [v['precision'] for v in metrics['per_class'].values()]
    recs = [v['recall'] for v in metrics['per_class'].values()]
    x = np.arange(len(names))
    w = 0.25
    ax3.bar(x - w, f1s, w, label='F1', color='#2196F3', alpha=0.85, edgecolor='white')
    ax3.bar(x, precs, w, label='Precision', color='#4CAF50', alpha=0.85, edgecolor='white')
    ax3.bar(x + w, recs, w, label='Recall', color='#FF9800', alpha=0.85, edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Score'); ax3.set_ylim(0, 1.05)
    ax3.set_title('Per-Appliance Performance'); ax3.legend(loc='upper right')

    # --- (1,2) F1 by number of components ---
    ax4 = fig.add_subplot(gs[1, 2])
    comp_data = metrics.get('per_n_components', {})
    if comp_data:
        nc_list = sorted(comp_data.keys())
        f1_s = [comp_data[nc]['f1_samples'] for nc in nc_list]
        f1_m = [comp_data[nc]['f1_macro'] for nc in nc_list]
        ax4.plot(nc_list, f1_s, 'o-', color='#2196F3', lw=2, ms=5, label='F1 (samples)')
        ax4.plot(nc_list, f1_m, 's--', color='#F44336', lw=1.5, ms=4, label='F1 (macro)')
        ax4.set_xlabel('# Active Appliances'); ax4.set_ylabel('F1')
        ax4.set_title('F1 vs Mixture Complexity')
        ax4.set_ylim(-0.05, 1.05); ax4.legend()
        ax4.set_xticks(nc_list)

    fig.suptitle('Fusion-ResNet — NILM Evaluation Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = f'{save_dir}/dashboard.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def generate_all_plots(history, metrics, save_dir='figures'):
    """Generate all paper-style plots."""
    print(f"\nGenerating plots → {save_dir}/")
    os.makedirs(save_dir, exist_ok=True)

    plot_training_curves(history, save_dir)
    plot_per_class_f1_bars(metrics, save_dir)
    plot_f1_by_components(metrics, save_dir)
    plot_per_class_heatmap(metrics, save_dir)
    plot_lr_schedule(history, save_dir)
    plot_summary_dashboard(history, metrics, save_dir)

    print(f"  All plots saved to {save_dir}/\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    dtype = torch.float64 if args.fp64 else torch.float32
    print(f"Device: {args.device} | Dtype: {dtype} | Variant: {args.variant}")

    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    X_real = np.load(f'{args.data_dir}/X_real.npy', allow_pickle=True)
    y_real = np.load(f'{args.data_dir}/y_real.npy', allow_pickle=True)

    # Try to load appliance names from label encoder
    appliance_names = DEFAULT_APPLIANCE_NAMES
    lenc_path = f'{args.data_dir}/real_label_encoder.npy'
    if os.path.exists(lenc_path):
        try:
            lenc = np.load(lenc_path, allow_pickle=True).item()
            appliance_names = list(lenc.classes_)
        except Exception:
            pass

    # Drop rare classes (< 10 samples)
    class_ids, counts = np.unique(y_real, return_counts=True)
    kept_class_ids = [int(cls) for cls, count in zip(class_ids, counts) if count >= 10]
    keep_mask = np.isin(y_real, kept_class_ids)
    y_real = y_real[keep_mask]
    X_real = X_real[keep_mask]

    n_classes = len(np.unique(y_real))
    signal_length = X_real.shape[1]
    appliance_names = [str(appliance_names[idx]) for idx in kept_class_ids]
    print(f"Classes: {n_classes} | Signal length: {signal_length} | "
          f"Samples: {len(X_real)}")
    print(f"Appliances: {appliance_names}")

    # ------------------------------------------------------------------
    # 2. Train/Val/Test split
    # ------------------------------------------------------------------
    print("\n[2/6] Splitting and composing mixtures...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=y_real)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.7, random_state=42, stratify=y_test)

    # Build source-level augmentor (train split only; val/test must stay clean)
    augmentor = None
    if any([args.aug_noise_sigma > 0, args.aug_amplitude_scale > 0,
            args.aug_phase_shift, args.aug_time_warp]):
        augmentor = SignatureAugmentor(
            noise_sigma=args.aug_noise_sigma,
            amplitude_scale=args.aug_amplitude_scale,
            phase_shift=args.aug_phase_shift,
            time_warp=args.aug_time_warp,
            seed=args.seed,
        )
        print(f"  Source augmentation: {augmentor.summary()}")

    # Generate mixture signals (augmentor applied only to training data)
    X_train, Y_train = compose(
        X_train, y_train, n_classes, args.n_samples,
        share=len(X_train) / len(X_real), augmentor=augmentor)
    X_val, Y_val = compose(
        X_val, y_val, n_classes, args.n_samples,
        share=len(X_val) / len(X_real))
    X_test, Y_test = compose(
        X_test, y_test, n_classes, args.n_samples,
        share=len(X_test) / len(X_real))

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Normalize to unit magnitude
    X_train = X_train / np.where(np.abs(X_train).max(axis=1, keepdims=True) == 0, 1.0,
                                 np.abs(X_train).max(axis=1, keepdims=True))
    X_val = X_val / np.where(np.abs(X_val).max(axis=1, keepdims=True) == 0, 1.0,
                             np.abs(X_val).max(axis=1, keepdims=True))
    X_test = X_test / np.where(np.abs(X_test).max(axis=1, keepdims=True) == 0, 1.0,
                               np.abs(X_test).max(axis=1, keepdims=True))

    # ------------------------------------------------------------------
    # 3. ICA for the ICA branch
    # ------------------------------------------------------------------
    print("\n[3/6] Fitting ICA...")
    fica = FastICA(n_classes + 1, whiten='unit-variance')
    fica.fit(X_train)

    m, s = get_stats(fica.transform(X_train))
    U = fica.components_.astype(np.float64)
    M = fica.mean_.astype(np.float64)
    print(f"ICA components: {U.shape}")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\n[4/6] Building Fusion-ResNet...")
    ModelClass = FusionResNet if args.variant == 'full' else FusionResNetLite

    model = ModelClass(
        n_classes=n_classes,
        signal_length=signal_length,
        U=U, M=M, m=m, s=s,
        dropout=args.dropout,
        mains_freq=args.mains_freq,
        mains_voltage=args.mains_volt,
    )

    if dtype == torch.float64:
        model = model.double()
    else:
        model = model.float()

    model = model.to(args.device)
    model_summary(model, input_shape=(2, signal_length))

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training for {args.epochs} epochs...")

    train_ds = NILMDataset(X_train, Y_train)
    val_ds = NILMDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(args.device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(args.device == 'cuda'))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )

    if args.use_pos_weight:
        pw = compute_pos_weight(Y_train, args.device, dtype)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"  pos_weight range: [{pw.min().item():.2f}, {pw.max().item():.2f}]")
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val = 0.0
    history = None
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=True)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if scheduler and resume_ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        start_epoch = int(resume_ckpt.get('epoch', -1)) + 1
        best_val = float(resume_ckpt.get('best_val_f1', 0.0))
        history = resume_ckpt.get('history')
        print(f"  Resume epoch: {start_epoch} | Best val F1 so far: {best_val:.4f}")

    history = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler,
        num_epochs=args.epochs, device=args.device, dtype=dtype,
        save_dir=args.save_dir,
        checkpoint_meta={
            "appliance_names": appliance_names,
            "kept_class_ids": kept_class_ids,
            "variant": args.variant,
            "signal_length": signal_length,
            "n_classes": n_classes,
            "model_version": args.model_version,
        },
        start_epoch=start_epoch,
        best_val=best_val,
        history=history,
        save_every=args.save_every,
        snapshot_every=args.snapshot_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate & Generate Plots
    # ------------------------------------------------------------------
    print(f"\n[6/6] Evaluating and generating plots...")
    threshold = history['threshold'][-1]

    # Load best checkpoint with versioning support
    versioned_ckpt_path = f"{args.save_dir}/latest_v{args.model_version}.pt"
    legacy_ckpt_path = f"{args.save_dir}/best.pt"
    ckpt_path = versioned_ckpt_path if os.path.exists(versioned_ckpt_path) else legacy_ckpt_path
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        threshold = ckpt.get('threshold', threshold)
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, "
              f"val F1 = {ckpt.get('best_val_f1', 'N/A'):.4f})")
        
        # Save versioned checkpoint if not already saved
        if ckpt_path == legacy_ckpt_path:  # Only if we loaded from legacy path
            torch.save(ckpt, versioned_ckpt_path)
            print(f"  Saved versioned checkpoint: {versioned_ckpt_path}")
    elif os.path.exists(legacy_ckpt_path):
        ckpt = torch.load(legacy_ckpt_path, map_location=args.device, weights_only=True)
        torch.save(ckpt, versioned_ckpt_path)
        print(f"  Saved versioned checkpoint: {versioned_ckpt_path}")

    Y_pred, Y_prob, metrics = evaluate(
        model, X_test, Y_test, threshold, args.device, dtype, n_classes,
        appliance_names=appliance_names, batch_size=args.batch_size,
    )

    # Save raw probabilities for calibration diagnostics
    os.makedirs(args.figures_dir, exist_ok=True)
    np.save(os.path.join(args.figures_dir, 'Y_prob.npy'), Y_prob)
    np.save(os.path.join(args.figures_dir, 'Y_true.npy'), Y_test)

    # Post-training temperature scaling calibration
    if args.calibrate:
        print("\n  Running temperature scaling calibration on validation set...")
        try:
            from temperature_scaling import TemperatureScaler
            # Collect validation logits with best model
            val_logits_list = []
            with torch.no_grad():
                model.eval()
                for i in range(0, len(X_val), args.batch_size):
                    batch = torch.tensor(X_val[i:i + args.batch_size], dtype=dtype, device=args.device)
                    val_logits_list.append(model(batch).cpu().numpy())
            val_logits = np.concatenate(val_logits_list, axis=0)
            n_half = len(val_logits) // 2
            cal_logits, rep_logits = val_logits[:n_half], val_logits[n_half:]
            cal_labels, rep_labels = Y_val[:n_half], Y_val[n_half:]

            scaler = TemperatureScaler()
            scaler.fit(cal_logits, cal_labels)
            cal_threshold = scaler.find_best_threshold(cal_logits, cal_labels)
            cal_metrics = scaler.calibration_metrics(rep_logits, rep_labels)

            print(f"  Calibrated threshold: {cal_threshold:.4f} (was {threshold:.4f})")
            print(f"  ECE: {cal_metrics['ece_before']:.4f} → {cal_metrics['ece_after']:.4f}")

            import json as _json
            cal_metrics['calibrated_threshold'] = cal_threshold
            cal_metrics['uncalibrated_threshold'] = float(threshold)
            with open(os.path.join(args.figures_dir, 'calibration_metrics.json'), 'w') as f:
                _json.dump(cal_metrics, f, indent=2)
            print(f"  Saved: {args.figures_dir}/calibration_metrics.json")

            # Re-evaluate test set with calibrated logits + threshold
            test_logits = np.log(Y_prob.clip(1e-7, 1 - 1e-7)) - np.log(1 - Y_prob.clip(1e-7, 1 - 1e-7))
            cal_test_probs = 1.0 / (1.0 + np.exp(-scaler.scale(test_logits)))
            Y_pred_cal = (cal_test_probs >= cal_threshold).astype(int)
            from sklearn.metrics import f1_score as _f1
            f1_cal = _f1(Y_test, Y_pred_cal, average='samples', zero_division=0)
            print(f"  Calibrated test F1 (samples): {f1_cal:.4f} (uncalibrated: {metrics['f1_samples']:.4f})")
        except ImportError:
            print("  [skip] temperature_scaling.py not found")

    # Save metrics to JSON
    import json
    metrics_serializable = {k: v for k, v in metrics.items() if k != 'per_class'}
    metrics_serializable['per_class'] = {
        k: {mk: float(mv) for mk, mv in v.items()}
        for k, v in metrics['per_class'].items()
    }
    metrics_serializable['per_n_components'] = {
        str(k): {mk: float(mv) for mk, mv in v.items()}
        for k, v in metrics.get('per_n_components', {}).items()
    }
    with open(os.path.join(args.figures_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    with open(os.path.join(args.figures_dir, f'test_metrics_{args.model_version}.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  Saved: {args.figures_dir}/test_metrics.json")

    # Generate all plots
    generate_all_plots(history, metrics, save_dir=args.figures_dir)

    print("Done!")


if __name__ == '__main__':
    main()

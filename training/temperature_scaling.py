"""
Temperature scaling calibration for multi-label classifiers.

Usage:
    scaler = TemperatureScaler()
    T = scaler.fit(cal_logits, cal_labels)          # fit on calibration set
    metrics = scaler.calibration_metrics(...)        # ECE before/after
    thresh = scaler.find_best_threshold(cal_logits, cal_labels)
    scaled_logits = scaler.scale(test_logits)        # apply to test set
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for probability calibration.

    Divides all logits by a learnable scalar T before sigmoid:
        calibrated_prob = sigmoid(logit / T)

    T > 1 lowers confidence (fixes overconfident models).
    T < 1 raises confidence (fixes under-confident models — expected here).
    T is optimized by minimizing BCE loss on a held-out calibration set.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    @property
    def T(self) -> float:
        return float(self.temperature.clamp(min=0.01).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Scale numpy logits by fitted temperature."""
        return logits / self.T

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = True,
    ) -> float:
        """Optimize temperature T on calibration data (LBFGS).

        Args:
            logits: Raw model logits, shape (N, n_classes).
            labels: Binary ground-truth labels, shape (N, n_classes).
            lr: LBFGS learning rate.
            max_iter: Maximum LBFGS iterations.
            verbose: Print fitted T.

        Returns:
            Fitted temperature value T.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature.clamp(min=0.01)
            loss = loss_fn(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = self.T
        if verbose:
            print(f"  TemperatureScaler: fitted T = {T:.4f}")
        return T

    @torch.no_grad()
    def find_best_threshold(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 37,
    ) -> float:
        """Grid-search optimal F1 threshold on calibrated logits.

        Should be called on the calibration set (NOT the test set).
        """
        scaled = self.scale(logits)
        probs = 1.0 / (1.0 + np.exp(-scaled))

        best_thresh, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, n_thresholds):
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, average='samples', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(t)
        return best_thresh

    def calibration_metrics(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Compute calibration statistics before and after temperature scaling.

        Args:
            logits: Raw model logits (N, n_classes) — from the REPORT split,
                    not the calibration split used to fit T.
            labels: Binary ground truth.
            n_bins: Number of bins for ECE computation.

        Returns:
            Dict with ECE, mean probabilities, and temperature.
        """
        probs_raw = 1.0 / (1.0 + np.exp(-logits))
        probs_cal = 1.0 / (1.0 + np.exp(-self.scale(logits)))

        def ece(probs: np.ndarray, labs: np.ndarray) -> float:
            flat_p = probs.ravel()
            flat_l = labs.ravel()
            bins = np.linspace(0, 1, n_bins + 1)
            total, n = 0.0, flat_p.size
            for i in range(n_bins):
                m = (flat_p >= bins[i]) & (flat_p < bins[i + 1])
                if not m.any():
                    continue
                total += m.sum() * abs(flat_l[m].mean() - flat_p[m].mean())
            return total / max(n, 1)

        labs_flat = labels.ravel()
        pos_mask = labs_flat == 1
        neg_mask = ~pos_mask

        return {
            'temperature': self.T,
            'ece_before': float(ece(probs_raw, labels)),
            'ece_after': float(ece(probs_cal, labels)),
            'mean_prob_before': float(probs_raw.ravel().mean()),
            'mean_prob_after': float(probs_cal.ravel().mean()),
            'mean_prob_positives_before': float(probs_raw.ravel()[pos_mask].mean()) if pos_mask.any() else 0.0,
            'mean_prob_positives_after': float(probs_cal.ravel()[pos_mask].mean()) if pos_mask.any() else 0.0,
            'mean_prob_negatives_before': float(probs_raw.ravel()[neg_mask].mean()) if neg_mask.any() else 0.0,
            'mean_prob_negatives_after': float(probs_cal.ravel()[neg_mask].mean()) if neg_mask.any() else 0.0,
        }

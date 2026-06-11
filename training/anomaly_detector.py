"""
Anomaly Detection for NILM Fault Monitoring
============================================

Detects electrical faults and appliance degradation by analyzing:
1. Unknown loads (no appliance gets high confidence)
2. Appliance degradation (confidence drift over time)
3. Unaccounted current (measured > predicted)

Usage:
    detector = AnomalyDetector(appliance_names)
    anomalies = detector.check_window(
        predictions=predictions_array,   # (n_appliances,)
        probabilities=probabilities_array,
        measured_current=measured_rms,   # from PZEM or similar
        timestamp=datetime.now(timezone.utc)
    )
    if anomalies:
        for alert in anomalies:
            print(alert)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    timestamp: str
    anomaly_type: str  # 'unknown_load', 'degradation', 'unaccounted_current', 'overcurrent'
    severity: str  # 'low', 'medium', 'high'
    message: str
    details: dict

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return f"[{self.severity.upper()}] {self.anomaly_type}: {self.message}"


class AnomalyDetector:
    """Detects electrical faults and appliance issues in real-time."""

    def __init__(
        self,
        appliance_names: list[str],
        history_file: Optional[str] = None,
        high_confidence_threshold: float = 0.5,
        degradation_threshold: float = 0.15,  # 15% confidence drop
        time_window_days: int = 7,
        current_mismatch_threshold: float = 0.2,  # 20% difference
    ):
        """
        Args:
            appliance_names: List of appliance names (order must match model output)
            history_file: Path to save/load confidence history (.json)
            high_confidence_threshold: Minimum confidence to consider appliance "detected"
            degradation_threshold: Confidence drop threshold to flag degradation
            time_window_days: Window for calculating degradation trend
            current_mismatch_threshold: Ratio threshold for unaccounted current
        """
        self.appliance_names = appliance_names
        self.n_appliances = len(appliance_names)
        self.high_confidence_threshold = high_confidence_threshold
        self.degradation_threshold = degradation_threshold
        self.time_window_days = time_window_days
        self.current_mismatch_threshold = current_mismatch_threshold
        self.history_file = history_file

        # Confidence history per appliance: {appliance_name: [(timestamp, confidence), ...]}
        self.confidence_history = {name: [] for name in appliance_names}

        # Load history if it exists
        if history_file and os.path.exists(history_file):
            self._load_history()

    def _load_history(self):
        """Load confidence history from disk."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.confidence_history = data.get('history', {})
            print(f"Loaded history from {self.history_file}")
        except Exception as e:
            print(f"Warning: Could not load history: {e}")

    def _save_history(self):
        """Save confidence history to disk."""
        if not self.history_file:
            return

        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        try:
            with open(self.history_file, 'w') as f:
                json.dump({'history': self.confidence_history}, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    def check_window(
        self,
        predictions: np.ndarray,  # (n_appliances,) binary
        probabilities: np.ndarray,  # (n_appliances,) float [0, 1]
        measured_current: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[Anomaly]:
        """
        Check a single inference window for anomalies.

        Args:
            predictions: Binary predictions (0 or 1) per appliance
            probabilities: Confidence scores per appliance
            measured_current: Measured RMS current from PZEM or sensor
            timestamp: Timestamp of this inference (default: now)

        Returns:
            List of detected anomalies
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        timestamp_str = timestamp.isoformat()
        anomalies = []

        # --- Check 1: Unknown Load (no appliance detected with high confidence) ---
        max_prob = probabilities.max()
        if max_prob < self.high_confidence_threshold:
            anomalies.append(Anomaly(
                timestamp=timestamp_str,
                anomaly_type='unknown_load',
                severity='medium',
                message=f'No appliance detected with confidence ≥ {self.high_confidence_threshold:.1%}',
                details={
                    'max_confidence': float(max_prob),
                    'threshold': self.high_confidence_threshold,
                }
            ))

        # --- Check 2: Appliance Degradation ---
        for i, name in enumerate(self.appliance_names):
            current_prob = float(probabilities[i])
            self.confidence_history[name].append((timestamp_str, current_prob))

            # Keep only recent history (within time window)
            cutoff_time = timestamp - timedelta(days=self.time_window_days)
            self.confidence_history[name] = [
                (ts, prob) for ts, prob in self.confidence_history[name]
                if datetime.fromisoformat(ts) >= cutoff_time
            ]

            # Check for degradation if we have enough history
            if len(self.confidence_history[name]) > 10:  # Need at least 10 samples
                probs = [prob for ts, prob in self.confidence_history[name]]
                recent_avg = np.mean(probs[-5:])  # Last 5 samples
                older_avg = np.mean(probs[:5])    # First 5 samples

                if older_avg > 0.1:  # Only check if it was previously detected
                    confidence_drop = (older_avg - recent_avg) / older_avg
                    if confidence_drop > self.degradation_threshold:
                        anomalies.append(Anomaly(
                            timestamp=timestamp_str,
                            anomaly_type='degradation',
                            severity='medium',
                            message=f'{name} confidence dropped {confidence_drop:.1%} over {self.time_window_days} days',
                            details={
                                'appliance': name,
                                'initial_confidence': float(older_avg),
                                'recent_confidence': float(recent_avg),
                                'drop_percent': float(confidence_drop * 100),
                            }
                        ))

        # --- Check 3: Unaccounted Current ---
        if measured_current is not None and measured_current > 0:
            # Estimated current = sum of detected appliance confidences
            # (rough approximation; real implementation would use per-appliance current ratings)
            predicted_current = probabilities.sum()  # Simple heuristic
            current_ratio = measured_current / max(predicted_current, 0.1)  # Avoid division by zero

            if current_ratio > (1.0 + self.current_mismatch_threshold):
                unaccounted = measured_current - predicted_current
                anomalies.append(Anomaly(
                    timestamp=timestamp_str,
                    anomaly_type='unaccounted_current',
                    severity='high',
                    message=f'Measured current {measured_current:.2f}A is {(current_ratio - 1):.1%} higher than predicted',
                    details={
                        'measured_current': float(measured_current),
                        'predicted_current': float(predicted_current),
                        'unaccounted': float(unaccounted),
                        'ratio': float(current_ratio),
                    }
                ))

        # Save updated history
        self._save_history()

        return anomalies

    def get_appliance_health(self, appliance_name: str, days: int = 7) -> dict:
        """
        Get health summary for a specific appliance.

        Args:
            appliance_name: Name of the appliance
            days: Number of days to look back

        Returns:
            Dict with health metrics
        """
        if appliance_name not in self.confidence_history:
            return {'error': f'Unknown appliance: {appliance_name}'}

        history = self.confidence_history[appliance_name]
        if not history:
            return {'appliance': appliance_name, 'message': 'No data yet'}

        # Filter to time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [prob for ts, prob in history if datetime.fromisoformat(ts) >= cutoff_time]

        if not recent:
            return {
                'appliance': appliance_name,
                'status': 'no_recent_data',
                'last_detection': history[-1][0] if history else None,
            }

        return {
            'appliance': appliance_name,
            'status': 'healthy' if np.mean(recent) > 0.3 else 'degraded',
            'avg_confidence': float(np.mean(recent)),
            'min_confidence': float(np.min(recent)),
            'max_confidence': float(np.max(recent)),
            'trend': 'improving' if recent[-1] > recent[0] else 'declining',
            'samples': len(recent),
            'time_span_days': days,
        }

    def get_system_health(self, days: int = 7) -> dict:
        """
        Get overall system health summary.

        Returns:
            Dict with system-wide metrics
        """
        health_per_appliance = {}
        for name in self.appliance_names:
            health_per_appliance[name] = self.get_appliance_health(name, days)

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'time_span_days': days,
            'appliances': health_per_appliance,
            'overall_status': 'healthy',  # Could add logic to summarize
        }

    def reset_history(self):
        """Clear all history."""
        self.confidence_history = {name: [] for name in self.appliance_names}
        self._save_history()
        print("History reset.")

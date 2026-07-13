"""
temporal_smoother.py
====================
Temporal postprocessing for Fusion-ResNet NILM predictions.

Operates on the (N_windows, N_classes) probability matrix produced by
inference_pipeline.py.  The model weights are never touched — this is a
pure signal-processing layer on top of per-window outputs.

Pipeline applied independently per appliance class:
  1. EMA smoothing  — smoothed[t] = alpha*p[t] + (1-alpha)*smoothed[t-1]
  2. Hysteresis     — ON  when smoothed prob >= on_threshold
                    — OFF when smoothed prob  < off_threshold
  3. Min-ON filter  — remove ON bursts shorter than min_on_windows
  4. Min-OFF filter — bridge OFF gaps shorter than min_off_windows
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassConfig:
    """Smoothing parameters for one appliance class."""
    alpha: float = 0.3            # EMA coefficient — 1=no smoothing, 0=fully averaged
    on_threshold: float = 0.45    # Smoothed probability to flip state ON
    off_threshold: float = 0.30   # Smoothed probability to flip state OFF (≤ on_threshold)
    min_on_windows: int = 3       # Minimum consecutive ON windows to keep an event
    min_off_windows: int = 2      # Minimum consecutive OFF windows; shorter gaps are bridged


@dataclass
class TemporalConfig:
    """Global default + optional per-class overrides."""
    default: ClassConfig = field(default_factory=ClassConfig)
    classes: Dict[str, ClassConfig] = field(default_factory=dict)

    def for_class(self, name: str) -> ClassConfig:
        return self.classes.get(name, self.default)

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalConfig":
        def _clean(raw: dict) -> dict:
            return {k: v for k, v in raw.items() if not k.startswith("_")}

        default = ClassConfig(**_clean(d.get("default", {})))
        classes = {
            k: ClassConfig(**_clean(v))
            for k, v in d.get("classes", {}).items()
            if not k.startswith("_")
        }
        return cls(default=default, classes=classes)

    @classmethod
    def from_json(cls, path: str) -> "TemporalConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "default": asdict(self.default),
            "classes": {k: asdict(v) for k, v in self.classes.items()},
        }


# ---------------------------------------------------------------------------
# Core smoother
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    EMA + hysteresis + duration filters on a (N, C) probability stream.

    Input:  probs          (N, C) raw sigmoid probabilities from the model
    Output: smoothed_probs (N, C) after EMA
            temporal_active (N, C) binary int8 after the full pipeline
    """

    def __init__(self, appliance_names: list[str], config: TemporalConfig):
        self.names = appliance_names
        self.config = config

    def smooth(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            probs: (N, C) raw probability matrix (float)

        Returns:
            smoothed_probs:  (N, C) EMA-smoothed probabilities
            temporal_active: (N, C) binary int8 after full pipeline
        """
        n, c = probs.shape
        smoothed_probs = np.empty((n, c), dtype=np.float64)
        temporal_active = np.zeros((n, c), dtype=np.int8)

        for i, name in enumerate(self.names):
            cfg = self.config.for_class(name)
            col = probs[:, i].astype(np.float64)

            ema = _ema(col, cfg.alpha)
            smoothed_probs[:, i] = ema

            states = _hysteresis(ema, cfg.on_threshold, cfg.off_threshold)
            states = _filter_short_runs(states, cfg.min_on_windows, target=1)
            states = _filter_short_runs(states, cfg.min_off_windows, target=0)
            temporal_active[:, i] = states

        return smoothed_probs, temporal_active

    def describe(self) -> str:
        """Return a compact log string of per-class configs."""
        d = self.config.default
        lines = [
            "  Temporal smoother config:",
            f"    default  alpha={d.alpha}  on_thr={d.on_threshold}"
            f"  off_thr={d.off_threshold}"
            f"  min_on={d.min_on_windows}w  min_off={d.min_off_windows}w",
        ]
        for name, cfg in self.config.classes.items():
            lines.append(
                f"    {name:<28s} alpha={cfg.alpha}  on_thr={cfg.on_threshold}"
                f"  off_thr={cfg.off_threshold}"
                f"  min_on={cfg.min_on_windows}w  min_off={cfg.min_off_windows}w"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average. alpha=1 → identity; alpha→0 → flat mean."""
    out = np.empty_like(x)
    out[0] = x[0]
    beta = 1.0 - alpha
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + beta * out[t - 1]
    return out


def _hysteresis(signal: np.ndarray, on_thresh: float,
                off_thresh: float) -> np.ndarray:
    """Two-threshold state machine — prevents chattering near the decision boundary."""
    states = np.zeros(len(signal), dtype=np.int8)
    s = 0
    for t, v in enumerate(signal):
        if s == 0 and v >= on_thresh:
            s = 1
        elif s == 1 and v < off_thresh:
            s = 0
        states[t] = s
    return states


def _filter_short_runs(states: np.ndarray, min_len: int,
                       target: int) -> np.ndarray:
    """
    Flip runs of `target` shorter than `min_len` to the opposite state.

    target=1 — remove short ON blips  (false-positive suppression)
    target=0 — fill short OFF gaps    (bridge nearby ON events)
    """
    if min_len <= 1:
        return states.copy()

    out = states.copy()
    other = 1 - target
    i, n = 0, len(states)

    while i < n:
        if states[i] == target:
            j = i
            while j < n and states[j] == target:
                j += 1
            if (j - i) < min_len:
                out[i:j] = other
            i = j
        else:
            i += 1

    return out


# ---------------------------------------------------------------------------
# Builder — called from inference_pipeline.py
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_temporal_config.json"


def build_smoother_from_args(
    appliance_names: list[str],
    config_path: Optional[str],
    alpha: float,
    on_threshold: float,
    off_threshold: float,
    min_on_windows: int,
    min_off_windows: int,
) -> TemporalSmoother:
    """
    Construct a TemporalSmoother from CLI arguments.

    Priority (highest wins):
      1. Per-class overrides in the JSON (--temporal-config or default_temporal_config.json)
      2. CLI flag values — become the effective default for unlisted classes
      3. Built-in ClassConfig defaults
    """
    cli_default = ClassConfig(
        alpha=alpha,
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        min_on_windows=min_on_windows,
        min_off_windows=min_off_windows,
    )

    json_path = config_path or (
        str(_DEFAULT_CONFIG_PATH) if _DEFAULT_CONFIG_PATH.exists() else None
    )

    if json_path:
        cfg = TemporalConfig.from_json(json_path)
        cfg.default = cli_default   # CLI flags win over the JSON [default] section
    else:
        cfg = TemporalConfig(default=cli_default)

    return TemporalSmoother(appliance_names, cfg)

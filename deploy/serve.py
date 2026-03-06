"""
Fusion-ResNet NILM — Cloud Inference Server (FastAPI)
=====================================================

Lightweight REST API that serves the Fusion-ResNet model.
Designed for cloud deployment (AWS/GCP/Azure) with the model
running on CPU (no GPU required at inference time).

Endpoints:
    POST /predict        — Classify a single window (400 samples)
    POST /predict/batch  — Classify multiple windows at once
    GET  /health         — Health check / model info

Usage:
    # Local testing
    uvicorn deploy.serve:app --host 0.0.0.0 --port 8000

    # Docker
    docker build -t fusion-resnet-nilm .
    docker run -p 8000:8000 fusion-resnet-nilm
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths — resolved relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "best.pt"),
)
THRESHOLD = float(os.environ.get("MODEL_THRESHOLD", "0"))
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "1024"))

# ---------------------------------------------------------------------------
# Appliance names (default PLAID-15)
# ---------------------------------------------------------------------------
DEFAULT_NAMES = [
    'Air Conditioner', 'Blender', 'Coffee maker',
    'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Laptop',
    'Microwave', 'Soldering Iron', 'Vacuum', 'Washing Machine',
]

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
model = None
threshold = 0.5
appliance_names: list[str] = []
n_classes = 0
startup_time = 0.0


def load_model_once():
    """Load and warm up the model. Called once at startup."""
    global model, threshold, appliance_names, n_classes, startup_time
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from fusion_resnet import FusionResNet

    t0 = time.time()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]

    # Detect class count
    cls_key = [k for k in state if "classifier" in k and "weight" in k][-1]
    n_classes = state[cls_key].shape[0]

    # ICA parameters baked into checkpoint
    U = state["ica_branch.ica.U"].numpy()
    M = state["ica_branch.ica.M"].numpy()
    m = state["ica_branch.norm.m"].numpy()
    s = state["ica_branch.norm.s"].numpy()

    model = FusionResNet(
        n_classes=n_classes,
        signal_length=400,
        U=U, M=M, m=m, s=s,
    ).float()
    model.load_state_dict(state)
    model.eval()

    # Threshold
    threshold = THRESHOLD if THRESHOLD > 0 else ckpt.get("threshold", 0.5)

    # Names
    appliance_names = DEFAULT_NAMES[:n_classes]

    # Warm-up forward pass
    with torch.no_grad():
        model(torch.randn(1, 400))

    startup_time = time.time() - t0
    print(f"Model loaded in {startup_time:.2f}s — "
          f"{n_classes} classes, threshold={threshold:.3f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_once()
    yield


app = FastAPI(
    title="Fusion-ResNet NILM",
    description="Non-Intrusive Load Monitoring — appliance detection from current waveform",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single window of 400 current samples (normalized to unit magnitude)."""
    signal: list[float] = Field(..., min_length=400, max_length=400,
                                description="400-sample current waveform window")


class BatchPredictRequest(BaseModel):
    """Batch of windows."""
    signals: list[list[float]] = Field(..., min_length=1,
                                       description="List of 400-sample windows")


class Prediction(BaseModel):
    appliances: list[str] = Field(description="Detected appliance names")
    probabilities: dict[str, float] = Field(description="Per-class probabilities")


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    n_windows: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    n_classes: int
    threshold: float
    device: str
    startup_s: float
    parameters: int


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run model on (N, 400) array, return predictions & probabilities."""
    x = torch.from_numpy(signals).float()
    logits = model(x)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    return preds, probs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="FusionResNet",
        n_classes=n_classes,
        threshold=threshold,
        device=DEVICE,
        startup_s=round(startup_time, 2),
        parameters=sum(p.numel() for p in model.parameters()),
    )


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    signal = np.array(req.signal, dtype=np.float32).reshape(1, -1)
    preds, probs = predict_batch(signal)

    detected = [appliance_names[i] for i in range(n_classes) if preds[0, i]]
    prob_dict = {appliance_names[i]: round(float(probs[0, i]), 4)
                 for i in range(n_classes)}

    return Prediction(appliances=detected, probabilities=prob_dict)


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch_endpoint(req: BatchPredictRequest):
    if len(req.signals) > MAX_BATCH:
        raise HTTPException(400, f"Max batch size is {MAX_BATCH}")

    for i, sig in enumerate(req.signals):
        if len(sig) != 400:
            raise HTTPException(400, f"Signal {i} has {len(sig)} samples, expected 400")

    signals = np.array(req.signals, dtype=np.float32)
    t0 = time.time()
    preds, probs = predict_batch(signals)
    latency = (time.time() - t0) * 1000

    results = []
    for i in range(len(signals)):
        detected = [appliance_names[j] for j in range(n_classes) if preds[i, j]]
        prob_dict = {appliance_names[j]: round(float(probs[i, j]), 4)
                     for j in range(n_classes)}
        results.append(Prediction(appliances=detected, probabilities=prob_dict))

    return BatchPrediction(
        predictions=results,
        n_windows=len(signals),
        latency_ms=round(latency, 2),
    )

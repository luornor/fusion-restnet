"""
Export Fusion-ResNet to TorchScript for production deployment.
Produces a self-contained .pt file that doesn't need Python source at runtime.

Usage:
    python deploy/export_torchscript.py
    python deploy/export_torchscript.py --checkpoint checkpoints/best.pt --output deploy/model_traced.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion_resnet import FusionResNet


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "deploy" / "model_traced.pt"))
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]

    # Detect n_classes
    cls_key = [k for k in state if "classifier" in k and "weight" in k][-1]
    n_classes = state[cls_key].shape[0]

    # ICA params
    U = state["ica_branch.ica.U"].numpy()
    M = state["ica_branch.ica.M"].numpy()
    m = state["ica_branch.norm.m"].numpy()
    s = state["ica_branch.norm.s"].numpy()

    model = FusionResNet(
        n_classes=n_classes, signal_length=400,
        U=U, M=M, m=m, s=s,
    ).float()
    model.load_state_dict(state)
    model.eval()

    # Trace
    dummy = torch.randn(1, 400)
    traced = torch.jit.trace(model, dummy)

    # Verify
    with torch.no_grad():
        out_orig = model(dummy)
        out_traced = traced(dummy)
    assert torch.allclose(out_orig, out_traced, atol=1e-5), "Traced model output mismatch!"

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    traced.save(args.output)
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"Saved TorchScript model: {args.output} ({size_mb:.2f} MB)")

    # Also save metadata alongside
    meta = {
        "n_classes": n_classes,
        "signal_length": 400,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "epoch": int(ckpt.get("epoch", 0)),
        "best_val_f1": float(ckpt.get("best_val_f1", 0)),
        "format": "torchscript_traced",
        "dtype": "float32",
    }
    import json
    meta_path = args.output.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

"""
Fusion-ResNet NILM — Deployment Guide
======================================

Model Specifications & System Requirements
"""


# =============================================================================
# MODEL PROFILE
# =============================================================================
MODEL_SPECS = {
    "architecture": "FusionResNet (4-branch: Raw + ICA + Fryze + FFT → Attention Fusion)",
    "parameters": "1,334,721 (~1.33M)",
    "checkpoint_fp32": "5.09 MB (weights only) / 15.4 MB (full checkpoint w/ optimizer)",
    "torchscript_traced": "5.52 MB",
    "input_shape": "(batch, 400)  — 400 current samples per window",
    "output_shape": "(batch, 15)  — 15 appliance logits (multi-label, sigmoid → threshold)",
    "dtype_training": "float32",
    "best_f1": 0.802,
    "threshold": 0.384,
}


# =============================================================================
# SYSTEM REQUIREMENTS
# =============================================================================
REQUIREMENTS = {
    # -------------------------------------------------------------------------
    # MINIMUM (edge / Raspberry Pi class)
    # -------------------------------------------------------------------------
    "minimum": {
        "cpu": "ARM Cortex-A72 (4-core) or equivalent x86 (e.g., Intel Atom)",
        "ram": "512 MB total (model uses ~80-120 MB with PyTorch runtime)",
        "storage": "200 MB (model + runtime; 1 GB with full PyTorch)",
        "gpu": "Not required — model runs entirely on CPU",
        "os": "Linux (aarch64 or x86_64), Windows, macOS",
        "python": "3.10+",
        "expected_latency": {
            "single_window": "~30-50 ms on ARM, ~7 ms on x86 laptop CPU",
            "batch_256": "~500 ms on ARM, ~85 ms on x86 laptop CPU",
            "throughput": "~500-3000 windows/sec depending on CPU",
        },
        "notes": "The model is only 1.3M params — very lightweight by DL standards. "
                 "A Raspberry Pi 4 (4GB) can run it comfortably.",
    },

    # -------------------------------------------------------------------------
    # RECOMMENDED (cloud / edge server)
    # -------------------------------------------------------------------------
    "recommended": {
        "cpu": "2 vCPUs (x86_64) — e.g., AWS t3.small, GCP e2-small",
        "ram": "2 GB (comfortable headroom for FastAPI + preprocessing)",
        "storage": "2 GB (model + dependencies + logs)",
        "gpu": "Not needed — CPU is sufficient for real-time inference",
        "docker_image_size": "~1.5 GB (python:3.11-slim + CPU PyTorch + deps)",
        "expected_latency": {
            "single_window": "~5-10 ms",
            "batch_256": "~50-100 ms",
            "throughput": "~3000-5000 windows/sec",
        },
    },

    # -------------------------------------------------------------------------
    # WHAT EACH WINDOW REPRESENTS
    # -------------------------------------------------------------------------
    "context": {
        "window_duration": "~167 ms (10 cycles of 60 Hz mains)",
        "real_time_requirement": "Process 1 window every 167 ms → 6 windows/sec",
        "implication": "Even the minimum setup (500 windows/sec) is 80x faster "
                       "than real-time. Batch processing of hours of data takes seconds.",
    },
}


# =============================================================================
# CLOUD DEPLOYMENT OPTIONS
# =============================================================================
DEPLOYMENT_OPTIONS = {

    # -------------------------------------------------------------------------
    # Option 1: Docker + Any Cloud (Recommended)
    # -------------------------------------------------------------------------
    "docker_cloud": {
        "description": "Containerized FastAPI server, deploy anywhere",
        "steps": [
            "1. Build:         docker build -t fusion-resnet-nilm .",
            "2. Test locally:  docker run -p 8000:8000 fusion-resnet-nilm",
            "3. Push to registry:",
            "   - AWS ECR:     aws ecr get-login-password | docker login ... && docker push",
            "   - GCP GCR:     gcloud auth configure-docker && docker push gcr.io/PROJECT/fusion-resnet-nilm",
            "   - Docker Hub:  docker push username/fusion-resnet-nilm",
            "4. Deploy to cloud:",
            "   - AWS ECS/Fargate: Create task definition → service (0.5 vCPU, 1 GB RAM)",
            "   - GCP Cloud Run:   gcloud run deploy --image=gcr.io/PROJECT/fusion-resnet-nilm --memory=1Gi --cpu=1",
            "   - Azure Container Instances: az container create ...",
        ],
        "estimated_cost": "$5-15/month for a small always-on instance",
        "pros": ["Portable", "Easy scaling", "Consistent environment"],
    },

    # -------------------------------------------------------------------------
    # Option 2: Serverless (AWS Lambda / GCP Cloud Functions)
    # -------------------------------------------------------------------------
    "serverless": {
        "description": "Pay-per-request, zero cost when idle",
        "feasibility": "POSSIBLE — model is small enough (5 MB weights + ~300 MB PyTorch layer)",
        "steps": [
            "1. Package model + code as Lambda layer or container image",
            "2. AWS Lambda: Use container image deployment (up to 10 GB image)",
            "3. Set memory to 1024-2048 MB, timeout 30s",
            "4. Cold start: ~5-10s (PyTorch import), warm: <100ms per request",
        ],
        "estimated_cost": "$0-2/month for light usage (pay per invocation)",
        "pros": ["Cheapest for sporadic use", "Auto-scaling to zero"],
        "cons": ["Cold starts (5-10s)", "Complex packaging"],
    },

    # -------------------------------------------------------------------------
    # Option 3: Edge Device (Raspberry Pi / Jetson Nano)
    # -------------------------------------------------------------------------
    "edge_device": {
        "description": "Run directly on edge hardware near the electrical panel",
        "supported_devices": [
            "Raspberry Pi 4/5 (4 GB RAM) — works great",
            "NVIDIA Jetson Nano — overkill but works",
            "Any ARM/x86 SBC with 512+ MB RAM",
            "Intel NUC or similar mini-PC",
        ],
        "steps": [
            "1. Install Python 3.10+ and PyTorch (CPU wheel for ARM)",
            "2. Copy model checkpoint + inference code",
            "3. Run FastAPI server or standalone inference script",
            "4. OR use TorchScript traced model (no Python source needed):",
            "   python deploy/export_torchscript.py",
            "5. Connect current sensor (e.g., SCT-013) via ADC to Pi GPIO",
        ],
        "pros": ["No cloud costs", "Low latency", "Data privacy"],
        "cons": ["Requires physical setup", "Manual updates"],
    },

    # -------------------------------------------------------------------------
    # Option 4: TorchScript Export (Any C++ / Mobile / Embedded)
    # -------------------------------------------------------------------------
    "torchscript": {
        "description": "Export to TorchScript for C++/mobile deployment without Python",
        "export_command": "python deploy/export_torchscript.py",
        "output_size": "5.52 MB",
        "usage": [
            "Python: model = torch.jit.load('deploy/model_traced.pt')",
            "C++:    auto model = torch::jit::load('model_traced.pt');",
            "Mobile: Use PyTorch Mobile (Android/iOS) with the same .pt file",
        ],
        "note": "ONNX export is NOT supported due to aten::fft_rfft operator. "
                "TorchScript tracing works perfectly.",
    },
}


# =============================================================================
# QUICK-START: Deploy to cloud in 5 minutes
# =============================================================================
QUICKSTART = """
╔══════════════════════════════════════════════════════════════════╗
║  QUICK-START: Deploy Fusion-ResNet NILM to the Cloud           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Install deployment deps:                                     ║
║     pip install fastapi uvicorn[standard]                        ║
║                                                                  ║
║  2. Test locally:                                                ║
║     cd Fusion-ResNet-NILM                                        ║
║     uvicorn deploy.serve:app --host 0.0.0.0 --port 8000         ║
║                                                                  ║
║  3. Test endpoint:                                               ║
║     curl http://localhost:8000/health                            ║
║     curl -X POST http://localhost:8000/predict                   ║
║       -H "Content-Type: application/json"                        ║
║       -d '{"signal": [0.1, 0.2, ...]}'     (400 samples)        ║
║                                                                  ║
║  4. Build Docker image:                                          ║
║     docker build -t fusion-resnet-nilm .                         ║
║                                                                  ║
║  5. Deploy to GCP Cloud Run (cheapest always-on):                ║
║     gcloud run deploy fusion-resnet-nilm                         ║
║       --image gcr.io/PROJECT/fusion-resnet-nilm                  ║
║       --memory 1Gi --cpu 1 --port 8000                           ║
║                                                                  ║
║  Estimated cloud cost: $5-15/month (always-on)                   ║
║                         $0-2/month (serverless / pay-per-use)    ║
╚══════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(QUICKSTART)

    print("\n" + "=" * 65)
    print("  MODEL SPECIFICATIONS")
    print("=" * 65)
    for k, v in MODEL_SPECS.items():
        print(f"  {k:<25s}: {v}")

    print("\n" + "=" * 65)
    print("  MINIMUM SYSTEM REQUIREMENTS")
    print("=" * 65)
    for k, v in REQUIREMENTS["minimum"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2:<20s}: {v2}")
        else:
            print(f"  {k:<25s}: {v}")

    print("\n" + "=" * 65)
    print("  RECOMMENDED (CLOUD)")
    print("=" * 65)
    for k, v in REQUIREMENTS["recommended"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2:<20s}: {v2}")
        else:
            print(f"  {k:<25s}: {v}")

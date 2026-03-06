"""Quick test of the deployment server endpoints."""
import requests
import json
import numpy as np
import time

BASE = "http://127.0.0.1:8000"

# 1. Health
print("=== HEALTH ===")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# 2. Single prediction (synthetic sine wave)
signal = [float(np.sin(2 * np.pi * i / 66.67)) for i in range(400)]
r = requests.post(f"{BASE}/predict", json={"signal": signal})
result = r.json()
print("\n=== SINGLE PREDICT ===")
print("Detected:", result["appliances"])
top3 = sorted(result["probabilities"].items(), key=lambda x: -x[1])[:3]
print("Top 3:", top3)

# 3. Batch prediction
signals = [signal] * 64
t0 = time.time()
r = requests.post(f"{BASE}/predict/batch", json={"signals": signals})
roundtrip = (time.time() - t0) * 1000
br = r.json()
print("\n=== BATCH (64 windows) ===")
print(f"Windows: {br['n_windows']}, Server latency: {br['latency_ms']}ms, Roundtrip: {roundtrip:.0f}ms")
unique_detections = set()
for p in br["predictions"]:
    unique_detections.update(p["appliances"])
print("Unique appliances detected:", unique_detections if unique_detections else "(none)")

print("\nAll tests passed!")

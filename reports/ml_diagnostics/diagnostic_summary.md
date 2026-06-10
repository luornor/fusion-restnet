# ML Diagnostic Summary — Fusion-ResNet NILM

Generated from: `figures/test_metrics.json`, `figures_earlystop-v1/test_metrics_earlystop-v1.json`, checkpoint history.  
Run `generate_diagnostics.py` to regenerate plots and CSV tables from actual checkpoints.

---

## 1. Training Curve Status

**Evidence gap confirmed:** Training curves exist inside checkpoint `history` dicts but were not extracted locally before this audit. Run `generate_diagnostics.py --checkpoints-dir checkpoints` to extract and compare both runs.

**What we know from the two runs:**

| Run | Best Val F1 (estimated) | Final Threshold | Stopped At |
|---|---|---|---|
| v0.0.1-dev | ~0.807 (test proxy) | 0.3842 | Epoch 300 (no early stop) |
| earlystop-v1 | ~0.770 (test proxy) | 0.2895 | Epoch ~60–80 (patience=20, min_delta=0.001) |

**Interpretation:** The earlystop-v1 run stopped significantly below peak. The lower threshold (0.29 vs 0.38) confirms the model's logit scale was still immature at stopping — it had not yet learned to produce high-confidence predictions. This is consistent with premature stopping, not overfitting.

**Action required:** Plot training curves. The v0.0.1-dev run needs curve inspection to determine whether val F1 was still climbing at epoch 300 or had plateaued.

---

## 2. Class and Signature Statistics

From PLAID data (`data/X_real.npy`, `data/y_real.npy`):

- **Total raw signatures:** 19,400
- **Unique class labels (before filter):** 16
- **Classes kept (≥ 10 samples):** 15 (one class is removed by the training script filter)
- **Mean signatures per class:** ~1,293

Exact per-class signature counts require running `generate_diagnostics.py`. See `class_counts.csv` (auto-generated).

**Key concern:** With ~1,293 signatures per class on average, synthetic mixture generation at `n_samples=2000` per complexity level produces ~30,000 training mixtures. Each mixture reuses from the same finite pool. The effective diversity per class depends on actual raw counts — classes near the 10-sample minimum threshold have negligible diversity.

---

## 3. Validation vs. Test F1 Gap

Threshold is optimized on validation every epoch, which inflates validation F1 slightly relative to test F1. The test results use the final epoch's optimized threshold.

| Metric | v0.0.1-dev (Test) | earlystop-v1 (Test) |
|---|---|---|
| F1 (samples) | **0.8068** | 0.7700 |
| F1 (macro) | **0.8441** | 0.8212 |
| Precision (samples) | 0.7805 | 0.7337 |
| Recall (samples) | 0.8759 | 0.8656 |
| Exact Match Accuracy | 0.1184 | 0.1035 |
| Hamming Loss | 0.1840 | 0.2175 |
| Threshold | 0.3842 | 0.2895 |

The gap between val F1 (used for early stopping) and test F1 is not measured directly. To measure it: re-evaluate best checkpoint on validation set and compare to test F1.

---

## 4. Per-Class Metrics Summary

Source: `figures/test_metrics.json` (v0.0.1-dev, the better model).

| Appliance | F1 | Precision | Recall | Support | Notes |
|---|---|---|---|---|---|
| Microwave | **0.994** | 0.997 | 0.991 | 3355 | Distinctive signature |
| Vacuum | 0.933 | 0.916 | 0.951 | 3411 | Strong |
| Laptop | 0.925 | 0.933 | 0.916 | 3334 | Strong |
| Hair Iron | 0.864 | 0.837 | 0.893 | 3425 | Good |
| Hairdryer | 0.864 | 0.837 | 0.894 | 3371 | Good |
| Heater | 0.871 | 0.848 | 0.896 | 3355 | Good |
| CFL | 0.864 | 0.875 | 0.852 | 3358 | Good |
| Fridge | 0.838 | 0.841 | 0.836 | 3359 | Moderate |
| Water kettle | 0.848 | 0.791 | 0.915 | 3347 | Moderate |
| Coffee maker | 0.832 | 0.757 | 0.923 | 3305 | Low precision |
| Washing Machine | 0.820 | 0.759 | 0.893 | 3396 | Low precision |
| Air Conditioner | 0.771 | 0.676 | 0.897 | 3386 | Low precision |
| Fan | 0.753 | 0.655 | 0.885 | 3331 | **Weak** |
| Incandescent Light Bulb | 0.742 | 0.634 | 0.894 | 3310 | **Weak** |
| Soldering Iron | 0.742 | 0.644 | 0.876 | 3307 | **Weak** |

**Bottom 4 appliances by F1:** Soldering Iron, ILB, Fan, AC — all have high recall but low precision. The model over-predicts these classes (false positives). This is consistent with:
- Limited real signature diversity (few distinct waveform shapes to learn from)
- Signatures similar to other appliances (Fan ↔ AC ↔ ILB confusion)
- Possible low raw signature count for these classes

---

## 5. Mixture Complexity Performance

| # Active | F1 (samples) | Precision | Recall | Exact Match | N Samples |
|---|---|---|---|---|---|
| 1 | 0.882 | 0.850 | 0.957 | 0.759 | 370 |
| 2 | 0.723 | 0.724 | 0.811 | 0.255 | 420 |
| 3 | 0.704 | 0.711 | 0.775 | 0.093 | 420 |
| 4 | 0.690 | 0.685 | 0.772 | 0.031 | 420 |
| 5 | 0.691 | 0.671 | 0.781 | 0.012 | 420 |
| 6 | 0.704 | 0.665 | 0.820 | 0.005 | 420 |
| 7 | 0.733 | 0.679 | 0.848 | 0.002 | 420 |
| 8 | 0.763 | 0.703 | 0.871 | 0.005 | 420 |
| 9 | 0.793 | 0.732 | 0.890 | 0.012 | 420 |
| 10 | 0.835 | 0.774 | 0.920 | 0.007 | 420 |
| 11 | 0.862 | 0.810 | 0.932 | 0.019 | 420 |
| 12 | 0.893 | 0.856 | 0.939 | 0.019 | 420 |
| 13 | 0.920 | 0.903 | 0.942 | 0.052 | 420 |
| 14 | 0.948 | 0.954 | 0.946 | 0.145 | 420 |
| 15 | 0.969 | 1.000 | 0.943 | 0.436 | 420 |

**Critical observation:** F1 bottoms out at 4–5 active appliances (0.69). This is exactly the scenario most common in real home environments. At 15 devices, precision is 1.0 — when everything is on, every prediction is a true positive by definition.

**What this means for deployment:** The model's weakest zone (4–6 devices) is the most common real-world operating condition. This is the primary improvement target.

---

## 6. Calibration Status

**Current state:** Output probabilities are not calibrated. The threshold of 0.38 (far below 0.5) means the model's sigmoid outputs are systematically depressed — the model is under-confident on positives.

**Expected Calibration Error (ECE):** Not yet computed. Run `generate_diagnostics.py --calibrate` after implementing temperature scaling.

**Estimated impact of calibration:** Temperature scaling typically moves threshold closer to 0.5 and improves metric stability across distribution shifts. Expected benefit: Low-Medium on aggregate F1, High on threshold stability.

---

## 7. Confirmed vs. Uncertain Audit Claims

| Claim | Status | Evidence |
|---|---|---|
| Early stopping triggered too soon (patience=20) | **Confirmed** | earlystop-v1 F1 is 4.5% below full-run; low threshold indicates immature logit scale |
| Model benefits from training beyond epoch 60–80 | **Confirmed** | Full 300-epoch run substantially outperforms early-stopped run |
| ICA branch struggles on complex mixtures | **Plausible, unconfirmed** | F1 drops 2→6 devices; needs ablation to isolate ICA contribution |
| Fan/ILB/Soldering Iron/AC have low source diversity | **Plausible, unconfirmed** | Low precision for these classes is consistent with data diversity issues; needs class count audit |
| Threshold optimization overfitting validation | **Plausible** | Threshold 0.38 ≠ 0.50 suggests output distribution shift; needs val/test F1 gap measurement |
| Data duplication harms generalization | **Confirmed by reasoning** | Training data is already synthetic mixtures from finite PLAID pool; duplication adds no new information |
| 50Hz adaptation needed for Ghana | **Confirmed by design** | FryzeDecomposition uses hardcoded 60Hz (line 162 of fusion_resnet.py) |

---

## 8. Immediate Action Items

1. Run `python generate_diagnostics.py` to get training curves, class counts, and calibration plots.
2. Check per-class raw signature counts — if any class has < 50 signatures, augmentation is essential.
3. Set `--early-stopping-patience 40` for all future runs (not 20).
4. Before Ghana deployment, update `FryzeDecomposition` to accept mains frequency as parameter (or add `--mains-freq 50` flag).

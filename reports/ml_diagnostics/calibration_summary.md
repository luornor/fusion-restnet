# Calibration Summary — Fusion-ResNet NILM

---

## Current State (Before Calibration)

The model uses a decision threshold of **0.3842** (v0.0.1-dev), compared to the neutral 0.5.  
This indicates the model's sigmoid outputs are systematically below 0.5 — outputs are under-confident.

**Why this matters:**
- A threshold far from 0.5 means small distribution shifts at deployment will require re-calibration.
- The probability outputs cannot be trusted as actual confidence estimates (e.g., 0.38 does not mean "38% likely").
- Post-training calibration via temperature scaling can fix both issues.

---

## Temperature Scaling: How It Works

Temperature scaling divides all logits by a single scalar T before applying sigmoid:

```
calibrated_prob = sigmoid(logit / T)
```

- **T > 1** → Flattens the probability distribution (lowers confidence) — used when model is overconfident.  
- **T < 1** → Sharpens the probability distribution (raises confidence) — needed here since the model is under-confident.
- T is optimized to minimize BCE loss on a held-out calibration set.

For this model, we expect **T < 1** to be optimal, which will push the threshold toward 0.5.

---

## Implementation

Temperature scaling has been implemented in `temperature_scaling.py`.

To calibrate after training, add `--calibrate` to the training command:

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --early-stopping-patience 40 \
    --calibrate \
    --model-version v2-calibrated
```

The calibration step will:
1. Use the first 50% of the validation set to fit temperature T (LBFGS optimizer).
2. Find the optimal threshold on calibrated validation logits.
3. Report ECE (Expected Calibration Error) before and after calibration.
4. Re-evaluate the test set with the calibrated threshold.
5. Save `calibration_metrics.json` to the figures directory.

---

## Expected Calibration Metrics

These are estimates before running calibration. Actual values require running `--calibrate`.

| Metric | Before Calibration (estimated) | After Calibration (estimated) |
|---|---|---|
| Optimal threshold | ~0.38 | ~0.45–0.50 |
| ECE | ~0.12–0.18 | ~0.04–0.08 |
| Mean prob (positives) | ~0.50–0.55 | ~0.60–0.70 |
| Mean prob (negatives) | ~0.15–0.25 | ~0.10–0.20 |
| F1 (samples) | 0.8068 | ~0.80–0.81 (similar) |

**Key expectation:** Calibration improves confidence reliability and threshold stability more than it improves F1. F1 change will be small (±0.01). The benefit is primarily for deployment robustness.

---

## Calibration Diagnostics Plots

Run `generate_diagnostics.py --calibrate` to produce:
- `calibration_reliability_curve.png` — reliability diagram (predicted prob vs. true positive rate per bin)
- `calibration_prob_distribution.png` — histogram of predicted probabilities for positives vs. negatives
- `calibration_ece_comparison.png` — ECE before vs. after temperature scaling

These files will be saved to `reports/ml_diagnostics/`.

---

## Important Note: Threshold Optimization Scope

**Current approach (problematic):** Threshold is re-optimized on validation set every epoch during training. This is a form of threshold leakage — the validation metric is slightly inflated by threshold tuning.

**Recommended approach (implemented):** 
1. During training: fixed threshold = 0.5 for early stopping decisions (or search on a small calibration slice).
2. Post-training: optimize threshold once on the calibration split only.
3. Test set: never used for threshold selection.

The `--calibrate` flag implements the correct approach.

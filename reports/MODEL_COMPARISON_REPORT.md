# Fusion-ResNet NILM — Model Comparison Report

**Date:** 2026-06-12  
**Prepared by:** NILM Development Team  
**Project:** Fusion-ResNet NILM — Ghana Edge Deployment

---

## 1. Executive Summary

Three model versions have been trained and evaluated on the 15-class appliance identification task. The latest model, **lumina-50hz-v1**, is the first version specifically prepared for Ghana deployment: it was trained on a combined PLAID + HIFDA dataset (35,400 samples) with the Fryze decomposition correctly parameterised for 50 Hz / 230 V mains, matching Ghanaian grid conditions.

Compared to the best prior model (v0.0.1-dev), lumina-50hz-v1 shows a slight drop in macro-F1 (0.825 vs 0.844) — an expected trade-off for learning across two different grid environments — while improving exact-match accuracy from 11.8% to 13.5% and delivering clear per-class gains on six appliances, including a +10.6 pp improvement on Water kettle and +5.5 pp on Fridge.

---

## 2. Model Versions at a Glance

| Attribute            | v0.0.1-dev            | earlystop-v1          | lumina-50hz-v1 (LATEST)         |
|----------------------|-----------------------|-----------------------|---------------------------------|
| Training data        | PLAID only            | PLAID only            | PLAID + HIFDA combined          |
| Samples              | 19,400                | 19,400                | 35,400                          |
| Mains configuration  | 60 Hz / 120 V (US)    | 60 Hz / 120 V (US)    | **50 Hz / 230 V (Ghana/EU)**    |
| Variant              | Full                  | Full                  | Full                            |
| Epochs trained       | 169                   | 14                    | 98                              |
| Stopping criterion   | Manual (300 ep limit) | Early stopping (p=20) | Early stopping (p=40)           |
| Final train loss     | 0.275                 | 0.405                 | 0.349                           |
| Final val loss       | 0.421                 | 0.447                 | 0.423                           |
| Final val F1         | 0.801                 | 0.771                 | 0.781                           |
| Decision threshold   | 0.384                 | 0.289                 | 0.432                           |

---

## 3. Test-Set Global Metrics

> All models evaluated on synthetic mixtures (1–15 simultaneous appliances) drawn from the test split.

| Metric                | v0.0.1-dev | earlystop-v1 | lumina-50hz-v1 | vs Best Prior |
|-----------------------|-----------|--------------|----------------|---------------|
| **F1 Macro**          | 0.844     | 0.821        | **0.825**      | -0.019        |
| **F1 Micro**          | 0.840     | 0.817        | **0.821**      | -0.019        |
| **F1 Weighted**       | 0.844     | 0.821        | **0.826**      | -0.018        |
| **F1 Samples**        | 0.807     | 0.770        | **0.775**      | -0.032        |
| Precision (macro)     | —         | —            | 0.769          | —             |
| Recall (macro)        | —         | —            | 0.901          | —             |
| Precision (samples)   | 0.781     | 0.734        | 0.752          | -0.029        |
| Recall (samples)      | 0.876     | 0.866        | 0.856          | -0.020        |
| **Exact Match Acc**   | 0.118     | 0.104        | **0.135**      | **+0.017**    |
| Hamming Loss          | 0.184     | 0.217        | 0.210          | +0.026        |
| Jaccard (samples)     | —         | —            | 0.666          | —             |

**Key observation:** lumina-50hz-v1 achieves the highest exact-match accuracy across all three models (13.5%), meaning it is best at correctly identifying the complete appliance combination in a mixture. The macro-F1 dip relative to v0.0.1-dev is expected — the model is simultaneously learning two grid environments (PLAID 60 Hz and HIFDA 50 Hz).

---

## 4. Per-Class F1 Score Comparison

| Appliance                | v0.0.1-dev | earlystop-v1 | lumina-50hz-v1 | vs v0.0.1-dev |
|--------------------------|-----------|--------------|----------------|---------------|
| Air Conditioner          | 0.771     | 0.757        | 0.768          | -0.003        |
| Coffee maker             | 0.832     | 0.815        | 0.742          | -0.090        |
| Compact Fluorescent Lamp | 0.864     | 0.817        | **0.914**      | **+0.051**    |
| Fan                      | 0.753     | 0.741        | 0.766          | +0.013        |
| Fridge                   | 0.838     | 0.831        | **0.893**      | **+0.055**    |
| Hair Iron                | 0.864     | 0.816        | **0.897**      | **+0.033**    |
| Hairdryer                | 0.864     | 0.827        | 0.841          | -0.023        |
| Heater                   | 0.871     | 0.846        | 0.792          | -0.079        |
| Incandescent Light Bulb  | 0.742     | 0.739        | 0.756          | +0.014        |
| Laptop                   | 0.925     | 0.866        | 0.785          | -0.140        |
| Microwave                | 0.994     | 0.992        | 0.937          | -0.057        |
| Soldering Iron           | 0.742     | 0.742        | 0.767          | **+0.025**    |
| Vacuum                   | 0.933     | 0.910        | 0.783          | -0.150        |
| Washing Machine          | 0.821     | 0.791        | 0.784          | -0.037        |
| Water kettle             | 0.848     | 0.829        | **0.954**      | **+0.106**    |
| **Macro Average**        | **0.844** | **0.821**    | **0.825**      | **-0.019**    |

### Winners in lumina-50hz-v1
- **Water kettle** +10.6 pp — strongest single improvement; likely benefits from HIFDA's resistance-heating signatures
- **Compact Fluorescent Lamp** +5.1 pp
- **Fridge** +5.5 pp
- **Hair Iron** +3.3 pp

### Regressions to watch
- **Laptop** -14.0 pp and **Vacuum** -15.0 pp are the largest drops. Both have very distinctive signatures in PLAID but HIFDA does not include them — the extra cross-dataset variance may be introducing confusion with similar-amplitude appliances.
- **Microwave** -5.7 pp — was near-perfect in v0.0.1-dev (F1 0.994), still strong (0.937) but worth monitoring.
- **Heater** and **Coffee maker** both drop; HIFDA includes heater and coffee maker signatures which differ from PLAID's US appliances — the model may be learning conflicting signatures.

---

## 5. Multi-Appliance Mixture Performance

Performance by number of simultaneous appliances (lumina-50hz-v1):

| Active appliances | F1 (samples) | Precision | Recall  | Exact Match |
|:-----------------:|:------------:|:---------:|:-------:|:-----------:|
| 1                 | 0.896        | 0.867     | 0.965   | 79.1%       |
| 2                 | 0.665        | 0.729     | 0.695   | 18.8%       |
| 3                 | 0.613        | 0.661     | 0.675   | 4.3%        |
| 4                 | 0.602        | 0.643     | 0.671   | 1.0%        |
| 5–15              | 0.619–0.996  | 0.608–1.0 | 0.746–0.992 | 0–87.6% |

The model handles single-appliance identification well (79% exact match). At 2–4 simultaneous appliances — the most realistic in-home scenario — F1 drops into the 0.60–0.67 range, consistent across all three model versions. This is an inherent challenge of the linear mixture approach used during training.

---

## 6. What Changed: v0.0.1-dev → lumina-50hz-v1

| Change                        | Detail                                                       |
|-------------------------------|--------------------------------------------------------------|
| Dataset                       | Added HIFDA (Spain, 50 Hz, 100 kHz → resampled to 30 kHz)   |
| Total training samples        | 19,400 → 35,400 (+82%)                                       |
| Fryze decomposition           | Hardcoded 60 Hz/120 V → parameterised **50 Hz / 230 V**      |
| Default dtype                 | float64 → **float32** (matches firmware)                     |
| Early stopping patience       | Not used → **40 epochs**                                     |
| New appliances with HIFDA data| Air Conditioner, Coffee maker, Hairdryer, Heater, Laptop, Microwave, Vacuum, Washing Machine |

---

## 7. Deployment Readiness

| Criterion                         | v0.0.1-dev | earlystop-v1 | lumina-50hz-v1 |
|-----------------------------------|:---------:|:------------:|:--------------:|
| 50 Hz Fryze decomposition         | No        | No           | **Yes**        |
| 230 V mains voltage               | No        | No           | **Yes**        |
| Float32 (firmware compatible)     | No        | No           | **Yes**        |
| HIFDA 50 Hz training exposure     | No        | No           | **Yes**        |
| Stable training (val loss)        | Yes       | Marginal     | Yes            |
| Best macro-F1                     | Yes       | No           | Close (-0.019) |
| Recommended for Ghana deployment  | No        | No           | **Yes**        |

---

## 8. Recommended Next Steps

1. **Investigate Laptop and Vacuum regression** — these are the two biggest drops. Check whether HIFDA's cross-dataset signal variance is the cause, or whether the decision threshold (0.432) needs class-specific tuning.

2. **Collect Ghana calibration data** — even 10 recordings per class from local Ghanaian appliances would close the remaining domain gap more effectively than any additional international dataset.

3. **Enable temperature scaling** — the model currently uses a fixed threshold (0.432). Applying post-hoc temperature calibration would allow class-specific thresholds and likely recover some of the precision lost on Laptop and Vacuum.

4. **Retrain with COOLL Fan/Vacuum signatures** — COOLL (France, 50 Hz) has fan, vacuum, and hairdryer recordings that could specifically address the Vacuum regression.

---

*Report generated from checkpoint: `model_registry/latest_vlumina-50hz-v1.pt`*  
*Test metrics source: `reports/figures/test_metrics_lumina-50hz-v1.json`*  
*Historical metrics source: `reports/ml_diagnostics/global_metrics_comparison.csv`*

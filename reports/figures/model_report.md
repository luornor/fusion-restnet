# Fusion-ResNet Model Report - multi-source-v1

## Global Metrics

| Metric | Value |
|---|---:|
| F1 samples | 0.7204 |
| F1 macro | 0.7874 |
| Precision samples | 0.6724 |
| Recall samples | 0.8677 |
| Exact match accuracy | 0.0996 |
| Hamming loss | 0.2714 |
| Threshold | 0.2895 |

## Training Snapshot

- Best validation F1: 0.7272 at epoch 100
- Final validation F1: 0.7139
- Final training F1: 0.8278

## Weakest Appliances

| Appliance | F1 | Precision | Recall | Support | Likely issue |
|---|---:|---:|---:|---:|---|
| Soldering Iron | 0.7216 | 0.5940 | 0.9191 | 3348 | false positives |
| Water kettle | 0.7231 | 0.5952 | 0.9210 | 3343 | false positives |
| Coffee maker | 0.7328 | 0.6022 | 0.9358 | 3350 | false positives |
| Washing Machine | 0.7553 | 0.6471 | 0.9069 | 3352 | false positives |
| Vacuum | 0.7565 | 0.6385 | 0.9278 | 3408 | false positives |

## Deployment-Relevant Mixture Band

| Active appliances | Mean F1 samples | Mean exact match | Samples |
|---:|---:|---:|---:|
| 2-5 | 0.5200 | 0.0232 | 1680 |

## Recommendations

- Prioritize false-positive reduction for: Soldering Iron, Water kettle, Coffee maker, Washing Machine, Vacuum. These classes have recall far above precision.
- Track the 2-5 active-appliance band separately; it is more deployment-relevant than the all-devices-on cases.
- Run calibration or class-specific threshold tuning before deployment; the global threshold is far from 0.50.

## Generated Artifacts

- `test_metrics.json` and versioned copy
- `global_metrics.csv`
- `per_class_metrics.csv`
- `mixture_complexity_metrics.csv`
- `training_curves.png`
- `per_appliance_f1.png`
- `f1_by_components.png`

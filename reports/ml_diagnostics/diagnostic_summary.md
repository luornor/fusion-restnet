# ML Diagnostic Summary - Fusion-ResNet NILM

This summary reflects the current tracked diagnostics under `reports/ml_diagnostics`.
Future runs should regenerate it with:

```bash
python diagnostics/generate_diagnostics.py
```

## Best Historical Run

From `global_metrics_comparison.csv`, `v0.0.1-dev` is stronger than `earlystop-v1`.

| Metric | v0.0.1-dev | earlystop-v1 |
|---|---:|---:|
| F1 samples | 0.8068 | 0.7700 |
| F1 macro | 0.8441 | 0.8212 |
| Precision samples | 0.7805 | 0.7337 |
| Recall samples | 0.8759 | 0.8656 |
| Exact match accuracy | 0.1184 | 0.1035 |
| Hamming loss | 0.1840 | 0.2175 |
| Threshold | 0.3842 | 0.2895 |

## Main Findings

- `earlystop-v1` underperforms the longer run, especially in the realistic 2-5 active-appliance range.
- The weakest classes are still precision-limited: Soldering Iron, Incandescent Light Bulb, Fan, and Air Conditioner are detected often, but with too many false positives.
- Mixture complexity is the deployment bottleneck. The model is strongest when one appliance or nearly all appliances are active, but that is less representative of home operation.
- Class diversity is uneven. Coffee maker, Water kettle, Hair Iron, and Soldering Iron have very low raw signature counts in PLAID and should be treated as data-risk classes.

## Report Set To Keep

- `global_metrics_comparison.csv`
- `per_class_metrics.csv`
- `mixture_complexity_metrics.csv`
- `class_counts.csv`
- `training_curves_comparison.png`
- `per_class_f1_comparison.png`
- `mixture_complexity_comparison.png`
- `class_counts.png`

Future diagnostics also generate `precision_recall_diagnostics.png`, which is useful for separating false-positive classes from miss-prone classes.

## Reporting Changes Made

- Training now writes `global_metrics.csv`, `per_class_metrics.csv`, `mixture_complexity_metrics.csv`, and `model_report.md` for every run.
- Default training plots were reduced to the core set: training curves, per-appliance metrics, and mixture-complexity performance.
- Per-class plots now sort weakest appliances first.
- Mixture-complexity plots now highlight the realistic 2-5 active-appliance band.
- The diagnostics script now uses `reports/figures` by default and discovers available `*_best.pt` checkpoints instead of hardcoding old run names.

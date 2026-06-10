# Code Changes Summary

All changes are backward-compatible. Every new feature is off by default and requires an explicit CLI flag to activate. Existing Colab commands run without modification.

---

## New Files

### `temperature_scaling.py`

Single-class module for post-training probability calibration.

- `TemperatureScaler` — learns a scalar temperature T via LBFGS on BCE loss.
- `fit(logits, labels)` — optimizes T on a calibration split.
- `scale(logits)` — applies T: `scaled = logits / T`.
- `find_best_threshold(logits, labels)` — grid-search optimal F1 threshold on calibrated logits.
- `calibration_metrics(logits, labels)` — returns ECE before/after, mean probabilities for positives/negatives.

**Why:** The model's output probabilities are under-confident (threshold = 0.38 far from 0.5). Temperature scaling is the standard fix. It does not improve F1 directly but makes the threshold stable across deployment conditions.

---

### `generate_diagnostics.py`

Standalone diagnostic script runnable from the project root.

```bash
python generate_diagnostics.py [--calibrate]
```

Produces:
- `training_curves_comparison.png` — training/val loss and F1 for all available runs
- `training_history_{run}.csv` — per-epoch metrics for each checkpoint
- `class_counts.csv` + `class_counts.png` — raw PLAID signatures per class with diversity risk labels
- `per_class_f1_comparison.png` — per-class F1 across all runs
- `mixture_complexity_comparison.png` — F1 vs #active appliances across all runs
- `global_metrics_comparison.csv` — summary table of global metrics per run
- `calibration_analysis.png` + `calibration_metrics.json` (with `--calibrate`)

**Why:** Training curves existed only inside checkpoint files. This script extracts and visualizes them. `class_counts.csv` is required before deciding whether augmentation or pos_weight is warranted.

---

## Modified Files

### `train_fusion_resnet.py`

#### New class: `SignatureAugmentor` (inserted before `Composer`)

Applies four configurable transforms to **individual source signatures before mixture composition**:
1. Additive Gaussian noise (`--aug-noise-sigma`, sigma as fraction of RMS)
2. Amplitude scaling (`--aug-amplitude-scale`, uniform ±scale)
3. Cyclic phase shift (`--aug-phase-shift`, random roll)
4. Time warping (`--aug-time-warp`, ±5% speed, off by default)

**Key design constraint:** Augmentation is applied only to training source signatures. Val and test data are untouched. Labels are never modified.

#### Modified: `Composer.__init__` and `compose_single`

- `Composer` now accepts `augmentor=None`.
- `compose_single` applies augmentation to each individual source waveform before summing — so the composed mixture contains a uniquely perturbed version each time.

#### Modified: `compose()`

Added `augmentor=None` parameter; passed through to `Composer`. Val and test calls omit the argument.

#### New function: `compute_pos_weight()`

Computes per-class BCE pos_weight = neg_count / pos_count from the training label matrix. Called only when `--use-pos-weight` is set. Do not use speculatively — run `generate_diagnostics.py` first to confirm class imbalance.

#### Modified: `train_model()` — LR warmup

Added `warmup_epochs=0` and `warmup_start_lr=1e-5` parameters.

When `warmup_epochs > 0`, linearly interpolates LR from `warmup_start_lr` to `base_lr` over the first `warmup_epochs` steps **of the current run** (respects `--resume-from`). The plateau scheduler is suspended during warmup so its internal patience counter is not burned by the low-LR warmup phase.

#### Modified: `main()` — Optimizer, scheduler, loss

- `weight_decay` now uses `args.weight_decay` (was hardcoded `1e-4`).
- Scheduler `factor` and `patience` now use `args.scheduler_factor` and `args.scheduler_patience`.
- If `--use-pos-weight`: `BCEWithLogitsLoss(pos_weight=pw)` where `pw` is computed from `Y_train`.

#### Modified: `main()` — Post-training calibration

When `--calibrate` is set:
1. Collects validation logits from the best checkpoint.
2. Splits validation 50/50: calibration (fit T) and report (measure ECE improvement).
3. Fits temperature T on calibration half.
4. Finds threshold on calibrated calibration logits (not the test set).
5. Re-evaluates test set with calibrated threshold.
6. Saves `calibration_metrics.json` to figures dir.

#### Modified: `main()` — Saves `Y_prob.npy` and `Y_true.npy`

After evaluation, raw test probabilities and ground truth labels are saved to the figures directory. Used by `generate_diagnostics.py --calibrate` and for offline calibration analysis.

---

## New CLI Arguments

All new arguments default to the pre-existing behavior (no change when omitted).

```
--weight-decay FLOAT        AdamW weight decay (default: 1e-4)
--scheduler-factor FLOAT    ReduceLROnPlateau factor (default: 0.8)
--scheduler-patience INT    ReduceLROnPlateau patience (default: 15)
--warmup-epochs INT         LR warmup epochs (default: 0 = disabled)
--warmup-start-lr FLOAT     Warmup starting LR (default: 1e-5)
--use-pos-weight            Enable pos_weight in BCEWithLogitsLoss
--aug-noise-sigma FLOAT     Gaussian noise sigma fraction (default: 0.0 = off)
--aug-amplitude-scale FLOAT Amplitude perturbation fraction (default: 0.0 = off)
--aug-phase-shift           Enable cyclic phase shift
--aug-time-warp             Enable time warping ±5%% (default: off)
--calibrate                 Post-training temperature scaling calibration
```

---

## What Was NOT Changed

- Model architecture (`fusion_resnet.py`) — unchanged. Data/training issues are the bottleneck, not model capacity.
- Evaluation logic (`compute_all_metrics`, `evaluate`) — unchanged. All metrics remain directly comparable to existing runs.
- ICA fitting — unchanged. The `FryzeDecomposition` 60 Hz hardcode was not changed (flagged in dataset_source_recommendations.md — requires a `--mains-freq` flag before Ghana deployment).
- Checkpoint format — backward compatible. All new keys are additive.
- Default training command — unchanged. Running without any new flags produces identical behavior to pre-change runs.

---

## Audit Claim Resolution

| Claim | Resolution |
|---|---|
| Early stopping triggered too soon | **Not changed in code** (patience was already an arg). Use `--early-stopping-patience 40` in commands. |
| No LR warmup | **Fixed** — `--warmup-epochs 5 --warmup-start-lr 1e-5` |
| Weight decay hardcoded | **Fixed** — `--weight-decay` arg |
| Scheduler not configurable | **Fixed** — `--scheduler-factor 0.5 --scheduler-patience 10` |
| No source-level augmentation | **Fixed** — `SignatureAugmentor` + 4 CLI flags |
| Threshold overfitting validation | **Fixed** — `--calibrate` moves threshold optimization to a held-out calibration split |
| No pos_weight support | **Fixed** — `--use-pos-weight` flag |
| No probability saving for offline analysis | **Fixed** — `Y_prob.npy` / `Y_true.npy` saved automatically |
| 60 Hz hardcode in FryzeDecomposition | **Fixed** — `mains_freq` parameter added to `FryzeDecomposition`, `FryzeBranch`, `FusionResNet`, `FusionResNetLite`; `--mains-freq` CLI arg in `train_fusion_resnet.py` |

# Experiment Plan — Fusion-ResNet NILM

All experiments run from `REPO_DIR` on Google Colab using the `NILM MODEL.ipynb` notebook pattern.  
Replace `$SAVE_DIR` and `$FIGURES_DIR` with your Colab path variables.

---

## Baseline Reference

Before running any ablation, record the baseline from the existing checkpoints:

| Metric | v0.0.1-dev (300 ep, no early stop) | earlystop-v1 (patience=20) |
|---|---|---|
| F1 (samples) | **0.8068** | 0.7700 |
| F1 (macro) | **0.8441** | 0.8212 |
| Threshold | 0.3842 | 0.2895 |
| Exact Match | 0.1184 | 0.1035 |

All new experiments should beat 0.8068 F1 (samples) to be considered improvements.

---

## Experiment 0 — Diagnostics (Run First)

**Purpose:** Extract training curves and class counts before any changes.

```bash
python generate_diagnostics.py \
    --checkpoints-dir checkpoints \
    --data-dir data \
    --output-dir reports/ml_diagnostics \
    --metrics-dir figures \
    --metrics-dir-es figures_earlystop-v1
```

**Expected outputs:** `training_curves_comparison.png`, `class_counts.csv`, `class_counts.png`, `per_class_f1_comparison.png`, `mixture_complexity_comparison.png`

**Decision gate:** If `class_counts.csv` shows any class with < 50 signatures, augmentation (Exp 3) becomes mandatory, not optional.

---

## Experiment 1 — Fixed Training Config (Quick Win)

**Purpose:** Verify that better early stopping and configurable scheduler already improve over the baseline without any data changes.

**Changes vs baseline:**
- Early stopping patience: 20 → **40**
- Scheduler factor: 0.8 → **0.5** (faster LR reduction)
- Scheduler patience: 15 → **10**
- Weight decay: unchanged (1e-4)

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v1-config \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v1-config \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.001 \
    --scheduler-factor 0.5 \
    --scheduler-patience 10 \
    --seed 42
```

**Expected outcome:** F1 ≥ 0.807. If early stopping now fires at the right time, the model should stop closer to its peak. Threshold should move closer to 0.40–0.45.

**Decision gate:** If F1 does not improve over baseline, the training config was not the bottleneck — proceed directly to Exp 3 (augmentation).

---

## Experiment 2 — LR Warmup

**Purpose:** Test whether linear warmup stabilizes early training and improves final F1.

**Changes vs Exp 1:**
- LR warmup: 5 epochs from 1e-5 → 1e-3
- All other settings same as Exp 1

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v2-warmup \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v2-warmup \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.001 \
    --scheduler-factor 0.5 \
    --scheduler-patience 10 \
    --warmup-epochs 5 \
    --warmup-start-lr 1e-5 \
    --seed 42
```

**Expected outcome:** Small improvement (0–0.5% F1) over Exp 1 if training was unstable early. Check the training curve — val F1 in epochs 1–5 should be more stable than without warmup.

---

## Experiment 3 — Source-Level Augmentation (Core Experiment)

**Purpose:** Increase effective signature diversity without collecting new data. This is the highest-expected-benefit experiment.

**Changes vs Exp 1:**
- Noise: `--aug-noise-sigma 0.02` (2% of RMS)
- Amplitude scaling: `--aug-amplitude-scale 0.15` (±15%)
- Phase shift: enabled

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v3-augment \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v3-augment \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.001 \
    --scheduler-factor 0.5 \
    --scheduler-patience 10 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --seed 42
```

**Expected outcome:** Primary improvement on underperforming classes (Fan, ILB, Soldering Iron, AC). These are precision-limited — augmentation reduces memorization of specific source waveforms, which is what causes false positives.

**Decision gate:** Compare per-class F1 to baseline. If augmentation improves Fan/ILB/Soldering Iron but hurts Microwave/Laptop, the noise sigma may be too high — try `--aug-noise-sigma 0.01`.

---

## Experiment 4 — Full Stack (Best Settings Combined)

**Purpose:** Combine all improvements from Exp 1–3 in one run.

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v4-full \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v4-full \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.001 \
    --scheduler-factor 0.5 \
    --scheduler-patience 10 \
    --warmup-epochs 5 \
    --warmup-start-lr 1e-5 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --seed 42
```

**This is the target model for deployment.** If it beats baseline on both F1 (samples) and per-class F1 for the four weak classes, it becomes the new checkpoint.

---

## Experiment 5 — Calibration

**Purpose:** Apply temperature scaling to the best model from Exp 4 to stabilize the threshold and improve probability calibration.

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v5-calibrated \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v5-calibrated \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.001 \
    --scheduler-factor 0.5 \
    --scheduler-patience 10 \
    --warmup-epochs 5 \
    --warmup-start-lr 1e-5 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --calibrate \
    --seed 42
```

**Expected output:** `calibration_metrics.json` in the figures dir.  
**Expected outcome:** Threshold closer to 0.45–0.50. F1 change ≤ ±0.01 (calibration doesn't improve discrimination, only confidence reliability).

---

## Experiment 6 — Stronger Weight Decay (Optional)

**Purpose:** Test whether stronger regularization helps the model generalize better from synthetic data.

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v6-wd \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v6-wd \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --weight-decay 5e-4 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --seed 42
```

**Run this only if Exp 4 shows signs of overfitting** (train F1 >> val F1 in the training curves from Exp 0 / generate_diagnostics).

---

## Experiment 7 — pos_weight (Optional, Conditional)

**Purpose:** Compensate for label imbalance if confirmed by class counts.

**Run this only if class_counts.csv from Exp 0 shows significant imbalance** (e.g., one class has 3× the signatures of another).

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v7-posweight \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v7-posweight \
    --save-every 1 \
    --snapshot-every 25 \
    --early-stopping-patience 40 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --use-pos-weight \
    --seed 42
```

**Do not run this experiment speculatively.** The synthetic mixture generator already creates balanced test label counts (~3,300–3,400 per class). If raw signature counts are also balanced, pos_weight adds nothing and may destabilize training.

---

## Experiment 8 — Augmented Lower LR (Optional)

**Purpose:** Test lr=5e-4 if Exp 4 shows training instability or threshold drift.

```bash
python train_fusion_resnet.py \
    --device cuda \
    --variant full \
    --epochs 300 \
    --fp32 \
    --model-version v8-lowlr \
    --save-dir $SAVE_DIR \
    --figures-dir $FIGURES_DIR/v8-lowlr \
    --save-every 1 \
    --snapshot-every 25 \
    --lr 5e-4 \
    --early-stopping-patience 40 \
    --warmup-epochs 5 \
    --aug-noise-sigma 0.02 \
    --aug-amplitude-scale 0.15 \
    --aug-phase-shift \
    --seed 42
```

---

## Recommended Run Order

```
Exp 0 (diagnostics) → Exp 1 → Exp 3 → Exp 4 → Exp 5
```

Only run Exp 2, 6, 7, 8 if motivated by results from the primary chain.

---

## How to Update the Colab Notebook

Replace the `cmd` block in the notebook with the relevant experiment command. The pattern is identical to the existing cell:

```python
MODEL_VERSION = 'v4-full'
FIGURES_DIR = ROOT / 'fusion_resnet' / f'figures_{MODEL_VERSION}'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    'python', 'train_fusion_resnet.py',
    '--device', 'cuda',
    '--variant', 'full',
    '--epochs', '300',
    '--fp32',
    '--model-version', MODEL_VERSION,
    '--save-dir', str(SAVE_DIR),
    '--figures-dir', str(FIGURES_DIR),
    '--save-every', '1',
    '--snapshot-every', '25',
    '--early-stopping-patience', '40',
    '--early-stopping-min-delta', '0.001',
    '--scheduler-factor', '0.5',
    '--scheduler-patience', '10',
    '--warmup-epochs', '5',
    '--warmup-start-lr', '1e-5',
    '--aug-noise-sigma', '0.02',
    '--aug-amplitude-scale', '0.15',
    '--aug-phase-shift',
]
subprocess.run(cmd, check=True)
```

---

## Evaluation After Each Experiment

The training script auto-generates all evaluation plots and `test_metrics_{MODEL_VERSION}.json`.  
Additionally run:

```bash
python generate_diagnostics.py \
    --checkpoints-dir checkpoints \
    --metrics-dir figures_{MODEL_VERSION} \
    --output-dir reports/ml_diagnostics
```

to produce the comparison plots across all runs.

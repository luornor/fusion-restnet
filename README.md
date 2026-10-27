# Fusion-ResNet for NILM

## What This Repo Is For

This project trains and evaluates a multi-branch NILM model that identifies which appliances are active from aggregate current measurements.

The repo now assumes a **hardware-only runtime target**:

- preprocessing can happen on the hardware
- the model can also run on the hardware
- this repo is mainly for training, evaluation, offline validation, and analysis

There is no separate server/deployment layer in the codebase anymore.

## Model Inputs

The model supports two practical workflows:

1. **Raw-window workflow**
   - training and evaluation from normalized `(N, 400)` waveform windows
   - the model computes ICA, Fryze, and FFT features internally

2. **Preprocessed-feature workflow**
   - offline inference from hardware-style features
   - input keys:
     - `raw_window`: `(N, 400)`
     - `fft_magnitude`: `(N, 200)`
     - `fryze_active`: `(N, 50)`
     - `fryze_reactive`: `(N, 50)`
     - `ica_features`: `(N, 16)`

This makes it possible to validate the exact feature representation your hardware will eventually produce.

## Can You Preprocess NumPy Data Before Training?

Yes.

If your final hardware path will feed **precomputed features** into the model, then the cleanest long-term design is to train on that same representation instead of computing some features inside the model during training and bypassing them later at inference time.

Right now the repo still trains from raw windows, but it now supports **checkpoint-consistent preprocessed inference** for hardware-style validation. That gives you a safe bridge:

- train from raw windows now
- generate hardware-style features from NumPy offline
- validate them with `inference_pipeline.py --preprocessed`
- later, if you want, we can add a dedicated training path for precomputed features

## Appliance Set

Class ids, names, typical currents and source licences live in
[preprocessing/class_registry.json](preprocessing/class_registry.json). Ids
0–15 are the original PLAID label space and never change. New classes are
appended:

| Id | Class | Seeded from | Units |
|---:|---|---|---:|
| 16 | Single-phase induction motor | LILAC | 1 |
| 17 | Three-phase induction motor | LILAC, KAIST HVAC | 4 |
| 18 | VSD-driven motor | LILAC | 1 |
| 19 | Fluorescent tube lighting | LILAC | 1 |
| 20 | Universal-motor tool | LILAC | 1 |
| 21 | Espresso machine | CAFFEINE | 1 |
| 22 | Inverter air conditioner | nonlinear_loads | 1 |
| 23 | UPS | nonlinear_loads | 1 |

Most new classes come from **one physical unit**. They are seeds for field
data, not deployable classes. Ids 22 and 23 have too few windows to train and
are dropped by `--min-class-samples`.

The model has no input for absolute power, because each window is scaled to
unit height. It cannot tell resistive loads apart by shape, so ovens, defrost
heaters and resistor banks belong in `Heater`. The VSD motor, inverter AC, UPS
and laptop all present a rectifier front end and look alike.

The firmware hardcodes 15 classes and 16 ICA components. A model trained on
more classes needs a matching firmware change.

## Main Files

- `fusion_resnet.py`: model definitions, including the wrapper for preprocessed-feature inference
- `train_fusion_resnet.py`: training, evaluation, checkpointing, plots
- `inference_pipeline.py`: offline inference for raw windows or preprocessed hardware features
- `anomaly_detector.py`: anomaly and health tracking
- `postprocess_timeline.py`: smoothing and interval extraction
- `preprocess_natural.py`: segment natural recordings into normalized windows
- `data_preprocessing.py`: PLAID preprocessing utilities

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_fusion_resnet.py --device cuda --variant full --model-version 0.0.1-dev
```

Useful variants:

```bash
python train_fusion_resnet.py --device cuda --variant full --epochs 300 --model-version 1.0.0
python train_fusion_resnet.py --device cuda --variant lite --batch-size 64 --fp32 --model-version 0.0.1-dev
python train_fusion_resnet.py --device cpu --variant lite --epochs 100 --model-version 0.0.1-dev
```

### Training data: time base

A window is 400 samples, and the repo has used two incompatible conventions:

- **cycles**: 8 mains cycles, 50 samples per cycle, starting at the voltage's
  rising zero crossing. This is what the model is built for, because
  `FryzeDecomposition` reshapes the window to 8 × 50 and multiplies each row by
  a one-cycle sine. PLAID `X_real.npy` is in this form.
- **raw30k**: 400 consecutive samples at 30 kHz, which is 13 ms or two-thirds
  of a cycle. `prepare_hifda.py`, `prepare_whited.py` and `prepare_cooll.py`
  produce this. The Fryze branch is meaningless on it.

`data/combined` and the shipped model mix both conventions, so every mixture
that combines PLAID with another source adds a 133 ms window to a 13 ms one.
New data uses **cycles**, and `merge_datasets.py` refuses to mix time bases.
HIFDA, WHITED and COOLL stay out of the new sets until their scripts cut
voltage-aligned cycle windows. `signature_utils.zc_windows` does that given
the voltage channel.

Inference on raw signals (`segment_continuous_signal`) resamples **10** cycles
into 400 samples, 40 per cycle. Neither PLAID nor the Fryze branch uses that.
Check it against the firmware before relying on raw-signal inference.

### Training data: build and train

```bash
# 1. Prepare each source into data/<source>/ (raw downloads live in ../datasets_raw/)
python preprocessing/prepare_plaid.py            # PLAID 2018 raw -> data/plaid_v2 (appliance-instance groups)
python preprocessing/prepare_lilac.py            # LILAC lab industrial loads
python preprocessing/prepare_caffeine.py         # CAFFEINE espresso machine
python preprocessing/prepare_kaist_hvac.py       # KAIST HVAC three-phase fan motor
python preprocessing/prepare_nonlinear_loads.py  # inverter AC, UPS snapshots

# 2. Merge. Default: every cycles-time-base source, plaid_v2 in place of legacy plaid.
python preprocessing/merge_datasets.py --out-dir data/combined_v2
python preprocessing/merge_datasets.py --commercial-only --out-dir data/combined_commercial

# 3. Train with an honest split and realistic mixture currents
python training/train_fusion_resnet.py --data-dir data/combined_v2 \
    --group-split --current-prior --calibrate \
    --mains-freq 50 --mains-volt 230 --model-version 2.0.0
```

- **`--group-split`** splits by appliance instance for PLAID v2 and by
  recording elsewhere. A class from one unit, such as Hair Iron, can only
  appear in one split, and training warns about it.
- **`--current-prior`** rescales each signature to a current drawn from its
  class range in the registry before mixing. Sources use different units, with
  PLAID in amps at 120 V and WHITED normalised, so without it the relative
  sizes inside a mixture are arbitrary.
- **`--commercial-only`** keeps only sources whose licence allows commercial
  use. COOLL is CC BY-NC. WHITED and LILAC state no licence, and LILAC is the
  only source for ids 16, 18, 19 and 20.
- **`--aug-bandlimit-prob`** matters only for raw30k data mixed with low-rate
  sources. The cycles time base already caps every source at 25 × mains.

To reproduce the legacy mixed set:
`merge_datasets.py --sources plaid hifda whited cooll --allow-mixed-timebase --out-dir data/combined`.

### Per-class thresholds

After training, a decision threshold is fitted per appliance on the validation
set and saved in the checkpoint as `class_thresholds`. Inference uses them
automatically. `--threshold` still forces one global value, and
`--no-class-thresholds` disables the fitting during training.

`--min-class-samples N` drops classes with fewer than N real windows. The
default is 10. Hair Iron has only 15 real windows, so `--min-class-samples 100`
drops it.

Checkpoints now store:

- threshold
- class_thresholds, one per appliance
- appliance names
- kept class ids
- variant
- signal length
- model version

## Offline Inference

### Raw windows

```bash
python inference_pipeline.py --checkpoint checkpoints/best.pt --input windows.npy --pre-segmented
```

### Hardware-style preprocessed features

```bash
python inference_pipeline.py --checkpoint checkpoints/best.pt \
    --input preprocessed_features.npz \
    --preprocessed \
    --device cpu --fp32
```

### With anomaly detection

```bash
python inference_pipeline.py --checkpoint checkpoints/best.pt \
    --input preprocessed_features.npz \
    --preprocessed \
    --enable-anomaly-detection \
    --measured-current 5.2
```

## Outputs

Inference writes:

- `predictions.json`
- `predictions.csv`
- `predictions.npy`
- `probabilities.npy`
- `windows.npy`
- `metadata.json`
- `mobile_payload.json`

## Notes

- Preprocessed inference now rebuilds the model directly from checkpoint metadata and saved ICA buffers.
- You do **not** need to re-fit ICA just to run inference.
- Timestamps are emitted in UTC for consistency.

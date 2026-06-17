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

With the current dataset filtering, the kept appliance classes are:

- Air Conditioner
- Coffee maker
- Compact Fluorescent Lamp
- Fan
- Fridge
- Hair Iron
- Hairdryer
- Heater
- Incandescent Light Bulb
- Laptop
- Microwave
- Soldering Iron
- Vacuum
- Washing Machine
- Water kettle

The code now preserves the correct class-name mapping after rare classes are removed.

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

Checkpoints now store:

- threshold
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

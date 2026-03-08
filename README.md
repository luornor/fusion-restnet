# Fusion-ResNet for NILM

## What This Project Does

This project identifies which household appliances are turned on just by looking at the total electrical current flowing into a house. This is called **Non-Intrusive Load Monitoring (NILM)** — "non-intrusive" because you only need one sensor at the main power line, not a separate sensor on every appliance.

Think of it like this: every appliance (fridge, microwave, laptop, etc.) draws electricity in a unique pattern. When multiple appliances run at the same time, their patterns overlap. This model learns to untangle those overlapping patterns and tell you which appliances are active.

## How It Works

The model takes in a short snapshot (~0.17 seconds) of the household's electrical current and looks at it from four different angles:

```
Electrical Current Snapshot (400 samples)
    │
    ├── Branch 1: Raw Signal        — the waveform as-is
    ├── Branch 2: ICA Decomposition  — statistically separated components
    ├── Branch 3: Fryze Decomposition — active vs reactive power
    └── Branch 4: FFT               — frequency content
    │
    └── Attention Fusion — the model learns which view matters most
            │
            └── Prediction: which appliances are ON
```

Each branch processes its view through a **ResNet** (a type of neural network good at learning from signals). Then an **attention module** decides how much to trust each branch before making a final prediction. This multi-view approach makes the model more robust than looking at the signal from just one angle.

## What the Model Expects (Input)

The model expects a **1D current waveform** of **400 samples**, normalized to unit magnitude. This represents roughly 10 electrical cycles (~0.17 seconds at 60 Hz mains).

In practice, you can feed it:

| Input Format | Description |
|---|---|
| **Pre-segmented `.npy`** | Array of shape `(N, 400)` — already windowed and ready |
| **Raw `.csv` file** | Continuous current waveform with a `Current` column — the inference pipeline automatically windows and resamples it |
| **Raw `.npy` file** | 1D continuous current signal — same automatic processing |

## What the Model Outputs

For each 400-sample window, the model outputs a **multi-label prediction** — a list of appliances and the confidence that each one is currently ON.

Example output for a single window:

```
Active Appliances:
  Fridge        (confidence: 0.94)
  Microwave     (confidence: 0.87)
  Laptop        (confidence: 0.72)
```

The model can detect these **15 appliances**:

> Air Conditioner, Blender, Coffee Maker, Compact Fluorescent Lamp, Fan, Fridge, Hair Iron, Hairdryer, Heater, Incandescent Light Bulb, Laptop, Microwave, Soldering Iron, Vacuum, Washing Machine

For continuous recordings, the inference pipeline slides a window across the signal and produces a prediction for each time step, giving you a timeline of appliance activity.

## Dataset

Trained on the [PLAID](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619) (Plug Load Appliance Identification Dataset) — real electrical measurements from US households sampled at 30 kHz.

During training, individual appliance recordings are mixed together to simulate real household scenarios where multiple appliances run simultaneously.

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# Google Colab with T4 GPU (recommended)
python train_fusion_resnet.py --device cuda --variant full --epochs 250 --fp32

# Laptop GPU (RTX 2050 / 4GB VRAM)
python train_fusion_resnet.py --device cuda --variant lite --batch-size 64 --fp32

# CPU only
python train_fusion_resnet.py --device cpu --variant lite --epochs 50
```

### Run Inference

```bash
# On a CSV file with continuous current data
python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
    --input recording.csv --sample-rate 30000 --device cpu --fp32

# On pre-segmented windows
python inference_pipeline.py --checkpoint checkpoints/fusion_resnet/best.pt \
    --input windows.npy --pre-segmented --device cpu --fp32
```

Results are saved to `inference_results/` as JSON, CSV, and NumPy files.

## Model Variants

| Variant | Parameters | VRAM | Best For |
|---------|-----------|------|----------|
| `full`  | ~1.3M    | ~1.5 GB (fp32) | Colab / desktop GPU |
| `lite`  | ~250K    | ~0.5 GB (fp32) | Laptop GPU / CPU |

## Training Output

The training script automatically generates:

- **Checkpoints** in `checkpoints/fusion_resnet/` — best and last model weights
- **Metrics** in `figures/test_metrics.json` — F1, Precision, Recall, Hamming Loss, etc.
- **Plots** in `figures/`:

| Plot | What It Shows |
|------|---------------|
| `training_curves.png` | Loss and F1 score over training epochs |
| `per_appliance_f1.png` | How well each appliance is detected |
| `f1_by_components.png` | Performance vs number of simultaneous appliances |
| `metrics_heatmap.png` | Per-appliance metrics at a glance |
| `dashboard.png` | Combined summary of all results |

## Project Structure

```
├── fusion_resnet.py           # Model architecture (branches, fusion, classifier)
├── train_fusion_resnet.py     # Training, evaluation, and plot generation
├── inference_pipeline.py      # Sliding-window inference on new data
├── fryze_utils.py             # Fryze power decomposition helpers
├── data_preprocessing.py      # Raw PLAID data processing (optional)
├── requirements.txt           # Python dependencies
├── data/                      # Pre-processed training data (.npy files)
├── checkpoints/               # Saved model weights
└── figures/                   # Generated plots and metrics
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- A GPU is recommended for training but not required for inference

## References

- Original ICAResNetFFN project: [ML2023SK Team 37](https://github.com/arx7ti/ML2023SK-final-project)
- PLAID Dataset: [Figshare](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619)




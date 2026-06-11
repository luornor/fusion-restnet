# HIFDA Preprocessing Report

Generated: 2026-06-11T08:58:21.033753

## Summary

| Item | Value |
|---|---|
| HIFDA root | `C:\Users\ASUS\Desktop\Projects\Fusion-ResNet\HIFDA_HF_electrical_signals_dataset\HIFDA_HF_electrical_signals_dataset` |
| Split used | `163.84ms_window_dataset` |
| Uncertain classes | excluded |
| Total HIFDA windows | 16000 |
| X_hifda shape | [16000, 400] |
| X_hifda dtype | float32 |
| y_hifda shape | [16000] |
| **SAFE_TO_TRAIN** | **[OK] YES** |

## Class Mapping

| HIFDA Folder | Model Class | Label ID | Status |
|---|---|---|---|
| Air_conditioner | Air Conditioner | 0 | included |
| Charger | ? | ? | excluded |
| CoffeeMaker | Coffee maker | 2 | included |
| Computer | ? | ? | excluded |
| EmptyGrid | ? | ? | excluded |
| Griddle | ? | ? | excluded |
| HairDryer | Hairdryer | 7 | included |
| Heater | Heater | 8 | included |
| Iron | Hair Iron | 6 | uncertain |
| Laptop | Laptop | 10 | included |
| Light | Incandescent Light Bulb | 9 | uncertain |
| Microwave | Microwave | 11 | included |
| Monitor | ? | ? | excluded |
| Vacuum | Vacuum | 13 | included |
| Washing_machine | Washing Machine | 14 | included |

## Per-Class Statistics

| Class | PLAID Count | HIFDA Count | Combined |
|---|---|---|---|
| Air Conditioner | 3841 | 2000 | 5841 |
| Blender | 2 | 0 | 2 |
| Coffee maker | 13 | 2000 | 2013 |
| Compact Fluorescent Lamp | 2218 | 0 | 2218 |
| Fan | 2946 | 0 | 2946 |
| Fridge | 1839 | 0 | 1839 |
| Hair Iron | 15 | 0 | 15 |
| Hairdryer | 2383 | 2000 | 4383 |
| Heater | 606 | 2000 | 2606 |
| Incandescent Light Bulb | 1788 | 0 | 1788 |
| Laptop | 638 | 2000 | 2638 |
| Microwave | 1813 | 2000 | 3813 |
| Soldering Iron | 28 | 0 | 28 |
| Vacuum | 292 | 2000 | 2292 |
| Washing Machine | 965 | 2000 | 2965 |
| Water kettle | 13 | 0 | 13 |

## Signal Statistics

| Metric | PLAID | HIFDA |
|---|---|---|
| global_min | -22.8019 | -1.1415 |
| global_max | 20.3017 | 1.1375 |
| global_mean | 0.0024 | -0.0007 |
| global_std | 3.1713 | 0.4163 |
| rms_mean | 1.7700 | 0.3362 |
| rms_std | 2.6314 | 0.2455 |
| rms_min | 0.0070 | 0.0095 |
| rms_max | 13.4042 | 0.8833 |

## [OK] All Validation Checks Passed


## Preprocessing Steps Applied

1. Load HIFDA Current .txt file (one float per line, no header)
2. Remove DC offset: subtract per-window mean (~1.647 ADC bias)
3. Resample: 100 kHz -> 30 kHz via polyphase filter (up=3, down=10)
4. Extract 12 non-overlapping 400-sample sub-windows per file
5. Reject all-zero windows (silent channel artefacts)
6. Cast to float32

## Reproduction Commands

```bash
# HIFDA only (exclude uncertain):
python prepare_hifda.py --hifda-dir "C:\Users\ASUS\Desktop\Projects\Fusion-ResNet\HIFDA_HF_electrical_signals_dataset" --out-dir "data/hifda" --reference-data-dir "data" --exclude-uncertain --seed 42
```

## Recommended Training Command

```bash
python train_fusion_resnet.py \
  --data-dir data \
  --mains-freq 50 --mains-volt 230 \
  --model-version combined-50hz-v1 \
  --epochs 300 --early-stopping-patience 40 \
  --variant lite --device cuda
```

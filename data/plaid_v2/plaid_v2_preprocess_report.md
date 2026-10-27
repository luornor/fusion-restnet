# plaid_v2 preprocessing report

- Generated: 2026-10-27T19:35:26+00:00
- Windows: 38,471 from 318 recordings
- Licence: CC BY 4.0 (commercial use: True)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 30000 Hz, mains 60.0 Hz, retimed to 50 Hz: True
- Usable bandwidth: up to 1500 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Air Conditioner | 6,317 | 22 | 0.396 |
| Blender | 6 | 1 | 0.743 |
| Coffee maker | 30 | 1 | 3.2001 |
| Compact Fluorescent Lamp | 3,304 | 41 | 0.1522 |
| Fan | 4,391 | 31 | 0.3443 |
| Fridge | 2,908 | 31 | 1.3545 |
| Hair Iron | 30 | 1 | 0.5705 |
| Hairdryer | 4,111 | 37 | 3.8308 |
| Heater | 1,508 | 13 | 6.0254 |
| Incandescent Light Bulb | 2,657 | 31 | 0.3063 |
| Laptop | 4,533 | 41 | 0.2965 |
| Microwave | 4,617 | 32 | 6.5586 |
| Soldering Iron | 60 | 1 | 0.0875 |
| Vacuum | 1,186 | 16 | 5.0886 |
| Washing Machine | 2,783 | 18 | 3.3412 |
| Water kettle | 30 | 1 | 4.5974 |

## Caveats

- US 120 V / 60 Hz appliances. Cycle-synchronous sampling removes the frequency difference; the current level reflects 120 V (use --current-prior when mixing).
- Groups are appliance instances, so --group-split tests generalisation to unseen appliances, not just unseen recordings.
- Blender (2 records) and Hair Iron, Coffee maker, Water kettle (10 each) remain thin.

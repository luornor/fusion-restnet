# nonlinear_loads preprocessing report

- Generated: 2026-10-27T19:36:58+00:00
- Windows: 19 from 5 recordings
- Licence: CC BY 4.0 (commercial use: True)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 7667.8 Hz, mains 60.06 Hz, retimed to 50 Hz: True
- Usable bandwidth: up to 1500 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Fridge | 4 | 1 | 1.8039 |
| Laptop | 4 | 1 | 0.5047 |
| Microwave | 3 | 1 | 13.5155 |
| Inverter air conditioner | 4 | 1 | 2.3614 |
| UPS | 4 | 1 | 1.5689 |

## Caveats

- Each class here is ONE ~0.2 s snapshot of ONE unit. Overlapping windows are near-duplicates.
- Inverter air conditioner and UPS are seeds only. Do not report accuracy on them until field data exists.
- ~3.8 kHz source bandwidth: irrelevant in the cycles time base; with raw30k, train with --aug-bandlimit-prob.
- Recorded on a 120/240 V 60 Hz grid; retimed to 50 Hz unless --no-retime.

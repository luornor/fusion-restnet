# blond preprocessing report

- Generated: 2026-10-31T15:27:23+00:00
- Windows: 2,805 from 20 recordings
- Licence: CC BY 4.0 (commercial use: True)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 6400 (BLOND-50), 50000 (BLOND-250) Hz, mains 50.0 Hz, retimed to 50 Hz: False
- Usable bandwidth: up to 1250 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Fan | 430 | 2 | 0.1389 |
| Heater | 10 | 1 | 6.5016 |
| Laptop | 2,335 | 15 | 0.2121 |
| Water kettle | 10 | 1 | 7.2594 |
| Universal-motor tool | 20 | 1 | 0.3621 |

## Caveats

- Office building in Germany, 230 V / 50 Hz, CC BY 4.0.
- Sampled download (fetch_blond.py): each socket-appliance instance on up to 3 weekdays.
- Groups are appliance models, so several identical laptops count as one unit (no split leakage).
- The Clatronic kettle on MEDAL-13 was not caught in the sampled office-hour files; Water kettle and Heater have one unit each, from the logged test sessions.

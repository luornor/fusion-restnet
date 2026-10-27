# lilac preprocessing report

- Generated: 2026-10-27T19:35:28+00:00
- Windows: 2,331 from 717 recordings
- Licence: none stated (commercial use: None)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 50000 Hz, mains 50 Hz, retimed to 50 Hz: False
- Usable bandwidth: up to 1250 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Coffee maker | 84 | 42 | 4.1751 |
| Fridge | 60 | 31 | 5.1308 |
| Hairdryer | 75 | 38 | 5.6166 |
| Heater | 363 | 91 | 8.1465 |
| Incandescent Light Bulb | 183 | 92 | 0.4147 |
| Vacuum | 82 | 41 | 6.9821 |
| Water kettle | 82 | 41 | 11.2845 |
| Single-phase induction motor | 144 | 72 | 3.2386 |
| Three-phase induction motor | 804 | 134 | 1.4825 |
| VSD-driven motor | 276 | 46 | 0.8959 |
| Fluorescent tube lighting | 90 | 45 | 0.4308 |
| Universal-motor tool | 88 | 44 | 1.9551 |

## Caveats

- One physical unit per LILAC appliance type: new classes seeded here will not generalise to other makes until field data is added.
- Windows from one record are rotations of the same cycle, not independent observations. In the cycles time base each window is that cycle repeated 8 times.
- Licence: none stated by LILAC. Do not ship a model trained on this without the authors' permission.
- Resistor and Raclette are mapped to Heater as electrically equivalent resistive loads.

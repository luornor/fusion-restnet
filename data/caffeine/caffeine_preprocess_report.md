# caffeine preprocessing report

- Generated: 2026-10-27T19:36:56+00:00
- Windows: 6,668 from 45 recordings
- Licence: CC BY 4.0 (commercial use: True)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 6250 Hz, mains 50.0 Hz, retimed to 50 Hz: False
- Usable bandwidth: up to 1250 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Espresso machine | 6,668 | 45 | 6.4538 |

## Caveats

- One physical machine: the Espresso machine class will not generalise to other makes until field data is added.
- 6.25 kHz source: nothing above 3.125 kHz. Irrelevant in the cycles time base (1.25 kHz cap); with raw30k, train with --aug-bandlimit-prob.
- Labels are whole-machine. Component states are saved in states_caffeine.npy.

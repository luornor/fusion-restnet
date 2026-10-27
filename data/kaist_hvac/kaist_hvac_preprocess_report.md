# kaist_hvac preprocessing report

- Generated: 2026-10-27T19:35:58+00:00
- Windows: 2,700 from 25 recordings
- Licence: CC BY 4.0 (commercial use: True)
- Time base: cycles (8 cycles x 50 samples)
- Source rate: 100000 Hz, mains 60.0 Hz, retimed to 50 Hz: True
- Usable bandwidth: up to 1500 Hz

| Class | Windows | Recordings | Median RMS (A) |
|---|---:|---:|---:|
| Three-phase induction motor | 2,700 | 25 | 2.5183 |

## Caveats

- One physical motor (HVAC fan via belt). Adds diversity to the Three-phase induction motor class but is not a fleet.
- Direct-on-line 60 Hz, 380 V class motor.
- Belt levels D0-D4 are kept as groups, not classes. They are useful later for motor-health work (see docs/MCSA_PLAN.md).

# Fusion-ResNet NILM: Hardware & Data Preprocessing Walkthrough

## Part 1: How the Hardware Captures Data

### The Physical Setup

The system is built around an ESP32 microcontroller that reads electrical measurements from sensors clamped onto the household main power line, then transmits the data over WiFi to a server for processing.

```
┌─────────── MAIN POWER LINE (120V AC, 60Hz) ───────────┐
│                                                         │
│  ┌──────────┐                                          │
│  │ SCT013   │ ← clamps around the live wire            │
│  │ (CT x2)  │   outputs 0-1V proportional to current   │
│  └────┬─────┘                                          │
│       │ analog signal                                   │
│       ▼                                                 │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │ Metering IC  │────▶│   ESP32      │                 │
│  │ (x2)        │SPI  │              │──── WiFi ──▶ Server
│  └──────────────┘     │              │                 │
│                        │   ┌────────┐│                 │
│  ┌──────────────┐     │   │LCD x2  ││                 │
│  │ PZEM-004T   │────▶│   │(I2C)   ││                 │
│  │ (x2)        │UART │   └────────┘│                 │
│  └──────────────┘     └──────────────┘                 │
│       │                     │                           │
│  ┌────┴─────┐         ┌────┴──────┐                   │
│  │Voltage   │         │DC-DC x3   │                   │
│  │Transformer│         │Buck/Boost │                   │
│  │120→12V   │         │+ 9V adapter│                   │
│  └──────────┘         └───────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Component-by-Component Breakdown

**1. SCT013 Current Transformers (x2)**

These clamp around the wire without cutting it. They convert the current flowing through the wire into a small 0–1V analog signal. There are two because the system monitors two circuits (e.g., two phases or two branch circuits).

**2. Voltage Transformer (120V → 12V)**

Steps down the mains voltage to a safe level for measurement.

**3. Metering IC Controllers (x2)**

This is the critical chip. It has a high-speed ADC (Analog-to-Digital Converter) that samples the SCT013 analog output at a high rate. Depending on the specific IC (ADE7753, ADE7953, ATM90E26, etc.), it can sample the waveform at 4–16 kHz and store it in an internal buffer.

**4. PZEM-004T Modules (x2)**

These measure RMS voltage, current, power, energy, and frequency. They are useful for the LCD display and total energy monitoring, but they do not provide raw waveforms — only aggregated readings.

**5. ESP32 Microcontroller**

The brain of the system. It reads waveform buffers from the metering ICs via SPI, reads power data from the PZEM-004T via UART, shows readings on the LCDs via I2C, and transmits data over WiFi to the server.

**6. DC-DC Buck Boost Converters (x3) + 9V Adapters**

Power supply for the ESP32, metering ICs, and sensors.

### What the ESP32 Sends

The ESP32 collects a short burst of raw current samples from the metering IC's waveform buffer and sends it over WiFi. A typical payload looks like:

```json
{
  "timestamp": 1709913600,
  "sample_rate": 8000,
  "current_samples": [0.12, 0.45, 0.78, 1.02, ...]
}
```

The server receives this and feeds it into the inference pipeline.


## The 30kHz Question

### Why 30kHz Matters

The PLAID dataset — which the model was trained on — was recorded at 30,000 samples per second using lab-grade equipment.

The hardware in this project cannot match that rate:

| Source             | Sample Rate  | Notes                            |
|--------------------|-------------|----------------------------------|
| PLAID dataset      | 30 kHz      | Lab-grade equipment              |
| ESP32 ADC directly | ~1–6 kHz    | Noisy, 12-bit, jittery timing   |
| Metering IC        | ~4–16 kHz   | Clean, consistent, purpose-built |

### Why It Still Works

The PLAID preprocessing pipeline converts 30kHz data down to 400 samples over 10 electrical cycles. That is effectively ~2,400 samples per second. So if the metering IC samples at 4kHz or higher, there is enough data — it just needs to be resampled to 400 points over the same time window.

The key requirement is capturing enough samples per cycle to preserve the waveform shape. At 8kHz, there are ~133 samples per cycle, which is enough to capture up to the 65th harmonic — more than sufficient for appliance identification.

The inference pipeline handles this resampling automatically.


## Part 2: The NILM Data Preprocessing Pipeline

Here is exactly what happens to the data, from raw electrical signal to model input.

### Stage 1: Raw Recording → Cycle-Aligned Signal

```
Raw current from sensor (e.g., 30kHz or whatever the hardware captures)
│
▼ FITPS transform (Fundamental Identification and Time-Period Standardization)
│
│  1. Finds the fundamental 60Hz frequency in the signal
│  2. Aligns the signal to exact cycle boundaries
│  3. Resamples each cycle to exactly 500 points
│
▼ Result: Clean signal with 500 samples per cycle
```

One cycle at 60Hz = 16.67 milliseconds = exactly 500 samples after FITPS.

### Stage 2: Steady-State Detection

```
Cycle-aligned signal
│
▼ where_jumps() — finds where appliances turn ON or OFF
│
│  Looks at the standard deviation of each cycle.
│  A sudden jump in standard deviation = something switched.
│  Between jumps = steady-state (appliances running normally).
│
▼ Result: list of stable segments
```

The model is designed to identify appliances during steady-state operation, not during transient events (the moment an appliance turns on or off).

### Stage 3: Windowing (10 Cycles Per Window)

```
Stable segments
│
▼ Extract 10-cycle windows
│
│  Each window = 10 cycles × 500 samples = 5,000 raw samples
│  Then FITPS resamples this to a compact representation
│  Final output: 400 samples per window (10 cycles × 40 samples/cycle)
│
▼ Result: shape (N, 400) — each row is one window
```

This is what "400 samples" means — 10 electrical cycles compressed to 400 points. In real time, each window covers approximately 167 milliseconds (10 × 16.67ms).

### Stage 4: Quality Filtering

```
Windows
│
▼ Remove bad windows:
│  - Signal too weak (max amplitude < 0.1) → noise, not a real appliance
│  - Inconsistent cycles (one cycle's mean 5x different from others) → transient
│
▼ Result: clean windows with labels
```

### Stage 5: Mixture Composition (Training Only)

In a real house, multiple appliances run at the same time. To simulate this, the training pipeline creates artificial mixtures:

```
Clean single-appliance windows
│
▼ Composer — creates realistic multi-appliance mixtures
│
│  Takes 2–15 individual appliance signals and ADDS them together.
│  Example: fridge_signal + laptop_signal + fan_signal = mixture
│  Label: [Fridge=1, Laptop=1, Fan=1, others=0]
│
▼ Result: (21,000 train / 2,666 val / 6,250 test) mixture windows
```

This step only happens during training. During inference on real household data, the signal coming from the main power line is already a natural mixture of all active appliances.

### Stage 6: Normalization

```
Mixture windows
│
▼ Normalize each window to unit magnitude
│  window = window / max(|window|)
│  So every window has values between -1 and 1
│
▼ Result: ready for the model
```

This ensures the model focuses on the shape of the waveform, not its absolute magnitude.


## Part 3: Inside the Model (4 Branches)

When a normalized 400-sample window enters the model, it is processed from four different perspectives simultaneously:

### Branch 1: Raw Signal

Takes the 400 samples exactly as they are. A 1D Convolutional ResNet extracts time-domain patterns — things like the shape of peaks, zero-crossings, and amplitude envelopes.

### Branch 2: ICA (Independent Component Analysis)

Multiplies the signal by a pre-computed ICA un-mixing matrix, separating it into 16 statistically independent components. Then applies an exponential transform and normalization before feeding into a Conv ResNet. This helps isolate individual source signals from the mixture.

### Branch 3: Fryze Power Decomposition

Reshapes the 400 samples into 8 cycles of 50 samples each. Averages across cycles and generates a synthetic 60Hz voltage waveform. Then computes:

- **Active current** — the part that does useful work (powers the appliance)
- **Non-active (reactive) current** — the part that flows back and forth without doing useful work

Different appliance types have very different active-to-reactive ratios. A heater is almost entirely active; a motor has significant reactive current.

### Branch 4: FFT (Fast Fourier Transform)

Computes the frequency spectrum of the 400 samples using `torch.fft.rfft()`. Takes the magnitude of the first 200 frequency bins. This reveals the harmonic content of the signal — the same harmonics that the hardware team mentioned.

Every appliance has a unique harmonic fingerprint. A laptop charger has strong odd harmonics (3rd, 5th, 7th). A heater has almost none. The FFT branch captures exactly this.

### Attention Fusion

All four branches produce a feature vector. The attention fusion module learns a weight for each branch — how much to trust each perspective for the current input. Some signals may be best identified by their raw shape, others by their frequency content.

The weighted combination is passed through a classification head that outputs 15 probabilities, one per appliance.


## The Complete Path: Hardware → Prediction

```
SCT013 sensor
    → Metering IC (samples at ~8kHz)
    → ESP32 reads ~1,333 samples (10 cycles at 8kHz ≈ 167ms)
    → Sends over WiFi to server
    → Server receives the data
    → inference_pipeline.py resamples 1,333 samples → 400 samples
    → Normalizes to [-1, 1]
    → Model processes through 4 branches
    → Output: "Fridge: 94%, Laptop: 87%, others: below threshold"
```

The ESP32 does not preprocess the data or run the model — it only captures and transmits. All preprocessing and inference happens on the server (a Raspberry Pi, laptop, or cloud endpoint).

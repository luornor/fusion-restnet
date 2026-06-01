# NILM Real-World Monitoring Pipeline

**Project:** Fusion-ResNet NILM (lumina-firmware on ESP32-S3)  
**Deployment target:** Ghana grid (230V / 50Hz)  
**Purpose:** Track model performance in the field, detect drift, and collect labelled data for future retraining.

---

## 1. Goals

| Goal | What it means in practice |
|---|---|
| **Performance visibility** | Know whether the model is making confident, stable predictions without needing ground-truth labels |
| **Drift detection** | Catch when the model starts behaving differently — new loads, grid harmonics, seasonal patterns |
| **Training data collection** | Build a real-world dataset from field events for the next model version |
| **Fault / anomaly tracking** | Surface `anomaly_detector.py` signals as operational metrics |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EDGE  (ESP32-S3 / lumina-firmware)                                      │
│                                                                          │
│  ┌──────────────┐   ┌───────────────────┐   ┌─────────────────────────┐ │
│  │ ZMPT101B     │──▶│  preprocessor.go  │──▶│  tflm.go (TFLite)       │ │
│  │ CT sensor    │   │  FFT / Fryze / ICA│   │  15-class inference     │ │
│  └──────────────┘   └───────────────────┘   └────────────┬────────────┘ │
│                                                           │              │
│                      ┌────────────────────────────────────▼───────────┐ │
│                      │  telemetry_agent.go                            │ │
│                      │  • Builds telemetry packet every 1s            │ │
│                      │  • Ring-buffers raw windows (30 min / PSRAM)   │ │
│                      │  • Triggers raw-window upload on events        │ │
│                      └──────────────────┬─────────────────────────────┘ │
└─────────────────────────────────────────┼────────────────────────────────┘
                                          │  WiFi
                              ┌───────────▼───────────┐
                              │   MQTT Broker          │
                              │   (Mosquitto / HiveMQ) │
                              └───────────┬───────────┘
                                          │
           ┌──────────────────────────────┼───────────────────────────┐
           │                              │                           │
 ┌─────────▼──────────┐       ┌──────────▼──────────┐   ┌───────────▼──────┐
 │  metrics_consumer  │       │  window_collector   │   │  alert_service   │
 │  (Python)          │       │  (Python)           │   │  (Python)        │
 │  • Parses telemetry│       │  • Saves raw windows│   │  • PSI drift     │
 │  • Computes KPIs   │       │    to object store  │   │  • Anomaly rates │
 │  • Writes to DB    │       │  • Tags for labeling│   │  • Email / Slack │
 └─────────┬──────────┘       └──────────┬──────────┘   └──────────────────┘
           │                             │
 ┌─────────▼──────────┐       ┌──────────▼──────────┐
 │  InfluxDB / SQLite │       │  Object Store        │
 │  (time-series KPIs)│       │  (MinIO / local dir) │
 └─────────┬──────────┘       └──────────┬──────────┘
           │                             │
 ┌─────────▼──────────┐       ┌──────────▼──────────┐
 │  Grafana Dashboard │       │  Label Studio        │
 │  (live monitoring) │       │  (human annotation)  │
 └────────────────────┘       └─────────────────────┘
```

---

## 3. Edge Telemetry Design

### 3.1 Telemetry Packet (sent every inference cycle, ~1 Hz)

Small JSON payload. Sent via MQTT topic `nilm/{device_id}/telemetry`.

```json
{
  "device_id": "lumina-001",
  "ts": 1748000000,
  "fw_version": "1.2.0",
  "predictions": [
    { "name": "Fridge",   "active": true,  "conf": 0.91 },
    { "name": "Fan",      "active": true,  "conf": 0.78 },
    { "name": "Laptop",   "active": false, "conf": 0.12 }
  ],
  "anomalies": [
    { "type": "unaccounted_current", "severity": "low", "residual_ratio": 0.08 }
  ],
  "metrics": {
    "inference_ms":    44,
    "preprocess_ms":   1,
    "max_conf":        0.91,
    "min_conf":        0.06,
    "pred_entropy":    0.31,
    "fryze_residual":  0.08,
    "n_active":        2,
    "rms_current":     1.42
  }
}
```

**Payload size estimate:** ~600 bytes. At 1 Hz over WiFi this is ~52 KB/min — well within typical home WiFi budgets.

### 3.2 Raw Window Upload (event-triggered)

Raw windows are expensive (400 samples × 4 bytes = 1.6 KB each) so they are uploaded selectively, not every second. Uploaded via MQTT topic `nilm/{device_id}/window` or HTTP POST.

**Upload triggers:**

| Trigger | Condition | Why it matters |
|---|---|---|
| **Low confidence** | `max_conf < 0.45` for ≥ 3 consecutive windows | Unknown load — high-value training sample |
| **Anomaly detected** | Any anomaly with severity ≥ medium | Real fault event |
| **State transition** | Any prediction flips from 0→1 or 1→0 | Clean on/off signatures are gold for training |
| **Scheduled sample** | 1 random window per minute | Unbiased coverage for distribution shift detection |
| **Manual trigger** | Via MQTT command topic | Debugging |

**Raw window packet:**
```json
{
  "device_id":     "lumina-001",
  "ts":            1748000000,
  "trigger":       "low_confidence",
  "raw_signal":    [0.012, -0.031, ...],
  "fft_magnitude": [0.002, 0.041, ...],
  "fryze_active":  [0.031, ...],
  "fryze_reactive":[0.012, ...],
  "ica_features":  [0.22, -0.14, ...],
  "predictions":   [...],
  "label":         null
}
```

### 3.3 On-Device Ring Buffer

The ESP32-S3 has 16 MB PSRAM. Allocate a 30-minute sliding window:
- 1,800 raw windows × 1.6 KB = **2.9 MB**
- This gives 30 minutes of context for retrospective upload after an anomaly is detected
- Implemented as a circular buffer index into PSRAM

---

## 4. Server-Side Services

### 4.1 `metrics_consumer.py`

Subscribes to `nilm/+/telemetry`. On each message:

1. Parse and validate the JSON packet
2. Write all numeric fields to InfluxDB (or SQLite with timestamp)
3. Update rolling statistics per device per appliance:
   - 1-hour rolling mean confidence
   - 24-hour prediction rate (% of windows where appliance is active)
   - Anomaly event counts
4. Feed into drift detector (see §5)

**Key InfluxDB measurements:**

```
measurement: inference
  tags: device_id, fw_version
  fields: inference_ms, preprocess_ms, pred_entropy, fryze_residual, n_active, rms_current

measurement: appliance_confidence
  tags: device_id, appliance_name
  fields: conf, active (0/1)

measurement: anomaly
  tags: device_id, anomaly_type, severity
  fields: residual_ratio (for unaccounted_current), confidence_delta (for degradation)
```

### 4.2 `window_collector.py`

Subscribes to `nilm/+/window`. On each message:

1. Save the raw window as a `.npz` file to object store: `windows/{device_id}/{year}/{month}/{ts}_{trigger}.npz`
2. Write metadata record to SQLite `windows` table (see §6.1)
3. If trigger is `low_confidence`, `anomaly`, or `transition` → add to labeling queue

### 4.3 `alert_service.py`

Runs every 10 minutes. Checks:

- **Confidence collapse**: mean confidence over last hour < 0.5 for any appliance previously stable above 0.7
- **PSI drift** (see §5): weekly PSI > 0.2
- **Anomaly rate spike**: anomaly events > 3× 7-day average in last hour
- **Device offline**: no telemetry received in 5 minutes

Alerts via email or Slack webhook (configured in `.env`).

---

## 5. Performance Monitoring Without Ground Truth

Since there are no real-time labels in deployment, we use **proxy metrics** that correlate with model quality.

### 5.1 Prediction Entropy

$$H = -\sum_{k=1}^{15} p_k \log p_k + (1 - p_k) \log(1 - p_k)$$

Low entropy = model is confident. High entropy = model is uncertain. Computed per window and tracked as a rolling mean.

**Baseline:** Compute mean entropy on the test set during training. Flag if field entropy exceeds baseline by > 50% sustained over 1 hour.

### 5.2 Confidence Distribution Drift (PSI)

Population Stability Index measures how much a distribution has shifted:

$$PSI = \sum_i (actual_i - expected_i) \times \ln\left(\frac{actual_i}{expected_i}\right)$$

**Procedure:**
1. After 2 weeks of deployment, record the confidence distribution per appliance as the **reference baseline** (save to `baseline_distributions.json`)
2. Every week, compute PSI for each appliance's confidence distribution vs. the baseline
3. Thresholds: PSI < 0.1 = stable, 0.1–0.2 = monitor, > 0.2 = alert (likely drift)

### 5.3 Prediction Stability Index

For consecutive windows *t* and *t+1*, a flip occurs when `active[k,t] ≠ active[k,t+1]`. Stability = 1 − (flip rate). Computed per appliance per hour.

**Normal baseline:** Most appliances should be stable 95%+ of the time within a 1-minute window (they don't turn on/off every second). A sustained stability drop signals model noise — often caused by a borderline confidence near the threshold.

### 5.4 Fryze Residual Ratio

`fryze_residual = (measured_rms_current − predicted_sum_current) / measured_rms_current`

If the model correctly identifies all active loads, this should be near 0. A sustained residual > 0.15 signals an unrecognised load — either a new appliance or a known appliance the model is missing.

**Track:** 1-hour rolling mean of `fryze_residual` per device. Threshold alert at > 0.2.

### 5.5 Metrics Summary Dashboard (Grafana)

| Panel | Metric | Alert threshold |
|---|---|---|
| Inference latency | `inference_ms` P95 | > 200 ms |
| Prediction entropy | rolling 1h mean | > 1.5× baseline |
| Per-appliance confidence | `conf` 24h rolling mean | Drop > 0.15 vs. 7-day avg |
| Fryze residual | rolling 1h mean | > 0.20 |
| Anomaly event rate | count per hour by type | > 3× 7-day avg |
| PSI (weekly) | per appliance | > 0.20 |
| Device uptime | last-seen heartbeat | offline > 5 min |
| Windows collected | count by trigger type | — |

---

## 6. Data Collection for Retraining

### 6.1 SQLite `windows` Table Schema

```sql
CREATE TABLE windows (
    id           INTEGER PRIMARY KEY,
    device_id    TEXT NOT NULL,
    ts           INTEGER NOT NULL,         -- Unix timestamp
    trigger      TEXT NOT NULL,            -- low_confidence | anomaly | transition | scheduled
    file_path    TEXT NOT NULL,            -- path to .npz in object store
    max_conf     REAL,
    pred_entropy REAL,
    fryze_resid  REAL,
    anomaly_type TEXT,
    label        TEXT,                     -- NULL until annotated
    label_source TEXT,                     -- human | pseudo | NULL
    labelled_at  INTEGER,
    in_queue     INTEGER DEFAULT 1         -- 1 = needs review, 0 = done
);

CREATE INDEX idx_windows_trigger ON windows(trigger);
CREATE INDEX idx_windows_label   ON windows(label);
CREATE INDEX idx_windows_ts      ON windows(ts);
```

### 6.2 Labeling Strategy

**Three tiers by cost:**

**Tier 1 — Pseudo-labels (free, automated)**
Windows where ≥ 1 appliance has `conf > 0.85` and `fryze_residual < 0.10`. The model is confident and energy-balanced — treat its prediction as the label. Tag `label_source = "pseudo"`. Use only for augmenting training, not for evaluation.

**Tier 2 — User-confirmed labels (low cost)**
At transition events (appliance turning on or off), the firmware can prompt the user via a companion app: *"We detected a new device. What did you just turn on?"* Maps directly to a label with zero extra sensor infrastructure.

**Tier 3 — Human annotation (highest cost, highest value)**
Upload `low_confidence` and `anomaly` windows to Label Studio. Annotator sees a waveform plot + model's best guess and confirms or corrects the label. Prioritize:
1. Windows with `trigger = low_confidence` — new or degraded loads
2. Windows with `anomaly_type = unknown_load` — genuinely unlabelled load types
3. A stratified random sample of `scheduled` windows — unbiased coverage

### 6.3 Retraining Triggers

Initiate a retraining cycle when any of the following are met:

| Condition | Threshold |
|---|---|
| New labelled windows collected | ≥ 500 human-labelled + ≥ 2,000 pseudo-labelled |
| PSI alert sustained | > 0.2 for ≥ 2 consecutive weekly checks |
| Mean confidence drop | > 0.10 sustained for 7 days on ≥ 3 appliances |
| New appliance class identified | ≥ 100 confirmed examples of a previously-unseen load |

When triggered, the pipeline:
1. Exports all labelled windows from object store as a new dataset split
2. Combines with original PLAID training data (weighted to prevent forgetting)
3. Retrains from the last checkpoint with a lower learning rate (fine-tuning)
4. Evaluates on a held-out real-world test set before deploying

---

## 7. Implementation Phases

### Phase 1 — Telemetry Baseline (MVP, ~1 week)

**Goal:** Get data flowing and a basic dashboard live before hardware ships.

- [ ] Add telemetry JSON serialisation to `telemetry_agent.go` in firmware
- [ ] Set up Mosquitto MQTT broker (Docker Compose, one command)
- [ ] Write `metrics_consumer.py` — subscribe, parse, write to SQLite
- [ ] Build Grafana dashboard with panels from §5.5
- [ ] Test with simulated telemetry packets from `inference_pipeline.py`

**Deliverable:** Dashboard showing live confidence, entropy, and anomaly rates.

### Phase 2 — Raw Window Collection (~1 week)

**Goal:** Start accumulating real-world training data.

- [ ] Implement PSRAM ring buffer + upload triggers in firmware
- [ ] Write `window_collector.py` — save `.npz` files, update SQLite
- [ ] Set up Label Studio with a waveform review task template
- [ ] Build `export_labelling_queue.py` — exports pending windows as a Label Studio import file
- [ ] Document labeling instructions for annotators

**Deliverable:** Object store filling with raw windows; labeling queue operational.

### Phase 3 — Drift Detection & Alerting (~3 days)

**Goal:** Automated early warning when the model starts degrading.

- [ ] Compute and save reference baseline distributions after 2 weeks of deployment
- [ ] Implement PSI calculation in `alert_service.py`
- [ ] Add confidence collapse and anomaly spike checks
- [ ] Wire up email/Slack alerts
- [ ] Add `fryze_residual` trending panel to Grafana

**Deliverable:** Team receives alerts if model behaviour deviates significantly.

### Phase 4 — Retraining Loop (~1 week, after enough data collected)

**Goal:** Close the loop — field data improves the model.

- [ ] Write `prepare_retraining_dataset.py` — merge PLAID + field windows, apply pseudo-label filtering
- [ ] Fine-tune checkpoint on new data (lower LR, frozen early ResNet stages)
- [ ] Evaluate on real-world held-out set; compare F1 vs. deployed model
- [ ] If improved: convert to TFLite via `FusionResNetSplit` + `transfer_weights()` (see alignment doc §6.1)
- [ ] OTA push to device

---

## 8. Quick-Start (Running Locally for Testing)

```bash
# 1. Start MQTT broker
docker run -d -p 1883:1883 eclipse-mosquitto

# 2. Start metrics consumer
python monitoring/metrics_consumer.py --broker localhost --db monitoring/nilm_metrics.db

# 3. Start window collector
python monitoring/window_collector.py --broker localhost --store monitoring/windows/

# 4. Simulate telemetry from a test inference run
python monitoring/simulate_telemetry.py \
    --checkpoint checkpoints/best_model.pt \
    --npz data/test_windows.npz \
    --broker localhost \
    --device-id lumina-test-01

# 5. Open Grafana at http://localhost:3000
#    Import dashboard from monitoring/grafana_dashboard.json
```

---

## 9. Key Decisions & Rationale

| Decision | Alternative considered | Why this choice |
|---|---|---|
| MQTT for telemetry | HTTP POST per window | MQTT is fire-and-forget; survives intermittent WiFi without blocking inference loop |
| SQLite for window metadata | PostgreSQL | Simpler ops; can migrate to Postgres when scale demands it |
| InfluxDB for time-series KPIs | SQLite only | Grafana native integration; efficient time-range queries |
| PSI for drift detection | KL divergence, MMD | PSI is interpretable, has established thresholds (0.1/0.2), and handles small samples well |
| Pseudo-labels at conf > 0.85 | conf > 0.7 | Lower threshold risks noise in training data; 0.85 matches the model's precision inflection point observed on the val set |
| Upload 1 scheduled window/min | Upload every window | 1.44 GB/day at 1 Hz would overwhelm storage; 1/min gives 2.3 MB/day with good coverage |

---

## 10. Files to Create

```
Fusion-ResNet-NILM/
└── monitoring/
    ├── metrics_consumer.py       # MQTT subscriber → InfluxDB/SQLite
    ├── window_collector.py       # Raw window saver + labeling queue
    ├── alert_service.py          # Drift detection + alerting
    ├── simulate_telemetry.py     # Test harness using inference_pipeline.py
    ├── export_labelling_queue.py # Export pending windows for Label Studio
    ├── prepare_retraining_dataset.py  # Merge PLAID + field data
    ├── grafana_dashboard.json    # Import-ready Grafana dashboard config
    ├── docker-compose.yml        # Mosquitto + InfluxDB + Grafana + MinIO
    └── README.md                 # Setup and usage
```

# Dataset Source Recommendations — Fusion-ResNet NILM

All information verified via web search. Confidence levels reflect how directly each dataset supports this project's 30 kHz / 400-sample window format and appliance classes.

---

## What This Model Needs

The model requires **single-appliance current waveforms** sampled at or near **30 kHz**, producing windows of **400 samples** (≈ one 10-cycle block resampled). The existing training data (PLAID) covers 15 appliance classes at exactly this specification.

Data additions should aim for:
1. More diverse signatures for underperforming classes (Fan, ILB, Soldering Iron, AC).
2. Coverage of 50 Hz mains (Ghana deployment target).
3. Real multi-appliance aggregate recordings for 2–5 device combinations.

---

## Dataset Comparison Table

| Dataset | Native Rate | Signals | Granularity | Mains Hz | Classes | Compatible? | License | Priority |
|---|---|---|---|---|---|---|---|---|
| PLAID *(baseline)* | 30 kHz | V + I | Single-appliance | 60 Hz | 15–17 | **Native match** | CC BY 4.0 | — |
| HIFDA | 100 kHz | V + I | Single-appliance | **50 Hz** | 14 | Yes (resample) | CC BY-NC-ND | **High** |
| WHITED | 44.1 kHz | V + I | Single-appliance (transient only) | 50 + 60 Hz | 47 types | Partial | Password-protected | Medium |
| COOLL | 100 kHz | V + I | Single-appliance | 50 Hz | 12 (power tools) | Yes (resample) | CC BY-NC | Low-Medium |
| BLOND-250 | 50 kSPS (indiv.) | V + I | Aggregate + labeled | 50 Hz | 17 (office) | Partial | CC BY 4.0 | Low |
| BLUED | 12 kHz | V + I (aggregate only) | Aggregate only | 60 Hz | ~43 (events) | **No** | Unspecified | Not recommended |
| UK-DALE | 16 kHz (aggregate) / 1/6 Hz (per-appliance) | Aggregate V+I; power only per-appliance | Aggregate only | 50 Hz | 50+ | **No** | CC BY 4.0 | Not recommended |

---

## 1. PLAID — Current Baseline

**Citation:** Medico et al., *Scientific Data* 7, 49 (2020). DOI: 10.1038/s41597-020-0389-7  
**Access:** https://figshare.com/articles/dataset/PLAID/10084619 | CC BY 4.0  
**Sampling rate:** 30 kHz  
**Signal:** Voltage + current, single-appliance  
**Region:** Pittsburgh, PA, USA — **60 Hz**  
**Classes:** 17 types, ~1,876 submetered records  

**Role:** Already in use. The source of all 19,400 training signatures.  
**Remaining value:** The expanded 2020 release (17 types, 330 unique makes/models) likely has more variants than the version currently in use. Downloading the full 2020 version and checking whether more Fan/ILB/Soldering Iron/AC signatures are available could increase diversity without adding any preprocessing burden.  
**Expected benefit:** Medium. Confidence: High.

---

## 2. HIFDA — Top Priority New Source

**Citation:** *Scientific Data* (2025). DOI: 10.1038/s41597-025-04859-3  
**Access:** https://zenodo.org/records/14886758 | **CC BY-NC-ND 4.0**  
**Sampling rate:** 100 kHz (steady-state)  
**Signal:** Voltage + current, single-appliance, steady-state only  
**Region:** Spain — **50 Hz**  
**Classes (14):** Air conditioner, charger, coffee maker, desktop computer, griddle, hair dryer, heater, iron, laptop, lamp, microwave, monitor, vacuum cleaner, washing machine

**Why this matters:**
- **50 Hz mains** — directly relevant for Ghana deployment. Training on HIFDA signatures alongside PLAID would expose all model branches (especially the FFT branch and the hardcoded 60 Hz Fryze voltage waveform) to 50 Hz appliance behavior.
- **Class overlap with current model:** Air conditioner, coffee maker, hair dryer (≈ hairdryer), heater, laptop, microwave, vacuum, washing machine — 8 of 15 model classes are covered.
- **Steady-state** — directly compatible with the model's steady-state window approach.
- **Preprocessing:** Downsample 100 kHz → 30 kHz (simple `scipy.signal.resample`). Extract 400-sample windows per 50 Hz cycle. Map class names to model's 15 class vocabulary.

**Limitations:**
- CC BY-NC-ND prohibits derivative datasets for commercial use. Verify project license before incorporating.
- `iron` maps to Hair Iron, but HIFDA likely measures clothes irons — different appliance. Flag this mapping as uncertain.
- `griddle` has no equivalent in the current 15 classes — exclude it.
- `charger`, `desktop computer`, `monitor` — not in the current 15 classes — exclude them.

**Expected benefit:** High (for 50 Hz adaptation + 8 overlapping classes). Confidence: High.

---

## 3. WHITED — Medium Priority

**Citation:** Kahl et al., *3rd NILM Workshop*, 2016  
**Access:** https://www.cs.cit.tum.de/dis/resources/whited/ | Password: `NILMTUMI13`  
**Sampling rate:** 44.1 kHz  
**Signal:** Voltage + current, single-appliance, **startup transient only** (5 second window)  
**Region:** Germany, Austria, Canada, Indonesia + more — **50 and 60 Hz**  
**Classes:** 47 types, 110 appliances, 1,259 measurements

**Why this matters:**
- Multi-region coverage at both 50 and 60 Hz.
- 47 appliance types — likely includes fans, heaters, and other underperforming classes.
- The **transient** nature is a significant caveat: WHITED captures startup events, not steady-state cycles. The current model uses steady-state windows. Transient signatures are useful for improving robustness at appliance turn-on events but are not directly interchangeable with steady-state PLAID signatures.

**Use strategy:** Extract the steady-state portion of each 5-second WHITED recording (last 1–2 seconds, after startup completes) and use those windows. This requires inspecting each recording. Alternatively, use WHITED signatures only for the FFT branch validation (startup harmonics differ from steady-state harmonics).

**Limitations:**
- Password-protected (not an open license).
- Must resample 44.1 kHz → 30 kHz.
- Transient-only is a compatibility risk — validate on a held-out set before mixing into training.

**Expected benefit:** Medium (startup transient diversity; multi-region 50 Hz). Confidence: Medium.

---

## 4. COOLL — Low-Medium Priority

**Citation:** Picon et al., arXiv:1611.05803 (2016)  
**Access:** https://coolldataset.github.io/ (registration form, Google Drive) | **CC BY-NC 4.0**  
**Sampling rate:** 100 kHz  
**Signal:** Voltage + current, single-appliance, controlled on/off  
**Region:** France — **50 Hz**  
**Classes (12):** Drill, fan, grinder, hair dryer, hedge trimmer, lamp, paint stripper, planer, router, sander, saw, vacuum cleaner

**Class overlap with current model:** Fan (partial — likely a different type than household fan), hair dryer, vacuum cleaner, lamp.

**Why this is low priority:** The majority of COOLL classes are power tools (drill, grinder, saw, etc.) that are not in the current model's 15-class vocabulary. Only 3–4 classes overlap. The Fan in COOLL is likely a workshop/industrial fan — a different load profile than the household Fan in PLAID.

**Use strategy:** Extract fan, hair dryer, and vacuum cleaner signatures only. Downsample 100 kHz → 30 kHz. Flag data as 50 Hz France for consistency tracking.

**Expected benefit:** Low-Medium for 3 overlapping classes. Confidence: Medium.

---

## 5. BLOND — Low Priority

**Citation:** Kriechbaumer et al., *Scientific Data* 5, 180048 (2018)  
**Access:** rsync `rsync://m1375836@dataserv.ub.tum.de/m1375836/` | CC BY 4.0 | **38.7 TB total**  
**Sampling rate:** Individual channels at 6.4 kSPS (BLOND-50) or 50 kSPS (BLOND-250); aggregate at 50/250 kSPS  
**Signal:** V + I, office building, 3-phase  
**Region:** Germany — **50 Hz**  
**Classes:** 17 office equipment types (monitors, laptops, printers, etc.)

**Why this is low priority:** Classes are all office equipment — monitors, laptops, printers, ICT infrastructure. The only overlap with the current model's residential appliance vocabulary is laptop. The 38.7 TB download requirement makes it impractical unless there is specific interest in office deployment scenarios.

**Expected benefit:** Low for current classes. Confidence: High.

---

## 6. BLUED — Not Recommended

**Reason:** Sampled at 12 kHz (below 30 kHz requirement). Provides only whole-house aggregate V+I waveforms — no isolated per-appliance waveforms. Useful only for aggregate-level validation of the full pipeline (post-disaggregation), not for training appliance signature classifiers.

---

## 7. UK-DALE — Not Recommended for Training

**Reason:** The 16 kHz whole-house V+I stream is aggregate (no per-appliance waveforms). Per-appliance channels are only 1/6 Hz active power readings — no waveform data. Useful for benchmarking aggregate disaggregation, not applicable to this model's waveform-level signature identification.

**Possible auxiliary use:** The whole-house 16 kHz aggregate stream (Houses 1, 2, 5) could be used to validate that the inference pipeline correctly handles real residential aggregate data from a 50 Hz grid, but this would require significant preprocessing and the model was not trained for aggregate-input inference.

---

## Ghana-Specific Recommendations

**Most important action for Ghana deployment:** Collect a small local calibration set.

**What to record:**
- 10–20 recordings per appliance class from Ghanaian appliances
- Sample at 30 kHz (or match whichever measurement device is used in the field sensor)
- Record each appliance in isolation for 5–10 seconds at steady-state operation
- Label with appliance type, brand, model if possible
- Record mains voltage simultaneously for Fryze decomposition accuracy

**Why this outweighs adding more external datasets:**
- The model's FryzeDecomposition module hardcodes 60 Hz (`line 162, fusion_resnet.py`). At 50 Hz, the active/reactive separation is systematically wrong.
- FFT harmonic positions differ between 50 and 60 Hz grids.
- Even 10 Ghanaian signatures per class used for fine-tuning would close the domain gap more effectively than 500 HIFDA signatures.

**Minimum viable Ghana calibration set:** 5 appliances × 10 recordings = 50 signatures, 2–3 hours of data collection with a laptop and a current clamp.

---

## Integration Priority Ranking

| Rank | Action | Expected Benefit | Effort | Confidence |
|---|---|---|---|---|
| 1 | Collect 10 Ghana 50 Hz signatures per underperforming class | **High** (direct domain match) | Medium | High |
| 2 | Download full PLAID 2020 (check for new Fan/ILB/Soldering Iron variants) | Medium | Low | High |
| 3 | Download HIFDA from Zenodo; add 8 overlapping classes at 50 Hz | High (50 Hz diversity) | Medium | High |
| 4 | Extract steady-state portion of WHITED; add 50 Hz variants | Medium | Medium-High | Medium |
| 5 | Extract Fan/Vacuum/Hairdryer from COOLL | Low-Medium | Medium | Medium |
| 6 | Fix `FryzeDecomposition` to accept `--mains-freq 50` flag | Medium (for 50 Hz deployment) | Low | High |

---

## Data Integration Guardrails

1. **No test set contamination:** Split by appliance instance or recording session, not by random window sampling from the same recording.
2. **Track provenance:** Add `dataset_source`, `region`, `mains_hz`, `appliance_instance_id` to every window's metadata.
3. **Hold out external-data-only test set:** Evaluate the model on a set that contains only HIFDA/WHITED/COOLL data (no PLAID) to measure cross-dataset generalization. A model that generalizes to HIFDA appliances without retraining is more deployment-ready.
4. **Verify class mapping before training:** `HIFDA iron` ≠ `Hair Iron (styling tool)`. Document every class mapping decision and note uncertainty.
5. **No resampling leakage:** Apply downsampling (100 kHz → 30 kHz) to source recordings before splitting into windows, not after.

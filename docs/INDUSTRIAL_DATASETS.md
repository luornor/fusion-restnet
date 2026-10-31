# Datasets for Commercial and Industrial Appliances: Assessment

**Date:** 2026-10-27
**Question:** Can public datasets let us retrain the NILM model for cold stores, supermarkets, hotels, hospitals, labs and data centres?
**Short answer:** Only for the small single-phase sites (offices, cafés, salons, guesthouses). For the large sites, there is almost no usable public data, and the hardware is the bigger blocker. We will have to collect our own signatures, and the pipeline already supports that.

---

## 1. What a dataset must have to be usable

The model classifies 400-sample windows of **raw current waveform at 30 kHz** from **one single-phase channel**. Training builds mixtures by adding single-appliance windows together. So a useful dataset needs:

1. **Raw current waveforms**, not RMS power or harmonic magnitudes.
2. **A sampling rate of 30 kHz or more.** Data at 6.4 kHz can be upsampled, but it has no content above 3.2 kHz, so the FFT branch sees an empty band that never occurs in real deployments. Mixing rates needs band-limiting augmentation, or the model learns the dataset instead of the appliance.
3. **Submetered, single-appliance recordings.** Aggregate-only data with event labels is much harder to use.
4. **A licence that allows commercial use.** We sell the product.

Most "industrial NILM" datasets fail the first point. They are power readings at 1 Hz or slower, which suits a separate low-rate model but not this one.

---

## 2. Two problems that come before data

### 2.1 The device is single-phase

The data model in `lumina-api` stores one current and one voltage per reading, and the model has one input channel. Most loads on the gap list are **three-phase, 400 V**: cold room and blast freezer compressors, compressor racks, chillers, CRAC units, lifts, commercial laundry, bakery ovens, large UPS systems, and X-ray, CT and MRI. A single CT clamp on one phase sees a third of a balanced load and none of the phase relationships. Large sites need three-phase metering with CTs rated for 100–1000 A before any model work matters.

### 2.2 One meter at the main incomer cannot separate a supermarket

A supermarket can have 20 identical display cases and several identical rack compressors. NILM cannot tell identical machines apart from one aggregate signal, and large loads swamp small ones like tills. Commercial NILM in practice meters **each distribution board or circuit**, then disaggregates within the circuit. Many large machines sit on dedicated circuits, where the job becomes direct metering plus health monitoring, not disaggregation. This is a product decision: a multi-channel unit per distribution board, not one unit per building.

### 2.3 A 13 ms window cannot tell variable-speed loads apart

Behind a variable speed drive (VSD), the mains sees the **drive's rectifier**, not the machine. A VSD pump, conveyor and fan draw near-identical input current. They differ in power level, duty cycle, start-up ramp and time of day, which a 13 ms window cannot see. Industrial classes will need longer-timescale features. That is the same conclusion as the second acquisition mode in `MCSA_PLAN.md` (workspace `docs/` folder, outside this repo).

---

## 3. Dataset catalogue

Ratings: **Usable** means it meets section 1. **Partial** means it is useful with work. **No** means it can't feed this model.

### 3.1 High-rate datasets with commercial or industrial loads

| Dataset | Loads | Rate | Licence | Access | Rating |
|---|---|---|---|---|---|
| **LILAC** (TUM, 2018) | 15 lab-measured industrial types, 1–3 concurrent. Includes three-phase induction motors and a motor on a frequency converter | 50 kHz, three-phase | Not stated | Official link is **dead** (404). Preprocessed subset on GitHub, downloaded | Partial |
| **BLOND** (TUM, 2018) | Office: 53 appliances in 16 classes, three-phase building | 50 or 250 kHz aggregate, 6.4 or 50 kHz per appliance | Not verified | mediaTUM, 38.7 TB, bot check | Partial (offices) |
| **CREAM** (TUM, 2020) | Two industrial coffee machines, component-level labels | 6.4 kHz | CC BY 4.0, per secondary source | mediaTUM, 103 GB, bot check | Partial (cafés) |
| **CAFFEINE** (2023) | 130 coffees: heating coil, grinder, vibration pump, infuser motor | 6.25 kHz | CC BY 4.0, verified | Zenodo 8105706, 1.4 GB, downloaded | Partial (cafés) |
| **KAIST AC motor testbed** (2025) | 1–7.5 HP three-phase 380 V induction motors, faults, VSD speed variation | 100 kHz current | CC BY 4.0, verified on one part | Mendeley, 7 parts, about 60 GB. **Some parts removed** | Partial (motors, MCSA) |
| **KAIST HVAC belt looseness** (2025) | HVAC fan motor, belt fault levels | High-rate current | CC BY 4.0, verified | Mendeley 5tt3zb2pns, 3.4 GB, downloaded | Partial (fans, MCSA) |

### 3.2 High-rate datasets we already use, with unused classes

| Dataset | Unused content relevant to the gap list | Licence |
|---|---|---|
| **COOLL** | 9 of 12 types unused: drills, angle grinders, saws, sanders, planer, router, hedge trimmers, paint stripper, lamps | **CC BY-NC 4.0: non-commercial only** |
| **WHITED** v1.1 | "Household and Industry" set, 1,259 measurements. Pipeline maps 13 types. Uncertain mappings include water heater, mixer and clothes iron | No licence stated, "for research purposes" |
| **PLAID** | 17 types, all residential | CC BY 4.0, verified |
| **HIFDA** | 14 household types, 100 kHz | CC BY 4.0, verified |

### 3.3 Checked and rejected for this model

| Dataset | Why not |
|---|---|
| **IMDELD** (Brazil feed mill) | 1 Hz power only. Meters sample at 8 kHz internally, but only 1 Hz is published |
| **Kulathilaka et al. 2024** (Sri Lanka) | Power every 6 s, assembled from UK-DALE, IMDELD, Pecan Street, plus 1 Hz local data |
| **IEEE DataPort "two industrial scenarios"** | 1 s RMS and harmonics. Subscription required |
| **SmartNIALMeter** (Switzerland) | 1–5 s power, residential |
| **fariddinar/nilm-dataset** (ESP32 + ATM90E36) | RMS and harmonic magnitudes every ~2 s, residential |
| **SIDED** digital twin | Synthetic, 1-minute power |
| **BLUED** | Residential US home, 12 kHz aggregate, event labels only, 57.7 GB. Original host is dead; Hugging Face mirror works |
| **FIRED, TDHA, LIT** | Residential |
| **ORNL Harmonic Signals Dataset** | Injected test signals, not appliances |

---

## 4. Coverage of the gap list

**Public data exists** means a waveform dataset contains the load or a close electrical equivalent. **Proxy** means a similar load type exists but the commercial version differs in power, phase or control. **None** means nothing public.

| Sector | Load | Coverage |
|---|---|---|
| Cold chain | Cold room and blast freezer compressors | Proxy: LILAC and KAIST induction motors. No commercial refrigeration data |
| Cold chain | Condenser and evaporator fans | Proxy: KAIST HVAC fan, fan classes |
| Cold chain | Defrost heaters | Proxy: resistive loads generalise well (heater, LILAC resistor) |
| Cold chain | Ice plant, display and chest freezers | None, or domestic fridges only |
| Food processing | Grinders, mills, mixers | Proxy: COOLL grinders (non-commercial licence), CAFFEINE grinder, WHITED mixer |
| Food processing | Conveyors and motors on VSDs | Proxy: LILAC frequency-converter motor, KAIST VSD motors |
| Food processing | Screw compressors, pasteurisers, bottling lines | None |
| Labs and hospitals | Centrifuges, autoclaves, analysers, imaging, sterilisers, oxygen plant | None |
| Data centres | UPS, rectifiers, CRAC, chillers | None at waveform level |
| Supermarkets | Display cases, compressor racks, bakery ovens, slicers | None. Ovens partly by resistive proxy |
| Supermarkets | Tills and POS | Proxy: BLOND office PCs and monitors |
| Hotels | Lifts, laundry, chillers, kitchen ranges, pumps | None, or domestic washing machines only |
| Guesthouses | Water pumps | None at waveform level |
| Guesthouses | Water heaters | Partial: WHITED water heater |
| Salons | Hood dryers, steamers | Proxy: hairdryer and heater classes |
| Salons | Clippers, sterilisers | None verified |
| Offices | Printers, desktop PCs | Partial: BLOND |
| Offices | Inverter vs split AC | Partial: PLAID AC is fixed-speed. Inverter AC needs collection |
| Cafés | Espresso machines | Usable with work: CAFFEINE, CREAM |
| Cafés | Display fridges, chest freezers | Proxy: domestic fridges |

---

## 5. Recommendation

1. **Ship the small-commercial segments first.** Offices, cafés, salons and guesthouses are single-phase and partly covered: CAFFEINE, CREAM, BLOND, and the unused WHITED classes. This needs no new hardware.
2. **Collect our own signatures. It is the only route for the large sites.** The training pipeline composes mixtures from single-appliance windows, so we only need **submetered single-machine recordings**, not labelled aggregates. That is cheap to collect. A technician clamps a CT on one machine's feed for 10–30 minutes across its operating states, with the device in raw capture mode. Phase 2 of [MONITORING_PIPELINE.md](../MONITORING_PIPELINE.md) already plans the firmware ring buffer and upload. Add a "signature capture" trigger and a labelling form for the appliance type, make, rating and state. Target 2–3 partner sites first: a cold store, a supermarket and a hotel. Field data is also Ghana-native at 50 Hz and 230/400 V, which public data is not.
3. **Decide the large-site hardware before training any industrial classes.** That means three-phase, multi-channel, and per-circuit metering. A model trained for one single-phase channel will not transfer.
4. **Resolve licensing before selling.** COOLL is non-commercial, and the shipped model already trains on 6,000 COOLL windows. WHITED and LILAC state no licence. Either get written permission from the authors or retrain without them. PLAID, HIFDA, CAFFEINE and the KAIST data are CC BY 4.0, which allows commercial use with attribution.

---

## 6. Downloaded so far

All downloads are in `Fusion-ResNet/datasets_raw/`, outside every git repo. Each dataset has its own folder. Treat the contents as untrusted: load `.npy` files with `allow_pickle=False`.

| Folder | Contents | Size |
|---|---|---|
| `lilac_awrgnilm_preprocessed/` | LILAC preprocessed by AWRGNILM. Submetered: 206 events. Aggregated: 1,324 events. Each is 3 phases × 1,000 samples (one 50 Hz cycle at 50 kHz), with labels and on/off state. No licence stated | 72 MB |
| `fariddinar_esp32_harmonics/` | Repo zip. Rejected for this model, kept for reference | 62 MB |
| `plaid2018_figshare_10084619/` | Raw PLAID 2018: 1,876 submetered recordings at 30 kHz plus metadata. MD5 matches figshare | 708 MB |
| `nonlinear_loads_mendeley_nkrhgfp9xy/` | Inverter AC, UPS, fridge, microwave, laptop and phone chargers, TV. One snapshot of about 12 cycles each. SHA-256 verified | 40 MB |
| `liman_c_mendeley_kccmrf3864/` | Metadata only. LIMAN-C current is at 4,096 Hz, kept for motor-health work, not NILM | <1 MB |
| `caffeine_zenodo_8105706/` | CAFFEINE coffee machine dataset, still zipped. MD5 matches Zenodo. 90 Parquet files, 1.9 GB uncompressed | 1.4 GB |
| `kaist_hvac_belt_mendeley_5tt3zb2pns/` | KAIST HVAC fan motor dataset, still zipped. SHA-256 matches Mendeley. Five current files for belt condition levels D0–D4 in TDMS format, plus vibration and RPM files. 12.5 GB uncompressed. Reading TDMS needs the `nptdms` package | 3.4 GB |
| `hifda_zenodo_14886758/` | HIFDA 2025 release, only `Full_time_window_dataset/` (current and voltage, 1,500 text files), fetched member by member from the 4.4 GB zip, CRC-checked | 8.3 GB |
| `whited_tum_gdrive/` | `WhiteD.zip` from the WHITED page's Google Drive link, read in place | 2.1 GB |
| `cooll_gdrive/` | COOLL: 4 current and 4 voltage zips (password in the release), scale factors, configs, headers. All zips pass `testzip` | 820 MB |
| `sided_huggingface/`, `hospital_phoenix_shahid_fakhri/` | Low-rate power data, section 9 | 593 MB |

---

## 7. Revisit list

None of these failures looked like a block by the office network. They are dead links, bot checks, logins, or email-gated passwords. Retry the bot-check sites in a real browser, on any network.

| Dataset | URL | Problem | Action |
|---|---|---|---|
| LILAC full (35 GB) | https://www.cs.cit.tum.de/dis/resources/lilacd/ | Nextcloud share returns 404 | Email matthias.kahl@tum.de for a new link and the licence terms |
| BLOND | https://mediatum.ub.tum.de/1375836 | **Resolved 2026-10-31**: the record page gives public FTP/rsync (user and password `m1375836`). CC BY 4.0 | Sampled download in progress, see section 11 |
| CREAM | https://mediatum.ub.tum.de/1554766 | Anubis bot check. 103 GB | Open in a browser. The rsync password is on the record page |
| PLAID 2017 (raw) | https://figshare.com/articles/dataset/PLAID_2017/11605215 | Bot check for scripts. The API works | 695 MB zip. Needed to rebuild PLAID with recording IDs |
| KAIST AC motor parts | https://data.mendeley.com/datasets/9r82jppsn7/1 and the six linked parts | Part AC.z03 "removed at author's request". 2026-10-29: the dataset API returns error 451 (unavailable for legal reasons) | Dead unless the KAIST authors re-release it. Without all 7 parts the archive won't unpack |
| LIT dataset | https://pessoal.dainf.ct.utfpr.edu.br/douglasrenaux/LIT_Dataset/ | HTTP 464 here. Registration form, password by email | Residential, low priority. Retry from home if wanted |
| COOLL licence | https://coolldataset.github.io/ | CC BY-NC 4.0 | Ask the authors at Univ. Orléans about commercial use, or drop COOLL |
| WHITED licence | https://www.cs.cit.tum.de/dis/resources/whited/ | No licence stated | Email matthias.kahl@tum.de together with the LILAC request |
| IEEE DataPort "two industrial scenarios" | https://ieee-dataport.org/documents/non-intrusive-load-monitoring-datasets-two-industrial-scenarios | Subscription | Only 1 s data. Skip unless there's a low-rate model |
| BLUED | https://huggingface.co/datasets/INFERLab/BLUED | Works, but 57.7 GB and residential | Skip for industrial work |
| **NLN-EMP pumps** (Univ. Twente) | https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0 | 2026-10-29: times out from here and is **refused** (ECONNREFUSED on 131.180.169.22:443) from an outside fetcher. The server is down, not the office network | Retry later on any network. Two centrifugal pumps on VSDs, 3-phase current at 20 kHz, CC0/CC BY. Check whether current was measured on the mains side of the drive |
| FIRED (Univ. Freiburg) | rsync://FIRED@clu.informatik.uni-freiburg.de/FIRED/ (password in the GitHub helper README) | 2026-10-29: port 873 times out. Port 443 answers with a certificate for another name, so no HTTPS route. 3.2 TB total | Try rsync from Colab, fetching only `highFreq` per-appliance files for a few days. 21 appliances at 2 kHz with voltage, CC0 per the helper repo |
| TDHA (NCEPU) | https://doi.org/10.57760/sciencedb.13172 | 2026-10-29: mirror f-lab.ncepu.edu.cn/TDHA lists 21 zips (~40 GB, CC BY 4.0), but every file link returns the site's HTML shell. Science Databank is a JS app and returns an empty body to scripts. In a browser it lists the same zips plus `Background.zip` (30 MB) and per-file MD5s | Download in a browser, one mode per appliance (section 11). 10 household types, 51.2 kHz V+I, 220 V/50 Hz |
| Houidi (IBISC, Univ. Evry) | https://ieee-dataport.org/documents/home-electrical-appliances-recordings-nilm | IEEE DataPort subscription | 61 appliances, 8 steady-state cycles of V+I at 50 Hz, 6.75 MB. Licence not stated |
| LIMAN-C full archive | https://data.mendeley.com/datasets/kccmrf3864/1 | Works. 255 MB | 4,096 Hz: no use for NILM, good for the MCSA plan (1.5 kW 50 Hz motor, 4 fault states) |

---

## 8. Analysis and preparation (2026-10-27, later)

### 8.1 Two problems found in the existing training data

- **Mixed time bases.** PLAID `X_real.npy` windows are 8 voltage-aligned mains
  cycles at 50 samples per cycle, which is what the model's Fryze branch
  needs. The HIFDA, WHITED and COOLL scripts cut 13 ms of raw 30 kHz signal,
  which is two-thirds of a cycle. The shipped model trained on mixtures adding
  one to the other. Inference on raw signals uses a third convention, 10 cycles
  at 40 samples per cycle.
- **Mixed current units.** PLAID is in amps at 120 V. WHITED is normalised per
  recording, so a laptop and a kettle are the same size. HIFDA and COOLL are in
  uncalibrated sensor volts. The relative size of appliances inside a training
  mixture was therefore arbitrary.

### 8.2 What was prepared

All new data is in the **cycles** time base, voltage-aligned like PLAID, in
amps, with recording group ids. Each script is in
`Fusion-ResNet-NILM/preprocessing/`.

| Source | Script | Windows | Groups | Classes added or extended |
|---|---|---:|---:|---|
| PLAID 2018 raw | `prepare_plaid.py` | 38,471 | 318 appliance instances | Rebuilds ids 0–15 with instance groups, replacing `X_real.npy` |
| LILAC | `prepare_lilac.py` | 2,331 | 717 | New: 16, 17, 18, 19, 20. Extends: bulb, coffee maker, fridge, hairdryer, heater, kettle, vacuum |
| CAFFEINE | `prepare_caffeine.py` | 6,668 | 45 trials | New: 21 Espresso machine |
| KAIST HVAC | `prepare_kaist_hvac.py` | 2,700 | 25 | Extends 17 Three-phase induction motor |
| Non-linear loads | `prepare_nonlinear_loads.py` | 19 | 5 | New: 22 Inverter AC, 23 UPS, about 4 windows each, too few to train |

Merged sets:

| Set | Windows | Classes | Use |
|---|---:|---:|---|
| `data/combined_v2` | 71,369 | 24 | Development. Includes LILAC and WHITED (no licence) and COOLL (CC BY-NC). Updated 2026-10-28, see section 10 |
| `data/combined_commercial` | 52,658 | 20 | Anything sold. Loses ids 16, 18, 19, 20 until LILAC's licence is cleared |

A 4-epoch CPU run on `combined_v2` completed end to end: 21 classes after
dropping thin ones, stratified group split over 1,107 groups, calibration,
per-class thresholds, and inference. It proves the pipeline works, not the
model.

### 8.3 Checks that passed

- Every source measures exactly 8.01 cycles per window.
- Phase alignment: resistive loads sit at −85 to −92° in every source, a sine
  from the voltage zero crossing. CAFFEINE and the non-linear loads analyser
  had reversed sensor polarity, now corrected automatically from the sign of
  average power.
- PLAID v2 reproduces legacy PLAID's phase signature: heater −86°, bulb −79°,
  laptop −70° in both.
- KAIST recorded no voltage, so its windows assume a 37° motor lag, a power
  factor of 0.8. That is the one alignment assumption in the set.

### 8.4 What the data can and cannot do

- **The new classes are seeds.** Apart from the three-phase motor, with 4
  units, each new class comes from one physical machine, so field accuracy on
  other makes is unknown.
- **The model cannot tell some new classes apart by shape.** The VSD motor,
  inverter AC and UPS all sit closest to Laptop in spectral profile, because
  each has a rectifier front end. Telling them apart needs power level or
  longer-timescale features.
- **Still needed:** commercial refrigeration, pumps, lifts, ovens, laundry and
  medical equipment remain uncovered. Field signature collection, section 5,
  is still the route.

---

## 9. Low-rate datasets: not for the waveform model (2026-10-28)

These two were suggested and are downloaded, but they can't train
Fusion-ResNet. It needs raw current waveforms at kHz rates. Both are power
readings at one per minute or slower.

| | Hospital (Shahid-Fakhri) | SIDED |
|---|---|---|
| Folder in `datasets_raw/` | `hospital_phoenix_shahid_fakhri/` | `sided_huggingface/` |
| What it is | One hospital in Phoenix, US, hourly kW for one year (8,760 rows) | Simulated facilities from a digital twin, 1-minute power for one year each |
| Real or simulated | **Simulated**, despite the README's "real-world" claim. The columns are EnergyPlus output variables, so this is the US DOE reference hospital from OpenEI | Simulated |
| Breakdown | Fans 9.6%, cooling 43.3%, lighting 11.4%, equipment 17.9%, other 17.7% | EV charging, cooling, background load, solar, combined heat and power |
| Scale | Median 1.1 MW, peak 1.6 MW | Median 25–320 kW |
| Licence | None. "For reference, research, and educational purposes" | None stated on GitHub or Hugging Face |
| Quirks | Gas column included | `Office_LA.csv` duplicates `Logistic_LA.csv` value for value: 8 distinct configurations, not 9. Temperatures are in kelvin |

What they are useful for:

- **Hospital: a sector benchmark.** The end-use split is a reasonable prior for
  what a hospital's bill is made of, such as cooling at about 43%, for the
  agent's benchmark and insight services. It also shows the hardware gap: a
  1.1 MW site draws about 1,600 A per phase at 400 V.
- **SIDED: a separate site-level model, if ever wanted.** A sequence model on
  1-minute smart-meter power could split a facility's total into cooling,
  background, EV and solar. Lumina already stores 1 Hz readings, which resample
  to 1-minute easily. Its relevance to Ghana is limited: background load is
  85–91% of consumption and cooling only 3–15%, far below a Ghanaian
  commercial site. Few Ghanaian facilities have combined heat and power or EV
  chargers.

Neither licence clearly allows commercial use. Treat both as research
references only.

---

## 10. HIFDA, WHITED and COOLL rebuilt (2026-10-28)

The processed arrays in the repo (`data/hifda`, `data/whited`, `data/cooll`,
used for `multi-source-v1`) are 13 ms raw30k windows without voltage, so they
cannot be converted to the cycles time base. All three were rebuilt from raw
data with `_v2` scripts, which replace the legacy sources in a default merge.

| Source | Windows | Units | Finding |
|---|---:|---:|---|
| `whited_v2` | 9,820 | 82 | **Legacy WHITED was voltage.** FLAC channel 0 is voltage (222 V after the kit factor, THD under 1% for every appliance) and channel 1 is current. `prepare_whited.py` read channel 0, so 26,000 of the 67,400 windows in `data/combined` were mains voltage labelled as appliances. v2 reads channel 1 in amps, handles the 60 Hz Canadian recordings, and adds bench grinder, drills, jigsaw, rotary tool, sewing machine and resistive cookers |
| `hifda_v2` | 4,800 | 6 | The current channel lags the voltage by 37.1° on resistive loads, an acquisition offset. Measured on the heater and compensated. The 2025 air conditioner and washing machine do not fit that calibration (current leading by 80° after correction) and are excluded by default |
| `cooll_v2` | 6,560 | 41 | All 12 types instead of 3, in amps via `scaleFactors.txt`, voltage-aligned. 24 power tools for Universal-motor tool. 100 kHz FLAC needs PyAV (`av`); libsndfile loses sync |

Resistive-load phase in the merged set: COOLL −89.9°, HIFDA −92.8°, LILAC
−91.8°, WHITED −85.9°, PLAID −80.5° (PLAID's figure includes bulbs at −79°).

**Implication for `multi-source-v1`:** its F1 macro of 0.79 came from 39%
voltage windows and from mixtures adding 133 ms PLAID windows to 13 ms
windows. It does not show how the model performs on real current.

---

## 11. BLOND and TDHA (2026-10-31)

### BLOND

- **Access:** `ftp://m1375836:m1375836@dataserv.ub.tum.de/BLOND/` (also rsync). CC BY 4.0, so it can go in the commercial set.
- **Layout:** `BLOND-50/<day>/` and `BLOND-250/<day>/`, each with:
  - `clear/`: the aggregate meter.
  - `medal-1` … `medal-15`: the per-appliance meters, 6 sockets each.
- **MEDAL files:** HDF5 with `current1` … `current6` and `voltage` (int16). Each dataset carries `calibration_factor` and `removed_offset` attributes, so calibrated voltage RMS comes out at 230 V.

| Subset | Days | MEDAL rate | File length | File size |
|---|---|---|---|---|
| BLOND-50 | 213 (2016-09-30 onwards) | 6.4 kHz (128 samples per cycle) | 15 min | ~30 MB |
| BLOND-250 | 50 (2017-05-12 onwards) | 50 kHz | 2 min | ~35 MB |

- **Appliances:** `appliance_log.json` maps socket to appliance over time. It lists 54 unique appliances:
  - 18 laptops, 10 USB chargers, 5 monitor models (several physical units each), 4 PCs, 3 dev boards
  - 2 battery chargers, 2 fans, 2 kettles
  - one each: printer, projector, projector screen motor, paper shredder, multi-tool, space heater, daylight lamp, toothbrush
- **Test sessions:** some appliances were plugged in only for logged slots:
  - kettle and multi-tool on MEDAL-2, 2016-11-21 at 11:15–11:20
  - shredder and multi-tool on MEDAL-1, 2016-12-06 and 2016-12-07
  - space heater on MEDAL-1, 2017-06-12 at 11:00–12:00
- **Channel skew:** resistive loads read −99.8° (kettle, BLOND-50, 1,578 W) and −99.0° (heater, BLOND-250, 1,465 W), with third harmonic below 1% of the fundamental. That is a ~9.5° current-channel lag on both rates, the same kind of offset as HIFDA. Compensate with `channel_skew_deg`.
- **Resolution:** 5.4 mA per LSB on sockets 2–6 and 15 mA on socket 1. USB chargers (5–15 W, about 0.05 A) are mostly quantisation noise, so filter windows by RMS.
- **Holidays:** 2016-10-03 is a German holiday and every socket was in standby. The downloader skips weekends and holidays.
- **Sampled download:** `preprocessing/fetch_blond.py` covers each of the 90 socket-appliance instances on up to 3 ordinary weekdays, fetching only the MEDAL that holds it. That is 2 files per day in BLOND-50 and 4 in BLOND-250, plus the test sessions: 422 files, 14 GB. A first plan that fetched every MEDAL for each layout change came to 84.5 GB.
- **Result (`prepare_blond.py`):** 2,805 windows in 20 appliance-model groups.

  | Class | Windows | Units |
  |---|---:|---:|
  | Laptop | 2,335 | 15 models |
  | Fan | 430 | 2 |
  | Universal-motor tool (multi-tool) | 20 | 1 |
  | Heater | 10 | 1 |
  | Water kettle | 10 | 1 |

  - Resistive phase after correction: −89.5° and −90.2°.
  - Monitors, PCs, chargers, the daylight lamp and the shredder are UNCERTAIN mappings, kept out of the default build.
  - The printer and projector are unmapped.
  - The Clatronic kettle (MEDAL-13) was not running in any sampled file.
  - Groups are model names, because the multi-tool moved between MEDAL-1 and MEDAL-2.

### TDHA

The Science Databank page lists 22 zips:

- `Aggregate.zip`: 4.3 GB
- `Background.zip`: 30 MB
- 20 single-appliance zips: 1.4–2.4 GB each, several operating modes of the same unit

Each appliance is probably one physical unit, so extra modes add little group diversity. Fetch one zip per appliance and skip `Aggregate`:

- `WaterHeater` (fetch first, for the resistive-phase check)
- `EggBoiler`
- `Fan_1stSpeed`
- `Hairdryer_1stSpeedLowTemperatureWind`
- `MicrowaveHighHeat`
- `Displayer`
- `TableLamp`
- `MobilePhone`
- `Router`
- `LargeHumidifierMaxPower`

That is about 18 GB.

---

## Sources

- LILAC: [paper](https://www.degruyterbrill.com/document/doi/10.1515/teme-2018-0038/html), [TUM page](https://www.cs.cit.tum.de/dis/resources/lilacd/), [preprocessed copy](https://github.com/sambaiga/AWRGNILM)
- BLOND: [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870472/), [data](https://mediatum.ub.tum.de/1375836)
- CREAM: [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7747629/), [data](https://mediatum.ub.tum.de/1554766)
- CAFFEINE: [Zenodo](https://zenodo.org/records/8105706)
- KAIST motor and HVAC: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361783/), [HVAC data](https://data.mendeley.com/datasets/5tt3zb2pns/1), [motor AC.z05](https://data.mendeley.com/datasets/gr8c9ckhxh)
- IMDELD: [IEEE DataPort](https://ieee-dataport.org/open-access/industrial-machines-dataset-electrical-load-disaggregation)
- Kulathilaka et al.: [Energies 2024](https://doi.org/10.3390/en17153802)
- SmartNIALMeter: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403396/)
- fariddinar: [GitHub](https://github.com/fariddinar/nilm-dataset)
- BLUED: [Hugging Face](https://huggingface.co/datasets/INFERLab/BLUED)
- COOLL: [site and licence](https://coolldataset.github.io/), [paper](https://arxiv.org/abs/1611.05803)
- PLAID: [figshare 2017](https://figshare.com/articles/PLAID_2017/11605215), [figshare 2018](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619)
- Non-linear household loads: [Mendeley](https://data.mendeley.com/datasets/nkrhgfp9xy/1), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12173743/)
- LIMAN-C: [Mendeley](https://data.mendeley.com/datasets/kccmrf3864/1)
- NLN-EMP pumps: [4TU](https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0), [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10751838/)
- Three-phase IM current under load: [Mendeley](https://data.mendeley.com/datasets/gxdd74czwh/1)
- HIFDA: [Zenodo](https://zenodo.org/records/13884628)
- SIDED digital twin: [arXiv 2506.20525](https://arxiv.org/html/2506.20525v2), [GitHub](https://github.com/ChristianInterno/SIDED), [Hugging Face](https://huggingface.co/datasets/CInterno/Synthetic_Industrial_Dataset_For_Energy_Disaggregation_SIDED)
- Hospital electricity consumption: [GitHub](https://github.com/Shahid-Fakhri/Electricity-Consumption)
- Industrial NILM on IMDELD: [arXiv 2607.28693](https://arxiv.org/pdf/2607.28693)
- VSD cooling NILM: [MIT](https://emsg.mit.edu/wp-content/uploads/2024/05/Nonintrusive_Load_Monitoring_of_Variable_Speed_Drive_Cooling_Systems.pdf)
- LIT: [site](https://pessoal.dainf.ct.utfpr.edu.br/douglasrenaux/LIT_Dataset)
- FIRED: [project](https://ira.informatik.uni-freiburg.de/en/src/projects_view.php%3FprojectID=92.html)
- ORNL HSD: [dataset](https://impact.ornl.gov/en/datasets/harmonic-signals-dataset/)

# Agent handoff: Fusion-ResNet NILM

Read this before changing anything. It records the project state and the rules
that are not obvious from the code. Last updated 2026-10-31.

The model identifies active appliances from one single-phase current channel.
Training runs in Google Colab. Preprocessing runs locally, because the raw data
(about 14 GB) lives outside this repo in `../datasets_raw/`.

Sibling repos in the same workspace: `lumina-agent` and `lumina-api`
(Centient-Technologies). The dataset research is in
[docs/INDUSTRIAL_DATASETS.md](docs/INDUSTRIAL_DATASETS.md):

- section 7: the revisit list
- section 10: the v2 rebuild
- section 11: BLOND and TDHA

## Current state

- All sources are rebuilt in the **cycles** time base: 8 mains cycles × 50
  samples, starting at the voltage's rising zero crossing. The scripts are
  `preprocessing/prepare_*.py` and the `_v2` variants.
- Merged sets, built by `preprocessing/merge_datasets.py`:
  - `data/combined_v2`: 74,174 windows, 24 classes, 1,301 groups. Development only.
  - `data/combined_commercial`: 55,463 windows, 21 classes, 461 groups. Use for any model that is sold.
  - BLOND (`data/blond`, 2,805 windows, 20 appliance-model groups) is included in both.
  - The `X`/`y`/groups arrays are gitignored. Rebuild them with the merge commands in README.md.
- **No model has been trained on the v2 data yet.** The shipped
  `multi-source-v1` (F1 macro 0.79) was trained on bad data:
  - 39% of its windows are WHITED **voltage**, not current.
  - It mixes 13 ms raw30k windows with 133 ms cycle windows.
  - Its test split did not keep each appliance unit on one side, so the 0.79 is inflated.
- Data collection is in progress. The user wants every usable dataset
  in before the final training run. A baseline run on `combined_v2` is
  recommended first, so each new source can be measured.

## Next steps, in order

1. **LIT** (UTFPR): the zips are in `../datasets_raw/lit_utfpr/`. Every inner file is zip-encrypted, so the password from the user's registration email is needed.
   - Use Synthetic `sy1`: 26 types × 16 recordings, one appliance each, with event labels. Its 3 `.bin` channels per recording are not yet decoded.
   - `sy2`, `sy3` and `sy8` are multi-load, Natural is a 2 h aggregate, and Simulated is model output: skip Simulated.
   - 127 V / 60 Hz, 15 kHz. Licence: check the registration email.
2. Prepare any other raw datasets the user drops in `../datasets_raw/`:
   - **TDHA** (Science Databank 10.57760/sciencedb.13172): 51.2 kHz V+I, 220 V/50 Hz. Docs section 11 lists which zips to fetch.
   - **Houidi** (IEEE DataPort): 61 appliances, 8 cycles of V+I at 50 Hz, already the window shape we use.
   - **CREAM** (mediaTUM, rsync details on the record page).
   - **BLOND, optional:** a targeted fetch of MEDAL-13 (Clatronic kettle, 2017-02 to 2017-06). The sampled office-hour files missed it.
   - **FIRED** (rsync, 2 kHz per appliance): the server was down on 2026-10-29.
   - **NLN-EMP pumps** (4TU): the server was down on 2026-10-29.

   For each new source:
   - Write `prepare_<name>.py` on the pattern of `prepare_whited_v2.py`, using `signature_utils.zc_windows`.
   - Add the source to `class_registry.json` and to `SOURCES` in `merge_datasets.py`.
   - Check the resistive phase (below), then re-merge.
3. Optional: extract event signatures from BLUED (Hugging Face `INFERLab/BLUED`, 57.7 GB). Use the steady state after each labelled event minus the steady state before it.
4. Train in Colab:
   ```bash
   python training/train_fusion_resnet.py --data-dir data/combined_v2 \
       --group-split --current-prior --calibrate \
       --mains-freq 50 --mains-volt 230 --model-version 2.0.0
   ```
   - Save checkpoints to Drive and resume with `--resume-from`.
   - Use `data/combined_commercial` for the sellable model.
   - Expect a lower F1 than v1. The new number is the honest one.
5. Firmware: it hardcodes 15 classes and 16 ICA components, so a model with more classes needs a matching firmware change. Also, `segment_continuous_signal` resamples **10** cycles into 400 samples. That does not match training; check it against the firmware's `preprocessor.go`.

## Rules that are easy to break

- **Never mix time bases.** `merge_datasets.py` refuses to unless `--allow-mixed-timebase` is passed. Legacy `data/hifda`, `data/whited` and `data/cooll` are raw30k: don't use them.
- **Check every new source with these two tests:**
  - The fundamental sits in FFT bin 8.
  - A resistive load (heater, kettle) has a phase of about −80° to −93° under the repo's convention.
  - A wrong channel or a channel skew shows up immediately.
- **WHITED:** FLAC channel 0 is voltage and channel 1 is current. Kit calibration factors are in `prepare_whited_v2.py`.
- **HIFDA:** the current channel lags voltage by 37° from acquisition. `prepare_hifda_v2.py` measures this on Heater records and compensates. Its AC and washing machine don't fit the calibration and are excluded unless `--include-uncertain`.
- **COOLL:** zip password `prismecooll`. The 100 kHz FLAC needs PyAV (`av`), because soundfile fails on it.
- **Class ids 0–15** are the original PLAID label space and never change. New classes are appended in `class_registry.json`.
- **Licences:**
  - COOLL is CC BY-NC.
  - WHITED and LILAC state no licence.
  - The commercial set excludes all three.
  - Email requests to the authors are pending.
- **Raw downloads are untrusted.** Load `.npy` files with `allow_pickle=False`, and keep scripts out of the data folders.
- Don't commit unless the user asks.

## Pending on the user's side

- **Browser downloads:**
  - TDHA: the 10 zips in docs section 11, into `../datasets_raw/tdha_scidb/`.
  - CREAM: send the rsync details from the record page.
  - Houidi: needs the DataPort subscription.
- **Emails to authors:**
  - matthias.kahl@tum.de: LILAC full link, plus WHITED and LILAC licences.
  - COOLL authors: commercial use.
- **LIT dataset registration:** https://pessoal.dainf.ct.utfpr.edu.br/douglasrenaux/LIT_Dataset/
- **Retry later:** 4TU and FIRED.

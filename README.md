# SHUDH-ASR — Phase 1: Baseline

This is the very first thing to run. Do these steps **in order, in your
VSCode terminal, on your own machine** (not in a notebook, keep it simple).

## Step 1 — create the folder and copy these files in

Put `requirements.txt`, `src/download_data.py`, and `src/run_baseline.py`
into a new folder called `shudh-asr` on your machine, matching this layout:

```
shudh-asr/
├── requirements.txt
├── data/              <- create this empty folder
└── src/
    ├── download_data.py
    └── run_baseline.py
```

```bash
mkdir -p shudh-asr/data shudh-asr/src
cd shudh-asr
# now copy the 3 files into place
```

## Step 2 — set up Python

```bash
conda create -n shudh-asr python=3.11 -y
conda activate shudh-asr
pip install -r requirements.txt
```

If you don't have conda, `python3 -m venv venv && source venv/bin/activate`
works the same way.

## Step 3 — download a small test set (takes ~1 minute)

```bash
cd src
python download_data.py
```

This pulls 20 real Hindi audio clips with correct transcripts from Google's
public FLEURS dataset (no login needed). It's just to prove your pipeline
works before you deal with the bigger, messier MUCS 2021 code-switching
dataset.

## Step 4 — run the baseline ASR and get your first WER number

```bash
python run_baseline.py
```

First run downloads the Whisper model (~1.5GB) — be patient, this only
happens once. At the end you'll see something like:

```
BASELINE RESULTS (openai/whisper-medium)
  Word Error Rate (WER): 14.32%
  Char Error Rate (CER): 6.10%
```

**Write this number down.** This is Row 1 of your final comparison table.

## What "success" looks like at this stage

You don't need a low WER yet — you need the *pipeline to run end-to-end*
without errors. A working 15-20% WER on FLEURS (clean, non-code-switched
Hindi) is completely normal and fine.

## Next steps (don't do these yet — one at a time)

1. Swap `MODEL_NAME` in `run_baseline.py` for the real IndicWhisper
   checkpoint from https://models.ai4bharat.org/ for Hindi/Telugu.
2. Download the real MUCS 2021 code-switching test set from
   https://openslr.org/104/ and re-run baseline on that — this is your
   real Phase 1 number (WER will be much higher, 28-34%, that's expected
   and matches the literature).
3. `src/phase2_naive_correction.py` and `src/phase3_build_kb.py` are
   already in this zip (written, syntax-checked) but NOT yet confirmed to
   run on your machine. Run Phase 1 successfully first, confirm the WER
   number prints, THEN run Phase 2:
   ```
   python phase2_naive_correction.py
   ```
   Report the output back before touching Phase 3.

## If something breaks

Paste the exact error message back here — don't guess at fixes, I'll tell
you exactly what's wrong and give you the corrected file.

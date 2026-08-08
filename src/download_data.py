"""
Windows-safe data download.

FLEURS provides each audio clip as raw encoded bytes (a real .wav file's
bytes). We grab those bytes with decode=False (so the `datasets` library
never tries to decode audio itself -- that's what was crashing on Windows)
and just write them straight to disk as .wav files. Loading them back later
with `soundfile` (in run_baseline.py) needs no ffmpeg/torchcodec at all.

Run:
    python download_data.py
"""

import os
import pickle
from datasets import load_dataset, Audio

DATA_DIR = "../data"
AUDIO_DIR = "../data/audio"

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("Downloading a small Hindi test set (FLEURS hi_in, test split)...")
    ds = load_dataset("google/fleurs", "hi_in", split="test", streaming=True)
    # decode=False = give us raw bytes, don't try to decode audio ourselves
    ds = ds.cast_column("audio", Audio(decode=False))

    metadata = []
    for i, sample in enumerate(ds):
        raw_bytes = sample["audio"]["bytes"]
        filename = f"sample_{i:02d}.wav"
        with open(f"{AUDIO_DIR}/{filename}", "wb") as f:
            f.write(raw_bytes)
        metadata.append({
            "file": filename,
            "reference_text": sample["transcription"],
        })
        if i >= 19:
            break

    with open(f"{DATA_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {len(metadata)} Hindi FLEURS clips and metadata to {DATA_DIR}/")
    print("\nNEXT STEP: once MUCS 2021 code-switching data matters for your real")
    print("results (Phase 8 comparison table), download it from:")
    print("  https://openslr.org/104/")

if __name__ == "__main__":
    main()

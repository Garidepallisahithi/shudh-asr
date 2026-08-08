"""
Phase 1 baseline -- Windows-safe version.

This version NEVER uses transformers.pipeline() and NEVER lets the
`datasets` library decode audio. It reads .wav files directly with
soundfile, and calls WhisperProcessor / WhisperForConditionalGeneration
directly. This avoids torchcodec entirely, which is what was crashing.

Expects:
  data/audio/sample_00.wav, sample_01.wav, ...
  data/metadata.pkl  -> list of dicts: {"file": "sample_00.wav", "reference_text": "..."}

Run:
    python run_baseline.py
"""

import pickle
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

MODEL_NAME = "openai/whisper-small"  # swap for IndicWhisper checkpoint later
DATA_DIR = "../data"
AUDIO_DIR = "../data/audio"

def load_model():
    print(f"Loading {MODEL_NAME} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}.")
    return processor, model, device

def read_wav(path, target_sr=16000):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:          # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # simple check -- FLEURS/Whisper expect 16kHz. If this ever prints,
        # tell me and I'll add real resampling (librosa.resample).
        print(f"  WARNING: {path} is {sr}Hz, Whisper expects {target_sr}Hz")
    return audio, sr

def transcribe(processor, model, device, audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def main():
    with open(f"{DATA_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded metadata for {len(metadata)} clips.")

    processor, model, device = load_model()

    hypotheses, references = [], []
    for i, item in enumerate(metadata):
        wav_path = f"{AUDIO_DIR}/{item['file']}"
        audio, sr = read_wav(wav_path)
        hyp = transcribe(processor, model, device, audio, sr)
        hypotheses.append(hyp)
        references.append(item["reference_text"])
        print(f"[{i+1}/{len(metadata)}]")
        print(f"  REF: {item['reference_text']}")
        print(f"  HYP: {hyp}")

    overall_wer = wer(references, hypotheses)
    overall_cer = cer(references, hypotheses)

    print("\n" + "=" * 50)
    print(f"BASELINE RESULTS ({MODEL_NAME})")
    print(f"  Word Error Rate (WER): {overall_wer * 100:.2f}%")
    print(f"  Char Error Rate (CER): {overall_cer * 100:.2f}%")
    print("=" * 50)

    with open(f"{DATA_DIR}/baseline_results.pkl", "wb") as f:
        pickle.dump({
            "references": references,
            "hypotheses": hypotheses,
            "wer": overall_wer,
            "cer": overall_cer,
            "model_name": MODEL_NAME,
        }, f)
    print("Saved to data/baseline_results.pkl for later phases.")

if __name__ == "__main__":
    main()

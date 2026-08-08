import pickle
import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
import faiss
from sentence_transformers import SentenceTransformer
 
MODEL_NAME = "openai/whisper-small"  # swap for IndicWhisper checkpoint later
DATA_DIR = "../data/domain"
AUDIO_DIR = "../data/domain/audio"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Retuned based on Phase 3's own sanity-check output: after enriching KB
# terms with descriptive context (to fix the earlier acronym false-positive
# problem), genuine correct matches now score 0.33-0.6, not 0.8+. The old
# 0.8 threshold was calibrated against the PRE-enrichment score distribution
# and made the enriched KB unable to fire on anything, clean or noisy.
MIN_SIMILARITY = 0.45
HIGH_THRESHOLD = 0.55
LOW_THRESHOLD = 0.45
 
 
# ---------- ASR (same approach as run_baseline.py, but reads mp3 via librosa) ----------
 
def load_asr_model():
    print(f"Loading {MODEL_NAME} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device
 
 
def transcribe(processor, model, device, audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
 
 
# ---------- SHUDH-ASR pipeline (same logic as phase7_explainability.py) ----------
 
def get_ngrams(text, min_n=2, max_n=4):
    words = text.split()
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i + n]))
    return ngrams
 
 
def retrieve(hyp, embed_model, index, terms, k=1):
    ngrams = get_ngrams(hyp)
    if not ngrams:
        return []
    embeddings = embed_model.encode(ngrams, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(embeddings, k)
    best_per_term = {}
    for ngram, ngram_scores, ngram_idxs in zip(ngrams, scores, idxs):
        for i, s in zip(ngram_idxs, ngram_scores):
            if s >= MIN_SIMILARITY:
                term = terms[i]
                if term not in best_per_term or s > best_per_term[term][1]:
                    best_per_term[term] = (ngram, float(s))
    ranked = sorted(best_per_term.items(), key=lambda x: -x[1][1])
    return [(ngram, term, score) for term, (ngram, score) in ranked][:3]
 
 
def correct(hyp, candidates):
    if not candidates:
        return hyp, None
    best_ngram, best_term, best_score = candidates[0]
    corrected = hyp.replace(best_ngram, best_term, 1) if best_ngram in hyp else hyp
    return corrected, (best_ngram, best_term, best_score)
 
 
def decide(confidence):
    if confidence >= HIGH_THRESHOLD:
        return "AUTO-CORRECT"
    elif confidence >= LOW_THRESHOLD:
        return "FLAG"
    else:
        return "ABSTAIN"
 
 
def shudh_asr_correct(hyp, index, terms, embed_model):
    candidates = retrieve(hyp, embed_model, index, terms)
    if not candidates:
        return hyp, "NO CORRECTION PROPOSED", None
    proposed, match_info = correct(hyp, candidates)
    confidence = match_info[2]
    decision = decide(confidence)
    final_text = proposed if decision == "AUTO-CORRECT" else hyp
    return final_text, decision, match_info
 
 
def main():
    with open(f"{DATA_DIR}/domain_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded {len(metadata)} domain test clips.")
 
    processor, model, device = load_asr_model()
 
    hypotheses, references = [], []
    for i, item in enumerate(metadata):
        path = f"{AUDIO_DIR}/{item['file']}"
        audio, sr = librosa.load(path, sr=16000)  # librosa handles mp3 via ffmpeg
        hyp = transcribe(processor, model, device, audio, sr)
        hypotheses.append(hyp)
        references.append(item["reference_text"])
        print(f"[{i+1}/{len(metadata)}]")
        print(f"  REF: {item['reference_text']}")
        print(f"  HYP: {hyp}")
 
    baseline_wer = wer(references, hypotheses)
    baseline_cer = cer(references, hypotheses)
    print(f"\nDOMAIN BASELINE: WER = {baseline_wer*100:.2f}%, CER = {baseline_cer*100:.2f}%")
 
    with open(f"{DATA_DIR}/domain_baseline_results.pkl", "wb") as f:
        pickle.dump({"references": references, "hypotheses": hypotheses,
                     "wer": baseline_wer, "cer": baseline_cer}, f)
 
    # Now run the full SHUDH-ASR pipeline on these real hypotheses
    print("\n" + "=" * 60)
    print("RUNNING FULL SHUDH-ASR PIPELINE ON DOMAIN TEST SET")
    print("=" * 60)
 
    index = faiss.read_index("../kb/domain_index.faiss")
    with open("../kb/domain_terms.pkl", "rb") as f:
        terms = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
 
    corrected_hyps = []
    auto_correct_count, flag_count, abstain_count, none_count = 0, 0, 0, 0
    for i, hyp in enumerate(hypotheses):
        final_text, decision, match_info = shudh_asr_correct(hyp, index, terms, embed_model)
        corrected_hyps.append(final_text)
        print(f"\n[{i+1}]")
        print(f"  ASR output: {hyp}")
        print(f"  Reference:  {references[i]}")
        print(f"  Decision:   {decision}")
        if match_info:
            print(f"  Match:      {match_info}")
        print(f"  Final:      {final_text}")
        if decision == "AUTO-CORRECT":
            auto_correct_count += 1
        elif decision == "FLAG":
            flag_count += 1
        elif decision == "ABSTAIN":
            abstain_count += 1
        else:
            none_count += 1
 
    shudh_wer = wer(references, corrected_hyps)
    shudh_cer = cer(references, corrected_hyps)
 
    print("\n" + "=" * 60)
    print("FINAL PHASE 8 COMPARISON -- REAL DOMAIN TEST SET")
    print("=" * 60)
    print(f"  Raw ASR baseline WER:        {baseline_wer*100:.2f}%")
    print(f"  SHUDH-ASR (full pipeline):   {shudh_wer*100:.2f}%")
    print(f"\n  Decisions made: {auto_correct_count} auto-corrected, "
          f"{flag_count} flagged for review, {abstain_count} abstained, "
          f"{none_count} no correction needed")
    print("=" * 60)
 
    with open(f"{DATA_DIR}/domain_shudh_results.pkl", "wb") as f:
        pickle.dump({"references": references, "hypotheses": corrected_hyps,
                     "wer": shudh_wer, "cer": shudh_cer}, f)
    print(f"\nSaved final results to {DATA_DIR}/domain_shudh_results.pkl")
 
 
if __name__ == "__main__":
    main()

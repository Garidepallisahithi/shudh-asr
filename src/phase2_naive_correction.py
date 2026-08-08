import pickle
from transformers import pipeline
from jiwer import wer, cer
 
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # small enough to run on CPU
 
def main():
    with open("../data/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
 
    print(f"Loading {MODEL_NAME} (small model, CPU-friendly, ~3GB download)...")
    corrector = pipeline("text-generation", model=MODEL_NAME, device_map="auto")
 
    corrected = []
    for i, hyp in enumerate(baseline["hypotheses"]):
        prompt = (
            "You are correcting errors in a Hindi speech-to-text transcript. "
            "The transcript below is written in Hindi (Devanagari script) and contains "
            "speech-recognition mistakes. Fix ONLY the spelling/word errors. "
            "Your output MUST be in Hindi (Devanagari script), the SAME language as "
            "the input. Do NOT translate to English. Do NOT explain. "
            "Return ONLY the corrected Hindi sentence.\n\n"
            f"Transcript: {hyp}\nCorrected (in Hindi):"
        )
        out = corrector(
            prompt,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.3,   # stops the model looping the same word
            no_repeat_ngram_size=3,   # extra guard against repeat loops
        )[0]["generated_text"]
        # strip the prompt back off, keep only what the model added
        fixed = out[len(prompt):].strip().split("\n")[0]
        corrected.append(fixed if fixed else hyp)
        print(f"[{i+1}/{len(baseline['hypotheses'])}] {hyp}  -->  {fixed}")
 
    new_wer = wer(baseline["references"], corrected)
    new_cer = cer(baseline["references"], corrected)
 
    print("\n" + "=" * 50)
    print("PHASE 2 RESULTS: naive LLM correction (no retrieval)")
    print(f"  Baseline WER (Phase 1):        {baseline['wer']*100:.2f}%")
    print(f"  Naive-correction WER (Phase 2): {new_wer*100:.2f}%")
    if new_wer > baseline["wer"]:
        print("  -> WER got WORSE. This is the expected, literature-documented result.")
    else:
        print("  -> WER improved -- note this, it's worth double-checking your prompt/model choice.")
    print("=" * 50)
 
    with open("../data/phase2_results.pkl", "wb") as f:
        pickle.dump({
            "references": baseline["references"],
            "hypotheses": corrected,
            "wer": new_wer,
            "cer": new_cer,
        }, f)
    print("Saved to data/phase2_results.pkl")
 
if __name__ == "__main__":
    main()
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from jiwer import wer, cer
 
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
 
def load_kb():
    index = faiss.read_index("../kb/domain_index.faiss")
    with open("../kb/domain_terms.pkl", "rb") as f:
        terms = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
    return index, terms, embed_model
 
def retrieve_candidates(query, embed_model, index, terms, k=3, min_similarity=0.5):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    # Only keep candidates that are ACTUALLY relevant. This is what stops the
    # model from being handed irrelevant banking terms for a sentence about
    # chocolate, which is what caused the hallucinated bank-account example.
    results = [
        terms[i] for i, s in zip(idxs[0], scores[0]) if s >= min_similarity
    ]
    return results
 
def main():
    with open("../data/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
 
    index, terms, embed_model = load_kb()
    print(f"Loading {LLM_MODEL} ...")
    corrector = pipeline("text-generation", model=LLM_MODEL, device_map="auto")
 
    corrected = []
    for i, hyp in enumerate(baseline["hypotheses"]):
        candidates = retrieve_candidates(hyp, embed_model, index, terms)
 
        if not candidates:
            # Nothing relevant found -- don't touch the transcript at all.
            # This is the correct, safe behavior: no grounding = no correction.
            corrected.append(hyp)
            print(f"[{i+1}/{len(baseline['hypotheses'])}]")
            print(f"  HYP:        {hyp}")
            print(f"  candidates: (none relevant enough -- left unchanged)")
            continue
 
        prompt = (
            "You are correcting errors in a Hindi speech-to-text transcript. "
            "The transcript is in Hindi (Devanagari script). Here are known "
            f"correct domain terms that MIGHT be relevant: {', '.join(candidates)}\n"
            "If one of these terms was misheard in the transcript, correct it "
            "using the exact term from the list. If none of them clearly apply, "
            "leave the transcript unchanged. Your output MUST be in Hindi "
            "(Devanagari script), the SAME language as the input -- do NOT "
            "translate to English, do NOT explain, return ONLY the corrected "
            "Hindi sentence.\n\n"
            f"Transcript: {hyp}\nCorrected (in Hindi):"
        )
        out = corrector(
            prompt,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )[0]["generated_text"]
        fixed = out[len(prompt):].strip().split("\n")[0]
        corrected.append(fixed if fixed else hyp)
        print(f"[{i+1}/{len(baseline['hypotheses'])}]")
        print(f"  HYP:        {hyp}")
        print(f"  candidates: {candidates}")
        print(f"  CORRECTED:  {fixed}")
 
    new_wer = wer(baseline["references"], corrected)
    new_cer = cer(baseline["references"], corrected)
 
    print("\n" + "=" * 50)
    print("PHASE 4 RESULTS: RAG-grounded correction")
    print(f"  Baseline WER (Phase 1):         {baseline['wer']*100:.2f}%")
    print(f"  RAG-grounded correction WER:    {new_wer*100:.2f}%")
    print("=" * 50)
    print("\nIMPORTANT CONTEXT: today's test set (FLEURS) is general Hindi news")
    print("speech, NOT banking/PMJDY speech -- so the domain KB has little to")
    print("retrieve here, and WER may not move much on THIS data. That is")
    print("expected and NOT a failure of the architecture. The real proof of")
    print("this mechanism needs domain-matched test data (banking-vocabulary")
    print("speech), which is a later data-collection step, not a code problem.")
    print("Today's goal was just: does the retrieval+LLM pipeline run correctly")
    print("end-to-end? If it printed candidates and corrections above, yes.")
 
    with open("../data/phase4_results.pkl", "wb") as f:
        pickle.dump({
            "references": baseline["references"],
            "hypotheses": corrected,
            "wer": new_wer,
            "cer": new_cer,
        }, f)
    print("\nSaved to data/phase4_results.pkl")
 
if __name__ == "__main__":
    main()
 












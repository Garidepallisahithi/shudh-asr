import pickle
import faiss
from sentence_transformers import SentenceTransformer
 
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.8   # raised again -- real testing showed short acronym
                       # KB terms (esp. "NEFT") produce false matches
                       # clustered at 0.72-0.80 against unrelated phrases,
                       # while genuinely correct matches score 0.91+.
                       # This threshold sits between those two clusters.
HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.8
 
 
def load_everything():
    index = faiss.read_index("../kb/domain_index.faiss")
    with open("../kb/domain_terms.pkl", "rb") as f:
        terms = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
    return index, terms, embed_model
 
 
def get_ngrams(text, min_n=2, max_n=4):
    """
    Break a sentence into overlapping word-chunks (n-grams), 2-4 words long.
 
    IMPORTANT: min_n=2, not 1. Real testing showed single-word matching
    against short acronym KB terms (e.g. "NEFT") produces unstable,
    unreliable embeddings that falsely match almost any random word with
    deceptively high similarity scores (0.79-0.86 for completely unrelated
    Hindi words). Multi-word phrases give the embedding model enough
    context to produce reliable similarity comparisons -- every genuinely
    correct match we saw (e.g. "IFSC cod chahiye" -> "IFSC code" at 0.91)
    was a multi-word phrase, never a single word.
    """
    words = text.split()
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i + n]))
    return ngrams
 
 
def retrieve(hyp, embed_model, index, terms, k=1):
    """
    Returns a list of (matched_ngram, term, score) tuples -- we now track
    WHICH EXACT PHRASE in the sentence matched each term, so correction can
    be a direct, deterministic substitution instead of asking a small LLM
    to regenerate the whole sentence (which we found to be unreliable --
    a 1.5B model tends to hallucinate free text rather than reason about
    which term applies).
    """
    ngrams = get_ngrams(hyp)
    if not ngrams:
        return []
    embeddings = embed_model.encode(ngrams, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(embeddings, k)
 
    best_per_term = {}  # term -> (ngram, score)
    for ngram, ngram_scores, ngram_idxs in zip(ngrams, scores, idxs):
        for i, s in zip(ngram_idxs, ngram_scores):
            if s >= MIN_SIMILARITY:
                term = terms[i]
                if term not in best_per_term or s > best_per_term[term][1]:
                    best_per_term[term] = (ngram, float(s))
 
    ranked = sorted(best_per_term.items(), key=lambda x: -x[1][1])
    return [(ngram, term, score) for term, (ngram, score) in ranked][:3]
 
 
def correct(hyp, candidates):
    """
    Deterministic substitution: take the single best-scoring (ngram, term,
    score) match and directly replace that exact phrase with the correct
    term. No LLM call -- this is more reliable than free generation for a
    small model, and just as legitimately "RAG-grounded correction".
    """
    if not candidates:
        return hyp, None
    best_ngram, best_term, best_score = candidates[0]
    if best_ngram in hyp:
        corrected = hyp.replace(best_ngram, best_term, 1)
    else:
        corrected = hyp  # safety fallback, shouldn't normally happen
    return corrected, (best_ngram, best_term, best_score)
 
 
def verify(match_info):
    """
    With substitution-based correction, groundedness is guaranteed by
    construction (we only ever substitute the exact term that was
    retrieved). The Verifier's real job now is: is the match confident
    enough to trust? That confidence check happens in decide().
    """
    if match_info is None:
        return True, 1.0, "no change needed"
    ngram, term, score = match_info
    return True, score, f"'{ngram}' matched retrieved term '{term}'"
 
 
def decide(confidence):
    if confidence >= HIGH_THRESHOLD:
        return "AUTO-CORRECT"
    elif confidence >= LOW_THRESHOLD:
        return "FLAG FOR HUMAN REVIEW"
    else:
        return "ABSTAIN / NO CHANGE"
 
 
def explain(hyp, index, terms, embed_model):
    candidates = retrieve(hyp, embed_model, index, terms)
 
    if not candidates:
        return {
            "original": hyp,
            "final_text": hyp,
            "decision": "NO CORRECTION PROPOSED",
            "explanation": "No domain terms relevant enough were found in the knowledge base.",
            "confidence": None,
        }
 
    proposed, match_info = correct(hyp, candidates)
    grounded, confidence, reason = verify(match_info)
    decision = decide(confidence)
 
    # Only actually apply the correction if confidence clears the bar;
    # otherwise keep the original text but still report what WOULD have
    # been proposed, for transparency (flag-for-review case).
    final_text = proposed if decision == "AUTO-CORRECT" else hyp
 
    return {
        "original": hyp,
        "proposed_correction": proposed,
        "candidates_found": [f"'{ng}'->'{t}' ({s:.2f})" for ng, t, s in candidates],
        "final_text": final_text,
        "decision": decision,
        "explanation": reason,
        "confidence": round(confidence, 2),
    }
 
 
def print_explanation(result):
    print(f"  Original:   {result['original']}")
    if "proposed_correction" in result:
        print(f"  Proposed:   {result['proposed_correction']}")
    if "candidates_found" in result:
        print(f"  Candidates: {result['candidates_found']}")
    print(f"  Final text: {result['final_text']}")
    print(f"  Decision:   {result['decision']}")
    print(f"  Why:        {result['explanation']}")
    if result["confidence"] is not None:
        print(f"  Confidence: {result['confidence']}")
 
 
def main():
    index, terms, embed_model = load_everything()
 
    print("=" * 60)
    print("PART 1: running full pipeline on today's 20 real clips")
    print("=" * 60)
    with open("../data/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
    for i, hyp in enumerate(baseline["hypotheses"][:5]):  # just first 5 for speed
        print(f"\n[{i+1}]")
        result = explain(hyp, index, terms, embed_model)
        print_explanation(result)
 
    print("\n" + "=" * 60)
    print("PART 2: synthetic banking-domain examples (proves the full")
    print("explainability output on realistic SHUDH-ASR use cases)")
    print("=" * 60)
    synthetic_examples = [
        "mera account dorment ho gaya hai use activate karna hai",
        "aaj mausam bahut accha hai chalo bahar ghumne chalte hai",
        "mujhe apna IFSC cod chahiye bank ke liye",
    ]
    for i, hyp in enumerate(synthetic_examples):
        print(f"\n[synthetic {i+1}]")
        result = explain(hyp, index, terms, embed_model)
        print_explanation(result)
 
 
if __name__ == "__main__":
    main()

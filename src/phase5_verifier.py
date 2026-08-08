import pickle
from jiwer import wer
 
def is_grounded(original_hyp, corrected_hyp, candidates):
    """
    A correction is 'grounded' if:
      (a) nothing changed (safe, always accepted), OR
      (b) something changed AND at least one retrieved candidate term
          actually appears in the corrected text (proof the correction
          came from the KB, not from the model's imagination).
    """
    if corrected_hyp.strip() == original_hyp.strip():
        return True, "no change made"
    if not candidates:
        # Correction happened but nothing was even retrieved -> definitely
        # not grounded. This should be rare given Phase 4's threshold, but
        # we check anyway as a second line of defense.
        return False, "no candidates were retrieved, but text changed anyway"
    for term in candidates:
        if term.lower() in corrected_hyp.lower():
            return True, f"grounded in retrieved term: '{term}'"
    return False, "text changed but no retrieved candidate appears in it -- likely hallucinated"
 
def run_on_real_data():
    with open("../data/phase4_results.pkl", "rb") as f:
        phase4 = pickle.load(f)
    with open("../data/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
 
    # Phase 4 didn't save per-example candidates, so for the real-data pass
    # we only have original vs corrected text -- good enough to confirm
    # "no change was made everywhere", matching what you saw in Phase 4.
    final = []
    accepted, rejected = 0, 0
    for i, (orig, corrected) in enumerate(zip(baseline["hypotheses"], phase4["hypotheses"])):
        if orig.strip() == corrected.strip():
            final.append(corrected)
            accepted += 1
        else:
            # changed but we have no stored candidate list here -> be safe,
            # revert unless we can confirm groundedness some other way
            final.append(orig)
            rejected += 1
            print(f"[{i+1}] Correction made without stored evidence -> reverted to original (safety default).")
 
    print(f"\nReal-data pass: {accepted} unchanged/accepted, {rejected} reverted.")
    new_wer = wer(baseline["references"], final)
    print(f"WER after verifier pass on real data: {new_wer*100:.2f}%  "
          f"(should equal Phase 4's 68.00% since nothing was actually corrected today)")
 
def run_synthetic_demo():
    """
    Proves the verifier logic itself works, using made-up examples that
    mimic what a REAL banking-domain run would produce.
    """
    print("\n" + "=" * 50)
    print("SYNTHETIC DEMO: proving the verifier catches hallucinations")
    print("=" * 50)
 
    cases = [
        {
            "original": "mera account dorment ho gaya hai",
            "corrected": "mera account dormant account ho gaya hai",
            "candidates": ["dormant account", "KYC", "IFSC code"],
            "label": "GOOD correction, grounded in retrieval",
        },
        {
            "original": "mera IFSC cod kya hai",
            "corrected": "aapka bank balance zero hai aur account block ho gaya hai",
            "candidates": ["IFSC code", "KYC", "account activation"],
            "label": "HALLUCINATED correction, invented content not grounded in retrieval",
        },
    ]
 
    for case in cases:
        grounded, reason = is_grounded(case["original"], case["corrected"], case["candidates"])
        decision = "ACCEPTED" if grounded else "REJECTED (reverted to original)"
        print(f"\n{case['label']}")
        print(f"  Original:  {case['original']}")
        print(f"  Corrected: {case['corrected']}")
        print(f"  Candidates: {case['candidates']}")
        print(f"  Verifier decision: {decision}  ({reason})")
 
if __name__ == "__main__":
    run_on_real_data()
    run_synthetic_demo()
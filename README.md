# SHUDH-ASR

Retrieval-grounded, agentic, confidence-calibrated post-ASR error correction
for code-switched Indian languages, with a domain focus on banking/PMJDY
terminology.

## Project status: all 8 phases complete

| Phase | Script | Status | Key result |
|---|---|---|---|
| 1. ASR baseline | `src/run_baseline.py` | ✅ Done | Whisper-small on FLEURS Hindi: WER 68.00% |
| 2. Naive LLM correction (no retrieval) | `src/phase2_naive_correction.py` | ✅ Done | WER rose to 98.10% — confirms the documented over-correction/hallucination failure mode |
| 3. Domain knowledge base + retrieval | `src/phase3_build_kb.py` | ✅ Done | FAISS + multilingual embeddings, English + Devanagari terms, n-gram span matching |
| 4. RAG-grounded correction | `src/phase4_rag_correction.py` | ✅ Done | Deterministic span substitution (not free LLM generation — found to be more reliable) |
| 5. Verifier agent | `src/phase5_verifier.py` | ✅ Done | Rejects corrections not grounded in retrieval evidence |
| 6. Confidence calibration + abstention | `src/phase6_calibration.py` | ✅ Done | 3-tier decision: auto-correct / flag for review / abstain |
| 7. Full pipeline + explainability | `src/phase7_explainability.py` | ✅ Done | End-to-end pipeline with human-readable correction rationale |
| 8. Domain-matched evaluation | `src/generate_domain_testset.py`, `src/run_domain_baseline.py` | ✅ Done | 15 TTS-synthesized banking/PMJDY sentences; see results below |

## Final results

**General-domain (FLEURS Hindi):**
- Baseline ASR: WER 68.00%
- + naive LLM correction: WER 98.10% (worse — confirms failure mode)
- + SHUDH-ASR (conservative threshold): WER 68.00% (unchanged — correctly abstains, no banking content present)

**Domain-matched (banking/PMJDY, TTS-synthesized):**
- Baseline ASR: WER 51.28%
- + SHUDH-ASR (conservative threshold, 0.8): WER 51.28% (unchanged — safe, but low recall)
- + SHUDH-ASR (permissive threshold, 0.45): WER 61.54% (worse — over-correction risk, motivates precision-first design)

**Key finding:** a precision-first (conservative threshold) design avoids introducing harm, at the cost of coverage. This motivates the paper's central claim and points to phonetic/transliteration-aware retrieval as necessary future work, since pure semantic-embedding retrieval is an unreliable bridge between Devanagari-script ASR errors and correct domain terms.

## Setup (if starting fresh on a new machine)

```powershell
cd shudh-asr
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the results, in order

```powershell
cd src
python download_data.py              # Phase 1 data
python run_baseline.py                # Phase 1
python phase2_naive_correction.py     # Phase 2
python phase3_build_kb.py             # Phase 3
python phase4_rag_correction.py       # Phase 4
python phase5_verifier.py             # Phase 5
python phase6_calibration.py          # Phase 6
python phase7_explainability.py       # Phase 7
python generate_domain_testset.py     # Phase 8 data (needs: pip install gTTS)
python run_domain_baseline.py         # Phase 8 final results
```

## Known limitations / honest follow-up work

1. **Corrector is currently deterministic span-substitution, not the LoRA-fine-tuned generative model** originally proposed. This was a deliberate simplification after finding that a small (1.5B) LLM's free-text regeneration was unreliable and prone to hallucination. LoRA fine-tuning on Colab/Kaggle GPU remains planned future work.
2. **Retrieval uses general-purpose semantic embeddings only.** Real testing showed this is an imperfect bridge between phonetic ASR errors (Devanagari script) and correct domain terms — a hybrid phonetic/transliteration-aware retrieval approach is the recommended next step.
3. **Domain test set is TTS-synthesized (15 sentences), not recorded real banking-call audio.** A larger, recorded, or the full MUCS 2021 code-switching benchmark would strengthen the results further.
4. **Confidence signal is currently the retrieval similarity score itself**, not real per-token ASR decoder confidence with a formally fitted calibration curve (temperature/Platt scaling) — noted as future work in the paper.

## Paper

An IEEE-format draft (`SHUDH-ASR_IEEE_Paper.docx`) is in progress, using the real results above.

## Repo

Pushed to: https://github.com/Garidepallisahithi/shudh-asr

## If something breaks

Paste the exact error message back to Claude — don't guess at fixes.
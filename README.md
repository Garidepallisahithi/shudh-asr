# SHUDH-ASR

**Retrieval-grounded, agentic post-ASR error correction for code-switched Indian languages.**

SHUDH-ASR is a correction layer for Automatic Speech Recognition (ASR) systems that targets a specific, unsolved problem in Indian-language voice interfaces: transcription errors in code-switched speech (Hindi-English) and domain-specific vocabulary (banking terms, government-scheme names) that general-purpose ASR models consistently get wrong — and that a naive "fix it with an LLM" approach makes *worse*, not better.

Rather than asking a language model to freely regenerate a transcript, SHUDH-ASR retrieves the correct domain term from a knowledge base, verifies the match is genuinely grounded (not hallucinated), and only applies a correction when it is calibrated-confident enough to trust — otherwise it abstains and flags the transcript for human review.

## Why this matters

Voice interfaces are often the only practical way for print-illiterate populations to access digital banking and government welfare schemes (e.g. PMJDY) in India. A mistranscribed account type or scheme name in this setting isn't a cosmetic error — it can misdirect an entire transaction. Existing Indic ASR systems leave residual errors concentrated in exactly these high-stakes terms, and the common fix (passing ASR output through a general LLM) is a documented failure mode that increases errors through over-correction and hallucination.

## Architecture

```
Audio → ASR (Whisper) → Domain Retriever (FAISS) → Corrector (span substitution)
      → Verifier + Confidence Calibration → Corrected Transcript + Explanation
```

Each ASR hypothesis is broken into overlapping word-level spans, compared against a bilingual (English + Devanagari) domain knowledge base using multilingual sentence embeddings, and the best match above a similarity threshold is substituted directly — never freely regenerated. A calibrated confidence score then decides whether to auto-correct, flag for human review, or leave the transcript untouched.

## Results

**General-domain speech (FLEURS, Hindi):**

| Configuration | WER |
|---|---|
| ASR baseline | 68.00% |
| + naive LLM correction (no retrieval) | 98.10% |
| + SHUDH-ASR | 68.00% |

**Domain-matched speech (banking/PMJDY):**

| Configuration | WER |
|---|---|
| ASR baseline | 51.28% |
| + SHUDH-ASR (conservative threshold) | 51.28% |
| + SHUDH-ASR (permissive threshold) | 61.54% |

Naive, ungrounded LLM correction raises error rate on general speech by nearly 30 points, directly confirming a failure mode reported in prior literature. A retrieval-grounded corrector reproduces the same risk when its similarity threshold is too permissive, while a conservative threshold avoids introducing errors — motivating the precision-first design used here.

## Tech Stack

- **ASR:** OpenAI Whisper (swappable for AI4Bharat IndicWhisper)
- **Retrieval:** FAISS, multilingual sentence embeddings (paraphrase-multilingual-MiniLM)
- **Correction:** Deterministic retrieval-grounded span substitution
- **Evaluation:** `jiwer` (WER/CER), gTTS (domain test-set synthesis)
- **Language:** Python

## Project Structure

```
shudh-asr/
├── src/
│   ├── run_baseline.py              # ASR baseline + WER
│   ├── phase2_naive_correction.py   # Naive LLM correction baseline
│   ├── phase3_build_kb.py           # Domain knowledge base + retrieval
│   ├── phase4_rag_correction.py     # Retrieval-grounded correction
│   ├── phase5_verifier.py           # Groundedness verification
│   ├── phase6_calibration.py        # Confidence calibration + abstention
│   ├── phase7_explainability.py     # Full pipeline with explanations
│   ├── generate_domain_testset.py   # Domain-matched test set (TTS)
│   └── run_domain_baseline.py       # Domain evaluation
├── kb/                               # Domain knowledge base (FAISS index)
├── data/                             # Test data (gitignored)
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Usage

```bash
cd src
python download_data.py && python run_baseline.py       # Baseline
python phase3_build_kb.py                                 # Build KB
python phase7_explainability.py                            # Full pipeline
python generate_domain_testset.py && python run_domain_baseline.py  # Domain eval
```

## Limitations & Future Work

- The corrector currently performs deterministic span substitution rather than a fine-tuned generative model; LoRA fine-tuning of an open LLM on GPU infrastructure is planned follow-up work.
- Retrieval relies on general-purpose semantic embeddings, which are an imperfect bridge between phonetic ASR errors (Devanagari script) and correct domain terms; phonetic/transliteration-aware retrieval is a recommended next step.
- Domain evaluation uses a 15-sentence TTS-synthesized test set; evaluation on the full MUCS 2021 code-switching benchmark and recorded real-world audio is planned.

## Authors

Garedepalli Sahithi
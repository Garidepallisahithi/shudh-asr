import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
 
# Each entry: (text used for embedding -- rich/descriptive, text used for
# the actual correction/display -- short form users expect)
DOMAIN_TERMS_RICH = [
    ("PMJDY Pradhan Mantri Jan Dhan Yojana bank scheme", "PMJDY"),
    ("Pradhan Mantri Jan Dhan Yojana government scheme", "Pradhan Mantri Jan Dhan Yojana"),
    ("dormant account bank inactive", "dormant account"),
    ("savings account bank", "savings account"),
    ("current account bank", "current account"),
    ("KYC Know Your Customer verification", "KYC"),
    ("Know Your Customer bank verification process", "Know Your Customer"),
    ("overdraft facility bank credit", "overdraft facility"),
    ("RuPay card bank debit card", "RuPay card"),
    ("Aadhaar seeding bank linking", "Aadhaar seeding"),
    ("account activation bank reactivate", "account activation"),
    ("zero balance account bank", "zero balance account"),
    ("beneficiary bank payment recipient", "beneficiary"),
    ("direct benefit transfer DBT government payment", "direct benefit transfer"),
    ("DBT direct benefit transfer scheme", "DBT"),
    ("mini statement bank transaction history", "mini statement"),
    ("passbook bank account record", "passbook"),
    ("IFSC code bank branch identifier", "IFSC code"),
    ("NEFT National Electronic Funds Transfer bank payment", "NEFT"),
    ("RTGS Real Time Gross Settlement bank payment", "RTGS"),
    ("IMPS Immediate Payment Service bank transfer", "IMPS"),
    ("UPI Unified Payments Interface bank payment app", "UPI"),
    ("PIN reset bank card security", "PIN reset"),
    ("account freeze bank blocked", "account freeze"),
    ("nominee bank account beneficiary", "nominee"),
    ("fixed deposit bank savings investment", "fixed deposit"),
    ("recurring deposit bank monthly savings", "recurring deposit"),
    ("loan against deposit bank credit", "loan against deposit"),
    ("PMJDY insurance cover accidental scheme", "insurance cover PMJDY"),
    ("accidental insurance cover PMJDY scheme benefit", "accidental insurance cover"),
    # --- Devanagari versions (real Hindi ASR output is in this script) ---
    # Same enrichment strategy: rich descriptive Devanagari text for the
    # embedding, correct short Devanagari spelling for the actual output.
    ("डॉर्मेंट खाता बैंक निष्क्रिय", "डॉर्मेंट"),
    ("आईएफएससी कोड बैंक शाखा पहचान संख्या", "आईएफएससी कोड"),
    ("प्रधानमंत्री जन धन योजना सरकारी बैंक योजना", "प्रधानमंत्री जन धन योजना"),
    ("केवाईसी दस्तावेज पहचान सत्यापन बैंक", "केवाईसी"),
    ("एनईएफटी पैसे भेजना बैंक भुगतान ट्रांसफर", "एनईएफटी"),
    ("जीरो बैलेंस खाता बैंक शून्य शेष", "जीरो बैलेंस"),
    ("रूपे कार्ड बैंक डेबिट कार्ड", "रूपे कार्ड"),
    ("मिनी स्टेटमेंट बैंक लेनदेन इतिहास विवरण", "मिनी स्टेटमेंट"),
    ("डायरेक्ट बेनिफिट ट्रांसफर सरकारी बैंक भुगतान", "डायरेक्ट बेनिफिट ट्रांसफर"),
    ("पिन रीसेट बैंक कार्ड सुरक्षा कोड बदलना", "पिन रीसेट"),
    ("ओवरड्राफ्ट सुविधा बैंक क्रेडिट अतिरिक्त राशि", "ओवरड्राफ्ट"),
    ("फिक्स्ड डिपॉजिट बैंक बचत निवेश सावधि जमा", "फिक्स्ड डिपॉजिट"),
    ("नॉमिनी बैंक खाता लाभार्थी उत्तराधिकारी", "नॉमिनी"),
    ("अकाउंट फ्रीज बैंक खाता अवरुद्ध बंद", "अकाउंट फ्रीज"),
    ("पासबुक बैंक खाता रिकॉर्ड डायरी", "पासबुक"),
    ("खाता सक्रिय करना बैंक चालू", "सक्रिय"),
]
 
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
 
def build_index():
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
 
    rich_texts = [rich for rich, display in DOMAIN_TERMS_RICH]
    display_terms = [display for rich, display in DOMAIN_TERMS_RICH]
 
    embeddings = model.encode(rich_texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
 
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
 
    faiss.write_index(index, "../kb/domain_index.faiss")
    # Save the DISPLAY terms (short form) -- this is what actually gets
    # substituted into corrected text, not the rich embedding text.
    with open("../kb/domain_terms.pkl", "wb") as f:
        pickle.dump(display_terms, f)
    print(f"Indexed {len(display_terms)} domain terms (with enriched embeddings) -> kb/domain_index.faiss")
    return model, index, display_terms
 
def test_retrieval(model, index, display_terms, query, k=3):
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    return [(display_terms[i], float(s)) for i, s in zip(idxs[0], scores[0])]
 
if __name__ == "__main__":
    model, index, display_terms = build_index()
 
    print("\n" + "=" * 50)
    print("PHASE 3 RESULTS: retrieval sanity check")
    print("=" * 50)
    test_queries = [
        "my account has become dorment",
        "I need to activate my jan dhan yojna",
        "what is my IFSC cod",
        "aaj mausam bahut accha hai",  # unrelated -- should score low now
    ]
    for q in test_queries:
        results = test_retrieval(model, index, display_terms, q)
        print(f"\nQuery: '{q}'")
        for term, score in results:
            print(f"   -> {term}   (similarity: {score:.3f})")
 
    print("\nExpected: banking queries should still match correctly, and the")
    print("weather query should now score noticeably LOWER than before,")
    print("since acronym embeddings are enriched with real context.")
 





























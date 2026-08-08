import os
import pickle
from gtts import gTTS
 
DATA_DIR = "../data/domain"
AUDIO_DIR = "../data/domain/audio"
 
# 15 realistic banking/PMJDY sentences -- covers most of the KB terms
# from phase3_build_kb.py, so retrieval has real material to work with.
SENTENCES = [
    "मेरा खाता डॉर्मेंट हो गया है इसे सक्रिय करना है",
    "मुझे अपना आईएफएससी कोड चाहिए",
    "प्रधानमंत्री जन धन योजना के तहत मेरा खाता खुला है",
    "केवाईसी दस्तावेज जमा करने हैं",
    "मुझे एनईएफटी से पैसे भेजने हैं",
    "मेरे खाते में जीरो बैलेंस है",
    "रूपे कार्ड कैसे बनवाएं",
    "मिनी स्टेटमेंट चाहिए",
    "डायरेक्ट बेनिफिट ट्रांसफर कब आएगा",
    "मेरा पिन रीसेट करना है",
    "ओवरड्राफ्ट सुविधा कैसे मिलेगी",
    "फिक्स्ड डिपॉजिट खोलना है",
    "नॉमिनी कैसे जोड़ें",
    "अकाउंट फ्रीज क्यों हुआ",
    "पासबुक अपडेट करानी है",
]
 
# NOTE: The knowledge base (phase3_build_kb.py) must contain the DEVANAGARI
# spelling of these same terms (e.g. "आईएफएससी कोड", not just "IFSC code"),
# since real Hindi ASR output is in Devanagari script. Comparing Devanagari
# ASR errors against English-only KB terms was found to produce zero
# matches -- this is the actual fix, see phase3_build_kb.py.
 
 
def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    metadata = []
 
    print(f"Generating {len(SENTENCES)} domain audio clips via gTTS...")
    for i, sentence in enumerate(SENTENCES):
        filename = f"domain_{i:02d}.mp3"
        path = f"{AUDIO_DIR}/{filename}"
        tts = gTTS(text=sentence, lang="hi")
        tts.save(path)
        metadata.append({"file": filename, "reference_text": sentence})
        print(f"[{i+1}/{len(SENTENCES)}] Saved: {sentence}")
 
    with open(f"{DATA_DIR}/domain_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
 
    print(f"\nSaved {len(metadata)} domain clips + metadata to {DATA_DIR}/")
    print("Next: run run_domain_baseline.py")
 
 
if __name__ == "__main__":
    main()

"""
test_arabic_rule.py
Quick unit test for the new Arabic-script disambiguation rule.
Tests the DialectDetector + LanguageRouter path for Urdu/Nastaliq text.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from language_module import DialectDetector, LanguageRouter

dd = DialectDetector()
router = LanguageRouter(confidence_threshold=0.60)

CASES = [
    # (description, transcript, whisper_lang, whisper_prob, ft_lang, ft_conf, mms_lang, mms_conf, expected_lang)

    # Typical Urdu failure: Whisper says 'hi', FastText 'unknown', MMS says 'ur'
    ("Urdu: Whisper=hi ft=unknown mms=ur",
     "میں نے اسے کہا کہ وہ یہاں آئے",      # Nastaliq Urdu
     "hi", 0.55, "unknown", 0.0, "ur", 0.72, "ur"),

    # Urdu: all three say unknown except MMS
    ("Urdu: Whisper=en ft=unknown mms=ur",
     "اس نے مجھ سے پوچھا کہ تم کون ہو",
     "en", 0.40, "unknown", 0.0, "ur", 0.68, "ur"),

    # Urdu: no Arabic-script source at all — should still default to ur via script
    ("Urdu: all sources wrong/unknown",
     "وہ فوجی دستہ شمال کی طرف بڑھ رہا ہے",
     "hi", 0.30, "unknown", 0.0, "hi", 0.35, "ur"),

    # Pashto (should still work via existing rule)
    ("Pashto: ft=ps mms=ps",
     "دا زما کور دی",                       # Pashto Nastaliq
     "ps", 0.60, "ps", 0.80, "ps", 0.75, "ps"),

    # Urdu with MMS=ur and FastText correctly says ur
    ("Urdu: ft=ur mms=ur (ideal case)",
     "وزیراعظم نے اعلان کیا",
     "hi", 0.55, "ur", 0.65, "ur", 0.78, "ur"),

    # Hindi (Devanagari) — should NOT trigger Arabic rule
    ("Hindi: Devanagari no interference",
     "मैंने उसे कहा था",
     "hi", 0.88, "hi", 0.90, "hi", 0.85, "hi"),

    # Punjabi (Gurmukhi) — should NOT trigger Arabic rule
    ("Punjabi: Gurmukhi no interference",
     "ਮੈਂ ਤੁਹਾਨੂੰ ਦੱਸਣਾ ਚਾਹੁੰਦਾ ਹਾਂ",
     "hi", 0.70, "pa", 0.75, "pa", 0.80, "pa"),
]

print(f"\n{'='*75}")
print(f"{'Description':<42} {'Dialect':<20} {'Got':>4} {'Exp':>4} {'Pass':>5}")
print(f"{'-'*75}")

passed = 0
for desc, transcript, wl, wp, fl, fc, ml, mc in [(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]) for c in CASES]:
    expected = CASES[[c[0] for c in CASES].index(desc)][8]
    dial = dd.detect_code_mix(transcript)
    d = dial["dialect"]
    result = router.detect_family(
        whisper_lang=wl, transcript=transcript,
        fasttext_lang=fl, fasttext_conf=fc,
        whisper_lang_prob=wp,
        dialect=d,
        mms_lang=ml, mms_conf=mc,
    )
    got = result["final_language"]
    ok = "PASS" if got == expected else "FAIL"
    if got == expected:
        passed += 1
    print(f"  {desc:<40} {d:<20} {got:>4} {expected:>4} {ok:>5}  [{result['vote_note']}]")

print(f"{'='*75}")
print(f"  {passed}/{len(CASES)} passed\n")

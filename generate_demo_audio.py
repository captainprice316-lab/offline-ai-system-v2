#!/usr/bin/env python3
"""Generate Hindi/Urdu demo intercept clips via edge-tts.

Each clip is scripted to exercise a threat level + a map location, mixing plain
keywords and coded terminology. Output: 16 kHz mono WAV in demo_audio/ plus a
manifest.json documenting expected behaviour for the demo operator.

Requires internet (edge-tts synthesis). WAVs then play fully offline.
"""
import asyncio, json, subprocess, tempfile, os
from pathlib import Path
import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path("demo_audio"); OUT.mkdir(exist_ok=True)

HI_M, HI_F = "hi-IN-MadhurNeural", "hi-IN-SwaraNeural"
UR_M, UR_F = "ur-PK-AsadNeural",   "ur-PK-UzmaNeural"
NE_M, NE_F = "ne-NP-SagarNeural",  "ne-NP-HemkalaNeural"
# Punjabi has no edge-tts voice — the pa showcase clips are real FLEURS eval
# speech copied in (they also best demonstrate the SeamlessM4T ASR quality win).
PA_EVAL_COPIES = [
    ("17_pa_quality_showcase.wav", "eval_audio/Punjabi_0000.wav",
     "REAL speech — SeamlessM4T ASR quality (was garbled on Whisper)"),
    ("18_pa_quality_showcase.wav", "eval_audio/Punjabi_0015.wav",
     "REAL speech — SeamlessM4T ASR quality"),
]

# id, voice, native text, expected threat, location, gloss (for operator)
CLIPS = [
    ("01_hi_critical_srinagar", HI_M,
     "श्रीनगर में दुश्मन पर हमला करो, बम तैयार है।",
     "CRITICAL", "Srinagar", "enemy + attack + bomb"),
    ("02_hi_coded_anantnag", HI_M,
     "अनंतनाग में बीस आलू और मिर्ची भेज दो, मेहमान तैयार हैं।",
     "HIGH", "Anantnag", "CODED: aloo=grenades, mirchi=bullets, mehmaan=infiltrators"),
    ("03_hi_high_pulwama", HI_F,
     "पुलवामा के उत्तर में गोला बारूद और बंदूक पहुँचाओ।",
     "HIGH", "Pulwama", "weapons + location (north)"),
    ("04_hi_high_anantnag", HI_M,
     "अनंतनाग से पीछे हटो, दो जवान शहीद हो गए।",
     "HIGH", "Anantnag", "movement (retreat) + casualties"),
    ("05_hi_medium_kupwara", HI_F,
     "कुपवाड़ा में उनकी चौकी पर नज़र रखो।",
     "MEDIUM", "Kupwara", "surveillance"),
    ("06_hi_medium_jammu", HI_M,
     "जम्मू में रसद और पानी का इंतज़ाम करो।",
     "MEDIUM", "Jammu", "logistics (supply + water)"),
    ("07_hi_low_comms", HI_M,
     "रेडियो चैनल बदलो और कॉलसाइन दोहराओ।",
     "LOW", "-", "comms (radio, channel, callsign)"),
    ("08_hi_clear", HI_F,
     "आज मौसम बहुत अच्छा है, शाम को चाय पीते हैं।",
     "CLEAR", "-", "benign chatter"),
    ("09_ur_critical_muzaffarabad", UR_M,
     "مظفرآباد میں دشمن پر حملہ کرو، دھماکہ خیز مواد تیار ہے۔",
     "CRITICAL", "Muzaffarabad", "enemy + attack + explosives"),
    ("10_ur_coded_baramulla", UR_M,
     "ڈاکٹر اور سامان بارہمولہ بھیج دو، مہمان آ رہے ہیں۔",
     "HIGH", "Baramulla", "CODED: doctor=IED maker, saman=weapons, mehmaan=infiltrators"),
    ("11_ur_high_baramulla", UR_F,
     "بارہمولہ کے مغرب میں تیاری کرو، سب اکٹھے ہو جاؤ۔",
     "HIGH", "Baramulla", "pre_attack (prepare, assemble) + location (west)"),
    ("12_ur_medium_command", UR_M,
     "حکم کا انتظار کرو، رابطہ برقرار رکھو۔",
     "MEDIUM", "-", "command (orders) + comms"),
    ("13_ur_clear", UR_F,
     "سب خیریت ہے، صبح بات کریں گے۔",
     "CLEAR", "-", "benign chatter"),
    # ── Nepali (routes to SeamlessM4T ASR) ──
    ("14_ne_critical_srinagar", NE_M,
     "श्रीनगरमा शत्रुमाथि आक्रमण गर, बम तयार छ।",
     "CRITICAL", "Srinagar", "enemy + attack + bomb"),
    ("15_ne_high_pulwama", NE_F,
     "पुलवामाको उत्तरमा हतियार र गोली पठाऊ।",
     "HIGH", "Pulwama", "weapons + location"),
    ("16_ne_medium_jammu", NE_M,
     "जम्मूमा खाना र पानीको प्रबन्ध गर।",
     "MEDIUM", "Jammu", "logistics (food + water)"),
]

async def synth(text, voice, mp3_path):
    await edge_tts.Communicate(text, voice, rate="-8%").save(mp3_path)

def to_wav16k(mp3_path, wav_path):
    subprocess.run([FFMPEG, "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1",
                    "-loglevel", "error", wav_path], check=True)

async def main():
    manifest = []
    for cid, voice, text, level, loc, gloss in CLIPS:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3 = f.name
        wav = str(OUT / f"{cid}.wav")
        await synth(text, voice, mp3)
        to_wav16k(mp3, wav)
        os.unlink(mp3)
        dur = subprocess.run([FFMPEG, "-i", wav, "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        print(f"  {cid:32s} [{level:8s}] {loc}")
        manifest.append({"file": f"{cid}.wav", "voice": voice, "text": text,
                         "expected_threat": level, "location": loc, "notes": gloss})
    # Punjabi showcase clips: copy real eval speech (no pa edge-tts voice)
    import shutil
    for dst, src, notes in PA_EVAL_COPIES:
        if Path(src).exists():
            shutil.copy(src, OUT / dst)
            manifest.append({"file": dst, "voice": "real-human (FLEURS)",
                             "text": "(real Punjabi speech)", "expected_threat": "n/a",
                             "location": "-", "notes": notes})
            print(f"  {dst:32s} [pa qual ] copied from {src}")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGenerated {len(manifest)} clips -> {OUT}/  (+ manifest.json)")

asyncio.run(main())

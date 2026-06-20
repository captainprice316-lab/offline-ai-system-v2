"""
download_papers.py — Download all 10 VANI literature review papers from arXiv
Saves PDFs to literature_papers/
"""
import os
import time
import urllib.request

OUTPUT_DIR = "literature_papers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAPERS = [
    {
        "id":       "P1",
        "arxiv":    "2212.04356",
        "filename": "P1_Whisper_Radford_2022.pdf",
        "title":    "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)",
    },
    {
        "id":       "P2",
        "arxiv":    "2207.04672",
        "filename": "P2_NLLB_MetaAI_2022.pdf",
        "title":    "No Language Left Behind: Scaling Human-Centered MT (NLLB-200)",
    },
    {
        "id":       "P3",
        "arxiv":    "2305.16307",
        "filename": "P3_IndicTrans2_AI4Bharat_2023.pdf",
        "title":    "IndicTrans2: Towards HQ MT for all 22 Scheduled Indian Languages",
    },
    {
        "id":       "P4",
        "arxiv":    "2305.13516",
        "filename": "P4_MMS_MetaAI_2023.pdf",
        "title":    "Scaling Speech Technology to 1,000+ Languages (MMS)",
    },
    {
        "id":       "P5",
        "arxiv":    "1607.01759",
        "filename": "P5_FastText_Joulin_2017.pdf",
        "title":    "Bag of Tricks for Efficient Text Classification (FastText)",
    },
    {
        "id":       "P6",
        "arxiv":    "1706.03762",
        "filename": "P6_AttentionIsAllYouNeed_Vaswani_2017.pdf",
        "title":    "Attention Is All You Need (Transformer)",
    },
    {
        "id":       "P7",
        "arxiv":    "2111.03945",
        "filename": "P7_IndicWav2Vec_AI4Bharat_2022.pdf",
        "title":    "IndicWav2Vec: A Multilingual Speech Model for Indian Languages",
    },
    {
        "id":       "P8",
        "arxiv":    "1911.02116",
        "filename": "P8_XLM_RoBERTa_Conneau_2020.pdf",
        "title":    "Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)",
    },
    {
        "id":       "P9",
        "arxiv":    "1912.08777",
        "filename": "P9_PEGASUS_Zhang_2020.pdf",
        "title":    "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization",
    },
    {
        "id":       "P10",
        "arxiv":    "2001.01980",
        "filename": "P10_pyannote_Bredin_2020.pdf",
        "title":    "pyannote.audio: neural building blocks for speaker diarization",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def download(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)

total_ok = 0
for p in PAPERS:
    dest = os.path.join(OUTPUT_DIR, p["filename"])
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        print(f"[SKIP] {p['id']} already downloaded - {p['filename']}")
        total_ok += 1
        continue

    url = f"https://arxiv.org/pdf/{p['arxiv']}.pdf"
    print(f"[DOWNLOAD] {p['id']} {p['title'][:60]}")
    print(f"           {url}")
    try:
        size = download(url, dest)
        print(f"           OK  {size/1024:.0f} KB  ->  {dest}")
        total_ok += 1
    except Exception as e:
        print(f"           FAILED: {e}")
        # Try alternate arXiv URL format (no .pdf extension)
        try:
            url2 = f"https://arxiv.org/pdf/{p['arxiv']}"
            size = download(url2, dest)
            print(f"           OK  (alt URL) {size/1024:.0f} KB  ->  {dest}")
            total_ok += 1
        except Exception as e2:
            print(f"           Alt also failed: {e2}")
    time.sleep(2)   # polite delay between requests

print(f"\nDone: {total_ok}/{len(PAPERS)} papers downloaded to ./{OUTPUT_DIR}/")

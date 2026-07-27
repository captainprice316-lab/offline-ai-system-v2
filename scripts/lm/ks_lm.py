# -*- coding: utf-8 -*-
"""ks_lm.py — interpolated Kneser-Ney trigram LM for Kashmiri n-best rescoring.

Why this exists: after the vocabulary repair (ks_cloud3, 50.26% L2 WER), 76% of
the remaining errors are word SUBSTITUTIONS — the error class an acoustic model
cannot fix but a language model can. The training corpus already carries 84,662
unique Kashmiri sentences / 2.33 M word tokens, which is enough for a trigram.

The corpus was deduplicated against the 372-clip IndicVoices-R test set during
prep (cloud/prep_ks_data.py applies a 403-sentence blocklist), so an LM trained
on it does not leak the evaluation references. train() re-checks this and warns.

Implementation is interpolated Kneser-Ney (Chen & Goodman) with a single
discount per order:

    P_KN(w|u,v) = max(c(u,v,w) - D, 0)/c(u,v) + D*N1+(u,v,.)/c(u,v) * P_KN(w|v)

with the lower orders using CONTINUATION counts rather than raw counts, which is
the part naive n-gram implementations get wrong.

Usage:
    python scripts/lm/ks_lm.py --train          # build + report held-out PPL
    from ks_lm import KNTrigramLM; lm = KNTrigramLM.load(path)
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

DEFAULT_OUT = ROOT / "models" / "ks_lm" / "ks_kn3.pkl"
BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class KNTrigramLM:
    def __init__(self, discount: float = 0.75):
        self.D = discount
        self.vocab: set[str] = set()
        # highest order: raw counts
        self.c3: Counter = Counter()          # (u,v,w)
        self.c2ctx: Counter = Counter()       # (u,v)      -> total
        self.n1_2ctx: Counter = Counter()     # (u,v)      -> #distinct w
        # middle order: continuation counts
        self.cont2: Counter = Counter()       # (v,w) -> #distinct u preceding
        self.cont1ctx: Counter = Counter()    # (v,)  -> sum of cont2 over w
        self.n1_1ctx: Counter = Counter()     # (v,)  -> #distinct w
        # lowest order: continuation counts
        self.cont1: Counter = Counter()       # w -> #distinct v preceding
        self.cont_total: int = 0

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, sentences: list[list[str]], min_count: int = 2):
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= min_count}
        self.vocab |= {BOS, EOS, UNK}

        def m(w):
            return w if w in self.vocab else UNK

        tri_seen, bi_seen = set(), set()
        for s in sentences:
            toks = [BOS, BOS] + [m(w) for w in s] + [EOS]
            for i in range(2, len(toks)):
                u, v, w = toks[i - 2], toks[i - 1], toks[i]
                self.c3[(u, v, w)] += 1
                self.c2ctx[(u, v)] += 1
                if (u, v, w) not in tri_seen:
                    tri_seen.add((u, v, w))
                    self.n1_2ctx[(u, v)] += 1
                    self.cont2[(v, w)] += 1          # distinct u preceding (v,w)
                if (v, w) not in bi_seen:
                    bi_seen.add((v, w))
                    self.cont1[w] += 1               # distinct v preceding w
                    self.n1_1ctx[(v,)] += 1
        for (v, w), c in self.cont2.items():
            self.cont1ctx[(v,)] += c
        self.cont_total = sum(self.cont1.values())
        return self

    # ── scoring ───────────────────────────────────────────────────────────────
    def _p_uni(self, w: str) -> float:
        # continuation probability, floored so OOV never yields -inf
        return max(self.cont1.get(w, 0), 0.1) / max(self.cont_total, 1)

    def _p_bi(self, v: str, w: str) -> float:
        denom = self.cont1ctx.get((v,), 0)
        if denom == 0:
            return self._p_uni(w)
        first = max(self.cont2.get((v, w), 0) - self.D, 0.0) / denom
        lam = self.D * self.n1_1ctx.get((v,), 0) / denom
        return first + lam * self._p_uni(w)

    def _p_tri(self, u: str, v: str, w: str) -> float:
        denom = self.c2ctx.get((u, v), 0)
        if denom == 0:
            return self._p_bi(v, w)
        first = max(self.c3.get((u, v, w), 0) - self.D, 0.0) / denom
        lam = self.D * self.n1_2ctx.get((u, v), 0) / denom
        return first + lam * self._p_bi(v, w)

    def logprob(self, words: list[str]) -> float:
        """Total log10 probability of a whitespace-tokenised sentence."""
        m = lambda w: w if w in self.vocab else UNK          # noqa: E731
        toks = [BOS, BOS] + [m(w) for w in words] + [EOS]
        lp = 0.0
        for i in range(2, len(toks)):
            p = self._p_tri(toks[i - 2], toks[i - 1], toks[i])
            lp += math.log10(max(p, 1e-12))
        return lp

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp = n = 0.0
        for s in sentences:
            lp += self.logprob(s)
            n += len(s) + 1                                   # +1 for </s>
        return 10 ** (-lp / max(n, 1))

    # ── persistence ───────────────────────────────────────────────────────────
    # Serialise plain data, never the instance: pickling `self` records the class
    # by import path, so a model saved while this file ran as __main__ fails to
    # load from any other module.
    _FIELDS = ("D", "vocab", "c3", "c2ctx", "n1_2ctx",
               "cont2", "cont1ctx", "n1_1ctx", "cont1", "cont_total")

    def save(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {k: getattr(self, k) for k in self._FIELDS}
        blob["__format__"] = "ks_kn3/1"
        with open(path, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: pathlib.Path) -> "KNTrigramLM":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if not isinstance(blob, dict) or blob.get("__format__") != "ks_kn3/1":
            raise ValueError(f"{path} is not a ks_kn3/1 model; retrain with --train")
        lm = KNTrigramLM(discount=blob["D"])
        for k in KNTrigramLM._FIELDS:
            setattr(lm, k, blob[k])
        return lm


# ── corpus + training entry point ─────────────────────────────────────────────

def load_corpus(exclude: set[str] | None = None):
    """L2-normalised training sentences (same ruler the WER is scored on).

    `exclude` drops sentences whose normalised form matches a test reference.
    This is NOT redundant with the corpus-prep blocklist: prep de-duplicated on
    RAW text, so two sentences differing only in diacritics survived it and then
    collide once L2 normalisation strips those diacritics. Measured on this
    corpus: 7 of the 372 test references leak in that way.
    """
    import pyarrow.parquet as pq
    from ks_ruler_study import norm

    man = pathlib.Path(r"E:\VANI\datasets\ks_combined\train_manifest.parquet")
    rows = pq.read_table(man, columns=["normalized"]).to_pydict()["normalized"]
    out, dropped = [], 0
    for s in rows:
        if not s or not s.strip():
            continue
        t = norm(s, 2)
        if exclude and t in exclude:
            dropped += 1
            continue
        out.append(t.split())
    if exclude:
        print(f"[leak-guard] dropped {dropped} training sentences matching a test reference")
    return out


def eval_reference_set():
    """The 372-clip test references, to assert the LM corpus does not leak."""
    from ks_ruler_study import norm, load_jsonl

    p = ROOT / "eval_data" / "ks_cloud3_seamless_hyps.jsonl"
    if not p.exists():
        return set()
    return {norm(r["ref"], 2) for r in load_jsonl(p)
            if r.get("set") == "indicvoices_test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--discount", type=float, default=0.75)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--heldout", type=int, default=3000)
    args = ap.parse_args()
    if not args.train:
        ap.error("nothing to do; pass --train")

    leak = eval_reference_set()
    if not leak:
        raise SystemExit("refusing to train: could not load the test references for the "
                         "leak guard (expected eval_data/ks_cloud3_seamless_hyps.jsonl)")
    sents = load_corpus(exclude=leak)
    print(f"[corpus] {len(sents):,} sentences, {sum(map(len, sents)):,} tokens")

    hits = sum(1 for s in sents if " ".join(s) in leak)
    print(f"[leak-check] test references remaining in LM corpus: {hits} "
          f"({'CLEAN' if hits == 0 else 'STILL CONTAMINATED'})")
    if hits:
        raise SystemExit("aborting: LM corpus is contaminated")

    held, train = sents[:args.heldout], sents[args.heldout:]
    lm = KNTrigramLM(discount=args.discount).fit(train, min_count=args.min_count)
    print(f"[lm] vocab {len(lm.vocab):,}  trigrams {len(lm.c3):,}  bigram-ctx {len(lm.c2ctx):,}")
    print(f"[lm] held-out perplexity ({len(held):,} sentences): {lm.perplexity(held):.1f}")

    out = pathlib.Path(args.out)
    lm.save(out)
    print(f"[saved] {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

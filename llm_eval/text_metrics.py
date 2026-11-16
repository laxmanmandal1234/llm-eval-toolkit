"""Reference-based metrics for generated text: EM, token F1, BLEU, ROUGE-L."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation/articles, squeeze spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: Sequence[str]) -> float:
    p = normalize(prediction)
    return float(any(p == normalize(r) for r in references))


def _f1(pred_toks: List[str], ref_toks: List[str]) -> float:
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    common = Counter(pred_toks) & Counter(ref_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Bag-of-tokens F1, maximised over references."""
    p = normalize(prediction).split()
    return max(_f1(p, normalize(r).split()) for r in references)


# ---------------------------------------------------------------- BLEU
def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions: Sequence[str], references: Sequence[Sequence[str]],
                max_n: int = 4, smooth: bool = False) -> float:
    """Corpus-level BLEU (Papineni et al., 2002).

    Clipped n-gram precisions are summed over the whole corpus, combined with
    a geometric mean, and multiplied by a brevity penalty. ``smooth`` applies
    add-one smoothing to n > 1 (Lin & Och, 2004), which is useful for short
    single sentences.
    """
    matches = [0] * max_n
    totals = [0] * max_n
    pred_len = ref_len = 0
    for pred, refs in zip(predictions, references):
        p = pred.lower().split()
        rs = [r.lower().split() for r in refs]
        pred_len += len(p)
        # closest reference length (ties -> shorter)
        ref_len += min((abs(len(r) - len(p)), len(r)) for r in rs)[1]
        for n in range(1, max_n + 1):
            p_counts = _ngrams(p, n)
            max_ref = Counter()
            for r in rs:
                max_ref |= _ngrams(r, n)
            matches[n - 1] += sum(min(c, max_ref[g]) for g, c in p_counts.items())
            totals[n - 1] += max(len(p) - n + 1, 0)

    log_p = 0.0
    for n in range(max_n):
        m, t = matches[n], totals[n]
        if smooth and n > 0:
            m, t = m + 1, t + 1
        if m == 0 or t == 0:
            return 0.0
        log_p += math.log(m / t) / max_n
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(log_p)


def sentence_bleu(prediction: str, references: Sequence[str], max_n: int = 4) -> float:
    return corpus_bleu([prediction], [references], max_n=max_n, smooth=True)


# ------------------------------------------------------------- ROUGE-L
def _lcs(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, references: Sequence[str], beta: float = 1.0) -> float:
    """ROUGE-L F-measure based on the longest common subsequence (Lin, 2004)."""
    p = normalize(prediction).split()
    best = 0.0
    for ref in references:
        r = normalize(ref).split()
        lcs = _lcs(p, r)
        if lcs == 0:
            continue
        prec, rec = lcs / len(p), lcs / len(r)
        f = (1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec)
        best = max(best, f)
    return best

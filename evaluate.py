"""Score one or more prediction files against references and compare systems.

Input format (JSON Lines), one object per example:
    {"id": "q1", "references": ["Paris"], "prediction": "Paris", "confidence": 0.92}
`confidence` is optional; when present, calibration metrics are reported.

    python evaluate.py examples/system_a.jsonl examples/system_b.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from llm_eval import (bootstrap_ci, brier_score, corpus_bleu, exact_match,
                      expected_calibration_error, paired_bootstrap_test, rouge_l, token_f1)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(rows):
    em = [exact_match(r["prediction"], r["references"]) for r in rows]
    f1 = [token_f1(r["prediction"], r["references"]) for r in rows]
    rl = [rouge_l(r["prediction"], r["references"]) for r in rows]
    out = {
        "EM": bootstrap_ci(em),
        "F1": bootstrap_ci(f1),
        "ROUGE-L": bootstrap_ci(rl),
        "BLEU": corpus_bleu([r["prediction"] for r in rows], [r["references"] for r in rows],
                            smooth=True),
        "_f1": f1,
    }
    if all("confidence" in r for r in rows):
        conf = [r["confidence"] for r in rows]
        correct = [e == 1.0 for e in em]
        out["ECE"] = expected_calibration_error(conf, correct, n_bins=5)
        out["Brier"] = brier_score(conf, correct)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    results = {Path(f).stem: score(load(f)) for f in args.files}
    print(f"{'system':<12} {'EM':>22} {'F1':>22} {'ROUGE-L':>22} {'BLEU':>6} {'ECE':>6} {'Brier':>6}")
    for name, r in results.items():
        ece = f"{r['ECE']:.3f}" if "ECE" in r else "-"
        brier = f"{r['Brier']:.3f}" if "Brier" in r else "-"
        print(f"{name:<12} {str(r['EM']):>22} {str(r['F1']):>22} {str(r['ROUGE-L']):>22} "
              f"{r['BLEU']:>6.3f} {ece:>6} {brier:>6}")
    print("(EM/F1/ROUGE-L: mean [95% bootstrap CI])")

    names = list(results)
    if len(names) >= 2:
        a, b = names[0], names[1]
        fa, fb = results[a]["_f1"], results[b]["_f1"]
        better, worse = (a, b) if np.mean(fa) >= np.mean(fb) else (b, a)
        p = paired_bootstrap_test(results[better]["_f1"], results[worse]["_f1"])
        print(f"\npaired bootstrap on F1: is {better} better than {worse}? p = {p:.4f}")


if __name__ == "__main__":
    main()

# llm-eval-toolkit

![tests](https://github.com/laxmanmandal1234/llm-eval-toolkit/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A lightweight toolkit for **evaluating LLM outputs rigorously**: reference-based metrics, confidence calibration, self-consistency, and — most importantly — **statistical uncertainty** (bootstrap confidence intervals and paired significance tests).

Every metric is implemented from its original definition in plain Python/NumPy and unit-tested against hand-computed values.

## Why

Leaderboard-style comparisons often report a single number per model on a few hundred examples. Two questions are usually left unanswered:

1. **Is the metric measuring what we care about?** (e.g. exact match marks *"The capital is Paris."* as wrong)
2. **Is the difference real or noise?** (a 3-point gap on 200 examples may not be significant)

This toolkit makes both questions easy to check.

## Contents

| Module | Metrics |
|---|---|
| `llm_eval/text_metrics.py` | SQuAD-style normalisation, **Exact Match**, **token F1**, corpus/sentence **BLEU** (clipped n-gram precision, brevity penalty, optional smoothing), **ROUGE-L** (LCS F-measure) |
| `llm_eval/calibration.py` | **Expected Calibration Error**, reliability-diagram bins, **Brier score**, **self-consistency** majority vote with vote-share confidence |
| `llm_eval/stats.py` | Percentile **bootstrap CI**, one-sided **paired bootstrap test** between two systems |
| `evaluate.py` | CLI: scores JSONL prediction files and compares systems |

## Usage

```bash
pip install -r requirements.txt
python examples/make_toy_data.py                 # creates two simulated systems
python evaluate.py examples/system_a.jsonl examples/system_b.jsonl
```

Input format (one JSON object per line; `confidence` is optional):

```json
{"id": "q1", "references": ["Paris"], "prediction": "The capital of France is Paris.", "confidence": 0.93}
```

Library use:

```python
from llm_eval import token_f1, expected_calibration_error, bootstrap_ci, paired_bootstrap_test

f1_a = [token_f1(p, refs) for p, refs in zip(preds_a, references)]
print(bootstrap_ci(f1_a))                      # 0.712 [0.658, 0.764]
print(paired_bootstrap_test(f1_b, f1_a))       # p-value that B is not better than A
```

## Worked example (simulated data)

`examples/make_toy_data.py` **simulates** two systems on 200 capital-city questions (these are not real model outputs; error rates are set by hand to illustrate the metrics):

- **system_a** — answers correctly 85 % of the time, but 30 % of answers are verbose sentences, and it is over-confident when wrong.
- **system_b** — answers correctly 75 % of the time, always tersely, with lower confidence when wrong.

| system | EM | F1 | ROUGE-L | BLEU | ECE | Brier |
|---|---|---|---|---:|---:|---:|
| system_a | 0.625 [0.560, 0.690] | 0.712 [0.658, 0.764] | 0.712 [0.658, 0.764] | 0.020 | 0.290 | 0.316 |
| system_b | 0.750 [0.690, 0.810] | 0.750 [0.690, 0.810] | 0.750 [0.690, 0.810] | 0.888 | 0.237 | 0.074 |

Paired bootstrap on F1 — *is system_b better than system_a?* **p = 0.17** (not significant at 0.05).

**What this shows**

- **Metric choice can flip the ranking.** system_a is the more *accurate* system, but EM ranks it lower because verbose correct answers fail exact match. Token F1 narrows the gap; an LLM-judge or answer-extraction step would be needed to give a fair score.
- **BLEU is the wrong tool for short answers**: it collapses to ≈ 0 for one-word predictions with any verbosity and is dominated by the brevity penalty.
- **Confidence intervals overlap** and the paired test is not significant, so with 200 examples we cannot claim either system is better on F1.
- **Calibration differs from accuracy**: system_b's Brier score is far lower because its confidence drops when it is wrong.

## Tests

```bash
pip install pytest
pytest -q
```

15 tests, including hand-computed values for F1, BLEU brevity penalty and clipping, ROUGE-L, ECE, and Brier score.

## References

- Rajpurkar et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.*
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.*
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
- Guo et al. (2017). *On Calibration of Modern Neural Networks.*
- Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.*
- Koehn (2004). *Statistical Significance Tests for Machine Translation Evaluation.*
- Dror et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP.*

## License

MIT © Laxman Mandal

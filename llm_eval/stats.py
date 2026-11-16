"""Uncertainty and significance for evaluation scores.

Reporting a single number on a few hundred examples hides a lot of noise.
These helpers quantify it with the bootstrap (Efron & Tibshirani, 1993) and the
paired bootstrap test commonly used in NLP (Koehn, 2004; Dror et al., 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Interval:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(scores: Sequence[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile bootstrap confidence interval for the mean of per-example scores."""
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    return Interval(float(x.mean()), float(np.quantile(means, alpha / 2)),
                    float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_test(scores_a: Sequence[float], scores_b: Sequence[float],
                          n_resamples: int = 10_000, seed: int = 0) -> float:
    """One-sided p-value for H0: system A is not better than system B.

    Both systems are scored on the *same* examples, so we resample example
    indices and recompute the difference in means.  The p-value is the share of
    resamples where A does not beat B.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("systems must be scored on the same examples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((diffs <= 0).mean())

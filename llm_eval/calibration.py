"""Confidence calibration and self-consistency."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .text_metrics import normalize


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool],
                     n_bins: int = 10) -> List[ReliabilityBin]:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidences must be in [0, 1]")
    edges = np.linspace(0, 1, n_bins + 1)
    # bin i holds (edges[i], edges[i+1]]; confidence 0 goes in the first bin
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    bins = []
    for i in range(n_bins):
        m = idx == i
        if m.any():
            bins.append(ReliabilityBin(edges[i], edges[i + 1], int(m.sum()),
                                       float(conf[m].mean()), float(corr[m].mean())))
    return bins


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    """ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|  (Guo et al., 2017)."""
    n = len(confidences)
    return float(sum(b.count / n * abs(b.accuracy - b.mean_confidence)
                     for b in reliability_bins(confidences, correct, n_bins)))


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    conf = np.asarray(confidences, dtype=float)
    return float(np.mean((conf - np.asarray(correct, dtype=float)) ** 2))


def self_consistency(samples: Sequence[str]) -> Tuple[str, float]:
    """Majority vote over sampled answers (Wang et al., 2023).

    Returns the most common normalised answer and its vote share, which is a
    simple, model-agnostic confidence estimate.
    """
    if not samples:
        raise ValueError("need at least one sample")
    normed = [normalize(s) for s in samples]
    counts = Counter(normed)
    # break ties by first occurrence for determinism
    best = max(counts, key=lambda a: (counts[a], -normed.index(a)))
    original = samples[normed.index(best)]
    return original, counts[best] / len(samples)

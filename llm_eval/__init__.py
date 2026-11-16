"""llm_eval: lightweight, dependency-light evaluation utilities for LLM outputs."""

from .calibration import (brier_score, expected_calibration_error, reliability_bins,
                          self_consistency)
from .stats import bootstrap_ci, paired_bootstrap_test
from .text_metrics import (corpus_bleu, exact_match, normalize, rouge_l, sentence_bleu,
                           token_f1)

__all__ = [
    "normalize", "exact_match", "token_f1", "corpus_bleu", "sentence_bleu", "rouge_l",
    "reliability_bins", "expected_calibration_error", "brier_score", "self_consistency",
    "bootstrap_ci", "paired_bootstrap_test",
]
__version__ = "0.1.0"
